"""Option E: Savitzky-Golay derivative + threshold detection.

Uses a Savitzky-Golay filter to compute a smoothed first derivative of the
signal, then detects regime transitions by thresholding the derivative.
Slopes are estimated from the derivative plateaus in the ramp regions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

from gate_analysis.common import GateData, generate_synthetic_data


def savitzky_golay(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    *,
    window_length: int = 51,
    polyorder: int = 3,
    slope_threshold: float = 1.0,
    slope_split_ratio: float = 0.5,
) -> dict[str, Any]:
    """Analyze gate closing curve using Savitzky-Golay derivative.

    Parameters
    ----------
    time : array
        Time values.
    position : array
        Gate position values (%).
    window_length : int
        SG filter window length (must be odd).
    polyorder : int
        SG filter polynomial order.
    slope_threshold : float
        Minimum absolute derivative (%/s) to consider as "closing".
    slope_split_ratio : float
        Ratio of max derivative to split fast/slow closing regions.

    Returns
    -------
    dict with keys: breakpoints, slopes, derivative, smoothed
    """
    dt = float(np.mean(np.diff(time)))

    # Smooth the signal
    smoothed = savgol_filter(position, window_length, polyorder)

    # Compute first derivative (slope) directly from SG filter
    derivative = savgol_filter(position, window_length, polyorder, deriv=1, delta=dt)

    # Find closing region: where derivative is significantly negative
    closing_mask = derivative < -slope_threshold
    if not np.any(closing_mask):
        print("Warning: no closing region detected. Try lowering slope_threshold.")
        return {
            "breakpoints": [],
            "slopes": [],
            "derivative": derivative,
            "smoothed": smoothed,
        }

    closing_indices = np.where(closing_mask)[0]
    t_start_closing = float(time[closing_indices[0]])
    t_end_closing = float(time[closing_indices[-1]])

    # Split closing region into fast and slow based on derivative magnitude
    closing_derivs = derivative[closing_mask]
    min_deriv = float(np.min(closing_derivs))  # most negative = fastest
    split_threshold = min_deriv * slope_split_ratio

    # Find the transition point between fast and slow
    # Look for where derivative crosses the split threshold
    in_closing = derivative[closing_indices]
    fast_mask = in_closing < split_threshold
    if np.any(fast_mask) and not np.all(fast_mask):
        # Find last index in fast region
        fast_indices = np.where(fast_mask)[0]
        last_fast = closing_indices[fast_indices[-1]]
        # Find first index in slow region after fast
        slow_after_fast = closing_indices[closing_indices > last_fast]
        if len(slow_after_fast) > 0:
            t_slope_change = float(time[last_fast])
        else:
            t_slope_change = (t_start_closing + t_end_closing) / 2
    else:
        t_slope_change = (t_start_closing + t_end_closing) / 2

    # Estimate slopes as median derivative in each sub-region
    fast_region = (time >= t_start_closing) & (time <= t_slope_change)
    slow_region = (time > t_slope_change) & (time <= t_end_closing)

    slope_fast = (
        float(np.median(derivative[fast_region])) if np.any(fast_region) else 0.0
    )
    slope_slow = (
        float(np.median(derivative[slow_region])) if np.any(slow_region) else 0.0
    )

    breakpoints = [t_start_closing, t_slope_change, t_end_closing]

    print("=== Option E: Savitzky-Golay Derivative + Threshold ===")
    print(f"SG params: window={window_length}, polyorder={polyorder}")
    print(f"Breakpoints: {[f'{bp:.3f}' for bp in breakpoints]}")
    print(f"Fast slope (median derivative): {slope_fast:.2f} %/s")
    print(f"Slow slope (median derivative): {slope_slow:.2f} %/s")

    return {
        "breakpoints": breakpoints,
        "slopes": [slope_fast, slope_slow],
        "derivative": derivative,
        "smoothed": smoothed,
    }


def _build_segments(
    data: GateData, result: dict[str, Any]
) -> list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    """Build smoothed curve as a single segment for plotting."""
    return [(data.time, result["smoothed"])]


if __name__ == "__main__":
    from bokeh.io import show
    from bokeh.layouts import column
    from bokeh.models import Label, Span
    from bokeh.plotting import figure

    data = generate_synthetic_data()
    result = savitzky_golay(data.time, data.position)

    # Top: signal + smoothed
    p1 = figure(
        width=1200,
        height=400,
        title="Option E: Savitzky-Golay Derivative + Threshold",
        y_axis_label="Gate position (%)",
    )
    p1.scatter(data.time, data.position, marker="circle", color="gray", alpha=0.3, size=2)
    p1.line(
        data.time, result["smoothed"], line_color="blue", line_width=1.5, legend_label="SG smoothed"
    )
    for bp in result["breakpoints"]:
        p1.add_layout(Span(location=bp, dimension="height", line_color="red", line_dash="dashed", line_alpha=0.7))
    for bp in data.breakpoints:
        p1.add_layout(Span(location=bp, dimension="height", line_color="green", line_dash="dotted", line_alpha=0.4))
    p1.legend.location = "top_right"
    p1.grid.grid_line_alpha = 0.3

    # Bottom: derivative (shares x range with top)
    p2 = figure(
        width=1200,
        height=400,
        title="First derivative (instantaneous slope)",
        x_axis_label="Time (s)",
        y_axis_label="Derivative (%/s)",
        x_range=p1.x_range,
    )
    p2.line(data.time, result["derivative"], line_color="blue", line_width=1)
    p2.add_layout(Span(location=0, dimension="width", line_color="black", line_width=0.5))
    for bp in result["breakpoints"]:
        p2.add_layout(Span(location=bp, dimension="height", line_color="red", line_dash="dashed", line_alpha=0.7))
    p2.grid.grid_line_alpha = 0.3

    info = f"Fast slope: {result['slopes'][0]:.2f} %/s\nSlow slope: {result['slopes'][1]:.2f} %/s"
    info += f"\nTrue: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s"
    p2.add_layout(
        Label(
            x=10, y=10, x_units="screen", y_units="screen",
            text=info, text_font_size="9pt",
            background_fill_color="wheat", background_fill_alpha=0.8,
        )
    )

    show(column(p1, p2))
