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
    import matplotlib.pyplot as plt

    data = generate_synthetic_data()
    result = savitzky_golay(data.time, data.position)
    segments = _build_segments(data, result)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: signal + smoothed
    axes[0].plot(data.time, data.position, ".", color="gray", alpha=0.3, markersize=2)
    axes[0].plot(
        data.time, result["smoothed"], "b-", linewidth=1.5, label="SG smoothed"
    )
    for bp in result["breakpoints"]:
        axes[0].axvline(bp, color="red", linestyle="--", alpha=0.7)
    for bp in data.breakpoints:
        axes[0].axvline(bp, color="green", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("Gate position (%)")
    axes[0].set_title("Option E: Savitzky-Golay Derivative + Threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: derivative
    axes[1].plot(data.time, result["derivative"], "b-", linewidth=1)
    axes[1].axhline(0, color="black", linewidth=0.5)
    for bp in result["breakpoints"]:
        axes[1].axvline(bp, color="red", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Derivative (%/s)")
    axes[1].set_title("First derivative (instantaneous slope)")
    axes[1].grid(True, alpha=0.3)

    info = f"Fast slope: {result['slopes'][0]:.2f} %/s\nSlow slope: {result['slopes'][1]:.2f} %/s"
    info += f"\nTrue: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s"
    axes[1].text(
        0.02,
        0.02,
        info,
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.show()
