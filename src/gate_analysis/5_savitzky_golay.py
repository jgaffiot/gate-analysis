"""Method 5: Savitzky-Golay derivative + peak detection.

Pipeline
--------
1. **First derivative via SG filter** – a Savitzky-Golay polynomial fit over
   ``window_length`` samples yields a smooth first derivative used later for
   slope estimation.

2. **Running step-score on the derivative** – for each sample ``i`` the score is

       step_score[i] = |mean(d1[i : i+gap]) − mean(d1[i−gap : i])|

   This is a matched filter for step changes in the derivative: it peaks
   *exactly at* each breakpoint, regardless of the magnitude of the slope
   change, because the mean on one side of the candidate point differs from
   the mean on the other side by the slope-change amplitude.

3. **Noise floor and gap choice** – the SG first derivative retains residual
   noise (σ_d1 ≈ 10 %/s for the default window and position noise of 1 %).
   Averaging ``gap`` samples on each side reduces the noise standard deviation
   of each mean to σ_d1/√gap.  The noise floor of the score (difference of two
   independent means) is therefore σ_d1·√(2/gap).  The default
   ``gap = min_peak_distance = 100`` samples (1 s) gives a noise floor of
   ≈ 1.4 %/s, well below the smallest expected slope change (5 %/s, SNR ≈ 3.5).
   Using a shorter gap (e.g. window_length//2 = 25) raises the noise floor to
   ≈ 2.8 %/s, at which point noise bumps can outrank the genuine small step.

4. **Peak selection** – ``scipy.signal.find_peaks`` with ``distance =
   min_peak_distance`` finds candidate peaks; they are then ranked by
   *prominence* (how much each peak stands out from its local baseline) rather
   than raw height, making the selection robust against broad noise humps.

5. **Slope estimation** – the median of the first derivative in each segment
   between consecutive detected breakpoints.

Tuning on real data
-------------------
- ``window_length``: larger values smooth more but widen the transition region.
- ``gap`` / ``min_peak_distance``: must satisfy ``gap < T_min/2`` where
  ``T_min`` is the shortest interval between breakpoints, while being large
  enough to average out derivative noise.  If breakpoints are closer together
  or noise is higher, adjust both parameters accordingly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks, peak_prominences, savgol_filter

from gate_analysis.common import GateData, generate_synthetic_data


def savitzky_golay(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    *,
    window_length: int = 51,
    polyorder: int = 3,
    n_breakpoints: int = 3,
    min_peak_distance: int = 100,
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
    n_breakpoints : int
        Expected number of breakpoints to detect.
    min_peak_distance : int
        Minimum separation between breakpoint candidates (samples).

    Returns
    -------
    dict with keys: breakpoints, slopes, derivative, second_deriv, smoothed
    """
    dt = float(np.mean(np.diff(time)))

    # Smooth the signal and compute the first derivative.
    smoothed = savgol_filter(position, window_length, polyorder)
    derivative = savgol_filter(position, window_length, polyorder, deriv=1, delta=dt)

    # Compute a two-stage second derivative for visualisation only.
    second_deriv = savgol_filter(
        derivative, window_length, polyorder, deriv=1, delta=dt
    )

    # --- Breakpoint detection via running step-score on the derivative ---
    # For each sample i, the step score is:
    #   step_score[i] = |mean(d1[i : i+gap]) − mean(d1[i-gap : i])|
    # This is the magnitude of the derivative jump across sample i averaged over
    # a window of `gap` samples on each side.  Using windowed means instead of
    # raw values suppresses local noise while preserving true slope transitions.
    # The gap must be large enough for the windowed means to average out the derivative
    # noise.  Using min_peak_distance (1 s) ensures the noise floor is well below the
    # smallest expected slope change (5 %/s), while keeping peaks separable.
    gap = min_peak_distance  # averaging window = 1 s by default
    cumsum = np.cumsum(derivative)
    i_min, i_max = gap, len(derivative) - gap
    idx = np.arange(i_min, i_max)
    after_mean = (cumsum[idx + gap] - cumsum[idx]) / gap
    before_mean = (cumsum[idx] - cumsum[idx - gap]) / gap
    step_score = np.zeros(len(derivative))
    step_score[i_min:i_max] = np.abs(after_mean - before_mean)

    # Find the n_breakpoints most prominent peaks in the step-score.
    peaks, _ = find_peaks(step_score, distance=min_peak_distance)

    if len(peaks) == 0:
        print("Warning: no breakpoints detected in step-score.")
        return {
            "breakpoints": [],
            "slopes": [],
            "derivative": derivative,
            "step_score": step_score,
            "second_deriv": second_deriv,
            "smoothed": smoothed,
        }

    # Rank by prominence so that noise-induced bumps don't beat real transitions.
    prominences, _, _ = peak_prominences(step_score, peaks)
    n = min(n_breakpoints, len(peaks))
    top_idx = np.argsort(prominences)[-n:]
    detected_peaks = np.sort(peaks[top_idx])
    breakpoints = [float(time[p]) for p in detected_peaks]

    # Estimate slopes as median first derivative in each ramp segment
    slopes: list[float] = []
    for i in range(len(breakpoints) - 1):
        mask = (time >= breakpoints[i]) & (time <= breakpoints[i + 1])
        slopes.append(float(np.median(derivative[mask])) if np.any(mask) else 0.0)

    print("=== Method 5: Savitzky-Golay Derivative + Peak Detection ===")
    print(f"SG params: window={window_length}, polyorder={polyorder}")
    print(f"Breakpoints: {[f'{bp:.3f}' for bp in breakpoints]}")
    for i, s in enumerate(slopes):
        print(f"Slope {i + 1}: {s:.2f} %/s")

    return {
        "breakpoints": breakpoints,
        "slopes": slopes,
        "derivative": derivative,
        "step_score": step_score,
        "second_deriv": second_deriv,
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

    def _add_breakpoint_spans(
        fig: figure, bps: list[float], *, gt: bool = False
    ) -> None:
        for bp in bps:
            fig.add_layout(
                Span(
                    location=bp,
                    dimension="height",
                    line_color="green" if gt else "red",
                    line_dash="dotted" if gt else "dashed",
                    line_alpha=0.4 if gt else 0.7,
                )
            )

    # Top: signal + smoothed
    p1 = figure(
        width=1200,
        height=350,
        title="Method 5: Savitzky-Golay Derivative + Peak Detection",
        y_axis_label="Gate position (%)",
    )
    p1.scatter(
        data.time, data.position, marker="circle", color="gray", alpha=0.3, size=2
    )
    p1.line(
        data.time,
        result["smoothed"],
        line_color="blue",
        line_width=1.5,
        legend_label="SG smoothed",
    )
    _add_breakpoint_spans(p1, result["breakpoints"])
    _add_breakpoint_spans(p1, data.breakpoints, gt=True)
    p1.legend.location = "top_right"
    p1.grid.grid_line_alpha = 0.3

    # Middle: first derivative
    p2 = figure(
        width=1200,
        height=300,
        title="First derivative dy/dt",
        y_axis_label="dy/dt (%/s)",
        x_range=p1.x_range,
    )
    p2.line(
        data.time,
        result["derivative"],
        line_color="blue",
        line_width=1,
        legend_label="dy/dt",
    )
    p2.add_layout(
        Span(location=0, dimension="width", line_color="black", line_width=0.5)
    )
    _add_breakpoint_spans(p2, result["breakpoints"])
    _add_breakpoint_spans(p2, data.breakpoints, gt=True)
    p2.legend.location = "top_right"
    p2.grid.grid_line_alpha = 0.3

    # Bottom: step-score + detected peaks
    p3 = figure(
        width=1200,
        height=300,
        title="Running step-score on derivative — peaks = detected breakpoints",
        x_axis_label="Time (s)",
        y_axis_label="Step score (%/s)",
        x_range=p1.x_range,
    )
    p3.line(
        data.time,
        result["step_score"],
        line_color="darkorange",
        line_width=1,
        legend_label="step score",
    )
    _add_breakpoint_spans(p3, result["breakpoints"])
    _add_breakpoint_spans(p3, data.breakpoints, gt=True)
    p3.grid.grid_line_alpha = 0.3

    slopes = result["slopes"]
    info_lines = [f"Slope {i + 1}: {s:.2f} %/s" for i, s in enumerate(slopes)]
    info_lines.append(f"True: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s")
    p3.add_layout(
        Label(
            x=10,
            y=10,
            x_units="screen",
            y_units="screen",
            text="\n".join(info_lines),
            text_font_size="9pt",
            background_fill_color="wheat",
            background_fill_alpha=0.8,
        )
    )

    show(column(p1, p2, p3))
