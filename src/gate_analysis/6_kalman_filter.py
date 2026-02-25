"""Method 6: Adaptive Kalman filter with innovation-based changepoint detection.

Uses a constant-velocity Kalman filter (filterpy). Changepoints are detected
online by monitoring a CUSUM test on the Normalized Innovation Squared (NIS).
When the NIS CUSUM exceeds a threshold, the filter is restarted by resetting
the velocity covariance at the estimated changepoint, enabling rapid
re-convergence to the new slope. Slopes are then estimated by OLS on each
detected segment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy import stats

from gate_analysis.common import GateData, generate_synthetic_data


def kalman_adaptive(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    *,
    measurement_noise: float = 1.0,
    process_noise: float = 0.1,
    cusum_slack: float = 3.0,
    cusum_threshold: float = 15.0,
    min_segment_samples: int = 50,
    reset_velocity_var: float = 500.0,
) -> dict[str, Any]:
    """Analyze gate closing using an adaptive Kalman filter with CUSUM restart.

    A constant-velocity Kalman filter estimates position and velocity at each
    step. The Normalized Innovation Squared (NIS = innovation² / S) should
    follow a chi²(1) distribution (mean = 1) when the model is correct.
    A CUSUM test on NIS detects sustained model mismatch caused by a slope
    change. When the CUSUM alarms, the velocity covariance is inflated at the
    estimated changepoint so the filter re-adapts within a few samples.
    Slopes are estimated by OLS on each detected segment.

    Parameters
    ----------
    time : array
        Time values.
    position : array
        Gate position values (%).
    measurement_noise : float
        Measurement noise standard deviation (R = noise²).
    process_noise : float
        Process noise variance for velocity (small: slope changes handled
        explicitly by restart).
    cusum_slack : float
        Reference value k for CUSUM: g = max(0, g + NIS - k). Should be
        slightly above 1 (the chi²(1) mean) to ignore normal noise.
    cusum_threshold : float
        CUSUM alarm threshold h. Larger values reduce false alarms at the
        cost of detection delay.
    min_segment_samples : int
        Minimum number of samples between two consecutive changepoints, to
        avoid double-detecting a single transition.
    reset_velocity_var : float
        Velocity variance injected into P when a changepoint is declared.
        Large value forces rapid re-adaptation to the new slope.

    Returns
    -------
    dict with keys: breakpoints, slopes, segments, filtered_position,
        filtered_velocity, innovations, nis_sequence, cusum,
        changepoint_indices.
    """
    dt = float(np.mean(np.diff(time)))
    n = len(time)

    # --- Build Kalman filter ---
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1.0, dt], [0.0, 1.0]])  # state transition
    kf.H = np.array([[1.0, 0.0]])  # observe position only
    kf.R = np.array([[measurement_noise**2]])  # measurement noise
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise**2)
    kf.x = np.array([[float(position[0])], [0.0]])
    kf.P *= 100.0  # large initial uncertainty

    filtered_pos = np.zeros(n)
    filtered_vel = np.zeros(n)
    innovations = np.zeros(n)
    nis_sequence = np.zeros(n)
    cusum_values = np.zeros(n)

    changepoint_indices: list[int] = []
    g = 0.0
    g_start = 0  # index where current CUSUM accumulation began
    last_cp_idx = 0  # index of last declared changepoint

    # --- Single-pass filter with online CUSUM detection ---
    for i in range(n):
        kf.predict()

        # After predict(), kf.x is the predicted state and kf.P the predicted
        # covariance.  Compute innovation and its covariance S before update.
        pred_obs = float((kf.H @ kf.x)[0, 0])
        innov = float(position[i]) - pred_obs
        S = float((kf.H @ kf.P @ kf.H.T + kf.R)[0, 0])
        nis = innov**2 / S

        innovations[i] = innov
        nis_sequence[i] = nis

        # CUSUM test — only armed after min_segment_samples from last event
        if i - last_cp_idx >= min_segment_samples:
            prev_g = g
            g = max(0.0, g + nis - cusum_slack)
            # Track start of current accumulation run
            if prev_g == 0.0 and g > 0.0:
                g_start = i
            cusum_values[i] = g

            if g > cusum_threshold:
                # Declare changepoint at g_start (first sample of excess NIS)
                changepoint_indices.append(g_start)
                last_cp_idx = g_start
                # Inflate velocity covariance → filter re-adapts fast
                kf.P[1, 1] = reset_velocity_var
                kf.P[0, 1] = 0.0
                kf.P[1, 0] = 0.0
                g = 0.0
        else:
            cusum_values[i] = g

        kf.update(np.array([[float(position[i])]]))
        filtered_pos[i] = float(kf.x[0, 0])
        filtered_vel[i] = float(kf.x[1, 0])

    # --- OLS slope estimation on each segment ---
    boundaries = [0] + changepoint_indices + [n]
    segments: list[dict[str, Any]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 2:
            continue
        t_seg = time[start:end]
        pos_seg = position[start:end]
        slope, intercept, *_ = stats.linregress(t_seg, pos_seg)
        segments.append(
            {
                "start_idx": start,
                "end_idx": end,
                "t_start": float(time[start]),
                "t_end": float(time[end - 1]),
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )

    # --- Identify closing ramps (significant negative slope) ---
    # Take segments in time order; the two most negative ones are fast/slow.
    closing_segs = sorted(
        [s for s in segments if s["slope"] < -1.0],
        key=lambda s: s["t_start"],
    )

    breakpoints = [float(time[idx]) for idx in changepoint_indices]

    slope_fast = closing_segs[0]["slope"] if len(closing_segs) >= 1 else float("nan")
    slope_slow = closing_segs[1]["slope"] if len(closing_segs) >= 2 else float("nan")

    print("=== Method 6: Adaptive Kalman Filter (CUSUM + restart) ===")
    print(f"KF params: R={measurement_noise:.2f}, Q_var={process_noise:.2f}")
    print(f"CUSUM params: slack={cusum_slack}, threshold={cusum_threshold}")
    print(f"Detected changepoints: {len(changepoint_indices)}")
    print(f"Breakpoints: {[f'{bp:.3f}' for bp in breakpoints]}")
    print(f"Fast slope (OLS): {slope_fast:.2f} %/s")
    print(f"Slow slope (OLS): {slope_slow:.2f} %/s")

    return {
        "breakpoints": breakpoints,
        "slopes": [slope_fast, slope_slow],
        "segments": segments,
        "filtered_position": filtered_pos,
        "filtered_velocity": filtered_vel,
        "innovations": innovations,
        "nis_sequence": nis_sequence,
        "cusum": cusum_values,
        "changepoint_indices": changepoint_indices,
    }


def _build_segments(
    data: GateData, result: dict[str, Any]
) -> list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    """Build OLS segment lines for plotting."""
    segs = []
    for seg in result["segments"]:
        t = data.time[seg["start_idx"] : seg["end_idx"]]
        y = seg["slope"] * t + seg["intercept"]
        segs.append((t, y))
    return segs


if __name__ == "__main__":
    import time as _time

    from bokeh.io import show
    from bokeh.layouts import column
    from bokeh.models import Label, Span
    from bokeh.plotting import figure

    data = generate_synthetic_data()

    t0 = _time.perf_counter()
    result = kalman_adaptive(data.time, data.position)
    elapsed = _time.perf_counter() - t0
    print(f"Runtime: {elapsed * 1000:.1f} ms")

    # --- Panel 1: position ---
    p1 = figure(
        width=1200,
        height=380,
        title="Method 6: Adaptive Kalman Filter (CUSUM + restart)",
        y_axis_label="Gate position (%)",
    )
    p1.scatter(data.time, data.position, color="gray", alpha=0.3, size=2)
    p1.line(
        data.time,
        result["filtered_position"],
        line_color="steelblue",
        line_width=1.5,
        legend_label="Kalman filtered",
    )
    # OLS segment lines
    for seg in result["segments"]:
        t = data.time[seg["start_idx"] : seg["end_idx"]]
        y = seg["slope"] * t + seg["intercept"]
        p1.line(t, y, line_color="darkorange", line_width=2, line_dash="dashed")

    _span_kw: dict[str, Any] = {"dimension": "height", "line_alpha": 0.7}
    for bp in result["breakpoints"]:
        p1.add_layout(
            Span(location=bp, line_color="red", line_dash="dashed", **_span_kw)
        )
    for bp in data.breakpoints:
        p1.add_layout(
            Span(location=bp, line_color="green", line_dash="dotted", **_span_kw)
        )
    p1.legend.location = "top_right"
    p1.grid.grid_line_alpha = 0.3

    # --- Panel 2: filtered velocity ---
    p2 = figure(
        width=1200,
        height=280,
        title="Kalman velocity (slope estimate)",
        y_axis_label="Velocity (%/s)",
        x_range=p1.x_range,
    )
    p2.line(
        data.time, result["filtered_velocity"], line_color="steelblue", line_width=1
    )
    p2.add_layout(
        Span(location=0, dimension="width", line_color="black", line_width=0.5)
    )
    for bp in result["breakpoints"]:
        p2.add_layout(
            Span(location=bp, line_color="red", line_dash="dashed", **_span_kw)
        )
    p2.grid.grid_line_alpha = 0.3

    # --- Panel 3: NIS and CUSUM ---
    p3 = figure(
        width=1200,
        height=280,
        title="CUSUM of Normalized Innovation Squared (NIS)",
        x_axis_label="Time (s)",
        y_axis_label="CUSUM / NIS",
        x_range=p1.x_range,
    )
    p3.line(
        data.time,
        result["nis_sequence"],
        line_color="lightgray",
        line_width=0.8,
        legend_label="NIS",
    )
    p3.line(
        data.time,
        result["cusum"],
        line_color="crimson",
        line_width=1.5,
        legend_label="CUSUM(NIS)",
    )
    p3.add_layout(
        Span(
            location=15.0,
            dimension="width",
            line_color="black",
            line_dash="dashed",
            line_width=1,
        )
    )
    for bp in result["breakpoints"]:
        p3.add_layout(
            Span(location=bp, line_color="red", line_dash="dashed", **_span_kw)
        )
    p3.legend.location = "top_right"
    p3.grid.grid_line_alpha = 0.3

    info = (
        f"Fast slope: {result['slopes'][0]:.2f} %/s  "
        f"Slow slope: {result['slopes'][1]:.2f} %/s\n"
        f"True: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s  |  "
        f"Runtime: {elapsed * 1000:.1f} ms"
    )
    p1.add_layout(
        Label(
            x=10,
            y=10,
            x_units="screen",
            y_units="screen",
            text=info,
            text_font_size="9pt",
            background_fill_color="wheat",
            background_fill_alpha=0.8,
        )
    )

    show(column(p1, p2, p3))
