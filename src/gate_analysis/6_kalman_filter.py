"""Option F: Kalman filter with manual regime detection.

Uses a constant-velocity Kalman filter (filterpy) to estimate position and
velocity (slope) states. Regime transitions are detected from the velocity
state, and closing slopes are extracted from the filtered velocity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

from gate_analysis.common import GateData, generate_synthetic_data


def kalman_filter(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    *,
    measurement_noise: float = 1.0,
    process_noise: float = 0.5,
    velocity_threshold: float = 1.0,
    slope_split_ratio: float = 0.5,
) -> dict[str, Any]:
    """Analyze gate closing using a constant-velocity Kalman filter.

    The state vector is [position, velocity]. The Kalman filter provides
    optimal estimation of the velocity (slope) at each time step.

    Parameters
    ----------
    time : array
        Time values.
    position : array
        Gate position values (%).
    measurement_noise : float
        Measurement noise standard deviation (%).
    process_noise : float
        Process noise factor for velocity changes.
    velocity_threshold : float
        Minimum absolute velocity (%/s) to consider as "closing".
    slope_split_ratio : float
        Ratio to split fast/slow regimes based on peak velocity.

    Returns
    -------
    dict with keys: breakpoints, slopes, filtered_position, filtered_velocity
    """
    dt = float(np.mean(np.diff(time)))
    n = len(time)

    # Set up constant-velocity Kalman filter
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # State transition: [pos, vel] -> [pos + vel*dt, vel]
    kf.F = np.array([[1.0, dt], [0.0, 1.0]])

    # Measurement: we observe position only
    kf.H = np.array([[1.0, 0.0]])

    # Measurement noise
    kf.R = np.array([[measurement_noise**2]])

    # Process noise
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise**2)

    # Initial state
    kf.x = np.array([[float(position[0])], [0.0]])
    kf.P *= 100.0  # large initial uncertainty

    # Run filter
    filtered_pos = np.zeros(n)
    filtered_vel = np.zeros(n)

    for i in range(n):
        kf.predict()
        kf.update(np.array([[float(position[i])]]))
        filtered_pos[i] = float(kf.x[0, 0])
        filtered_vel[i] = float(kf.x[1, 0])

    # Detect closing region from velocity
    closing_mask = filtered_vel < -velocity_threshold
    if not np.any(closing_mask):
        print("Warning: no closing region detected.")
        return {
            "breakpoints": [],
            "slopes": [],
            "filtered_position": filtered_pos,
            "filtered_velocity": filtered_vel,
        }

    closing_indices = np.where(closing_mask)[0]
    t_start = float(time[closing_indices[0]])
    t_end = float(time[closing_indices[-1]])

    # Split fast/slow based on velocity magnitude
    min_vel = float(np.min(filtered_vel[closing_mask]))
    split_thresh = min_vel * slope_split_ratio

    closing_vels = filtered_vel[closing_indices]
    fast_mask = closing_vels < split_thresh

    if np.any(fast_mask) and not np.all(fast_mask):
        fast_idx = np.where(fast_mask)[0]
        last_fast_global = closing_indices[fast_idx[-1]]
        t_change = float(time[last_fast_global])
    else:
        t_change = (t_start + t_end) / 2

    # Estimate slopes from filtered velocity
    fast_region = (time >= t_start) & (time <= t_change)
    slow_region = (time > t_change) & (time <= t_end)

    slope_fast = (
        float(np.median(filtered_vel[fast_region])) if np.any(fast_region) else 0.0
    )
    slope_slow = (
        float(np.median(filtered_vel[slow_region])) if np.any(slow_region) else 0.0
    )

    breakpoints = [t_start, t_change, t_end]

    print("=== Option F: Kalman Filter with Regime Detection ===")
    print(f"KF params: R={measurement_noise:.1f}, Q_var={process_noise:.1f}")
    print(f"Breakpoints: {[f'{bp:.3f}' for bp in breakpoints]}")
    print(f"Fast slope (median filtered velocity): {slope_fast:.2f} %/s")
    print(f"Slow slope (median filtered velocity): {slope_slow:.2f} %/s")

    return {
        "breakpoints": breakpoints,
        "slopes": [slope_fast, slope_slow],
        "filtered_position": filtered_pos,
        "filtered_velocity": filtered_vel,
    }


def _build_segments(
    data: GateData, result: dict[str, Any]
) -> list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    """Build filtered position as a single segment for plotting."""
    return [(data.time, result["filtered_position"])]


if __name__ == "__main__":
    from bokeh.io import show
    from bokeh.layouts import column
    from bokeh.models import Label, Span
    from bokeh.plotting import figure

    data = generate_synthetic_data()
    result = kalman_filter(data.time, data.position)

    # Top: position
    p1 = figure(
        width=1200,
        height=400,
        title="Option F: Kalman Filter with Regime Detection",
        y_axis_label="Gate position (%)",
    )
    p1.scatter(data.time, data.position, marker="circle", color="gray", alpha=0.3, size=2)
    p1.line(
        data.time, result["filtered_position"], line_color="blue", line_width=1.5, legend_label="Kalman filtered"
    )
    for bp in result["breakpoints"]:
        p1.add_layout(Span(location=bp, dimension="height", line_color="red", line_dash="dashed", line_alpha=0.7))
    for bp in data.breakpoints:
        p1.add_layout(Span(location=bp, dimension="height", line_color="green", line_dash="dotted", line_alpha=0.4))
    p1.legend.location = "top_right"
    p1.grid.grid_line_alpha = 0.3

    # Bottom: velocity (shares x range with top)
    p2 = figure(
        width=1200,
        height=400,
        title="Kalman-estimated velocity (instantaneous slope)",
        x_axis_label="Time (s)",
        y_axis_label="Velocity (%/s)",
        x_range=p1.x_range,
    )
    p2.line(data.time, result["filtered_velocity"], line_color="blue", line_width=1)
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
