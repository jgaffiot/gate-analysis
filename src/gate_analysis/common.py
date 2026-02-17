"""Synthetic data generation and plotting utilities for gate closing analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


@dataclass
class GateData:
    """Container for synthetic gate closing data with ground-truth parameters."""

    time: npt.NDArray[np.floating[Any]]
    position: npt.NDArray[np.floating[Any]]
    # Ground-truth parameters
    breakpoints: list[float] = field(default_factory=list)
    slopes: list[float] = field(default_factory=list)
    plateaus: tuple[float, float] = (98.0, 2.0)


def generate_synthetic_data(
    *,
    dt: float = 0.01,
    plateau_high: float = 98.0,
    plateau_low: float = 2.0,
    t_start_closing: float = 2.0,
    t_slope_change: float = 5.0,
    t_end_closing: float = 9.0,
    t_total: float = 12.0,
    slope_fast: float = -25.0,
    slope_slow: float = -5.0,
    noise_std: float = 1.0,
    seed: int = 42,
) -> GateData:
    """Generate a realistic synthetic gate closing signal.

    The signal has four phases:
    1. High plateau (~plateau_high%)
    2. Fast linear closing (steep negative slope)
    3. Slow linear closing (gentle negative slope)
    4. Low plateau (~plateau_low%)

    The slopes are given in %/s. Default fast slope is -25%/s and slow is -5%/s.
    The breakpoint between fast and slow closing is at t_slope_change.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0, t_total, dt)
    position = np.empty_like(time)

    # Compute position values at breakpoints for continuity
    pos_at_start = plateau_high
    pos_at_slope_change = pos_at_start + slope_fast * (t_slope_change - t_start_closing)

    for i, t in enumerate(time):
        if t < t_start_closing:
            position[i] = plateau_high
        elif t < t_slope_change:
            position[i] = pos_at_start + slope_fast * (t - t_start_closing)
        elif t < t_end_closing:
            position[i] = pos_at_slope_change + slope_slow * (t - t_slope_change)
        else:
            position[i] = plateau_low

    # Clamp to physical range before adding noise
    position = np.clip(position, plateau_low, plateau_high)

    # Add sensor noise
    position = position + rng.normal(0, noise_std, size=position.shape)

    return GateData(
        time=time,
        position=position,
        breakpoints=[t_start_closing, t_slope_change, t_end_closing],
        slopes=[slope_fast, slope_slow],
        plateaus=(plateau_high, plateau_low),
    )


def plot_results(
    data: GateData,
    title: str,
    *,
    fitted_segments: list[
        tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
    ]
    | None = None,
    detected_breakpoints: list[float] | None = None,
    estimated_slopes: list[float] | None = None,
) -> plt.Figure:
    """Plot the raw data with optional fitted segments and detected breakpoints."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        data.time,
        data.position,
        ".",
        color="gray",
        alpha=0.3,
        markersize=2,
        label="Raw data",
    )

    if fitted_segments:
        for i, (t_seg, y_seg) in enumerate(fitted_segments):
            ax.plot(t_seg, y_seg, linewidth=2, label=f"Segment {i}")

    if detected_breakpoints:
        for bp in detected_breakpoints:
            ax.axvline(
                bp, color="red", linestyle="--", alpha=0.7, label=f"BP @ {bp:.2f}s"
            )

    # Ground truth breakpoints
    for bp in data.breakpoints:
        ax.axvline(bp, color="green", linestyle=":", alpha=0.4)

    info_parts = []
    if estimated_slopes:
        for i, s in enumerate(estimated_slopes):
            info_parts.append(f"Slope {i + 1}: {s:.2f} %/s")
    info_parts.append(f"True slopes: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s")

    ax.text(
        0.02,
        0.02,
        "\n".join(info_parts),
        transform=ax.transAxes,
        verticalalignment="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gate position (%)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    return fig


if __name__ == "__main__":
    data = generate_synthetic_data()
    print(f"Generated {len(data.time)} samples")
    print(f"True breakpoints: {data.breakpoints}")
    print(f"True slopes: {data.slopes}")
    fig = plot_results(data, "Synthetic Gate Closing Data")
    plt.show()
