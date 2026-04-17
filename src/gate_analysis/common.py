"""Synthetic data generation and plotting utilities for gate closing analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from bokeh.models import Label, Span
from bokeh.palettes import Category10
from bokeh.plotting import figure


@dataclass
class GateData:
    """Container for gate closing data with ground-truth parameters.

    The DataFrame ``df`` has a ``date_time`` column (:pyclass:`Datetime("us")`)
    and one or more gate position columns.  Ground-truth parameters are shared
    across gates in synthetic data.  When a trapezoid channel is present
    (``gate_1``), ``trapezoid_decrease_rate`` holds its decrease rate in %/s.
    """

    df: pl.DataFrame
    # Ground-truth parameters
    breakpoints: list[float] = field(default_factory=list)
    slopes: list[float] = field(default_factory=list)
    plateaus: tuple[float, float] = (98.0, 2.0)
    trapezoid_decrease_rate: float | None = None

    @property
    def gate_columns(self) -> list[str]:
        """Names of gate position columns (everything except ``date_time``)."""
        return [c for c in self.df.columns if c != "date_time"]

    @property
    def time(self) -> npt.NDArray[np.floating[Any]]:
        """Elapsed time in seconds from the first timestamp."""
        dt_col = self.df["date_time"]
        durations = dt_col - dt_col[0]
        return (
            durations.dt.total_microseconds().cast(pl.Float64) / 1_000_000.0
        ).to_numpy()


def generate_synthetic_data(
    *,
    dt: float = 0.01,
    plateau_high: float = 98.0,
    plateau_low: float = 2.0,
    t_start_closing: float = 2.0,
    t_slope_change: float = 5.0,
    t_end_closing: float = 9.0,
    t_total: float = 20.0,
    slope_fast: float = -25.0,
    slope_slow: float = -5.0,
    noise_std: float = 1.0,
    n_gates: int = 2,
    trapezoid_decrease_rate: float | None = None,
    trapezoid_rise_duration: float = 1.0,
    start_datetime: datetime | None = None,
    seed: int = 42,
) -> GateData:
    """Generate a realistic synthetic gate closing signal.

    ``gate_0`` has four phases:
    1. High plateau (~plateau_high%)
    2. Fast linear closing (steep negative slope)
    3. Slow linear closing (gentle negative slope)
    4. Low plateau (~plateau_low%)

    ``gate_1`` (when ``n_gates >= 2``) is a trapezoid channel:
    - 0 % before the gate starts closing
    - 100 % plateau while the gate closes fast
    - linear decrease at ``trapezoid_decrease_rate`` %/s until it reaches 0

    Additional gates (``gate_2`` and above) follow the same perturbed-slope
    pattern as ``gate_0``.

    Parameters
    ----------
    n_gates : int
        Number of gate columns to generate (default 2).  Gates other than
        ``gate_1`` receive slightly perturbed slopes (±5 %) and noise (±10 %).
    trapezoid_decrease_rate : float or None
        Decrease rate of the trapezoid channel in %/s.  Defaults to
        ``2 * abs(slope_slow)``.
    trapezoid_rise_duration : float
        Duration of the linear rise from 0 to 100 % in seconds (default 1.0).
    start_datetime : datetime or None
        Start timestamp for the ``date_time`` column.  Defaults to
        ``datetime(2024, 1, 1)``.
    """
    if trapezoid_decrease_rate is None:
        trapezoid_decrease_rate = 2.0 * abs(slope_slow)

    rng = np.random.default_rng(seed)

    if start_datetime is None:
        start_datetime = datetime(2024, 1, 1)

    time_arr = np.arange(0, t_total, dt)

    # Build datetime column
    date_times = [start_datetime + timedelta(seconds=float(t)) for t in time_arr]

    # Build gate columns
    gate_data: dict[str, npt.NDArray[np.floating[Any]]] = {}
    for g in range(n_gates):
        noise_std_g = noise_std * (1.0 + 0.1 * abs(rng.standard_normal()))

        if g == 1:
            # Trapezoid: 0 → linear rise → plateau at 100 % → linear decrease → 0
            t_rise_end = t_start_closing + trapezoid_rise_duration
            rise = 100.0 * (time_arr - t_start_closing) / trapezoid_rise_duration
            decrease = np.maximum(
                0.0,
                100.0 - trapezoid_decrease_rate * (time_arr - t_slope_change),
            )
            position = np.where(
                time_arr < t_start_closing,
                0.0,
                np.where(
                    time_arr < t_rise_end,
                    rise,
                    np.where(time_arr < t_slope_change, 100.0, decrease),
                ),
            )
            position = np.clip(position, 0.0, 100.0)
        else:
            slope_fast_g = slope_fast * (1.0 + 0.05 * rng.standard_normal())
            slope_slow_g = slope_slow * (1.0 + 0.05 * rng.standard_normal())

            pos_at_start = plateau_high
            pos_at_slope_change = pos_at_start + slope_fast_g * (
                t_slope_change - t_start_closing
            )

            position = np.where(
                time_arr < t_start_closing,
                plateau_high,
                np.where(
                    time_arr < t_slope_change,
                    pos_at_start + slope_fast_g * (time_arr - t_start_closing),
                    np.where(
                        time_arr < t_end_closing,
                        pos_at_slope_change
                        + slope_slow_g * (time_arr - t_slope_change),
                        plateau_low,
                    ),
                ),
            )
            position = np.clip(position, plateau_low, plateau_high)

        position = position + rng.normal(0, noise_std_g, size=position.shape)

        gate_data[f"gate_{g}"] = position

    df = pl.DataFrame({"date_time": date_times, **gate_data}).cast(
        {"date_time": pl.Datetime("us")}
    )

    return GateData(
        df=df,
        breakpoints=[t_start_closing, t_slope_change, t_end_closing],
        slopes=[slope_fast, slope_slow],
        plateaus=(plateau_high, plateau_low),
        trapezoid_decrease_rate=trapezoid_decrease_rate,
    )


def plot_results(
    data: GateData,
    title: str,
    *,
    fitted_segments: dict[
        str,
        list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]],
    ]
    | None = None,
    detected_breakpoints: dict[str, list[float]] | None = None,
    estimated_slopes: dict[str, list[float]] | None = None,
) -> figure:
    """Plot raw data for all gates with optional fitted segments and breakpoints.

    Parameters are dicts keyed by gate column name to support multi-gate
    overlay on a single figure.
    """
    fig = figure(
        width=1200,
        height=600,
        title=title,
        x_axis_label="Time (s)",
        y_axis_label="Gate position (%)",
    )

    colors = Category10[10]
    time = data.time

    for g_idx, col in enumerate(data.gate_columns):
        color = colors[g_idx % 10]
        position = data.df[col].to_numpy()

        # Raw data scatter
        fig.scatter(
            time,
            position,
            marker="circle",
            color=color,
            alpha=0.2,
            size=2,
            legend_label=col,
        )

        # Fitted segments
        if fitted_segments and col in fitted_segments:
            for i, (t_seg, y_seg) in enumerate(fitted_segments[col]):
                kw: dict[str, Any] = {"line_width": 2, "color": color}
                if i == 0:
                    kw["legend_label"] = f"{col} fit"
                fig.line(t_seg, y_seg, **kw)

        # Detected breakpoints
        if detected_breakpoints and col in detected_breakpoints:
            for bp in detected_breakpoints[col]:
                fig.add_layout(
                    Span(
                        location=bp,
                        dimension="height",
                        line_color=color,
                        line_dash="dashed",
                        line_alpha=0.7,
                    )
                )
            # Single legend entry for this gate's breakpoints
            fig.line(
                [],
                [],
                line_color=color,
                line_dash="dashed",
                line_alpha=0.7,
                legend_label=f"{col} BPs",
            )

    # Ground truth breakpoints (shared across gates)
    for bp in data.breakpoints:
        fig.add_layout(
            Span(
                location=bp,
                dimension="height",
                line_color="green",
                line_dash="dotted",
                line_alpha=0.4,
            )
        )

    info_parts: list[str] = []
    if estimated_slopes:
        for col, slopes in estimated_slopes.items():
            for i, s in enumerate(slopes):
                info_parts.append(f"{col} slope {i + 1}: {s:.2f} %/s")
    info_parts.append(f"True slopes: {data.slopes[0]:.1f}, {data.slopes[1]:.1f} %/s")

    fig.add_layout(
        Label(
            x=10,
            y=10,
            x_units="screen",
            y_units="screen",
            text="\n".join(info_parts),
            text_font_size="9pt",
            background_fill_color="wheat",
            background_fill_alpha=0.8,
        )
    )

    fig.legend.location = "top_right"
    fig.legend.label_text_font_size = "8pt"
    fig.grid.grid_line_alpha = 0.3

    return fig


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    print(f"Generated {len(data.df)} samples, {len(data.gate_columns)} gates")
    print(f"Gate columns: {data.gate_columns}")
    print(f"True breakpoints: {data.breakpoints}")
    print(f"True slopes: {data.slopes}")
    fig = plot_results(data, "Synthetic Gate Closing Data")
    show(fig)
