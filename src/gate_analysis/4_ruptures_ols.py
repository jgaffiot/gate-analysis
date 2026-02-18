"""Option D: Change-point detection with ruptures + OLS slope estimation.

Uses the ruptures library for segmentation (PELT or Dynp with a linear cost
model), then fits OLS on each segment to estimate slopes with confidence
intervals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import ruptures as rpt
from scipy import stats

from gate_analysis.common import GateData, generate_synthetic_data, plot_results


def _ols_segment(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """Fit OLS on a single segment and return slope, intercept, and CI."""
    result = stats.linregress(time, position)
    n = len(time)
    # 95% confidence interval on slope
    t_crit = stats.t.ppf(0.975, df=n - 2)
    slope_ci = t_crit * result.stderr
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_squared": float(result.rvalue**2),
        "slope_stderr": float(result.stderr),
        "slope_ci_95": (float(result.slope - slope_ci), float(result.slope + slope_ci)),
    }


def ruptures_ols(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    *,
    n_breakpoints: int | None = 3,
    penalty: float | None = None,
    method: str = "dynp",
) -> dict[str, Any]:
    """Detect change points with ruptures and estimate slopes with OLS.

    Parameters
    ----------
    time : array
        Time values.
    position : array
        Gate position values (%).
    n_breakpoints : int or None
        Number of breakpoints (used with dynp). If None, penalty must be set.
    penalty : float or None
        Penalty for PELT. If None, n_breakpoints must be set.
    method : str
        Search method: "dynp" (dynamic programming) or "pelt".

    Returns
    -------
    dict with keys: breakpoints, segments, slopes
    """
    signal = position.reshape(-1, 1)

    if method == "pelt":
        algo = rpt.Pelt(model="l2", min_size=20).fit(signal)
        if penalty is None:
            penalty = 10.0
        change_indices = algo.predict(pen=penalty)
    else:
        algo = rpt.Dynp(model="l2", min_size=20).fit(signal)
        if n_breakpoints is None:
            n_breakpoints = 3
        change_indices = algo.predict(n_bkps=n_breakpoints)

    # change_indices includes the last point (n), remove it
    bp_indices = [i for i in change_indices if i < len(time)]

    breakpoints = [
        float(time[i]) if i < len(time) else float(time[-1]) for i in bp_indices
    ]

    # Build segments and fit OLS on each
    all_bounds = [0, *bp_indices, len(time)]
    segments_info: list[dict[str, Any]] = []

    for i in range(len(all_bounds) - 1):
        i_start, i_end = all_bounds[i], all_bounds[i + 1]
        t_seg = time[i_start:i_end]
        p_seg = position[i_start:i_end]

        if len(t_seg) < 3:
            segments_info.append({"slope": 0.0, "intercept": float(np.mean(p_seg))})
            continue

        ols = _ols_segment(t_seg, p_seg)
        segments_info.append(
            {
                **ols,
                "t_start": float(t_seg[0]),
                "t_end": float(t_seg[-1]),
                "n_points": len(t_seg),
            }
        )

    # Identify the two ramp segments (largest absolute slopes)
    slopes_abs = [
        (i, abs(seg.get("slope", 0.0))) for i, seg in enumerate(segments_info)
    ]
    slopes_abs.sort(key=lambda x: x[1], reverse=True)
    ramp_indices = [slopes_abs[0][0], slopes_abs[1][0]]
    ramp_indices.sort()

    estimated_slopes = [segments_info[i]["slope"] for i in ramp_indices]

    print("=== Option D: ruptures + OLS ===")
    print(f"Change points (indices): {bp_indices}")
    print(f"Breakpoints (time): {[f'{bp:.3f}' for bp in breakpoints]}")
    for i, seg in enumerate(segments_info):
        slope = seg.get("slope", 0.0)
        ci = seg.get("slope_ci_95", (0, 0))
        print(
            f"  Segment {i}: slope={slope:.2f} %/s, 95% CI=[{ci[0]:.2f}, {ci[1]:.2f}]"
        )
    print(f"Ramp slopes: {[f'{s:.2f}' for s in estimated_slopes]}")

    return {
        "breakpoints": breakpoints,
        "breakpoint_indices": bp_indices,
        "segments": segments_info,
        "slopes": estimated_slopes,
    }


def _build_segments(
    data: GateData, result: dict[str, Any]
) -> list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    """Build fitted line segments for plotting."""
    bp_indices = result["breakpoint_indices"]
    all_bounds = [0, *bp_indices, len(data.time)]
    segments = []
    for i, seg in enumerate(result["segments"]):
        i_start, i_end = all_bounds[i], all_bounds[i + 1]
        t_seg = data.time[i_start:i_end]
        y_seg = seg.get("intercept", 0.0) + seg.get("slope", 0.0) * t_seg
        segments.append((t_seg, y_seg))
    return segments


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    result = ruptures_ols(data.time, data.position)
    segments = _build_segments(data, result)
    fig = plot_results(
        data,
        "Option D: ruptures + OLS",
        fitted_segments=segments,
        detected_breakpoints=result["breakpoints"],
        estimated_slopes=result["slopes"],
    )
    show(fig)
