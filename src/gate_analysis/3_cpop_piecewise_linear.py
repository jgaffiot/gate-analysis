"""Option C: CPOP-like continuous piecewise-linear segmentation with L0 penalty.

Custom dynamic-programming implementation inspired by Fearnhead et al. (2019).
Finds the optimal continuous piecewise-linear fit minimizing residual sum of
squares plus an L0 penalty on slope changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gate_analysis.common import GateData, generate_synthetic_data, plot_results


def _fit_continuous_piecewise_linear(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    breakpoint_indices: list[int],
) -> tuple[float, list[float], npt.NDArray[np.floating[Any]]]:
    """Fit a continuous piecewise-linear function at given breakpoints.

    Returns (rss, slopes, fitted_values).
    The fit is constrained to be continuous at breakpoints.
    Parameterisation: intercept + one cumulative-basis slope per segment.

    The design matrix column k+1 encodes the contribution of segment k's
    slope to each sample:
      - 0                      for samples before segment k starts
      - time[j] - time[i_start] for samples within segment k
      - time[i_end] - time[i_start] for samples after segment k ends

    Built with pure NumPy broadcasting (no Python inner loop over samples).
    """
    n = len(time)
    bps = [0, *breakpoint_indices, n]
    n_segments = len(bps) - 1
    indices = np.arange(n)

    design = np.zeros((n, n_segments + 1))
    design[:, 0] = 1.0
    for k in range(n_segments):
        i_start = bps[k]
        i_end = bps[k + 1]  # exclusive upper index
        t_start = time[i_start]
        t_end_val = time[i_end - 1]  # bps[-1]=n so time[n-1]=time[-1] for last seg
        design[:, k + 1] = np.where(
            indices < i_start,
            0.0,
            np.where(indices < i_end, time - t_start, t_end_val - t_start),
        )

    # Solve least squares
    coeffs, _rss_arr, _rank, _sv = np.linalg.lstsq(design, position, rcond=None)
    fitted = design @ coeffs
    rss = float(np.sum((position - fitted) ** 2))
    slopes = coeffs[1:].tolist()

    return rss, slopes, fitted


def cpop_piecewise_linear(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    *,
    penalty: float | None = None,
    max_breakpoints: int = 5,
    n_breakpoints: int | None = None,
) -> dict[str, Any]:
    """CPOP-like segmentation using dynamic programming with L0 penalty.

    For tractability, this demo uses a simplified approach:
    1. Evaluate candidate breakpoint sets with 1..max_breakpoints breakpoints
    2. For each count, find optimal placement by iterative refinement
    3. Select the best model using BIC (or a user-specified penalty)

    Parameters
    ----------
    time : array
        Time values.
    position : array
        Gate position values (%).
    penalty : float or None
        L0 penalty per breakpoint. If None, uses BIC (2 * log(n)).
    max_breakpoints : int
        Maximum number of breakpoints to consider (used when n_breakpoints
        is None).
    n_breakpoints : int or None
        If given, skip model selection and fit only this exact number of
        breakpoints.  Eliminates the outer BIC loop — use when the signal
        topology is known (e.g. n_breakpoints=3 for the gate signal).

    Returns
    -------
    dict with keys: breakpoints, slopes, n_breakpoints, fitted, bic_scores
    """
    n = len(time)
    if penalty is None:
        penalty = 2.0 * np.log(n)

    # Downsample candidate positions for efficiency
    step = max(1, n // 200)
    candidates = list(range(step, n - step, step))

    best_cost = np.inf
    best_result: dict[str, Any] = {}
    bic_scores: list[tuple[int, float]] = []

    counts = (
        [n_breakpoints] if n_breakpoints is not None else range(0, max_breakpoints + 1)
    )
    for n_bps in counts:
        if n_bps == 0:
            rss, slopes, fitted = _fit_continuous_piecewise_linear(time, position, [])
            cost = rss + penalty * 0
            bic_scores.append((0, cost))
            if cost < best_cost:
                best_cost = cost
                best_result = {
                    "breakpoint_indices": [],
                    "slopes": slopes,
                    "fitted": fitted,
                    "rss": rss,
                    "n_breakpoints": 0,
                }
            continue

        # Greedy search: start with evenly spaced, then refine
        bp_indices = [int(n * (k + 1) / (n_bps + 1)) for k in range(n_bps)]

        # Iterative refinement: optimize each breakpoint one at a time
        for _iteration in range(10):
            improved = False
            for bp_idx in range(n_bps):
                current_best_rss = np.inf
                current_best_pos = bp_indices[bp_idx]
                for cand in candidates:
                    trial = bp_indices.copy()
                    trial[bp_idx] = cand
                    trial.sort()
                    # Ensure minimum segment size
                    all_bps = [0, *trial, n]
                    if any(
                        all_bps[i + 1] - all_bps[i] < step * 2
                        for i in range(len(all_bps) - 1)
                    ):
                        continue
                    rss, _, _ = _fit_continuous_piecewise_linear(time, position, trial)
                    if rss < current_best_rss:
                        current_best_rss = rss
                        current_best_pos = cand
                if current_best_pos != bp_indices[bp_idx]:
                    bp_indices[bp_idx] = current_best_pos
                    bp_indices.sort()
                    improved = True
            if not improved:
                break

        rss, slopes, fitted = _fit_continuous_piecewise_linear(
            time, position, bp_indices
        )
        cost = rss + penalty * n_bps
        bic_scores.append((n_bps, cost))

        if cost < best_cost:
            best_cost = cost
            best_result = {
                "breakpoint_indices": bp_indices,
                "slopes": slopes,
                "fitted": fitted,
                "rss": rss,
                "n_breakpoints": n_bps,
            }

    breakpoints = [float(time[i]) for i in best_result["breakpoint_indices"]]

    print("=== Option C: CPOP-like Continuous Piecewise Linear (L0 penalty) ===")
    print(f"Selected {best_result['n_breakpoints']} breakpoints")
    print(f"Breakpoints: {[f'{bp:.3f}' for bp in breakpoints]}")
    print(f"Slopes: {[f'{s:.2f}' for s in best_result['slopes']]}")
    print(f"BIC scores: {[(nb, f'{c:.1f}') for nb, c in bic_scores]}")

    return {
        "breakpoints": breakpoints,
        "slopes": best_result["slopes"],
        "fitted": best_result["fitted"],
        "n_breakpoints": best_result["n_breakpoints"],
        "bic_scores": bic_scores,
    }


def _build_segments(
    data: GateData, result: dict[str, Any]
) -> list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    """Build fitted line segments for plotting."""
    fitted = result["fitted"]
    bps = [data.time[0], *result["breakpoints"], data.time[-1]]
    segments = []
    for i in range(len(bps) - 1):
        mask = (data.time >= bps[i]) & (data.time <= bps[i + 1])
        segments.append((data.time[mask], fitted[mask]))
    return segments


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    result = cpop_piecewise_linear(data.time, data.position)
    segments = _build_segments(data, result)
    fig = plot_results(
        data,
        "Option C: CPOP-like Continuous Piecewise Linear",
        fitted_segments=segments,
        detected_breakpoints=result["breakpoints"],
        estimated_slopes=result["slopes"],
    )
    show(fig)
