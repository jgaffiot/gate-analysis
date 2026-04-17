"""Method 3: CPOP-like continuous piecewise-linear segmentation with L0 penalty.

Custom dynamic-programming implementation inspired by Fearnhead et al. (2019).
Finds the optimal continuous piecewise-linear fit minimizing residual sum of
squares plus an L0 penalty on slope changes.

Optimization uses unconstrained per-segment OLS (fast, correct BIC landscape).
The final output uses a continuous piecewise-linear fit for smooth plotting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from gate_analysis.common import GateData, generate_synthetic_data, plot_results


def _unconstrained_rss(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    breakpoint_indices: list[int],
) -> float:
    """Sum of per-segment OLS residuals (unconstrained, no continuity requirement).

    Much faster than the constrained fit and gives a well-behaved BIC landscape:
    the continuity constraint can add hundreds of RSS units at the true breakpoints,
    corrupting BIC model-selection when comparing different breakpoint counts.
    """
    bps = [0, *breakpoint_indices, len(time)]
    total = 0.0
    for i in range(len(bps) - 1):
        t_seg = time[bps[i] : bps[i + 1]]
        p_seg = position[bps[i] : bps[i + 1]]
        t_mean = t_seg.mean()
        p_mean = p_seg.mean()
        xm = t_seg - t_mean
        denom = float(np.dot(xm, xm))
        if denom < 1e-12:
            total += float(np.dot(p_seg - p_mean, p_seg - p_mean))
        else:
            slope = float(np.dot(xm, p_seg - p_mean) / denom)
            resid = p_seg - p_mean - slope * xm
            total += float(np.dot(resid, resid))
    return total


def _fit_continuous_piecewise_linear(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    breakpoint_indices: list[int],
) -> tuple[float, list[float], npt.NDArray[np.floating[Any]]]:
    """Fit a continuous piecewise-linear function at given breakpoints.

    Returns (rss, slopes, fitted_values).  Called once per n_bps after
    the optimal breakpoints are found, purely for the final smooth output.

    Design matrix column k+1: 0 before segment k, (time - t_start) within
    segment k, constant (t_end - t_start) after segment k ends.  This
    encodes a continuous model where each coeff[k+1] is the slope of segment k.
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
    dt = float(time[1] - time[0]) if n > 1 else 0.01
    if penalty is None:
        penalty = 2.0 * np.log(n)

    # Minimum segment length: enough samples for a meaningful linear fit
    min_seg = max(10, n // 100)
    t_lo = float(time[min_seg])
    t_hi = float(time[n - 1 - min_seg])

    best_cost = np.inf
    best_result: dict[str, Any] = {}
    bic_scores: list[tuple[int, float]] = []

    # Helper: convert time values → valid, sorted integer indices
    def _to_idxs(ts: list[float]) -> list[int]:
        return sorted(
            int(np.clip(np.searchsorted(time, t), min_seg, n - 1 - min_seg)) for t in ts
        )

    def _valid(idxs: list[int]) -> bool:
        all_bps = [0, *idxs, n]
        return all(
            all_bps[i + 1] - all_bps[i] >= min_seg for i in range(len(all_bps) - 1)
        )

    counts = (
        [n_breakpoints] if n_breakpoints is not None else range(0, max_breakpoints + 1)
    )
    for n_bps in counts:
        if n_bps == 0:
            rss_unc = _unconstrained_rss(time, position, [])
            cost = rss_unc
            bic_scores.append((0, cost))
            if cost < best_cost:
                best_cost = cost
                rss, slopes, fitted = _fit_continuous_piecewise_linear(
                    time, position, []
                )
                best_result = {
                    "breakpoint_indices": [],
                    "slopes": slopes,
                    "fitted": fitted,
                    "rss": rss,
                    "n_breakpoints": 0,
                }
            continue

        # Phase 1: forward-greedy initialisation with unconstrained RSS.
        # Add one breakpoint at a time at the position giving the largest RSS
        # reduction.  Unconstrained RSS gives a smooth, correct landscape for
        # BIC — the continuity constraint can add 500–1500 RSS units at the
        # true breakpoints, making BIC over-select breakpoints.
        n_search = 30
        t_grid = np.linspace(t_lo, t_hi, n_search + 2)[1:-1]
        greedy_times: list[float] = []
        for _step in range(n_bps):
            best_rss_g = np.inf
            best_t_g = float(t_grid[len(t_grid) // 2])
            for t_cand in t_grid:
                trial_times = sorted(greedy_times + [float(t_cand)])
                idxs = _to_idxs(trial_times)
                if not _valid(idxs):
                    continue
                rss_g = _unconstrained_rss(time, position, idxs)
                if rss_g < best_rss_g:
                    best_rss_g = rss_g
                    best_t_g = float(t_cand)
            greedy_times.append(best_t_g)

        x0 = np.array(sorted(greedy_times))

        # Phase 2: Nelder-Mead joint refinement from the greedy solution.
        # All breakpoints are optimised simultaneously, so the simplex can
        # escape the shallow troughs that trap coordinate-descent.
        def _objective(
            t_bps: npt.NDArray[np.floating[Any]],
        ) -> float:
            idxs = _to_idxs(list(t_bps))
            if not _valid(idxs):
                return 1e15
            return _unconstrained_rss(time, position, idxs)

        res = minimize(
            _objective,
            x0,
            method="Nelder-Mead",
            options={
                "xatol": dt * 2,
                "fatol": 1.0,
                "maxiter": 5_000,
                "adaptive": True,
            },
        )
        bp_indices = _to_idxs(list(res.x))

        # BIC uses unconstrained RSS (2 params per segment → 2*log(n) per breakpoint)
        rss_unc = _unconstrained_rss(time, position, bp_indices)
        cost = rss_unc + penalty * n_bps
        bic_scores.append((n_bps, cost))

        # Continuous fit computed once per n_bps, only for final output
        rss, slopes, fitted = _fit_continuous_piecewise_linear(
            time, position, bp_indices
        )

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

    print("=== Method 3: CPOP-like Continuous Piecewise Linear (L0 penalty) ===")
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


def analyze(
    data: GateData,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run CPOP piecewise-linear analysis on all gate columns.

    Returns (results, segments) dicts keyed by gate column name.
    """
    time = data.time
    results: dict[str, dict[str, Any]] = {}
    segments: dict[str, Any] = {}
    for col in data.gate_columns:
        position = data.df[col].to_numpy()
        r = cpop_piecewise_linear(time, position)
        results[col] = r
        segments[col] = _build_segments(data, r)
    return results, segments


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    results, segments = analyze(data)
    fig = plot_results(
        data,
        "Method 3: CPOP-like Continuous Piecewise Linear",
        fitted_segments=segments,
        detected_breakpoints={c: r["breakpoints"] for c, r in results.items()},
        estimated_slopes={c: r["slopes"] for c, r in results.items()},
    )
    show(fig)
