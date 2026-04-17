"""Method 1: Segmented regression using the piecewise-regression package.

Uses Muggeo's iterative method to fit a piecewise-linear model with
automatically estimated breakpoints and slopes, including confidence intervals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import piecewise_regression

from gate_analysis.common import GateData, generate_synthetic_data, plot_results


def segmented_regression(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    n_breakpoints: int = 3,
    start_values: list[float] | None = None,
) -> dict[str, Any]:
    """Fit a piecewise-linear model using Muggeo's segmented regression.

    Parameters
    ----------
    time : array
        Time values.
    position : array
        Gate position values (%).
    n_breakpoints : int
        Number of breakpoints to fit (default 3: start of closing,
        slope change, end of closing).
    start_values : list of float, optional
        Initial breakpoint positions passed to Muggeo's algorithm.
        Required when the signal has very abrupt slope changes (e.g.
        the trapezoid's ~100 %/s rise) that cause convergence failure
        from the default evenly-spaced initialisation.

    Returns
    -------
    dict with keys: breakpoints, slopes, summary, model
    """
    model = piecewise_regression.Fit(
        time.tolist(),
        position.tolist(),
        n_breakpoints=n_breakpoints,
        start_values=start_values,
        n_boot=10,
    )

    results = model.get_results()
    if results is None:
        msg = "Piecewise regression failed to converge"
        raise RuntimeError(msg)

    breakpoints: list[float] = []
    # piecewise-regression returns alpha_k = absolute slope of segment k
    # (beta_k = kink magnitude = alpha_{k+1} - alpha_k, also in the results)
    all_slopes: list[float] = []

    for bp_key in sorted(k for k in results["estimates"] if k.startswith("breakpoint")):
        bp_est = results["estimates"][bp_key]["estimate"]
        breakpoints.append(float(bp_est))

    for alpha_key in sorted(k for k in results["estimates"] if k.startswith("alpha")):
        slope_est = results["estimates"][alpha_key]["estimate"]
        all_slopes.append(float(slope_est))

    # Pick the two segments with the largest absolute slope for reporting.
    # For the gate (3 bps): plateau, fast, slow, plateau → indices 1, 2.
    # For the trapezoid (4 bps): flat, rise, plateau, decrease, flat → indices 1, 3.
    indexed = sorted(enumerate(all_slopes), key=lambda x: abs(x[1]), reverse=True)
    top2_idx = sorted(i for i, _ in indexed[:2])
    slopes = [all_slopes[i] for i in top2_idx]

    print("=== Method 1: Segmented Regression (Muggeo) ===")
    print(f"Breakpoints: {[f'{bp:.3f}' for bp in breakpoints]}")
    print(f"Slopes: {[f'{s:.2f}' for s in slopes]}")
    model.summary()

    return {
        "breakpoints": breakpoints,
        "slopes": slopes,
        "all_slopes": all_slopes,
        "summary": results,
        "model": model,
    }


def _build_segments(
    data: GateData, result: dict[str, Any]
) -> list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    """Build fitted line segments for plotting."""
    model = result["model"]
    y_hat = np.array(model.predict(data.time.tolist()))
    bps = [data.time[0], *result["breakpoints"], data.time[-1]]
    segments = []
    for i in range(len(bps) - 1):
        mask = (data.time >= bps[i]) & (data.time <= bps[i + 1])
        segments.append((data.time[mask], y_hat[mask]))
    return segments


def analyze(
    data: GateData,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run segmented regression on all gate columns.

    Returns (results, segments) dicts keyed by gate column name.
    """
    time = data.time
    results: dict[str, dict[str, Any]] = {}
    segments: dict[str, Any] = {}
    for g_idx, col in enumerate(data.gate_columns):
        position = data.df[col].to_numpy()
        is_trapezoid = g_idx == 1 and data.trapezoid_decrease_rate is not None
        if is_trapezoid:
            # Provide initial guesses: rise_start, rise_end, decrease_start, zero_crossing.
            # Muggeo's algorithm diverges on the steep rise (~100 %/s) without them.
            t_lo, t_hi = float(time[0]), float(time[-1])
            t_range = t_hi - t_lo
            start_values = [
                t_lo + 0.10 * t_range,
                t_lo + 0.15 * t_range,
                t_lo + 0.25 * t_range,
                t_lo + 0.75 * t_range,
            ]
            r = segmented_regression(
                time, position, n_breakpoints=4, start_values=start_values
            )
        else:
            r = segmented_regression(time, position, n_breakpoints=3)
        results[col] = r
        segments[col] = _build_segments(data, r)
    return results, segments


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    results, segments = analyze(data)
    fig = plot_results(
        data,
        "Method 1: Segmented Regression (Muggeo)",
        fitted_segments=segments,
        detected_breakpoints={c: r["breakpoints"] for c, r in results.items()},
        estimated_slopes={c: r["slopes"] for c, r in results.items()},
    )
    show(fig)
