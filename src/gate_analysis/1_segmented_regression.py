"""Option A: Segmented regression using the piecewise-regression package.

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

    Returns
    -------
    dict with keys: breakpoints, slopes, summary, model
    """
    model = piecewise_regression.Fit(
        time.tolist(),
        position.tolist(),
        n_breakpoints=n_breakpoints,
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

    # all_slopes = [plateau_high, fast_slope, slow_slope, plateau_low]
    # Keep only the two closing slopes for reporting (consistent with other methods)
    slopes = all_slopes[1:3]

    print("=== Option A: Segmented Regression (Muggeo) ===")
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


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    result = segmented_regression(data.time, data.position)
    segments = _build_segments(data, result)
    fig = plot_results(
        data,
        "Option A: Segmented Regression (Muggeo)",
        fitted_segments=segments,
        detected_breakpoints=result["breakpoints"],
        estimated_slopes=result["slopes"],
    )
    show(fig)
