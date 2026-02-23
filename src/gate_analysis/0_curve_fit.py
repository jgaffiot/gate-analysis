"""Method 0: Direct curve_fit of a 4-piece piecewise linear model.

Fits the known signal structure (high plateau → fast ramp → slow ramp →
low plateau) directly using scipy.optimize.curve_fit with 6 free parameters:

    t1, t2, t3  — three breakpoints
    v_high      — high plateau value
    a1, a2      — fast and slow slopes

The low plateau value is determined by continuity.  Because the signal
topology is hard-coded, no model selection is needed and the covariance
matrix returned by curve_fit provides standard errors on all parameters,
including slopes, at negligible cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from gate_analysis.common import GateData, generate_synthetic_data, plot_results


def _gate_model(
    t: npt.NDArray[np.floating[Any]],
    t1: float,
    t2: float,
    t3: float,
    v_high: float,
    a1: float,
    a2: float,
) -> npt.NDArray[np.floating[Any]]:
    """4-piece piecewise linear model with known gate-closing topology."""
    v2 = v_high + a1 * (t2 - t1)  # value at t2 (fast→slow transition)
    v3 = v2 + a2 * (t3 - t2)  # value at t3 (end of closing)
    return np.piecewise(
        t,
        [t < t1, (t >= t1) & (t < t2), (t >= t2) & (t < t3), t >= t3],
        [
            v_high,
            lambda t: v_high + a1 * (t - t1),
            lambda t: v2 + a2 * (t - t2),
            v3,
        ],
    )


def curve_fit_piecewise(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    p0: list[float] | None = None,
) -> dict[str, Any]:
    """Fit the gate closing signal with a direct 4-piece piecewise linear model.

    Uses scipy.optimize.curve_fit (Trust Region Reflective) to estimate all
    parameters simultaneously.  A coarse initial guess based on time-range
    fractions is sufficient for convergence; the optimizer then refines all
    six parameters jointly against the least-squares criterion.

    Parameters
    ----------
    time : array
        Time values.
    position : array
        Gate position values (%).
    p0 : list of float, optional
        Initial guess [t1, t2, t3, v_high, a1, a2].
        Defaults to [20 %, 50 %, 75 % of time range, max(position), −20, −3].

    Returns
    -------
    dict with keys: breakpoints, slopes, slope_stderr, params, pcov
    """
    if p0 is None:
        t_range = float(time[-1] - time[0])
        v_max = float(np.max(position))
        p0 = [
            float(time[0]) + 0.20 * t_range,
            float(time[0]) + 0.50 * t_range,
            float(time[0]) + 0.75 * t_range,
            v_max,
            -20.0,
            -3.0,
        ]

    t_min, t_max = float(time[0]), float(time[-1])
    lower = [t_min, t_min, t_min, -np.inf, -np.inf, -np.inf]
    upper = [t_max, t_max, t_max, np.inf, 0.0, 0.0]

    popt, pcov = curve_fit(
        _gate_model,
        time,
        position,
        p0=p0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=10_000,
    )

    t1, t2, t3, v_high, a1, a2 = popt
    perr = np.sqrt(np.diag(pcov))

    print("=== Method 0: Direct curve_fit (scipy) ===")
    print(f"Breakpoints: {t1:.3f}, {t2:.3f}, {t3:.3f} s")
    print(f"Fast slope:  {a1:.2f} ± {perr[4]:.2f} %/s")
    print(f"Slow slope:  {a2:.2f} ± {perr[5]:.2f} %/s")
    print(f"High plateau: {v_high:.2f} %")

    return {
        "breakpoints": [t1, t2, t3],
        "slopes": [a1, a2],
        "slope_stderr": [perr[4], perr[5]],
        "params": popt,
        "pcov": pcov,
    }


def _build_segments(
    data: GateData, result: dict[str, Any]
) -> list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    """Build fitted line segments for plotting."""
    y_hat = _gate_model(data.time, *result["params"])
    bps = [data.time[0], *result["breakpoints"], data.time[-1]]
    segments = []
    for i in range(len(bps) - 1):
        mask = (data.time >= bps[i]) & (data.time <= bps[i + 1])
        segments.append((data.time[mask], y_hat[mask]))
    return segments


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    result = curve_fit_piecewise(data.time, data.position)
    segments = _build_segments(data, result)
    fig = plot_results(
        data,
        "Method 0: Direct Curve Fit (scipy)",
        fitted_segments=segments,
        detected_breakpoints=result["breakpoints"],
        estimated_slopes=result["slopes"],
    )
    show(fig)
