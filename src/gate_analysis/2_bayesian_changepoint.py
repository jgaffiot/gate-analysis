"""Method 2: Bayesian 4-regime change-point model — MAP + Laplace approximation.

Defines the same probabilistic model as a full Bayesian treatment:
  - 3 ordered change points via positive offsets (tau1 free; d12, d23 > 0)
  - Gaussian priors on plateau levels and slopes
  - Half-Normal priors on the gap widths and noise standard deviation
  - Gaussian likelihood

Inference is done with two cheap steps instead of ADVI/MCMC:

1. **MAP** — minimise the negative log-posterior with
   ``scipy.optimize.minimize`` (Nelder-Mead, gradient-free, robust to the
   piecewise kinks in the likelihood landscape).

2. **Laplace approximation** — compute the numerical Hessian of the negative
   log-posterior at the MAP.  Its inverse is the Gaussian posterior covariance;
   the square-root of the diagonal gives parameter standard errors and, by
   the delta method, 95 % credible intervals on breakpoints and slopes.

For a unimodal, well-identified posterior like this one the Laplace
approximation is equivalent to ADVI's Gaussian variational family, but runs
in ~0.5 s instead of ~8 s without any probabilistic-programming dependency.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from gate_analysis.common import GateData, generate_synthetic_data, plot_results


def _numerical_hessian(
    f: Callable[[npt.NDArray[np.floating[Any]]], float],
    x: npt.NDArray[np.floating[Any]],
    eps: float = 1e-4,
) -> npt.NDArray[np.floating[Any]]:
    """Numerical Hessian via central finite differences (O(n²) evaluations)."""
    n = len(x)
    f0 = f(x)
    hess = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        hess[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / eps**2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = eps
            h = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                4.0 * eps**2
            )
            hess[i, j] = hess[j, i] = h
    return hess


def bayesian_changepoint(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    *,
    seed: int = 42,  # kept for API compatibility
) -> dict[str, Any]:
    """Fit a Bayesian 4-regime change-point model via MAP + Laplace approximation.

    Probabilistic model
    -------------------
    tau1       ~ Uniform(t_min, t_max)
    d12, d23   ~ HalfNormal(sigma=3)          → tau2 = tau1+d12, tau3 = tau2+d23
    level_high ~ Normal(98, 10)
    level_low  ~ Normal(2, 10)
    slope_fast, slope_slow ~ Normal(0, 15)
    sigma_obs  ~ HalfNormal(5)
    obs        ~ Normal(mu(t), sigma_obs)

    The internal parameter vector is
        [tau1, d12, d23, v_high, v_low, a1, a2, log_sigma]

    Parameters
    ----------
    time : array
        Time values.
    position : array
        Gate position values (%).
    seed : int
        Unused (kept for API compatibility with earlier ADVI version).

    Returns
    -------
    dict with keys: breakpoints, slopes, plateaus, slope_stderr, tau_stderr, cov
    """
    n = len(time)
    t_min, t_max = float(time.min()), float(time.max())
    t_range = t_max - t_min

    # ------------------------------------------------------------------
    # Negative log-posterior
    # params = [tau1, d12, d23, v_high, v_low, a1, a2, log_sigma]
    # ------------------------------------------------------------------
    def neg_log_posterior(params: npt.NDArray[np.floating[Any]]) -> float:
        tau1, d12, d23, v_high, v_low, a1, a2, log_sigma = params
        if d12 <= 0.0 or d23 <= 0.0:
            return 1e15
        tau2 = tau1 + d12
        tau3 = tau2 + d23
        sigma = np.exp(log_sigma)

        v2 = v_high + a1 * (tau2 - tau1)
        mu = np.piecewise(
            time,
            [
                time < tau1,
                (time >= tau1) & (time < tau2),
                (time >= tau2) & (time < tau3),
                time >= tau3,
            ],
            [
                v_high,
                lambda t: v_high + a1 * (t - tau1),
                lambda t: v2 + a2 * (t - tau2),
                v_low,
            ],
        )

        # Negative log-likelihood
        nll = 0.5 * np.sum(((position - mu) / sigma) ** 2) + n * log_sigma

        # Negative log-priors (constants dropped)
        nlp = (
            0.5 * (d12 / 3.0) ** 2  # d12    ~ HalfNormal(3)
            + 0.5 * (d23 / 3.0) ** 2  # d23    ~ HalfNormal(3)
            + 0.5 * ((v_high - 98.0) / 10.0) ** 2  # level_high ~ Normal(98,10)
            + 0.5 * ((v_low - 2.0) / 10.0) ** 2  # level_low  ~ Normal(2,10)
            + 0.5 * (a1 / 15.0) ** 2  # slope_fast ~ Normal(0,15)
            + 0.5 * (a2 / 15.0) ** 2  # slope_slow ~ Normal(0,15)
            + 0.5 * (sigma / 5.0) ** 2  # sigma_obs  ~ HalfNormal(5)
        )
        return nll + nlp

    # ------------------------------------------------------------------
    # MAP estimation
    # ------------------------------------------------------------------
    x0 = np.array(
        [
            t_min + 0.20 * t_range,  # tau1
            0.30 * t_range,  # d12
            0.35 * t_range,  # d23
            float(np.max(position)),  # v_high
            float(np.min(position)),  # v_low
            -20.0,  # a1 (fast slope)
            -3.0,  # a2 (slow slope)
            0.0,  # log_sigma → sigma=1
        ]
    )

    result = minimize(
        neg_log_posterior,
        x0,
        method="Powell",
        options={"xtol": 1e-6, "ftol": 1e-6, "maxiter": 100_000},
    )

    tau1, d12, d23, v_high, v_low, a1, a2, log_sigma = result.x
    tau2 = tau1 + d12
    tau3 = tau2 + d23
    sigma_noise = np.exp(log_sigma)

    # ------------------------------------------------------------------
    # Laplace approximation: H = d²(-log p) / dθ²  →  cov ≈ H⁻¹
    # ------------------------------------------------------------------
    hess = _numerical_hessian(neg_log_posterior, result.x)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.full((8, 8), np.nan)

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    # Propagate to derived quantities via delta method
    # tau2 = tau1 + d12
    se_tau2 = float(np.sqrt(max(0.0, cov[0, 0] + cov[1, 1] + 2.0 * cov[0, 1])))
    # tau3 = tau1 + d12 + d23
    se_tau3 = float(
        np.sqrt(
            max(
                0.0,
                cov[0, 0]
                + cov[1, 1]
                + cov[2, 2]
                + 2.0 * cov[0, 1]
                + 2.0 * cov[0, 2]
                + 2.0 * cov[1, 2],
            )
        )
    )

    print("=== Method 2: Bayesian MAP + Laplace (scipy) ===")
    print(
        f"Breakpoints: tau1={tau1:.3f} ± {se[0]:.3f} s, "
        f"tau2={tau2:.3f} ± {se_tau2:.3f} s, "
        f"tau3={tau3:.3f} ± {se_tau3:.3f} s"
    )
    print(
        f"Fast slope: {a1:.2f} ± {se[5]:.2f} %/s"
        f"  →  95 % CI: [{a1 - 1.96 * se[5]:.2f}, {a1 + 1.96 * se[5]:.2f}]"
    )
    print(
        f"Slow slope: {a2:.2f} ± {se[6]:.2f} %/s"
        f"  →  95 % CI: [{a2 - 1.96 * se[6]:.2f}, {a2 + 1.96 * se[6]:.2f}]"
    )
    print(
        f"High plateau: {v_high:.1f} %  "
        f"Low plateau: {v_low:.1f} %  "
        f"Noise σ: {sigma_noise:.2f} %"
    )

    return {
        "breakpoints": [tau1, tau2, tau3],
        "slopes": [a1, a2],
        "plateaus": (v_high, v_low),
        "slope_stderr": [float(se[5]), float(se[6])],
        "tau_stderr": [float(se[0]), se_tau2, se_tau3],
        "cov": cov,
    }


def bayesian_changepoint_trapezoid(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """Fit a Bayesian trapezoid change-point model via MAP + Laplace approximation.

    Probabilistic model
    -------------------
    tau1            ~ Uniform(t_min, t_max)
    d12             ~ HalfNormal(sigma=3)     → tau2 = tau1 + d12 (decrease start)
    log_rise_rate   ~ Normal(log(100), 0.5)   → rise_rate = exp(log_rise_rate)
    log_dec_rate    ~ Normal(log(10), 1.0)    → dec_rate  = exp(log_dec_rate)
    sigma_obs       ~ HalfNormal(5)

    The internal parameter vector is
        [tau1, d12, log_rise_rate, log_dec_rate, log_sigma]

    Returns breakpoints [tau1, t_rise_end, tau2] and slopes [rise_rate, -dec_rate].
    """
    n = len(time)
    t_min, t_max = float(time.min()), float(time.max())
    t_range = t_max - t_min

    def neg_log_posterior(params: npt.NDArray[np.floating[Any]]) -> float:
        tau1, d12, log_rise_rate, log_dec_rate, log_sigma = params
        if d12 <= 0.0:
            return 1e15
        tau2 = tau1 + d12
        rise_rate = np.exp(log_rise_rate)
        dec_rate = np.exp(log_dec_rate)
        sigma = np.exp(log_sigma)

        t_rise_end = tau1 + 100.0 / rise_rate
        rise = rise_rate * (time - tau1)
        decrease = np.maximum(0.0, 100.0 - dec_rate * (time - tau2))
        mu = np.where(
            time < tau1,
            0.0,
            np.where(time < t_rise_end, rise, np.where(time < tau2, 100.0, decrease)),
        )

        nll = 0.5 * np.sum(((position - mu) / sigma) ** 2) + n * log_sigma

        nlp = (
            0.5 * (d12 / 3.0) ** 2
            + 0.5 * ((log_rise_rate - np.log(100.0)) / 0.5) ** 2
            + 0.5 * ((log_dec_rate - np.log(10.0)) / 1.0) ** 2
            + 0.5 * (sigma / 5.0) ** 2
        )
        return nll + nlp

    x0 = np.array(
        [
            t_min + 0.15 * t_range,  # tau1
            0.25 * t_range,  # d12
            np.log(100.0),  # log_rise_rate
            np.log(10.0),  # log_dec_rate
            0.0,  # log_sigma → sigma=1
        ]
    )

    result = minimize(
        neg_log_posterior,
        x0,
        method="Powell",
        options={"xtol": 1e-6, "ftol": 1e-6, "maxiter": 100_000},
    )

    tau1, d12, log_rise_rate, log_dec_rate, log_sigma = result.x
    tau2 = tau1 + d12
    rise_rate = np.exp(log_rise_rate)
    dec_rate = np.exp(log_dec_rate)
    t_rise_end = tau1 + 100.0 / rise_rate
    sigma_noise = np.exp(log_sigma)

    hess = _numerical_hessian(neg_log_posterior, result.x)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.full((5, 5), np.nan)

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    # Delta method: se(rise_rate) = se(log_rise_rate) * rise_rate
    se_rise_rate = float(se[2]) * rise_rate
    se_dec_rate = float(se[3]) * dec_rate
    # tau2 = tau1 + d12 → var(tau2) = var(tau1) + var(d12) + 2*cov(tau1,d12)
    se_tau2 = float(np.sqrt(max(0.0, cov[0, 0] + cov[1, 1] + 2.0 * cov[0, 1])))

    print("=== Method 2: Bayesian Trapezoid MAP + Laplace (scipy) ===")
    print(f"Rise start: tau1={tau1:.3f} ± {se[0]:.3f} s")
    print(f"Rise end:   {t_rise_end:.3f} s")
    print(f"Decrease start: tau2={tau2:.3f} ± {se_tau2:.3f} s")
    print(
        f"Rise rate:  {rise_rate:.2f} ± {se_rise_rate:.2f} %/s"
        f"  →  95 % CI: [{rise_rate - 1.96 * se_rise_rate:.2f}, {rise_rate + 1.96 * se_rise_rate:.2f}]"
    )
    print(
        f"Dec rate:   {dec_rate:.2f} ± {se_dec_rate:.2f} %/s"
        f"  →  95 % CI: [{dec_rate - 1.96 * se_dec_rate:.2f}, {dec_rate + 1.96 * se_dec_rate:.2f}]"
    )
    print(f"Noise σ: {sigma_noise:.2f} %")

    return {
        "breakpoints": [tau1, t_rise_end, tau2],
        "slopes": [rise_rate, -dec_rate],
        "slope_stderr": [se_rise_rate, se_dec_rate],
        "tau_stderr": [float(se[0]), float(np.nan), se_tau2],
        "cov": cov,
        "model": "trapezoid",
    }


def _build_segments(
    data: GateData, result: dict[str, Any]
) -> list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    """Build fitted line segments for plotting."""
    if result.get("model") == "trapezoid":
        t1, t_rise_end, t2 = result["breakpoints"]
        rise_rate, neg_dec_rate = result["slopes"]
        dec_rate = -neg_dec_rate
        segments = []
        for t_lo, t_hi, fn in [
            (data.time[0], t1, lambda t: np.zeros_like(t)),
            (t1, t_rise_end, lambda t: rise_rate * (t - t1)),
            (t_rise_end, t2, lambda t: np.full_like(t, 100.0)),
            (t2, data.time[-1], lambda t: np.maximum(0.0, 100.0 - dec_rate * (t - t2))),
        ]:
            mask = (data.time >= t_lo) & (data.time <= t_hi)
            t_seg = data.time[mask]
            segments.append((t_seg, fn(t_seg)))
        return segments

    tau1, tau2, tau3 = result["breakpoints"]
    s_fast, s_slow = result["slopes"]
    level_high, level_low = result["plateaus"]

    pos_at_tau2 = level_high + s_fast * (tau2 - tau1)

    segments = []
    for t_lo, t_hi, fn in [
        (data.time[0], tau1, lambda t: np.full_like(t, level_high)),
        (tau1, tau2, lambda t: level_high + s_fast * (t - tau1)),
        (tau2, tau3, lambda t: pos_at_tau2 + s_slow * (t - tau2)),
        (tau3, data.time[-1], lambda t: np.full_like(t, level_low)),
    ]:
        mask = (data.time >= t_lo) & (data.time <= t_hi)
        t_seg = data.time[mask]
        segments.append((t_seg, fn(t_seg)))

    return segments


def analyze(
    data: GateData,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run Bayesian changepoint analysis on all gate columns.

    Dispatches to the trapezoid model for ``gate_1`` when
    ``data.trapezoid_decrease_rate`` is set.

    Returns (results, segments) dicts keyed by gate column name.
    """
    time = data.time
    results: dict[str, dict[str, Any]] = {}
    segments: dict[str, Any] = {}
    for g_idx, col in enumerate(data.gate_columns):
        position = data.df[col].to_numpy()
        is_trapezoid = g_idx == 1 and data.trapezoid_decrease_rate is not None
        r = (
            bayesian_changepoint_trapezoid(time, position)
            if is_trapezoid
            else bayesian_changepoint(time, position)
        )
        results[col] = r
        segments[col] = _build_segments(data, r)
    return results, segments


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    results, segments = analyze(data)
    fig = plot_results(
        data,
        "Method 2: Bayesian MAP + Laplace (scipy)",
        fitted_segments=segments,
        detected_breakpoints={c: r["breakpoints"] for c, r in results.items()},
        estimated_slopes={c: r["slopes"] for c, r in results.items()},
    )
    show(fig)
