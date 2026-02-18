"""Option B: Bayesian change-point model using PyMC.

Defines a 4-regime model (high plateau, fast slope, slow slope, low plateau)
with unknown change points, and uses ADVI (variational inference) to get
approximate posterior distributions on all parameters including slopes.

ADVI replaces NUTS MCMC: it finds a Gaussian approximation to the posterior
via gradient-based optimisation, running in seconds instead of minutes while
giving equally accurate point estimates and reasonable credible intervals for
this unimodal, well-identified model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pymc as pm

from gate_analysis.common import GateData, generate_synthetic_data, plot_results


def bayesian_changepoint(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    *,
    n_samples: int = 2000,
    n_iter: int = 30_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit a Bayesian 4-regime change-point model using PyMC + ADVI.

    The model has:
    - 3 change points (tau1, tau2, tau3) with ordering enforced by an
      offset parameterisation (tau1 free; tau2 = tau1 + d12; tau3 = tau2 + d23
      with d12, d23 ~ HalfNormal > 0)
    - 2 plateau levels and 2 slopes
    - Gaussian noise

    Inference uses ADVI (Automatic Differentiation Variational Inference)
    instead of NUTS MCMC, reducing runtime from ~5 min to ~5 s.

    Parameters
    ----------
    time : array
        Time values.
    position : array
        Gate position values (%).
    n_samples : int
        Number of samples drawn from the variational approximation.
    n_iter : int
        Number of ADVI optimisation iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: breakpoints, slopes, plateaus, trace
    """
    t_min, t_max = float(time.min()), float(time.max())

    with pm.Model() as model:
        # Ordered change points via positive offsets — clean geometry for ADVI:
        #   tau1 ~ Uniform  (first breakpoint)
        #   d12, d23 ~ HalfNormal > 0  (interval lengths, automatically positive)
        #   tau2 = tau1 + d12,  tau3 = tau2 + d23
        tau1 = pm.Uniform("tau1", lower=t_min, upper=t_max)
        d12 = pm.HalfNormal("d12", sigma=3.0)
        d23 = pm.HalfNormal("d23", sigma=3.0)
        tau2 = pm.Deterministic("tau2", tau1 + d12)  # type: ignore[operator]
        tau3 = pm.Deterministic("tau3", tau2 + d23)

        # Plateau levels
        level_high = pm.Normal("level_high", mu=98, sigma=10)
        level_low = pm.Normal("level_low", mu=2, sigma=10)

        # Slopes (negative, since we're closing)
        slope_fast = pm.Normal("slope_fast", mu=-25, sigma=15)
        slope_slow = pm.Normal("slope_slow", mu=-5, sigma=15)

        # Noise
        sigma = pm.HalfNormal("sigma", sigma=5)

        # Build the piecewise-linear mean function
        # Phase 1: constant at level_high
        # Phase 2: level_high + slope_fast * (t - tau1)
        # Phase 3: continuation + slope_slow * (t - tau2)
        # Phase 4: constant at level_low
        pos_at_tau2 = level_high + slope_fast * (tau2 - tau1)
        mu = pm.math.switch(
            time < tau1,  # type: ignore[operator]
            level_high,
            pm.math.switch(
                time < tau2,
                level_high + slope_fast * (time - tau1),  # type: ignore[operator]
                pm.math.switch(
                    time < tau3,
                    pos_at_tau2 + slope_slow * (time - tau2),  # type: ignore[operator]
                    level_low,
                ),
            ),
        )

        pm.Normal("obs", mu=mu, sigma=sigma, observed=position)

        # ADVI: finds a Gaussian approximation to the posterior via optimisation.
        # Orders of magnitude faster than NUTS for this smooth, unimodal model.
        approx = pm.fit(n=n_iter, method="advi", random_seed=seed, progressbar=True)
        trace = approx.sample(draws=n_samples, random_seed=seed)

    # Extract posterior means (InferenceData.posterior is dynamic from arviz)
    posterior: Any = trace.posterior  # type: ignore[attr-defined]
    tau1_est = float(posterior["tau1"].mean())
    tau2_est = float(posterior["tau2"].mean())
    tau3_est = float(posterior["tau3"].mean())
    slope_fast_est = float(posterior["slope_fast"].mean())
    slope_slow_est = float(posterior["slope_slow"].mean())
    level_high_est = float(posterior["level_high"].mean())
    level_low_est = float(posterior["level_low"].mean())

    print("=== Option B: Bayesian Change-Point Model (PyMC + ADVI) ===")
    print(f"Breakpoints: tau1={tau1_est:.3f}, tau2={tau2_est:.3f}, tau3={tau3_est:.3f}")
    print(f"Fast slope: {slope_fast_est:.2f} %/s")
    print(f"Slow slope: {slope_slow_est:.2f} %/s")
    print(f"High plateau: {level_high_est:.1f}%, Low plateau: {level_low_est:.1f}%")

    # Credible intervals (approximate, from the variational Gaussian)
    for param in ["slope_fast", "slope_slow", "tau1", "tau2", "tau3"]:
        vals = posterior[param].values.flatten()
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"  {param}: 95% CI = [{lo:.3f}, {hi:.3f}]")

    return {
        "breakpoints": [tau1_est, tau2_est, tau3_est],
        "slopes": [slope_fast_est, slope_slow_est],
        "plateaus": (level_high_est, level_low_est),
        "trace": trace,
        "model": model,
    }


def _build_segments(
    data: GateData, result: dict[str, Any]
) -> list[tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]]:
    """Build fitted line segments for plotting."""
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


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    result = bayesian_changepoint(data.time, data.position)
    segments = _build_segments(data, result)
    fig = plot_results(
        data,
        "Option B: Bayesian Change-Point Model (PyMC + ADVI)",
        fitted_segments=segments,
        detected_breakpoints=result["breakpoints"],
        estimated_slopes=result["slopes"],
    )
    show(fig)
