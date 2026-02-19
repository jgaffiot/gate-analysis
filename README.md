# Gate Analysis -- Pipe Gate Closing Curve Analysis Demos

## Goal

This project demonstrates seven different methods for analyzing **pipe gate
closing curves**. The signal of interest is a gate position (in %) recorded
over time, with the following structure:

1. **High plateau** (~100%): gate wide open, full flow
2. **Fast closing ramp**: steep linear decrease
3. **Slow closing ramp**: gentle linear decrease
4. **Low plateau** (~0%): gate fully closed

Due to sensor imperfections the signal has additive noise, and the plateau
values may not be exactly 100% or 0%. The primary goal is to **accurately
measure the two closing slopes**, in order to track mechanical aging
(slope flattening over successive closing events).

A shared synthetic data generator (`common.py`) produces a realistic test
signal with known ground-truth parameters (breakpoints at 2.0, 5.0, 9.0 s;
slopes of -25.0 and -5.0 %/s; Gaussian noise with 1% std).

## Options

### 1. Segmented Regression (`1_segmented_regression.py`)

**Library:** `piecewise-regression` (Python port of Muggeo's 2003 iterative
method)

Fits a continuous piecewise-linear model with a specified number of
breakpoints. The algorithm iteratively linearizes the breakpoint estimation
problem -- no grid search needed. Provides confidence intervals on both
breakpoints and slopes via bootstrap. The most straightforward and
battle-tested approach.

### 2. Bayesian Changepoint Model (`2_bayesian_changepoint.py`)

**Library:** `pymc`

Defines a full probabilistic model with 4 regimes (two plateaus + two ramps),
3 ordered change points, and Gaussian noise. Uses NUTS MCMC sampling to
produce posterior distributions on all parameters. Gives credible intervals
on slopes and breakpoints for free. The model encodes the known physical
structure as priors, which helps convergence. Slowest option by far due to
MCMC, but best for principled uncertainty quantification.

### 3. CPOP-like Continuous Piecewise Linear (`3_cpop_piecewise_linear.py`)

**Library:** custom implementation (numpy only)

Inspired by Fearnhead et al. (2019), this implements a simplified
dynamic-programming approach with an L0 penalty on slope changes. For each
candidate number of breakpoints (0 to 5), it finds optimal breakpoint
placement by iterative coordinate-wise refinement, then selects the best
model using BIC. Enforces continuity at breakpoints by construction.

### 4. ruptures + OLS (`4_ruptures_ols.py`)

**Library:** `ruptures`, `scipy.stats`

Uses the `ruptures` library for change-point detection (dynamic programming
with an L2 cost model), then fits ordinary least squares on each segment to
estimate slopes with 95% confidence intervals. The L2 cost detects shifts in
mean rather than slope, which makes it less well-suited for this specific
signal shape -- a `clinear` cost model would be more appropriate.

### 5. Savitzky-Golay Derivative + Threshold (`5_savitzky_golay.py`)

**Library:** `scipy.signal`

Applies a Savitzky-Golay filter to simultaneously smooth the signal and
compute its first derivative analytically from the fitted polynomial. The
closing region is detected by thresholding the derivative, and the fast/slow
transition is found by splitting on derivative magnitude. Slopes are estimated
as the median derivative in each sub-region. Simple and fast, but no formal
statistical inference.

### 6. Kalman Filter with Regime Detection (`6_kalman_filter.py`)

**Library:** `filterpy`

Sets up a constant-velocity Kalman filter (state = [position, velocity]) to
optimally estimate the instantaneous slope at each time step. Regime
transitions are detected from the filtered velocity signal using the same
threshold/split heuristic as option 5. Provides online-capable processing,
but the heuristic regime splitting limits accuracy compared to model-based
approaches.

### 8. NOT — Narrowest-Over-Threshold (`8_not_detection.py`)

**Library:** custom implementation (numpy + scipy only)

Python reimplementation of the NOT algorithm (Baranowski, Chen & Fryzlewicz,
2019). Draws a large number of random sub-intervals, computes an F-type
contrast statistic (RSS reduction from splitting at a kink) at every
candidate position using O(1) prefix-sum OLS, and iteratively declares the
narrowest interval whose maximum statistic exceeds a Bonferroni-calibrated
threshold as a change point. The narrowest-first extraction rule gives sharper
localisation of closely-spaced breakpoints than standard binary segmentation.
No need to specify the number of breakpoints.

## Results

All demos are run on the same synthetic signal (1200 samples, dt=0.01s,
seed=42). Ground truth: breakpoints at **2.0, 5.0, 9.0 s**, slopes
**-25.0, -5.0 %/s**.

| # | Method | Breakpoints (s) | Fast slope (%/s) | Slow slope (%/s) | Time |
|---|--------|-----------------|-------------------|-------------------|------|
| 1 | Segmented regression | 2.000, 5.003, 9.145 | -24.99 | -5.03 | ~2 s |
| 2 | Bayesian changepoint | 2.003, 4.985, 9.257 | -25.20 | -4.96 | ~5 min |
| 3 | CPOP piecewise linear | 1.980, 5.040, 9.180 | -24.79 | -4.96 | ~4 s |
| 4 | ruptures + OLS | 2.850, 4.300, 6.900 | -25.18 | -8.64 | ~1 s |
| 5 | Savitzky-Golay | 0.390, 5.000, 11.990 | -22.67 | -3.41 | ~1 s |
| 6 | Kalman filter | 0.040, 8.010, 11.800 | -11.91 | -4.78 | ~1 s |
| 8 | NOT | 2.030, 5.040, 9.100 | -24.99 | -5.05 | ~1 s |
| | **Ground truth** | **2.000, 5.000, 9.000** | **-25.00** | **-5.00** | |

### Observations

- **Options 1-3** produce accurate slope estimates (within 1% of true values)
  and good breakpoint detection. These are model-based approaches that fit
  the known piecewise-linear structure directly.
- **Option 4** (ruptures) finds the fast slope well but misplaces breakpoints
  and overestimates the slow slope, because the L2 cost function detects
  mean shifts rather than slope changes. Using `model="clinear"` (continuous
  piecewise linear) would improve this.
- **Options 5-6** (SG derivative and Kalman) are signal-processing approaches
  that rely on heuristic thresholds for regime detection. They correctly
  identify the general shape but are less precise on breakpoint locations
  and slope magnitude. Both would benefit from tuning their parameters to
  the specific signal characteristics.
- **Option 2** (Bayesian) is the only one providing principled credible
  intervals (e.g., fast slope 95% CI: [-25.39, -24.98] %/s), but is
  ~150x slower than option 1.
- **Option 8** (NOT) achieves accuracy on par with options 1–3 while
  requiring no prior knowledge of the number of breakpoints. The
  narrowest-first rule makes it robust when change points are closely spaced.
  The Bonferroni-calibrated threshold provides statistical control over false
  positives.
- For batch processing many closing events, **option 1** (segmented
  regression) offers the best accuracy/speed trade-off.

## Usage

```bash
uv sync
MPLBACKEND=Agg uv run python -m gate_analysis.1_segmented_regression
# or interactively (with plot display):
uv run python -m gate_analysis.1_segmented_regression
```

Replace `1_segmented_regression` with any of the 7 module names above.
