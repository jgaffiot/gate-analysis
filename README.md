# Gate Analysis -- Pipe Gate Closing Curve Analysis Demos

## Goal

This project demonstrates several different methods for analyzing **pipe gate
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

## Classical methods

### 0. Direct Curve Fit (`0_curve_fit.py`)

**Library:** `scipy.optimize` (numpy only)

Fits the known 4-piece piecewise linear model directly using
`scipy.optimize.curve_fit` (Trust Region Reflective algorithm). Because the
signal topology — high plateau → fast ramp → slow ramp → low plateau — is
fully known in advance, the model has exactly 6 free parameters (`t1`, `t2`,
`t3`, `v_high`, `a1`, `a2`). No model selection is needed. The covariance
matrix returned by `curve_fit` provides standard errors on all parameters,
including slopes, at negligible extra cost. This is the simplest approach
when the signal structure is known, and typically the fastest.

### 1. Segmented Regression (`1_segmented_regression.py`)

**Library:** `piecewise-regression` (Python port of Muggeo's 2003 iterative
method)

Fits a continuous piecewise-linear model with a specified number of
breakpoints. The algorithm iteratively linearizes the breakpoint estimation
problem -- no grid search needed. Provides confidence intervals on both
breakpoints and slopes via bootstrap. The most straightforward and
battle-tested approach.

### 2. Bayesian Changepoint Model (`2_bayesian_changepoint.py`)

**Library:** `scipy.optimize` (numpy only)

Defines the same probabilistic model as a full Bayesian treatment: 3 ordered
change points via positive offsets, Gaussian priors on plateau levels and
slopes, half-Normal priors on gap widths and noise. Inference uses two cheap
steps instead of MCMC:
1. **MAP** — `scipy.optimize.minimize` (Powell, gradient-free) finds the
   mode of the posterior in ~0.1 s.
2. **Laplace approximation** — the numerical Hessian of the negative
   log-posterior at the MAP gives a Gaussian posterior approximation. Its
   inverse is the covariance matrix; diagonal square roots are parameter
   standard errors; 95 % credible intervals follow by the delta method.

For a unimodal, well-identified posterior like this one the Laplace
approximation is equivalent to ADVI's Gaussian variational family — the same
uncertainty estimates — but runs in ~0.25 s without any probabilistic-
programming dependency.

### 3. CPOP-like Continuous Piecewise Linear (`3_cpop_piecewise_linear.py`)

**Library:** custom implementation (numpy only)

Inspired by Fearnhead et al. (2019), this implements a simplified
dynamic-programming approach with an L0 penalty on slope changes. For each
candidate number of breakpoints (0 to 5), it finds optimal breakpoint
placement by iterative coordinate-wise refinement, then selects the best
model using BIC. Enforces continuity at breakpoints by construction. Pass
`n_breakpoints=3` to skip model selection when the topology is known (~6×
faster: ~0.3 s vs ~2 s).

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

### 6. Adaptive Kalman Filter with CUSUM Restart (`6_kalman_filter.py`)

**Library:** `filterpy`, `scipy.stats`

Sets up a constant-velocity Kalman filter (state = [position, velocity]).
Instead of post-hoc heuristic thresholding, changepoints are detected
**online** using a CUSUM test on the Normalized Innovation Squared (NIS =
innovation² / S). The NIS follows a chi²(1) distribution (mean = 1) when
the filter model matches the data; a sustained excess signals a slope change.
When the CUSUM exceeds its threshold, the velocity covariance is inflated at
the estimated changepoint, forcing rapid re-adaptation to the new slope within
a few samples. Slopes are then estimated by OLS on each detected segment,
giving much more accurate results than the filtered-velocity median heuristic.

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
| 0 | Direct curve fit (scipy) | 2.000, 5.003, 9.165 | -24.99 | -5.02 | <1 s |
| 1 | Segmented regression | 2.000, 5.003, 9.145 | -24.99 | -5.03 | ~0.9 s |
| 2 | Bayesian MAP + Laplace | 1.999, 5.004, 8.980 | -24.98 | -5.01 | ~0.25 s |
| 3 | CPOP piecewise linear | 1.980, 5.040, 9.180 | -24.79 | -4.96 | ~2 s (~0.3 s with n_breakpoints=3) |
| 4 | ruptures + OLS | 2.850, 4.300, 6.900 | -25.18 | -8.64 | ~1 s |
| 5 | Savitzky-Golay | 0.390, 5.000, 11.990 | -22.67 | -3.41 | ~1 s |
| 6 | Kalman adaptive (CUSUM restart) | 2.040, 5.060, 9.550 | -24.99 | -4.94 | ~0.05 s |
| 8 | NOT | 2.030, 5.040, 9.100 | -24.99 | -5.05 | ~1 s |
| | **Ground truth** | **2.000, 5.000, 9.000** | **-25.00** | **-5.00** | |

### Observations

- **Option 0** (direct curve fit) achieves near-exact results (~0.01 %/s
  error) in under a millisecond by exploiting full knowledge of the signal
  topology. It also provides standard errors on slopes directly from the
  covariance matrix (e.g. fast slope: −24.99 ± 0.07 %/s). The only caveat
  is that convergence depends on a reasonable initial guess, which is easy to
  supply when the signal structure is known.
- **Options 1-3** produce accurate slope estimates (within 1% of true values)
  and good breakpoint detection. These are model-based approaches that fit
  the known piecewise-linear structure directly.
- **Option 4** (ruptures) finds the fast slope well but misplaces breakpoints
  and overestimates the slow slope, because the L2 cost function detects
  mean shifts rather than slope changes. Using `model="clinear"` (continuous
  piecewise linear) would improve this.
- **Option 5** (SG derivative) is a signal-processing approach relying on
  heuristic thresholds for regime detection. It correctly identifies the
  general shape but is less precise on breakpoint locations and slope
  magnitude, and would benefit from parameter tuning.
- **Option 6** (adaptive Kalman) replaces the original heuristic velocity
  threshold with a principled CUSUM test on the Normalized Innovation Squared.
  The filter is restarted at each detected changepoint, enabling near-instant
  re-convergence to the new slope. Combined with OLS per segment, slope
  accuracy improves from ~(-11.91, -4.78) to ~(-24.99, -4.94) %/s — on par
  with the model-based methods — while being the fastest approach (~50 ms).
  The third breakpoint (end of closing) is slightly overestimated (~9.55 vs
  9.00 s) because the CUSUM needs a few plateau samples to accumulate; this
  is a known detection-delay artefact of CUSUM-based methods.
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

### Individual scripts

```bash
# Install dependencies
uv sync

# Run headlessly (no display required, e.g. on a server)
MPLBACKEND=Agg uv run python -m gate_analysis.1_segmented_regression

# Run interactively (opens a browser window with the plot)
uv run python -m gate_analysis.1_segmented_regression
```

Replace `1_segmented_regression` with any of the 7 module names above.

### Notebook (`classical_methods_nb.py`)

The notebook is written with [marimo](https://marimo.io) and covers all seven
methods on a single page.

**Open as an interactive notebook** (live reactive UI in the browser):

```bash
uv run marimo edit classical_methods_nb.py
```

**Run as a read-only app** (no editing, cleaner UI):

```bash
uv run marimo run classical_methods_nb.py
```

**Export to a self-contained HTML file** (no server needed, shareable):

```bash
uv run marimo export html classical_methods_nb.py -o classical_methods_nb.html
```

**Run as a plain Python script** (prints results to stdout, no browser):

```bash
uv run python classical_methods_nb.py
```

### Transformer-based methods

Transformer-based approach (section 9 of `biblio.md`).

#### Notebook (`transformer_methods_nb.py`)

A dedicated marimo notebook for the Transformer-based approach.

It walks through:

1. Loading `AutonLab/MOMENT-1-large` from HuggingFace (~800 MB, cached after
   first run)
2. **Zero-shot reconstruction** — sliding-window reconstruction error as a
   change-point signal (with honest assessment of its limitations)
3. **Fine-tuned Conv1D head** — training a lightweight head on frozen MOMENT
   patch embeddings using entirely synthetic data from `generate_synthetic_data()`
4. **Inference** — vote-accumulation strategy over overlapping windows
5. **Results and discussion** — comparison with classical methods

It requires heavy libraries (including PyTorch), totaling several Go on disk.

```bash
# Install dependencies
uv sync --group transformer

# Interactive notebook
uv run marimo edit transformer_methods_nb.py

# Read-only app
uv run marimo run transformer_methods_nb.py

# Self-contained HTML export
uv run marimo export html transformer_methods_nb.py -o transformer_methods_nb.html
```
