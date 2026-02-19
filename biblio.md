# Bibliography: Pipe Gate Closing Curve Analysis

## 1. Problem Statement

Accurately measuring the two closing slopes (fast then slow) of a pipe gate
from noisy sensor data (position in % vs. time), in order to track mechanical
aging (slope flattening over time).

The signal has **five phases**: constant high plateau (~100%), fast linear
closing, slow linear closing, constant low plateau (~0%), with additive sensor
noise throughout and possible calibration offsets on the plateaus.

The core analytical challenge decomposes into:

1. **Noise reduction / signal preprocessing**
2. **Segmentation**: detecting the 3-4 change points (plateau-to-fast,
   fast-to-slow, slow-to-plateau, and possibly plateau boundaries)
3. **Slope estimation** on each linear segment, with uncertainty
   quantification

---

## 2. Signal Preprocessing (Denoising)

Before fitting, reducing sensor noise improves change-point detection and
slope accuracy.

### 2.1 Moving Average / Exponential Moving Average

- Simplest approach. Smooths out high-frequency noise.
- Drawback: introduces lag and rounds off the transition corners (change
  points), which biases breakpoint location.

### 2.2 Savitzky-Golay Filter

- Fits successive subsets of adjacent data points with a low-degree polynomial
  via least squares. Preserves signal shape (peaks, slopes) better than
  moving averages.
- The filter can also directly estimate derivatives (slope) from the
  polynomial coefficients.
- Reference: Savitzky, A., Golay, M.J.E. (1964). "Smoothing and
  Differentiation of Data by Simplified Least Squares Procedures." *Analytical
  Chemistry*, 36(8), 1627-1639.
- See also: Schafer, R.W. (2011). "What Is a Savitzky-Golay Filter?"
  *IEEE Signal Processing Magazine*, 28(4), 111-117.
- Implementation: `scipy.signal.savgol_filter` in Python, `sgolayfilt` in
  MATLAB.
- Caveat: Marcotte (2022) argues SG filters should be replaced by regularized
  approaches in some contexts. See: [ACS Measurement Science Au](https://pubs.acs.org/doi/10.1021/acsmeasuresciau.1c00054).

### 2.3 Kalman Filter

- Optimal for Gaussian noise on a linear dynamic system. Can model the gate
  as a state-space system (position + velocity), naturally estimating the
  slope as the velocity state.
- More complex to set up (requires defining process/measurement noise
  covariances Q and R), but adapts to changing dynamics.
- Reference: Kalman, R.E. (1960). "A New Approach to Linear Filtering and
  Prediction Problems." *Journal of Basic Engineering*, 82(1), 35-45.
- Python: `filterpy`, `pykalman`.

### 2.4 LOESS / LOWESS (Locally Weighted Scatterplot Smoothing)

- Non-parametric local regression. Very flexible, but computationally heavier
  and does not enforce the known piecewise-linear structure.
- Reference: Cleveland, W.S. (1979). "Robust Locally Weighted Regression
  and Smoothing Scatterplots." *JASA*, 74(368), 829-836.

### 2.5 Recommendation

For this problem, **Savitzky-Golay** is a good default: it preserves the
linear segments and the corners. However, if the noise model is well
characterized, a **Kalman filter** with a piecewise-linear state model is
theoretically optimal. In practice, preprocessing is optional if the
segmentation/regression method is robust to noise (e.g., Bayesian methods
or robust regression).

---

## 3. Segmentation: Change-Point / Breakpoint Detection

This is the central problem: finding where the signal transitions between
its phases (plateaus and linear ramps).

### 3.1 Segmented (Broken-Line) Regression -- Muggeo's Method

- **Key paper**: Muggeo, V.M.R. (2003). "Estimating regression models with
  unknown break-points." *Statistics in Medicine*, 22, 3055-3071.
  [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/sim.1545)
- Iterative linearization: given an initial guess for breakpoints, the
  nonlinear problem is approximated by a linear model, and breakpoints are
  updated at each iteration. Computationally efficient, no grid search.
- Extended in: Muggeo, V.M.R. (2008). "segmented: An R Package to Fit
  Regression Models with Break-Point Relationships." *R News*, 8(1), 20-25.
  [PDF](https://journal.r-project.org/articles/RN-2008-004/RN-2008-004.pdf)
- **R package**: [`segmented`](https://cran.r-project.org/web/packages/segmented/)
  -- the most mature tool for this specific task.
- Supports automatic selection of the number of breakpoints via BIC/AIC
  (`selgmented` function).
- Provides **confidence intervals on breakpoint locations and slopes**.
- **Highly recommended** for this use case.

### 3.2 Bayesian Multiple Change Point Regression (mcp)

- **R package**: [`mcp`](https://lindeloev.github.io/mcp/) -- Bayesian
  inference for regression with one or more change points.
- Allows specifying a priori knowledge about the number and form of
  segments (e.g., "plateau, then linear, then linear, then plateau").
- Uses MCMC (JAGS) to produce full posterior distributions over breakpoint
  locations and slopes -- natural uncertainty quantification.
- Reference: Lindelov, J.K. (2020). "mcp: An R Package for Regression
  With Multiple Change Points." [OSF Preprint](https://osf.io/fzqxv/).
- **Advantage**: can encode the known structure (4 segments, 2 plateaus + 2
  slopes) as a prior, making it very well suited here.
- **Drawback**: slower than frequentist methods (MCMC), but this is
  irrelevant for single-curve analysis.

### 3.3 Pruned Exact Linear Time (PELT)

- Algorithm for optimal segmentation under a penalized cost function.
  Computational complexity is linear in the number of data points (with
  pruning).
- Reference: Killick, R., Fearnhead, P., Eckley, I.A. (2012). "Optimal
  Detection of Changepoints with a Linear Computational Cost." *JASA*,
  107(500), 1590-1598.
- Typically detects changes in **mean** or **variance**, not directly in
  slope. Can be adapted for piecewise-linear signals with appropriate cost
  functions.
- **R package**: `changepoint`. **Python**: `ruptures` (see below).

### 3.4 ruptures (Python)

- **Python library**: [ruptures](https://centre-borelli.github.io/ruptures-docs/)
  -- offline change point detection.
- Reference: Truong, C., Oudre, L., Vayatis, N. (2020). "Selective review
  of offline change point detection methods." *Signal Processing*, 167,
  107299.
- Supports multiple search algorithms: dynamic programming (`Dynp`), PELT,
  binary segmentation, bottom-up, window-based.
- Supports cost functions for piecewise-linear signals (`CostCLinear`,
  `CostLinear`) -- directly relevant.
- **Good Python-native option**, but less specialized for slope inference
  (no built-in confidence intervals on slopes).

### 3.5 CPOP (Changepoint for Piecewise Linear Signals)

- Dynamic programming method using an L0 penalty on changes in slope.
  Specifically designed for **continuous piecewise-linear** signals.
- Reference: Fearnhead, P., Maidstone, R., Letchford, A. (2019). "Detecting
  Changes in Slope With an L0 Penalty." *Journal of Computational and
  Graphical Statistics*, 28(2), 265-275.
  [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/10618600.2018.1512868)
- **R package**: `cpop`.
- Enforces continuity at change points (the fitted line is continuous),
  which matches the physical reality of a gate closing.
- **Well suited** for this application.

### 3.6 Bai-Perron Multiple Structural Breaks

- Classical econometrics approach for detecting multiple structural breaks
  in linear regression.
- Reference: Bai, J., Perron, P. (2003). "Computation and Analysis of
  Multiple Structural Change Models." *Journal of Applied Econometrics*,
  18, 1-22.
  [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.659)
- Uses dynamic programming (O(T^2)).
- **R package**: `strucchange`.
- Provides sup-F tests for the number of breaks.
- More oriented toward econometric time series; less natural for
  engineering signal analysis.

### 3.7 Hidden Markov Models (HMM)

- Model the signal as switching between hidden states (plateau, fast ramp,
  slow ramp, plateau), each with its own emission distribution.
- Can handle noise naturally. The Viterbi algorithm gives the most likely
  state sequence.
- Reference: Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and
  Selected Applications in Speech Recognition." *Proc. IEEE*, 77(2),
  257-286.
- Python: `hmmlearn`, `pomegranate`.
- **Drawback**: does not natively model linear trends within states (would
  need autoregressive HMM or switching linear dynamical system).

### 3.8 L1 Trend Filtering (Total Variation Denoising for Piecewise-Linear Signals)

- Solves a convex optimisation problem with an L1 penalty on the second
  differences of the signal. The solution is guaranteed to be piecewise
  linear with automatically selected knot locations.
- Reference: Kim, S., Koh, K., Boyd, S., Gorinevsky, D. (2009). "ℓ1 Trend
  Filtering." *SIAM Review*, 51(2), 339-360.
- Extended to adaptive piecewise polynomials of arbitrary degree by:
  Tibshirani, R.J. (2014). "Adaptive piecewise polynomial estimation via
  trend filtering." *Annals of Statistics*, 42(1), 285-323.
- **Python**: `cvxpy` (general convex solver). **R**: `genlasso`.
- No need to specify the number of breakpoints; the penalty parameter λ
  controls complexity (larger λ → fewer segments). Can be tuned via
  cross-validation or BIC.
- Breakpoint locations are implicit (found as kinks in the fitted piecewise-
  linear solution); slopes are read off per segment after fitting.
- **Well suited** for this application: directly targets piecewise-linear
  signals, convex problem (global optimum guaranteed), handles noise well.
- **Drawback**: does not provide formal confidence intervals on breakpoints
  or slopes without post-hoc bootstrap.

### 3.9 Narrowest-Over-Threshold (NOT)

- Searches for the narrowest data interval in which a CUSUM-type statistic
  exceeds a threshold, and identifies a change point within that interval.
  Iterates to find all change points.
- Reference: Baranowski, R., Chen, Y., Fryzlewicz, P. (2019). "Narrowest-
  Over-Threshold Detection of Multiple Change Points and Change-Point-Like
  Features." *Journal of the Royal Statistical Society: Series B*, 81(3),
  649-672.
- Handles both piecewise-constant (change in mean) and piecewise-linear
  (change in slope / kinks) signals within the same framework.
- **R package**: `not`. No mature Python implementation.
- Sharper localisation of closely-spaced breakpoints than binary
  segmentation, because it focuses on the narrowest informative interval
  rather than the widest.
- **Well suited** for this application given the known presence of distinct
  ramp segments.
- **Drawback**: R-only; no built-in slope confidence intervals.

### 3.10 Wild Binary Segmentation (WBS) and Seeded Binary Segmentation (SBS)

- **WBS**: improves over standard binary segmentation by drawing a large
  number of random sub-intervals and applying a CUSUM test on each, then
  selecting the most significant split. Much more robust than vanilla binary
  segmentation when breakpoints are closely spaced or have small amplitudes.
- WBS reference: Fryzlewicz, P. (2014). "Wild binary segmentation for
  multiple change-point detection." *Annals of Statistics*, 42(6), 2243-2281.
- **SBS**: replaces the random sub-intervals of WBS with a deterministic
  "seeded" set, giving reproducible results with near-linear computational
  complexity and proven minimax optimality.
- SBS reference: Kovács, S., Li, H., Bühlmann, P., Munk, A. (2023).
  "Seeded Binary Segmentation: A General Methodology for Fast and Optimal
  Change Point Detection." *Biometrika*, 110(1), 249-256.
- **R packages**: `wbs` (WBS), `breakfast` (SBS and variants). Python:
  limited; SBS has a [GitHub implementation](https://github.com/kovacssolt/ChangePoints).
- Primarily detect changes in mean; adapting to changes in slope requires
  an appropriate CUSUM statistic.
- **Useful** as a fast pre-screening step or for benchmarking against
  the more specialised CPOP / NOT methods.

---

## 4. Slope Estimation and Uncertainty

Once segments are identified, slopes must be estimated accurately.

### 4.1 Ordinary Least Squares (OLS)

- Standard linear regression on each segment.
- Assumes noise only on the dependent variable (position).
- Provides standard errors, confidence intervals, and prediction intervals
  via classical statistics.
- Slope attenuation bias is negligible when the independent variable (time)
  is measured precisely (which it is, from a clock).

### 4.2 Total Least Squares / Orthogonal Distance Regression

- Accounts for errors in both variables.
- Reference: Van Huffel, S., Vandewalle, J. (1991). *The Total Least
  Squares Problem: Computational Aspects and Analysis*. SIAM.
- **Not needed here**: time is measured precisely; only position has noise.
  OLS is appropriate.

### 4.3 Robust Regression (M-estimators, RANSAC, Theil-Sen)

- If outliers or heavy-tailed noise are present:
  - **Theil-Sen estimator**: median of all pairwise slopes. Breakdown point
    of ~29%. Very robust.
  - **RANSAC**: iteratively fits to random subsets, rejects outliers.
  - **Huber/Bisquare M-estimators**: downweight outliers.
- Python: `scipy.stats.theilslopes`, `sklearn.linear_model.RANSACRegressor`,
  `statsmodels.RLM`.
- Reference: Theil, H. (1950). "A rank-invariant method of linear and
  polynomial regression analysis." *Indagationes Mathematicae*, 12, 85-91.

### 4.4 Bayesian Linear Regression

- Full posterior distribution on slope, giving credible intervals.
- Natural integration with Bayesian change-point models (e.g., `mcp`).
- Particularly useful for tracking slope evolution across many closing
  events (hierarchical models).

### 4.5 Bootstrap Confidence Intervals

- Non-parametric uncertainty quantification: resample the data, refit,
  and compute empirical confidence intervals on slopes and breakpoints.
- Works with any method. Useful when analytic standard errors are not
  available.

---

## 5. End-to-End Approaches for the Specific Problem

Given the known structure (plateau -> fast ramp -> slow ramp -> plateau),
several integrated strategies are available:

### Option A: Segmented Regression (Recommended starting point)

- **Tool**: R `segmented` or Python `piecewise-regression`
- **How**: fit a linear model with 3-4 breakpoints. The package estimates
  breakpoints and slopes simultaneously with confidence intervals.
- **Pros**: simple, fast, well-tested, gives CIs. Good for batch
  processing many closing events.
- **Cons**: frequentist; breakpoint number must be specified (but can be
  selected via BIC).

### Option B: Bayesian Change-Point Model

- **Tool**: R `mcp`
- **How**: define the model as `position ~ 1 ~ time ~ time ~ 1` (constant,
  then two linear segments, then constant). Run MCMC.
- **Pros**: encodes known structure, full posterior on all parameters,
  principled uncertainty. Can extend to hierarchical models across many
  closing events.
- **Cons**: requires R and JAGS, slower.

### Option C: CPOP (Continuous Piecewise Linear Segmentation)

- **Tool**: R `cpop`
- **How**: automatic detection of change points in a continuous piecewise
  linear signal with an L0 penalty.
- **Pros**: no need to specify the number of breakpoints; enforces
  continuity (physically realistic). Designed exactly for this signal type.
- **Cons**: less control over segment structure; need to tune the penalty.

### Option D: ruptures + OLS

- **Tool**: Python `ruptures` with `CostCLinear` or `CostLinear`
- **How**: detect change points, then fit OLS on each segment.
- **Pros**: pure Python, flexible, good for integration into a larger
  pipeline. Multiple search algorithms available.
- **Cons**: no built-in slope CIs from ruptures; must compute separately.

### Option E: Savitzky-Golay Derivative + Threshold Detection

- **How**: apply SG filter with derivative output to estimate instantaneous
  slope. Detect transitions by thresholding the derivative.
- **Pros**: simple, no model assumption beyond smoothness. Direct slope
  readout.
- **Cons**: sensitive to filter parameters (window size, polynomial order).
  No formal statistical inference on slopes or breakpoints.

### Option F: Kalman Filter with Regime Switching

- **How**: model the gate as a switching linear dynamical system with 4
  regimes. Use an interacting multiple model (IMM) filter or switching
  Kalman filter.
- **Pros**: online processing, optimal for Gaussian noise, naturally
  estimates velocity (slope) as a state variable.
- **Cons**: complex to implement and tune. Overkill for offline analysis
  of a known-structure signal.

### Option G: L1 Trend Filtering

- **Tool**: Python `cvxpy` or R `genlasso`
- **How**: minimise the L1 norm of the second differences subject to a
  data-fidelity term. The penalty λ controls the number of segments; tune
  via BIC or cross-validation.
- **Pros**: convex problem (global optimum guaranteed), no need to specify
  the number of breakpoints, directly produces a piecewise-linear fit.
- **Cons**: no built-in confidence intervals; breakpoint locations are
  implicit (read off as kinks in the solution); λ must be tuned.

### Option H: NOT (Narrowest-Over-Threshold)

- **Tool**: R `not`
- **How**: iteratively find the narrowest interval in which a CUSUM-type
  statistic exceeds a threshold, identify a change point within that
  interval, and repeat until no further change points are found. Then fit
  OLS on each identified segment.
- **Pros**: sharper breakpoint localisation than binary segmentation when
  the fast/slow ramp boundary is tight; no need to specify the number of
  breakpoints; well suited to this signal structure (handles piecewise-
  linear / kink signals natively).
- **Cons**: R-only; no built-in slope or breakpoint confidence intervals
  (use bootstrap post-hoc). WBS/SBS (`wbs`/`breakfast` packages) can serve
  as a faster pre-screening alternative before applying NOT, but require
  adapting the CUSUM statistic to detect slope changes rather than mean
  shifts.

---

## 6. Tracking Degradation Over Time

To monitor aging (slope flattening across many closing events):

- Estimate the two slopes for each closing event using any method above.
- Plot slopes vs. event number (or calendar time / cycle count).
- Apply **trend analysis**: simple linear regression on slope vs. time,
  or control charts (Shewman/CUSUM) to detect when degradation exceeds
  a threshold.
- **Bayesian hierarchical models** (via `mcp` or `brms` in R) can
  pool information across events and estimate the degradation rate with
  uncertainty.

---

## 7. Software Summary

| Tool | Language | Method | Slope CIs | Breakpoint CIs | Piecewise Linear |
|------|----------|--------|-----------|----------------|------------------|
| `segmented` | R | Iterative (Muggeo) | Yes | Yes | Yes |
| `mcp` | R | Bayesian MCMC | Yes (posterior) | Yes (posterior) | Yes |
| `cpop` | R | DP + L0 penalty | Manual | Manual | Yes (continuous) |
| `ruptures` | Python | PELT/DP/BinSeg | Manual | No | Yes (cost model) |
| `piecewise-regression` | Python | Muggeo-like | Yes | Yes | Yes |
| `changepoint` | R | PELT | No | No | Mean/variance only |
| `strucchange` | R | Bai-Perron | Yes | Yes | General regression |
| `scipy.signal.savgol_filter` | Python | SG filter | No | No | Preprocessing |
| `statsmodels` | Python | OLS/RLM | Yes | N/A | Post-segmentation |
| `cvxpy` / `genlasso` | Python / R | L1 trend filtering | Manual | Manual | Yes (automatic knots) |
| `not` | R | Narrowest-Over-Threshold | Manual | No | Yes (kinks) |
| `wbs` / `breakfast` | R | WBS / SBS | Manual | No | Via CUSUM statistic |

---

## 8. Key References

1. Muggeo, V.M.R. (2003). "Estimating regression models with unknown
   break-points." *Statistics in Medicine*, 22, 3055-3071.
2. Muggeo, V.M.R. (2008). "segmented: An R Package to Fit Regression
   Models with Break-Point Relationships." *R News*, 8(1), 20-25.
3. Truong, C., Oudre, L., Vayatis, N. (2020). "Selective review of offline
   change point detection methods." *Signal Processing*, 167, 107299.
4. Killick, R., Fearnhead, P., Eckley, I.A. (2012). "Optimal Detection of
   Changepoints with a Linear Computational Cost." *JASA*, 107(500),
   1590-1598.
5. Fearnhead, P., Maidstone, R., Letchford, A. (2019). "Detecting Changes
   in Slope With an L0 Penalty." *JCGS*, 28(2), 265-275.
6. Bai, J., Perron, P. (2003). "Computation and Analysis of Multiple
   Structural Change Models." *J. Applied Econometrics*, 18, 1-22.
7. Savitzky, A., Golay, M.J.E. (1964). "Smoothing and Differentiation of
   Data by Simplified Least Squares Procedures." *Analytical Chemistry*,
   36(8), 1627-1639.
8. Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction
   Problems." *J. Basic Engineering*, 82(1), 35-45.
9. Cleveland, W.S. (1979). "Robust Locally Weighted Regression and
   Smoothing Scatterplots." *JASA*, 74(368), 829-836.
10. Van Huffel, S., Vandewalle, J. (1991). *The Total Least Squares
    Problem*. SIAM.
11. Lindelov, J.K. (2020). "mcp: An R Package for Regression With Multiple
    Change Points." OSF Preprint.
12. Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models." *Proc.
    IEEE*, 77(2), 257-286.
13. Kim, S., Koh, K., Boyd, S., Gorinevsky, D. (2009). "ℓ1 Trend
    Filtering." *SIAM Review*, 51(2), 339-360.
14. Tibshirani, R.J. (2014). "Adaptive piecewise polynomial estimation via
    trend filtering." *Annals of Statistics*, 42(1), 285-323.
15. Baranowski, R., Chen, Y., Fryzlewicz, P. (2019). "Narrowest-Over-
    Threshold Detection of Multiple Change Points and Change-Point-Like
    Features." *JRSS-B*, 81(3), 649-672.
16. Fryzlewicz, P. (2014). "Wild binary segmentation for multiple change-
    point detection." *Annals of Statistics*, 42(6), 2243-2281.
17. Kovács, S., Li, H., Bühlmann, P., Munk, A. (2023). "Seeded Binary
    Segmentation: A General Methodology for Fast and Optimal Change Point
    Detection." *Biometrika*, 110(1), 249-256.

---

## 9. Transformer-Based Methods

Modern deep-learning time-series research is dominated by Transformer
architectures. This section surveys pre-trained foundation models (usable
zero-shot or with light fine-tuning) and untrained architectures (requiring
training from scratch), and assesses their practical relevance to the
gate-closing problem.

**Critical caveat:** the vast majority of time-series Transformer foundation
models are **forecasting models** — they output a probability distribution
over future values. Change-point detection and slope estimation are
fundamentally different tasks. Applicability must be evaluated carefully for
each model.

---

### 9.1 Pre-trained Foundation Models

#### 9.1.1 MOMENT — the only broadly applicable foundation model

- **HuggingFace:** `AutonLab/MOMENT-1-large` (also `-base`, `-small`)
- **Paper:** Goswami, M., et al. (2024). "MOMENT: A Family of Open
  Time-series Foundation Models." *ICML 2024*.
  [arXiv:2402.03885](https://arxiv.org/abs/2402.03885)
- **Package:** `pip install momentfm`
- **Architecture:** ViT-style patch Transformer encoder pre-trained on the
  "Time Series Pile" (large diverse collection of public time series) via a
  masked patch reconstruction objective.
- **Supported tasks:** forecasting, classification, anomaly detection
  (reconstruction-based), imputation. The only foundation model with
  explicit non-forecasting task support.
- **Applicability to this problem:**
  - *Reconstruction / anomaly mode (zero-shot):* MOMENT computes a
    per-timestep reconstruction error. Phase transitions (change points) are
    locally novel patterns and appear as peaks in this error — they can be
    thresholded or peak-detected to locate breakpoints. No training needed.
  - *Classification mode (light fine-tuning):* add a lightweight
    classification head on top of frozen MOMENT patch embeddings and train
    it to label each patch as one of four regimes (plateau-high, fast-ramp,
    slow-ramp, plateau-low). Breakpoints are then the boundaries between
    predicted classes. Training data can be generated synthetically in
    unlimited quantity.

  ```python
  from momentfm import MOMENTPipeline
  import torch

  pipe = MOMENTPipeline.from_pretrained(
      "AutonLab/MOMENT-1-large",
      model_kwargs={"task_name": "reconstruction"},
  )
  pipe.init()
  # Input shape: (batch, n_channels, seq_len); MOMENT expects ≤512 samples
  x = torch.tensor(position).float().unsqueeze(0).unsqueeze(0)
  out = pipe(x)
  recon_error = (x - out.reconstruction).abs().squeeze()
  # peaks in recon_error ≈ change-point locations
  ```

- **Limitations:** fixed 512-token context window; reconstruction-based
  anomaly detection is designed for point anomalies, not structural breaks —
  post-processing is needed; fine-tuning requires labelled regime examples
  (synthetic generation solves this).

#### 9.1.2 Forecasting-only models — not directly applicable

The following models are included for completeness. All are pre-trained
probabilistic forecasters; none produce segmentation or change-point outputs
natively. They could in principle detect change points indirectly via
likelihood-ratio tests (high forecast error at transitions), but this is
fragile and roundabout compared to the methods in sections 3–4.

| Model | Organisation | HuggingFace ID | Notes |
|-------|-------------|----------------|-------|
| Chronos-Bolt | Amazon | `amazon/chronos-bolt-base` | Tokenises values, cross-entropy loss; no segmentation output. [arXiv:2403.07815](https://arxiv.org/abs/2403.07815) |
| Moirai | Salesforce | `Salesforce/moirai-1.1-R-large` | Patch encoder, probabilistic forecasting; Moirai-MoE variant uses sparse MoE. |
| TimesFM | Google | `google/timesfm-1.0-200m` | Forecasting only. |
| TimeGPT | Nixtla | API (closed weights) | Forecasting + residual anomaly detection; API-only, not open weights. |

---

### 9.2 Untrained Architectures (Require Training)

The gate-closing problem is unusually well-suited to training from scratch:
`generate_synthetic_data()` provides an unlimited supply of perfectly
labelled synthetic examples at negligible cost. All architectures below
would be trained on such data.

#### 9.2.1 PatchTST — patch Transformer for classification (available in `transformers`)

- **HuggingFace:** integrated in `transformers` as
  `PatchTSTForClassification` / `PatchTSTForPrediction`.
  [Model doc](https://huggingface.co/docs/transformers/en/model_doc/patchtst)
- **Paper:** Nie, Y., et al. (2023). "A Time Series is Worth 64 Words:
  Long-term Forecasting with Transformers." *ICLR 2023*.
  [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- **Architecture:** divides the signal into fixed-length patches (tokens)
  and applies a channel-independent Transformer encoder. `PatchTSTFor
  Classification` adds a classification head over the patch embeddings.
- **Approach for this problem:** frame it as per-timestep regime labelling
  (4 classes). Breakpoints are read off as class-label boundaries.

  ```python
  from transformers import PatchTSTConfig, PatchTSTForClassification

  config = PatchTSTConfig(
      num_input_channels=1,
      context_length=1200,
      patch_length=16,        # 75 tokens for n=1200
      num_targets=4,          # 4 regimes
      d_model=64,
      num_attention_heads=4,
      num_hidden_layers=3,
  )
  model = PatchTSTForClassification(config)
  # Train on synthetic data: 10 000–100 000 examples, minutes on CPU
  ```

- **Advantage:** in the standard HuggingFace ecosystem; self-supervised
  pre-training (masked patch reconstruction) also supported, enabling
  pre-training on unlabelled closing curves before supervised fine-tuning.

#### 9.2.2 Anomaly Transformer — anomaly-attention mechanism

- **GitHub:** [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer)
- **Paper:** Xu, J., et al. (2022). "Anomaly Transformer: Time Series
  Anomaly Detection with Association Discrepancy." *ICLR 2022 Spotlight*.
- **Architecture:** introduces an "Anomaly Attention" block that computes an
  *association discrepancy* between point-wise and patch-wise attention
  distributions. Normal segments have high discrepancy; anomalies (and by
  extension, structural breaks) disrupt this pattern.
- **Limitation for this problem:** designed for point anomalies in
  multivariate industrial signals, not for piecewise-linear structural
  breaks in a univariate signal. Would require significant adaptation and
  domain-specific training. Likely overkill.

#### 9.2.3 TCDformer — Transformer with explicit change-point preprocessing

- **Paper:** Wan, R., et al. (2024). "TCDformer: A Transformer Framework
  for Non-stationary Time Series Forecasting Based on Trend and
  Change-Point Detection." *Neural Networks*, 173, 106196.
  [doi:10.1016/j.neunet.2024.106196](https://www.sciencedirect.com/science/article/abs/pii/S0893608024001205)
- **Architecture:** introduces a Local Linear Scaling Approximation (LLSA)
  module that explicitly detects change points as a preprocessing step,
  then feeds the reconstructed signal into a Transformer forecaster using
  wavelet attention for seasonal components and an MLP for trend.
- **Relevance:** the LLSA front-end is directly relevant — it is a learnable
  change-point detector that could be extracted and used standalone. The
  full TCDformer is forecasting-oriented and not applicable as-is.

#### 9.2.4 Custom Transformer with segmentation head (most principled approach)

For a task with perfectly known structure and unlimited synthetic training
data, a small purpose-built Transformer is arguably the cleanest option:

- **Input:** signal split into patches of length *p* (e.g., *p*=16 → 75
  tokens for *n*=1200)
- **Encoder:** standard Transformer (4 layers, 8 heads, d_model=64, ~500k
  parameters)
- **Head A:** per-token regime classifier (4 classes); breakpoints are
  class-label boundaries
- **Head B (optional):** per-segment slope regressor fitted after
  classification
- **Constraint (optional):** enforce the known regime ordering (0→1→2→3)
  via a CRF layer or a Viterbi decoding step over the classifier output
- **Training:** fully synthetic via `generate_synthetic_data()` with
  randomised noise levels, slopes, and breakpoint positions — infinite
  supply, no annotation cost
- **Training time:** minutes on CPU for a reasonably sized synthetic dataset

---

### 9.3 Comparison Table

| Model | Type | HuggingFace / Source | Applicable? | Effort |
|-------|------|----------------------|-------------|--------|
| MOMENT | Pre-trained encoder | [`AutonLab/MOMENT-1-large`](https://huggingface.co/AutonLab/MOMENT-1-large) | ✓ Anomaly/classification | Light fine-tuning or zero-shot |
| PatchTST | Untrained (HF arch) | [`transformers.PatchTST`](https://huggingface.co/docs/transformers/en/model_doc/patchtst) | ✓ Per-patch classification | Train on synthetic data |
| Custom Transformer | Untrained | PyTorch standard | ✓ Best structural fit | Train on synthetic data |
| TCDformer LLSA | Untrained (front-end) | [Neural Networks 2024](https://www.sciencedirect.com/science/article/abs/pii/S0893608024001205) | ~ LLSA front-end only | Adapt + train |
| Anomaly Transformer | Untrained | [`thuml/Anomaly-Transformer`](https://github.com/thuml/Anomaly-Transformer) | ~ Indirect, needs adapting | Adapt + train |
| Chronos-Bolt | Pre-trained forecaster | [`amazon/chronos-bolt-base`](https://huggingface.co/amazon/chronos-bolt-base) | ✗ Forecasting only | N/A |
| Moirai | Pre-trained forecaster | `Salesforce/moirai-1.1-R-large` | ✗ Forecasting only | N/A |
| TimesFM | Pre-trained forecaster | `google/timesfm-1.0-200m` | ✗ Forecasting only | N/A |
| TimeGPT | Pre-trained forecaster | API (closed weights) | ✗ Forecasting + API only | N/A |

### 9.4 Practical Recommendation

Start with **MOMENT in reconstruction mode** (zero-shot, ~5 lines of code)
to get a quick baseline: reconstruction-error peaks can be compared directly
against the breakpoints found by the classical methods in sections 3–4.

If accuracy is insufficient, train a **PatchTST classifier** on synthetic
data generated by `generate_synthetic_data()`. The per-patch classification
framing maps cleanly to the four-regime structure, the architecture is
available off-the-shelf from `transformers`, and the unlimited synthetic
supply makes annotation cost zero.

A **custom Transformer with a constrained segmentation head** (ordered-regime
CRF decoding) would be the most principled design if the goal is production
deployment with strict accuracy requirements.

---

### 9.5 Key References

18. Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., Dubrawski, A.
    (2024). "MOMENT: A Family of Open Time-series Foundation Models."
    *ICML 2024*. [arXiv:2402.03885](https://arxiv.org/abs/2402.03885)
19. Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J. (2023). "A Time
    Series is Worth 64 Words: Long-term Forecasting with Transformers."
    *ICLR 2023*. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
20. Xu, J., Wu, H., Wang, J., Long, M. (2022). "Anomaly Transformer: Time
    Series Anomaly Detection with Association Discrepancy." *ICLR 2022
    Spotlight*. [openreview](https://openreview.net/forum?id=LzQQ89U1qm_)
21. Wan, R., Xia, Y., et al. (2024). "TCDformer: A Transformer Framework
    for Non-stationary Time Series Forecasting Based on Trend and
    Change-Point Detection." *Neural Networks*, 173, 106196.
    [doi:10.1016/j.neunet.2024.106196](https://www.sciencedirect.com/science/article/abs/pii/S0893608024001205)
22. Ansari, A.F., et al. (2024). "Chronos: Learning the Language of Time
    Series." [arXiv:2403.07815](https://arxiv.org/abs/2403.07815)
23. Woo, G., et al. (2024). "Unified Training of Universal Time Series
    Forecasting Transformers (Moirai)."
    [Salesforce blog](https://www.salesforce.com/blog/moirai/)

---

## 10. Practical Recommendation

For a first pass, start with **Option A** (`segmented` in R or
`piecewise-regression` in Python). It is the simplest method that directly
addresses the problem, provides confidence intervals on both slopes and
breakpoints, and is fast enough for batch processing hundreds of closing
events.

If you need principled uncertainty quantification or want to encode the
known four-phase structure as a prior, move to **Option B** (`mcp`).

If you work in Python and need integration into a larger pipeline,
**Option D** (`ruptures` + `statsmodels` OLS) is the pragmatic choice.
