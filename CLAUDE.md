# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run a specific analysis method (headless)
MPLBACKEND=Agg uv run python -m gate_analysis.1_segmented_regression

# Run with plot display
uv run python -m gate_analysis.1_segmented_regression

# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Type check
uv run ty check src/
```

Replace `1_segmented_regression` with any of the 8 module names.

## Architecture

This is a demo/research project (Python 3.13, managed with `uv`) that implements and compares eight methods for detecting breakpoints and measuring slopes in pipe gate closing signals.

**Package layout:** `src/gate_analysis/` (src layout, installed as editable via `uv_build`).

**`common.py`** is the shared foundation used by all demo scripts:
- `GateData` dataclass holds a **polars DataFrame** (`df`) with a `date_time` column (`Datetime[us]`) and one or more gate position columns, plus ground-truth `breakpoints`, `slopes`, `plateaus`.  The `time` property returns elapsed seconds as a numpy array.  The `gate_columns` property lists the position column names.
- `generate_synthetic_data(n_gates=2)` produces the canonical test signal (dt=0.01s, seed=42) with breakpoints at 2.0, 5.0, 9.0 s and slopes of -25.0 and -5.0 %/s.  Each gate receives slightly perturbed slopes (±5 %) and noise (±10 %).
- `plot_results()` overlays all gates on a single figure; its `fitted_segments`, `detected_breakpoints`, and `estimated_slopes` parameters are dicts keyed by gate column name.

**Each numbered script** (`0_…` through `6_…` and `8_…`) is self-contained and follows the same pattern:
1. A core analysis function (e.g. `curve_fit_piecewise(time, position)`) that operates on numpy arrays — unchanged.
2. An `analyze(data: GateData)` function that loops over all gate columns, returning `(results, segments)` dicts keyed by gate column name.
3. An `if __name__ == "__main__"` guard that calls `analyze()`, then `plot_results()` with the per-gate results; invoked as modules (`python -m gate_analysis.<name>`).

**Signal structure the methods target:** high plateau → fast linear ramp → slow linear ramp → low plateau (4 regimes, 3 breakpoints).

## Method summary

| # | File | Library | Accuracy |
|---|------|---------|----------|
| 0 | `0_curve_fit.py` | `scipy.optimize` | Best accuracy + speed; requires known topology |
| 1 | `1_segmented_regression.py` | `piecewise-regression` | Accurate, ~0.9 s (n_boot=10) |
| 2 | `2_bayesian_changepoint.py` | `scipy.optimize` (MAP + Laplace) | Best UQ, ~0.25 s |
| 3 | `3_cpop_piecewise_linear.py` | numpy only (DP + BIC) | Accurate, ~2 s; ~0.3 s with n_breakpoints=3 |
| 4 | `4_ruptures_ols.py` | `ruptures` + `scipy` | Less accurate (L2 cost) |
| 5 | `5_savitzky_golay.py` | `scipy.signal` | Heuristic, fast |
| 6 | `6_kalman_filter.py` | `filterpy` + `scipy.stats` | Accurate (~50 ms); CUSUM restart, online-capable |
| 8 | `8_not_detection.py` | numpy + scipy | Accurate, ~1 s; no breakpoint count needed |
