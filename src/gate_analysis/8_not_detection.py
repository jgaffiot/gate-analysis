"""Option H: Narrowest-Over-Threshold (NOT) change-point detection.

Python reimplementation of the NOT algorithm (Baranowski, Chen & Fryzlewicz,
2019, JRSS-B 81(3):649-672) adapted for kink detection (change in slope) in
piecewise-linear signals.

Algorithm outline
-----------------
1. Draw M random sub-intervals [s_m, e_m] of the signal.
2. For each interval, sweep all candidate kink positions and record the
   maximum RSS reduction (delta_RSS = RSS_null - RSS_left - RSS_right) and
   its location.  OLS fits are evaluated in O(1) via prefix sums, so the
   inner sweep is fully vectorised.
3. Retain "significant" intervals: those where max delta_RSS > threshold.
   The threshold is calibrated via a Bonferroni bound so that the
   false-positive rate across all sampled (interval, split) pairs is < 1 %.
4. Iteratively declare change points (narrowest-first rule):
   a. Pick the narrowest remaining significant interval.
   b. Declare its argmax position as a change point.
   c. Discard all remaining significant intervals that contain this position.
   d. Repeat until none remain.
5. Estimate slopes by OLS on each resulting segment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from gate_analysis.common import GateData, generate_synthetic_data, plot_results


# ---------------------------------------------------------------------------
# Prefix-sum helpers for O(1) OLS RSS on arbitrary sub-intervals
# ---------------------------------------------------------------------------


def _build_prefix_sums(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Return five prefix-sum arrays (length n+1) for OLS sufficient statistics."""
    z = np.zeros(1)
    return (
        np.concatenate([z, np.cumsum(time)]),
        np.concatenate([z, np.cumsum(time**2)]),
        np.concatenate([z, np.cumsum(position)]),
        np.concatenate([z, np.cumsum(time * position)]),
        np.concatenate([z, np.cumsum(position**2)]),
    )


def _rss_scalar(
    p_t: npt.NDArray[np.floating[Any]],
    p_t2: npt.NDArray[np.floating[Any]],
    p_y: npt.NDArray[np.floating[Any]],
    p_ty: npt.NDArray[np.floating[Any]],
    p_y2: npt.NDArray[np.floating[Any]],
    s: int,
    e: int,
) -> float:
    """OLS RSS for segment [s, e] (inclusive). O(1) via prefix sums."""
    n = e - s + 1
    if n < 2:
        return 0.0
    st = p_t[e + 1] - p_t[s]
    st2 = p_t2[e + 1] - p_t2[s]
    sy = p_y[e + 1] - p_y[s]
    sty = p_ty[e + 1] - p_ty[s]
    sy2 = p_y2[e + 1] - p_y2[s]
    sty_c = sty - st * sy / n
    st2_c = st2 - st**2 / n
    sy2_c = sy2 - sy**2 / n
    if abs(st2_c) < 1e-12:
        return max(float(sy2_c), 0.0)
    return max(float(sy2_c - sty_c**2 / st2_c), 0.0)


def _rss_vectorized(
    p_t: npt.NDArray[np.floating[Any]],
    p_t2: npt.NDArray[np.floating[Any]],
    p_y: npt.NDArray[np.floating[Any]],
    p_ty: npt.NDArray[np.floating[Any]],
    p_y2: npt.NDArray[np.floating[Any]],
    s_arr: npt.NDArray[np.intp],
    e_arr: npt.NDArray[np.intp],
) -> npt.NDArray[np.floating[Any]]:
    """Vectorised OLS RSS for multiple segments [s_arr[i], e_arr[i]]."""
    n = (e_arr - s_arr + 1).astype(float)
    st = p_t[e_arr + 1] - p_t[s_arr]
    st2 = p_t2[e_arr + 1] - p_t2[s_arr]
    sy = p_y[e_arr + 1] - p_y[s_arr]
    sty = p_ty[e_arr + 1] - p_ty[s_arr]
    sy2 = p_y2[e_arr + 1] - p_y2[s_arr]
    sty_c = sty - st * sy / n
    st2_c = st2 - st**2 / n
    sy2_c = sy2 - sy**2 / n
    safe = np.abs(st2_c) > 1e-12
    rss = np.where(safe, sy2_c - sty_c**2 / np.where(safe, st2_c, 1.0), sy2_c)
    return np.maximum(rss, 0.0)


# ---------------------------------------------------------------------------
# NOT main function
# ---------------------------------------------------------------------------


def not_detection(
    time: npt.NDArray[np.floating[Any]],
    position: npt.NDArray[np.floating[Any]],
    *,
    threshold: float | None = None,
    n_intervals: int = 5000,
    min_segment_len: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Narrowest-Over-Threshold detection of slope change points.

    Parameters
    ----------
    time : array
        Uniformly-spaced time values.
    position : array
        Gate position values (%).
    threshold : float or None
        RSS-reduction threshold for significance.  If None, derived from the
        noise level via a Bonferroni bound over all sampled (interval, split)
        pairs so that the global false-positive rate is < 1 %.
    n_intervals : int
        Number of random sub-intervals to draw (default 5 000).
    min_segment_len : int
        Minimum samples in each half of a candidate kink split (default 10).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        breakpoints          – detected breakpoint times (s)
        slopes               – OLS slope per segment (%/s)
        fitted               – OLS fitted values (same length as ``time``)
        change_point_indices – sample indices of the detected change points
        n_significant        – number of significant sub-intervals found
    """
    n = len(time)
    rng = np.random.default_rng(seed)

    # ── Noise estimate via MAD of first differences ──────────────────────────
    # For i.i.d. N(0, sigma^2) noise, differences are N(0, 2*sigma^2).
    # MAD(|diff|) = 0.6745 * sqrt(2) * sigma ≈ 0.9535 * sigma.
    diffs = np.diff(position)
    sigma_noise = float(np.median(np.abs(diffs))) / 0.9535
    sigma2 = sigma_noise**2

    # ── Threshold ────────────────────────────────────────────────────────────
    # Under H0, delta_RSS(k) ~ chi^2(2) * sigma^2 for each split k, i.e.
    # P(delta_RSS > t) = exp(-t / (2*sigma^2)).
    # Bonferroni over n_intervals * n (interval, split) pairs:
    #   threshold = sigma^2 * 2 * log(n_intervals * n / alpha)
    if threshold is None:
        alpha = 0.01
        threshold = sigma2 * 2.0 * np.log(n_intervals * n / alpha)

    # ── Prefix sums ──────────────────────────────────────────────────────────
    p_t, p_t2, p_y, p_ty, p_y2 = _build_prefix_sums(time, position)

    # ── Random sub-intervals ─────────────────────────────────────────────────
    # Each interval must be long enough to accommodate two sub-segments.
    min_len = 4 * min_segment_len
    starts = rng.integers(0, n - min_len, size=n_intervals)
    # Lengths drawn uniformly between min_len and the remaining signal length.
    max_extra = n - starts - min_len
    lengths = min_len + (rng.uniform(size=n_intervals) * max_extra).astype(int)
    ends = np.minimum(starts + lengths, n - 1)

    # ── Evaluate each interval (vectorised inner loop) ────────────────────────
    # significant: list of (interval_length, s, e, tau)
    significant: list[tuple[int, int, int, int]] = []

    for s, e in zip(starts.tolist(), ends.tolist()):
        s, e = int(s), int(e)
        k_min = s + min_segment_len - 1  # last index of left sub-segment
        k_max = e - min_segment_len  # right sub-segment starts at k+1
        if k_min > k_max:
            continue

        ks = np.arange(k_min, k_max + 1, dtype=np.intp)
        rss_null = _rss_scalar(p_t, p_t2, p_y, p_ty, p_y2, s, e)
        rss_L = _rss_vectorized(p_t, p_t2, p_y, p_ty, p_y2, np.full_like(ks, s), ks)
        rss_R = _rss_vectorized(p_t, p_t2, p_y, p_ty, p_y2, ks + 1, np.full_like(ks, e))
        delta = rss_null - (rss_L + rss_R)
        best_idx = int(np.argmax(delta))
        if float(delta[best_idx]) > threshold:
            significant.append((e - s, s, e, int(ks[best_idx])))

    # ── Narrowest-first change-point extraction ───────────────────────────────
    # Sort ascending by interval length; narrowest interval is always at index 0.
    significant.sort(key=lambda x: x[0])
    change_point_indices: list[int] = []
    remaining = list(significant)

    while remaining:
        _, _s, _e, tau_star = remaining[0]
        change_point_indices.append(tau_star)
        # Discard every remaining interval that contains tau_star.
        remaining = [
            item for item in remaining[1:] if not (item[1] <= tau_star <= item[2])
        ]

    change_point_indices.sort()

    # ── OLS slopes and fitted values per segment ──────────────────────────────
    breakpoints_t = [float(time[cp]) for cp in change_point_indices]
    seg_bounds = [0, *[cp + 1 for cp in change_point_indices], n]
    slopes: list[float] = []
    fitted = np.empty(n)

    for i in range(len(seg_bounds) - 1):
        sl, sr = seg_bounds[i], seg_bounds[i + 1]
        if sr - sl < 2:
            slopes.append(float("nan"))
            fitted[sl:sr] = position[sl:sr]
            continue
        lr = stats.linregress(time[sl:sr], position[sl:sr])
        slopes.append(float(lr.slope))
        fitted[sl:sr] = lr.intercept + lr.slope * time[sl:sr]

    print("=== Option H: Narrowest-Over-Threshold (NOT) ===")
    print(f"Noise estimate: {sigma_noise:.3f} %,  threshold: {threshold:.1f}")
    print(f"Significant sub-intervals: {len(significant)} / {n_intervals}")
    print(f"Breakpoints: {[f'{bp:.3f}' for bp in breakpoints_t]}")
    print(f"Slopes: {[f'{s:.2f}' for s in slopes]} %/s")

    return {
        "breakpoints": breakpoints_t,
        "slopes": slopes,
        "fitted": fitted,
        "change_point_indices": change_point_indices,
        "n_significant": len(significant),
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


if __name__ == "__main__":
    from bokeh.io import show

    data = generate_synthetic_data()
    result = not_detection(data.time, data.position)
    segments = _build_segments(data, result)
    fig = plot_results(
        data,
        "Option H: Narrowest-Over-Threshold (NOT)",
        fitted_segments=segments,
        detected_breakpoints=result["breakpoints"],
        estimated_slopes=result["slopes"],
    )
    show(fig)
