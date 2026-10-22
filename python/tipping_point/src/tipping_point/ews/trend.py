"""Trend scores: Kendall's tau of an indicator against time, and the pipeline that turns a series
into the score s(t) of Bury et al. 2021 (PNAS): detrend, rolling indicator, then Kendall's tau of
the indicator over the trailing segment that ends at t.

kendall_tau is tau-b, scipy.stats.kendalltau(time, x), batched over rows and O(n log n) per row:
the discordant pairs are the row's inversions, counted by a merge sort run top-down on its ranks.
Large batches run in row chunks on a thread pool (numpy releases the GIL inside its loops).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import indicators

INDICATORS: dict[str, Callable[..., np.ndarray]] = {
    "variance": lambda x, window, lag: indicators.rolling_variance(x, window),
    "ac": lambda x, window, lag: indicators.rolling_autocorrelation(x, window, lag),
    "skewness": lambda x, window, lag: indicators.rolling_skewness(x, window),
}
CHUNK = 1 << 17  # elements per row chunk: small enough to stay in cache
THREADS = min(8, os.cpu_count() or 1)


def rowwise(fn: Callable[[np.ndarray], np.ndarray], rows: np.ndarray) -> np.ndarray:
    """fn over chunks of the rows of a 2D array, in parallel, outputs stacked in row order."""
    step = max(1, CHUNK // max(1, rows.shape[1]))
    starts = range(0, len(rows), step)
    if len(starts) <= 1 or THREADS == 1:
        return fn(rows)
    with ThreadPoolExecutor(THREADS) as pool:
        parts = list(pool.map(lambda s: fn(rows[s : s + step]), starts))
    return np.concatenate(parts)


def _discordant(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row of finite values: (discordant pairs, tied pairs) against time.

    Rows are padded with +inf to M = 2^k; pads rank last, so they pair with nothing discordantly.
    Ranks break ties by position, so a tied pair is never counted discordant. At each level the
    row is a sequence of groups of S positions (one aligned block each) in rank order; a position
    in the block's right half is discordant with every left-half position ranked above it. The
    groups are then split stably by that bit, which gives the next level's groups."""
    b, m = rows.shape
    M = 1 << (m - 1).bit_length()
    small = np.int16 if M <= 1 << 15 else np.int32
    padded = rows
    if m < M:
        padded = np.full((b, M), np.inf)
        padded[:, :m] = rows
    order = np.argsort(padded, axis=1, kind="stable")
    ordered = np.take_along_axis(padded, order, axis=1)[:, :m]
    tied = np.zeros(b, dtype=np.int64)
    same = ordered[:, 1:] == ordered[:, :-1]
    if same.any():  # a run of t equal values holds t(t-1)/2 tied pairs
        index = np.arange(m)
        first = np.ones((b, m), dtype=bool)
        first[:, 1:] = ~same
        start = np.maximum.accumulate(np.where(first, index, 0), axis=1)
        tied = (index - start).sum(axis=1)
    seq = order.astype(small)
    discordant = np.zeros(b, dtype=np.int64)
    base = (np.arange(b, dtype=np.intp) * M)[:, None]
    S = M
    while S >= 2:
        half, groups = S // 2, M // S
        left = ((seq & half) == 0).reshape(b, groups, S)
        seen = np.cumsum(left, axis=2, dtype=small)  # left positions ranked at or below each
        # sum over right positions of (half - seen); the left ones' seen sum to half(half+1)/2
        right_seen = seen.sum(axis=(1, 2), dtype=np.int64) - groups * half * (half + 1) // 2
        discordant += half * (M // 2) - right_seen
        if S > 2:
            within = np.where(left, seen - 1, half + np.arange(S, dtype=small) - seen)
            within += (np.arange(groups, dtype=small) * S)[:, None]
            split = np.empty_like(seq)
            split.ravel()[(within.reshape(b, M) + base).ravel()] = seq.ravel()
            seq = split
        S = half
    return discordant, tied


def kendall_tau(x: np.ndarray) -> np.ndarray:
    """Kendall's tau-b of each row of x (..., n) against time 0..n-1; a row holding a NaN or an
    infinity, or a constant row, gives NaN."""
    x = np.asarray(x, dtype=np.float64)
    *lead, n = x.shape
    rows = x.reshape(-1, n)
    tau = np.full(len(rows), np.nan)
    good = np.isfinite(rows).all(axis=1)
    if n < 2 or not good.any():
        return tau.reshape(lead)
    counts = rowwise(lambda r: np.stack(_discordant(r), axis=1), rows[good])
    discordant, tied = counts[:, 0].astype(np.float64), counts[:, 1].astype(np.float64)
    pairs = n * (n - 1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        tau[good] = (pairs - tied - 2 * discordant) / np.sqrt((pairs - tied) * pairs)
    return tau.reshape(lead)


@dataclass(frozen=True)
class Pipeline:
    """Series -> score: detrend (Gaussian, or none), rolling indicator, Kendall's tau of the last
    `length` indicator values. A score needs `span` = window + length - 1 samples."""

    indicator: str  # a key of INDICATORS
    window: int  # samples per rolling window
    length: int  # indicator values the trend is scored over: the trailing segment
    bandwidth: float | None = None  # Gaussian detrending bandwidth in samples; None: no detrend
    lag: int = 1  # autocorrelation lag in samples

    def __post_init__(self) -> None:
        if self.indicator not in INDICATORS:
            raise ValueError(f"indicator must be one of {sorted(INDICATORS)}: {self.indicator!r}")
        if self.length < 2 or self.window < 2:
            raise ValueError(f"window and length must be >= 2: {self.window}, {self.length}")

    @property
    def span(self) -> int:
        return self.window + self.length - 1

    def residuals(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.bandwidth is None:
            return x
        return indicators.gaussian_detrend(x, self.bandwidth)

    def trend(self, residuals: np.ndarray) -> np.ndarray:
        """Kendall's tau of the indicator over the last `length` samples of each row."""
        residuals = np.asarray(residuals, dtype=np.float64)
        *lead, n = residuals.shape
        if n < self.span:
            raise ValueError(f"a score needs {self.span} samples, got {n}")
        tail = residuals.reshape(-1, n)[:, n - self.span :]
        compute = INDICATORS[self.indicator]
        values = rowwise(lambda r: compute(r, self.window, self.lag)[:, self.window - 1 :], tail)
        return kendall_tau(values).reshape(lead)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.trend(self.residuals(x))


def trailing_score(
    x: np.ndarray, pipeline: Pipeline, every: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """s(t) along each row of x (..., n): the pipeline applied to the span samples ending at t,
    each slice detrended on its own so no score sees past its t. Returns (t, scores (..., len t))
    for t = span - 1, span - 1 + every, ... < n."""
    x = np.asarray(x, dtype=np.float64)
    *lead, n = x.shape
    if n < pipeline.span:
        raise ValueError(f"s(t) needs {pipeline.span} samples, got {n}")
    times = np.arange(pipeline.span - 1, n, every)
    slices = np.lib.stride_tricks.sliding_window_view(x, pipeline.span, axis=-1)[..., ::every, :]
    scores = pipeline(slices.reshape(-1, pipeline.span))
    return times, scores.reshape(*lead, len(times))
