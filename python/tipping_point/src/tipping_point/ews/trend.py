"""Trend scores: Kendall's tau of an indicator against time, and the pipeline that turns a series
into the score s(t) of Bury et al. 2021 (PNAS): detrend, rolling indicator, then Kendall's tau of
the indicator over the trailing segment that ends at t.

kendall_tau is tau-b, scipy.stats.kendalltau(time, x), batched over rows: the discordant pairs are
the row's inversions, counted in O(n^1.5) by blocks of sqrt(n) (_discordant). Large batches run in
row chunks on a thread pool (numpy releases the GIL inside its loops).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import indicators


def _threads() -> int:
    value = os.environ.get("TP_THREADS")
    if not value:
        return min(8, os.cpu_count() or 1)
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f"TP_THREADS must be a positive integer: {value!r}")
    return int(value)


INDICATORS: dict[str, Callable[..., np.ndarray]] = {
    "variance": lambda x, window, lag: indicators.rolling_variance(x, window),
    "ac": lambda x, window, lag: indicators.rolling_autocorrelation(x, window, lag),
    "skewness": lambda x, window, lag: indicators.rolling_skewness(x, window),
    "level": lambda x, window, lag: indicators.rolling_mean(x, window),
    "slope": lambda x, window, lag: indicators.rolling_slope(x, window),
}
RAW = frozenset({"level", "slope"})  # read the raw series: residuals have no level to read
CHUNK = 1 << 17  # elements per row chunk: small enough to stay in cache
# 8 spares a shared machine; many-core hosts lift it, changing no result: rows are independent
THREADS = _threads()
SLICES = 1 << 23  # elements of s(t) slices held at once (64 MB)


def rowwise(
    fn: Callable[[np.ndarray], np.ndarray], rows: np.ndarray, width: int | None = None
) -> np.ndarray:
    """fn over chunks of the rows of a 2D array, in parallel, outputs stacked in row order. `width`
    is the elements fn handles per row, when that is not the row's own length."""
    step = max(1, CHUNK // max(1, width or rows.shape[1]))
    starts = range(0, len(rows), step)
    if len(starts) <= 1:
        return fn(rows)
    if THREADS == 1:  # no pool, but chunks still keep each pass in cache
        return np.concatenate([fn(rows[s : s + step]) for s in starts])
    with ThreadPoolExecutor(THREADS) as pool:
        parts = list(pool.map(lambda s: fn(rows[s : s + step]), starts))
    return np.concatenate(parts)


def _discordant(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row of finite values: (discordant pairs, tied pairs) against time.

    Ranks break ties by position, so a tied pair is never counted discordant. Positions are cut
    into blocks of about sqrt(n). A pair inside one block is discordant when its values say so,
    compared directly. A pair across two is discordant when the later block comes first in rank
    order: an inversion of the block ids read in rank order, which are cut the same way, into
    buckets of ranks. A pair inside one bucket is compared directly again, and pairs across two
    are counted from a table of how many of each block each bucket holds. The comparisons run with
    time along the first axis, as contiguous loops of whole rows."""
    b, m = rows.shape
    order = np.argsort(rows, axis=1)
    same = np.take_along_axis(rows, order, axis=1)
    same = same[:, 1:] == same[:, :-1]
    tied = np.zeros(b, dtype=np.int64)
    if same.any():  # a run of t equal values holds t(t-1)/2 tied pairs, in position order
        ties = same.any(axis=1)
        order[ties] = np.argsort(rows[ties], axis=1, kind="stable")
        index = np.arange(m)
        first = np.ones((b, m), dtype=bool)
        first[:, 1:] = ~same
        start = np.maximum.accumulate(np.where(first, index, 0), axis=1)
        tied = (index - start).sum(axis=1)
    shift = m.bit_length() // 2
    size, count = 1 << shift, -(-m // (1 << shift))
    width, symbols = count * size, count + 1  # pads: +inf and a last block, above all that precedes
    values = np.full((b, width), np.inf)
    values[:, :m] = rows
    blocks = np.full((b, width), count, dtype=np.int32)
    blocks[:, :m] = order >> shift
    cells = np.arange(b)[:, None] * count * symbols + (np.arange(width) >> shift) * symbols + blocks
    held = np.bincount(cells.ravel(), minlength=b * count * symbols).reshape(b, count, symbols)
    earlier = np.zeros_like(held)
    earlier[:, 1:] = np.cumsum(held[:, :-1], axis=1)
    above = np.zeros_like(held)
    above[:, :, :-1] = np.cumsum(earlier[:, :, :0:-1], axis=2)[:, :, ::-1]
    discordant = (held * above).sum(axis=(1, 2))

    def flat(a: np.ndarray) -> np.ndarray:
        return a.reshape(b, count, size).transpose(2, 0, 1).reshape(size, b * count)

    values, ids = flat(values), flat(blocks.astype(np.int16 if symbols < 1 << 15 else np.int32))
    kind = np.uint8 if size <= 1 << 8 else np.uint32  # counts per position: at most size - 1
    seen, passed = np.zeros((2, size, b * count), dtype=kind)
    mask = np.empty((size, b * count), dtype=bool)
    counts = mask.view(np.uint8)
    for d in range(1, size):
        r = size - d
        np.greater(values[:r], values[d:], out=mask[:r])
        np.add(seen[d:], counts[:r], out=seen[d:])
        np.greater(ids[:r], ids[d:], out=mask[:r])
        np.add(passed[d:], counts[:r], out=passed[d:])
    inside = seen.sum(axis=0, dtype=np.int64) + passed.sum(axis=0, dtype=np.int64)
    return discordant + inside.reshape(b, count).sum(axis=1), tied


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
    """Series -> score: detrend (Gaussian, or none; an indicator in RAW reads the raw series
    whatever the bandwidth), rolling indicator, Kendall's tau of the last `length` indicator values.
    A score needs `span` = window + length - 1 samples."""

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
        bandwidth = self.bandwidth
        if bandwidth is None or self.indicator in RAW:
            return x
        rows = x.reshape(-1, x.shape[-1])
        return rowwise(lambda r: indicators.gaussian_detrend(r, bandwidth), rows).reshape(x.shape)

    def rolling(self, x: np.ndarray) -> np.ndarray:
        """The indicator along each whole row of x, detrended as a score's input is: value t
        describes the window ending at t (NaN for the first window - 1). `length` plays no part."""
        return INDICATORS[self.indicator](self.residuals(x), self.window, self.lag)

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
    span = pipeline.span
    if n < span:
        raise ValueError(f"s(t) needs {span} samples, got {n}")
    times = np.arange(span - 1, n, every)
    rows = x.reshape(-1, n)
    scores = np.empty((len(rows), len(times)))
    per = max(1, SLICES // (len(rows) * span))  # times per batch: the slices are copies
    for k in range(0, len(times), per):
        t = times[k : k + per]
        slices = rows[:, t[:, None] - span + 1 + np.arange(span)]  # (rows, len t, span)
        scores[:, k : k + per] = pipeline(slices.reshape(-1, span)).reshape(len(rows), len(t))
    return times, scores.reshape(*lead, len(times))
