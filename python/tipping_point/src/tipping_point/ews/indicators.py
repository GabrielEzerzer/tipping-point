"""Classic early-warning indicators on (..., n) arrays: time is the last axis, and every window,
lag and bandwidth is in samples.

The definitions are ewstools 2.1.3's, pinned by a golden fixture (tests/fixtures/ews_golden.json):
  gaussian_smooth          scipy.ndimage.gaussian_filter(x, sigma, mode="reflect"), the kernel's
                           quartiles at +-bandwidth/4 (sigma = 0.25/0.675 bandwidth) and cut at
                           4 sigma; computed here by FFT
  rolling_variance         sample variance (ddof 1) of each trailing window
  rolling_autocorrelation  Pearson correlation of the window's (x_s, x_s+lag) pairs, as pandas'
                           Series.autocorr: each side of the pairs keeps its own mean and spread
  rolling_skewness         bias-corrected sample skewness G1 = sqrt(w (w-1)) / (w-2) m3 / m2^1.5
The rolling value at index t describes the window x[t - window + 1 .. t]. The first window - 1
values are NaN, as in pandas, so an output lines up with its input. A window without spread has
variance 0 and no autocorrelation or skewness (NaN).

Windows are summed from running sums, O(n) per series; each value carries a bound on its rounding
error, and any window whose bound exceeds 1e-9 (a flat window, or one whose level sits far above
its own spread) is recomputed two-pass from its own samples.

Bursts: configurations saved in bursts of consecutive sweeps give one-sweep pairs even when the
bursts are far apart; burst_autocorrelation pools those pairs and never pairs two bursts.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SIGMA_PER_BANDWIDTH = 0.25 / 0.675  # a Gaussian's quartiles sit at +-0.675 sigma
TRUNCATE = 4.0  # scipy's default: the kernel stops at 4 sigma
_ROUNDING = 16 * float(np.finfo(np.float64).eps)  # per running-sum term, with margin
_TRUST = 1e-9  # a fast value that may be off by more is recomputed from its window


def gaussian_kernel(sigma: float) -> np.ndarray:
    """scipy.ndimage's normalised kernel: radius int(4 sigma + 0.5), so a tiny sigma gives [1]."""
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    radius = int(TRUNCATE * sigma + 0.5)
    t = np.arange(-radius, radius + 1)
    phi = np.exp(-0.5 / (sigma * sigma) * t**2)
    return phi / phi.sum()


def _fast_size(n: int) -> int:
    """The smallest 2^a 3^b 5^c >= n: lengths the FFT handles without a large prime factor."""
    best = 1 << max(0, (n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            size = p35
            while size < n:
                size *= 2
            best = min(best, size)
            p35 *= 3
        p5 *= 5
    return best


def gaussian_smooth(x: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian smoothing along the last axis, edges reflected (d c b a | a b c d | d c b a).

    Equal to scipy.ndimage.gaussian_filter(x, SIGMA_PER_BANDWIDTH * bandwidth, mode="reflect")
    to rounding; the convolution runs by FFT, so its cost does not grow with the bandwidth."""
    x = np.asarray(x, dtype=np.float64)
    kernel = gaussian_kernel(SIGMA_PER_BANDWIDTH * bandwidth)
    r = len(kernel) // 2
    if r == 0:
        return x.copy()
    n = x.shape[-1]
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(r, r)], mode="symmetric")
    size = _fast_size(n + 4 * r)  # the full linear convolution: no wrap-around
    spectrum = np.fft.rfft(padded, size, axis=-1) * np.fft.rfft(kernel, size)
    return np.fft.irfft(spectrum, size, axis=-1)[..., 2 * r : 2 * r + n]


def gaussian_detrend(x: np.ndarray, bandwidth: float) -> np.ndarray:
    """Residuals after Gaussian smoothing: x - gaussian_smooth(x, bandwidth)."""
    x = np.asarray(x, dtype=np.float64)
    return x - gaussian_smooth(x, bandwidth)


def _series(x: np.ndarray, window: int, smallest: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError("expected a series, got a scalar")
    if not smallest <= window <= x.shape[-1]:
        raise ValueError(f"window {window} must lie in [{smallest}, {x.shape[-1]}] samples")
    return x


def _window_sums(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum over every full trailing window of the last axis (n - window + 1 values), and the
    rounding error each can carry: it is a difference of two running sums, and a running sum of
    t terms is off by about eps sqrt(t) times its size."""
    c = np.cumsum(y, axis=-1)
    sums = c[..., window - 1 :].copy()
    sums[..., 1:] -= c[..., :-window]
    size = np.abs(c[..., window - 1 :])
    size[..., 1:] += np.abs(c[..., :-window])
    terms = np.arange(window, y.shape[-1] + 1)
    return sums, _ROUNDING * np.sqrt(terms) * size


def _centred(x: np.ndarray) -> np.ndarray:
    """(rows, n), each row minus its mean: running sums of a centred row stay small, so their
    differences keep their digits."""
    rows = x.reshape(-1, x.shape[-1])
    return rows - rows.mean(axis=-1, keepdims=True)


def _redo(
    y: np.ndarray,
    out: np.ndarray,
    unsure: np.ndarray,
    window: int,
    exact: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Recompute, two-pass from the window itself, every window whose fast value may be off by
    more than _TRUST: flat windows, and windows a level far above their own spread swamps."""
    rows, starts = np.nonzero(unsure)
    step = max(1, (1 << 20) // window)
    for k in range(0, len(rows), step):
        r, p = rows[k : k + step], starts[k : k + step]
        out[r, p] = exact(y[r[:, None], p[:, None] + np.arange(window)])


def _lined_up(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Back to the input's shape, NaN first, so that output index t is the window ending at t."""
    out = np.full(shape, np.nan)
    out.reshape(-1, shape[-1])[:, shape[-1] - values.shape[-1] :] = values
    return out


def _deviations(x: np.ndarray) -> np.ndarray:
    """x minus its mean along the last axis, shifted by its first value before averaging: a
    constant stretch then gives exact zeros, where its rounded mean would leave 1-ulp noise."""
    d = x - x[..., :1]
    return d - d.mean(axis=-1, keepdims=True)


def _exact_variance(windows: np.ndarray) -> np.ndarray:
    d = _deviations(windows)
    return np.sum(d * d, axis=-1) / (windows.shape[-1] - 1)


def rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    x = _series(x, window, 2)
    y = _centred(x)
    s1, e1 = _window_sums(y, window)
    s2, e2 = _window_sums(y * y, window)
    ss = s2 - s1 * s1 / window  # sum of squared deviations from the window mean
    error = e2 + 2.0 * np.sqrt(np.abs(s2) / window) * e1
    var = ss / (window - 1)
    _redo(y, var, ~(error <= _TRUST * ss), window, _exact_variance)
    return _lined_up(var, x.shape)


def rolling_autocorrelation(x: np.ndarray, window: int, lag: int = 1) -> np.ndarray:
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    x = _series(x, window, lag + 2)  # a correlation needs at least two pairs
    y = _centred(x)
    k = window - lag  # pairs per window
    a, b = y[:, :-lag], y[:, lag:]
    (sa, ea), (sb, eb) = _window_sums(a, k), _window_sums(b, k)
    (saa, eaa), (sbb, ebb) = _window_sums(a * a, k), _window_sums(b * b, k)
    sab, eab = _window_sums(a * b, k)
    ssa, ssb, cov = saa - sa * sa / k, sbb - sb * sb / k, sab - sa * sb / k
    err_a = eaa + 2.0 * np.sqrt(np.abs(saa) / k) * ea
    err_b = ebb + 2.0 * np.sqrt(np.abs(sbb) / k) * eb
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(ssa * ssb)
        error = (eab + (np.abs(sb) * ea + np.abs(sa) * eb) / k) / np.sqrt(np.abs(ssa * ssb))
        error += np.abs(r) * (err_a / np.abs(ssa) + err_b / np.abs(ssb)) / 2
    unsure = ~(error <= _TRUST) | (ssa <= 0) | (ssb <= 0)  # a negative spread is rounding
    _redo(y, r, unsure, window, lambda w: _pearson(w[:, :-lag], w[:, lag:]))
    return _lined_up(r, x.shape)


def _exact_skewness(windows: np.ndarray) -> np.ndarray:
    w = windows.shape[-1]
    d = _deviations(windows)
    m2, m3 = np.mean(d * d, axis=-1), np.mean(d**3, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = np.sqrt(w * (w - 1.0)) / (w - 2) * m3 / m2**1.5
    return np.where(m2 > 0, g1, np.nan)


def rolling_skewness(x: np.ndarray, window: int) -> np.ndarray:
    x = _series(x, window, 3)
    y = _centred(x)
    w = window
    (s1, e1), (s2, e2), (s3, e3) = (_window_sums(y**p, w) for p in (1, 2, 3))
    mean = s1 / w
    m2 = s2 / w - mean * mean  # biased central moments of the window
    m3 = s3 / w - 3 * mean * s2 / w + 2 * mean**3
    err2 = (e2 + 2.0 * np.sqrt(np.abs(s2) / w) * e1) / w
    err3 = (e3 + 3 * np.abs(mean) * e2 + (3 * np.abs(s2) / w + 6 * mean * mean) * e1) / w
    scale = np.sqrt(w * (w - 1.0)) / (w - 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = scale * m3 / m2**1.5
        error = scale * (err3 / m2**1.5 + 1.5 * np.abs(m3) * err2 / m2**2.5)
    _redo(y, g1, ~(error <= _TRUST), w, _exact_skewness)
    return _lined_up(g1, x.shape)


def _pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation along the last axis, two-pass; NaN when either side has no spread."""
    da, db = _deviations(a), _deviations(b)
    saa, sbb = np.sum(da * da, axis=-1), np.sum(db * db, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.sum(da * db, axis=-1) / np.sqrt(saa * sbb)
    return np.where((saa > 0) & (sbb > 0), r, np.nan)


def burst_autocorrelation(x: np.ndarray, lag: int = 1) -> np.ndarray:
    """Pearson correlation of the (frame f, frame f + lag) pairs of x (..., bursts, frames),
    pooled over the bursts; frames of different bursts are never paired."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 2 or not 1 <= lag < x.shape[-1]:
        raise ValueError(f"need (..., bursts, frames) with frames > lag = {lag}, got {x.shape}")
    pooled = (*x.shape[:-2], -1)
    return _pearson(x[..., :-lag].reshape(pooled), x[..., lag:].reshape(pooled))
