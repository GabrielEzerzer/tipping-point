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

Bursts: configurations saved in bursts of consecutive sweeps give one-sweep pairs even when the
bursts are far apart; burst_autocorrelation pools those pairs and never pairs two bursts.
"""

from __future__ import annotations

import numpy as np

SIGMA_PER_BANDWIDTH = 0.25 / 0.675  # a Gaussian's quartiles sit at +-0.675 sigma
TRUNCATE = 4.0  # scipy's default: the kernel stops at 4 sigma
_FLAT = 1e-10  # a window whose variance is below this share of the series' is treated as flat


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


def _window_sums(y: np.ndarray, window: int) -> np.ndarray:
    """Sum over every full trailing window of the last axis: n - window + 1 values."""
    c = np.cumsum(y, axis=-1)
    sums = c[..., window - 1 :].copy()
    sums[..., 1:] -= c[..., :-window]
    return sums


def _centred(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x minus its mean, and the flatness floor: running sums of a centred series stay small,
    so their differences keep their digits; a window's sum of squares below the floor is 0."""
    y = x - x.mean(axis=-1, keepdims=True)
    return y, _FLAT * np.mean(y * y, axis=-1, keepdims=True)


def _lined_up(values: np.ndarray, n: int) -> np.ndarray:
    """Prefix NaN so that output index t is the window ending at input index t."""
    out = np.full((*values.shape[:-1], n), np.nan)
    out[..., n - values.shape[-1] :] = values
    return out


def rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    x = _series(x, window, 2)
    y, floor = _centred(x)
    s1, s2 = _window_sums(y, window), _window_sums(y * y, window)
    ss = s2 - s1 * s1 / window  # sum of squared deviations from the window mean
    var = np.where(ss <= floor * window, 0.0, ss / (window - 1))
    return _lined_up(var, x.shape[-1])


def rolling_autocorrelation(x: np.ndarray, window: int, lag: int = 1) -> np.ndarray:
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    x = _series(x, window, lag + 2)  # a correlation needs at least two pairs
    y, floor = _centred(x)
    k = window - lag  # pairs per window
    a, b = y[..., :-lag], y[..., lag:]
    sa, sb = _window_sums(a, k), _window_sums(b, k)
    ssa = _window_sums(a * a, k) - sa * sa / k
    ssb = _window_sums(b * b, k) - sb * sb / k
    cov = _window_sums(a * b, k) - sa * sb / k
    flat = (ssa <= floor * k) | (ssb <= floor * k)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(flat, np.nan, cov / np.sqrt(np.abs(ssa * ssb)))
    return _lined_up(r, x.shape[-1])


def rolling_skewness(x: np.ndarray, window: int) -> np.ndarray:
    x = _series(x, window, 3)
    y, floor = _centred(x)
    w = window
    mean = _window_sums(y, w) / w
    m2_raw, m3_raw = _window_sums(y * y, w) / w, _window_sums(y**3, w) / w
    m2 = m2_raw - mean * mean  # biased central moments of the window
    m3 = m3_raw - 3 * mean * m2_raw + 2 * mean**3
    flat = m2 * w <= floor * w
    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = np.sqrt(w * (w - 1.0)) / (w - 2) * m3 / np.abs(m2) ** 1.5
    return _lined_up(np.where(flat, np.nan, g1), x.shape[-1])


def _pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation along the last axis, two-pass; NaN when either side has no spread."""
    da = a - a.mean(axis=-1, keepdims=True)
    db = b - b.mean(axis=-1, keepdims=True)
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
