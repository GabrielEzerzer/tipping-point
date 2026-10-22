"""Indicators against ewstools 2.1.3's own outputs, scipy's Gaussian filter, the AR(1) closed
forms and a direct per-window computation.

AR(1), x_t = phi x_(t-1) + sigma e_t, has Var = sigma^2 / (1 - phi^2), lag-k correlation phi^k
and zero skewness, so the rolling indicators must find those on a long stationary series.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import lfilter
from tipping_point.ews import indicators as ind
from tipping_point.ews import trend

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "ews_golden.json").read_text("utf-8")
)
ROLLING = {
    "variance": lambda x, case: ind.rolling_variance(x, case["window"]),
    "autocorrelation": lambda x, case: ind.rolling_autocorrelation(x, case["window"], case["lag"]),
    "skewness": lambda x, case: ind.rolling_skewness(x, case["window"]),
}


def _array(values: list) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values])


def _ar1(phi: float, replicas: int, n: int, seed: int, sigma: float = 1.0) -> np.ndarray:
    """Stationary AR(1) rows: x_0 is drawn from the stationary distribution."""
    noise = sigma * np.random.default_rng(seed).standard_normal((replicas, n))
    noise[:, 0] /= np.sqrt(1 - phi**2)
    series = lfilter([1.0], [1.0, -phi], noise, axis=1)
    assert isinstance(series, np.ndarray)
    return series


@pytest.mark.parametrize("name", sorted(GOLDEN["cases"]))
def test_matches_the_ewstools_golden_fixture(name: str) -> None:
    """Catches any drift from ewstools' definitions: kernel width, reflection, ddof, the pair
    convention of autocorr, the skewness bias correction, NaN alignment, Kendall's variant."""
    case = GOLDEN["cases"][name]
    x = _array(case["x"])
    residuals = x
    if case["bandwidth"] is not None:
        smooth = ind.gaussian_smooth(x, case["bandwidth"])
        assert np.allclose(smooth, _array(case["smoothing"]), rtol=1e-12, atol=1e-12)
        residuals = ind.gaussian_detrend(x, case["bandwidth"])
        assert np.allclose(residuals, _array(case["residuals"]), rtol=0, atol=1e-12)
    for key, rolling in ROLLING.items():
        ours, theirs = rolling(residuals, case), _array(case[key])
        assert np.array_equal(np.isnan(ours), np.isnan(theirs)), key
        assert np.nanmax(np.abs(ours - theirs)) <= 1e-12, key
        tau = trend.kendall_tau(ours[case["window"] - 1 :])
        assert tau == pytest.approx(case["ktau"][key], abs=1e-12), key


@pytest.mark.parametrize("bandwidth", [0.2, 3.0, 25.0, 140.0, 2000.0])
def test_fft_smoothing_matches_scipy_reflect(bandwidth: float) -> None:
    """Catches a kernel radius off by one, a wrapped FFT, or a reflection that is not scipy's
    half-sample one; 0.2 is a radius-0 kernel and 2000 reflects several times past both ends."""
    x = np.cumsum(np.random.default_rng(1).standard_normal((3, 301)), axis=1) + 40.0
    sigma = ind.SIGMA_PER_BANDWIDTH * bandwidth
    expected = gaussian_filter1d(x, sigma, axis=-1, mode="reflect")
    assert np.allclose(ind.gaussian_smooth(x, bandwidth), expected, rtol=0, atol=1e-11)
    assert np.allclose(  # ewstools calls the n-dimensional filter on one series
        ind.gaussian_smooth(x[0], bandwidth), gaussian_filter(x[0], sigma, mode="reflect")
    )


def test_ar1_closed_forms() -> None:
    """Catches a wrong variance normalisation or a lag misaligned inside the window: the window
    means must find sigma^2/(1-phi^2), phi and phi^3, and zero skewness."""
    phi, sigma, window = 0.9, 0.5, 2500
    x = _ar1(phi, replicas=32, n=10_000, seed=2, sigma=sigma)
    var = np.nanmean(ind.rolling_variance(x, window))
    assert var == pytest.approx(sigma**2 / (1 - phi**2), rel=0.05)
    assert np.nanmean(ind.rolling_autocorrelation(x, window, 1)) == pytest.approx(phi, abs=0.01)
    assert np.nanmean(ind.rolling_autocorrelation(x, window, 3)) == pytest.approx(phi**3, abs=0.02)
    assert np.nanmean(ind.rolling_skewness(x, window)) == pytest.approx(0.0, abs=0.03)


def test_rolling_sums_keep_their_digits_on_long_offset_series() -> None:
    """Catches the cancellation running sums suffer on a large mean: each window must equal a
    direct two-pass computation (n = 5,000, window 1,000, as in the research benchmark)."""
    window, lag = 1000, 2
    x = _ar1(0.95, replicas=1, n=5000, seed=3)[0] + 1e3 + np.linspace(0, 50, 5000)
    view = np.lib.stride_tricks.sliding_window_view(x, window, axis=-1)
    a, b = view[..., :-lag], view[..., lag:]
    da = a - a.mean(axis=-1, keepdims=True)
    db = b - b.mean(axis=-1, keepdims=True)
    ac = (da * db).sum(-1) / np.sqrt((da * da).sum(-1) * (db * db).sum(-1))
    d = view - view.mean(axis=-1, keepdims=True)
    m2, m3 = (d**2).mean(-1), (d**3).mean(-1)
    skew = np.sqrt(window * (window - 1)) / (window - 2) * m3 / m2**1.5
    valid = slice(window - 1, None)
    assert np.allclose(ind.rolling_variance(x, window)[valid], view.var(-1, ddof=1), rtol=1e-11)
    assert np.allclose(ind.rolling_autocorrelation(x, window, lag)[valid], ac, atol=1e-11)
    assert np.allclose(ind.rolling_skewness(x, window)[valid], skew, atol=1e-10)


def test_flat_windows_have_zero_variance_and_no_correlation_or_skewness() -> None:
    """A frozen stretch (|m| = 1 in an ordered lattice) must not produce rounding garbage."""
    x = np.random.default_rng(4).standard_normal(400)
    x[150:300] = 0.7
    var = ind.rolling_variance(x, 50)
    assert np.all(var[199:300] == 0.0) and np.all(var[100:149] > 0)
    assert np.all(np.isnan(ind.rolling_autocorrelation(x, 50)[199:300]))
    assert np.all(np.isnan(ind.rolling_skewness(x, 50)[199:300]))
    assert np.all(ind.rolling_variance(np.full(80, 3.0), 10)[9:] == 0.0)


def test_burst_pairs_never_cross_bursts() -> None:
    """Within each burst the series climbs by 1 a frame; between bursts it jumps by 200. Every
    within-burst pair is (a, a + 1), so pooling only those gives exactly 1; one pair across a
    jump would pull it down, as the plain rolling correlation shows."""
    offsets = 100.0 * (-1.0) ** np.arange(30)[:, None]
    bursts = np.tile(np.arange(8.0), (30, 1)) + offsets
    assert ind.burst_autocorrelation(bursts, 1) == pytest.approx(1.0, abs=1e-12)
    assert ind.rolling_autocorrelation(bursts.ravel(), 240)[-1] < 0.9


def test_burst_autocorrelation_of_ar1_is_phi() -> None:
    """Bursts of 8 consecutive samples cut from AR(1) chains, far apart: lag-1 pairs give phi."""
    phi = 0.8
    chains = _ar1(phi, replicas=4, n=200_000, seed=6)
    bursts = chains.reshape(4, -1, 40)[:, :, :8]  # keep 8 of every 40 samples
    r = ind.burst_autocorrelation(bursts, 1)
    assert r.shape == (4,)
    assert np.allclose(r, phi, atol=0.02)


def test_windows_outside_the_series_are_refused() -> None:
    x = np.zeros(10)
    with pytest.raises(ValueError):
        ind.rolling_variance(x, 11)
    with pytest.raises(ValueError):
        ind.rolling_autocorrelation(x, 3, lag=2)  # one pair is no correlation
    with pytest.raises(ValueError):
        ind.rolling_skewness(x, 2)
    with pytest.raises(ValueError):
        ind.gaussian_smooth(x, 0.0)
