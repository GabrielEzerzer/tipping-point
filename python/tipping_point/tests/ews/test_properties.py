"""Property tests on the EWS core: each rolling indicator against its definition on every window,
Gaussian smoothing and Kendall's tau against scipy, AUC and the alarm threshold against brute
force, the scaled-temperature inverse pair, the power-law fit and the phase surrogates' spectrum."""

import numpy as np
import scipy.ndimage
import scipy.stats
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.lib.stride_tricks import sliding_window_view
from tipping_point.ews import fss, indicators, scoring, surrogates, trend

EIGHTHS = st.one_of(st.integers(-2, 2), st.integers(-800, 800)).map(lambda k: k / 8)
OFFSETS = st.sampled_from([0.0, 1e6])


def series(min_size: int = 3, max_size: int = 60) -> st.SearchStrategy[np.ndarray]:
    return arrays(np.float64, st.integers(min_size, max_size), elements=EIGHTHS)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@given(series(2), OFFSETS, st.data())
def test_rolling_variance_is_each_windows_sample_variance(x, offset, data):
    window = data.draw(st.integers(2, len(x)))
    got = indicators.rolling_variance(x + offset, window)
    want = np.var(sliding_window_view(x, window), axis=-1, ddof=1)
    assert np.isnan(got[: window - 1]).all()
    np.testing.assert_allclose(got[window - 1 :], want, rtol=1e-7, atol=1e-9)


@given(series(4), OFFSETS, st.data())
def test_rolling_autocorrelation_is_each_windows_lagged_pearson(x, offset, data):
    lag = data.draw(st.integers(1, len(x) - 3))
    window = data.draw(st.integers(lag + 2, len(x)))
    got = indicators.rolling_autocorrelation(x + offset, window, lag)
    want = [pearson(w[:-lag], w[lag:]) for w in sliding_window_view(x, window)]
    assert np.isnan(got[: window - 1]).all()
    np.testing.assert_allclose(got[window - 1 :], want, rtol=1e-7, atol=1e-9)


@given(series(3), OFFSETS, st.data())
def test_rolling_skewness_is_each_windows_corrected_skewness(x, offset, data):
    window = data.draw(st.integers(3, len(x)))
    got = indicators.rolling_skewness(x + offset, window)
    windows = sliding_window_view(x, window)
    flat = np.ptp(windows, axis=-1) == 0
    want = np.full(len(windows), np.nan)
    want[~flat] = scipy.stats.skew(windows[~flat], axis=-1, bias=False)
    assert np.isnan(got[: window - 1]).all()
    np.testing.assert_allclose(got[window - 1 :], want, rtol=1e-6, atol=1e-7)


@given(series(1, 80), st.floats(0.5, 60.0))
def test_gaussian_smooth_is_scipys_reflected_gaussian_filter(x, bandwidth):
    sigma = indicators.SIGMA_PER_BANDWIDTH * bandwidth
    want = scipy.ndimage.gaussian_filter1d(x, sigma, mode="reflect", truncate=indicators.TRUNCATE)
    np.testing.assert_allclose(indicators.gaussian_smooth(x, bandwidth), want, atol=1e-9)


@given(
    st.integers(2, 300).flatmap(
        lambda n: arrays(
            np.float64, (3, n), elements=st.integers(-3, 3) | st.integers(-(10**6), 10**6)
        )
    )
)
def test_kendall_tau_is_scipys_tau_b_against_time(rows):
    time = np.arange(rows.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        want = np.array([scipy.stats.kendalltau(time, row)[0] for row in rows])
    np.testing.assert_allclose(trend.kendall_tau(rows), want, rtol=1e-12, atol=1e-12)


SCORES = arrays(
    np.float64, st.integers(0, 12), elements=st.integers(-4, 4).map(float) | st.just(np.nan)
)


@given(SCORES, SCORES)
def test_auc_is_the_pairwise_win_rate_and_swapping_classes_complements_it(positives, negatives):
    pos, neg = positives[~np.isnan(positives)], negatives[~np.isnan(negatives)]
    got = scoring.auc(positives, negatives)
    if len(pos) == 0 or len(neg) == 0:
        assert np.isnan(got)
        return
    wins = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    np.testing.assert_allclose(got, np.mean(wins), rtol=1e-12)
    np.testing.assert_allclose(got + scoring.auc(negatives, positives), 1.0, rtol=1e-12)


@given(
    arrays(
        np.float64,
        st.tuples(st.integers(1, 3), st.integers(1, 15)),
        elements=st.integers(-5, 5).map(float) | st.just(np.nan),
    ),
    st.floats(0.0, 3.0),
    st.floats(1.0, 20.0),
)
def test_threshold_at_far_is_the_lowest_theta_that_keeps_every_higher_one_under_far(
    scores, far, duration
):
    found = scoring.threshold_at_far(scores, far, duration)
    finite = scores[np.isfinite(scores)]
    candidates = np.unique(np.concatenate([[-np.inf], finite]))
    counts = np.array([scoring.alarms(scores, c).sum() for c in candidates])
    assert found.far_max == counts.max() / duration
    if not found.attainable:
        assert np.isnan(found.theta)
        assert (counts <= far * duration).all()
        return
    assert (counts[candidates >= found.theta] <= far * duration).all()
    assert counts[candidates < found.theta][-1] > far * duration


@given(st.floats(0.5, 5.0), st.integers(4, 256))
def test_temperature_inverts_scaled_temperature(T, L):
    np.testing.assert_allclose(fss.temperature(fss.scaled_temperature(T, L), L), T, rtol=1e-12)


@given(
    st.lists(st.integers(4, 512), min_size=3, max_size=6, unique=True),
    st.floats(-3.0, 3.0),
    st.floats(0.01, 100.0),
    st.booleans(),
)
def test_power_law_recovers_the_exponent_of_an_exact_power_law(sizes, kappa, scale, correction):
    values = scale * np.asarray(sizes, dtype=np.float64) ** kappa
    np.testing.assert_allclose(fss.power_law(sizes, values, correction), kappa, atol=1e-8)


@given(series(2, 64), st.integers(0, 2**32 - 1))
def test_phase_surrogates_keep_the_amplitude_spectrum(x, seed):
    assume(np.ptp(x) > 0)
    drawn = surrogates.phase_randomised(x, 3, np.random.default_rng(seed))
    want = np.abs(np.fft.rfft(x))
    got = np.abs(np.fft.rfft(drawn, axis=-1))
    np.testing.assert_allclose(got, np.broadcast_to(want, got.shape), atol=1e-8 * want.max())
