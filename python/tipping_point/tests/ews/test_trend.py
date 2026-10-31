"""Kendall's tau against scipy, and the pipeline's score and s(t) against their definitions."""

import numpy as np
import pytest
from scipy.stats import kendalltau
from tipping_point.ews import indicators as ind
from tipping_point.ews import trend
from tipping_point.ews.trend import Pipeline


def _scipy(rows: np.ndarray) -> np.ndarray:
    return np.array([kendalltau(np.arange(rows.shape[1]), r)[0] for r in rows])


@pytest.mark.parametrize("n", [2, 3, 7, 64, 65, 513, 1000, 16385])
def test_kendall_tau_matches_scipy(n: int) -> None:
    """Catches a miscounted merge level, padding that pairs with real values, or a length that
    is not a power of two going wrong (65, 513 pad almost 2x; 64 pads nothing); 16385 pads to
    2^15, past what 16-bit ranks hold."""
    rows = np.cumsum(np.random.default_rng(n).standard_normal((40 if n < 10_000 else 2, n)), axis=1)
    assert np.allclose(trend.kendall_tau(rows), _scipy(rows), rtol=0, atol=1e-14)


def test_kendall_tau_b_with_ties_matches_scipy() -> None:
    """Catches ties counted as discordant, or tau-a where scipy's tau-b corrects for ties."""
    rows = np.round(np.cumsum(np.random.default_rng(7).standard_normal((50, 300)), axis=1) / 4)
    assert np.allclose(trend.kendall_tau(rows), _scipy(rows), rtol=0, atol=1e-14)


def test_kendall_tau_extremes_and_undefined_rows() -> None:
    """A rising row is +1, a falling one -1; a constant row, a NaN or an inf has no tau."""
    rows = np.array([np.arange(9.0), -np.arange(9.0), np.ones(9), np.arange(9.0), np.arange(9.0)])
    rows[3, 4], rows[4, 2] = np.nan, np.inf
    tau = trend.kendall_tau(rows)
    assert tau[0] == 1.0 and tau[1] == -1.0
    assert np.isnan(tau[2:]).all()
    assert trend.kendall_tau(np.zeros((2, 3, 5))).shape == (2, 3)


def test_kendall_tau_is_the_same_in_any_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows are independent: splitting the batch across threads must not change a digit."""
    rows = np.random.default_rng(8).standard_normal((300, 200))
    whole = trend.kendall_tau(rows)
    monkeypatch.setattr(trend, "CHUNK", 1000)
    assert np.array_equal(trend.kendall_tau(rows), whole)


def test_pipeline_scores_the_trailing_segment_of_the_indicator() -> None:
    """The score is Kendall's tau of the indicator's last `length` values, after detrending the
    whole input; it reads nothing before the span it needs."""
    x = np.random.default_rng(9).standard_normal((4, 260)).cumsum(axis=1)
    p = Pipeline("variance", window=40, length=150, bandwidth=30.0)
    residuals = ind.gaussian_detrend(x, 30.0)
    expected = trend.kendall_tau(ind.rolling_variance(residuals[:, -p.span :], 40)[:, 39:])
    assert np.array_equal(p.trend(residuals), expected)
    assert np.array_equal(p(x), expected)
    early = residuals.copy()
    early[:, : 260 - p.span] = 1e6  # outside the span: must not matter
    assert np.array_equal(p.trend(early), expected)


def test_trailing_score_detrends_each_slice_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """s(t) uses only samples up to t: changing the future leaves every earlier score alone, and
    batching the slices to bound memory changes nothing."""
    x = np.random.default_rng(10).standard_normal((3, 400)).cumsum(axis=1)
    p = Pipeline("ac", window=30, length=60, bandwidth=20.0)
    times, scores = trend.trailing_score(x, p, every=7)
    assert times[0] == p.span - 1 and np.all(np.diff(times) == 7) and times[-1] < 400
    for k in (0, 5, len(times) - 1):
        t = times[k]
        assert np.array_equal(scores[:, k], p(x[:, t - p.span + 1 : t + 1]))
    future = x.copy()
    future[:, 300:] += 100.0
    _, changed = trend.trailing_score(future, p, every=7)
    before = times < 300
    assert np.array_equal(changed[:, before], scores[:, before])
    monkeypatch.setattr(trend, "SLICES", 500)  # one evaluation time per batch
    assert np.array_equal(trend.trailing_score(x, p, every=7)[1], scores)


def _ar1(phi: float, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(shape)
    x = np.zeros(shape)
    for t in range(1, shape[1]):
        x[:, t] = phi * x[:, t - 1] + noise[:, t]
    return x


@pytest.mark.parametrize(("rise", "sign"), [(0.02, 1.0), (-0.02, -1.0)])
def test_level_takes_the_sign_of_a_trend_through_ar1(rise: float, sign: float) -> None:
    """AR(1) on a line: the rolling mean follows the line, so the score has the line's sign in every
    row and sits near +-1, where the same noise without the trend scores on both sides. Run through
    a detrend, the rolling mean of residuals has no trend to find, which is why level skips it."""
    window, length = 50, 400
    rng = np.random.default_rng(14)
    noise = _ar1(0.5, (8, window + length - 1), rng)
    x = noise + rise * np.arange(noise.shape[1])
    score = Pipeline("level", window, length, bandwidth=None)(x)
    assert np.all(sign * score > 0.8), score
    alone = Pipeline("level", window, length, bandwidth=None)(noise)
    assert np.abs(alone).mean() < 0.5, alone
    residuals = ind.gaussian_detrend(x, 30.0)
    detrended = trend.kendall_tau(ind.rolling_mean(residuals[:, -length:], window)[:, window - 1 :])
    assert np.abs(detrended).mean() < 0.5, detrended


def test_slope_takes_the_sign_of_a_steepening_trend_through_ar1() -> None:
    """A parabola through AR(1): a rising slope scores near +1, a falling slope near -1."""
    window, length = 80, 400
    rng = np.random.default_rng(15)
    noise = _ar1(0.5, (8, window + length - 1), rng)
    t = np.arange(noise.shape[1])
    for curve, sign in ((1e-3 * t**2, 1.0), (-1e-3 * t**2, -1.0)):
        score = Pipeline("slope", window, length, bandwidth=None)(noise + curve)
        assert np.all(sign * score > 0.8), score


@pytest.mark.parametrize("name", sorted(trend.RAW))
def test_a_raw_indicator_reads_the_series_whatever_the_bandwidth(name: str) -> None:
    """The bypass is the indicator's: score, rolling values and residuals (what a surrogate test
    draws from) are the raw series', with or without a bandwidth, and a detrended indicator's are
    not."""
    x = np.random.default_rng(16).standard_normal((3, 300)).cumsum(axis=1)
    raw, given = (Pipeline(name, 40, 100, bandwidth=b) for b in (None, 25.0))
    assert np.array_equal(given(x), raw(x))
    assert np.array_equal(given.rolling(x), raw.rolling(x), equal_nan=True)
    assert np.array_equal(given.residuals(x), x)
    detrended = Pipeline("variance", 40, 100, bandwidth=25.0)
    assert not np.array_equal(detrended.residuals(x), x)
    rolling = {"level": ind.rolling_mean, "slope": ind.rolling_slope}[name]
    assert np.array_equal(given.rolling(x), rolling(x, 40), equal_nan=True)


def test_pipeline_refuses_unknown_indicators_and_short_inputs() -> None:
    with pytest.raises(ValueError):
        Pipeline("kurtosis", window=10, length=10)
    with pytest.raises(ValueError):
        Pipeline("variance", window=10, length=10).trend(np.zeros((2, 18)))
