"""Scoring functions against brute-force definitions."""

import itertools

import numpy as np
import pytest
from tipping_point.ews import scoring
from tipping_point.ews.trend import Pipeline


def _brute_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pairs = [(p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg)]
    return float(np.mean(pairs))


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_the_pair_count_with_ties(seed: int) -> None:
    """Catches ties counted as 0 or 1 instead of 1/2, or ranks off by one."""
    rng = np.random.default_rng(seed)
    pos, neg = rng.integers(0, 6, 37).astype(float), rng.integers(0, 5, 23).astype(float)
    assert scoring.auc(pos, neg) == pytest.approx(_brute_auc(pos, neg), abs=1e-12)


def test_auc_extremes_and_missing_scores() -> None:
    assert scoring.auc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert scoring.auc([1.0, 2.0], [3.0, 4.0]) == 0.0
    assert scoring.auc([1.0, 2.0], [1.0, 2.0]) == 0.5
    assert scoring.auc([np.nan, 3.0], [1.0, np.nan]) == 1.0  # undefined scores are dropped
    assert np.isnan(scoring.auc([np.nan], [1.0]))


def test_surrogate_pvalue_is_one_plus_count_over_one_plus_n() -> None:
    """Never 0; ties and undefined surrogates count against the observed score."""
    surrogates = np.array([[0.1, 0.2, 0.5, np.nan], [0.1, 0.2, 0.3, 0.4], [0.9, 0.9, 0.9, 0.9]])
    p = scoring.surrogate_pvalue(np.array([0.5, 0.95, 0.1]), surrogates)
    assert p.tolist() == [3 / 5, 1 / 5, 5 / 5]
    assert np.isnan(scoring.surrogate_pvalue(np.array([np.nan]), np.zeros((1, 3))))


def test_a_surrogate_identical_to_its_row_never_rejects() -> None:
    """The identity revert check in miniature: the row and its copies score in one batch, so
    every copy ties the observed score and p = 1 exactly."""
    x = np.random.default_rng(1).standard_normal((6, 300)).cumsum(axis=1)
    p = Pipeline("variance", window=40, length=200, bandwidth=30.0)
    copies = lambda r, n, rng: np.repeat(r[:, None, :], n, axis=1)  # noqa: E731
    observed, pvalues = scoring.surrogate_test(x, p, copies, 19, np.random.default_rng(2))
    assert np.array_equal(observed, p(x))
    assert np.all(pvalues == 1.0)


def test_alarms_count_rises_above_the_threshold() -> None:
    s = np.array([[0.1, 0.6, 0.7, 0.2, 0.9, np.nan, 0.8], [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    assert scoring.alarms(s, 0.5).tolist() == [3, 1]  # after a gap, the count starts afresh
    assert scoring.false_alarm_rate(s, 0.5, duration=8.0) == 0.5


def test_threshold_at_far_is_the_lowest_safe_threshold() -> None:
    """Brute force over every candidate: the chosen theta and all above it are within budget,
    and the candidate just below it is not."""
    rng = np.random.default_rng(3)
    s = rng.standard_normal((20, 60)).cumsum(axis=1) / 10
    s[3, 10:15] = np.nan
    duration, far = 1200.0, 0.02
    theta = scoring.threshold_at_far(s, far, duration)
    candidates = np.unique(s[~np.isnan(s)])
    rate = {c: scoring.false_alarm_rate(s, c, duration) for c in candidates}
    assert all(rate[c] <= far for c in candidates if c >= theta)
    below = candidates[candidates < theta]
    assert len(below) and rate[below.max()] > far


def test_first_alarm_and_lead_times() -> None:
    times = np.array([10.0, 20.0, 30.0, 40.0])
    s = np.array([[0.0, 0.6, 0.2, 0.9], [0.0, 0.1, 0.2, 0.3]])
    alarm = scoring.first_alarm(times, s, 0.5)
    assert alarm[0] == 20.0 and np.isnan(alarm[1])  # the second row is a miss
    lead = scoring.lead_times(35.0, alarm)
    assert lead[0] == 15.0 and np.isnan(lead[1])


def test_hierarchical_bootstrap_resamples_seeds_then_replicas() -> None:
    """Two seeds whose replicas are all 0 and all 10: a resample's mean is 0, 5 or 10 with
    probabilities 1/4, 1/2, 1/4. A flat bootstrap over replicas would centre tightly on 5."""
    seeds = [np.zeros(50), np.full(50, 10.0)]
    means = scoring.hierarchical_bootstrap(np.mean, [seeds], 4000, np.random.default_rng(4))
    values, counts = np.unique(means, return_counts=True)
    assert values.tolist() == [0.0, 5.0, 10.0]
    assert counts / 4000 == pytest.approx([0.25, 0.5, 0.25], abs=0.03)
    one = scoring.hierarchical_bootstrap(
        np.mean, [[np.array([0.0, 1.0])]], 2000, np.random.default_rng(5)
    )
    assert set(np.unique(one).tolist()) == {0.0, 0.5, 1.0}  # replicas resampled within the seed


def test_bootstrap_takes_several_samples_for_two_sample_statistics() -> None:
    pos = [np.array([0.9, 0.8]), np.array([0.7, 0.95])]
    neg = [np.array([0.1, 0.2, 0.3])]
    draws = scoring.hierarchical_bootstrap(scoring.auc, [pos, neg], 50, np.random.default_rng(6))
    assert np.all(draws == 1.0)
    assert scoring.percentile_interval(np.arange(101.0)) == pytest.approx((2.5, 97.5))
