"""The labeller's parts (#29 S2; research pass section 2 items 3-11, section 4's checks): EM
recovers planted parameters from a poor start, label matching from swapped starts, the degeneracy
guard, missing bars, the hysteresis and episode rules on hand-checked paths, null stretches, the
refit calendar, and planted episodes recovered as events."""

import math
from datetime import date
from typing import Any

import numpy as np
import pytest
from market_planted import hmm, regimes, rv_series
from tipping_point.market import msvol
from tipping_point.market.msvol import Params, Theta

PLANTED = Theta(-0.3, 0.9, 0.4, 0.5, 0.985, 0.95)
FIRST = date(2022, 7, 4)  # a Monday
SHORT = Params(warmup_days=28, baseline_days=14)  # seven weeks hold a warm-up and three refits


def test_em_recovers_planted_parameters_from_a_poor_start() -> None:
    z, _ = hmm(PLANTED, 8000, np.random.default_rng(1))
    poor = msvol.fit(z, Theta(-0.05, 0.05, 1.5, 1.5, 0.6, 0.6), 1e-10, 1000)
    near = msvol.fit(z, PLANTED, 1e-10, 1000)
    first = msvol.fit(z, msvol.start(z, Params()), 1e-10, 1000)
    for result in (poor, near, first):
        assert result.converged and result.failure is None
        assert np.allclose(result.theta, near.theta, rtol=0, atol=1e-4)  # one optimum
    tolerance = (0.03, 0.03, 0.03, 0.03, 0.01, 0.03)  # about five standard errors at 8,000 bars
    for got, want, tol in zip(poor.theta, PLANTED, tolerance, strict=True):
        assert abs(got - want) < tol
    assert abs(poor.shares[1] - 0.23) < 0.05  # the planted chain spends 3/13 of its time high


def test_a_fit_from_swapped_states_ends_with_state_1_high() -> None:
    z, _ = hmm(PLANTED, 4000, np.random.default_rng(2))
    swapped = Theta(PLANTED.mu1, PLANTED.mu0, PLANTED.sigma1, PLANTED.sigma0, 0.95, 0.985)
    a = msvol.fit(z, swapped, 1e-10, 500)
    b = msvol.fit(z, PLANTED, 1e-10, 500)
    assert a.theta.mu1 > a.theta.mu0 and a.theta.sigma1 > a.theta.sigma0
    assert np.allclose(a.theta, b.theta, rtol=0, atol=1e-6)


def test_the_guard_names_each_degeneracy() -> None:
    good = msvol.Fit(PLANTED, -1.0, 5, True, (0.8, 0.2), None)
    assert msvol.guard(good, Params()) is None
    sigma = msvol.guard(good._replace(theta=PLANTED._replace(sigma1=0.04)), Params())
    assert sigma == "sigma 0.04 < 0.05"
    share = msvol.guard(good._replace(shares=(0.99, 0.01)), Params())
    assert share == "smoothed share 0.01 < 0.02"
    gap = msvol.guard(good._replace(theta=PLANTED._replace(mu1=-0.1)), Params())
    assert gap is not None and gap.startswith("mu1 - mu0 0.2 < 0.25")
    failed = good._replace(failure="EM failed numerically")
    assert msvol.guard(failed, Params()) == "EM failed numerically"


def test_a_state_that_collapses_onto_a_few_bars_is_caught() -> None:
    rng = np.random.default_rng(3)
    z = np.concatenate([rng.normal(0.0, 0.5, 3000), np.full(40, 4.0)])
    result = msvol.fit(z, Theta(0.0, 4.0, 0.5, 0.5, 0.95, 0.9), 1e-6, 200)
    assert msvol.guard(result, Params()) is not None


def test_a_degenerate_refit_is_flagged_and_the_theta_before_stays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ts, rv, _ = rv_series(FIRST, 49, np.random.default_rng(5))
    real, starts = msvol.fit, []

    def fit(z: np.ndarray, theta: Theta, tol: float, max_iter: int) -> msvol.Fit:
        starts.append(theta)
        result = real(z, theta, tol, max_iter)
        if len(starts) == 2:  # the second refit comes back degenerate
            return result._replace(theta=result.theta._replace(sigma1=0.01))
        return result

    monkeypatch.setattr(msvol, "fit", fit)
    out = msvol.label(ts, rv, SHORT)
    first, second, third = out.refits
    assert not first["flagged"] and not third["flagged"]
    assert second["flagged"] and second["reason"] == "sigma 0.01 < 0.05"
    assert second["sigma1"] == 0.01  # the refit row records what the fit gave
    r1, _, r3 = (r["bar"] for r in out.refits)
    assert np.all(out.refit[r1:r3] == 0) and np.all(out.refit[r3:] == 2)
    assert np.all(out.refit[:r1] == -1) and np.all(np.isnan(out.p[:r1]))
    assert starts[2] == Theta(
        **{name: first[name] for name in Theta._fields}
    )  # warm from θ in force


def test_a_run_of_missing_bars_neither_alarms_nor_crashes() -> None:
    rng = np.random.default_rng(6)
    states = regimes(49 * 288, rng)
    gap = slice(35 * 288, 37 * 288)  # two whole days, as when a day's file is missing
    states[gap.start - 300 : gap.stop + 300] = 0
    ts, rv, _ = rv_series(FIRST, 49, rng, states)
    rv[gap] = np.nan
    out = msvol.label(ts, rv, SHORT)
    assert out.refits and not any(r["flagged"] for r in out.refits)
    assert np.all(out.h[gap] == 0) and np.all(np.isfinite(out.p[gap]))
    assert np.all(np.isnan(out.x[gap])) and np.all(np.isnan(out.z[gap]))
    for e in out.episodes:
        assert e["onset"] >= gap.stop or (e["end"] is not None and e["end"] <= gap.start)
    r = out.refits[out.refit[gap.stop - 1]]
    theta = Theta(**{name: r[name] for name in Theta._fields})
    stationary = (1 - theta.p00) / (2 - theta.p00 - theta.p11)
    assert abs(out.p[gap.stop - 1] - stationary) < 1e-3  # P relaxes to the chain's own share
    nothing = msvol.label(ts, np.full(len(ts), np.nan), SHORT)  # no data at all
    assert all(r["flagged"] for r in nothing.refits) and np.all(nothing.h == -1)
    assert nothing.episodes == [] and nothing.nulls == []


def test_hysteresis_enters_at_0p8_and_leaves_at_0p2() -> None:
    p = np.array([math.nan, math.nan, 0.5, 0.8, 0.5, 0.21, 0.2, 0.5, 0.79, 0.95, 0.19])
    assert msvol.hysteresis(p, 0.8, 0.2).tolist() == [-1, -1, 0, 1, 1, 1, 0, 0, 0, 1, 0]


def _path(n: int, runs: list[tuple[int, int]], unlabelled: int = 0) -> np.ndarray:
    h = np.zeros(n, np.int8)
    h[:unlabelled] = -1
    for a, b in runs:
        h[a:b] = 1
    return h


def test_episodes_are_classed_by_length_clean_window_and_refractory_period() -> None:
    params = Params(d_min=3, clean=4, refractory=10)
    runs = [(5, 9), (14, 16), (18, 21), (28, 33), (38, 41), (50, 53), (60, 62), (69, 72)]
    h = _path(72, runs, unlabelled=2)
    missing = np.zeros(72, bool)
    missing[29] = True
    found = [
        (e["onset"], e["end"], e["bars"], e["kind"], e["confirm"], e["missing"])
        for e in msvol.episodes(h, missing, params)
    ]
    assert found == [
        (5, 9, 4, "episode", None, 0),  # long enough, but its history is unknown
        (14, 16, 2, "flicker", None, 0),
        (18, 21, 3, "episode", None, 0),  # the flicker sits in its clean window
        (28, 33, 5, "event", 31, 1),
        (38, 41, 3, "episode", None, 0),  # 10 bars after an event onset: refractory
        (50, 53, 3, "event", 53, 0),
        (60, 62, 2, "flicker", None, 0),
        (69, None, 3, "event", 72, 0),  # still open at the end, already confirmed
    ]
    tail = msvol.episodes(_path(22, [(20, 22)]), np.zeros(22, bool), params)
    assert [e["kind"] for e in tail] == ["pending"]


def test_null_stretches_keep_clear_of_episodes_flickers_and_unknown_history() -> None:
    params = Params(d_min=3, clean=4, refractory=10)
    missing = np.zeros(100, bool)
    missing[[50, 51]] = True

    def stretches(h: np.ndarray) -> list[tuple[int, int, int, int]]:
        found = msvol.episodes(h, missing, params)
        assert found[0]["kind"] == "event" and found[1]["kind"] == "flicker"
        return [
            (s["start"], s["end"], s["bars"], s["missing"])
            for s in msvol.nulls(h, missing, found, params)
        ]

    # an event at 30-34 clears [20, 45); a flicker at 60-61 clears [56, 66); the first and last
    # 10 bars are unknown history
    h = _path(100, [(30, 35), (60, 62)])
    assert stretches(h) == [(10, 20, 10, 0), (45, 56, 11, 2), (66, 90, 24, 0)]
    h = _path(100, [(30, 35), (60, 62), (98, 100)])  # a pending run counts as an episode
    assert stretches(h) == [(10, 20, 10, 0), (45, 56, 11, 2), (66, 88, 22, 0)]


def test_refits_fall_after_the_warm_up_then_on_mondays() -> None:
    first = 1_656_633_600 * 10**9  # 2022-07-01T00:00:00Z, a Friday
    ts = first + msvol.BAR_NS * np.arange(31 * 288, dtype=np.int64)
    # the warm-up ends on Sunday 07-03; then Mondays 07-04, 07-11, 07-18 and 07-25
    assert msvol.refit_bars(ts, Params(warmup_days=2)) == [d * 288 for d in (2, 3, 10, 17, 24)]
    assert msvol.refit_bars(ts, Params(warmup_days=29, refit_days=1)) == [29 * 288, 30 * 288]
    assert msvol.refit_bars(ts, Params(warmup_days=40)) == []


def test_bars_off_the_grid_or_thresholds_out_of_order_are_refused() -> None:
    ts, rv, _ = rv_series(FIRST, 2, np.random.default_rng(7))
    with pytest.raises(ValueError, match="gapless"):
        msvol.label(np.delete(ts, 100), np.delete(rv, 100))
    with pytest.raises(ValueError, match="gapless"):
        msvol.label(ts[1:], rv[1:])
    with pytest.raises(ValueError, match="exit"):
        msvol.label(ts, rv, Params(enter=0.3, exit=0.6))


def test_planted_episodes_come_out_as_events(planted: Any) -> None:
    full, params = planted.full, planted.params
    first = int(np.flatnonzero(full.h >= 0)[0])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], planted.states, [0]]).astype(np.int8)))
    eligible = [
        int(a)
        for a, b in zip(edges[0::2], edges[1::2], strict=True)
        if b - a >= params.d_min and a >= first + params.refractory
    ]
    events = [e["onset"] for e in full.episodes if e["kind"] == "event"]
    delays = [[e - a for e in events if -2 <= e - a <= params.d_min] for a in eligible]
    recalled = [d[0] for d in delays if d]
    matched = [e for e in events if any(-2 <= e - a <= params.d_min for a in eligible)]
    assert len(eligible) >= 20
    assert len(recalled) >= 0.9 * len(eligible) and len(matched) >= 0.9 * len(events)
    assert np.median(recalled) <= 2 and max(recalled) <= 6  # the filter's onset delay, in bars
    assert not any(r["flagged"] for r in full.refits)
