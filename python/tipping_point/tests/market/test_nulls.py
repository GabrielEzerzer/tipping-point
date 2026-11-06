"""The market's nulls (#29 S3; research pass section 4, "Circularity, dynamic", section 2 item 12).

The GARCH simulator against its closed forms; 1-minute OHLC from it, through the real bars; the
delay simulation by brute force; segments and slices that never see the onset; the shift null; and
the floor end to end with the real labeller, the real variance and AC1 Pipelines and scoring.auc.

The floor's parameters here are stated synthetic ones, not a fit: a minute GARCH of 60% a year
with a half-life of 19 hours, bridged to 10 seconds. The pilot year's fit is
harness/market_garch.py's (test_garch_fit.py). The expected ranges come from eleven seeds of three
paths each (variance at 6 bars 0.61-0.73, AC1 0.35-0.69, shift medians 0.46-0.56 for the trends and
0.44-0.54 for the level control), widened to hold any seed rather than this one.
"""

import math
from datetime import date
from typing import Any

import numpy as np
import pytest
from tipping_point.ews import scoring
from tipping_point.ews.trend import Pipeline
from tipping_point.market import archive, bars, msvol, nulls, reports
from tipping_point.market.__main__ import CONFIRMATORY as CLI_CONFIRMATORY
from tipping_point.market.nulls import Garch

MODERATE = Garch(5e-9, 0.05, 0.90, 10.0)
PERSISTENT = Garch(5e-10, 0.01, 0.985, 10.0)
STEPS = 1_000_000
FLOOR_GARCH = Garch(4.1e-10, 0.012, 0.9874, 6.0)
PARAMS = msvol.Params(warmup_days=28, baseline_days=14)
HORIZONS = (0, 3, 6, 12, 24)
SMALL = msvol.Params(warmup_days=14, baseline_days=7, max_iter_warm=10)
PLANTED = msvol.Theta(-0.3, 0.9, 0.4, 0.5, 0.985, 0.95)
START_MS = archive.day_ms(nulls.FIRST)


def _closed_forms(g: Garch) -> tuple[float, float]:
    """The kurtosis of r and the lag-1 autocorrelation of r^2, from E[h_t^2] = E[(omega + alpha
    r_t-1^2 + beta h_t-1)^2] and E[r_t^2 r_t-1^2] = E[h_t r_t-1^2]."""
    kappa = 3 * (g.nu - 2) / (g.nu - 4)  # the kurtosis of a Student-t
    s = g.alpha + g.beta
    h2 = (1 - s**2) / (1 - s**2 - (kappa - 1) * g.alpha**2)  # E[h^2] / sigma^4
    kurtosis = kappa * h2
    cross = (1 - s) + g.alpha * kurtosis + g.beta * h2  # E[r_t^2 r_t-1^2] / sigma^4
    return kurtosis, (cross - 1) / (kurtosis - 1)


def _acf(x: np.ndarray, lag: int) -> float:
    d = x - x.mean()
    return float((d[:-lag] * d[lag:]).sum() / (d * d).sum())


@pytest.mark.parametrize("garch", [MODERATE, PERSISTENT], ids=["moderate", "persistent"])
def test_the_long_run_variance_matches_its_closed_form(garch: Garch) -> None:
    """Within five standard errors of the sample variance, by the closed-form kurtosis and
    autocorrelation of r^2: 1.8% for the moderate process, 2.6% for the persistent one."""
    kurtosis, rho = _closed_forms(garch)
    s = garch.alpha + garch.beta
    standard_error = math.sqrt((kurtosis - 1) * (1 + 2 * rho / (1 - s)) / STEPS)
    assert 5 * standard_error < 0.03
    r = nulls.simulate(garch, STEPS, np.random.default_rng(29))
    assert garch.variance == pytest.approx(garch.omega / (1 - s))
    assert abs(r.var() / garch.variance - 1) < 5 * standard_error
    assert abs(r.mean()) < 5 * math.sqrt(garch.variance / STEPS)


def test_the_tails_and_the_clustering_match_their_closed_forms() -> None:
    r = nulls.simulate(MODERATE, STEPS, np.random.default_rng(30))
    kurtosis, rho = _closed_forms(MODERATE)
    s = MODERATE.alpha + MODERATE.beta
    assert kurtosis > 4  # heavier than a normal's 3 and than the t's own (4 at nu = 10)
    assert float((r**4).mean() / (r**2).mean() ** 2) == pytest.approx(kurtosis, rel=0.05)
    assert _acf(r * r, 1) == pytest.approx(rho, abs=0.005)
    assert _acf(r * r, 5) == pytest.approx(rho * s**4, abs=0.005)
    assert abs(_acf(r, 1)) < 0.005  # clusters in size, not in sign


def test_the_recursion_is_the_stated_one_and_the_shocks_have_unit_variance() -> None:
    g = MODERATE
    n = 1000
    r = nulls.simulate(g, n, np.random.default_rng(3))
    shocks = np.random.default_rng(3).standard_t(g.nu, n)
    h = g.variance
    for t in range(n):
        assert r[t] == pytest.approx(math.sqrt(h) * shocks[t] * math.sqrt((g.nu - 2) / g.nu))
        h = g.omega + g.alpha * r[t] ** 2 + g.beta * h


@pytest.mark.parametrize(
    "kwargs, says",
    [
        ({"omega": 0.0, "alpha": 0.1, "beta": 0.8, "nu": 5.0}, "omega"),
        ({"omega": 1e-9, "alpha": -0.1, "beta": 0.8, "nu": 5.0}, "alpha, beta"),
        ({"omega": 1e-9, "alpha": 0.2, "beta": 0.8, "nu": 5.0}, "alpha \\+ beta"),
        ({"omega": 1e-9, "alpha": 0.1, "beta": 0.8, "nu": 2.0}, "nu"),
    ],
)
def test_a_process_without_a_finite_variance_is_refused(
    kwargs: dict[str, float], says: str
) -> None:
    with pytest.raises(ValueError, match=says):
        Garch(**kwargs)


def test_the_simulator_repeats_under_its_seed() -> None:
    a = nulls.simulate(FLOOR_GARCH, 5000, np.random.default_rng(1))
    b = nulls.simulate(FLOOR_GARCH, 5000, np.random.default_rng(1))
    c = nulls.simulate(FLOOR_GARCH, 5000, np.random.default_rng(2))
    assert a.tobytes() == b.tobytes() and a.tobytes() != c.tobytes()


def test_klines_are_consistent_ohlc_of_the_10_second_path() -> None:
    days = 1
    r = nulls.simulate(MODERATE, days * nulls.DAY_STEPS, np.random.default_rng(4))
    k = nulls.minute_klines(START_MS, r, price=100.0)
    minutes = 1440 * days
    assert k["open"][0] == 100.0
    assert np.array_equal(k["open"][1:], k["close"][:-1])
    assert np.all(k["high"] >= np.maximum(k["open"], k["close"]))
    assert np.all(k["low"] <= np.minimum(k["open"], k["close"]))
    assert np.array_equal(k["open_time"], START_MS + 60_000 * np.arange(minutes))
    # the high and low are the path's own, over the open and six closes of the minute
    path = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    assert k["high"][7] == pytest.approx(path[42:49].max())
    assert k["low"][7] == pytest.approx(path[42:49].min())
    returns = nulls.close_returns(k)
    assert np.allclose(returns, r.reshape(-1, 6).sum(axis=1), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="whole minutes"):
        nulls.minute_klines(START_MS, r[:-1])


def test_the_bars_read_the_klines_as_they_read_the_archive() -> None:
    days = 7
    r = nulls.simulate(MODERATE, days * nulls.DAY_STEPS, np.random.default_rng(5))
    k = nulls.minute_klines(START_MS, r)
    bar, checks = bars.bars(START_MS, days, k, nulls._NO_METRICS, nulls._NO_FUNDING)
    assert checks["bars_without_rv"] == 0 and checks["bars_incomplete"] == 0
    assert np.array_equal(bar["close"], k["close"][4::5])
    per_bar = 5 * nulls.STEPS_PER_MINUTE * MODERATE.variance
    # A range read off seven points misses much of a continuous path's, so Garman-Klass reads well
    # under the variance here. Nothing downstream sees the level: z is relative to its own past.
    assert 0.35 < bar["rv"].mean() / per_bar < 0.55
    i = 123
    one = bars.GK * np.log(k["close"][5 * i : 5 * i + 5] / k["open"][5 * i : 5 * i + 5]) ** 2
    two = 0.5 * np.log(k["high"][5 * i : 5 * i + 5] / k["low"][5 * i : 5 * i + 5]) ** 2
    assert bar["rv"][i] == pytest.approx((two - one).sum())


# ---------------------------------------------------------------- the detection delay


def test_delays_match_brute_force_through_the_filter_and_the_hysteresis() -> None:
    n = 20_000
    lengths, delay = nulls.delays(PLANTED, n, np.random.default_rng(6), 0.8, 0.2)
    rng = np.random.default_rng(6)
    states = nulls._chain(PLANTED, n, rng)
    noise = rng.normal(size=n)
    z = np.where(states, PLANTED.mu1 + PLANTED.sigma1 * noise, PLANTED.mu0 + PLANTED.sigma0 * noise)
    e0, e1, _ = msvol._emissions(z, PLANTED)
    h = msvol.hysteresis(msvol._forward(e0, e1, PLANTED), 0.8, 0.2)
    want_lengths, want_delay = [], []
    t = 1
    while t < n:
        if states[t] and not states[t - 1]:
            end = t
            while end < n and states[end]:
                end += 1
            if end < n:
                want_lengths.append(end - t)
                risen = [i - t for i in range(t, end) if h[i] == 1]
                want_delay.append(risen[0] if risen else -1)
            t = end
        t += 1
    assert lengths.tolist() == want_lengths and delay.tolist() == want_delay
    assert len(want_delay) > 100


def test_the_delays_repeat_under_their_seed() -> None:
    a = nulls.delays(PLANTED, 5000, np.random.default_rng(1))
    b = nulls.delays(PLANTED, 5000, np.random.default_rng(1))
    c = nulls.delays(PLANTED, 5000, np.random.default_rng(2))
    assert [x.tobytes() for x in a] == [x.tobytes() for x in b]
    assert [x.tobytes() for x in a] != [x.tobytes() for x in c]


def test_the_chain_holds_its_planted_persistence() -> None:
    states = nulls._chain(PLANTED, 400_000, np.random.default_rng(7))
    assert states.mean() == pytest.approx(0.015 / (0.015 + 0.05), abs=0.01)
    starts, ends = msvol.runs(states)
    assert (ends - starts).mean() == pytest.approx(1 / 0.05, rel=0.05)


def test_the_delay_is_what_the_research_pass_measured_and_grows_with_the_threshold() -> None:
    """Section 2, item 12: at the planted persistence, median 1 and p90 3 bars crossing 0.5, 3.9%
    of episodes missed. The pass did not state its means and spreads, so this is a neighbourhood."""
    n = 400_000

    def read(theta: msvol.Theta, enter: float, leave: float) -> tuple[float, float, float]:
        _, delay = nulls.delays(theta, n, np.random.default_rng(8), enter, leave)
        risen = delay[delay >= 0]
        return float(np.median(risen)), float(np.percentile(risen, 90)), float(np.mean(delay < 0))

    median, p90, missed = read(PLANTED, 0.5, 0.5)
    assert median <= 1 and p90 <= 3 and 0.02 < missed < 0.06
    at_entry = read(PLANTED, 0.8, 0.2)
    assert at_entry[0] >= median and at_entry[1] >= p90 and at_entry[2] > missed
    sharper = PLANTED._replace(mu1=2.1)
    assert read(sharper, 0.8, 0.2)[1] <= at_entry[1]


def test_delay_rows_split_off_the_episodes_that_can_become_events() -> None:
    params = msvol.Params()
    every, longer = nulls.delay_rows(PLANTED, params, 200_000, np.random.default_rng(9))
    lengths, delay = nulls.delays(
        PLANTED, 200_000, np.random.default_rng(9), params.enter, params.exit
    )
    risen = delay[(lengths >= params.d_min) & (delay >= 0)]
    assert longer["median"] == np.median(risen)
    assert longer["p90"] == np.percentile(risen, 90, method="higher") == int(longer["p90"])
    assert longer["max"] == risen.max() and longer["episodes"] == np.sum(lengths >= params.d_min)
    assert (every["population"], longer["population"]) == ("all", "at_least_d_min")
    assert every["episodes"] > longer["episodes"] > 100
    assert longer["missed"] == 0 < every["missed"]  # 12 bars at this separation always rise
    assert every["missed_share"] == pytest.approx(every["missed"] / every["episodes"])
    assert 0 <= longer["median"] <= longer["p90"] <= longer["max"]


# ---------------------------------------------------------------- segments and slices


def test_a_segment_ends_just_before_its_horizon_and_never_sees_the_onset() -> None:
    minutes = np.arange(2000.0)
    rows = nulls.pre_onset_rows(minutes, np.array([100, 300]), 2, 60)
    assert rows.tolist() == [
        list(np.arange(430.0, 490.0)),  # onset bar 100 is minute 500; 2 bars before it, minute 490
        list(np.arange(1430.0, 1490.0)),
    ]
    assert nulls.pre_onset_rows(minutes, np.array([13]), 2, 60).shape == (0, 60)  # 55 minutes back
    assert nulls.pre_onset_rows(minutes, np.array([14]), 2, 60).shape == (1, 60)  # 60 back


@pytest.mark.parametrize("horizon", [0, 6])
def test_no_score_moves_when_the_onset_bar_and_all_after_it_are_rewritten(horizon: int) -> None:
    rng = np.random.default_rng(10)
    minutes = rng.normal(size=5000)
    onsets = np.array([300, 700])
    scorer = Pipeline("variance", 60, 240)
    before = scorer(nulls.pre_onset_rows(minutes, onsets, horizon, scorer.span))
    later = minutes.copy()
    later[5 * (onsets[0] - horizon) : 5 * onsets[0] + 5] = 99.0  # up to the first onset's bar
    first = scorer(nulls.pre_onset_rows(later, onsets[:1], horizon, scorer.span))
    assert first[0] == before[0]
    wider = minutes.copy()
    wider[5 * (onsets[0] - horizon) - 1] = 99.0  # the segment's own last minute does move it
    assert scorer(nulls.pre_onset_rows(wider, onsets[:1], horizon, scorer.span))[0] != before[0]


def test_null_slices_are_non_overlapping_spans_inside_their_stretch() -> None:
    minutes = np.arange(1000.0)
    stretches = [{"start": 10, "bars": 30}, {"start": 100, "bars": 11}, {"start": 120, "bars": 12}]
    rows = nulls.null_rows(minutes, stretches, 60)
    # 150 minutes hold two spans, 55 none, 60 exactly one
    assert rows[:, 0].tolist() == [50.0, 110.0, 600.0]
    assert rows[:, -1].tolist() == [109.0, 169.0, 659.0]
    assert nulls.null_rows(minutes, [], 60).shape == (0, 60)


def test_eligible_bars_by_hand() -> None:
    params = msvol.Params(d_min=3, clean=4, refractory=10)
    h = np.zeros(40, np.int8)
    h[:5] = -1  # labels start at bar 5, so history is known from bar 15
    h[20:22] = 1  # bars 21-25 have these inside their clean window
    h[30] = 1  # bars 31-34 have this one
    want = [*range(15, 21), *range(26, 31), *range(35, 40)]
    assert nulls.eligible_onsets(h, params).tolist() == want
    assert nulls.eligible_onsets(np.full(40, -1, np.int8), params).size == 0


@pytest.mark.parametrize("clean, refractory", [(4, 10), (12, 4)])
def test_eligible_bars_are_exactly_where_the_event_rule_would_place_an_event(
    clean: int, refractory: int
) -> None:
    """Plant a run of d_min bars at each labelled bar in turn and let msvol.episodes class it. The
    flickers bar clean windows and start no refractory period, which the mask leaves out."""
    params = msvol.Params(d_min=3, clean=clean, refractory=refractory)
    h = np.zeros(70, np.int8)
    h[:5] = -1
    h[20:22] = 1
    h[30] = 1
    h[45:47] = 1
    missing = np.zeros(len(h), bool)
    eligible = set(nulls.eligible_onsets(h, params).tolist())
    assert eligible
    for bar in range(5, len(h) - params.d_min):
        trial = h.copy()
        trial[bar : bar + params.d_min] = 1
        kinds = {e["onset"]: e["kind"] for e in msvol.episodes(trial, missing, params)}
        assert (kinds.get(bar) == "event") == (bar in eligible), bar


def test_every_real_event_onset_is_eligible(planted: Any) -> None:
    onsets = [e["onset"] for e in planted.full.episodes if e["kind"] == "event"]
    assert len(onsets) >= 20
    assert np.isin(onsets, nulls.eligible_onsets(planted.full.h, planted.params)).all()


def test_every_shifted_onset_lands_on_an_eligible_bar_at_least_a_day_away() -> None:
    day = 288
    eligible = np.concatenate([np.arange(1000, 3000), np.arange(5000, 7000)])
    onsets = np.array([1000, 2500, 6000])
    moved, kept = nulls.shift_onsets(onsets, eligible, day, 500, np.random.default_rng(11))
    assert moved.shape == (500, 3) and kept.all()
    assert np.isin(moved, eligible).all()
    assert np.all(np.abs(moved - onsets) >= day)
    assert np.unique(moved[:, 2]).size > 300  # spread, not a fixed offset
    assert np.all(np.ptp(moved - onsets, axis=1) > 0)  # in each draw, each onset has its own offset
    again, _ = nulls.shift_onsets(onsets, eligible, day, 500, np.random.default_rng(11))
    assert again.tobytes() == moved.tobytes()


def test_a_shift_draws_uniformly_from_the_bars_it_may_use() -> None:
    eligible = np.arange(0, 4000, 2)  # 2,000 bars
    wide, _ = nulls.shift_onsets(np.array([0]), eligible, 288, 40_000, np.random.default_rng(12))
    far = eligible[eligible >= 288]
    deciles = np.histogram(wide, bins=10, range=(288, 4000))[0]
    assert deciles.sum() == 40_000 and far.size == 2000 - 144
    assert deciles.min() > 0.9 * 4000 and deciles.max() < 1.1 * 4000


def test_an_onset_with_no_eligible_bar_is_left_out_not_drawn_from_elsewhere() -> None:
    eligible = np.array([100, 150, 200])
    rng = np.random.default_rng(13)
    # 150 has all three within a day; 1000 none; 388 only 100, a day away exactly
    moved, kept = nulls.shift_onsets(np.array([150, 1000, 388]), eligible, 288, 50, rng)
    assert kept.tolist() == [False, True, True]
    assert moved.shape == (50, 2)
    assert set(moved[:, 0].tolist()) == {100, 150, 200}
    assert set(moved[:, 1].tolist()) == {100}
    moved, kept = nulls.shift_onsets(np.array([5]), np.empty(0, np.int64), 288, 3, rng)
    assert moved.shape == (3, 0) and kept.tolist() == [False]
    moved, kept = nulls.shift_onsets(np.empty(0, np.int64), eligible, 288, 3, rng)
    assert moved.shape == (3, 0) and kept.size == 0


# ---------------------------------------------------------------- the floor


class Level:
    """A control with no trend in it: the variance of a segment's last half hour. Volatility near
    an onset is high, so right before one it separates; two hours back much less."""

    span = 299

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[:, -30:].var(axis=1)


@pytest.fixture(scope="module")
def floor() -> nulls.Floor:
    scorers = {
        "variance": Pipeline("variance", 60, 240),
        "ac": Pipeline("ac", 60, 240),
        "level": Level(),
    }
    return nulls.circularity_floor(
        FLOOR_GARCH, scorers, scoring.auc, HORIZONS, days=90, paths=3, seed=29, params=PARAMS
    )


def _cell(found: nulls.Floor, scorer: str, horizon: int) -> dict[str, Any]:
    (cell,) = [c for c in found.cells if (c["scorer"], c["horizon_bars"]) == (scorer, horizon)]
    return cell


def test_the_floor_runs_end_to_end_on_events_and_null_slices(floor: nulls.Floor) -> None:
    assert len(floor.paths) == 3 and len(floor.cells) == 3 * len(HORIZONS)
    for path in floor.paths:
        assert path["bars"] == 90 * 288 and path["labelled_bars"] == 62 * 288
        assert path["refits"] == 9 and path["events"] >= 5 and path["null_bars"] >= 299 // 5
    assert len({(p["events"], p["null_bars"]) for p in floor.paths}) == 3  # each its own draws
    n_events = sum(p["events"] for p in floor.paths)
    for cell in floor.cells:
        assert cell["n_pos"] == n_events and cell["n_neg"] >= 50
        assert cell["shift_draws"] == nulls.SHIFT_DRAWS
        key = cell["scorer"], cell["horizon_bars"]
        own = [c for c in floor.path_cells if (c["scorer"], c["horizon_bars"]) == key]
        assert [c["path"] for c in own] == [0, 1, 2]
        assert sum(c["n_pos"] for c in own) == cell["n_pos"]
        assert sum(c["n_neg"] for c in own) == cell["n_neg"]
    assert len(floor.path_cells) == 3 * len(floor.cells)


class Recording:
    """A scorer that keeps the rows it is handed."""

    def __init__(self, span: int = 299) -> None:
        self.span = span
        self.calls: list[np.ndarray] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls.append(x.copy())
        return x[:, -1]


def _rebuilt(seed: int, days: int, params: msvol.Params) -> tuple[np.ndarray, msvol.Labels, Any]:
    """A floor's first path rebuilt from its seed by the documented steps: its 1-minute
    close-to-close returns, its labels and its event onsets."""
    rng = np.random.default_rng([seed, 0])
    returns = nulls.simulate(FLOOR_GARCH, days * nulls.DAY_MINUTES, rng, nulls.STEPS_PER_MINUTE)
    klines = nulls.minute_klines(START_MS, returns)
    bar, _ = bars.bars(START_MS, days, klines, nulls._NO_METRICS, nulls._NO_FUNDING)
    out = msvol.label(bar["ts_ns"], bar["rv"], params)
    minutes = np.diff(np.log(klines["close"]), prepend=np.log(klines["open"][0]))
    onsets = np.array([e["onset"] for e in out.episodes if e["kind"] == "event"], np.int64)
    return minutes, out, onsets


def _moved_bars(minutes: np.ndarray, rows: np.ndarray, horizon: int) -> np.ndarray:
    """The onset bar each segment was cut for, read from the last return it holds."""
    ends = np.array([int(np.flatnonzero(minutes == row[-1])[0]) + 1 for row in rows])
    assert np.all(ends % 5 == 0)
    return ends // 5 + horizon


def test_the_worst_known_delay_sets_the_contaminated_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any path can set the bound, and one with no delay known neither sets nor hides it."""
    p90s = iter([math.nan, 5.0, 7.0])

    def fake(*_: Any) -> list[dict[str, float]]:
        return [{"p90": math.nan}, {"p90": next(p90s)}]

    monkeypatch.setattr(nulls, "delay_rows", fake)
    found = nulls.circularity_floor(
        FLOOR_GARCH,
        {"variance": Pipeline("variance", 60, 240)},
        scoring.auc,
        (6, 7),
        days=21,
        paths=3,
        seed=1,
        params=SMALL,
        draws=1,
    )
    assert [c["delay_p90"] for c in found.cells] == [7.0, 7.0]
    assert [c["contaminated"] for c in found.cells] == [True, False]


def test_the_floor_scores_the_labellers_own_segments_of_close_to_close_returns() -> None:
    """The rows the scorer was handed are the null slices of the labeller's stretches and the
    segments before its events, of 1-minute close-to-close returns, and each shifted segment is cut
    for a bar a day from its event, with the same clean window a real onset has."""
    seed, days, h, draws = 8, 35, 6, 20  # this path holds events and null slices; not all do
    recording = Recording()
    found = nulls.circularity_floor(
        FLOOR_GARCH,
        {"recording": recording},
        scoring.auc,
        (h,),
        days=days,
        paths=1,
        seed=seed,
        params=SMALL,
        draws=draws,
    )
    minutes, out, onsets = _rebuilt(seed, days, SMALL)
    assert onsets.size == found.paths[0]["events"] > 0 and out.nulls
    assert found.paths[0]["unshifted_events"] == 0
    assert found.cells[0]["n_pos"] == onsets.size
    null_slices, segments, shifted = recording.calls
    assert null_slices.tobytes() == nulls.null_rows(minutes, out.nulls, 299).tobytes()
    assert len(null_slices) == found.cells[0]["n_neg"] > 0
    assert segments.tobytes() == nulls.pre_onset_rows(minutes, onsets, h, 299).tobytes()
    assert shifted.shape == (draws * onsets.size, 299)
    moved = _moved_bars(minutes, shifted, h)
    assert np.all(np.abs(moved.reshape(draws, -1) - onsets) >= 288)
    for bar in moved.tolist():
        assert np.all(out.h[bar - SMALL.clean : bar] == 0) and out.h[bar] >= 0
        assert np.all(out.h[bar - SMALL.refractory : bar] >= 0)


def test_each_paths_auc_reads_its_own_segments_against_its_own_slices() -> None:
    """The floor's own spread (#111): per path and cell, the AUC of that path's pre-onset segments
    against that path's null slices alone, which the pooled AUC mixes over every path."""
    seed, days, horizons, paths = 8, 35, (0, 6), 2  # both paths hold events and null slices
    recording = Recording()
    found = nulls.circularity_floor(
        FLOOR_GARCH,
        {"recording": recording},
        scoring.auc,
        horizons,
        days=days,
        paths=paths,
        seed=seed,
        params=SMALL,
        draws=3,
    )
    per_path = 1 + 2 * len(horizons)  # the null slices, then each horizon's segments and shifts
    assert len(recording.calls) == paths * per_path
    want = []
    for path in range(paths):
        null_slices, *cut = recording.calls[path * per_path : (path + 1) * per_path]
        for h, segments in zip(horizons, cut[::2], strict=True):
            assert len(segments) and len(null_slices)
            score = scoring.auc(segments[:, -1], null_slices[:, -1])
            want.append((path, "recording", h, score, len(segments), len(null_slices)))
    keys = ("path", "scorer", "horizon_bars", "auc", "n_pos", "n_neg")
    assert [tuple(c[k] for k in keys) for c in found.path_cells] == want
    assert want[0][3] != want[len(horizons)][3]  # the paths differ: the spread is there to see
    for cell in found.cells:
        own = [c for c in found.path_cells if c["horizon_bars"] == cell["horizon_bars"]]
        assert sum(c["n_pos"] for c in own) == cell["n_pos"]
    report = reports.floor(found, FLOOR_GARCH, {})
    table = report.tables["path_cells"]
    assert table.schema.equals(reports.FLOOR["path_cells"])
    assert table.to_pylist() == found.path_cells
    assert "## AUC per path: the floor's own spread" in report.summary


def test_an_event_that_cannot_move_a_day_is_counted_and_the_rest_stay_on_eligible_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the bars just before the first event are eligible here. That event has none a day away;
    the rest, a refractory period on at least, can only move there."""
    seed, days, h, draws = 8, 35, 6, 20
    minutes, _, onsets = _rebuilt(seed, days, SMALL)
    near = np.arange(onsets[0] - 200, onsets[0])
    monkeypatch.setattr(nulls, "eligible_onsets", lambda *_: near)
    recording = Recording()
    found = nulls.circularity_floor(
        FLOOR_GARCH,
        {"recording": recording},
        scoring.auc,
        (h,),
        days=days,
        paths=1,
        seed=seed,
        params=SMALL,
        draws=draws,
    )
    (path,), (cell,) = found.paths, found.cells
    assert path["events"] == onsets.size > 1 and path["unshifted_events"] == 1
    assert cell["n_pos"] == onsets.size and cell["shift_draws"] == draws
    shifted = recording.calls[-1]
    assert shifted.shape == (draws * (onsets.size - 1), 299)
    assert np.isin(_moved_bars(minutes, shifted, h), near).all()
    report = reports.floor(found, FLOOR_GARCH, {})
    assert report.tables["paths"]["unshifted_events"].to_pylist() == [1]
    assert f"1 of the {onsets.size} events had no eligible bar a day away" in report.summary


def test_a_path_too_short_to_move_any_event_a_day_says_so_and_shifts_nothing() -> None:
    """Fifteen days with fourteen of warm-up label one day, so no bar lies a day from any event."""
    params = msvol.Params(
        warmup_days=14, baseline_days=7, max_iter_warm=10, clean=12, refractory=24, d_min=6
    )
    recording = Recording()
    found = nulls.circularity_floor(
        FLOOR_GARCH,
        {"recording": recording},
        scoring.auc,
        (0,),
        days=15,
        paths=1,
        seed=1,  # this path holds events; not all do
        params=params,
        draws=20,
    )
    (path,), (cell,) = found.paths, found.cells
    assert path["labelled_bars"] == 288 and path["events"] == path["unshifted_events"] > 0
    assert cell["n_pos"] == path["events"]
    assert cell["shift_draws"] == 0 and math.isnan(cell["shift_auc_median"])
    assert sum(len(rows) for rows in recording.calls) == cell["n_pos"] + cell["n_neg"]
    report = reports.floor(found, FLOOR_GARCH, {})
    assert report.tables["cells"]["shift_auc_median"].null_count == 1
    assert report.tables["paths"]["unshifted_events"].to_pylist() == [path["events"]]
    n = path["events"]
    assert f"{n} of the {n} events had no eligible bar a day away" in report.summary


def test_a_shifted_segment_always_has_the_history_its_span_needs() -> None:
    """A span longer than the warm-up: bars too early to hold one are not drawn at all."""
    seed, days, h, draws, span = 1, 35, 6, 5, 25_000
    recording = Recording(span)
    found = nulls.circularity_floor(
        FLOOR_GARCH,
        {"recording": recording},
        scoring.auc,
        (h,),
        days=days,
        paths=1,
        seed=seed,
        params=SMALL,
        draws=draws,
    )
    minutes, _, onsets = _rebuilt(seed, days, SMALL)
    assert onsets[0] < span // 5 + h  # the first label comes too early to hold a whole span
    (path,) = found.paths
    shifted = recording.calls[-1]
    assert shifted.shape == (draws * (onsets.size - path["unshifted_events"]), span)
    assert np.all(_moved_bars(minutes, shifted, h) >= span // 5 + h)


def test_the_variance_floor_sits_above_chance_and_falls_with_the_horizon(
    floor: nulls.Floor,
) -> None:
    """What the floor is for: a clustering process with no critical dynamics hands the variance
    pipeline this AUC by construction. Over eleven seeds its mean ran from 0.75 at no horizon to
    0.62 at two hours."""
    assert 0.60 <= _cell(floor, "variance", 0)["auc"] <= 0.85
    assert 0.55 <= _cell(floor, "variance", 6)["auc"] <= 0.80
    assert 0.42 <= _cell(floor, "variance", 24)["auc"] <= 0.72
    assert _cell(floor, "variance", 0)["auc"] > _cell(floor, "variance", 24)["auc"]


def test_the_ac1_floor_sits_at_chance(floor: nulls.Floor) -> None:
    """GARCH returns have no autocorrelation to trend: 0.35-0.69 over eleven seeds, any horizon, the
    minute returns' t tails making AC1 noisy, and about 0.50 on average."""
    for h in HORIZONS:
        assert 0.30 <= _cell(floor, "ac", h)["auc"] <= 0.74


def test_the_shift_null_sits_at_chance_for_every_scorer(floor: nulls.Floor) -> None:
    """A shifted segment sits where a real one could, inside the clean window, so only its alignment
    differs. That holds for a scorer of the level too: it read 0.73 when segments could land
    anywhere, and now runs 0.44-0.47 on this fixture. Over eleven seeds every median lay in
    0.43-0.56, about 0.50 on average."""
    for cell in floor.cells:
        assert cell["shift_auc_lo"] <= cell["shift_auc_median"] <= cell["shift_auc_hi"]
        assert 0.38 <= cell["shift_auc_median"] <= 0.64


def test_the_segments_are_aligned_with_the_onsets(floor: nulls.Floor) -> None:
    """The level control: high right before an onset, at chance shifted, and fading going back."""
    near, far = _cell(floor, "level", 0), _cell(floor, "level", 24)
    assert near["auc"] > 0.75 and near["auc"] > near["shift_auc_hi"]
    assert far["auc"] < near["auc"] - 0.1


def test_a_horizon_below_the_detection_delay_is_contaminated(floor: nulls.Floor) -> None:
    """Under these fits the p90 delay ran 5 or 6 bars, as the floor's own delay simulation says."""
    worst = max(p["delay_p90"] for p in floor.paths)
    assert 2 <= worst < 12
    for cell in floor.cells:
        assert cell["delay_p90"] == worst
        assert cell["contaminated"] == (cell["horizon_bars"] < worst)
    assert _cell(floor, "variance", 0)["contaminated"]
    assert not _cell(floor, "variance", 24)["contaminated"]


def _small_floor(seed: int) -> nulls.Floor:
    scorers = {"variance": Pipeline("variance", 60, 240)}
    return nulls.circularity_floor(
        FLOOR_GARCH, scorers, scoring.auc, (6,), days=35, paths=1, seed=seed, params=SMALL, draws=20
    )


def test_the_floor_and_its_report_repeat_under_a_seed(tmp_path: Any) -> None:
    a, b = _small_floor(8), _small_floor(8)
    assert repr(a) == repr(b)
    assert repr(_small_floor(9)) != repr(a)
    assert a.cells[0]["n_pos"] > 0 and a.cells[0]["n_neg"] > 0  # something to repeat
    written = [
        reports.write(reports.floor(found, FLOOR_GARCH, {}), tmp_path / name)
        for found, name in ((a, "a"), (b, "b"))
    ]
    names = sorted(p.name for p in written[0].iterdir())
    assert names == [
        "cells.parquet",
        "garch.parquet",
        "path_cells.parquet",
        "paths.parquet",
        "summary.md",
    ]
    for name in names:
        assert (written[0] / name).read_bytes() == (written[1] / name).read_bytes()
    summary = (written[0] / "summary.md").read_text(encoding="utf-8")
    assert "alpha = 0.012" in summary and "beta = 0.9874" in summary
    assert "no eligible bar" not in summary  # every event here could move


def test_a_process_with_no_clustering_gives_no_labels_and_no_crash() -> None:
    """Independent shocks: the guard refuses every fit, so no bar is labelled and every cell says
    so. A floor with nothing to report must not read as a floor of 0.5."""
    scorers = {"variance": Pipeline("variance", 60, 240)}
    params = msvol.Params(warmup_days=14, baseline_days=7, max_iter=20, max_iter_warm=5)
    found = nulls.circularity_floor(
        Garch(1e-7, 0.0, 0.0, 6.0),
        scorers,
        scoring.auc,
        (6,),
        days=28,
        paths=1,
        seed=1,
        params=params,
        draws=5,
    )
    (path,), (cell,) = found.paths, found.cells
    assert path["labelled_bars"] == 0 and path["flagged_refits"] == path["refits"] > 0
    assert path["events"] == path["unshifted_events"] == 0 and math.isnan(path["delay_p90"])
    assert math.isnan(cell["auc"]) and cell["n_pos"] == cell["n_neg"] == 0
    assert cell["contaminated"] and cell["shift_draws"] == 0
    report = reports.floor(found, Garch(1e-7, 0.0, 0.0, 6.0), {})
    assert report.tables["cells"]["auc"].null_count == 1


def test_a_cell_with_no_positives_has_no_auc_whatever_the_auc_function_returns() -> None:
    """The empty floor of a short half-life (#138): null slices but no pre-onset segment. A
    horizon longer than the path leaves every event without one, as a path with no events would.
    The AUC passed in answers 0.5 even then, so only the floor's own rule keeps the cell empty,
    and the report says how many are."""
    seed, days, far = 8, 35, 35 * 288  # this path holds events and null slices
    found = nulls.circularity_floor(
        FLOOR_GARCH,
        {"recording": Recording()},
        lambda *_: 0.5,
        (0, far),
        days=days,
        paths=1,
        seed=seed,
        params=SMALL,
        draws=3,
    )
    (path,) = found.paths
    near, empty = found.cells
    assert path["events"] > 0 and near["n_pos"] == path["events"] and near["auc"] == 0.5
    assert empty["n_pos"] == 0 and empty["n_neg"] == near["n_neg"] > 0
    assert math.isnan(empty["auc"]) and math.isnan(found.path_cells[1]["auc"])
    assert found.path_cells[0]["auc"] == 0.5
    # the far horizon leaves no bar to shift to, so neither horizon has a shift null either
    assert near["shift_draws"] == empty["shift_draws"] == 0
    report = reports.floor(found, FLOOR_GARCH, {})
    assert report.tables["cells"]["auc"].to_pylist() == [0.5, None]
    paragraph = report.summary.split("\n\n")[1]
    assert (
        "1 of the 2 cells and 1 of the 2 AUCs per path have no pre-onset segment or no null "
        "slice, so their AUC is n/a." in paragraph
    )
    assert "no pre-onset segment" not in reports.floor(_small_floor(8), FLOOR_GARCH, {}).summary


def test_the_floor_stays_out_of_the_confirmatory_span() -> None:
    assert nulls.CONFIRMATORY == CLI_CONFIRMATORY == date(2024, 1, 1)
    scorers = {"variance": Pipeline("variance", 60, 240)}
    with pytest.raises(ValueError, match="market-labels-v1"):
        nulls.circularity_floor(
            FLOOR_GARCH,
            scorers,
            scoring.auc,
            (6,),
            days=3,
            paths=1,
            seed=1,
            first=date(2023, 12, 30),
        )
