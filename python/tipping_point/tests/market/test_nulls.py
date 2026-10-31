"""The market's nulls (#29 S3; research pass section 4, "Circularity, dynamic", section 2 item 12).

The GARCH simulator against its closed forms; 1-minute OHLC from it, through the real bars; the
delay simulation by brute force; segments and slices that never see the onset; the shift null; and
the floor end to end with the real labeller, the real variance and AC1 Pipelines and scoring.auc.

The floor's parameters here are stated synthetic ones, not a fit: 60% a year in 10-second variance,
a half-life of 19 hours. S5 fits the 2023 pilot year. Their expected ranges come from ten seeds of
three paths each (variance at 6 bars 0.61-0.71, AC1 0.45-0.57, shift medians 0.46-0.55), widened
to hold any seed rather than this one.
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
FLOOR_GARCH = Garch(1.14e-11, 0.005, 0.9949, 6.0)
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


def test_every_shifted_onset_moves_at_least_a_day_and_stays_in_range() -> None:
    low, high, day = 1000, 11_000, 288
    onsets = np.array([1000, 1100, 6000, 10_999])
    moved = nulls.shift_onsets(onsets, low, high, day, 500, np.random.default_rng(11))
    assert moved.shape == (500, 4)
    assert np.all((moved >= low) & (moved < high))
    assert np.all(np.abs(moved - onsets) >= day)
    assert np.unique(moved[:, 2]).size > 400  # spread, not a fixed offset
    offsets = (moved - onsets) % (high - low)
    assert np.all(np.ptp(offsets, axis=1) > 0)  # in each draw, each onset has its own offset
    wide = nulls.shift_onsets(onsets[:1], low, high, day, 20_000, np.random.default_rng(12))
    deciles = np.histogram(wide, bins=10, range=(low, high))[0]
    assert deciles[1:9].min() > 1800  # about 2,100 each
    # the offset goes around the range, so an onset at its start bars the day at either end
    assert max(deciles[0], deciles[9]) < deciles[1:9].min()
    again = nulls.shift_onsets(onsets, low, high, day, 500, np.random.default_rng(11))
    assert again.tobytes() == moved.tobytes()


def test_a_shift_with_no_room_is_refused() -> None:
    rng = np.random.default_rng(13)
    with pytest.raises(ValueError, match="shift range"):
        nulls.shift_onsets(np.array([5]), 10, 2000, 288, 3, rng)
    with pytest.raises(ValueError, match="either way"):
        nulls.shift_onsets(np.array([15]), 10, 500, 288, 3, rng)
    assert nulls.shift_onsets(np.empty(0, np.int64), 10, 2000, 288, 3, rng).shape == (3, 0)


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


class Recording:
    """A scorer that keeps the rows it is handed."""

    span = 299

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls.append(x.copy())
        return x[:, -1]


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
    """Rebuild the path from its seed by the documented steps: the rows the scorer was handed are
    the null slices of the labeller's stretches and the segments before its events, of 1-minute
    close-to-close returns, and the shifted segments lie in the bars the shift was given."""
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
    rng = np.random.default_rng([seed, 0])
    returns = nulls.simulate(FLOOR_GARCH, days * nulls.DAY_STEPS, rng)
    klines = nulls.minute_klines(START_MS, returns)
    bar, _ = bars.bars(START_MS, days, klines, nulls._NO_METRICS, nulls._NO_FUNDING)
    out = msvol.label(bar["ts_ns"], bar["rv"], SMALL)
    minutes = np.diff(np.log(klines["close"]), prepend=np.log(klines["open"][0]))
    onsets = np.array([e["onset"] for e in out.episodes if e["kind"] == "event"], np.int64)
    assert onsets.size == found.paths[0]["events"] > 0 and out.nulls
    assert found.cells[0]["n_pos"] == onsets.size
    null_slices, segments, shifted = recording.calls
    assert null_slices.tobytes() == nulls.null_rows(minutes, out.nulls, 299).tobytes()
    assert len(null_slices) == found.cells[0]["n_neg"] > 0
    assert segments.tobytes() == nulls.pre_onset_rows(minutes, onsets, h, 299).tobytes()
    assert shifted.shape == (draws * onsets.size, 299)
    # every shifted segment ends h bars before a labelled bar at least a day from its event
    ends = np.array([int(np.flatnonzero(minutes == row[-1])[0]) + 1 for row in shifted])
    moved = ends // 5 + h
    assert np.all(ends % 5 == 0) and np.all(moved >= 14 * 288) and np.all(moved < days * 288)
    assert np.all(np.abs(moved.reshape(draws, -1) - onsets) >= 288)


def test_the_variance_floor_sits_above_chance_and_falls_with_the_horizon(
    floor: nulls.Floor,
) -> None:
    """What the floor is for: a clustering process with no critical dynamics hands the variance
    pipeline this AUC by construction. Over ten seeds its mean ran from 0.72 at no horizon to 0.58
    at two hours."""
    assert 0.60 <= _cell(floor, "variance", 0)["auc"] <= 0.85
    assert 0.55 <= _cell(floor, "variance", 6)["auc"] <= 0.80
    assert 0.42 <= _cell(floor, "variance", 24)["auc"] <= 0.72
    assert _cell(floor, "variance", 0)["auc"] > _cell(floor, "variance", 24)["auc"]


def test_the_ac1_floor_sits_at_chance(floor: nulls.Floor) -> None:
    """GARCH returns have no autocorrelation to trend: over ten seeds 0.42-0.60 at every horizon."""
    for h in HORIZONS:
        assert 0.38 <= _cell(floor, "ac", h)["auc"] <= 0.66


def test_the_shift_null_sits_at_chance_for_a_trend(floor: nulls.Floor) -> None:
    """A shifted segment can land anywhere, a null slice only in the calm, so a scorer of the level
    reads above chance on it (0.73 for the control below). A trend does not."""
    for cell in floor.cells:
        assert cell["shift_auc_lo"] <= cell["shift_auc_median"] <= cell["shift_auc_hi"]
        if cell["scorer"] != "level":
            assert 0.42 <= cell["shift_auc_median"] <= 0.60


def test_the_segments_are_aligned_with_the_onsets(floor: nulls.Floor) -> None:
    """The level control: high right before an onset, unlike its shifted self, and fading back."""
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
        reports.write(reports.floor(found, FLOOR_GARCH), tmp_path / name)
        for found, name in ((a, "a"), (b, "b"))
    ]
    names = sorted(p.name for p in written[0].iterdir())
    assert names == ["cells.parquet", "garch.parquet", "paths.parquet", "summary.md"]
    for name in names:
        assert (written[0] / name).read_bytes() == (written[1] / name).read_bytes()
    summary = (written[0] / "summary.md").read_text(encoding="utf-8")
    assert "alpha = 0.005" in summary and "beta = 0.9949" in summary


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
    assert path["events"] == 0 and math.isnan(path["delay_p90"])
    assert math.isnan(cell["auc"]) and cell["n_pos"] == cell["n_neg"] == 0
    assert cell["contaminated"] and cell["shift_draws"] == 0
    report = reports.floor(found, Garch(1e-7, 0.0, 0.0, 6.0))
    assert report.tables["cells"]["auc"].null_count == 1


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
