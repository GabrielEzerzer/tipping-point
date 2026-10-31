"""The cascade proxy (#29 S4; research pass section 3), on planted bars and a fake archive: a
planted cascade is detected, no leg alone makes one, the legs cover the same 30 minutes, the
thresholds hold at equality, the volume median and the onset read only their trailing days, gaps
are skipped and counted, merging and pooling follow the rule with both counts reported, nothing
looks ahead (truncation and perturbation, bit for bit), and the pooled list lands as a table in the
labels run layout. None of it is market data, except the tests named on_real_days at the end."""

import dataclasses
import hashlib
import json
import os
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pytest
from market_planted import KLINES_HEADER, METRICS_HEADER, Vision, funding_csv
from tipping_point import engine
from tipping_point.market import archive, bars, cascade, msvol
from tipping_point.market.__main__ import ARCHIVE, CONFIRMATORY
from tipping_point.market.cascade import LONG, SHORT, Params

Series = dict[str, np.ndarray]

ROOT = Path(__file__).resolve().parents[4]
FIRST = date(2023, 1, 2)
T0 = archive.day_ms(FIRST) * 1_000_000
BAR = msvol.BAR_NS
HOUR = 12  # bars
DAY = bars.DAY_BARS
W = 6
TRAILING = 28 * DAY
FIRST_WINDOW = 2 * W + TRAILING - 1  # the first window whose trailing days all exist
AT = 30 * DAY + 10 * HOUR  # where the single-cascade tests plant theirs
SQUEEZE = {"price": 0.06, "share": 0.8}
# The sweeps and the archive use a week of trailing days: the rule reads them the same however
# many there are, and it keeps those tests to seconds.
WEEK = Params(trailing_days=7)
WEEK_FIRST_WINDOW = 2 * W + 7 * DAY - 1


def calm(days: int = 34, seed: int = 1) -> Series:
    """Quiet bars: price and open interest walks far too small to reach the thresholds, balanced
    flow, ordinary volume."""
    rng = np.random.default_rng(seed)
    n = days * DAY
    return {
        "ts_ns": T0 + BAR * np.arange(n, dtype=np.int64),
        "close": 20_000 * np.exp(np.cumsum(rng.normal(0, 0.0004, n))),
        "volume": 100 * np.exp(rng.normal(0, 0.2, n)),
        "taker_buy_share": rng.uniform(0.47, 0.53, n),
        "oi": 100_000 * np.exp(np.cumsum(rng.normal(0, 0.0002, n))),
    }


def plant(
    series: Series,
    at: int,
    *,
    oi: float = -0.05,
    price: float = -0.06,
    share: float = 0.2,
    volume: float = 8.0,
) -> Series:
    """In place, over the W bars from `at`: price and open interest step evenly to these log
    amounts and stay there, each close at its bar's end and each open interest in the snapshot at
    its bar's start, as the feed has them; flow and volume change for those bars alone. With the
    defaults the windows closing at at+4..at+8 fire: open interest has fallen 3.33 % over the window
    closing at at+4 and only 2.5 % at at+3, and is back to that at at+9."""
    steps = np.arange(len(series["ts_ns"])) - at
    series["close"] *= np.exp(price * np.clip((steps + 1) / W, 0.0, 1.0))
    series["oi"] *= np.exp(oi * np.clip(steps / W, 0.0, 1.0))
    series["volume"][at : at + W] *= volume
    series["taker_buy_share"][at : at + W] = share
    return series


def flat(days: int = 30) -> Series:
    """Bars that never move, so a window's every measure can be worked by hand."""
    n = days * DAY
    return {
        "ts_ns": T0 + BAR * np.arange(n, dtype=np.int64),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 10.0),
        "taker_buy_share": np.full(n, 0.5),
        "oi": np.full(n, 1000.0),
    }


def _bar(ts_ns: int) -> int:
    return (ts_ns - T0) // BAR


def _fired(found: list[dict[str, Any]]) -> list[int]:
    return [_bar(c["fired_ts_ns"]) for c in found]


def _utc(ts_ns: int) -> str:
    return f"{datetime.fromtimestamp(ts_ns / 10**9, UTC):%Y-%m-%d %H:%M}"


def _counts_add_up(windows: dict[str, int]) -> None:
    parts = ("missing_oi", "missing_bars", "no_history", "evaluated")
    assert sum(windows[f"windows_{part}"] for part in parts) == windows["windows"]


# ---------------------------------------------------------------- the rule


def test_the_start_values_are_the_research_passs() -> None:
    p = Params()
    assert (p.a, p.b, p.c, p.v, p.window, p.trailing_days) == (0.03, 0.025, 0.15, 3.0, 6, 28)
    assert (p.merge_hours, p.both_hours, p.bounds_hours, p.onset_percentile) == (24, 1, 12, 99.0)
    # How open interest lines up with price is not here, so no count of events can choose it.
    assert len(dataclasses.fields(Params)) == 10


def test_a_planted_long_cascade_is_detected() -> None:
    series = plant(calm(), AT)
    found, windows = cascade.candidates(series)
    assert _fired(found) == list(range(AT + 4, AT + 9))
    assert {c["kind"] for c in found} == {LONG}
    assert _bar(found[0]["ts_ns"]) == AT  # the first bar that moved
    assert found[0]["flow"] < -0.5
    _counts_add_up(windows)


def test_open_interest_price_and_flow_cover_the_same_30_minutes() -> None:
    """Read by instants, not by index: a bar's oi is the open interest at its start and its close
    the price at its end. Each leg of every candidate spans the 30 minutes up to the instant it
    fired, and the volume and flow come from the six bars inside them."""
    series = plant(calm(), AT)
    ts, volume = series["ts_ns"], series["volume"]
    bought = series["taker_buy_share"] * volume

    def oi_at(t: int) -> float:
        return float(series["oi"][ts == t][0])

    def price_at(t: int) -> float:
        return float(series["close"][ts == t - BAR][0])

    found, _ = cascade.candidates(series)
    assert len(found) == 5
    for c in found:
        end = c["fired_ts_ns"]
        start = end - W * BAR
        inside = (start <= ts) & (ts < end)
        assert c["oi_change"] == pytest.approx(np.log(oi_at(end) / oi_at(start)), rel=1e-12)
        assert c["price_change"] == pytest.approx(
            np.log(price_at(end) / price_at(start)), rel=1e-12
        )
        assert c["volume"] == volume[inside].sum()
        expected = (2 * bought[inside].sum() - c["volume"]) / c["volume"]
        assert c["flow"] == pytest.approx(expected, rel=1e-12)
        assert start <= c["ts_ns"] < end


def test_a_short_squeeze_mirrors_price_and_flow_over_the_same_fall_in_open_interest() -> None:
    found, _ = cascade.candidates(plant(calm(), AT, **SQUEEZE))
    assert _fired(found) == list(range(AT + 4, AT + 9))
    assert {c["kind"] for c in found} == {SHORT}
    assert _bar(found[0]["ts_ns"]) == AT
    assert found[0]["oi_change"] < -0.03 and found[0]["price_change"] > 0.025
    assert found[0]["flow"] > 0.5


@pytest.mark.parametrize(
    "legs",
    [
        pytest.param({"price": 0.0, "share": 0.5, "volume": 1.0}, id="open-interest-drop-alone"),
        pytest.param({"share": 0.8}, id="price-falls-while-flow-buys"),
        pytest.param({"price": 0.06}, id="price-rises-while-flow-sells"),
        pytest.param({"oi": 0.0}, id="no-open-interest-drop"),
        pytest.param({"oi": 0.05}, id="positions-opening-into-the-move"),
        pytest.param({"volume": 1.0}, id="ordinary-volume"),
        pytest.param({"price": -0.01}, id="small-price-move"),
        pytest.param({"share": 0.5}, id="balanced-flow"),
    ],
)
def test_no_candidate_without_every_leg_pointing_the_same_way(legs: dict[str, float]) -> None:
    found, windows = cascade.candidates(plant(calm(), AT, **legs))
    assert found == []
    assert windows["windows_evaluated"] == len(calm()["ts_ns"]) - FIRST_WINDOW


def _exact(side: int) -> tuple[Series, int]:
    """One window, closing at k, with V exactly three times the median and F exactly 0.25 either
    way; the price steps on its last bar and open interest in the snapshot at its close."""
    k = 29 * DAY
    series = flat()
    series["oi"][k:] = 960.0
    series["close"][k - 1 :] = 100.0 + side * 3.0
    series["volume"][k - W : k] = 30.0
    series["taker_buy_share"][k - W : k] = 0.5 + side * 0.125
    return series, k


@pytest.mark.parametrize(("side", "kind"), [(-1, LONG), (1, SHORT)])
def test_each_threshold_holds_at_equality_and_fails_just_past_it(side: int, kind: str) -> None:
    series, k = _exact(side)
    loose, _ = cascade.candidates(series, Params(a=0.01, b=0.01, c=0.1, v=2.0))
    at_k = next(c for c in loose if _bar(c["fired_ts_ns"]) == k)
    assert at_k["flow"] == side * 0.25 and at_k["volume"] == 180.0
    assert at_k["volume_median"] == 60.0
    exact = Params(
        a=-at_k["oi_change"], b=side * at_k["price_change"], c=side * at_k["flow"], v=3.0
    )
    found, _ = cascade.candidates(series, exact)
    assert _fired(found) == [k] and found[0]["kind"] == kind
    for name in ("a", "b", "c", "v"):
        past = float(np.nextafter(getattr(exact, name), np.inf))
        assert cascade.candidates(series, dataclasses.replace(exact, **{name: past}))[0] == []


def test_the_volume_median_reads_the_trailing_28_days_and_not_the_window() -> None:
    series = plant(calm(), AT)
    found, _ = cascade.candidates(series)
    volume = series["volume"]
    for c in found:
        b = _bar(c["fired_ts_ns"]) - 1  # the window's last bar
        windows = [volume[j - W + 1 : j + 1].sum() for j in range(b - W - TRAILING + 1, b - W + 1)]
        assert c["volume_median"] == np.median(windows)
    b = _bar(found[0]["fired_ts_ns"]) - 1
    older = {**series, "volume": volume.copy()}
    older["volume"][: b - W - TRAILING - W + 2] *= 1000.0  # bars of no window in its trailing days
    assert cascade.candidates(older)[0][0] == found[0]
    busier = {**series, "volume": volume.copy()}
    busier["volume"][b - W - TRAILING : b - W] *= 10.0  # all of them: the median follows
    assert cascade.candidates(busier)[0] != found
    surge = {**series, "volume": volume.copy()}
    surge["volume"][b + 1 :] *= 1000.0  # what comes later cannot raise the bar
    assert cascade.candidates(surge)[0][0] == found[0]


def _stepped(returns: np.ndarray, moves: list[float], side: int = -1) -> tuple[Series, int, Params]:
    """A flat series whose closes follow `returns` and then the window's `moves`, its open interest
    stepping in the snapshot at the window's close, its volume and flow set for those bars."""
    series = flat(30)
    at = 29 * DAY
    r = returns.copy()
    r[at : at + W] = moves
    r[at + W :] = 0.0
    series["close"] = 100.0 * np.exp(np.cumsum(r))
    series["oi"][at + W :] = 960.0
    series["volume"][at : at + W] = 30.0
    series["taker_buy_share"][at : at + W] = 0.5 + side * 0.25
    return series, at, Params(b=0.025, v=2.9)


def _alternating(up: float, down: float) -> np.ndarray:
    """Returns up on odd bars and -down on even ones: bar 29 * DAY - 1, before any window, is up."""
    n = 30 * DAY
    return np.where(np.arange(n) % 2 == 1, up, -down)


def test_the_onset_is_the_first_bar_past_its_trailing_percentile() -> None:
    series, at, params = _stepped(
        _alternating(0.006, 0.006), [-0.001, -0.002, -0.007, -0.006, -0.006, -0.006]
    )
    found, _ = cascade.candidates(series, params)
    assert _bar(found[0]["fired_ts_ns"]) == at + W
    assert _bar(found[0]["ts_ns"]) == at + 2  # 0.007 beats the trailing 0.006


def test_the_onset_is_the_windows_first_bar_when_no_bar_passes() -> None:
    series, at, params = _stepped(_alternating(0.006, 0.006), [-0.005] * 6)
    found, _ = cascade.candidates(series, params)
    assert _bar(found[0]["fired_ts_ns"]) == at + W and _bar(found[0]["ts_ns"]) == at


@pytest.mark.parametrize("side", [-1, 1])
def test_the_onset_reads_returns_in_the_candidates_own_direction(side: int) -> None:
    """Large moves the other way never set the bar: |return| would have put this onset on the
    window's first bar, since nothing in the window reaches 0.02."""
    moves = [side * m for m in (0.0005, 0.007, 0.007, 0.007, 0.007, 0.007)]
    returns = -side * _alternating(0.02, 0.001)
    series, at, params = _stepped(returns, moves, side)
    found, _ = cascade.candidates(series, params)
    assert found[0]["kind"] == (SHORT if side > 0 else LONG)
    assert _bar(found[0]["fired_ts_ns"]) == at + W and _bar(found[0]["ts_ns"]) == at + 1


def test_the_onset_percentile_reads_only_the_bars_own_trailing_days() -> None:
    """Moves older than a bar's trailing days or later than the bar would lift the percentile past
    everything in the window, and put the onset on the window's first bar."""
    returns = _alternating(0.006, 0.006)
    returns[:150] = -0.05  # older than 28 days before any bar of the window
    series, at, params = _stepped(returns, [-0.001, -0.002, -0.007, -0.006, -0.006, -0.006])
    later = at + W + 1
    series["close"][later:] *= np.exp(np.cumsum(np.full(len(series["close"]) - later, -0.05)))
    found, _ = cascade.candidates(series, params)
    assert _bar(found[0]["fired_ts_ns"]) == at + W and _bar(found[0]["ts_ns"]) == at + 2


def test_a_bar_off_the_grid_is_refused() -> None:
    series = calm(2)
    gapped = {name: np.delete(column, 100) for name, column in series.items()}
    with pytest.raises(ValueError, match="gapless"):
        cascade.candidates(gapped)


# ---------------------------------------------------------------- the alignment on a sharp V


def _v_shape(at: int) -> Series:
    """2023-08-17's shape: the whole fall and its selling in bar `at`, the drop in open interest
    showing only in the snapshot stamped at its close, and in the next bar half the volume buying
    back part of the fall."""
    series = calm()
    series["close"][at:] *= np.exp(-0.10)
    series["close"][at + 1 :] *= np.exp(0.045)
    series["oi"][at + 1 :] *= np.exp(-0.08)
    series["volume"][at] *= 100.0
    series["taker_buy_share"][at] = 0.3
    series["volume"][at + 1] *= 50.0
    series["taker_buy_share"][at + 1] = 0.7
    return series


def test_a_sharp_v_fires_on_its_fall_and_gives_no_short_squeeze() -> None:
    series = _v_shape(AT)
    found, windows = cascade.candidates(series)
    assert [(c["kind"], _bar(c["ts_ns"]), _bar(c["fired_ts_ns"])) for c in found] == [
        (LONG, AT, AT + 1)
    ]
    assert found[0]["flow"] < -0.3
    assert found[0]["price_change"] == pytest.approx(-0.10, abs=0.005)
    assert found[0]["volume"] == series["volume"][AT - 5 : AT + 1].sum()
    assert windows["windows"] == len(series["ts_ns"]) - W - 1
    _counts_add_up(windows)


def test_open_interest_read_one_stamp_stale_misses_the_fall_and_invents_a_squeeze() -> None:
    """What pairing oi[k] with close[k] does, shown by shifting the column, which is why the rule
    does not. The window that first holds the drop holds the rebound too, so its flow fails, and the
    last window still holding the drop holds the rebound alone."""
    series = _v_shape(AT)
    stale = {**series, "oi": np.append(np.nan, series["oi"][:-1])}
    found, _ = cascade.candidates(stale)
    assert [(c["kind"], _bar(c["fired_ts_ns"])) for c in found] == [(SHORT, AT + 7)]


# ---------------------------------------------------------------- gaps


@pytest.mark.parametrize(
    ("step", "fires", "why"),
    [
        (AT + 3, [], "inside every window that would fire, the two ends of each intact"),
        (AT - 2, [AT + 5, AT + 6, AT + 7, AT + 8], "the far end of the first window"),
        (AT + 8, [AT + 4, AT + 5, AT + 6, AT + 7], "the near end of the last window"),
    ],
)
def test_a_window_touching_a_missing_open_interest_step_is_counted_not_evaluated(
    step: int, fires: list[int], why: str
) -> None:
    series = plant(calm(), AT)
    series["oi"][step] = np.nan
    found, windows = cascade.candidates(series)
    assert _fired(found) == fires, why
    assert windows["windows_missing_oi"] == W + 1  # k - W <= step <= k
    assert windows["windows_evaluated"] == len(series["ts_ns"]) - FIRST_WINDOW - (W + 1)
    _counts_add_up(windows)


def test_an_open_interest_of_zero_counts_as_missing() -> None:
    """Binance's metrics hold such rows: 2023-08-09 02:35, on both symbols."""
    series = plant(calm(), AT)
    series["oi"][AT + 3] = 0.0
    found, windows = cascade.candidates(series)
    assert found == [] and windows["windows_missing_oi"] == W + 1


def test_a_missing_bar_is_counted_apart_and_a_step_missing_too_counts_once() -> None:
    series = plant(calm(), AT)
    for name in ("close", "volume", "taker_buy_share"):  # as bars.py leaves an incomplete bar
        series[name][AT + 2] = np.nan
    found, windows = cascade.candidates(series)
    assert found == []
    assert (windows["windows_missing_bars"], windows["windows_missing_oi"]) == (W + 1, 0)
    _counts_add_up(windows)
    series["oi"][AT + 3] = np.nan  # the snapshot at that bar's close touches the same windows
    _, windows = cascade.candidates(series)
    assert (windows["windows_missing_bars"], windows["windows_missing_oi"]) == (0, W + 1)
    _counts_add_up(windows)


def test_a_bar_that_traded_nothing_is_not_a_gap() -> None:
    series = plant(calm(), AT)
    series["volume"][AT + 1] = 0.0
    series["taker_buy_share"][AT + 1] = np.nan  # as bars.py leaves it
    found, windows = cascade.candidates(series)
    assert windows["windows_missing_bars"] == 0
    assert _fired(found) == list(range(AT + 4, AT + 9))
    assert -1.0 <= found[0]["flow"] < -0.5


def test_a_window_without_its_trailing_days_is_counted_not_evaluated() -> None:
    series = plant(calm(), FIRST_WINDOW - 6)  # fires at FIRST_WINDOW - 2 .. FIRST_WINDOW + 2
    found, windows = cascade.candidates(series)
    assert _fired(found) == [FIRST_WINDOW, FIRST_WINDOW + 1, FIRST_WINDOW + 2]
    assert windows["windows_no_history"] == FIRST_WINDOW - W - 1
    _counts_add_up(windows)


def test_trailing_days_with_no_complete_window_are_no_history() -> None:
    series = calm(60)
    back = 30 * DAY
    for name in ("close", "volume", "taker_buy_share"):
        series[name][DAY:back] = np.nan
    plant(series, back + 2 * W)  # fires on a median of its few windows since the outage
    found, windows = cascade.candidates(series)
    # The first day's windows have no history yet. Back at bar `back`, windows closing at back+7..
    # touch no missing bar, but the first complete window ends with bar back+5, which the trailing
    # days reach only from the window closing at back+12. In between, a window touches missing bars
    # and is counted so.
    assert windows["windows_no_history"] == (DAY - W) + 5
    assert _fired(found)[0] == back + 2 * W + 4
    _counts_add_up(windows)


# ---------------------------------------------------------------- merging and pooling


def _candidate(fired: int, onset: int | None = None, kind: str = LONG) -> dict[str, Any]:
    start = fired - 3 if onset is None else onset
    return {"ts_ns": T0 + start * BAR, "fired_ts_ns": T0 + fired * BAR, "kind": kind}


def test_candidates_within_24_hours_merge_into_the_earliest() -> None:
    far = 100 * DAY
    rows = cascade.merge(
        [
            _candidate(far, onset=far - 4),
            _candidate(far + 1),
            _candidate(far + 200, kind=SHORT),  # the other side merges too
            _candidate(far + DAY),  # exactly 24 hours after the earliest: within
            _candidate(far + DAY + 1),  # past it, however close to the one before: a new one
            _candidate(far + 3 * DAY),
        ]
    )
    assert [_bar(r["ts_ns"]) for r in rows] == [far - 4, far + DAY + 1 - 3, far + 3 * DAY - 3]
    assert [r["candidates"] for r in rows] == [4, 1, 1]
    assert rows[0]["kind"] == LONG
    assert [_bar(r["last_fired_ts_ns"]) for r in rows] == [far + DAY, far + DAY + 1, far + 3 * DAY]
    assert cascade.merge([]) == []


def test_both_symbols_firing_within_an_hour_are_one_market_event_with_two_instances() -> None:
    far = 100 * DAY
    rows = cascade.pool(
        {
            "BTCUSDT": [_candidate(far), _candidate(far + 3 * DAY), _candidate(far + 6 * DAY)],
            "ETHUSDT": [
                _candidate(far + HOUR),  # exactly an hour later: within
                _candidate(far + 3 * DAY + HOUR + 1),  # past it
                _candidate(far + 6 * DAY - 5),  # ETH first this time
            ],
        }
    )
    assert [(_bar(r["fired_ts_ns"]) - far, r["symbol"], r["market_event"]) for r in rows] == [
        (0, "BTCUSDT", 0),
        (HOUR, "ETHUSDT", 0),
        (3 * DAY, "BTCUSDT", 1),
        (3 * DAY + HOUR + 1, "ETHUSDT", 2),
        (6 * DAY - 5, "ETHUSDT", 3),
        (6 * DAY, "BTCUSDT", 3),
    ]
    for r in rows:
        assert r["window_start_ts_ns"] == r["ts_ns"] - 12 * HOUR * BAR
        assert r["window_end_ts_ns"] == r["ts_ns"] + 12 * HOUR * BAR
    assert cascade.pool({"BTCUSDT": [], "ETHUSDT": []}) == []


def test_a_group_never_chains_past_its_first_firing() -> None:
    """The first of a market event anchors it, as the earliest candidate anchors an instance."""
    far = 100 * DAY
    rows = cascade.pool(
        {
            "A": [_candidate(far)],
            "B": [_candidate(far + HOUR)],
            "C": [_candidate(far + HOUR + 1)],  # within the hour of B, not of A
        }
    )
    assert [r["market_event"] for r in rows] == [0, 0, 1]


# ---------------------------------------------------------------- nothing looks ahead


PLANTS: dict[str, list[tuple[int, dict[str, float]]]] = {
    "BTCUSDT": [
        (8 * DAY + 10 * HOUR, {}),  # with ETH's, within the hour: one market event
        (8 * DAY + 15 * HOUR, {}),  # merges into the one above
        (11 * DAY, SQUEEZE),
        (11 * DAY + 23 * HOUR, {}),  # the other side, inside the squeeze's 24 hours: merges
        (12 * DAY + 2 * HOUR, {}),  # past them, though within 24 hours of the one above
        (17 * DAY + 3 * HOUR, {}),
    ],
    "ETHUSDT": [
        (8 * DAY + 10 * HOUR + 8, {}),
        (14 * DAY + 6 * HOUR, SQUEEZE),
        (19 * DAY + 20 * HOUR, {}),
    ],
}
SEEDED = 20
REACH = 3 * DAY  # bars a perturbed run carries past T


@pytest.fixture(scope="module")
def pair() -> dict[str, Series]:
    out = {"BTCUSDT": calm(23, 1), "ETHUSDT": calm(23, 2)}
    for symbol, plants in PLANTS.items():
        for at, legs in plants:
            plant(out[symbol], at, **legs)
    out["ETHUSDT"]["oi"][17 * DAY] = np.nan
    return out


def _cut(series: Series, T: int) -> Series:
    return {name: column[: T + 1] for name, column in series.items()}


def _moved(series: Series, T: int, seed: int, reach: int | None = REACH) -> Series:
    """The series through T + reach, every bar after T shaken and 5 % of them dropped, with a fresh
    cascade soon after T: any bar's dependence on what follows it has plenty to read."""
    rng = np.random.default_rng([seed, T])
    out = {
        name: column[: None if reach is None else T + 1 + reach].copy()
        for name, column in series.items()
    }
    size = len(out["ts_ns"]) - T - 1
    if size <= 0:
        return out
    after, at = T + 1, T + 2
    out["close"][after:] *= np.exp(rng.normal(0.0, 0.001, size))
    out["oi"][after:] *= np.exp(rng.normal(0.0, 0.001, size))
    out["volume"][after:] *= np.exp(rng.normal(0.0, 1.0, size))
    out["taker_buy_share"][after:] = rng.uniform(0.0, 1.0, size)
    if at + W < len(out["ts_ns"]):
        plant(out, at)
    clear = max(0, at + 2 * W - after)  # so the fresh cascade stays whole and fires
    for name in ("close", "oi", "volume"):
        out[name][after:][clear:][rng.uniform(size=size)[clear:] < 0.05] = np.nan
    return out


def _dumped(found: list[dict[str, Any]], T_ns: int) -> list[str]:
    return [json.dumps(c) for c in found if c["fired_ts_ns"] <= T_ns]


def _settled(rows: list[dict[str, Any]], T_ns: int) -> list[str]:
    """The pooled rows fired by T: only a group still open at T can yet gain a firing."""
    day_ns = 24 * cascade.HOUR_NS
    out = []
    for r in rows:
        if r["fired_ts_ns"] <= T_ns:
            open_keys = ("candidates", "last_fired_ts_ns")
            closed = r["fired_ts_ns"] + day_ns <= T_ns
            out.append(json.dumps({k: v for k, v in r.items() if closed or k not in open_keys}))
    return out


def _through(every: dict[str, int], later: dict[str, int]) -> dict[str, int]:
    return {key: every[key] - later[key] for key in every}


def _cuts(n: int) -> list[int]:
    planted = [at + d for plants in PLANTS.values() for at, _ in plants for d in (-1, 3, 4, 8, 9)]
    seeded = np.random.default_rng(4).integers(WEEK_FIRST_WINDOW, n - 1, SEEDED)
    return sorted({*planted, WEEK_FIRST_WINDOW - 1, WEEK_FIRST_WINDOW, n - 1, *seeded.tolist()})


def _runs(pair: dict[str, Series]) -> dict[str, tuple[list[dict[str, Any]], dict[str, int]]]:
    return {symbol: cascade.candidates(series, WEEK) for symbol, series in pair.items()}


def test_no_candidate_at_or_before_t_depends_on_a_later_bar(pair: dict[str, Series]) -> None:
    full = _runs(pair)
    found = {symbol: run[0] for symbol, run in full.items()}
    pooled = cascade.pool(found)
    assert sum(len(some) for some in found.values()) == 5 * sum(len(p) for p in PLANTS.values())
    assert {r["kind"] for r in pooled} == {LONG, SHORT} and len(pooled) >= 6
    assert any(r["candidates"] > 5 for r in pooled)  # a real merge
    assert len({r["market_event"] for r in pooled}) < len(pooled)  # and a real pairing
    n = len(pair["BTCUSDT"]["ts_ns"])
    cuts = _cuts(n)
    assert len(cuts) >= 2 * len(PLANTS) + SEEDED
    btc, changed = "BTCUSDT", 0
    for T in cuts:
        T_ns = int(pair[btc]["ts_ns"][T])
        truncated = _runs({symbol: _cut(series, T) for symbol, series in pair.items()})
        moved = {symbol: _moved(pair[symbol], T, 7 + i) for i, symbol in enumerate(pair)}
        perturbed = _runs(moved)
        for variant, runs in (("truncated", truncated), ("perturbed", perturbed)):
            rows = {symbol: run[0] for symbol, run in runs.items()}
            for symbol in pair:
                got, want = _dumped(rows[symbol], T_ns), _dumped(found[symbol], T_ns)
                assert got == want, f"{symbol} candidates differ by T = {T}, {variant}"
            got, want = _settled(cascade.pool(rows), T_ns), _settled(pooled, T_ns)
            assert got == want, f"the pooled rows differ by T = {T}, {variant}"
        counted = truncated[btc][1]  # the windows counted by T are the same in all three
        later = cascade.candidates(pair[btc], WEEK, since=T + 1)[1]
        assert counted == _through(full[btc][1], later), f"counts differ by T = {T}"
        later = cascade.candidates(moved[btc], WEEK, since=T + 1)[1]
        assert counted == _through(perturbed[btc][1], later), f"counts differ by T = {T}"
        end = moved[btc]["ts_ns"][-1]
        after = [c for c in perturbed[btc][0] if c["fired_ts_ns"] > T_ns]
        changed += after != [c for c in found[btc] if T_ns < c["fired_ts_ns"] <= end]
    # At every cut with room for it, the bars after T changed what fired after T.
    assert changed == sum(n > T + 2 + W for T in cuts)


def test_the_same_to_the_end_of_the_series(pair: dict[str, Series]) -> None:
    """One cut perturbed all the way to the end of the series, not just past its reach."""
    series = pair["BTCUSDT"]
    T = PLANTS["BTCUSDT"][1][0] + 8
    T_ns = int(series["ts_ns"][T])
    moved = _moved(series, T, 11, reach=None)
    assert len(moved["ts_ns"]) == len(series["ts_ns"])
    full, shaken = cascade.candidates(series, WEEK)[0], cascade.candidates(moved, WEEK)[0]
    assert _dumped(shaken, T_ns) == _dumped(full, T_ns) != []
    assert len(shaken) > len(_dumped(shaken, T_ns))


def test_starting_later_or_asking_later_changes_nothing_inside_what_is_asked(
    pair: dict[str, Series],
) -> None:
    """What recall relies on: the rule at a bar reads only its trailing days, so a read starting a
    day more than those early gives what a run from the series start gives, and `since` holds back
    the rest."""
    series = pair["BTCUSDT"]
    full, every = cascade.candidates(series, WEEK)
    since = 11 * DAY
    start = since - 8 * DAY
    asked = [c for c in full if _bar(c["fired_ts_ns"]) >= since]
    part = {name: column[start:] for name, column in series.items()}
    found, windows = cascade.candidates(part, WEEK, since=since - start)
    assert asked and json.dumps(found) == json.dumps(asked)
    assert windows["windows"] == len(series["ts_ns"]) - since
    gone = since - WEEK_FIRST_WINDOW
    assert windows["windows_evaluated"] == every["windows_evaluated"] - gone
    held, _ = cascade.candidates(series, WEEK, since=since)
    assert json.dumps(held) == json.dumps(asked)


# ---------------------------------------------------------------- the archive, the run, the CLI


def _publish(
    fake: Vision,
    symbol: str,
    series: Series,
    drop_minutes: tuple[int, ...] = (),
    drop_steps: tuple[int, ...] = (),
) -> None:
    """Binance's files for `series`: each bar's move in five equal log steps, volume in five equal
    parts, and the metrics carrying contracts and their USDT value, which falls with the price."""
    log_close = np.log(series["close"])
    before = np.concatenate([[log_close[0]], log_close[:-1]])
    steps = np.arange(1, 6) / 5
    closes = np.exp(before[:, None] + (log_close - before)[:, None] * steps).ravel()
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs, lows = np.maximum(opens, closes), np.minimum(opens, closes)
    volume = np.repeat(series["volume"] / 5, 5)
    taker = (volume * np.repeat(series["taker_buy_share"], 5)).tolist()
    opens, highs, lows, closes = opens.tolist(), highs.tolist(), lows.tolist(), closes.tolist()
    volume = volume.tolist()
    contracts, close = series["oi"].tolist(), series["close"].tolist()
    days = len(log_close) // DAY
    for d in range(days):
        day = FIRST + timedelta(days=d)
        start = archive.day_ms(day)
        rows = [KLINES_HEADER]
        for m in range(1440):
            i = d * 1440 + m
            if i not in drop_minutes:
                rows.append(
                    f"{start + m * 60_000},{opens[i]:.4f},{highs[i]:.4f},{lows[i]:.4f},"
                    f"{closes[i]:.4f},{volume[i]:.6f},{start + (m + 1) * 60_000 - 1},0,10,"
                    f"{taker[i]:.6f},0,0"
                )
        fake.add(
            archive.Source("klines", symbol, f"{day:%Y-%m-%d}"), ("\n".join(rows) + "\n").encode()
        )
        midnight = datetime(day.year, day.month, day.day, tzinfo=UTC)
        rows = [METRICS_HEADER]
        for step in range(DAY):
            b = d * DAY + step
            if b not in drop_steps:
                stamp = midnight + timedelta(minutes=5 * step)
                rows.append(
                    f"{stamp:%Y-%m-%d %H:%M:%S},{symbol},{contracts[b]:.8f},"
                    f"{contracts[b] * close[b]:.8f},1.9,1.05,2.1,0.95"
                )
        fake.add(
            archive.Source("metrics", symbol, f"{day:%Y-%m-%d}"), ("\n".join(rows) + "\n").encode()
        )
    months = {(FIRST + timedelta(days=d)).replace(day=1) for d in range(days)}
    for month in sorted(months):
        fake.add(archive.Source("fundingRate", symbol, f"{month:%Y-%m}"), funding_csv(month))


DAYS = 14
END = FIRST + timedelta(days=DAYS - 1)
SPAN = ["--start", FIRST.isoformat(), "--end", END.isoformat()]
BTC_AT = 8 * DAY + 10 * HOUR
ETH_AT = BTC_AT + 8  # 40 minutes later
ETH_SQUEEZE = 9 * DAY + 12 * HOUR
BTC_GAP = 10 * DAY + 12 * HOUR  # a metrics row absent
ETH_GAP = 10 * DAY + 6 * HOUR  # a minute's kline absent
FLAT_CONTRACTS = 11 * DAY + 14 * HOUR  # price, flow and volume; the USDT value falls, contracts not


@pytest.fixture(scope="module")
def fetched(tmp_path_factory: pytest.TempPathFactory) -> Path:
    btc, eth = calm(DAYS, 1), calm(DAYS, 2)
    plant(btc, BTC_AT)
    plant(btc, BTC_AT + 5 * HOUR)
    plant(eth, ETH_AT)
    plant(eth, ETH_SQUEEZE, **SQUEEZE)
    plant(eth, FLAT_CONTRACTS, oi=0.0)
    fake = Vision()
    _publish(fake, "BTCUSDT", btc, drop_steps=(BTC_GAP,))
    _publish(fake, "ETHUSDT", eth, drop_minutes=(ETH_GAP * 5 + 2,))
    dest = tmp_path_factory.mktemp("archive")
    done = archive.fetch(archive.SYMBOLS, FIRST, END, dest, fake.get)
    assert {e["status"] for e in done.values()} == {"verified"}
    return dest


def _run(fetched: Path, data: Path, tag: str, *more: str) -> Path:
    where = ["--archive", str(fetched), "--data-dir", str(data), "--tag", tag]
    assert cascade.main([*SPAN, *where, "--trailing-days", "7", *more]) == 0
    (run_dir,) = (data / "labels").glob(f"*-{tag}")
    return run_dir


@pytest.fixture(scope="module")
def run(fetched: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _run(fetched, tmp_path_factory.mktemp("data"), "a")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_pooled_list_lands_as_a_table_in_the_labels_run_layout(run: Path) -> None:
    manifest, complete = _json(run / "manifest.json"), _json(run / "complete.json")
    assert (manifest["domain"], manifest["key"], manifest["tables"]) == (
        "labels",
        "ts_ns",
        ["cascades"],
    )
    assert manifest["params"]["cascade"] == dataclasses.asdict(WEEK)
    assert (manifest["licence"], manifest["attribution"]) == ("CC BY-NC-SA 4.0", "Binance Vision")
    assert len(manifest["sources"]) == 2 * (2 * DAYS + 1) and manifest["missing_sources"] == []
    (path,) = (run / "cascades").glob("*.parquet")
    table = pq.read_table(path)
    assert table.schema.equals(cascade.SCHEMA, check_metadata=False)
    assert complete["rows"] == {"cascades": 3} and complete["clean"]
    relative = f"cascades/{path.name}"
    assert complete["files"][relative] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert int(path.stem) == table["ts_ns"][0].as_py()
    data = run.parent.parent
    rows = engine.read(data, "labels", run.name, "cascades").to_pylist()
    assert [
        (_bar(r["ts_ns"]), r["symbol"], r["market_event"], r["kind"], r["candidates"]) for r in rows
    ] == [
        (BTC_AT, "BTCUSDT", 0, LONG, 10),  # the second plant, five hours on, merged into it
        (ETH_AT, "ETHUSDT", 0, LONG, 5),
        (ETH_SQUEEZE, "ETHUSDT", 1, SHORT, 5),
    ]
    first = rows[0]
    assert _bar(first["fired_ts_ns"]) == BTC_AT + 4
    assert _bar(first["last_fired_ts_ns"]) == BTC_AT + 5 * HOUR + 8
    assert first["window_start_ts_ns"] == first["ts_ns"] - 12 * 3_600 * 10**9
    assert first["window_end_ts_ns"] == first["ts_ns"] + 12 * 3_600 * 10**9
    assert first["oi_change"] < -0.03 and first["price_change"] < -0.025 and first["flow"] < -0.15
    assert first["volume"] >= 3 * first["volume_median"]


def test_the_run_reports_both_counts_and_every_gap(run: Path) -> None:
    checks = _json(run / "complete.json")["checks"]
    assert (checks["asset_instances"], checks["market_events"]) == (3, 2)
    assert checks["liquidations"] == {"status": "not recorded"}
    btc, eth = checks["BTCUSDT"], checks["ETHUSDT"]
    assert (btc["windows_missing_oi"], btc["windows_missing_bars"]) == (W + 1, 0)
    assert (eth["windows_missing_oi"], eth["windows_missing_bars"]) == (0, W + 1)
    assert btc["bars_without_oi"] == 1 and eth["bars_incomplete"] == 1
    assert (btc["candidates"], eth["candidates"]) == (10, 10)
    for symbol in (btc, eth):
        _counts_add_up(symbol)
        assert symbol["windows_no_history"] == WEEK_FIRST_WINDOW - W - 1


def test_two_runs_write_identical_tables(fetched: Path, run: Path) -> None:
    again = _run(fetched, run.parent.parent, "b")
    files = [_json(r / "complete.json")["files"] for r in (run, again)]
    assert files[0] == files[1] and len(files[0]) == 1


def test_every_parameter_is_an_option(fetched: Path, tmp_path: Path) -> None:
    run_dir = _run(fetched, tmp_path, "tight", "--a", "0.2", "--merge-hours", "6")
    assert _json(run_dir / "complete.json")["rows"] == {"cascades": 0}
    params = _json(run_dir / "manifest.json")["params"]["cascade"]
    assert (params["a"], params["merge_hours"]) == (0.2, 6)
    assert (run_dir / "cascades" / f"{0:020d}.parquet").is_file()


def test_the_confirmatory_span_and_unverified_files_are_refused(
    fetched: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert date(2024, 1, 1) == CONFIRMATORY
    late = ["--start", "2023-12-30", "--end", "2024-01-02", "--archive", str(fetched)]
    data = tmp_path / "data"
    assert cascade.main([*late, "--data-dir", str(data)]) == 1
    assert "market-labels-v1" in capsys.readouterr().err and not data.exists()
    where = ["--archive", str(tmp_path / "nothing-here"), "--data-dir", str(data)]
    assert cascade.main([*SPAN, *where]) == 1
    assert "not fetched" in capsys.readouterr().err and not data.exists()


# ---------------------------------------------------------------- the liquidations hook


def _recorded(data: Path, run_id: str, rows: dict[str, list[Any]]) -> None:
    folder = data / "market" / run_id / "liquidations"
    folder.mkdir(parents=True)
    pq.write_table(pa.table(rows), folder / f"{0:020d}.parquet")


def test_no_liquidations_table_is_not_recorded_and_no_failure(tmp_path: Path) -> None:
    rows = [{"ts_ns": T0, "fired_ts_ns": T0, "symbol": "BTCUSDT"}]
    assert cascade.liquidations(tmp_path, rows) == {"status": "not recorded"}
    (tmp_path / "market" / "a-run" / "trades").mkdir(parents=True)  # a market run, no such table
    assert cascade.liquidations(tmp_path, rows) == {"status": "not recorded"}
    columns = {"ts_ns": [], "symbol": [], "price": [], "qty": []}
    schema = pa.schema(
        [
            ("ts_ns", pa.int64()),
            ("symbol", pa.string()),
            ("price", pa.float64()),
            ("qty", pa.float64()),
        ]
    )
    folder = tmp_path / "market" / "a-run" / "liquidations"
    folder.mkdir()
    pq.write_table(pa.table(columns, schema=schema), folder / f"{0:020d}.parquet")
    assert cascade.liquidations(tmp_path, rows) == {"status": "not recorded"}


def test_a_recorded_table_gives_each_covered_row_its_30_minute_notional(tmp_path: Path) -> None:
    fired = T0 + 100 * BAR
    start, end = fired - 6 * BAR, fired  # the 30 minutes of the window: bars 94-99
    _recorded(
        tmp_path,
        "bybit-1",
        {
            "ts_ns": [T0, start - 1, start, start + BAR, end - 1, end, start, T0 + 5000 * BAR],
            "symbol": ["BTCUSDT"] * 6 + ["ETHUSDT", "BTCUSDT"],
            "price": [100.0] * 8,
            "qty": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        },
    )
    _recorded(
        tmp_path,
        "bybit-2",
        {
            "ts_ns": [T0 + 5001 * BAR],
            "symbol": ["BTCUSDT"],
            "price": [100.0],
            "qty": [9.0],
        },
    )
    rows = [
        {"ts_ns": fired - 3 * BAR, "fired_ts_ns": fired, "symbol": "BTCUSDT"},
        {"ts_ns": fired - 3 * BAR, "fired_ts_ns": fired, "symbol": "ETHUSDT"},
        {"ts_ns": T0, "fired_ts_ns": T0 + 2 * BAR, "symbol": "BTCUSDT"},  # starts before the record
        {"ts_ns": T0, "fired_ts_ns": T0 + 6000 * BAR, "symbol": "BTCUSDT"},  # ends after it
    ]
    report = cascade.liquidations(tmp_path, rows)
    assert report["status"] == "recorded" and report["rows"] == 9
    assert (report["first_ts_ns"], report["last_ts_ns"]) == (T0, T0 + 5001 * BAR)
    assert report["cascades"] == [
        {"ts_ns": fired - 3 * BAR, "symbol": "BTCUSDT", "notional": 100.0 * (3 + 4 + 5)},
        {"ts_ns": fired - 3 * BAR, "symbol": "ETHUSDT", "notional": 700.0},
    ]


def test_the_run_reads_a_recorded_table(fetched: Path, tmp_path: Path) -> None:
    last_bar = T0 + (BTC_AT + 3) * BAR  # of the window that fired first, at BTC_AT + 4
    _recorded(
        tmp_path,
        "bybit-1",
        {
            "ts_ns": [T0 + DAY * BAR, last_bar, T0 + 13 * DAY * BAR],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "price": [100.0, 100.0, 100.0],
            "qty": [1.0, 2.0, 1.0],
        },
    )
    run_dir = _run(fetched, tmp_path, "with-liquidations")
    report = _json(run_dir / "complete.json")["checks"]["liquidations"]
    assert report["status"] == "recorded"
    by_row = {(c["symbol"], _bar(c["ts_ns"])): c["notional"] for c in report["cascades"]}
    assert by_row == {
        ("BTCUSDT", BTC_AT): 200.0,
        ("ETHUSDT", ETH_AT): 0.0,
        ("ETHUSDT", ETH_SQUEEZE): 0.0,
    }


# ---------------------------------------------------------------- the documented events


def test_the_documented_list_is_committed_with_its_sources() -> None:
    events = cascade.documented()
    assert [e["name"] for e in events] == [
        "2023-08-17",
        "2024-08-05",
        "2024-12",
        "2025-02",
        "2025-04",
        "2025-10-10",
    ]
    for event in events:
        first, last = event["first_day"], event["last_day"]
        assert isinstance(first, date) and first <= last
        assert event["name"] in (first.isoformat(), f"{first:%Y-%m}")
        assert event["sources"] and all(s.startswith("https://") for s in event["sources"])
    days = {e["name"]: (e["first_day"], e["last_day"]) for e in events}
    assert days["2024-12"] == (date(2024, 12, 1), date(2024, 12, 31))
    assert days["2025-02"] == (date(2025, 2, 1), date(2025, 2, 28))
    assert days["2025-04"] == (date(2025, 4, 1), date(2025, 4, 30))
    # One event lies in the pilot year, the rest in the confirmatory span: see the recall test.
    assert [e["name"] for e in events if e["last_day"] < CONFIRMATORY] == ["2023-08-17"]


def test_recall_refuses_without_its_switch(tmp_path: Path) -> None:
    for environ in ({}, {cascade.SWITCH: ""}):
        with pytest.raises(PermissionError, match=cascade.SWITCH):
            cascade.recall(tmp_path, environ=environ)


def test_recall_says_only_what_the_rule_says_inside_each_events_days(fetched: Path) -> None:
    """On the fake archive. `quiet`'s read holds the cascades of day 8 in its trailing days, and
    they do not come out: only the windows ending inside the event's own day are asked for."""

    def on(day: int, name: str) -> dict[str, Any]:
        at = FIRST + timedelta(days=day)
        return {"name": name, "first_day": at, "last_day": at}

    events = [on(8, "hit"), on(11, "quiet"), on(18, "unfetched")]
    switched = {cascade.SWITCH: "1"}
    hit, quiet, unfetched = cascade.recall(fetched, WEEK, events, environ=switched)
    assert hit["recalled"] is True
    assert hit["symbols"]["BTCUSDT"]["candidates"] == 10
    assert _bar(hit["symbols"]["BTCUSDT"]["first"]["ts_ns"]) == BTC_AT
    assert _bar(hit["symbols"]["ETHUSDT"]["first"]["ts_ns"]) == ETH_AT
    assert quiet["recalled"] is False
    for symbol in quiet["symbols"].values():
        assert (symbol["status"], symbol["candidates"], symbol["first"]) == ("read", 0, None)
        assert symbol["windows"] == symbol["windows_evaluated"] == DAY
    assert unfetched["recalled"] is None
    assert {s["status"] for s in unfetched["symbols"].values()} == {"not read"}


def _real_columns(symbol: str, first: date, last: date) -> dict[str, dict[str, np.ndarray]]:
    """A span's columns by kind, from days kept locally (gitignored), or a visible skip. The callers
    keep to 2023: nothing confirmatory is read."""
    assert last < CONFIRMATORY
    try:
        span = archive.collect([symbol], first, last, ROOT / ARCHIVE)
    except archive.ArchiveError as error:
        pytest.skip(f"real days not here: {error}")
    return {kind: span.columns[(symbol, kind)] for kind in archive.KINDS}


def test_2023_08_17_fires_on_both_symbols_on_real_days() -> None:
    """The pilot year's documented cascade, at the start values, pinned here so that the pilot's
    choice of thresholds leaves this alone. BTC's window reads the whole 30-minute fall, and ETH
    fires too, which it does not when open interest is read a stamp stale."""
    day = date(2023, 8, 17)
    first = day - timedelta(days=29)
    start_values = Params(a=0.03, b=0.025, c=0.15, v=3.0)
    fired = {}
    for symbol in archive.SYMBOLS:
        columns = _real_columns(symbol, first, day)
        bar, _ = bars.bars(archive.day_ms(first), 30, *(columns[kind] for kind in archive.KINDS))
        found, windows = cascade.candidates(bar, start_values, since=29 * DAY)
        assert windows["windows_evaluated"] == DAY
        fired[symbol] = found[0]
        stale = {**bar, "oi": np.append(np.nan, bar["oi"][:-1])}
        fired[symbol, "stale"] = cascade.candidates(stale, start_values, since=29 * DAY)[0]
    btc, eth = fired["BTCUSDT"], fired["ETHUSDT"]
    assert _utc(btc["fired_ts_ns"]) == _utc(eth["fired_ts_ns"]) == "2023-08-17 21:45"
    assert (btc["kind"], eth["kind"]) == (LONG, LONG)
    assert btc["price_change"] == pytest.approx(-0.106, abs=0.001)
    assert btc["flow"] == pytest.approx(-0.303, abs=0.001)
    assert eth["flow"] < -0.15 and eth["price_change"] < -0.025 and eth["oi_change"] < -0.03
    assert fired["ETHUSDT", "stale"] == []
    assert fired["BTCUSDT", "stale"][0]["flow"] == pytest.approx(-0.183, abs=0.001)


def _taker_ratios(symbol: str, first: date, last: date) -> tuple[np.ndarray, np.ndarray]:
    """The metrics' taker ratio, a column archive.read leaves out, with its stamps."""
    stamps, ratios = [], []
    for d in range((last - first).days + 1):
        source = archive.Source("metrics", symbol, f"{first + timedelta(days=d):%Y-%m-%d}")
        with zipfile.ZipFile(ROOT / ARCHIVE / source.path) as bundle:
            table = pacsv.read_csv(pa.py_buffer(bundle.read(bundle.namelist()[0])))
        stamps.append(table["create_time"].to_numpy().astype("datetime64[ms]").astype(np.int64))
        ratios.append(table["sum_taker_long_short_vol_ratio"].to_numpy().astype(np.float64))
    return np.concatenate(stamps), np.concatenate(ratios)


@pytest.mark.parametrize(
    ("symbol", "first", "last"),
    [
        ("BTCUSDT", date(2023, 3, 1), date(2023, 3, 31)),
        ("ETHUSDT", date(2023, 11, 1), date(2023, 11, 30)),
    ],
    ids=["btc-2023-03", "eth-2023-11"],
)
def test_a_metrics_stamp_marks_the_end_of_its_step_on_real_days(
    symbol: str, first: date, last: date
) -> None:
    """What fixes the rule's alignment, on months with no documented cascade. The change in open
    interest between stamps t - 5 min and t correlates most with the 1-minute returns between them,
    not with the step after, as it would if the stamp marked the start of its step. And the taker
    ratio stamped t is the klines' over [t - 5 min, t), and no other step's. Minutes are labelled by
    their open."""
    minute = 60_000
    columns = _real_columns(symbol, first, last)
    klines, metrics = columns["klines"], columns["metrics"]
    start = int(klines["open_time"].min())
    returns = np.full(int(klines["open_time"].max() - start) // minute + 1, np.nan)
    returns[(klines["open_time"] - start) // minute] = np.abs(
        np.log(klines["close"] / klines["open"])
    )
    order = np.argsort(metrics["create_time"])
    stamps, oi = metrics["create_time"][order], metrics["sum_open_interest"][order]
    stamps, oi = stamps[oi > 0], oi[oi > 0]
    joined = np.diff(stamps) == 5 * minute
    change = np.abs(np.log(oi[1:] / oi[:-1]))[joined]
    at = (stamps[1:][joined] - start) // minute

    def correlation(offset: int) -> float:
        pick = at + offset
        keep = (pick >= 0) & (pick < len(returns))
        x, y = change[keep], returns[pick[keep]]
        return float(np.corrcoef(x[~np.isnan(y)], y[~np.isnan(y)])[0, 1])

    by_offset = {offset: correlation(offset) for offset in range(-10, 7)}
    assert set(sorted(by_offset, key=by_offset.__getitem__)[-5:]) <= set(range(-5, 1))
    between = np.mean([by_offset[m] for m in range(-5, 0)])
    assert between > np.mean([by_offset[m] for m in range(0, 5)]) + 0.05
    assert between > np.mean([by_offset[m] for m in range(-10, -5)]) + 0.05

    step = (klines["open_time"] - start) // (5 * minute)
    volume = np.bincount(step, weights=klines["volume"])
    buys = np.bincount(step, weights=klines["taker_buy_volume"])
    with np.errstate(divide="ignore", invalid="ignore"):
        kline_ratio = buys / (volume - buys)
    ratio_stamps, ratio = _taker_ratios(symbol, first, last)
    index = (ratio_stamps - start) // (5 * minute)

    def ratio_gap(steps: int) -> float:
        pick = index + steps
        keep = (pick >= 0) & (pick < len(kline_ratio))
        return float(np.nanmedian(np.abs(ratio[keep] - kline_ratio[pick[keep]])))

    gaps = {steps: ratio_gap(steps) for steps in (-3, -2, -1, 0, 1, 2)}
    assert gaps[-1] < 0.001  # the step ending at the stamp; a median, for the few outlier rows
    assert all(gap > 0.05 for steps, gap in gaps.items() if steps != -1)


def test_documented_cascades_on_real_days(capsys: pytest.CaptureFixture[str]) -> None:
    """The recall check on real days kept locally (gitignored, harness/fixtures/archives/binance).

    It reads confirmatory dates: five of the six events lie from 2024-01-01 on, and that span stays
    closed until market-labels-v1 is frozen (#29 S5). So it runs only with TP_CASCADE_RECALL set
    and otherwise skips, and it reads no rule output but each documented event's own days.

    It is a report, and nothing is to be tuned on it. It asserts that the check ran and printed,
    and no recall floor: a floor would invite moving a, b, c or v until it passes, and thresholds
    chosen so would have seen the confirmatory span, which is what the split exists to prevent.
    The pilot sets them on 2023 alone; a documented event they miss is a finding for the paper.

    `TP_CASCADE_RECALL=1 uv run pytest python/tipping_point/tests/market/test_cascade.py -k
    documented_cascades_on_real_days`, after fetching each event's days and the 29 before them.
    """
    root = ROOT / ARCHIVE
    if not os.environ.get(cascade.SWITCH):
        pytest.skip(f"documented-event recall not run: set {cascade.SWITCH} to read {root}")
    rows = cascade.recall(root)
    read = [r for r in rows if r["recalled"] is not None]
    if not read:
        pytest.skip(f"documented-event recall not run: no documented event's days in {root}")
    with capsys.disabled():
        print()
        for row in rows:
            print(f"{row['name']:<11} recalled {row['recalled']}")
            for symbol, found in row["symbols"].items():
                if found["status"] == "read":
                    first = found["first"]
                    print(
                        f"  {symbol:<8} {found['candidates']} candidates, "
                        f"{found['windows_evaluated']} of {found['windows']} windows evaluated"
                        + (
                            f", first {first['kind']} at {_utc(first['fired_ts_ns'])}"
                            if first
                            else ""
                        )
                    )
                else:
                    print(f"  {symbol:<8} {found['status']}: {found['error']}")
        print(f"recall {sum(bool(r['recalled']) for r in read)} of {len(read)} read")
    assert [r["name"] for r in rows] == [e["name"] for e in cascade.documented()]
    for row, event in zip(rows, cascade.documented(), strict=True):
        start = archive.day_ms(event["first_day"]) * 1_000_000
        end = archive.day_ms(event["last_day"] + timedelta(days=1)) * 1_000_000
        for found in row["symbols"].values():
            if found["status"] == "read" and found["first"]:
                assert start <= found["first"]["fired_ts_ns"] < end
