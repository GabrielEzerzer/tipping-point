# The labels on planted bars and a fake archive: cascade.py's windows, trailing daily thresholds,
# day-by-day detection from before the grid, the grid's label, skip and censor edges, the run.

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from market_planted import vision
from tipping_point.market import archive, bars, cascade, msvol
from tipping_point.panel import labels, span
from tipping_point.panel.__main__ import main

BAR = msvol.BAR_NS
HOUR = labels.HOUR_NS
DAY = bars.DAY_BARS
DAY_NS = labels.DAY_NS
T0 = labels.ns(date(2022, 1, 1))
MINUTE = 60 * 10**9


def planted(days: int = 40, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = days * DAY
    return {
        "ts_ns": T0 + BAR * np.arange(n, dtype=np.int64),
        "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))),
        "volume": 100 * np.exp(rng.normal(0, 0.3, n)),
        "taker_buy_share": rng.uniform(0.4, 0.6, n),
        "oi": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.001, n))),
    }


def plant(
    columns: dict[str, np.ndarray], day: int, hour: int, side: int = -1, minute: int = 0
) -> None:
    first = day * DAY + hour * 12 + minute // 5
    for i in range(first, first + 6):
        columns["close"][i:] *= np.exp(side * 0.01)
        columns["oi"][i:] *= np.exp(-0.005)
        columns["volume"][i] *= 10
        columns["taker_buy_share"][i] = 0.5 + side * 0.45


def gapped() -> dict[str, np.ndarray]:
    columns = planted()
    columns["oi"][31 * DAY + 5] = np.nan
    columns["oi"][33 * DAY] = -1.0
    columns["close"][34 * DAY + 100 : 34 * DAY + 103] = np.nan
    columns["volume"][2 * DAY : 3 * DAY] = np.nan
    return columns


def test_the_windows_are_the_ones_cascade_reads() -> None:
    columns = gapped()
    win = labels.windows(columns)
    k = np.arange(cascade.Params().window + 1, len(columns["ts_ns"]))
    assert np.array_equal(win["ts_ns"], columns["ts_ns"][k])
    with np.errstate(invalid="ignore"):
        expected = np.log(columns["close"][k - 1] / columns["close"][k - 7])
    assert np.array_equal(win["price_change"], expected, equal_nan=True)
    oi = np.where(columns["oi"] > 0, columns["oi"], np.nan)
    with np.errstate(invalid="ignore"):
        expected = np.log(oi[k] / oi[k - 6])
    assert np.array_equal(win["oi_change"], expected, equal_nan=True)
    for since in (0, 29 * DAY, 35 * DAY):
        _, counts = cascade.candidates(columns, cascade.Params(), since)
        assert counts["windows_evaluated"] == int(win["evaluated"][k >= since].sum())
        assert counts["windows"] == int((k >= since).sum())


def test_the_evaluated_mask_matches_cascade_window_by_window_at_the_gaps() -> None:
    columns = gapped()
    mask = labels.windows(columns)["evaluated"]
    first = cascade.Params().window + 1
    flips = np.flatnonzero(np.diff(mask)) + first
    assert flips.size >= 7
    near = np.unique(np.concatenate([flips + d for d in range(-3, 5)]))
    near = near[(near >= first) & (near < len(columns["ts_ns"]) - 1)]

    def evaluated(since: int) -> int:
        return cascade.candidates(columns, cascade.Params(), int(since))[1]["windows_evaluated"]

    counts = {s: evaluated(s) for s in np.unique(np.concatenate([near, near + 1]))}
    expected = np.array([counts[s] - counts[s + 1] for s in near], bool)
    assert expected.any() and not expected.all()
    assert np.array_equal(mask[near - first], expected)
    assert not np.array_equal(np.roll(mask, 1)[near - first], expected)
    assert not np.array_equal(np.roll(mask, -1)[near - first], expected)


def test_a_close_of_zero_or_below_is_refused() -> None:
    columns = planted(days=2)
    columns["close"][100] = 0.0
    with pytest.raises(ValueError, match="zero or below"):
        labels.windows(columns)


def test_a_days_threshold_reads_every_known_change_of_its_90_days() -> None:
    columns = planted(days=100)
    columns["oi"][70 * DAY : 71 * DAY] = np.nan
    win = labels.windows(columns)
    rows = labels.daily(win, T0, 100)
    counts = np.array([r["windows"] for r in rows])
    has = np.array([r["b"] is not None for r in rows])
    assert np.array_equal(has, counts >= 28 * DAY * 0.8)
    first = int(np.argmax(has))
    assert 40 < first < 60 and has[first:].all()
    since = win["ts_ns"][win["evaluated"]][0]
    for d in (first, 75, 99):
        day = T0 + d * DAY_NS
        closing = (win["ts_ns"] >= max(day - 90 * DAY_NS, since)) & (win["ts_ns"] < day)
        ret, oi = (
            win[name][closing & np.isfinite(win[name])] for name in ("price_change", "oi_change")
        )
        assert rows[d]["windows"] == int((closing & win["evaluated"]).sum())
        assert rows[d]["oi_unknown"] == int(closing.sum()) - len(oi)
        assert rows[d]["sd_ret"] == float(np.std(ret, ddof=1))
        assert rows[d]["sd_oi"] == float(np.std(oi, ddof=1))
        assert (rows[d]["a"], rows[d]["b"]) == (3.7 * rows[d]["sd_oi"], 5.0 * rows[d]["sd_ret"])
    assert rows[first]["oi_unknown"] == 0 and rows[99]["oi_unknown"] == DAY + 6
    assert rows[99]["windows"] < 90 * DAY - DAY


def test_a_days_threshold_forgets_the_windows_closing_before_its_90_days() -> None:
    columns = planted(days=130)
    win = labels.windows(columns)
    d = 125
    edge = int(np.searchsorted(win["ts_ns"], T0 + (d - 90) * DAY_NS))
    assert (
        win["ts_ns"][edge] == T0 + (d - 90) * DAY_NS and win["evaluated"][edge - 1 : edge + 1].all()
    )
    row = labels.daily(win, T0, 130)[d]
    assert row["windows"] == 90 * DAY
    assert row["sd_ret"] == float(np.std(win["price_change"][edge : edge + 90 * DAY], ddof=1))
    for i, windows in ((edge, 90 * DAY - 1), (edge - 1, 90 * DAY)):
        off = {**win, "evaluated": win["evaluated"].copy()}
        off["evaluated"][i] = False
        assert labels.daily(off, T0, 130)[d]["windows"] == windows


def test_a_flat_history_gives_no_threshold_so_a_volume_burst_is_no_cascade() -> None:
    for flat in (("close",), ("oi",)):
        columns = planted(days=60)
        for name in flat:
            columns[name][:] = 100.0
        rows = labels.daily(labels.windows(columns), T0, 60)
        assert rows[55]["windows"] >= labels.MIN_WINDOWS and rows[55]["missing"] == 0
        assert all(r["b"] is None for r in rows), flat
    columns = planted(days=60)
    columns["close"][:], columns["oi"][:] = 100.0, 1e6
    burst = slice(55 * DAY + 144, 55 * DAY + 150)
    columns["volume"][burst] *= 10
    columns["taker_buy_share"][burst] = 0.05
    zero = cascade.Params(a=0.0, b=0.0, c=0.15, v=3.0)
    assert cascade.candidates(columns, zero, 55 * DAY)[0]
    thresholds, found, _, coin = labels.coin(columns, T0 + 60 * DAY_NS)
    assert found == [] and all(r["b"] is None for r in thresholds)
    got = labels.grid(T0 + 55 * DAY_NS + HOUR * np.arange(24, dtype=np.int64), coin, [])
    assert all(got[f"censored_{h}"].all() for h in labels.HORIZONS)


def test_a_threshold_needs_95_percent_of_its_windows_since_the_first_evaluated_one() -> None:
    at = T0 + 95 * DAY_NS + 12 * HOUR
    for gap, kept in ((2, True), (4, False)):
        columns = planted(days=100)
        columns["oi"][60 * DAY : (60 + gap) * DAY] = np.nan
        win = labels.windows(columns)
        since = win["ts_ns"][win["evaluated"]][0]
        thresholds, _, _, coin = labels.coin(columns, T0 + 100 * DAY_NS)
        row = thresholds[95]
        assert row["windows"] >= labels.MIN_WINDOWS and row["missing"] == gap * DAY + 6
        closing = (win["ts_ns"] >= since) & (win["ts_ns"] < T0 + 95 * DAY_NS)
        assert row["windows"] + row["missing"] == int(closing.sum())
        assert (labels.missing_share(row) <= 0.05) == kept == (row["b"] is not None)
        on = (coin.windows_ts >= T0 + 95 * DAY_NS) & (coin.windows_ts < T0 + 96 * DAY_NS)
        assert coin.clean[on].all() == kept and coin.clean[on].any() == kept
        got = labels.grid(np.array([at], np.int64), coin, [])
        assert all(got[f"censored_{h}"][0] != kept for h in labels.HORIZONS)


def test_a_threshold_counts_the_known_changes_of_windows_the_rule_could_not_evaluate() -> None:
    columns = planted(days=100)
    gap = slice(60 * DAY, 62 * DAY)
    steps = np.diff(np.log(columns["close"]), prepend=0.0)
    steps[gap] *= 20
    columns["close"] = np.exp(np.cumsum(steps))
    columns["oi"][gap] = np.nan
    lone = 80 * DAY + 100
    columns["oi"][lone] = np.nan
    columns["oi"][lone + 1 :] *= np.exp(-0.3)
    win = labels.windows(columns)
    row = labels.daily(win, T0, 100)[95]
    since = win["ts_ns"][win["evaluated"]][0]
    closing = (win["ts_ns"] >= since) & (win["ts_ns"] < T0 + 95 * DAY_NS)
    assert row["missing"] == 2 * DAY + 6 + 7 and row["oi_unknown"] == 2 * DAY + 6 + 2
    assert labels.missing_share(row) <= labels.MAX_MISSING
    for name, leg in (("price_change", "sd_ret"), ("oi_change", "sd_oi")):
        known = closing & np.isfinite(win[name])
        assert row[leg] == float(np.std(win[name][known], ddof=1))
        assert row[leg] > 2 * np.std(win[name][closing & win["evaluated"]], ddof=1)
    carried = closing & ~win["evaluated"] & (np.abs(win["oi_change"]) > 0.2)
    assert int(carried.sum()) == 5


def test_open_interest_that_starts_late_is_not_counted_missing() -> None:
    columns = planted(days=100)
    columns["oi"][: 40 * DAY] = np.nan
    win = labels.windows(columns)
    since = int(win["ts_ns"][win["evaluated"]][0])
    assert T0 + 40 * DAY_NS < since < T0 + 41 * DAY_NS
    rows = labels.daily(win, T0, 100)
    assert all(r["missing"] == 0 for r in rows)
    first = next(d for d, r in enumerate(rows) if r["b"] is not None)
    assert rows[first - 1]["windows"] < labels.MIN_WINDOWS <= rows[first]["windows"]
    assert all(r["b"] is not None for r in rows[first:])


def test_a_days_threshold_does_not_change_when_that_day_or_later_changes() -> None:
    columns = planted(days=70)
    base = labels.daily(labels.windows(columns), T0, 70)
    rng = np.random.default_rng(5)
    on = columns["ts_ns"] >= T0 + 60 * DAY_NS
    changed = {name: values.copy() for name, values in columns.items()}
    for name in ("close", "oi", "volume"):
        changed[name][on] *= np.exp(rng.normal(0, 0.05, int(on.sum())))
    changed["oi"][on & (rng.uniform(size=len(on)) < 0.01)] = np.nan
    again = labels.daily(labels.windows(changed), T0, 70)
    assert again[:61] == base[:61]
    assert again[61] != base[61]


def test_day_by_day_detection_equals_one_call_when_the_threshold_is_constant() -> None:
    columns = gapped()
    for day, hour in ((30, 6), (31, 23), (35, 12), (38, 0)):
        plant(columns, day, hour)
    plant(columns, 36, 9, side=1)
    p = cascade.Params(a=0.01, b=0.02, c=0.15, v=3.0)
    since = 29
    whole, whole_counts = cascade.candidates(columns, p, since * DAY)
    found, counts, per_day = labels.by_day(columns, [None] * since + [p] * (40 - since))
    assert len(whole) >= 5 and {r["kind"] for r in whole} == {cascade.LONG, cascade.SHORT}
    assert found == whole
    assert counts == whole_counts
    assert per_day.sum() == whole_counts["windows_evaluated"] and not per_day[:since].any()


def fired_day(row: dict[str, Any]) -> int:
    return (row["fired_ts_ns"] - T0) // DAY_NS


def test_each_day_finds_only_what_its_own_rule_admits() -> None:
    columns = planted()
    for day in range(30, 38):
        plant(columns, day, 12)
    strict = cascade.Params(a=0.01, b=1.0, c=0.15, v=3.0)
    loose = cascade.Params(a=0.01, b=0.02, c=0.15, v=3.0)
    rules = [None] * 29 + [[strict, loose][d % 2] for d in range(29, 40)]
    found, _, _ = labels.by_day(columns, rules)
    whole, _ = cascade.candidates(columns, loose, 29 * DAY)
    assert {fired_day(r) for r in whole} == set(range(30, 38))
    assert found == [r for r in whole if fired_day(r) % 2]


def test_a_window_across_midnight_takes_its_closing_days_rule_and_merges_across_it() -> None:
    columns = planted()
    plant(columns, 34, 23, minute=40)
    midnight = T0 + 35 * DAY_NS
    strict = cascade.Params(a=0.01, b=1.0, c=0.15, v=3.0)
    loose = cascade.Params(a=0.01, b=0.02, c=0.15, v=3.0)
    whole, _ = cascade.candidates(columns, loose, 29 * DAY)
    after = [r for r in whole if r["fired_ts_ns"] >= midnight]
    assert any(r["fired_ts_ns"] - labels.LEAD_NS < midnight < r["fired_ts_ns"] for r in after)
    assert len(after) < len(whole)
    for day_34, day_35, kept in ((strict, loose, after), (loose, strict, whole[: -len(after)])):
        rules = [None] * 29 + [loose] * 5 + [day_34, day_35] + [loose] * 4
        assert labels.by_day(columns, rules)[0] == kept
    (row,) = cascade.pool({"X": labels.by_day(columns, [None] * 29 + [loose] * 11)[0]})
    assert row["fired_ts_ns"] < midnight < row["last_fired_ts_ns"] and row["candidates"] >= 2


def thresholded(days: int = 60) -> dict[str, np.ndarray]:
    columns = planted(days=days)
    rows = labels.daily(labels.windows(columns), T0, days)
    assert rows[52]["b"] is not None
    return columns


def test_detection_starts_before_the_grid_so_an_earlier_episode_is_known() -> None:
    columns = thresholded()
    plant(columns, 55, 20)
    plant(columns, 56, 1)
    end = T0 + 60 * DAY_NS
    _, found, _, coin = labels.coin(columns, end)
    first = T0 + 56 * DAY_NS
    rows, table = labels.label({"X": found}, {"X": coin}, first, end)
    (row,) = rows
    assert not row["in_span"] and row["candidates"] >= 2
    assert row["fired_ts_ns"] < first <= row["last_fired_ts_ns"]
    at = table.slice(0, 1).to_pylist()[0]
    assert at["ts_ns"] == first and at["skip"] and not at["long_1"] and not at["long_6"]
    assert not any(table["long_24"].to_pylist())


def test_a_burst_raises_the_next_days_threshold_and_not_its_own() -> None:
    columns = planted(days=70)
    steps = np.zeros(70 * DAY)
    steps[60 * DAY : 60 * DAY + 144] = np.random.default_rng(7).normal(0, 0.06, 144)
    columns["close"] *= np.exp(np.cumsum(steps))
    plant(columns, 60, 18)
    plant(columns, 61, 12)
    thresholds, found, _, _ = labels.coin(columns, T0 + 70 * DAY_NS)
    assert thresholds[60]["b"] < 0.03 and thresholds[61]["b"] > 0.09
    assert {fired_day(r) for r in found} == {60}
    lax, _ = cascade.candidates(columns, labels.params(thresholds[60]), 61 * DAY)
    assert {fired_day(r) for r in lax} == {61}


def test_windows_before_the_first_day_with_a_threshold_are_not_clean() -> None:
    columns = thresholded()
    thresholds, _, _, coin = labels.coin(columns, T0 + 60 * DAY_NS)
    first = next(d for d, r in enumerate(thresholds) if r["b"] is not None)
    start = T0 + first * DAY_NS
    win = labels.windows(columns)
    before = win["ts_ns"] < start
    assert win["evaluated"][before].sum() > DAY
    assert not coin.clean[before].any() and coin.clean[~before].all()
    got = labels.grid(start + HOUR * np.arange(-1, 30, dtype=np.int64), coin, [])
    assert all(hours_where(got[f"censored_{h}"]) == list(range(25)) for h in labels.HORIZONS)


def test_a_short_squeeze_within_a_day_of_a_long_cascade_is_part_of_it() -> None:
    columns = thresholded()
    plant(columns, 55, 10)
    plant(columns, 55, 18, side=1)
    end = T0 + 60 * DAY_NS
    _, found, _, coin = labels.coin(columns, end)
    assert {r["kind"] for r in found} == {cascade.LONG, cascade.SHORT}
    rows, table = labels.label({"X": found}, {"X": coin}, T0 + 55 * DAY_NS, end)
    (row,) = rows
    assert row["kind"] == cascade.LONG and row["in_span"] and row["candidates"] >= 2
    assert any(table["long_1"].to_pylist())
    assert not any(any(table[f"short_{h}"].to_pylist()) for h in labels.HORIZONS)


def test_days_between_losing_and_regaining_a_threshold_detect_nothing_and_censor() -> None:
    columns = planted(days=130)
    columns["oi"][70 * DAY : 74 * DAY] = np.nan
    plant(columns, 90, 12)
    plant(columns, 120, 12)
    thresholds, found, _, coin = labels.coin(columns, T0 + 130 * DAY_NS)
    has = np.array([r["b"] is not None for r in thresholds])
    assert has[51:73].all() and not has[73:109].any() and has[109:].all()
    assert {fired_day(r) for r in found} == {120}
    win = labels.windows(columns)
    without = (win["ts_ns"] > T0 + 74 * DAY_NS + HOUR) & (win["ts_ns"] < T0 + 109 * DAY_NS)
    assert win["evaluated"][without].all() and not coin.clean[without].any()
    hours = T0 + HOUR * np.arange(90 * 24, 112 * 24, dtype=np.int64)
    got = labels.grid(hours, coin, [])
    for h in labels.HORIZONS:
        assert got[f"censored_{h}"][: 20 * 24].all() and not got[f"censored_{h}"][-24:].any()


def test_the_coin_reads_and_counts_nothing_from_the_end_on() -> None:
    columns = thresholded()
    plant(columns, 59, 12)
    end = T0 + 58 * DAY_NS
    long = labels.coin(columns, end)
    cut = labels.coin({name: values[: 58 * DAY] for name, values in columns.items()}, end)
    assert long[:3] == cut[:3]
    assert np.array_equal(long[3].windows_ts, cut[3].windows_ts)
    assert np.array_equal(long[3].clean, cut[3].clean)
    assert len(long[0]) == 58 and long[3].windows_ts.max() < end
    assert all(r["fired_ts_ns"] < end for r in long[1])


def test_a_missing_bar_in_the_trailing_history_censors_without_changing_detection() -> None:
    whole = planted(days=90)
    plant(whole, 75, 12)
    columns = {name: values.copy() for name, values in whole.items()}
    m = 60 * DAY + 144
    columns["close"][m] = np.nan
    end = T0 + 90 * DAY_NS
    win = labels.windows(columns)
    _, found, _, coin = labels.coin(columns, end)
    i, j = np.searchsorted(coin.windows_ts, columns["ts_ns"][[m + 8075, m + 8076]])
    assert win["evaluated"][i] and win["history_gap"][i] and not coin.clean[i]
    assert coin.clean[j] and not win["history_gap"][j]
    at = T0 + 75 * DAY_NS + 12 * HOUR
    assert found == labels.coin(whole, end)[1]
    assert {(r["fired_ts_ns"] - at) // HOUR for r in found} == {0}
    got = labels.grid(np.array([at - HOUR, at], np.int64), coin, cascade.pool({"X": found}))
    assert got["long_1"].any() and all(got[f"censored_{h}"].all() for h in labels.HORIZONS)


def cascade_row(kind: str, begin: int, last_fired: int, onset: int | None = None) -> dict[str, Any]:
    fired = begin + labels.LEAD_NS
    return {
        "ts_ns": begin if onset is None else onset,
        "kind": kind,
        "fired_ts_ns": fired,
        "last_fired_ts_ns": max(last_fired, fired),
    }


def windows_from(
    last: int = T0 + 80 * HOUR, unclean: tuple[int, ...] = (), first: int = T0 - 25 * HOUR
) -> labels.Coin:
    closes = np.arange(first, last + BAR, BAR, dtype=np.int64)
    return labels.Coin(closes, ~np.isin(closes, np.array(unclean, np.int64)))


HOURS = T0 + HOUR * np.arange(48, dtype=np.int64)


def hours_where(values: np.ndarray) -> list[int]:
    return np.flatnonzero(values).tolist()


def test_a_label_counts_window_starts_in_t_to_t_plus_h() -> None:
    rows = [
        cascade_row(cascade.LONG, T0 + 5 * HOUR, T0 + 5 * HOUR + 30 * MINUTE, T0 + 5 * HOUR),
        cascade_row(cascade.SHORT, T0 + 10 * HOUR + 25 * MINUTE, T0 + 11 * HOUR),
    ]
    got = labels.grid(HOURS, windows_from(), rows)
    assert hours_where(got["long_1"]) == [5]
    assert hours_where(got["long_6"]) == [0, 1, 2, 3, 4, 5]
    assert hours_where(got["long_24"]) == [0, 1, 2, 3, 4, 5]
    assert hours_where(got["short_1"]) == [10]
    assert hours_where(got["short_6"]) == [5, 6, 7, 8, 9, 10]
    assert hours_where(got["short_24"]) == list(range(11))


def test_an_hour_between_a_window_start_and_its_firing_is_an_ordinary_row() -> None:
    begin, onset = T0 + 19 * HOUR + 40 * MINUTE, T0 + 19 * HOUR + 50 * MINUTE
    rows = [cascade_row(cascade.LONG, begin, begin + labels.LEAD_NS, onset)]
    got = labels.grid(HOURS, windows_from(), rows)
    assert hours_where(got["long_1"]) == [19]
    assert not got["skip"][20] and not got["long_1"][20] and not got["long_24"][20]
    assert got["skip"][21]
    on_the_hour = [cascade_row(cascade.LONG, T0 + 19 * HOUR + 30 * MINUTE, 0)]
    assert labels.grid(HOURS, windows_from(), on_the_hour)["skip"][20]


def test_an_instance_is_dated_by_its_first_window_not_by_its_onset() -> None:
    begin, onset = T0 + 19 * HOUR + 45 * MINUTE, T0 + 20 * HOUR + 5 * MINUTE
    got = labels.grid(HOURS, windows_from(), [cascade_row(cascade.LONG, begin, 0, onset)])
    assert got["long_1"][19] and not got["long_1"][20]


def test_skip_runs_from_the_first_firing_to_a_day_after_the_last() -> None:
    rows = [
        cascade_row(cascade.LONG, T0 + 5 * HOUR, T0 + 5 * HOUR + 30 * MINUTE),
        cascade_row(cascade.SHORT, T0 + 30 * HOUR + 25 * MINUTE, T0 + 31 * HOUR),
    ]
    got = labels.grid(HOURS, windows_from(), rows)
    assert hours_where(got["skip"]) == list(range(6, 30)) + list(range(31, 48))
    edge = [cascade_row(cascade.LONG, T0 + 5 * HOUR, T0 + 6 * HOUR)]
    assert hours_where(labels.grid(HOURS, windows_from(), edge)["skip"]) == list(range(6, 31))


def test_censoring_reads_the_day_before_t_and_the_windows_starting_before_t_plus_h() -> None:
    got = labels.grid(HOURS, windows_from(unclean=(T0 + 20 * HOUR,)), [])
    assert hours_where(got["censored_1"]) == list(range(19, 45))
    assert hours_where(got["censored_6"]) == list(range(14, 45))
    assert hours_where(got["censored_24"]) == list(range(0, 45))
    found = [cascade_row(cascade.LONG, T0 + 30 * HOUR, 0)]
    assert labels.grid(HOURS, windows_from(unclean=(T0 + 20 * HOUR,)), found)["censored_1"][30]


def test_a_gap_55_minutes_before_t_censors_t() -> None:
    gap = windows_from(unclean=(T0 + 20 * HOUR - 55 * MINUTE,))
    got = labels.grid(HOURS, gap, [])
    assert all(got[f"censored_{h}"][20] for h in labels.HORIZONS)
    assert got["censored_1"][43] and not got["censored_1"][44]


def test_the_end_needs_exactly_the_last_window_starting_before_t_plus_h() -> None:
    late = labels.grid(HOURS, windows_from(unclean=(T0 + 20 * HOUR + 25 * MINUTE,)), [])
    assert hours_where(late["censored_1"])[0] == 19
    later = labels.grid(HOURS, windows_from(unclean=(T0 + 20 * HOUR + 30 * MINUTE,)), [])
    assert hours_where(later["censored_1"])[0] == 20
    enough = labels.grid(HOURS, windows_from(last=T0 + 20 * HOUR + 25 * MINUTE), [])
    assert hours_where(enough["censored_1"]) == list(range(20, 48))
    short = labels.grid(HOURS, windows_from(last=T0 + 20 * HOUR + 20 * MINUTE), [])
    assert hours_where(short["censored_1"]) == list(range(19, 48))
    young = labels.grid(HOURS, windows_from(first=T0 - 23 * HOUR), [])
    assert hours_where(young["censored_1"]) == [0]


def test_the_run_refuses_held_back_days(tmp_path: Path) -> None:
    for start, dev, end in (
        (date(2023, 10, 1), date(2023, 12, 1), date(2024, 1, 1)),
        (date(2023, 10, 1), date(2024, 1, 1), date(2024, 1, 1)),
    ):
        with pytest.raises(span.HeldBackError):
            labels.run(tmp_path, tmp_path, ["BTCUSDT"], start, dev, end, "x")
    assert not (tmp_path / "labels").exists()


SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIRST = date(2022, 11, 1)
DAYS = 61
SPAN = ["--start", "2022-11-01", "--dev-start", "2022-12-29", "--end", "2022-12-31"]


@pytest.fixture(scope="module")
def fetched(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fake = vision(SYMBOLS, FIRST, DAYS, seed=11)
    dest = tmp_path_factory.mktemp("archive")
    archive.fetch(SYMBOLS, FIRST, date(2022, 12, 31), dest, fake.get)
    return dest


def test_the_command_writes_a_labels_run(
    fetched: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    where = ["--archive", str(fetched), "--data-dir", str(tmp_path), "--symbols", *SYMBOLS]
    assert main(["labels", *SPAN, *where, "--end", "2024-01-01"]) == 1
    assert main(["labels", *SPAN, *where, "--dev-start", "2024-01-01"]) == 1
    assert main(["labels", *SPAN, *where, "--dev-start", "2022-10-01"]) == 1
    assert "error: the span must run" in capsys.readouterr().err
    assert not (tmp_path / "labels").exists()
    assert main(["labels", *SPAN, *where, "BTCUSDT", "--tag", "t"]) == 0
    (run,) = (tmp_path / "labels").glob("*-t")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    assert manifest["tables"] == ["thresholds", "cascades", "grid"] and complete["clean"]
    params = manifest["params"]
    assert params["symbols"] == SYMBOLS
    assert (params["start"], params["dev_start"], params["end"]) == (
        "2022-11-01",
        "2022-12-29",
        "2022-12-31",
    )
    assert params["constants"] == labels.constants()
    assert params["constants"]["sd_oi_multiple"] == 3.7
    assert (
        params["code"]["uv_lock_sha256"]
        == hashlib.sha256(
            (Path(__file__).resolve().parents[4] / "uv.lock").read_bytes()
        ).hexdigest()
    )
    assert params["code"]["head"] == manifest["producer"]["git_sha"]
    assert len(manifest["sources"]) == 2 * (2 * DAYS + 2)

    tables = {n: pq.read_table(next((run / n).glob("*.parquet"))) for n in manifest["tables"]}
    assert tables["thresholds"].schema.equals(labels.THRESHOLDS, check_metadata=False)
    assert tables["cascades"].schema.equals(labels.CASCADES, check_metadata=False)
    assert tables["grid"].schema.equals(labels.GRID, check_metadata=False)
    thresholds = tables["thresholds"].to_pylist()
    assert len(thresholds) == 2 * DAYS
    assert [r["symbol"] for r in thresholds[:2]] == SYMBOLS
    assert thresholds[0]["ts_ns"] == labels.ns(FIRST) and thresholds[0]["b"] is None
    assert all(r["b"] is not None for r in thresholds[-2 * 8 :])
    grid = tables["grid"]
    assert grid.num_rows == 2 * 72
    assert grid["ts_ns"].to_pylist()[:2] == [labels.ns(date(2022, 12, 29))] * 2
    assert grid["symbol"].to_pylist()[:2] == SYMBOLS
    last = grid.slice(grid.num_rows - 2)
    assert all(last[f"censored_{h}"].to_pylist() == [True, True] for h in labels.HORIZONS)
    assert not any(grid["censored_1"].to_pylist()[:-2])
    checks = complete["checks"]
    assert checks["grid"] == labels.shares(grid)
    files = json.loads((fetched / "manifest.json").read_text(encoding="utf-8"))["files"]
    for symbol in SYMBOLS:
        counted = checks[symbol]["row_checks"]
        assert (counted["klines"]["files"], counted["metrics"]["files"]) == (DAYS, DAYS)
        assert counted["fundingRate"]["files"] == 2
        assert counted["klines"]["rows"] == sum(
            e["checks"]["rows"] for p, e in files.items() if p.startswith(f"klines/{symbol}/")
        )
        assert counted["metrics"]["off_grid"] == 0
    out = capsys.readouterr().out
    assert str(run) in out and "market events" in out


def test_each_day_detects_with_its_own_threshold(fetched: Path, tmp_path: Path) -> None:
    _, checks = labels.run(
        fetched, tmp_path, SYMBOLS, FIRST, date(2022, 12, 29), date(2022, 12, 31), "own"
    )
    for symbol in SYMBOLS:
        got = archive.collect([symbol], FIRST, date(2022, 12, 31), fetched)
        columns, _ = bars.bars(
            archive.day_ms(FIRST),
            DAYS,
            *(got.columns[(symbol, kind)] for kind in archive.KINDS),
        )
        rows = labels.daily(labels.windows(columns), labels.ns(FIRST), DAYS)
        found = 0
        for d, row in enumerate(rows):
            p = labels.params(row)
            if p is None:
                continue
            day = labels.ns(FIRST) + d * DAY_NS
            some, _ = cascade.candidates(columns, p, d * DAY)
            found += sum(day <= r["fired_ts_ns"] < day + DAY_NS for r in some)
        assert checks[symbol]["candidates"] == found
        assert checks[symbol]["days_with_threshold"] == sum(r["b"] is not None for r in rows)
