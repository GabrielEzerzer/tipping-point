"""The label-count and stability reports (#29 S3; research pass section 4): every item the section
lists, by hand on a path small enough to count and on a synthetic fixture run through the real bars
and labeller; what an optional input changes; the committed release calendar; byte-for-byte
repeats; the confirmatory refusal."""

import csv
import hashlib
import importlib.resources
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from market_planted import regimes
from tipping_point.market import archive, bars, msvol, nulls, reports

FIRST = date(2022, 7, 4)  # a Monday
START_MS = archive.day_ms(FIRST)
DAYS = 70
PARAMS = msvol.Params(warmup_days=28, baseline_days=14)
SYMBOLS = ("BTCUSDT", "ETHUSDT")
SHARED_FROM = 49 * 288  # the planted paths coincide over the last three weeks
GAP = 72  # BTC loses six hours of klines in a calm stretch of its labelled span
MINUTE_NS = 60 * 10**9


@dataclass(frozen=True)
class Fixture:
    series: dict[str, tuple[dict[str, np.ndarray], msvol.Labels]]
    parkinson: dict[str, np.ndarray]
    states: dict[str, np.ndarray]
    gap: slice


def _klines(states: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Klines of the 10-second returns of a planted path: log volatility up 0.6 in the high state,
    with a daily sine and quieter weekends, as market_planted plants them."""
    steps = states.repeat(30)
    t = np.arange(len(steps))
    day = t // nulls.DAY_STEPS
    weekend = (START_MS // 86_400_000 + day + 3) % 7 >= 5
    log_vol = -9.0 + 0.6 * steps + 0.25 * np.sin(2 * np.pi * t / nulls.DAY_STEPS) - 0.15 * weekend
    return nulls.minute_klines(START_MS, rng.normal(size=len(steps)) * np.exp(log_vol))


def _build(days: int, seed: int, symbols: tuple[str, ...] = SYMBOLS) -> Fixture:
    rng = np.random.default_rng(seed)
    n = days * 288
    btc = regimes(n, rng)
    eth = regimes(n, rng)
    eth[SHARED_FROM:] = btc[SHARED_FROM:]
    states = {"BTCUSDT": btc, "ETHUSDT": eth}
    empty = nulls._NO_METRICS, nulls._NO_FUNDING
    calm = np.convolve(btc, np.ones(600), "valid") == 0  # calm from i to i + 600
    start = int(np.flatnonzero(calm[35 * 288 :])[0]) + 35 * 288 + 200
    gap = slice(start, start + GAP)
    series, parkinson = {}, {}
    for symbol in symbols:
        klines = _klines(states[symbol], rng)
        if symbol == "BTCUSDT":
            keep = np.ones(5 * n, bool)
            keep[5 * gap.start : 5 * gap.stop] = False
            klines = {name: column[keep] for name, column in klines.items()}
        bar, _ = bars.bars(START_MS, days, klines, *empty)
        series[symbol] = (bar, msvol.label(bar["ts_ns"], bar["rv"], PARAMS))
        parkinson[symbol] = reports.parkinson_rv(START_MS, days, klines)
    return Fixture(series, parkinson, states, gap)


@pytest.fixture(scope="module")
def fixture() -> Fixture:
    return _build(DAYS, 29)


@pytest.fixture(scope="module")
def counted(fixture: Fixture) -> reports.Report:
    return reports.label_count(fixture.series)


@pytest.fixture(scope="module")
def stable(fixture: Fixture) -> reports.Report:
    return reports.stability(fixture.series, PARAMS, fixture.parkinson, seed=29)


def _rows(report: reports.Report, table: str, **where: Any) -> list[dict[str, Any]]:
    rows = report.tables[table].to_pylist()
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def _sections(summary: str) -> dict[str, str]:
    parts = summary.split("\n## ")
    return {p.split("\n", 1)[0]: p.split("\n", 1)[1] for p in parts[1:]}


# ---------------------------------------------------------------- the label-count report, by hand


HAND = msvol.Params(d_min=3, clean=4, refractory=10)
HAND_START = archive.day_ms(date(2023, 1, 30)) * 1_000_000  # a Monday
HAND_BARS = 10 * 288
HAND_TS = HAND_START + msvol.BAR_NS * np.arange(HAND_BARS, dtype=np.int64)


def _hand_symbol(runs: list[tuple[int, int]], gap: slice | None = None) -> Any:
    """Bars 0-287 (Monday) are unlabelled; a NaN rv makes a bar missing."""
    rv = np.ones(HAND_BARS)
    if gap:
        rv[gap] = np.nan
    h = np.zeros(HAND_BARS, np.int8)
    h[:288] = -1
    for a, b in runs:
        h[a:b] = 1
    missing = np.isnan(rv)
    found = msvol.episodes(h, missing, HAND)
    nan = np.full(HAND_BARS, np.nan)
    labels = msvol.Labels(
        nan, nan, nan, h, np.full(HAND_BARS, -1), [], found, msvol.nulls(h, missing, found, HAND)
    )
    return {"ts_ns": HAND_TS, "rv": rv}, labels


@pytest.fixture(scope="module")
def hand() -> dict[str, Any]:
    btc = [(300, 310), (400, 402), (600, 606), (608, 612), (2878, 2880)]
    eth = [(306, 316), (612, 620), (1200, 1210)]
    return {"BTCUSDT": _hand_symbol(btc, slice(1000, 1060)), "ETHUSDT": _hand_symbol(eth)}


def test_the_monthly_table_counts_by_hand(hand: dict[str, Any]) -> None:
    kinds = [e["kind"] for e in hand["BTCUSDT"][1].episodes]
    assert kinds == ["event", "flicker", "event", "episode", "pending"]
    report = reports.label_count(hand)
    january, february = _rows(report, "monthly", symbol="BTCUSDT")
    assert january == {
        "symbol": "BTCUSDT",
        "month": "2023-01",
        "labelled_bars": 288,  # Tuesday the 31st; Monday is unlabelled
        "missing_bars": 0,
        "share_high": pytest.approx(12 / 288),
        "events": 1,
        "episodes": 0,
        "flickers": 1,
        "pending": 0,
        "event_bars_median": 10.0,
        "event_bars_max": 10,
        "episode_bars_median": None,
        "episode_bars_max": None,
        "flicker_bars_median": 2.0,
    }
    assert february == {
        "symbol": "BTCUSDT",
        "month": "2023-02",
        "labelled_bars": HAND_BARS - 576,
        "missing_bars": 60,
        "share_high": pytest.approx(12 / (HAND_BARS - 576)),
        "events": 1,
        "episodes": 1,
        "flickers": 0,
        "pending": 1,
        "event_bars_median": 6.0,
        "event_bars_max": 6,
        "episode_bars_median": 4.0,
        "episode_bars_max": 4,
        "flicker_bars_median": None,
    }


def test_onsets_are_clocked_by_utc_hour_and_weekday_by_hand(hand: dict[str, Any]) -> None:
    report = reports.label_count(hand)
    rows = _rows(report, "onset_clock", symbol="BTCUSDT")
    hours = {r["value"]: (r["events"], r["onsets"]) for r in rows if r["unit"] == "hour"}
    days = {r["value"]: (r["events"], r["onsets"]) for r in rows if r["unit"] == "weekday"}
    assert set(hours) == set(range(24)) and set(days) == set(range(7))
    # 01:00 and 09:20 on Tuesday the 31st; 02:00 and 02:40 on Wednesday the 1st; 23:50 on the 8th
    assert {k: v for k, v in hours.items() if v != (0, 0)} == {
        1: (1, 1),
        2: (1, 2),
        9: (0, 1),
        23: (0, 1),
    }
    assert {k: v for k, v in days.items() if v != (0, 0)} == {1: (1, 2), 2: (1, 3)}
    assert "by weekday Mon-Sun: 0 2 3 0 0 0 0" in report.summary


def test_coincidence_counts_onsets_within_an_hour_either_way_by_hand(hand: dict[str, Any]) -> None:
    report = reports.label_count(hand)
    events, onsets = report.tables["coincidence"].to_pylist()
    # BTC events 300, 600; ETH events 306 (30 minutes on), 612 (the hour exactly), 1200
    assert events == {
        "symbol_a": "BTCUSDT",
        "symbol_b": "ETHUSDT",
        "population": "events",
        "count_a": 2,
        "count_b": 3,
        "a_near_b": 2,
        "b_near_a": 2,
    }
    # BTC's other onsets, 400, 608 and 2878: 608 lies within the hour of 612
    assert onsets["population"] == "onsets"
    assert (onsets["count_a"], onsets["count_b"]) == (5, 3)
    assert (onsets["a_near_b"], onsets["b_near_a"]) == (3, 2)


def test_null_hours_and_four_hour_slices_by_hand(hand: dict[str, Any]) -> None:
    report = reports.label_count(hand)
    stretches = [(s["start"], s["end"]) for s in hand["BTCUSDT"][1].nulls]
    assert stretches == [(320, 396), (406, 590), (622, 2868)]
    (row,) = _rows(report, "nulls", symbol="BTCUSDT")
    assert row == {
        "symbol": "BTCUSDT",
        "stretches": 3,
        "bars": 76 + 184 + 2246,
        "hours": pytest.approx(2506 / 12),
        "missing_bars": 60,
        "slices_4h": 1 + 3 + 46,
        "slices_4h_gapless": 50 - 3,  # the gap at 1000-1060 touches 958-1006, 1006-1054, 1054-1102
    }
    assert "209 null hours hold 50 4-hour slices (47 without a missing bar)" in report.summary


def test_scheduled_releases_are_counted_within_15_minutes_by_hand(hand: dict[str, Any]) -> None:
    calendar = {
        "CPI": np.array([HAND_TS[300] + 15 * MINUTE_NS, HAND_TS[2000], HAND_TS[0]]),
        "FOMC": np.array([HAND_TS[600] + 16 * MINUTE_NS]),
        "NFP": np.empty(0, np.int64),
    }
    report = reports.label_count(hand, releases=calendar)
    btc = {r["kind"]: r for r in _rows(report, "releases", symbol="BTCUSDT")}
    assert list(btc) == ["CPI", "FOMC", "NFP"]
    assert btc["CPI"]["releases_in_span"] == 2  # the third falls on the unlabelled Monday
    assert (btc["CPI"]["events_near"], btc["CPI"]["onsets_near"]) == (1, 1)  # 15 minutes is near
    assert (btc["FOMC"]["events_near"], btc["FOMC"]["onsets_near"]) == (0, 0)  # 16 is not
    assert btc["NFP"]["releases_in_span"] == 0
    assert _rows(report, "releases", symbol="ETHUSDT", kind="CPI")[0]["events_near"] == 1  # 15 on
    section = _sections(report.summary)
    title = "Onsets within +-15 minutes of scheduled releases (descriptive, no hypothesis)"
    assert "| CPI | 2 | 1 | 1 |" in section[title].replace("| BTCUSDT ", "")
    assert _rows(report, "inputs", item="releases")[0]["status"] == "supplied"


def test_cascade_candidates_overlap_regime_onsets_by_hand(hand: dict[str, Any]) -> None:
    candidates = np.array(
        [HAND_TS[300] + 60 * MINUTE_NS, HAND_TS[400] + 61 * MINUTE_NS, HAND_TS[1500]]
    )
    report = reports.label_count(hand, cascades={"BTCUSDT": candidates})
    assert report.tables["cascades"].to_pylist() == [
        {
            "symbol": "BTCUSDT",  # a symbol without candidates has no row
            "candidates": 3,
            "candidates_near_event": 1,  # the hour exactly is near
            "candidates_near_onset": 1,  # 61 minutes from the flicker at 400 is not
            "events": 2,
            "events_near_candidate": 1,
        }
    ]
    assert _rows(report, "inputs", item="cascades")[0]["status"] == "supplied"
    assert _rows(report, "inputs", item="releases")[0]["status"] == reports.NOT_SUPPLIED


# ---------------------------------------------------------------- the label-count report, whole


def test_without_its_inputs_the_label_count_report_says_not_supplied(
    counted: reports.Report,
) -> None:
    assert {r["item"]: r["status"] for r in counted.tables["inputs"].to_pylist()} == {
        "releases": reports.NOT_SUPPLIED,
        "cascades": reports.NOT_SUPPLIED,
    }
    assert counted.tables["releases"].num_rows == counted.tables["cascades"].num_rows == 0
    section = _sections(counted.summary)
    title = "Onsets within +-15 minutes of scheduled releases (descriptive, no hypothesis)"
    assert section[title].strip() == reports.NOT_SUPPLIED
    title = "Cascade candidates and regime onsets within +-1 hour"
    assert section[title].strip() == reports.NOT_SUPPLIED


def test_the_label_count_report_adds_up_on_the_fixture(
    fixture: Fixture, counted: reports.Report
) -> None:
    months = counted.tables["monthly"].to_pylist()
    assert sorted({r["month"] for r in months}) == ["2022-08", "2022-09"]  # the labelled span
    for symbol, (_, out) in fixture.series.items():
        own = [m for m in months if m["symbol"] == symbol]
        kinds = [e["kind"] for e in out.episodes]
        assert kinds.count("event") > 0 and kinds.count("flicker") > 0
        for kind, column in (("event", "events"), ("flicker", "flickers"), ("episode", "episodes")):
            assert sum(m[column] for m in own) == kinds.count(kind)
        assert sum(m["labelled_bars"] for m in own) == int(np.sum(out.h >= 0))
        high = sum(m["share_high"] * m["labelled_bars"] for m in own)
        assert high == pytest.approx(np.sum(out.h == 1))
        clock = _rows(counted, "onset_clock", symbol=symbol)
        for unit in ("hour", "weekday"):
            assert sum(r["events"] for r in clock if r["unit"] == unit) == kinds.count("event")
            assert sum(r["onsets"] for r in clock if r["unit"] == unit) == len(kinds)
    assert sum(m["missing_bars"] for m in months if m["symbol"] == "BTCUSDT") == GAP
    assert sum(m["missing_bars"] for m in months if m["symbol"] == "ETHUSDT") == 0


def test_planted_coincidence_shows_on_the_fixture(counted: reports.Report) -> None:
    """The paths coincide over the last three weeks only; independent paths seldom would."""
    events = _rows(counted, "coincidence", population="events")[0]
    assert events["a_near_b"] >= 4 and events["b_near_a"] >= 4
    assert events["a_near_b"] < events["count_a"] and events["b_near_a"] < events["count_b"]


def test_the_null_table_holds_slices_for_30(fixture: Fixture, counted: reports.Report) -> None:
    for symbol, (_, out) in fixture.series.items():
        (row,) = _rows(counted, "nulls", symbol=symbol)
        assert row["stretches"] == len(out.nulls) > 0
        assert row["hours"] == pytest.approx(sum(s["bars"] for s in out.nulls) / 12)
        assert row["slices_4h"] == sum(s["bars"] // 48 for s in out.nulls) > 0
        assert row["slices_4h_gapless"] <= row["slices_4h"]


# ---------------------------------------------------------------- the committed release calendar

KINDS = ("CPI", "FOMC", "NFP")
EASTERN = {"CPI": (8, 30), "NFP": (8, 30), "FOMC": (14, 0)}  # the release slots, Eastern time


def _ns(text: str) -> int:
    return int(datetime.fromisoformat(text).timestamp()) * 10**9


def _utc_of(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns // 10**9, UTC)


def _eastern_offset(day: date) -> int:
    """Hours behind UTC: US daylight time runs from the second Sunday of March to the first Sunday
    of November (no release here falls on a Sunday, so the 2 a.m. change needs no hour)."""

    def sunday(month: int, nth: int) -> date:
        first = date(day.year, month, 1)
        return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (nth - 1))

    return 4 if sunday(3, 2) <= day < sunday(11, 1) else 5


def test_the_committed_calendar_holds_18_cpi_18_nfp_and_12_fomc_times() -> None:
    calendar = reports.release_calendar()
    assert tuple(calendar) == KINDS  # the names label_count's `releases` takes
    assert {kind: len(times) for kind, times in calendar.items()} == {
        "CPI": 18,
        "FOMC": 12,
        "NFP": 18,
    }
    first, last = _ns("2022-07-01T00:00:00Z"), _ns("2024-01-01T00:00:00Z")
    for kind, times in calendar.items():
        assert times.dtype == np.int64
        assert np.all(np.diff(times) > 0), kind  # sorted, no repeats
        assert first <= times[0] and times[-1] < last, kind
        for ts in times.tolist():
            when = _utc_of(ts)
            assert ts % (60 * 10**9) == 0 and when.weekday() < 5, (kind, when)
            local = when - timedelta(hours=_eastern_offset(when.date()))
            assert (local.hour, local.minute) == EASTERN[kind], (kind, when)
    # first, last and the clock changes, as the research record gives them
    for kind, text in [
        ("CPI", "2022-07-13T12:30:00Z"),
        ("CPI", "2022-11-10T13:30:00Z"),  # first EST row
        ("CPI", "2023-03-14T12:30:00Z"),  # first EDT row
        ("CPI", "2023-12-12T13:30:00Z"),
        ("NFP", "2022-07-08T12:30:00Z"),
        ("NFP", "2022-11-04T12:30:00Z"),  # still EDT
        ("NFP", "2023-03-10T13:30:00Z"),  # still EST
        ("NFP", "2023-12-08T13:30:00Z"),
        ("FOMC", "2022-07-27T18:00:00Z"),
        ("FOMC", "2022-12-14T19:00:00Z"),
        ("FOMC", "2023-11-01T18:00:00Z"),  # the time read off the statement page
        ("FOMC", "2023-12-13T19:00:00Z"),
    ]:
        assert _ns(text) in calendar[kind], (kind, text)
    assert calendar["CPI"][0] == _ns("2022-07-13T12:30:00Z")
    assert calendar["FOMC"][-1] == _ns("2023-12-13T19:00:00Z")


def test_every_calendar_row_names_its_primary_source() -> None:
    with reports.CALENDAR.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["kind", "time_utc", "source"] and len(rows) == 48
    for row in rows:
        year = row["time_utc"][:4]
        if row["kind"] == "FOMC":
            assert row["source"] in {
                f"https://www.federalreserve.gov/newsevents/pressreleases/{year}-press-fomc.htm",
                "https://www.federalreserve.gov/newsevents/pressreleases/monetary20231101a.htm",
            }, row
        else:
            assert row["source"] == f"https://www.bls.gov/schedule/{year}/home.htm", row


def test_the_calendar_refuses_a_time_that_is_not_utc(tmp_path: Path) -> None:
    path = tmp_path / "releases.csv"
    head = "kind,time_utc,source\n"
    path.write_text(head + "CPI,2022-07-13T12:30:00+00:00,x\nNFP,2022-07-08T12:30:00Z,x\n")
    found = {kind: times.tolist() for kind, times in reports.release_calendar(path).items()}
    assert found == {"CPI": [_ns("2022-07-13T12:30:00Z")], "NFP": [_ns("2022-07-08T12:30:00Z")]}
    for stamp in ("2022-07-13T08:30:00-04:00", "2022-07-13T12:30:00"):
        path.write_text(head + f"CPI,{stamp},x\n")
        with pytest.raises(ValueError, match="is not UTC"):
            reports.release_calendar(path)


def test_the_calendar_is_read_from_the_installed_package(tmp_path: Path) -> None:
    """Through the package, not the checkout: from a folder outside the repository, by a fresh
    interpreter, as a kernel installing the wheel reads it."""
    packaged = importlib.resources.files("tipping_point.market") / "releases.csv"
    assert Path(str(packaged)) == reports.CALENDAR and packaged.is_file()
    code = (
        "from tipping_point.market import reports\n"
        "print({k: len(v) for k, v in reports.release_calendar().items()})"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "{'CPI': 18, 'FOMC': 12, 'NFP': 18}"


def test_the_calendar_fills_the_releases_table_on_the_fixture(fixture: Fixture) -> None:
    calendar = reports.release_calendar()
    report = reports.label_count(fixture.series, releases=calendar)
    rows = report.tables["releases"].to_pylist()
    assert [(r["symbol"], r["kind"]) for r in rows] == [(s, k) for s in SYMBOLS for k in KINDS]
    window = reports.RELEASE_NS
    for row in rows:
        bar, out = fixture.series[row["symbol"]]
        ts, times = bar["ts_ns"], calendar[row["kind"]]
        labelled = ts[out.h >= 0]
        inside = (times >= labelled[0]) & (times <= labelled[-1])
        assert row["releases_in_span"] == int(inside.sum())
        for population, kind in (("events_near", "event"), ("onsets_near", None)):
            at = ts[[e["onset"] for e in out.episodes if kind is None or e["kind"] == kind]]
            near = np.abs(at[:, None] - times[None, :]).min(axis=1) <= window
            assert row[population] == int(near.sum()), (row, population)
    # the labelled span, 2022-08-01 to 2022-09-11, holds CPI on 08-10 and payrolls on 08-05 and
    # 09-02; the FOMC statements of 07-27 and 09-21 fall outside it
    spans = {(r["symbol"], r["kind"]): r["releases_in_span"] for r in rows}
    assert spans == {(s, k): n for s in SYMBOLS for k, n in zip(KINDS, (1, 0, 2), strict=True)}
    assert _rows(report, "inputs", item="releases")[0]["status"] == "supplied"
    section = _sections(report.summary)
    title = "Onsets within +-15 minutes of scheduled releases (descriptive, no hypothesis)"
    assert reports.NOT_SUPPLIED not in section[title]
    assert "| BTCUSDT | CPI | 1 |" in section[title] and "| ETHUSDT | NFP | 2 |" in section[title]
    # where the times come from, under the heading and before the table (#142)
    source, table = section[title].strip().split("\n\n")
    assert source == (
        "Calendar: market/releases.csv; 11 of the 12 FOMC times are the standard 14:00 Eastern "
        "slot, not read off each statement page."
    )
    assert table.startswith("| symbol | kind |")
    caveat = " ".join((reports.release_calendar.__doc__ or "").split())
    assert source.split("; ")[1] in caveat  # the caveat the calendar's reader states


# ---------------------------------------------------------------- the stability report's parts


def test_kappa_by_hand() -> None:
    a = np.array([True, True, False, False])
    assert reports.kappa(a, np.array([True, False, False, False])) == pytest.approx(0.5)
    assert reports.kappa(a, a) == 1.0
    assert reports.kappa(a, ~a) == -1.0
    assert math.isnan(reports.kappa(np.zeros(3, bool), np.zeros(3, bool)))
    assert math.isnan(reports.kappa(np.empty(0, bool), np.empty(0, bool)))
    rng = np.random.default_rng(1)
    far = reports.kappa(rng.uniform(size=20_000) < 0.2, rng.uniform(size=20_000) < 0.2)
    assert abs(far) < 0.03  # independent paths agree only by chance


def test_onsets_match_one_to_one_within_the_tolerance() -> None:
    a, b = np.array([10, 20, 30]), np.array([11, 13, 28, 100])
    assert reports.match_onsets(a, b, 3) == [(10, 11), (30, 28)]
    assert reports.match_onsets(np.array([10, 11]), np.array([11]), 3) == [(10, 11)]
    assert reports.match_onsets(np.array([10]), np.array([7, 13]), 3) == [(10, 7)]
    assert reports.match_onsets(np.array([10]), np.array([14]), 3) == []
    assert reports.match_onsets(np.empty(0), np.array([1]), 3) == []


def test_the_plain_rule_by_hand() -> None:
    z = np.array([9.0, 2, 2, 2, -1, 2, 2, -1, 2, np.nan, 2, 2, 2, -1])
    h = np.zeros(len(z), np.int8)
    h[0] = -1
    sd = float(np.std(z[1:][~np.isnan(z[1:])]))
    assert 0.5 * sd < 2  # so 2 is above the line and -1 is not
    out = reports.plain_path(z, h, 0.5, 3)
    # the first bar is unlabelled; the run at 5-6 and the one at 8 (a NaN breaks it) are too short
    assert out.tolist() == [-1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    assert reports.plain_path(np.full(4, np.nan), np.zeros(4, np.int8), 0.5, 1).tolist() == [0] * 4


def test_parkinson_rv_by_hand_and_with_the_bars_missing_pattern() -> None:
    minutes = 288 * 5
    high, low = 101.0, 99.0
    klines = {
        "open_time": START_MS + 60_000 * np.arange(minutes, dtype=np.int64),
        "open": np.full(minutes, 100.0),
        "high": np.full(minutes, high),
        "low": np.full(minutes, low),
        "close": np.full(minutes, 100.0),
        "count": np.full(minutes, 10, np.int64),
        "volume": np.ones(minutes),
        "taker_buy_volume": np.full(minutes, 0.5),
    }
    klines["count"][7] = 0  # bar 1: a minute without trades adds nothing
    klines["count"][15:20] = 0  # bar 3: no trades at all: missing
    keep = np.arange(minutes) != 12  # bar 2: a minute gone: missing
    klines = {name: column[keep] for name, column in klines.items()}
    rv = reports.parkinson_rv(START_MS, 1, klines)
    one = math.log(high / low) ** 2 / (4 * math.log(2))
    assert rv[0] == pytest.approx(5 * one) and rv[1] == pytest.approx(4 * one)
    assert math.isnan(rv[2]) and math.isnan(rv[3]) and rv[4] == pytest.approx(5 * one)
    bar, _ = bars.bars(START_MS, 1, klines, nulls._NO_METRICS, nulls._NO_FUNDING)
    assert np.array_equal(np.isnan(rv), np.isnan(bar["rv"]))


def _y(rv: np.ndarray) -> np.ndarray:
    return np.where(rv > 0, np.log(np.where(rv > 0, rv, 1.0)), np.nan)


def test_the_repeated_refit_loop_is_the_labellers_bit_for_bit(planted: Any) -> None:
    """relabelled_p repeats msvol.label's loop; on the real slots the two must not differ at all."""
    slot = msvol._slots(planted.ts)
    p = reports.relabelled_p(planted.ts, _y(planted.rv), slot, planted.params)
    assert p.tobytes() == planted.full.p.tobytes()


def test_the_repeated_refit_loop_keeps_the_theta_before_a_degenerate_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build(49, 5, ("ETHUSDT",))
    bar, _ = fixture.series["ETHUSDT"]
    real, calls = msvol.fit, []

    def fit(z: np.ndarray, theta: msvol.Theta, tol: float, max_iter: int) -> msvol.Fit:
        calls.append(theta)
        result = real(z, theta, tol, max_iter)
        if len(calls) % 3 == 2:  # the second refit of each run comes back degenerate
            return result._replace(theta=result.theta._replace(sigma1=0.01))
        return result

    monkeypatch.setattr(msvol, "fit", fit)
    out = msvol.label(bar["ts_ns"], bar["rv"], PARAMS)
    assert [r["flagged"] for r in out.refits] == [False, True, False]
    slot = msvol._slots(bar["ts_ns"])
    p = reports.relabelled_p(bar["ts_ns"], _y(bar["rv"]), slot, PARAMS)
    assert p.tobytes() == out.p.tobytes()


def test_one_slot_for_every_bar_is_no_deseasonalisation(planted: Any) -> None:
    y, days = _y(planted.rv), PARAMS.baseline_days
    r = 40 * 288
    _, z = msvol._features(y, np.zeros(len(y), np.int64), r, r + 288, days)
    window = y[r - days * 288 : r]
    assert np.allclose(
        z[r : r + 288], y[r : r + 288] - np.nanmedian(window), rtol=0, atol=1e-9, equal_nan=True
    )


def _smoothed(z: np.ndarray, t: msvol.Theta) -> list[float]:
    """P(high | all of z) by a plain forward-backward pass, written out apart from msvol's: a
    missing bar updates nothing, and the chain starts stationary."""
    filtered, predicted = [], []
    a = (1 - t.p00) / (2 - t.p00 - t.p11)
    for value in z.tolist():
        b = (1 - t.p00) * (1 - a) + t.p11 * a
        predicted.append(b)
        if not math.isnan(value):
            e0 = math.exp(-0.5 * ((value - t.mu0) / t.sigma0) ** 2) / t.sigma0
            e1 = math.exp(-0.5 * ((value - t.mu1) / t.sigma1) ** 2) / t.sigma1
            a = b * e1 / (b * e1 + (1 - b) * e0)
        else:
            a = b
        filtered.append(a)
    out = filtered[:]
    for i in range(len(z) - 2, -1, -1):
        high = out[i + 1] / predicted[i + 1]
        low = (1 - out[i + 1]) / (1 - predicted[i + 1])
        keep_high = filtered[i] * (t.p11 * high + (1 - t.p11) * low)
        keep_low = (1 - filtered[i]) * ((1 - t.p00) * high + t.p00 * low)
        out[i] = keep_high / (keep_high + keep_low)
    return out


def test_the_oracle_is_the_final_thetas_full_sample_smoother(fixture: Fixture) -> None:
    bar, out = fixture.series["ETHUSDT"]
    y, ts = _y(bar["rv"]), bar["ts_ns"]
    theta = nulls.final_theta(out)
    assert theta is not None
    last = out.refits[-1]["bar"]
    _, z = msvol._features(y, msvol._slots(ts), last, len(y), PARAMS.baseline_days)
    want = msvol.hysteresis(np.array(_smoothed(z, theta)), PARAMS.enter, PARAMS.exit)
    want[out.h < 0] = -1
    assert reports.oracle_path(ts, y, out, PARAMS).tobytes() == want.tobytes()


def test_the_smoothed_oracle_leads_the_as_of_labels_and_reads_the_plant_better(
    fixture: Fixture, stable: reports.Report
) -> None:
    for symbol, (bar, out) in fixture.series.items():
        oracle = reports.oracle_path(bar["ts_ns"], _y(bar["rv"]), out, PARAMS)
        labelled = out.h >= 0
        assert np.array_equal(oracle < 0, ~labelled)
        truth = fixture.states[symbol][labelled] == 1
        as_of = np.mean((out.h[labelled] == 1) == truth)
        smoothed = np.mean((oracle[labelled] == 1) == truth)
        assert smoothed > as_of > 0.97
        shifts = [r["shift_bars"] for r in _rows(stable, "onset_shifts", symbol=symbol)]
        assert sum(shifts) > 0 and min(shifts) >= -1  # the filter comes after the smoother


# ---------------------------------------------------------------- the stability report, whole


def test_every_declared_variant_has_its_row(stable: reports.Report) -> None:
    want = {
        "oracle",
        "d_min=6",
        "d_min=36",
        "thresholds=0.7/0.3",
        "thresholds=0.9/0.1",
        "baseline_days=28",
        "baseline_days=56",
        "parkinson",
        "no_deseasonalisation",
        "plain_z>0.5sd_for_12_bars",
    }
    for symbol in SYMBOLS:
        rows = _rows(stable, "agreement", symbol=symbol)
        assert {r["against"] for r in rows} == want and len(rows) == len(want)
        for row in rows:
            assert row["matched"] <= min(row["onsets"], row["onsets_other"])
    assert "| BTCUSDT | no_deseasonalisation |" in stable.summary


def test_the_as_of_labels_follow_the_oracle(stable: reports.Report) -> None:
    for symbol in SYMBOLS:
        (row,) = _rows(stable, "agreement", symbol=symbol, against="oracle")
        assert row["agreement"] > 0.98 and row["kappa"] > 0.9
        assert row["matched"] >= 0.9 * row["onsets"] > 0
        assert 0 <= row["shift_median"] <= 2  # the filter comes after the smoother, not before it
        shifts = _rows(stable, "onset_shifts", symbol=symbol)
        assert len(shifts) == row["matched"]
        for shift in shifts:
            assert (shift["ts_ns"] - shift["oracle_ts_ns"]) / msvol.BAR_NS == shift["shift_bars"]
            assert abs(shift["shift_bars"]) <= reports.ONSET_TOLERANCE
        assert f"- {symbol}: as-of labels agree with the oracle" in stable.summary


def test_the_variants_move_the_labels_as_they_should(stable: reports.Report) -> None:
    for symbol in SYMBOLS:
        rows = {r["against"]: r for r in _rows(stable, "agreement", symbol=symbol)}
        row = rows.__getitem__
        oracle = row("oracle")
        assert row("d_min=6")["agreement"] == row("d_min=36")["agreement"] == 1.0  # H is unchanged
        assert row("d_min=36")["onsets_other"] <= oracle["onsets"]
        assert 0.99 < row("thresholds=0.7/0.3")["agreement"] < 1.0
        assert 0.99 < row("thresholds=0.9/0.1")["agreement"] < 1.0
        # another range estimator: nearly the same labels, not the same
        assert row("parkinson")["kappa"] > 0.9 and row("parkinson")["agreement"] < 1.0
        # the labels do not hinge on the HMM
        assert row("plain_z>0.5sd_for_12_bars")["kappa"] > 0.8
        # the fixture plants a daily sine: without its slots the labels follow the clock instead
        assert row("no_deseasonalisation")["kappa"] < 0.5
        # a longer baseline starts later: 28 days loses its first refit, 56 all but the last ones
        first = oracle["compared_bars"]
        assert row("baseline_days=56")["compared_bars"] < row("baseline_days=28")["compared_bars"]
        assert row("baseline_days=28")["compared_bars"] < first
        assert row("baseline_days=56")["agreement"] > 0.9
        # only the bars both label are compared, so only their onsets are matched
        assert 0 < row("baseline_days=56")["onsets"] < oracle["onsets"]


def test_the_theta_path_holds_durations_and_boundary_jumps(
    fixture: Fixture, stable: reports.Report
) -> None:
    for symbol, (bar, out) in fixture.series.items():
        rows = _rows(stable, "refits", symbol=symbol)
        assert len(rows) == len(out.refits)
        for row, refit in zip(rows, out.refits, strict=True):
            r = refit["bar"]
            assert row["ts_ns"] == bar["ts_ns"][r] and row["flagged"] is False
            assert row["low_duration_bars"] == pytest.approx(1 / (1 - refit["p00"]))
            assert row["high_duration_bars"] == pytest.approx(1 / (1 - refit["p11"]))
            assert row["mu1"] == refit["mu1"]
            assert row["p_before"] == (None if r == 0 or np.isnan(out.p[r - 1]) else out.p[r - 1])
        assert rows[0]["jump"] is None  # no theta before the first
        assert "| 2022-08-01 00:00 | " + symbol in stable.summary  # the first refit, in UTC
        for row, refit in zip(rows[1:], out.refits[1:], strict=True):
            # one filter step on from P refiltered at the bar before must land on P at the refit
            r = refit["bar"]
            theta = nulls.theta_of(refit)
            e0, e1, _ = msvol._emissions(out.z[r : r + 1], theta)
            b1 = (1 - theta.p00) + (theta.p00 + theta.p11 - 1) * row["p_refiltered"]
            next_p = b1 * e1[0] / (b1 * e1[0] + (1 - b1) * e0[0])
            assert next_p == pytest.approx(out.p[r], abs=1e-9)
            assert row["jump"] == pytest.approx(row["p_refiltered"] - row["p_before"])
            assert abs(row["jump"]) < 0.5  # weekly refits barely move a settled theta


def test_a_flagged_refit_shows_in_the_report(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _build(49, 7, ("ETHUSDT",))
    bar, _ = fixture.series["ETHUSDT"]
    real, calls = msvol.fit, []

    def fit(z: np.ndarray, theta: msvol.Theta, tol: float, max_iter: int) -> msvol.Fit:
        calls.append(theta)
        result = real(z, theta, tol, max_iter)
        if len(calls) == 2:
            return result._replace(theta=result.theta._replace(sigma1=0.01))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(msvol, "fit", fit)
        out = msvol.label(bar["ts_ns"], bar["rv"], PARAMS)
    series = {"ETHUSDT": (bar, out)}
    report = reports.stability(series, PARAMS, fixture.parkinson, seed=1)
    first, second, third = report.tables["refits"].to_pylist()
    assert second["flagged"] and second["reason"] == "sigma 0.01 < 0.05"
    assert second["sigma1"] == 0.01  # the row records what the fit gave
    assert not first["flagged"] and not third["flagged"]
    assert "1 of 3 refits flagged" in report.summary and "sigma 0.01 < 0.05" in report.summary
    # the theta in force did not change, only the z it reads: the jump is measured all the same
    assert second["p_refiltered"] is not None


def test_the_delay_distribution_is_reported_under_the_fitted_theta(
    fixture: Fixture, stable: reports.Report
) -> None:
    for symbol, (_, out) in fixture.series.items():
        every, longer = _rows(stable, "delay", symbol=symbol)
        assert (every["population"], longer["population"]) == ("all", "at_least_d_min")
        assert every["episodes"] > longer["episodes"] > 100
        assert 0 <= longer["median"] <= longer["p90"] <= longer["max"] <= 12
        theta = nulls.final_theta(out)
        assert theta is not None and theta == nulls.theta_of(out.refits[-1])
        assert f"p90 detection delay {longer['p90']:.0f} bars" in stable.summary


def test_without_aggtrades_the_report_says_not_supplied(stable: reports.Report) -> None:
    assert stable.tables["aggtrades"].num_rows == 0
    assert stable.tables["inputs"].to_pylist() == [
        {"item": "aggtrades", "status": reports.NOT_SUPPLIED}
    ]
    section = _sections(stable.summary)["aggTrades 10-second RV against Garman-Klass"]
    assert section.strip() == reports.NOT_SUPPLIED


def test_with_aggtrades_the_report_compares_the_estimators() -> None:
    fixture = _build(42, 8)
    rng = np.random.default_rng(8)
    bar, _ = fixture.series["BTCUSDT"]
    other = bar["rv"] * np.exp(0.2 + 0.1 * rng.normal(size=len(bar["rv"])))
    report = reports.stability(
        fixture.series, PARAMS, fixture.parkinson, seed=1, aggtrades={"BTCUSDT": other}
    )
    (row,) = report.tables["aggtrades"].to_pylist()  # ETH was not supplied, so it has no row
    assert row["symbol"] == "BTCUSDT" and row["bars"] == 42 * 288 - GAP
    assert row["log_corr"] > 0.9 and row["median_log_ratio"] == pytest.approx(0.2, abs=0.01)
    assert report.tables["inputs"].to_pylist() == [{"item": "aggtrades", "status": "supplied"}]
    assert reports.NOT_SUPPLIED not in report.summary
    with pytest.raises(ValueError, match="aggTrades bars"):
        reports.stability(
            fixture.series, PARAMS, fixture.parkinson, seed=1, aggtrades={"BTCUSDT": other[:-1]}
        )


def test_a_series_with_no_labels_reports_as_much() -> None:
    ts = START_MS * 1_000_000 + msvol.BAR_NS * np.arange(35 * 288, dtype=np.int64)
    rv = np.full(len(ts), np.nan)
    out = msvol.label(ts, rv, msvol.Params(warmup_days=14, baseline_days=7))
    assert np.all(out.h == -1) and all(r["flagged"] for r in out.refits)
    series = {"BTCUSDT": ({"ts_ns": ts, "rv": rv}, out)}
    counted = reports.label_count(series)
    assert counted.tables["monthly"].num_rows == 0
    assert _rows(counted, "nulls")[0]["slices_4h"] == 0
    assert "0 events" in counted.summary and "of 0 labelled bars" in counted.summary
    params = msvol.Params(warmup_days=14, baseline_days=7)
    report = reports.stability(series, params, {"BTCUSDT": rv}, seed=1)
    assert report.tables["delay"].num_rows == 0
    for row in report.tables["agreement"].to_pylist():
        assert row["compared_bars"] == 0 and row["agreement"] is None and row["kappa"] is None
    assert "no fitted theta, so no detection delay" in report.summary
    assert report.tables["refits"]["jump"].null_count == report.tables["refits"].num_rows


# ---------------------------------------------------------------- what both reports share


def test_the_reports_write_parquet_and_a_summary(
    tmp_path: Path, counted: reports.Report, stable: reports.Report
) -> None:
    for report, schemas in ((counted, reports.LABEL_COUNT), (stable, reports.STABILITY)):
        folder = reports.write(report, tmp_path / "out")
        assert folder == tmp_path / "out" / report.name
        files = sorted(p.name for p in folder.iterdir())
        assert files == sorted([*(f"{name}.parquet" for name in schemas), "summary.md"])
        for name, schema in schemas.items():
            table = pq.read_table(folder / f"{name}.parquet")
            assert table.schema.equals(schema, check_metadata=False), name
            assert table.equals(report.tables[name]), name
        assert (folder / "summary.md").read_text(encoding="utf-8") == report.summary
        assert report.summary.startswith("# ") and report.summary.endswith("\n")
        with pytest.raises(FileExistsError):
            reports.write(report, tmp_path / "out")


def _digests(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}


def test_two_runs_of_each_report_are_identical_byte_for_byte(
    tmp_path: Path, counted: reports.Report, stable: reports.Report
) -> None:
    again = _build(DAYS, 29)  # from the seed up: the fixture, the labels and both reports
    runs = [
        (counted, stable),
        (
            reports.label_count(again.series),
            reports.stability(again.series, PARAMS, again.parkinson, seed=29),
        ),
    ]
    digests = []
    for k, reports_run in enumerate(runs):
        digests.append([_digests(reports.write(r, tmp_path / str(k))) for r in reports_run])
    assert digests[0] == digests[1]
    assert len(digests[0][0]) == len(reports.LABEL_COUNT) + 1
    assert len(digests[0][1]) == len(reports.STABILITY) + 1


def test_the_delay_simulation_is_the_only_thing_the_seed_moves() -> None:
    small = _build(42, 9, ("ETHUSDT",))
    a = reports.stability(small.series, PARAMS, small.parkinson, seed=1)
    b = reports.stability(small.series, PARAMS, small.parkinson, seed=2)
    assert a.tables["delay"].num_rows == 2 and not a.tables["delay"].equals(b.tables["delay"])
    for name in set(reports.STABILITY) - {"delay"}:
        assert a.tables[name].equals(b.tables[name]), name
    # so the report names its seed (#137): with the run's labeller, enough to repeat it
    for report, seed in ((a, 1), (b, 2)):
        line = report.summary.split("\n\n")[2]
        assert line.startswith("Labeller parameters: ")
        assert line.endswith(f". The delay simulation drew from seed = {seed}.")


def _ending_at(last: np.ndarray) -> dict[str, Any]:
    bar = {"ts_ns": last, "rv": np.ones(len(last))}
    nan = np.full(len(last), np.nan)
    h = np.full(len(last), -1, np.int8)
    return {"BTCUSDT": (bar, msvol.Labels(nan, nan, nan, h, np.full(len(last), -1), [], [], []))}


def test_nothing_is_read_from_the_confirmatory_span() -> None:
    boundary = archive.day_ms(date(2024, 1, 1)) * 1_000_000
    before = _ending_at(boundary - msvol.BAR_NS * np.arange(3, 0, -1, dtype=np.int64))
    after = _ending_at(boundary - msvol.BAR_NS * np.arange(2, -1, -1, dtype=np.int64))
    assert reports.label_count(before).tables["nulls"].num_rows == 1
    with pytest.raises(ValueError, match="market-labels-v1"):
        reports.label_count(after)
    with pytest.raises(ValueError, match="market-labels-v1"):
        reports.stability(after, PARAMS, {"BTCUSDT": np.ones(3)}, seed=1)
