# The labels on planted bars and a fake archive: cascade.py's windows, thresholds from their span
# only, the grid's label, skip and censor edges, the run's layout and its held-back refusal.

import dataclasses
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
T0 = labels.ns(date(2022, 1, 1))


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


def test_the_windows_are_the_ones_cascade_reads() -> None:
    columns = planted()
    columns["oi"][31 * DAY + 5] = np.nan
    columns["oi"][33 * DAY] = -1.0
    columns["close"][34 * DAY + 100 : 34 * DAY + 103] = np.nan
    columns["volume"][2 * DAY : 3 * DAY] = np.nan
    win = labels.windows(columns)
    k = np.arange(cascade.Params().window + 1, len(columns["ts_ns"]))
    assert np.array_equal(win["ts_ns"], columns["ts_ns"][k])
    with np.errstate(invalid="ignore"):
        expected = np.log(columns["close"][k - 1] / columns["close"][k - 7])
    assert np.array_equal(win["price_change"], expected, equal_nan=True)
    for since in (0, 29 * DAY, 35 * DAY):
        _, counts = cascade.candidates(columns, cascade.Params(), since)
        assert counts["windows_evaluated"] == int(win["evaluated"][k >= since].sum())
        assert counts["windows"] == int((k >= since).sum())


def test_the_evaluated_mask_matches_cascade_window_by_window_at_the_gaps() -> None:
    columns = planted()
    columns["oi"][31 * DAY + 5] = np.nan
    columns["oi"][33 * DAY] = -1.0
    columns["close"][34 * DAY + 100 : 34 * DAY + 103] = np.nan
    columns["volume"][2 * DAY : 3 * DAY] = np.nan
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


def test_thresholds_read_only_the_windows_closing_in_their_span() -> None:
    columns = planted()
    first, end = T0 + 32 * DAY * BAR, T0 + 38 * DAY * BAR
    base = labels.thresholds(labels.windows(columns), first, end)
    win = labels.windows(columns)
    inside = win["evaluated"] & (win["ts_ns"] >= first) & (win["ts_ns"] < end)
    assert base["windows"] == int(inside.sum()) == 6 * DAY
    assert base["sd_ret"] == float(np.std(win["price_change"][inside], ddof=1))
    assert (base["a"], base["b"]) == (3.7 * base["sd_oi"], 5.0 * base["sd_ret"])
    rng = np.random.default_rng(9)
    ts = columns["ts_ns"]
    outside = (ts < first - 7 * BAR) | (ts >= end)
    changed = {name: values.copy() for name, values in columns.items()}
    for name in ("close", "oi"):
        changed[name][outside] *= np.exp(rng.normal(0, 0.05, int(outside.sum())))
    assert labels.thresholds(labels.windows(changed), first, end) == base
    changed["close"][ts == first - 7 * BAR] *= 1.01
    assert labels.thresholds(labels.windows(changed), first, end) != base


def test_too_few_windows_are_refused() -> None:
    columns = planted(days=10)
    with pytest.raises(ValueError, match="too few"):
        labels.thresholds(labels.windows(columns), T0, T0 + 10 * DAY * BAR)


MINUTE = 60 * 10**9


def cascade_row(kind: str, begin: int, last_fired: int, onset: int | None = None) -> dict[str, Any]:
    fired = begin + labels.LEAD_NS
    return {
        "ts_ns": begin if onset is None else onset,
        "kind": kind,
        "fired_ts_ns": fired,
        "last_fired_ts_ns": max(last_fired, fired),
    }


def coin(last_end_ns: int, unevaluated: tuple[int, ...] = (), hours: int = 60) -> labels.Coin:
    closes = T0 + BAR * np.arange(1, hours * 12 + 1, dtype=np.int64)
    evaluated = ~np.isin(closes, np.array(unevaluated, np.int64))
    return labels.Coin(closes, evaluated, last_end_ns)


HOURS = T0 + HOUR * np.arange(48, dtype=np.int64)
END = T0 + 48 * HOUR


def hours_where(values: np.ndarray) -> list[int]:
    return np.flatnonzero(values).tolist()


def test_a_label_counts_window_starts_in_t_to_t_plus_h() -> None:
    rows = [
        cascade_row(cascade.LONG, T0 + 5 * HOUR, T0 + 5 * HOUR + 30 * MINUTE, T0 + 5 * HOUR),
        cascade_row(cascade.SHORT, T0 + 10 * HOUR + 25 * MINUTE, T0 + 11 * HOUR),
    ]
    got = labels.grid(HOURS, coin(T0 + 60 * HOUR), END, rows)
    assert hours_where(got["long_1"]) == [5]
    assert hours_where(got["long_6"]) == [0, 1, 2, 3, 4, 5]
    assert hours_where(got["long_24"]) == [0, 1, 2, 3, 4, 5]
    assert hours_where(got["short_1"]) == [10]
    assert hours_where(got["short_6"]) == [5, 6, 7, 8, 9, 10]
    assert hours_where(got["short_24"]) == list(range(11))


def test_a_cascade_under_way_at_t_is_skipped_not_positive() -> None:
    begin, onset = T0 + 19 * HOUR + 50 * MINUTE, T0 + 20 * HOUR
    rows = [cascade_row(cascade.LONG, begin, begin + labels.LEAD_NS, onset)]
    got = labels.grid(HOURS, coin(T0 + 60 * HOUR), END, rows)
    assert hours_where(got["long_1"]) == [19]
    assert not got["long_1"][20] and not got["long_6"][20] and not got["long_24"][20]
    assert got["skip"][20] and not got["skip"][19]


def test_skip_runs_from_after_the_window_start_to_a_day_after_the_last_firing() -> None:
    rows = [
        cascade_row(cascade.LONG, T0 + 5 * HOUR, T0 + 5 * HOUR + 30 * MINUTE),
        cascade_row(cascade.SHORT, T0 + 30 * HOUR + 25 * MINUTE, T0 + 31 * HOUR),
    ]
    got = labels.grid(HOURS, coin(T0 + 60 * HOUR), END, rows)
    assert hours_where(got["skip"]) == list(range(6, 30)) + list(range(31, 48))
    edge = [cascade_row(cascade.LONG, T0 + 5 * HOUR, T0 + 6 * HOUR)]
    assert hours_where(labels.grid(HOURS, coin(T0 + 60 * HOUR), END, edge)["skip"]) == list(
        range(6, 31)
    )


def test_censoring_follows_the_windows_starting_in_t_to_t_plus_h_the_end_and_the_last_bar() -> None:
    got = labels.grid(HOURS, coin(T0 + 60 * HOUR, unevaluated=(T0 + 20 * HOUR,)), END, [])
    assert hours_where(got["censored_1"]) == [19, 47]
    assert hours_where(got["censored_6"]) == [14, 15, 16, 17, 18, 19, 42, 43, 44, 45, 46, 47]
    assert hours_where(got["censored_24"]) == list(range(0, 20)) + list(range(24, 48))
    straddling = coin(T0 + 60 * HOUR, unevaluated=(T0 + 20 * HOUR + 15 * MINUTE,))
    assert hours_where(labels.grid(HOURS, straddling, END, [])["censored_1"]) == [19, 47]
    got = labels.grid(HOURS, coin(T0 + 40 * HOUR), END, [])
    assert hours_where(got["censored_1"]) == list(range(39, 48))
    got = labels.grid(HOURS, coin(T0 + 60 * HOUR, hours=47), END, [])
    assert hours_where(got["censored_1"]) == [46, 47]


def test_the_run_refuses_held_back_days(tmp_path: Path) -> None:
    for start, dev, end in (
        (date(2023, 10, 1), date(2023, 12, 1), date(2024, 1, 1)),
        (date(2023, 10, 1), date(2024, 1, 1), date(2024, 1, 1)),
    ):
        with pytest.raises(span.HeldBackError):
            labels.run(tmp_path, tmp_path, ["BTCUSDT"], start, dev, end, "x")
    assert not (tmp_path / "labels").exists()


SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIRST = date(2022, 11, 28)
SPAN = ["--start", "2022-11-28", "--dev-start", "2022-12-29", "--end", "2022-12-31"]


@pytest.fixture(scope="module")
def fetched(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fake = vision(SYMBOLS, FIRST, 34, seed=11)
    dest = tmp_path_factory.mktemp("archive")
    archive.fetch(SYMBOLS, FIRST, date(2022, 12, 31), dest, fake.get)
    return dest


def test_the_command_writes_a_labels_run(
    fetched: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    where = ["--archive", str(fetched), "--data-dir", str(tmp_path), "--symbols", *SYMBOLS]
    assert main(["labels", *SPAN, *where, "--end", "2024-01-01"]) == 1
    assert main(["labels", *SPAN, *where, "--dev-start", "2024-01-01"]) == 1
    assert not (tmp_path / "labels").exists()
    assert main(["labels", *SPAN, *where, "--tag", "t"]) == 0
    (run,) = (tmp_path / "labels").glob("*-t")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    assert manifest["tables"] == ["thresholds", "cascades", "grid"] and complete["clean"]
    params = manifest["params"]
    assert params["symbols"] == SYMBOLS
    assert (params["start"], params["dev_start"], params["end"]) == (
        "2022-11-28",
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
    assert len(manifest["sources"]) == 2 * (2 * 34 + 2)

    tables = {n: pq.read_table(next((run / n).glob("*.parquet"))) for n in manifest["tables"]}
    assert tables["thresholds"].schema.equals(labels.THRESHOLDS, check_metadata=False)
    assert tables["cascades"].schema.equals(cascade.SCHEMA, check_metadata=False)
    assert tables["grid"].schema.equals(labels.GRID, check_metadata=False)
    assert tables["thresholds"]["symbol"].to_pylist() == SYMBOLS
    assert tables["thresholds"]["windows"].to_pylist() == [3 * DAY] * 2
    grid = tables["grid"]
    assert grid.num_rows == 2 * 72
    assert grid["ts_ns"].to_pylist()[:2] == [labels.ns(date(2022, 12, 29))] * 2
    assert grid["symbol"].to_pylist()[:2] == SYMBOLS
    last = grid.slice(grid.num_rows - 2)
    assert all(last[f"censored_{h}"].to_pylist() == [True, True] for h in labels.HORIZONS)
    assert not any(grid["censored_1"].to_pylist()[:-2])
    assert complete["checks"]["grid"] == labels.shares(grid)
    out = capsys.readouterr().out
    assert str(run) in out and "market events" in out


def test_the_run_uses_each_coins_own_threshold(fetched: Path, tmp_path: Path) -> None:
    run, checks = labels.run(
        fetched,
        tmp_path,
        SYMBOLS,
        FIRST,
        date(2022, 12, 29),
        date(2022, 12, 31),
        "own",
    )
    table = pq.read_table(next((run / "thresholds").glob("*.parquet"))).to_pylist()
    for row in table:
        got = archive.collect([row["symbol"]], FIRST, date(2022, 12, 31), fetched)
        columns, _ = bars.bars(
            archive.day_ms(FIRST),
            34,
            *(got.columns[(row["symbol"], kind)] for kind in archive.KINDS),
        )
        p = labels.params(row)
        assert dataclasses.replace(cascade.Params(), a=p.a, b=p.b, c=0.15, v=3.0) == p
        since = 31 * DAY
        found, _ = cascade.candidates(columns, p, since)
        assert checks[row["symbol"]]["candidates"] == len(found)
