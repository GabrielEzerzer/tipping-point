# The held-back limit, root and prefix check: the default limit writes the tables written before
# it existed, a raised limit runs fetch, labels and inputs across 2024 into held-back/ only, every
# development command refuses that root and its runs, and the prefix check ties both spans.

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from market_planted import Vision, vision
from tipping_point.market import archive, bars
from tipping_point.panel import compare, fetch, inputs, labels, prefix, sequence, span, store
from tipping_point.panel.__main__ import main

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIRST = date(2022, 11, 1)
DAYS = 61


def digest(table: pa.Table) -> str:
    return hashlib.sha256(repr((table.schema.names, table.to_pylist())).encode()).hexdigest()


def tables_of(run: Path) -> dict[str, pa.Table]:
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    return {
        name.split("/")[0].removesuffix(".parquet"): pq.read_table(run / name)
        for name in complete["files"]
        if name.endswith(".parquet")
    }


def market_labels(folder: Path, coins: list[str], first: date, dev: date, last: date) -> Path:
    names = sorted(coins)
    hours = np.arange(labels.ns(dev), labels.ns(last + timedelta(days=1)), labels.HOUR_NS)
    days = [labels.ns(dev + timedelta(days=d)) for d in range((last - dev).days + 1)]
    tables = {
        "grid/g.parquet": pa.table(
            {"ts_ns": np.repeat(hours, len(names)), "symbol": np.tile(names, len(hours))}
        ),
        "thresholds/t.parquet": pa.table(
            {
                "ts_ns": np.repeat(days, len(names)),
                "symbol": np.tile(names, len(days)),
                "sd_ret": np.full(len(days) * len(names), 0.004),
                "b": np.full(len(days) * len(names), 0.02),
            }
        ),
    }
    files = {}
    for name, table in tables.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, folder / name)
        files[name] = store.sha256(folder / name)
    params = {"start": str(first), "dev_start": str(dev), "end": str(last), "symbols": coins}
    (folder / "manifest.json").write_text(json.dumps({"run_id": "t", "params": params}), "utf-8")
    (folder / "complete.json").write_text(json.dumps({"files": files, "clean": True}), "utf-8")
    return folder


PINNED = {
    "thresholds": "4cb0dcc64c384caa90955d23b79e15a7edd2d5c1c346302f34c46fab9be2e6ac",
    "cascades": "4a879fd583f1251cb96931ea9f336332ca61f195e31cb83e4d2b2dd98129e6f3",
    "grid": "68b8567354c707c9162a9489ee80ce5de62384e51d39cb72e6fd4eb7ff2b5758",
    "inputs": "abf9120ac093eb9f2dfd823ac8fb00b27bfa2922bf66e4d0d6884f64b11bfb12",
    "market": "cbfe93269f7ad134a48378cff261c7d539d8a27335259cabbeb4ca2cca5e706b",
}


def test_the_default_limit_writes_the_tables_written_before_it_existed(tmp_path: Path) -> None:
    dest = tmp_path / "archive"
    last = FIRST + timedelta(days=DAYS - 1)
    archive.fetch(SYMBOLS, FIRST, last, dest, vision(SYMBOLS, FIRST, DAYS, seed=11).get)
    run, _ = labels.run(dest, tmp_path, SYMBOLS, FIRST, date(2022, 12, 20), last, "t")
    got = {name: digest(table) for name, table in tables_of(run).items()}
    made, _ = inputs.run(run, dest, tmp_path, "t")
    got["inputs"] = digest(tables_of(made)["inputs"])

    coins = [f"C{k:02d}USDT" for k in range(inputs.MARKET_MIN + 1)][::-1]
    first, dev, end = date(2022, 11, 28), date(2022, 12, 4), date(2022, 12, 7)
    market = tmp_path / "market"
    fake = vision(coins, first, (end - first).days + 1, 5)
    archive.fetch(coins, first, end, market, fake.get)
    folder = market_labels(tmp_path / "labels" / "m", coins, first, dev, end)
    made, _ = inputs.run(folder, market, tmp_path / "m", "t")
    got["market"] = digest(tables_of(made)["inputs"])
    assert got == PINNED


LIMIT = date(2024, 2, 1)
START, DEV, END, LAST = date(2023, 11, 1), date(2023, 12, 20), date(2023, 12, 31), date(2024, 1, 5)


@dataclass(frozen=True)
class Spans:
    base: Path
    fake: Vision
    dev_root: Path
    held_root: Path
    dev_labels: Path
    held_labels: Path
    dev_inputs: Path
    held_inputs: Path


@pytest.fixture(scope="module")
def spans(tmp_path_factory: pytest.TempPathFactory) -> Spans:
    base = tmp_path_factory.mktemp("data")
    held = base / span.HELD_BACK_DIR
    fake = vision(SYMBOLS, START, (LAST - START).days + 1, seed=11)
    dev_root, held_root = base / "archives" / "binance", held / "archives" / "binance"
    fetch.fetch(SYMBOLS, START, END, dev_root, fake.get)
    fetch.fetch(SYMBOLS, START, LAST, held_root, fake.get, limit=LIMIT)
    dev_labels, _ = labels.run(dev_root, base, SYMBOLS, START, DEV, END, "dev")
    held_labels, _ = labels.run(held_root, held, SYMBOLS, START, DEV, LAST, "whole", limit=LIMIT)
    dev_inputs, _ = inputs.run(dev_labels, dev_root, base, "dev")
    held_inputs, _ = inputs.run(held_labels, held_root, held, "whole", limit=LIMIT)
    fake.requests.clear()
    return Spans(base, fake, dev_root, held_root, dev_labels, held_labels, dev_inputs, held_inputs)


def params(run: Path) -> dict[str, Any]:
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))["params"]


def test_a_raised_limit_runs_every_data_stage_into_the_held_back_root(spans: Spans) -> None:
    held = spans.base / span.HELD_BACK_DIR
    assert spans.held_labels.parent == held / "labels"
    assert spans.held_inputs.parent == held / "panel-inputs"
    assert [p.name for p in (spans.base / "labels").iterdir()] == [spans.dev_labels.name]
    assert [p.name for p in (spans.base / "panel-inputs").iterdir()] == [spans.dev_inputs.name]
    archived = json.loads((spans.held_root / "manifest.json").read_text(encoding="utf-8"))
    assert archived["limit"] == str(LIMIT)
    assert any("2024-01-05" in name for name in archived["files"])
    dev_archive = json.loads((spans.dev_root / "manifest.json").read_text(encoding="utf-8"))
    assert "limit" not in dev_archive
    for dev, whole in (
        (spans.dev_labels, spans.held_labels),
        (spans.dev_inputs, spans.held_inputs),
    ):
        assert "limit" not in params(dev) and params(whole)["limit"] == str(LIMIT)
    for run, source in (
        (spans.dev_inputs, spans.dev_labels),
        (spans.held_inputs, spans.held_labels),
    ):
        assert params(run)["labels_manifest_sha256"] == store.sha256(source / "manifest.json")
    grid = tables_of(spans.held_labels)["grid"]["ts_ns"].to_numpy()
    assert grid.max() == labels.ns(LAST) + 23 * labels.HOUR_NS
    rows = tables_of(spans.held_inputs)["inputs"]
    assert np.isfinite(rows["vol_1h"].to_numpy()[grid >= labels.ns(span.HELD_BACK)]).any()


def test_a_raised_limit_reads_and_writes_only_under_held_back(spans: Spans, tmp_path: Path) -> None:
    held = spans.base / span.HELD_BACK_DIR
    outside = tmp_path / "out"
    attempts = [
        lambda: fetch.fetch(SYMBOLS, START, LAST, outside, spans.fake.get, limit=LIMIT),
        lambda: labels.run(spans.held_root, outside, SYMBOLS, START, DEV, LAST, "x", limit=LIMIT),
        lambda: labels.run(spans.dev_root, held, SYMBOLS, START, DEV, END, "x", limit=LIMIT),
        lambda: inputs.run(spans.held_labels, spans.held_root, outside, "x", limit=LIMIT),
        lambda: inputs.run(spans.dev_labels, spans.held_root, held, "x", limit=LIMIT),
        lambda: inputs.run(spans.held_labels, spans.dev_root, held, "x", limit=LIMIT),
    ]
    for attempt in attempts:
        with pytest.raises(span.HeldBackError, match="not under a held-back folder"):
            attempt()
    with pytest.raises(span.HeldBackError, match="from 2024-02-01 on"):
        labels.run(spans.held_root, held, SYMBOLS, START, DEV, LIMIT, "x", limit=LIMIT)
    with pytest.raises(span.HeldBackError):
        fetch.guarded(spans.fake.get, limit=LIMIT)(
            archive.Source("metrics", "BTCUSDT", str(LIMIT)).url
        )
    grid = next((spans.held_labels / "grid").glob("*.parquet"))
    with pytest.raises(span.HeldBackError, match="reaches"):
        store.refuse_held_back(grid)
    store.refuse_held_back(grid, limit=LIMIT)
    assert spans.fake.requests == [] and not outside.exists()
    assert len(list((held / "labels").iterdir())) == 1


def test_every_development_command_refuses_the_held_back_root_and_its_runs(
    spans: Spans,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(archive, "http_get", spans.fake.get)
    out = tmp_path / "out"
    data = ["--data-dir", str(out)]
    days = ["--symbols", *SYMBOLS, "--start", str(DEV), "--end", str(END)]
    commands = [
        ["fetch", *days, "--dest", str(spans.held_root)],
        ["labels", *days, "--dev-start", str(END), "--archive", str(spans.held_root), *data],
        ["inputs", "--labels", str(spans.held_labels), "--archive", str(spans.dev_root), *data],
        ["inputs", "--labels", str(spans.dev_labels), "--archive", str(spans.held_root), *data],
        ["ladder", "--inputs", str(spans.held_inputs), *data],
        ["sequence", "--inputs", str(spans.held_inputs), *data],
        ["classical", "--inputs", str(spans.held_inputs), *data],
    ]
    for command in commands:
        assert main(command) == 1, command
    errors = capsys.readouterr().err.splitlines()
    assert len(errors) == len(commands)
    assert all("held-back test span" in line for line in errors), errors
    plan, _, _ = compare.snapshot()
    arms = [f"--arm={name}={tmp_path / name}" for name in plan.names]
    assert main(["compare", "--inputs", str(spans.held_inputs), *arms, *data]) == 1
    with pytest.raises(span.HeldBackError):
        sequence.read_columns(spans.held_inputs, spans.base / span.HELD_BACK_DIR, ["vol_1h"])
    assert spans.fake.requests == [] and not out.exists()


def copied(run: Path, into: Path) -> Path:
    target = into / run.parent.name / run.name
    shutil.copytree(run, target)
    return target


def rewrite(run: Path, prefix_: str, change: Any) -> str:
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    (name,) = [n for n in complete["files"] if n.startswith(prefix_)]
    pq.write_table(change(pq.read_table(run / name)), run / name)
    complete["files"][name] = store.sha256(run / name)
    (run / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    return complete["files"][name]


def changed(table: pa.Table, column: str, ts: int, symbol: str = "ETHUSDT") -> pa.Table:
    hits = np.flatnonzero(
        (table["ts_ns"].to_numpy() == ts) & (np.array(table["symbol"].to_pylist()) == symbol)
    )
    assert hits.size == 1
    i = int(hits[0])
    values = table[column].to_pylist()
    old = values[i]
    if isinstance(old, bool):
        values[i] = not old
    elif old is None or math.isnan(old):
        values[i] = 1.0
    else:
        values[i] = old * 1.0000001
    index = table.schema.get_field_index(column)
    return table.set_column(index, column, pa.array(values, table.schema.field(column).type))


def altered(spans: Spans, tmp_path: Path, what: str, column: str, ts: int) -> tuple[Path, Path]:
    held = tmp_path / span.HELD_BACK_DIR
    labels_run = copied(spans.held_labels, held)
    inputs_run = copied(spans.held_inputs, held)
    if what == "inputs.parquet":
        rewrite(inputs_run, what, lambda t: changed(t, column, ts))
        return labels_run, inputs_run
    sha = rewrite(labels_run, what, lambda t: changed(t, column, ts))
    manifest = json.loads((inputs_run / "manifest.json").read_text(encoding="utf-8"))
    manifest["params"]["labels_grid" if what == "grid/" else "labels_thresholds"]["sha256"] = sha
    (inputs_run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return labels_run, inputs_run


def before_held(hours: float) -> int:
    return labels.ns(span.HELD_BACK) - int(hours * labels.HOUR_NS)


def test_the_prefix_check_passes_equal_runs_whose_labels_differ_only_in_the_window(
    spans: Spans,
) -> None:
    got = prefix.check(spans.held_labels, spans.held_inputs, spans.dev_labels, spans.dev_inputs)
    dev = tables_of(spans.dev_labels)
    assert got["rows"] == {
        "thresholds": dev["thresholds"].num_rows,
        "grid": dev["grid"].num_rows,
        "inputs": dev["grid"].num_rows,
    }
    assert got["window_after_ns"] == {
        "1h": before_held(1.5),
        "6h": before_held(6.5),
        "24h": before_held(24.5),
    }
    whole = tables_of(spans.held_labels)["grid"].slice(0, dev["grid"].num_rows)
    ts = dev["grid"]["ts_ns"].to_numpy()
    for h in labels.HORIZONS:
        name = f"censored_{h}"
        differ = whole[name].to_numpy() != dev["grid"][name].to_numpy()
        assert differ.any() and (ts[differ] > prefix.window_ns(h)).all(), name
    assert np.isfinite(dev["thresholds"]["b"].to_numpy(zero_copy_only=False)[-2:]).all()


@pytest.mark.parametrize(
    ("what", "column", "hours"),
    [
        ("inputs.parquet", "vol_24h", 30.0),
        ("inputs.parquet", "m_ret_1h", 1.0),
        ("grid/", "long_1", 2.0),
        ("grid/", "censored_6", 7.0),
        ("grid/", "censored_24", 25.0),
        ("grid/", "skip", 1.0),
        ("thresholds/", "b", 24.0),
        ("thresholds/", "sd_ret", 72.0),
    ],
)
def test_the_prefix_check_refuses_a_change_before_2024_outside_the_window(
    spans: Spans, tmp_path: Path, what: str, column: str, hours: float
) -> None:
    labels_run, inputs_run = altered(spans, tmp_path, what, column, before_held(hours))
    with pytest.raises(prefix.PrefixError, match=f"{column} differs"):
        prefix.check(labels_run, inputs_run, spans.dev_labels, spans.dev_inputs)


@pytest.mark.parametrize(
    ("column", "hours"),
    [("long_1", 1.0), ("censored_6", 6.0), ("short_24", 24.0), ("censored_24", 1.0)],
)
def test_the_prefix_check_passes_a_change_inside_the_window(
    spans: Spans, tmp_path: Path, column: str, hours: float
) -> None:
    labels_run, inputs_run = altered(spans, tmp_path, "grid/", column, before_held(hours))
    prefix.check(labels_run, inputs_run, spans.dev_labels, spans.dev_inputs)


def test_the_prefix_check_refuses_runs_it_cannot_tie_together(spans: Spans, tmp_path: Path) -> None:
    with pytest.raises(prefix.PrefixError, match="not made from the grid"):
        prefix.check(spans.held_labels, spans.dev_inputs, spans.dev_labels, spans.dev_inputs)
    with pytest.raises(prefix.PrefixError, match="must end on"):
        prefix.check(spans.held_labels, spans.held_inputs, spans.held_labels, spans.held_inputs)
    labels_run = copied(spans.held_labels, tmp_path / span.HELD_BACK_DIR)
    manifest = json.loads((labels_run / "manifest.json").read_text(encoding="utf-8"))
    manifest["params"]["seed"] = 1
    (labels_run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(prefix.PrefixError, match="does not record the manifest"):
        prefix.check(labels_run, spans.held_inputs, spans.dev_labels, spans.dev_inputs)


def gone(fake: Vision, symbol: str, kinds: tuple[str, ...], first: date, last: date) -> None:
    day = first
    while day <= last:
        for kind in kinds:
            url = archive.Source(kind, symbol, f"{day:%Y-%m-%d}").url
            del fake.files[url], fake.files[url + ".CHECKSUM"]
        day += timedelta(days=1)


def unpublished(fake: Vision, symbol: str, month: str) -> None:
    url = archive.Source("fundingRate", symbol, month).url
    del fake.files[url], fake.files[url + ".CHECKSUM"]


def row_mask(table: pa.Table, symbol: str) -> np.ndarray:
    return np.array(table["symbol"].to_pylist()) == symbol


def test_a_coin_whose_bars_end_is_censored_and_has_no_inputs_after_its_last_bar(
    tmp_path: Path,
) -> None:
    first, stop = date(2022, 11, 1), date(2022, 12, 28)
    dev, end = date(2022, 12, 24), date(2023, 1, 3)
    fake = vision(SYMBOLS, first, (end - first).days + 1, seed=11)
    gone(fake, "ETHUSDT", ("klines", "metrics"), stop, end)
    unpublished(fake, "ETHUSDT", "2023-01")
    root = tmp_path / "archive"
    fetch.fetch(SYMBOLS, first, end, root, fake.get)
    run, checks = labels.run(root, tmp_path, SYMBOLS, first, dev, end, "t")
    assert checks["ETHUSDT"]["files_missing"] == 2 * ((end - stop).days + 1) + 1
    made, _ = inputs.run(run, root, tmp_path, "t")
    grid, rows = tables_of(run)["grid"], tables_of(made)["inputs"]
    ts = grid["ts_ns"].to_numpy()
    after = ts > labels.ns(stop)
    eth = row_mask(grid, "ETHUSDT")
    for h in labels.HORIZONS:
        censored = grid[f"censored_{h}"].to_numpy()
        assert censored[eth & (ts + h * labels.HOUR_NS + labels.LEAD_NS > labels.ns(stop))].all()
        assert not censored[eth & (ts < labels.ns(stop) - 2 * labels.DAY_NS)].all()
        assert not censored[~eth & after].all()
    own = np.column_stack([rows[n].to_numpy() for n in inputs.OWN])
    assert np.isnan(own[eth & after]).all()
    assert np.isfinite(own[eth & (ts == labels.ns(stop))]).any()
    assert np.isfinite(rows["vol_28d"].to_numpy()[~eth & after]).all()
    days = tables_of(run)["thresholds"]
    assert (row_mask(days, "ETHUSDT") & (days["ts_ns"].to_numpy() > labels.ns(stop))).any()

    got = archive.collect(["ETHUSDT"], first, end, root)
    columns, _ = bars.bars(
        archive.day_ms(first),
        (end - first).days + 1,
        *(got.columns[("ETHUSDT", k)] for k in archive.KINDS),
    )
    stop_bar = (labels.ns(stop) + 2 * labels.HOUR_NS - labels.ns(first)) // inputs.BAR_NS
    flat = np.full(len(columns["ts_ns"]), np.nan)
    stale = inputs.coin(columns, flat, np.array([stop_bar]), np.array([0.004]), np.array([0.02]))
    assert np.isfinite(stale["vol_28d"][0]) and np.isfinite(stale["funding"][0])


def funding_until(month: date, last: datetime) -> bytes:
    stamps = archive.day_ms(month) + 8 * 3_600_000 * np.arange(31 * 3)
    kept = stamps[stamps <= int(last.timestamp()) * 1000]
    rows = [f"{int(s) + (i % 3) * 4},8,0.00010000" for i, s in enumerate(kept)]
    header = "calc_time,funding_interval_hours,last_funding_rate"
    return ("\n".join([header, *rows]) + "\n").encode()


def test_a_funding_series_that_ends_is_missing_after_its_carry_while_bars_go_on(
    tmp_path: Path,
) -> None:
    first, dev, end = date(2022, 11, 1), date(2022, 12, 20), date(2022, 12, 31)
    fake = vision(SYMBOLS, first, (end - first).days + 1, seed=11)
    settled = datetime(2022, 12, 20, 8, tzinfo=UTC)
    december = archive.Source("fundingRate", "ETHUSDT", "2022-12")
    fake.add(december, funding_until(date(2022, 12, 1), settled))
    root = tmp_path / "archive"
    fetch.fetch(SYMBOLS, first, end, root, fake.get)
    folder = market_labels(tmp_path / "labels" / "f", SYMBOLS, first, dev, end)
    made, _ = inputs.run(folder, root, tmp_path, "t")
    rows = tables_of(made)["inputs"]
    ts = rows["ts_ns"].to_numpy()
    eth = row_mask(rows, "ETHUSDT")
    carry_end = int(settled.timestamp()) * 10**9 + 8 * labels.HOUR_NS + inputs.BAR_NS
    for name in ("funding", "funding_rel"):
        values = rows[name].to_numpy()
        assert np.isfinite(values[eth & (ts <= carry_end)]).all(), name
        assert np.isnan(values[eth & (ts > carry_end)]).all(), name
        assert np.isfinite(values[~eth]).all(), name
    assert np.isfinite(rows["vol_1h"].to_numpy()[eth & (ts > carry_end)]).all()


def test_a_funding_file_is_needed_inside_a_coins_trading_life_only(tmp_path: Path) -> None:
    first, dev, end = date(2022, 11, 28), date(2022, 11, 29), date(2022, 12, 3)
    cases = {"trading": None, "stopped": date(2022, 12, 1), "one-day": date(2022, 12, 2)}
    for name, stop in cases.items():
        fake = vision(SYMBOLS, first, (end - first).days + 1, seed=5)
        unpublished(fake, "ETHUSDT", "2022-12")
        if stop is not None:
            gone(fake, "ETHUSDT", ("klines", "metrics"), stop, end)
        root = tmp_path / name / "archive"
        fetch.fetch(SYMBOLS, first, end, root, fake.get)
        folder = market_labels(tmp_path / name / "labels" / "f", SYMBOLS, first, dev, end)
        if stop is None or stop != date(2022, 12, 1):
            with pytest.raises(archive.ArchiveError, match="funding files missing"):
                inputs.run(folder, root, tmp_path / name, "t")
            continue
        made, _ = inputs.run(folder, root, tmp_path / name, "t")
        rows = tables_of(made)["inputs"]
        late = row_mask(rows, "ETHUSDT") & (rows["ts_ns"].to_numpy() > labels.ns(stop))
        assert late.any() and np.isnan(rows["vol_1h"].to_numpy()[late]).all()


def test_a_coin_counts_in_the_market_means_up_to_its_last_bar_only(tmp_path: Path) -> None:
    coins = [f"C{k:02d}USDT" for k in range(inputs.MARKET_MIN)]
    first, dev = date(2022, 11, 28), date(2022, 12, 4)
    stop, end = date(2022, 12, 6), date(2022, 12, 7)
    fake = vision(coins, first, (end - first).days + 1, 5)
    gone(fake, "C00USDT", ("klines", "metrics"), stop, end)
    root = tmp_path / "archive"
    fetch.fetch(coins, first, end, root, fake.get)
    folder = market_labels(tmp_path / "labels" / "m", coins, first, dev, end)
    made, _ = inputs.run(folder, root, tmp_path, "t")
    rows = tables_of(made)["inputs"]
    ts = rows["ts_ns"].to_numpy()
    mine = row_mask(rows, "C00USDT")
    vol, market = rows["vol_24h"].to_numpy(), rows["m_vol_24h"].to_numpy()
    hours = ts[mine]
    counted = np.isfinite(vol.reshape(len(hours), len(coins))).sum(axis=1)
    assert (counted[hours <= labels.ns(stop)] == inputs.MARKET_MIN).all()
    assert (counted[hours > labels.ns(stop)] == inputs.MARKET_MIN - 1).all()
    assert np.isfinite(market[ts <= labels.ns(stop)]).all()
    assert np.isnan(market[ts > labels.ns(stop)]).all()
    assert np.isnan(vol[mine & (ts > labels.ns(stop))]).all()
