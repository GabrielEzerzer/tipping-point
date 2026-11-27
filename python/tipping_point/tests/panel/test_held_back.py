# The held-back limit, root and prefix check: the default limit writes the tables written before
# it existed, a raised limit runs fetch, labels and inputs across 2024 into held-back/ only, no
# development command reads or writes there, and the prefix check ties both spans.

import calendar
import hashlib
import json
import math
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from market_planted import Vision, vision
from tipping_point.market import archive
from tipping_point.panel import fetch, inputs, labels, prefix, sequence, span, store
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
HELD_NS = labels.ns(span.HELD_BACK)


def settlements(month: date, keep: Callable[[datetime], bool]) -> bytes:
    days = calendar.monthrange(month.year, month.month)[1]
    stamps = [
        datetime(month.year, month.month, 1, tzinfo=UTC) + timedelta(hours=8 * i)
        for i in range(3 * days)
    ]
    rows = [
        f"{int(s.timestamp()) * 1000 + (i % 3) * 4},8,0.00010000"
        for i, s in enumerate(stamps)
        if keep(s)
    ]
    header = "calc_time,funding_interval_hours,last_funding_rate"
    return ("\n".join([header, *rows]) + "\n").encode()


def gone(fake: Vision, symbol: str, kinds: tuple[str, ...], first: date, last: date) -> None:
    day = first
    while day <= last:
        for kind in kinds:
            url = archive.Source(kind, symbol, f"{day:%Y-%m-%d}").url
            del fake.files[url], fake.files[url + ".CHECKSUM"]
        day += timedelta(days=1)


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
    gone(fake, "ETHUSDT", ("klines", "metrics"), END, END)
    december = archive.Source("fundingRate", "ETHUSDT", "2023-12")
    fake.add(
        december,
        settlements(date(2023, 12, 1), lambda s: s < datetime(2023, 12, 31, 16, tzinfo=UTC)),
    )
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
    assert np.isfinite(rows["vol_1h"].to_numpy()[grid >= HELD_NS]).any()


def test_a_lowered_limit_is_recorded_too(tmp_path: Path) -> None:
    lowered = date(2023, 12, 1)
    fake = vision(SYMBOLS, START, 3, seed=11)
    root = tmp_path / "archive"
    fetch.fetch(SYMBOLS, START, START + timedelta(days=2), root, fake.get, limit=lowered)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["limit"] == str(lowered)
    with pytest.raises(span.HeldBackError, match="from 2023-12-01 on"):
        fetch.fetch(SYMBOLS, START, lowered, root, fake.get, limit=lowered)


def test_a_raised_limit_reads_and_writes_only_under_held_back(spans: Spans, tmp_path: Path) -> None:
    held = spans.base / span.HELD_BACK_DIR
    outside = tmp_path / "out"
    attempts = [
        lambda: fetch.fetch(SYMBOLS, START, LAST, outside, spans.fake.get, limit=LIMIT),
        lambda: labels.run(spans.held_root, outside, SYMBOLS, START, DEV, LAST, "x", limit=LIMIT),
        lambda: labels.run(spans.dev_root, held, SYMBOLS, START, DEV, END, "x", limit=LIMIT),
        lambda: labels.run(
            spans.held_root, held / "..", SYMBOLS, START, DEV, LAST, "x", limit=LIMIT
        ),
        lambda: inputs.run(spans.held_labels, spans.held_root, outside, "x", limit=LIMIT),
        lambda: inputs.run(spans.dev_labels, spans.held_root, held, "x", limit=LIMIT),
        lambda: inputs.run(spans.held_labels, spans.dev_root, held, "x", limit=LIMIT),
        lambda: inputs.run(
            spans.held_labels, held / "labels" / ".." / "..", held, "x", limit=LIMIT
        ),
    ]
    for attempt in attempts:
        with pytest.raises(span.HeldBackError, match="not under a held-back folder"):
            attempt()
    with pytest.raises(span.HeldBackError, match="from 2024-02-01 on"):
        labels.run(spans.held_root, held, SYMBOLS, START, DEV, LIMIT, "x", limit=LIMIT)
    with pytest.raises(span.HeldBackError, match="from 2024-02-01 on"):
        fetch.fetch(SYMBOLS, START, LIMIT, spans.held_root, spans.fake.get, limit=LIMIT)
    with pytest.raises(span.HeldBackError):
        fetch.guarded(spans.fake.get, limit=LIMIT)(
            archive.Source("metrics", "BTCUSDT", str(LIMIT)).url
        )
    with pytest.raises(span.HeldBackError, match="from 2024-02-01 on"):
        inputs.ratios(spans.held_root, {}, "BTCUSDT", START, LIMIT, limit=LIMIT)
    grid = next((spans.held_labels / "grid").glob("*.parquet"))
    with pytest.raises(span.HeldBackError, match="reaches"):
        store.refuse_held_back(grid)
    store.refuse_held_back(grid, limit=LIMIT)
    assert spans.fake.requests == [] and not outside.exists()
    assert len(list((held / "labels").iterdir())) == 1


def test_a_development_run_neither_reads_nor_writes_under_held_back(
    spans: Spans, tmp_path: Path
) -> None:
    held = spans.base / span.HELD_BACK_DIR
    outside = tmp_path / "out"
    through = spans.base / "archives" / ".." / span.HELD_BACK_DIR / "archives" / "binance"
    attempts = [
        lambda: fetch.fetch(SYMBOLS, DEV, END, spans.held_root, spans.fake.get),
        lambda: fetch.fetch(SYMBOLS, DEV, END, through, spans.fake.get),
        lambda: labels.run(spans.held_root, outside, SYMBOLS, START, DEV, END, "x"),
        lambda: labels.run(spans.dev_root, held, SYMBOLS, START, DEV, END, "x"),
        lambda: inputs.run(spans.held_labels, spans.dev_root, outside, "x"),
        lambda: inputs.run(spans.dev_labels, spans.held_root, outside, "x"),
        lambda: inputs.run(spans.dev_labels, spans.dev_root, held, "x"),
    ]
    for attempt in attempts:
        with pytest.raises(span.HeldBackError, match="is under a held-back folder"):
            attempt()
    assert spans.fake.requests == [] and not outside.exists()
    assert len(list((held / "labels").iterdir())) == 1
    assert len(list((held / "panel-inputs").iterdir())) == 1


def opened(*args: object, **kwargs: object) -> None:
    raise AssertionError("a refused root was read")


def test_a_refused_root_is_never_listed_or_opened(
    spans: Spans, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for module, name in ((span, "manifest_files"), (inputs, "labels_run"), (archive, "collect")):
        monkeypatch.setattr(module, name, opened)
    held = spans.base / span.HELD_BACK_DIR
    outside = tmp_path / "out"
    attempts = [
        lambda: fetch.fetch(SYMBOLS, START, LAST, outside, spans.fake.get, limit=LIMIT),
        lambda: labels.run(outside, held, SYMBOLS, START, DEV, LAST, "x", limit=LIMIT),
        lambda: inputs.run(spans.held_labels, outside, held, "x", limit=LIMIT),
        lambda: inputs.run(outside, spans.held_root, held, "x", limit=LIMIT),
        lambda: fetch.fetch(SYMBOLS, DEV, END, spans.held_root, spans.fake.get),
        lambda: labels.run(spans.held_root, outside, SYMBOLS, START, DEV, END, "x"),
        lambda: inputs.run(spans.dev_labels, spans.held_root, outside, "x"),
        lambda: inputs.run(spans.held_labels, spans.dev_root, outside, "x"),
    ]
    for attempt in attempts:
        with pytest.raises(span.HeldBackError, match="held-back folder"):
            attempt()
    assert spans.fake.requests == [] and not outside.exists()


def linked(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            pytest.skip("this platform does not let the test make a directory link")
        import _winapi

        try:
            _winapi.CreateJunction(str(target), str(link))
        except OSError:
            pytest.skip("this platform does not let the test make a directory link")


def test_a_folder_reached_through_a_directory_link_is_checked_where_it_resolves(
    spans: Spans, tmp_path: Path
) -> None:
    held = tmp_path / span.HELD_BACK_DIR
    linked(held / "labels", tmp_path / "dev" / "labels")
    with pytest.raises(span.HeldBackError, match="not under a held-back folder"):
        labels.run(spans.held_root, held, SYMBOLS, START, DEV, LAST, "x", limit=LIMIT)
    linked(held / "panel-inputs", tmp_path / "dev" / "panel-inputs")
    with pytest.raises(span.HeldBackError, match="not under a held-back folder"):
        inputs.run(spans.held_labels, spans.held_root, held, "x", limit=LIMIT)
    archive_root = held / "archive"
    linked(archive_root / "klines" / "BTCUSDT", tmp_path / "dev" / "klines")
    with pytest.raises(span.HeldBackError, match="not under a held-back folder"):
        fetch.fetch(SYMBOLS, START, LAST, archive_root, spans.fake.get, limit=LIMIT)
    run = copied(spans.held_labels, held / "copy")
    shutil.move(run / "grid", tmp_path / "dev" / "grid")
    linked(run / "grid", tmp_path / "dev" / "grid")
    with pytest.raises(span.HeldBackError, match="is not inside the labels run"):
        inputs.run(run, spans.held_root, held / "copy", "x", limit=LIMIT)
    dev = tmp_path / "work"
    linked(dev / "labels", spans.base / span.HELD_BACK_DIR / "labels")
    with pytest.raises(span.HeldBackError, match="is under a held-back folder"):
        labels.run(spans.dev_root, dev, SYMBOLS, START, DEV, END, "x")
    assert spans.fake.requests == []
    assert [p.name for p in (tmp_path / "dev" / "labels").iterdir()] == []
    assert len(list((spans.base / span.HELD_BACK_DIR / "labels").iterdir())) == 1


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
    moved = tmp_path / "moved"
    copy_root = moved / "archive"
    shutil.copytree(spans.held_root, copy_root)
    copy_labels, copy_inputs = copied(spans.held_labels, moved), copied(spans.held_inputs, moved)
    commands = [
        ["fetch", *days, "--dest", str(spans.held_root)],
        ["fetch", *days, "--dest", str(copy_root)],
        ["labels", *days, "--dev-start", str(END), "--archive", str(spans.held_root), *data],
        ["labels", *days, "--dev-start", str(END), "--archive", str(copy_root), *data],
        ["inputs", "--labels", str(spans.held_labels), "--archive", str(spans.dev_root), *data],
        ["inputs", "--labels", str(copy_labels), "--archive", str(spans.dev_root), *data],
        ["inputs", "--labels", str(spans.dev_labels), "--archive", str(spans.held_root), *data],
        ["inputs", "--labels", str(spans.dev_labels), "--archive", str(copy_root), *data],
        ["ladder", "--inputs", str(spans.held_inputs), *data],
        ["ladder", "--inputs", str(copy_inputs), *data],
        ["sequence", "--inputs", str(spans.held_inputs), *data],
        ["classical", "--inputs", str(spans.held_inputs), *data],
    ]
    for command in commands:
        assert main(command) == 1, command
    errors = capsys.readouterr().err.splitlines()
    assert len(errors) == len(commands)
    assert all("held-back test span" in line for line in errors), errors
    assert sum("held-back folder" in line for line in errors) == 4, errors
    with pytest.raises(span.HeldBackError):
        sequence.read_columns(spans.held_inputs, spans.base / span.HELD_BACK_DIR, ["vol_1h"])
    with pytest.raises(span.HeldBackError):
        sequence.read_columns(copy_inputs, moved, ["vol_1h"])
    assert spans.fake.requests == [] and not out.exists()


def test_the_named_days_are_refused_from_the_last_day_of_their_period() -> None:
    def first_day_rule(url: str) -> bool:
        found = re.findall(r"(?<!\d)(\d{4})-(?:(\d{2})(?:-(\d{2}))?(?!\d))?", url)
        return any(date(int(y), int(m or 1), int(d or 1)) >= span.HELD_BACK for y, m, d in found)

    def refused(url: str, limit: date = span.HELD_BACK) -> bool:
        try:
            span.refuse_url(url, limit=limit)
        except span.HeldBackError:
            return True
        return False

    months = [f"{y}-{m:02d}" for y in (2021, 2022, 2023, 2024, 2025) for m in range(1, 13)]
    days = [f"{date(2023, 12, 25) + timedelta(days=d)}" for d in range(14)]
    names = [
        *(f"BTCUSDT-fundingRate-{m}.zip" for m in months),
        *(f"klines/BTCUSDT/BTCUSDT-1m-{d}.zip" for d in days),
        "klines/BTCUSDT/BTCUSDT-1m-2023-01-30.zip",
        "klines/ETHUSDT/ETHUSDT-1m-2023-02-02.zip",
        "klines/BTCUSDT/BTCUSDT-1m-2024-01-02.zip",
        "report-2023-x",
        "report-2024-x",
        "report-2025-x",
        archive.Source("metrics", "BTCUSDT", "2023-12-31").url,
        archive.Source("fundingRate", "BTCUSDT", "2024-01").url,
    ]
    for name in names:
        assert refused(name) == first_day_rule(name), name
    mid = date(2024, 6, 15)
    assert refused("BTCUSDT-fundingRate-2024-06.zip", mid)
    assert not refused("BTCUSDT-fundingRate-2024-05.zip", mid)
    assert not refused("BTCUSDT-1m-2024-06-14.zip", mid)
    assert refused("BTCUSDT-1m-2024-06-15.zip", mid)
    assert refused("report-2024-x", mid) and not refused("report-2023-x", mid)
    assert refused("BTCUSDT-fundingRate-2024-01.zip", date(2024, 1, 31))
    assert not refused("BTCUSDT-fundingRate-2024-01.zip", date(2024, 2, 1))


def test_an_archive_naming_a_day_from_a_raised_limit_on_is_refused(tmp_path: Path) -> None:
    root = tmp_path / span.HELD_BACK_DIR / "archive"
    late = root / "klines" / "BTCUSDT" / "BTCUSDT-1m-2024-02-01.zip"
    early = root / "klines" / "BTCUSDT" / "BTCUSDT-1m-2024-01-31.zip"
    early.parent.mkdir(parents=True)
    early.write_bytes(b"")
    span.refuse_archive(root, limit=LIMIT)
    assert span.manifest_files(root, limit=LIMIT) == {}
    late.write_bytes(b"")
    with pytest.raises(span.HeldBackError, match="holds"):
        span.refuse_archive(root, limit=LIMIT)
    late.unlink()
    listed = {"files": {"klines/BTCUSDT/BTCUSDT-1m-2024-02-01.zip": {"status": "missing"}}}
    (root / "manifest.json").write_text(json.dumps(listed), encoding="utf-8")
    with pytest.raises(span.HeldBackError, match="lists in its manifest"):
        span.manifest_files(root, limit=LIMIT)


def test_a_parquet_file_reaching_a_raised_limit_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"ts_ns": [labels.ns(START), labels.ns(LIMIT) - 1]}), path)
    store.refuse_held_back(path, limit=LIMIT)
    pq.write_table(pa.table({"ts_ns": [labels.ns(START), labels.ns(LIMIT)]}), path)
    with pytest.raises(span.HeldBackError, match="reaches"):
        store.refuse_held_back(path, limit=LIMIT)


def test_a_grid_hour_in_the_held_back_span_is_refused_even_past_the_file_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = market_labels(tmp_path / "labels" / "g", SYMBOLS, START, DEV, END)
    late = pa.table({"ts_ns": [HELD_NS, HELD_NS], "symbol": SYMBOLS})
    grid = next((folder / "grid").glob("*.parquet"))
    pq.write_table(pa.concat_tables([pq.read_table(grid), late]), grid)
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    complete["files"]["grid/g.parquet"] = store.sha256(grid)
    (folder / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(span.HeldBackError, match="reaches"):
        inputs.run(folder, tmp_path / "archive", tmp_path, "t")
    monkeypatch.setattr(store, "refuse_held_back", lambda *args, **kwargs: None)
    with pytest.raises(span.HeldBackError, match="the grid holds an hour"):
        inputs.run(folder, tmp_path / "archive", tmp_path, "t")


def copied(run: Path, into: Path) -> Path:
    target = into / run.parent.name / run.name
    shutil.copytree(run, target)
    return target


def edit_json(path: Path, change: Callable[[dict[str, Any]], object]) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    change(record)
    path.write_text(json.dumps(record), encoding="utf-8")


def edit_manifest(run: Path, change: Callable[[dict[str, Any]], object]) -> None:
    edit_json(run / "manifest.json", change)
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    if "manifest_sha256" in complete:
        complete["manifest_sha256"] = store.sha256(run / "manifest.json")
        (run / "complete.json").write_text(json.dumps(complete), encoding="utf-8")


def rewrite(run: Path, prefix_: str, change: Callable[[pa.Table], pa.Table]) -> str:
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    (name,) = [n for n in complete["files"] if n.startswith(prefix_)]
    pq.write_table(change(pq.read_table(run / name)), run / name)
    complete["files"][name] = store.sha256(run / name)
    (run / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    return complete["files"][name]


def pointed(inputs_run: Path, key: str, sha: str) -> None:
    edit_manifest(inputs_run, lambda m: m["params"][key].__setitem__("sha256", sha))


@dataclass(frozen=True)
class Runs:
    labels: Path
    inputs: Path
    dev_labels: Path
    dev_inputs: Path

    def args(self) -> tuple[Path, Path, Path, Path]:
        return self.labels, self.inputs, self.dev_labels, self.dev_inputs

    def relabel(
        self, prefix_: str, change: Callable[[pa.Table], pa.Table], dev: bool = False
    ) -> None:
        labels_run, inputs_run = (
            (self.dev_labels, self.dev_inputs) if dev else (self.labels, self.inputs)
        )
        sha = rewrite(labels_run, prefix_, change)
        pointed(inputs_run, "labels_grid" if prefix_ == "grid/" else "labels_thresholds", sha)


def copies(spans: Spans, tmp_path: Path) -> Runs:
    held, dev = tmp_path / span.HELD_BACK_DIR, tmp_path / "dev"
    return Runs(
        copied(spans.held_labels, held),
        copied(spans.held_inputs, held),
        copied(spans.dev_labels, dev),
        copied(spans.dev_inputs, dev),
    )


def cell(table: pa.Table, column: str, i: int, value: object) -> pa.Table:
    values = table[column].to_pylist()
    values[i] = value
    index = table.schema.get_field_index(column)
    return table.set_column(index, column, pa.array(values, table.schema.field(column).type))


def changed(table: pa.Table, column: str, ts: int, symbol: str = "ETHUSDT") -> pa.Table:
    hits = np.flatnonzero(
        (table["ts_ns"].to_numpy() == ts) & (np.array(table["symbol"].to_pylist()) == symbol)
    )
    assert hits.size == 1
    i = int(hits[0])
    old = table[column].to_pylist()[i]
    if isinstance(old, bool):
        return cell(table, column, i, not old)
    if old is None or math.isnan(old):
        return cell(table, column, i, 1.0)
    return cell(table, column, i, old * 1.0000001)


def before_held(hours: float) -> int:
    return HELD_NS - int(hours * labels.HOUR_NS)


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


def test_a_coin_with_a_gap_at_the_end_of_2023_that_trades_in_2024_passes_the_prefix_check(
    spans: Spans,
) -> None:
    dev, whole = tables_of(spans.dev_inputs)["inputs"], tables_of(spans.held_inputs)["inputs"]
    ts, eth = dev["ts_ns"].to_numpy(), row_mask(dev, "ETHUSDT")
    gap = eth & (ts >= labels.ns(END) + labels.HOUR_NS)
    assert gap.any() and np.isnan(dev["vol_1h"].to_numpy()[gap]).all()
    assert np.isfinite(
        dev["vol_24h"].to_numpy()[eth & (ts == labels.ns(END) + labels.HOUR_NS)]
    ).all()
    assert np.isfinite(dev["funding"].to_numpy()[eth]).all()
    later = row_mask(whole, "ETHUSDT") & (whole["ts_ns"].to_numpy() >= HELD_NS + labels.HOUR_NS)
    assert np.isfinite(whole["vol_1h"].to_numpy()[later]).any()
    assert digest(whole.slice(0, dev.num_rows)) == digest(dev)


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
    runs = copies(spans, tmp_path)
    if what == "inputs.parquet":
        rewrite(runs.inputs, what, lambda t: changed(t, column, before_held(hours), "BTCUSDT"))
    else:
        runs.relabel(what, lambda t: changed(t, column, before_held(hours), "BTCUSDT"))
    with pytest.raises(prefix.PrefixError, match=f"{column} differs"):
        prefix.check(*runs.args())


@pytest.mark.parametrize(
    ("column", "hours"),
    [("long_1", 1.0), ("censored_6", 6.0), ("short_24", 24.0), ("censored_24", 1.0)],
)
def test_the_prefix_check_passes_a_change_inside_the_window(
    spans: Spans, tmp_path: Path, column: str, hours: float
) -> None:
    runs = copies(spans, tmp_path)
    runs.relabel("grid/", lambda t: changed(t, column, before_held(hours), "BTCUSDT"))
    prefix.check(*runs.args())


def first_row(table: pa.Table, column: str = "ts_ns") -> int:
    values = table[column].to_numpy(zero_copy_only=False).astype(np.float64)
    return int(np.flatnonzero(row_mask(table, "BTCUSDT") & np.isfinite(values))[0])


def swap(table: pa.Table, i: int, j: int) -> pa.Table:
    order = np.arange(table.num_rows)
    order[[i, j]] = order[[j, i]]
    return table.take(order)


def bump_time(table: pa.Table) -> pa.Table:
    ts = table["ts_ns"].to_numpy()
    i = int(np.flatnonzero(np.diff(ts) > 0)[2])
    return cell(table, "ts_ns", i, int(ts[i]) + 1)


def cases() -> dict[str, tuple[Callable[[Spans, Runs], tuple[Path, ...]], str]]:
    def itself(s: Spans, r: Runs) -> tuple[Path, ...]:
        return s.dev_labels, s.dev_inputs, s.dev_labels, s.dev_inputs

    def same_inputs(s: Spans, r: Runs) -> tuple[Path, ...]:
        return r.labels, r.dev_inputs, r.dev_labels, r.dev_inputs

    def symbols(s: Spans, r: Runs) -> tuple[Path, ...]:
        edit_manifest(r.labels, lambda m: m["params"].__setitem__("symbols", SYMBOLS[::-1]))
        return r.args()

    def start(s: Spans, r: Runs) -> tuple[Path, ...]:
        edit_manifest(r.labels, lambda m: m["params"].__setitem__("start", "2023-10-31"))
        return r.args()

    def unclean(s: Spans, r: Runs) -> tuple[Path, ...]:
        edit_json(r.labels / "complete.json", lambda c: c.__setitem__("clean", False))
        return r.args()

    def manifest_moved(s: Spans, r: Runs) -> tuple[Path, ...]:
        edit_json(r.inputs / "manifest.json", lambda m: m.__setitem__("note", "x"))
        return r.args()

    def bytes_moved(s: Spans, r: Runs) -> tuple[Path, ...]:
        grid = next((r.labels / "grid").glob("*.parquet"))
        pq.write_table(pq.read_table(grid), grid, compression="none")
        return r.args()

    def thresholds_sha(s: Spans, r: Runs) -> tuple[Path, ...]:
        pointed(r.inputs, "labels_thresholds", "0" * 64)
        return r.args()

    def dev_grid_sha(s: Spans, r: Runs) -> tuple[Path, ...]:
        pointed(r.dev_inputs, "labels_grid", "0" * 64)
        return r.args()

    def two_grids(s: Spans, r: Runs) -> tuple[Path, ...]:
        def more(c: dict[str, Any]) -> None:
            c["files"]["grid/extra.parquet"] = next(
                v for k, v in c["files"].items() if k.startswith("grid/")
            )

        edit_json(r.labels / "complete.json", more)
        return r.args()

    def extra_column(s: Spans, r: Runs) -> tuple[Path, ...]:
        r.relabel("grid/", lambda t: t.append_column("note", pa.array([0] * t.num_rows)))
        return r.args()

    def absent_in_both(s: Spans, r: Runs) -> tuple[Path, ...]:
        r.relabel("grid/", lambda t: t.drop_columns(["skip"]))
        r.relabel("grid/", lambda t: t.drop_columns(["skip"]), dev=True)
        return r.args()

    def unsorted(s: Spans, r: Runs) -> tuple[Path, ...]:
        r.relabel(
            "grid/",
            lambda t: t.take(np.r_[0 : t.num_rows - 4, t.num_rows - 1 : t.num_rows - 5 : -1]),
        )
        return r.args()

    def dev_row_missing(s: Spans, r: Runs) -> tuple[Path, ...]:
        r.relabel("grid/", lambda t: t.slice(0, t.num_rows - 1), dev=True)
        return r.args()

    def null_for_nan(s: Spans, r: Runs) -> tuple[Path, ...]:
        def blank(t: pa.Table) -> pa.Table:
            i = int(np.flatnonzero(np.isnan(t["m_vol_1h"].to_numpy(zero_copy_only=False)))[0])
            return cell(t, "m_vol_1h", i, None)

        rewrite(r.inputs, "inputs.parquet", blank)
        return r.args()

    def null_for_zero(s: Spans, r: Runs) -> tuple[Path, ...]:
        row = first_row(pq.read_table(r.dev_inputs / "inputs.parquet"), "vol_24h")
        rewrite(r.dev_inputs, "inputs.parquet", lambda t: cell(t, "vol_24h", row, 0.0))
        rewrite(r.inputs, "inputs.parquet", lambda t: cell(t, "vol_24h", row, None))
        return r.args()

    def negative_zero(s: Spans, r: Runs) -> tuple[Path, ...]:
        rewrite(
            r.dev_inputs,
            "inputs.parquet",
            lambda t: cell(t, "vol_24h", first_row(t, "vol_24h"), 0.0),
        )
        rewrite(
            r.inputs, "inputs.parquet", lambda t: cell(t, "vol_24h", first_row(t, "vol_24h"), -0.0)
        )
        return r.args()

    def one_ulp(s: Spans, r: Runs) -> tuple[Path, ...]:
        def nudge(t: pa.Table) -> pa.Table:
            i = first_row(t, "vol_24h")
            return cell(t, "vol_24h", i, float(np.nextafter(t["vol_24h"][i].as_py(), np.inf)))

        rewrite(r.inputs, "inputs.parquet", nudge)
        return r.args()

    def time_moved(s: Spans, r: Runs) -> tuple[Path, ...]:
        rewrite(r.inputs, "inputs.parquet", bump_time)
        return r.args()

    def symbols_swapped(s: Spans, r: Runs) -> tuple[Path, ...]:
        rewrite(r.inputs, "inputs.parquet", lambda t: swap(t, 6, 7))
        return r.args()

    def empty_prefix(s: Spans, r: Runs) -> tuple[Path, ...]:
        r.relabel("thresholds/", lambda t: t.filter(pa.array(t["ts_ns"].to_numpy() >= HELD_NS)))
        return r.args()

    def no_2024(s: Spans, r: Runs) -> tuple[Path, ...]:
        r.relabel("thresholds/", lambda t: t.filter(pa.array(t["ts_ns"].to_numpy() < HELD_NS)))
        return r.args()

    def fewer_inputs(s: Spans, r: Runs) -> tuple[Path, ...]:
        rewrite(r.inputs, "inputs.parquet", lambda t: t.slice(0, t.num_rows - 2))
        return r.args()

    return {
        "itself": (itself, "same folder"),
        "same inputs": (same_inputs, "same folder"),
        "symbols": (symbols, "differ in symbols"),
        "start": (start, "differ in start"),
        "unclean": (unclean, "does not mark the run clean"),
        "manifest moved": (manifest_moved, "sha256 of manifest.json differs"),
        "bytes moved": (bytes_moved, "its sha256 differs from complete.json's"),
        "thresholds sha": (thresholds_sha, "not made from the thresholds of"),
        "dev grid sha": (dev_grid_sha, "-dev: it was not made from the grid of"),
        "two grids": (two_grids, "lists 2 files for grid/"),
        "extra column": (extra_column, "not the code's grid schema"),
        "absent in both": (absent_in_both, "not the code's grid schema"),
        "unsorted": (unsorted, "not sorted by ts_ns"),
        "dev row missing": (dev_row_missing, "the development table has"),
        "null for nan": (null_for_nan, "m_vol_1h differs"),
        "null for zero": (null_for_zero, "vol_24h differs"),
        "negative zero": (negative_zero, "vol_24h differs"),
        "one ulp": (one_ulp, "vol_24h differs"),
        "time moved": (time_moved, "ts_ns differs"),
        "symbols swapped": (symbols_swapped, "symbol differs"),
        "empty prefix": (empty_prefix, "no rows before"),
        "no 2024": (no_2024, "no rows from"),
        "fewer inputs": (fewer_inputs, "the inputs tables have"),
    }


@pytest.mark.parametrize("name", list(cases()))
def test_the_prefix_check_refuses_runs_it_cannot_tie_together(
    spans: Spans, tmp_path: Path, name: str
) -> None:
    make, message = cases()[name]
    with pytest.raises(prefix.PrefixError, match=message):
        prefix.check(*make(spans, copies(spans, tmp_path)))


def test_the_prefix_check_refuses_a_labels_manifest_the_inputs_run_does_not_record(
    spans: Spans, tmp_path: Path
) -> None:
    runs = copies(spans, tmp_path)
    edit_manifest(runs.labels, lambda m: m["params"].__setitem__("seed", 1))
    with pytest.raises(prefix.PrefixError, match="does not record the manifest"):
        prefix.check(*runs.args())


def test_the_prefix_check_passes_a_null_in_an_integer_column_equal_on_both_sides(
    spans: Spans, tmp_path: Path
) -> None:
    runs = copies(spans, tmp_path)
    for dev in (False, True):
        runs.relabel("thresholds/", lambda t: cell(t, "windows", first_row(t), None), dev=dev)
    prefix.check(*runs.args())


def row_mask(table: pa.Table, symbol: str) -> np.ndarray:
    return np.array(table["symbol"].to_pylist()) == symbol


def unpublished(fake: Vision, symbol: str, month: str) -> None:
    url = archive.Source("fundingRate", symbol, month).url
    del fake.files[url], fake.files[url + ".CHECKSUM"]


def test_a_coin_whose_bars_end_is_censored_and_fades_out_of_its_inputs(tmp_path: Path) -> None:
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
    stop_ns = labels.ns(stop)
    eth = row_mask(grid, "ETHUSDT")
    for h in labels.HORIZONS:
        censored = grid[f"censored_{h}"].to_numpy()
        assert censored[eth & (ts + h * labels.HOUR_NS + labels.LEAD_NS > stop_ns)].all()
        assert not censored[eth & (ts < stop_ns - 2 * labels.DAY_NS)].all()
        assert not censored[~eth & (ts > stop_ns)].all()
    values = {name: rows[name].to_numpy() for name in ("vol_1h", "vol_24h", "vol_28d", "funding")}
    assert np.isfinite(values["vol_1h"][eth & (ts == stop_ns)]).all()
    assert np.isnan(values["vol_1h"][eth & (ts > stop_ns)]).all()
    assert np.isfinite(values["vol_24h"][eth & (ts == stop_ns + labels.HOUR_NS)]).all()
    assert np.isnan(values["vol_24h"][eth & (ts >= stop_ns + 5 * labels.HOUR_NS)]).all()
    assert np.isfinite(
        values["vol_28d"][eth & (ts > stop_ns) & (ts <= stop_ns + labels.DAY_NS)]
    ).all()
    assert np.isnan(values["vol_28d"][eth & (ts >= stop_ns + 2 * labels.DAY_NS)]).all()
    settled = labels.ns(date(2023, 1, 1)) - 8 * labels.HOUR_NS
    carry = settled + inputs.FUNDING_CAP_BARS * inputs.BAR_NS + inputs.BAR_NS
    assert np.isfinite(values["funding"][eth & (ts <= carry)]).all()
    assert np.isnan(values["funding"][eth & (ts > carry)]).all()
    assert np.isfinite(values["vol_28d"][~eth & (ts > stop_ns)]).all()


def test_a_coin_that_stops_fades_out_as_a_gap_of_the_same_length_does(tmp_path: Path) -> None:
    coins = [f"C{k:02d}USDT" for k in range(inputs.MARKET_MIN + 1)]
    first, dev, end = date(2022, 11, 28), date(2022, 12, 3), date(2022, 12, 8)
    stop, resumed = date(2022, 12, 4), date(2022, 12, 6)
    tables = {}
    for name, last in (("stopped", end), ("gap", resumed - timedelta(days=1))):
        fake = vision(coins, first, (end - first).days + 1, 5)
        gone(fake, "C00USDT", ("klines", "metrics"), stop, last)
        root = tmp_path / name / "archive"
        fetch.fetch(coins, first, end, root, fake.get)
        folder = market_labels(tmp_path / name / "labels" / "m", coins, first, dev, end)
        made, _ = inputs.run(folder, root, tmp_path / name, "t")
        tables[name] = tables_of(made)["inputs"]
    ts = tables["gap"]["ts_ns"].to_numpy()
    upto = int(np.searchsorted(ts, labels.ns(resumed)))
    assert digest(tables["stopped"].slice(0, upto)) == digest(tables["gap"].slice(0, upto))
    assert digest(tables["stopped"]) != digest(tables["gap"])
    mine = row_mask(tables["gap"], "C00USDT")
    during = mine & (ts > labels.ns(stop)) & (ts <= labels.ns(resumed))
    vol_1h, vol_24h = tables["gap"]["vol_1h"].to_numpy(), tables["gap"]["vol_24h"].to_numpy()
    assert np.isnan(vol_1h[during]).all()
    assert np.isfinite(vol_24h[mine & (ts == labels.ns(stop) + labels.HOUR_NS)]).all()
    assert np.isfinite(tables["gap"]["m_vol_24h"].to_numpy()[ts <= labels.ns(resumed)]).all()
    after = mine & (ts >= labels.ns(resumed) + labels.HOUR_NS)
    assert np.isfinite(vol_1h[after]).all()
    assert np.isnan(tables["stopped"]["vol_1h"].to_numpy()[after]).all()


def test_the_funding_cap_keeps_a_settlement_for_16_hours_and_not_one_bar_more() -> None:
    settled = labels.ns(date(2023, 3, 1))
    cap = inputs.FUNDING_CAP_BARS * inputs.BAR_NS
    assert cap == 16 * labels.HOUR_NS
    ts = settled + np.array([-inputs.BAR_NS, 0, cap - inputs.BAR_NS, cap, cap + inputs.BAR_NS])
    rate = np.array([np.nan, 1e-4, 1e-4, 1e-4, 1e-4])
    calc = np.array([settled // 1_000_000 + 4])
    got = inputs.funding_capped(rate, ts, calc)
    assert np.isnan(got[[0, 1]]).all() and np.isfinite(got[[2, 3]]).all() and np.isnan(got[4])
    later = np.array([settled // 1_000_000, (settled + cap) // 1_000_000 + 1])
    assert np.isfinite(inputs.funding_capped(rate, ts, later)[1:]).all()
    assert np.isnan(inputs.funding_capped(rate, ts, np.empty(0, np.int64))).all()


def test_a_funding_gap_longer_than_the_cap_is_missing_until_the_series_resumes(
    tmp_path: Path,
) -> None:
    first, dev, end = date(2022, 11, 1), date(2022, 12, 5), date(2022, 12, 20)
    fake = vision(SYMBOLS, first, (end - first).days + 1, seed=11)
    before = datetime(2022, 12, 10, tzinfo=UTC)
    after = datetime(2022, 12, 11, 16, tzinfo=UTC)
    december = archive.Source("fundingRate", "ETHUSDT", "2022-12")
    fake.add(december, settlements(date(2022, 12, 1), lambda s: s <= before or s >= after))
    root = tmp_path / "archive"
    fetch.fetch(SYMBOLS, first, end, root, fake.get)
    folder = market_labels(tmp_path / "labels" / "f", SYMBOLS, first, dev, end)
    made, _ = inputs.run(folder, root, tmp_path, "t")
    rows = tables_of(made)["inputs"]
    ts, eth = rows["ts_ns"].to_numpy(), row_mask(rows, "ETHUSDT")
    cap = inputs.FUNDING_CAP_BARS * inputs.BAR_NS
    lapse = int(before.timestamp()) * 10**9 + cap + inputs.BAR_NS
    back = int(after.timestamp()) * 10**9 + inputs.BAR_NS
    funding = rows["funding"].to_numpy()
    assert np.isfinite(funding[eth & (ts <= lapse)]).all()
    assert np.isnan(funding[eth & (ts > lapse) & (ts <= back)]).all()
    assert eth[(ts > lapse) & (ts <= back)].sum() == 40 - cap // labels.HOUR_NS
    assert np.isfinite(funding[eth & (ts > back)]).all()
    assert np.isfinite(funding[~eth]).all()
    assert np.isfinite(rows["funding_rel"].to_numpy()[eth & (ts > back + labels.DAY_NS)]).all()


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
