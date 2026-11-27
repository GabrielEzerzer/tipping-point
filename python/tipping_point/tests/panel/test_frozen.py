# The held-back test command on a synthetic span: the frozen plan's loader, the data step and the
# run step end to end with the support on disk before any score, each provenance check refusing its
# broken record, one test per plan, and every development command refusing the test's runs.

import json
import math
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from market_planted import Vision, checksum, vision
from tipping_point.ising import io
from tipping_point.panel import (
    fetch,
    frozen,
    heldback,
    inputs,
    labels,
    prefix,
    sequence,
    span,
    store,
)
from tipping_point.panel.__main__ import main

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER = "f" * 40
ZERO = "0" * 64
CODE = "c0de" * 16
REAL_CODE = frozen.code_sha256
REAL_CHECKOUT = frozen.checkout
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
START, DEV, END, LAST = date(2023, 8, 10), date(2023, 9, 30), date(2023, 12, 31), date(2024, 1, 5)
LIMIT = date(2024, 2, 1)
HELD = span.HELD_BACK_DIR

LADDER = """
start = 2023-09-30
end = 2023-12-31
horizons = [24, 1]
sides = ["long", "short", "any"]
without = []

[model]
c = 0.01
clip = 5.0

[folds]
gap_days = 31
forward_first = 1

[bootstrap]
block_days = 14
draws = 50
seed = 1

[[rungs]]
name = "full"
groups = ["vol", "price", "flow", "lev"]
own_only = ["vol", "price", "flow", "lev"]

[[rungs]]
name = "vol"
groups = ["vol"]
own_only = ["vol"]
"""

FINAL = """
ladder = "ladder.toml"
variants = ["vol"]
rungs = ["full", "vol"]
models = ["jepa"]
schemes = ["forward"]
seeds = SEEDS
folds = [2]
target = "future"
threads = 1

[window]
length = 36
future = 4
shifts = [2, 4]

[encoder]
patch = 2
width = 8
conv_blocks = 1
attention_blocks = 1
heads = 2
kernel = 3
expansion = 2

[predictor]
hidden = 8

[jepa]
steps = 4
batch = 16
lr = 1e-3
final_lr = 1e-4
warmup = 1
weight_decay = 0.05
curve_every = 2
momentum_start = 0.9
momentum_end = 1.0
prediction = 1.0
objective = "z"
variance = 1.0
covariance = 0.04

[supervised]
steps = 4
batch = 16
lr = 1e-3
final_lr = 1e-4
warmup = 1
weight_decay = 0.1
curve_every = 2
dropout = 0.2

[selection]
share = 0.2
every = 2
negatives = 200

[probe]
batch = 4096
readouts = ["last", "means", "spreads"]
c = [0.01, 1.0]
"""


def put(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {value(x)}" for k, x in v.items()) + "}"
    return repr(v) if isinstance(v, float) else str(v)


def toml(raw: dict[str, Any]) -> str:
    return "".join(f"{k} = {value(v)}\n" for k, v in raw.items())


def record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def edit(path: Path, change: Callable[[dict[str, Any]], Any]) -> None:
    found = record(path)
    change(found)
    path.write_text(json.dumps(found, indent=2), encoding="utf-8")


def rebind(run: Path) -> str:
    digest = store.sha256(run / "manifest.json")
    edit(run / "complete.json", lambda c: c.__setitem__("manifest_sha256", digest))
    return digest


@pytest.fixture(scope="module", autouse=True)
def committed() -> Iterator[None]:
    real = labels.grid

    def planted(hours: np.ndarray, coin: Any, rows: Any) -> dict[str, np.ndarray]:
        out = real(hours, coin, rows)
        k = hours // labels.HOUR_NS
        for h in labels.HORIZONS:
            free = ~out[f"censored_{h}"]
            out[f"long_{h}"] = out[f"long_{h}"] | ((k % 23 == 0) & free)
            out[f"short_{h}"] = out[f"short_{h}"] | ((k % 29 == 5) & free)
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io, "code_version", lambda: {"git_sha": SHA, "dirty": False})
        patch.setattr(frozen, "code_sha256", lambda *args: CODE)
        patch.setattr(frozen, "checkout", lambda *args: None)
        patch.setattr(labels, "grid", planted)
        yield


@dataclass(frozen=True)
class World:
    base: Path
    fake: Vision
    dev_labels: Path
    dev_inputs: Path
    pieces: tuple[Path, ...]
    final: Path
    plan: dict[str, Any]
    made: dict[str, Any]


def pin(run: Path) -> dict[str, str]:
    return {"run": run.name, "manifest_sha256": store.sha256(run / "manifest.json")}


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    base = tmp_path_factory.mktemp("data")
    fake = vision(SYMBOLS, START, (LAST - START).days + 1, seed=11)
    root = base / "archives" / "binance"
    fetch.fetch(SYMBOLS, START, END, root, fake.get)
    dev_labels, _ = labels.run(root, base, SYMBOLS, START, DEV, END, "dev")
    dev_inputs, _ = inputs.run(dev_labels, root, base, "dev")
    plans = tmp_path_factory.mktemp("plans")
    put(plans / "ladder.toml", LADDER)
    for name, seeds in (("final", "[3, 4]"), ("piece3", "[3]"), ("piece4", "[4]")):
        put(plans / f"{name}.toml", FINAL.replace("SEEDS", seeds))
    pieces = tuple(
        sequence.run(dev_inputs, base, plans / f"piece{s}.toml", f"piece{s}")[0] for s in (3, 4)
    )
    final, _ = sequence.merge_run(list(pieces), plans / "final.toml", base, "final")
    files = record(final / "complete.json")["files"]
    encoders = record(final / "manifest.json")["params"]["encoders"]
    plan = {
        "end": LAST,
        "code_sha256": CODE,
        "approved": [SHA],
        "pairs": [["jepa", "window"], ["jepa", "full"], ["full", "vol"]],
        "primary": ["jepa", "full"],
        "side": "long+short",
        "horizon": 24,
        "runs": {
            "labels": pin(dev_labels),
            "inputs": pin(dev_inputs),
            "final": {
                **pin(final),
                "plan_sha256": files["plan.toml"],
                "ladder_sha256": files["ladder.toml"],
            },
        },
        "rule": {
            "bound": "q05",
            "draws": "all",
            "min_quarters": 1,
            "min_complete": 0.0,
            "min_blocks": 1,
            "min_supported": 1,
        },
        "arms": [
            {
                "name": "jepa",
                "kind": "sequence",
                "model": "jepa",
                "variant": "vol",
                "readout": "means",
                "c": 0.01,
                "seeds": [3, 4],
                "encoders": dict(sorted(encoders.items())),
            },
            {
                "name": "window",
                "kind": "classical",
                "model": "window",
                "variant": "vol",
                "readout": "means",
                "c": 1.0,
            },
            {"name": "full", "kind": "rung", "rung": "full"},
            {"name": "vol", "kind": "rung", "rung": "vol"},
        ],
    }
    path = put(tmp_path_factory.mktemp("plan") / "test.toml", toml(plan))
    made = frozen.data(base, fake.get, path)
    return World(base, fake, dev_labels, dev_inputs, pieces, final, plan, made)


def copy(world: World, folder: Path) -> Path:
    data = folder / "data"
    for part in (
        "labels",
        "panel-inputs",
        "panel-sequence",
        f"{HELD}/labels",
        f"{HELD}/panel-inputs",
    ):
        shutil.copytree(world.base / part, data / part)
    return data


def changed(plan: dict[str, Any], **change: Any) -> dict[str, Any]:
    return json.loads(json.dumps(plan, default=str)) | {"end": LAST} | change


@dataclass(frozen=True)
class Done:
    data: Path
    path: Path
    run: Path
    out: dict[str, Any]
    expected: dict[str, Any]


@pytest.fixture(scope="module")
def tested(world: World, tmp_path_factory: pytest.TempPathFactory) -> Done:
    folder = tmp_path_factory.mktemp("tested")
    data, path = copy(world, folder), put(folder / "test.toml", toml(world.plan))
    c = frozen.checked(data, path)
    t = c.test
    expected = heldback.run(
        c.values, c.grid, c.plan, t.end, c.arms, t.pairs, t.primary, t.side, t.horizon, t.rule
    )
    run, out = frozen.run(data, path)
    return Done(data, path, run, out, expected)


def test_a_frozen_plan_loads_and_with_none_both_steps_refuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if frozen.PLAN.exists():
        assert frozen.load().path == frozen.PLAN
        return
    for step in ("data", "run"):
        assert main(["test", step, "--data-dir", str(tmp_path / "out")]) == 1
        assert "the test plan is not frozen" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_the_data_step_makes_the_whole_span_runs_under_the_held_back_root(world: World) -> None:
    held = world.base / HELD
    assert world.made["made"] == [world.made["labels"], world.made["inputs"]]
    assert world.made["labels"].parent == held / "labels"
    assert world.made["inputs"].parent == held / "panel-inputs"
    for run in (world.made["labels"], world.made["inputs"]):
        params = record(run / "manifest.json")["params"]
        assert params["limit"] == str(LIMIT) and params["end"] == str(LAST)
    assert world.made["prefix"]["rows"]["grid"] > 0
    assert world.made["missing"] == [
        {"symbol": s, "missing": 0, "missing_in_test": 0, "first": None, "last": None}
        for s in SYMBOLS
    ]
    assert [p.name for p in (world.base / "labels").iterdir()] == [world.dev_labels.name]


def test_the_data_step_run_again_reuses_its_runs_and_fetches_nothing(world: World) -> None:
    held = world.base / HELD
    before = {
        p: p.read_bytes() for p in held.rglob("*") if p.is_file() and "archives" not in p.parts
    }
    world.fake.requests.clear()
    path = put(world.base.parent / "again.toml", toml(world.plan))
    again = frozen.data(world.base, world.fake.get, path)
    assert again["made"] == [] and world.fake.requests == []
    assert (again["labels"], again["inputs"]) == (world.made["labels"], world.made["inputs"])
    after = {
        p: p.read_bytes() for p in held.rglob("*") if p.is_file() and "archives" not in p.parts
    }
    assert after == before


def test_the_data_step_refuses_a_half_written_or_second_whole_span_run(
    world: World, tmp_path: Path
) -> None:
    data = copy(world, tmp_path)
    path = put(tmp_path / "test.toml", toml(world.plan))
    whole = data / HELD / "labels" / world.made["labels"].name
    half = whole.with_name("20240101T000000Z-whole")
    shutil.copytree(whole, half)
    (half / "complete.json").unlink()
    with pytest.raises(ValueError, match="did not finish"):
        frozen.data(data, world.fake.get, path)
    (half / "complete.json").write_bytes((whole / "complete.json").read_bytes())
    with pytest.raises(ValueError, match="takes exactly one"):
        frozen.data(data, world.fake.get, path)
    assert world.fake.requests == []


def test_the_tables_read_for_scoring_are_the_recorded_bytes(world: World) -> None:
    labels_dir, inputs_dir = world.made["labels"], world.made["inputs"]
    manifest = record(inputs_dir / "manifest.json")
    complete = record(inputs_dir / "complete.json")
    values, _, read = frozen.read_tables(labels_dir, inputs_dir, manifest, complete, ["vol_1h"])
    assert read["inputs"]["sha256"] == complete["files"]["inputs.parquet"]
    assert len(values["vol_1h"]) > 0
    grid = json.loads(json.dumps(manifest))
    grid["params"]["labels_grid"]["sha256"] = ZERO
    table = json.loads(json.dumps(complete))
    table["files"]["inputs.parquet"] = ZERO
    for m, c, what in ((grid, complete, "grid"), (manifest, table, "inputs table")):
        with pytest.raises(ValueError, match=f"the whole-span {what}: its sha256 differs"):
            frozen.read_tables(labels_dir, inputs_dir, m, c, ["vol_1h"])


def held_free(world: World, folder: Path) -> Path:
    data = copy(world, folder)
    shutil.rmtree(data / HELD)
    return data


def served(world: World, answers: dict[str, Any]) -> Callable[[str], bytes | None]:
    def get(url: str) -> bytes | None:
        for end, answer in answers.items():
            if url.endswith(end):
                if isinstance(answer, Exception):
                    raise answer
                return answer
        return world.fake.files.get(url)

    return get


def test_the_data_step_says_what_a_re_run_does_for_each_bad_download(
    world: World, tmp_path: Path
) -> None:
    data = held_free(world, tmp_path)
    path = put(tmp_path / "test.toml", toml(world.plan))
    junk = b"not a zip"
    late = {"ETHUSDT-metrics-2024-01-05.zip.CHECKSUM": None}
    get = served(
        world,
        late
        | {
            "BTCUSDT-1m-2024-01-02.zip": OSError("connection reset"),
            "ETHUSDT-1m-2024-01-03.zip": b"changed",
            "BTCUSDT-metrics-2024-01-04.zip.CHECKSUM": checksum(
                "BTCUSDT-metrics-2024-01-04.zip", junk
            ),
            "BTCUSDT-metrics-2024-01-04.zip": junk,
        },
    )
    kept = data / HELD / "archives" / "binance" / "metrics" / "BTCUSDT"
    kept = kept / "BTCUSDT-metrics-2024-01-04.zip"
    with pytest.raises(ValueError) as caught:
        frozen.data(data, get, path)
    message = str(caught.value)
    assert message.startswith("no labels run was made: ")
    assert "1 refused (the first klines/ETHUSDT/ETHUSDT-1m-2024-01-03.zip)" in message
    assert "1 failed (the first klines/BTCUSDT/BTCUSDT-1m-2024-01-02.zip)" in message
    assert "a re-run fetches it again" in message
    assert "1 unreadable" in message and f"{kept} and {kept}.CHECKSUM" in message
    assert kept.is_file() and not (data / HELD / "labels").exists()
    with pytest.raises(ValueError) as caught:
        frozen.data(data, served(world, late), path)
    assert "refused" not in str(caught.value) and "failed" not in str(caught.value)
    assert f"{kept} and {kept}.CHECKSUM" in str(caught.value)
    kept.unlink()
    kept.with_name(kept.name + ".CHECKSUM").unlink()
    out = frozen.data(data, served(world, late), path)
    assert out["missing"] == [
        {"symbol": "BTCUSDT", "missing": 0, "missing_in_test": 0, "first": None, "last": None},
        {
            "symbol": "ETHUSDT",
            "missing": 1,
            "missing_in_test": 1,
            "first": "2024-01-05",
            "last": "2024-01-05",
        },
    ]


def test_the_missing_files_table_reads_each_coins_days_from_2024_on() -> None:
    base = "https://data.binance.vision/data/futures/um"
    manifest = {
        "params": {"symbols": SYMBOLS},
        "missing_sources": [
            f"{base}/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2023-12-31.zip",
            f"{base}/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-03.zip",
            f"{base}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip",
        ],
    }
    assert frozen.missing_files(manifest, LAST, "x") == [
        {
            "symbol": "BTCUSDT",
            "missing": 3,
            "missing_in_test": 2,
            "first": "2024-01-01",
            "last": "2024-01-05",
        },
        {"symbol": "ETHUSDT", "missing": 0, "missing_in_test": 0, "first": None, "last": None},
    ]
    for urls, problem in (
        ([f"{base}/daily/klines/XRPUSDT/1m/XRPUSDT-1m-2024-01-02.zip"], "'.*XRPUSDT.*' is not"),
        ([3], "3 is not an archive file"),
        ("x", "it records no list of missing archive files"),
    ):
        with pytest.raises(ValueError, match=f"x: {problem}"):
            frozen.missing_files(manifest | {"missing_sources": urls}, LAST, "x")


def archive_changed(data: Path, world: World, change: str) -> None:
    labels_run, inputs_run = whole(data, world, "labels"), whole(data, world, "inputs")
    read = record(labels_run / "manifest.json")["sources"]
    url = next(s["url"] for s in read if "/klines/" in s["url"])

    def without(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [s for s in sources if s["url"] != url]

    def missing(m: dict[str, Any]) -> None:
        m["sources"] = without(m["sources"])
        m["missing_sources"].append(url)

    if change == "sha256":
        edit(
            inputs_run / "manifest.json",
            lambda m: next(s for s in m["params"]["sources"] if s["url"] == url).__setitem__(
                "sha256", ZERO
            ),
        )
    elif change == "missing for labels":
        edit(labels_run / "manifest.json", missing)
        digest = store.sha256(labels_run / "manifest.json")
        edit(inputs_run / "manifest.json", set_param("labels_manifest_sha256", digest))
    elif change == "missing for inputs":
        edit(
            inputs_run / "manifest.json",
            lambda m: m["params"].__setitem__("sources", without(m["params"]["sources"])),
        )
    if change:
        rebind(inputs_run)


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ("", None),
        ("sha256", f"klines/.* read by labels as [0-9a-f]{{64}}, by inputs as {ZERO}"),
        ("missing for labels", "klines/.* missing for labels and read by inputs"),
        ("missing for inputs", "klines/.* read by labels and not by inputs"),
    ],
)
@pytest.mark.parametrize("step", ["data", "run"])
def test_both_steps_take_whole_span_runs_only_when_they_read_the_same_archive_files(
    world: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    step: str,
    change: str,
    problem: str | None,
) -> None:
    data = copy(world, tmp_path)
    archive_changed(data, world, change)
    path = put(tmp_path / "test.toml", toml(world.plan))
    opened: list[str] = []
    monkeypatch.setattr(prefix, "check", lambda *a: opened.append("prefix") or pytest.fail())
    go = (
        (lambda: frozen.data(data, lambda url: pytest.fail(url), path))
        if step == "data"
        else (lambda: frozen.run(data, path))
    )
    if problem is None:
        with pytest.raises(pytest.fail.Exception):
            go()
        assert opened == ["prefix"]
        return
    with pytest.raises(ValueError, match="read different archive files .*" + problem):
        go()
    assert opened == []
    assert not (data / HELD / frozen.DOMAIN).exists()


def test_the_data_step_refuses_an_inputs_run_with_no_labels_run(
    world: World, tmp_path: Path
) -> None:
    data = copy(world, tmp_path)
    shutil.rmtree(data / HELD / "labels")
    path = put(tmp_path / "test.toml", toml(world.plan))
    with pytest.raises(ValueError, match="is a whole-span inputs run with no whole-span labels"):
        frozen.data(data, lambda url: pytest.fail(url), path)


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda p: p["pairs"].append(["jepa", "nope"]), "pair jepa - nope names an unknown arm"),
        (lambda p: p.__setitem__("primary", ["window", "vol"]), "is not one of the pairs"),
        (lambda p: p.__setitem__("horizon", 6), "horizon 6 "),
        (lambda p: p["arms"][1].__setitem__("readout", "median"), "the arms: arm window"),
        (lambda p: p["arms"][0].__setitem__("seeds", [3]), "not the final run's models"),
    ],
)
def test_the_data_step_refuses_what_the_run_step_would_before_it_fetches(
    world: World, tmp_path: Path, change: Callable[[dict[str, Any]], Any], problem: str
) -> None:
    data = held_free(world, tmp_path)
    plan = changed(world.plan)
    change(plan)
    path = put(tmp_path / "test.toml", toml(plan))
    with pytest.raises(ValueError, match=problem):
        frozen.data(data, lambda url: pytest.fail(url), path)
    assert not (data / HELD).exists()


def test_the_test_runs_only_the_code_the_plan_names(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = put(tmp_path / "test.toml", toml(changed(world.plan, code_sha256=ZERO)))
    monkeypatch.setattr(frozen, "development", lambda *args: pytest.fail("read a run"))
    data = tmp_path / "absent"
    for step in (
        lambda: frozen.data(data, lambda url: pytest.fail(url), path),
        lambda: frozen.run(data, path),
    ):
        with pytest.raises(ValueError, match=f"is {CODE}, not the plan's code_sha256 {ZERO}"):
            step()
    assert not data.exists()


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)


def test_the_code_sha256_covers_the_package_and_its_locks_and_not_the_plan(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "core.autocrlf", "false")
    source = "python/tipping_point/src/tipping_point/panel/frozen.py"
    kept = [
        frozen.PLAN_FILE,
        "docs/market.md",
        "python/tipping_point/tests/panel/test_frozen.py",
        "notebooks/run.py",
    ]
    changing = [source, "python/tipping_point/pyproject.toml", "pyproject.toml", "uv.lock"]

    def commit(name: str, text: str) -> str:
        put(tmp_path / name, text)
        git(tmp_path, "add", name)
        git(tmp_path, "commit", "-q", "-m", name)
        return REAL_CODE(tmp_path)

    for name in [*changing, *kept]:
        put(tmp_path / name, "one\n")
        git(tmp_path, "add", name)
    git(tmp_path, "commit", "-q", "-m", "first")
    digest = REAL_CODE(tmp_path)
    for name in kept:
        assert commit(name, "two\n") == digest, name
    for name in changing:
        new = commit(name, "two\n")
        assert new != digest, name
        digest = new
    put(tmp_path / source, "uncommitted\n")
    assert REAL_CODE(tmp_path) == digest
    git(tmp_path, "rm", "-q", "uv.lock")
    git(tmp_path, "commit", "-q", "-m", "no lock")
    with pytest.raises(ValueError, match=r"HEAD tracks no uv.lock"):
        REAL_CODE(tmp_path)


def git_out(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=True, text=True
    ).stdout.strip()


def test_the_checkout_is_the_running_commit_with_nothing_untracked_or_loadable_hidden(
    tmp_path: Path,
) -> None:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "core.autocrlf", "false")
    git(tmp_path, "config", "status.showUntrackedFiles", "no")
    package = "python/tipping_point/src/tipping_point"
    put(tmp_path / ".gitignore", "__pycache__/\nshadow/\n")
    for name in (f"{package}/panel/compare.py", frozen.PLAN_FILE):
        put(tmp_path / name, "one\n")
    git(tmp_path, "add", ".gitignore", f"{package}/panel/compare.py", frozen.PLAN_FILE)
    git(tmp_path, "commit", "-q", "-m", "first")
    head = git_out(tmp_path, "rev-parse", "HEAD")
    put(tmp_path / f"{package}/panel/__pycache__/compare.cpython-314.pyc", "bytes")
    REAL_CHECKOUT(head, tmp_path)
    with pytest.raises(ValueError, match=f"HEAD is {head}, not the running code's commit {SHA}"):
        REAL_CHECKOUT(SHA, tmp_path)
    shadow = put(tmp_path / f"{package}/panel/compare/__init__.py", "")
    assert git_out(tmp_path, "status", "--porcelain") == ""
    with pytest.raises(ValueError, match="uncommitted changes or untracked files"):
        REAL_CHECKOUT(head, tmp_path)
    shadow.unlink()
    put(tmp_path / f"{package}/shadow/__init__.py", "")
    with pytest.raises(ValueError, match=f"Python could import .*{package}/shadow/__init__.py"):
        REAL_CHECKOUT(head, tmp_path)
    shutil.rmtree(tmp_path / f"{package}/shadow")
    git(tmp_path, "rm", "-q", "--cached", frozen.PLAN_FILE)
    put(tmp_path / ".gitignore", f"__pycache__/\nshadow/\n{frozen.PLAN_FILE}\n")
    git(tmp_path, "add", ".gitignore")
    git(tmp_path, "commit", "-q", "-m", "plan ignored")
    head = git_out(tmp_path, "rev-parse", "HEAD")
    assert (tmp_path / frozen.PLAN_FILE).is_file()
    with pytest.raises(ValueError, match=r"HEAD does not track .*test\.toml"):
        REAL_CHECKOUT(head, tmp_path)


def test_the_plan_path_and_the_code_sha256_are_this_checkouts() -> None:
    assert frozen.PLAN == frozen.REPO / frozen.PLAN_FILE
    source = io._source_tree(Path(frozen.__file__).resolve().parents[1])
    assert source == frozen.REPO / "python" / "tipping_point" / "pyproject.toml"
    assert re.fullmatch(r"[0-9a-f]{64}", REAL_CODE())


@pytest.mark.parametrize("damage", ["manifest", "complete", "file"])
def test_an_unreadable_earlier_run_is_refused_before_any_held_back_read(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    data, path = copy(world, tmp_path), put(tmp_path / "test.toml", toml(world.plan))
    folder = data / HELD / frozen.DOMAIN
    if damage == "file":
        put(folder, "x")
    else:
        run = folder / "20240101T000000Z-test"
        manifest = json.dumps({"params": {"plan_sha256": ZERO}})
        put(run / "manifest.json", "{" if damage == "manifest" else manifest)
        if damage == "complete":
            put(run / "complete.json", "{")
    monkeypatch.setattr(prefix, "check", lambda *args: pytest.fail("prefix"))
    monkeypatch.setattr(frozen, "read_tables", lambda *args: pytest.fail("tables"))
    with pytest.raises(
        ValueError, match="is not a folder" if damage == "file" else "cannot be read"
    ):
        frozen.run(data, path)
    assert folder.is_file() if damage == "file" else len(list(folder.iterdir())) == 1


def test_the_tables_read_for_scoring_are_one_grid_and_its_rows(world: World) -> None:
    labels_dir, inputs_dir = world.made["labels"], world.made["inputs"]
    manifest = record(inputs_dir / "manifest.json")
    complete = record(inputs_dir / "complete.json")
    name = Path(manifest["params"]["labels_grid"]["file"]).name
    for file in (f"other/{name}", f"grid/x/{name}", name):
        moved = json.loads(json.dumps(manifest))
        moved["params"]["labels_grid"]["file"] = file
        with pytest.raises(ValueError, match="is not a grid table"):
            frozen.read_tables(labels_dir, inputs_dir, moved, complete, ["vol_1h"])
    other = json.loads(json.dumps(manifest))
    other["params"]["labels_grid"] = record(world.dev_inputs / "manifest.json")["params"][
        "labels_grid"
    ]
    with pytest.raises(ValueError, match="rows are not its grid's rows"):
        frozen.read_tables(world.dev_labels, inputs_dir, other, complete, ["vol_1h"])


def test_a_table_changed_after_the_prefix_check_is_refused_before_the_run_writes(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, path = copy(world, tmp_path), put(tmp_path / "test.toml", toml(world.plan))
    table = whole(data, world, "inputs") / "inputs.parquet"
    real = prefix.check

    def then_changed(*args: Any) -> Any:
        found = real(*args)
        before = table.read_bytes()
        pq.write_table(pq.read_table(table), table, compression="gzip")
        assert table.read_bytes() != before
        return found

    monkeypatch.setattr(prefix, "check", then_changed)
    with pytest.raises(ValueError, match="the whole-span inputs table: its sha256 differs"):
        frozen.run(data, path)
    assert not (data / HELD / frozen.DOMAIN).exists()


def test_the_core_scores_the_checked_model_bytes_at_the_plans_threads(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, path = copy(world, tmp_path), put(tmp_path / "test.toml", toml(world.plan))
    pinned = world.plan["arms"][0]["encoders"]
    real_support, real_threads = heldback.support, torch.set_num_threads
    threads: list[int] = []

    def removed(*args: Any, **kwargs: Any) -> Any:
        for name in pinned:
            (final_of(data, world) / name).unlink()
        return real_support(*args, **kwargs)

    def counted(n: int) -> None:
        threads.append(n)
        real_threads(n)

    monkeypatch.setattr(heldback, "support", removed)
    monkeypatch.setattr(torch, "set_num_threads", counted)
    run, _ = frozen.run(data, path)
    assert threads == [1] and record(run / "manifest.json")["params"]["threads"] == 1
    assert record(run / "complete.json")["checks"]["encoders_used"] == {"jepa": pinned}


def test_a_share_with_no_scoreable_quarter_is_written_as_null(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, path = copy(world, tmp_path), put(tmp_path / "test.toml", toml(world.plan))
    real = heldback.support

    def empty(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **kwargs) | {"complete_draws": math.nan, "causes": ["none scoreable"]}

    monkeypatch.setattr(heldback, "support", empty)
    run, out = frozen.run(data, path)

    def strict(name: str) -> Any:
        text = (run / name).read_text(encoding="utf-8")
        return json.loads(text, parse_constant=lambda c: pytest.fail(f"{name}: {c}"))

    assert out["support"]["complete_draws"] is None
    assert out["decision"]["support"]["complete_draws"] is None
    assert json.loads(frozen._json({"a": [math.inf, np.float64("nan")], "b": 1.5})) == {
        "a": [None, None],
        "b": 1.5,
    }
    assert strict("support.json")["complete_draws"] is None
    assert strict("decision.json")["support"]["complete_draws"] is None
    assert "share of complete draws - of" in (run / "report.md").read_text(encoding="utf-8")
    other = put(tmp_path / "other.toml", toml(changed(world.plan, horizon=1)))
    assert frozen.checked(data, other).earlier[0]["outcome"] == "inconclusive"


def test_the_run_step_writes_the_cores_decision_and_tables(world: World, tested: Done) -> None:
    decision = record(tested.run / "decision.json")
    want = json.loads(frozen._json(tested.expected["decision"]))
    assert decision == want
    assert decision["outcome"] in ("supported", "not supported")
    complete = record(tested.run / "complete.json")
    assert complete["checks"]["outcome"] == decision["outcome"]
    assert complete["manifest_sha256"] == store.sha256(tested.run / "manifest.json")
    names = {"test.toml", "support.json", "decision.json", "report.md"}
    names |= {f"{n}.parquet" for n in ("comparisons", "scores", "folds", "rows")}
    names |= {f"arm-{a['name']}.parquet" for a in world.plan["arms"]}
    assert set(complete["files"]) == names
    for name, digest in complete["files"].items():
        assert store.sha256(tested.run / name) == digest
    for name, table in tested.expected["tables"].items():
        assert pq.read_table(tested.run / f"arm-{name}.parquet").equals(table)
    rows = pq.read_table(tested.run / "comparisons.parquet").to_pylist()
    assert json.loads(frozen._json(rows)) == json.loads(
        frozen._json(tested.expected["comparisons"])
    )
    accounting = tested.expected["accounting"]
    for name, want in (
        ("scores", tested.expected["scores"]),
        ("folds", tested.expected["folds"]),
        ("rows", [{"horizon": h, **row} for h, part in accounting.items() for row in part]),
    ):
        got = pq.read_table(tested.run / f"{name}.parquet").to_pylist()
        assert got and json.loads(frozen._json(got)) == json.loads(frozen._json(want)), name
    params = record(tested.run / "manifest.json")["params"]
    assert params["plan_sha256"] == frozen.load(tested.path).sha256
    assert params["code"]["git_sha"] == SHA and params["earlier"] == []
    assert params["code"]["code_sha256"] == CODE and params["threads"] == 1
    installed = {"torch": torch.__version__, "numpy": np.__version__, "pyarrow": pa.__version__}
    assert {k: params[k] for k in installed} == installed
    assert params["runtime"] == json.loads(json.dumps(sequence.runtime()))
    assert params["runs"]["final"]["pieces"] == [pin(p) | {"commit": SHA} for p in world.pieces]
    assert params["runs"]["whole_labels"]["commit"] == params["runs"]["whole_inputs"]["commit"]
    assert params["runs"]["whole_inputs"]["commit"] == SHA
    assert params["missing"] == world.made["missing"]
    report = (tested.run / "report.md").read_text(encoding="utf-8")
    assert f"**{decision['outcome']}**" in report and "## Primary" in report
    assert f"code sha256 {CODE}" in report and "| BTCUSDT | 0 | 0 | - | - |" in report
    assert f"not checked against uv.lock: torch {torch.__version__}" in report
    assert (tested.run / "test.toml").read_bytes() == tested.path.read_bytes()


def test_the_models_the_core_loads_are_the_final_runs(world: World, tested: Done) -> None:
    used = record(tested.run / "complete.json")["checks"]["encoders_used"]
    saved = record(world.final / "manifest.json")["params"]["encoders"]
    files = record(world.final / "complete.json")["files"]
    pinned = world.plan["arms"][0]["encoders"]
    assert used == {"jepa": pinned} and set(pinned) == set(saved)
    for name, digest in pinned.items():
        assert saved[name] == files[name] == digest == store.sha256(world.final / name)
    assert tested.expected["encoders"] == used


def test_the_support_is_on_disk_before_the_first_score(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, path = copy(world, tmp_path), put(tmp_path / "test.toml", toml(world.plan))
    folder = data / HELD / frozen.DOMAIN

    def failed(*args: Any, **kwargs: Any) -> Any:
        (run,) = folder.iterdir()
        assert (run / "support.json").is_file() and (run / "manifest.json").is_file()
        raise RuntimeError("scoring failed on purpose")

    monkeypatch.setattr(heldback, "scored", failed)
    with pytest.raises(RuntimeError, match="on purpose"):
        frozen.run(data, path)
    (run,) = folder.iterdir()
    assert sorted(p.name for p in run.iterdir()) == ["manifest.json", "support.json", "test.toml"]
    support = record(run / "support.json")
    assert support["causes"] == [] and support["scoreable"] == 1
    with pytest.raises(ValueError, match=r"already ran this plan .* did not finish"):
        frozen.run(data, path)


def test_one_test_per_plan_and_another_plan_lists_the_earlier_runs(
    world: World, tested: Done, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=r"already ran this plan .* finished: one test per plan"):
        frozen.run(tested.data, tested.path)
    other = put(tmp_path / "other.toml", toml(changed(world.plan, pairs=[["jepa", "full"]])))
    run, out = frozen.run(tested.data, other)
    first = {
        "run": tested.run.name,
        "plan_sha256": frozen.load(tested.path).sha256,
        "finished": True,
        "outcome": tested.out["decision"]["outcome"],
    }
    assert record(run / "manifest.json")["params"]["earlier"] == [first]
    assert f"`{tested.run.name}`" in (run / "report.md").read_text(encoding="utf-8")
    assert out["support"] == json.loads(frozen._json(tested.out["support"]))


def test_an_inconclusive_support_scores_nothing(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = copy(world, tmp_path)
    rule = world.plan["rule"] | {"min_quarters": 2}
    path = put(tmp_path / "test.toml", toml(changed(world.plan, rule=rule)))
    monkeypatch.setattr(heldback, "scored", lambda *a, **k: pytest.fail("scored"))
    run, _ = frozen.run(data, path)
    decision = record(run / "decision.json")
    assert decision["outcome"] == "inconclusive" and decision["causes"]
    assert record(run / "complete.json")["rows"] == {}
    assert not list(run.glob("*.parquet"))
    text = (run / "report.md").read_text(encoding="utf-8")
    assert "no arm was scored" in text and "no arm was fitted or scored" in text
    assert "No arm or probe was refitted." in text
    assert not re.search(r"fitted once|scores the test span|refitted every", text)


def write_plan(tmp_path: Path, text: str) -> Path:
    return put(tmp_path / "test.toml", text)


def loaded_problem(world_plan: dict[str, Any], tmp_path: Path, **change: Any) -> str:
    with pytest.raises(ValueError) as caught:
        frozen.load(write_plan(tmp_path, toml(changed(world_plan, **change))))
    return str(caught.value)


@pytest.fixture
def base_plan() -> dict[str, Any]:
    arm = {
        "name": "jepa",
        "kind": "sequence",
        "model": "jepa",
        "variant": "vol",
        "readout": "means",
        "c": 0.01,
        "seeds": [3, 4],
        "encoders": {"encoders/vol-forward-fold3-seed3-jepa.pt": ZERO},
    }
    return {
        "end": LAST,
        "code_sha256": CODE,
        "approved": [SHA],
        "pairs": [["jepa", "full"]],
        "primary": ["jepa", "full"],
        "side": "long+short",
        "horizon": 24,
        "runs": {
            "labels": {"run": "a", "manifest_sha256": ZERO},
            "inputs": {"run": "b", "manifest_sha256": ZERO},
            "final": {
                "run": "c",
                "manifest_sha256": ZERO,
                "plan_sha256": ZERO,
                "ladder_sha256": ZERO,
            },
        },
        "rule": {
            "bound": "q05",
            "draws": "all",
            "min_quarters": 1,
            "min_complete": 0.0,
            "min_blocks": 1,
            "min_supported": 1,
        },
        "arms": [arm, {"name": "full", "kind": "rung", "rung": "full"}],
    }


def test_the_loader_reads_a_whole_plan_and_only_its_file(
    base_plan: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TP_DATA_DIR", str(tmp_path / "elsewhere"))
    path = write_plan(tmp_path, toml(base_plan))
    test = frozen.load(path)
    assert test.end == LAST and test.limit == LIMIT and test.approved == (SHA,)
    assert test.final.run == "c" and test.primary == ("jepa", "full")
    (arm, rung) = test.arms
    assert isinstance(arm, heldback.SequenceArm) and arm.seeds == (3, 4) and arm.c == 0.01
    assert rung == heldback.RungArm("full", "full")
    assert test.rule == heldback.Rule("q05", "all", 1, 0.0, 1, 1)
    assert test.sha256 == store.sha256(path)


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"extra": 1}, "the plan: unknown key extra"),
        ({"runs": {"labels": {"run": "a", "manifest_sha256": ZERO}}}, "runs: missing key final"),
        ({"rule": {"bound": "q05"}}, "rule: missing key draws"),
        ({"end": "2024-01-05"}, "end '2024-01-05'"),
        ({"end": date(2023, 12, 31)}, "a date from 2024-01-01 on"),
        ({"horizon": 24.0}, "horizon 24.0 must be a whole number"),
        ({"horizon": True}, "horizon True must be a whole number"),
        ({"approved": [SHA[:12]]}, "approved must list distinct full commit sha1s"),
        ({"approved": [SHA, SHA]}, "approved must list distinct full commit sha1s"),
        ({"pairs": [["jepa"]]}, "pairs must list pairs of arm names"),
        ({"side": 1}, "side must be text"),
    ],
)
def test_the_loader_refuses_unknown_missing_and_mistyped_keys(
    base_plan: dict[str, Any], tmp_path: Path, change: dict[str, Any], problem: str
) -> None:
    raw = base_plan | change
    with pytest.raises(ValueError) as caught:
        frozen.load(write_plan(tmp_path, toml(raw)))
    assert problem in str(caught.value)


def runs_of(p: dict[str, Any]) -> dict[str, Any]:
    return p["runs"]


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (
            lambda p: runs_of(p)["labels"].__setitem__("run", "a/b"),
            "runs.labels: run 'a/b' is not a run id",
        ),
        (
            lambda p: runs_of(p)["inputs"].__setitem__("manifest_sha256", "x"),
            "runs.inputs: manifest_sha256 'x' is not a sha256",
        ),
        (
            lambda p: runs_of(p)["final"].__setitem__("ladder_sha256", 1),
            "runs.final: ladder_sha256 1 is not a sha256",
        ),
        (lambda p: runs_of(p).__setitem__("final", 1), "runs.final is not a table"),
        (lambda p: p.__setitem__("runs", 1), "runs is not a table"),
        (lambda p: p.__setitem__("rule", 1), "rule is not a table"),
        (lambda p: p.__setitem__("arms", []), "arms must list at least one arm"),
        (lambda p: p.__setitem__("arms", "jepa"), "arms must list at least one arm"),
        (lambda p: p.__setitem__("primary", ["jepa"]), "primary must be a pair of arm names"),
        (lambda p: p.__setitem__("primary", "jepa"), "primary must be a pair of arm names"),
        (lambda p: p.__setitem__("code_sha256", "x"), "code_sha256 'x' is not a sha256"),
        (lambda p: p.pop("code_sha256"), "the plan: missing key code_sha256"),
    ],
)
def test_the_loader_refuses_bad_pins_tables_and_pairs(
    base_plan: dict[str, Any], tmp_path: Path, change: Callable[[dict[str, Any]], Any], problem: str
) -> None:
    change(base_plan)
    with pytest.raises(ValueError) as caught:
        frozen.load(write_plan(tmp_path, toml(base_plan)))
    assert problem in str(caught.value)


def test_the_loader_refuses_a_duplicate_key(base_plan: dict[str, Any], tmp_path: Path) -> None:
    text = toml(base_plan) + "horizon = 1\n"
    with pytest.raises(ValueError, match=r"test\.toml: Cannot overwrite a value"):
        frozen.load(write_plan(tmp_path, text))


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"seeds": [3, 3]}, "seeds [3, 3]"),
        ({"seeds": [-1]}, "seeds [-1]"),
        ({"seeds": [3.0]}, "seeds [3.0]"),
        ({"seeds": []}, "seeds []"),
        ({"c": 0.0}, "c 0.0"),
        ({"c": True}, "c True"),
        ({"model": "probe"}, "model 'probe'"),
        ({"readout": "median"}, "readout 'median'"),
        ({"model": "supervised"}, "the supervised model takes no readout or c"),
        ({"encoders": {"x": "1"}}, "encoders must map each model file"),
        ({"kind": "boosted"}, "kind 'boosted'"),
        ({"kind": ["sequence"]}, "kind ['sequence']"),
        ({"kind": {"sequence": 1}}, "kind {'sequence': 1}"),
        ({"name": "a b"}, "letters, digits"),
        ({"name": 3}, "letters, digits"),
        ({"c": [1]}, "c [1] (a number above 0)"),
        ({"c": "x"}, "c 'x' (a number above 0)"),
        ({"rung": "full"}, "unknown key rung"),
        ({"variant": 3}, "variant must be text"),
    ],
)
def test_the_loader_refuses_an_arm_the_core_would_refuse(
    base_plan: dict[str, Any], tmp_path: Path, change: dict[str, Any], problem: str
) -> None:
    base_plan["arms"][0] |= change
    with pytest.raises(ValueError) as caught:
        frozen.load(write_plan(tmp_path, toml(base_plan)))
    assert problem in str(caught.value)


@pytest.mark.parametrize(
    ("arm", "problem"),
    [
        (
            {
                "name": "w",
                "kind": "classical",
                "model": "rung",
                "variant": "vol",
                "readout": "last",
                "c": 1.0,
            },
            "a classical arm",
        ),
        (
            {"name": "w", "kind": "classical", "model": "window", "variant": "vol", "c": 1.0},
            "a classical arm",
        ),
        ({"name": "JEPA", "kind": "rung", "rung": "vol"}, "arm names must differ, ignoring case"),
        ({"name": 3, "kind": "rung", "rung": "vol"}, "a name is letters, digits"),
        *(
            (
                {
                    "name": "w",
                    "kind": "classical",
                    "model": "window",
                    "variant": "vol",
                    "readout": "means",
                    "c": c,
                },
                "a classical arm",
            )
            for c in (0.0, -1.0, True, "x", [1.0])
        ),
        ({"name": "x", "kind": "rung"}, "missing key rung"),
    ],
)
def test_the_loader_refuses_a_classical_or_rung_arm_the_core_would_refuse(
    base_plan: dict[str, Any], tmp_path: Path, arm: dict[str, Any], problem: str
) -> None:
    base_plan["arms"].append(arm)
    with pytest.raises(ValueError) as caught:
        frozen.load(write_plan(tmp_path, toml(base_plan)))
    assert problem in str(caught.value)


def test_the_loader_refuses_a_rule_the_core_would_refuse(
    base_plan: dict[str, Any], tmp_path: Path
) -> None:
    base_plan["rule"]["bound"] = "q10"
    with pytest.raises(ValueError, match="the decision rule: bound q10"):
        frozen.load(write_plan(tmp_path, toml(base_plan)))


Break = Callable[[Path, dict[str, Any], pytest.MonkeyPatch], Path | None]


def final_of(data: Path, world: World) -> Path:
    return data / "panel-sequence" / world.final.name


def repin(plan: dict[str, Any], key: str, run: Path) -> None:
    plan["runs"][key]["manifest_sha256"] = store.sha256(run / "manifest.json")


def edited(data: Path, world: World, plan: dict[str, Any], change: Callable[[dict], Any]) -> None:
    final = final_of(data, world)
    edit(final / "manifest.json", change)
    rebind(final)
    repin(plan, "final", final)


def piece_edited(
    data: Path, world: World, plan: dict[str, Any], change: Callable[[dict], Any]
) -> None:
    piece = data / "panel-sequence" / world.pieces[0].name
    edit(piece / "manifest.json", change)
    digest = rebind(piece)

    def resourced(m: dict[str, Any]) -> None:
        m["params"]["sources"][0]["manifest_sha256"] = digest

    edited(data, world, plan, resourced)


def replanned(
    data: Path, world: World, plan: dict[str, Any], name: str, old: str, new: str
) -> None:
    final = final_of(data, world)
    text = (final / name).read_text(encoding="utf-8")
    assert old in text
    put(final / name, text.replace(old, new))
    digest = store.sha256(final / name)
    edit(final / "complete.json", lambda c: c["files"].__setitem__(name, digest))
    key = "plan" if name == "plan.toml" else "ladder_plan"
    edited(data, world, plan, lambda m: m["params"][key].__setitem__("sha256", digest))
    plan["runs"]["final"]["plan_sha256" if name == "plan.toml" else "ladder_sha256"] = digest


def whole(data: Path, world: World, kind: str) -> Path:
    run = world.made[kind]
    return data / HELD / run.parent.name / run.name


def set_param(key: str, value: Any) -> Callable[[dict], Any]:
    return lambda m: m["params"].__setitem__(key, value)


def set_producer(key: str, value: Any) -> Callable[[dict], Any]:
    return lambda m: m["producer"].__setitem__(key, value)


def rewritten_inputs(data: Path, world: World) -> None:
    run = whole(data, world, "inputs")
    table = pq.read_table(run / "inputs.parquet")
    column = table["vol_1h"].to_numpy().copy()
    i = int(np.flatnonzero(np.isfinite(column))[0])
    column[i] += 1.0
    table = table.set_column(table.schema.get_field_index("vol_1h"), "vol_1h", pa.array(column))
    pq.write_table(table, run / "inputs.parquet", compression="zstd")
    digest = store.sha256(run / "inputs.parquet")
    edit(run / "complete.json", lambda c: c["files"].__setitem__("inputs.parquet", digest))
    rebind(run)


def guards(world: World) -> dict[str, tuple[Break, str]]:
    def dev(kind: str, change: Callable[[dict], Any], key: str) -> Break:
        def go(data: Path, plan: dict[str, Any], patch: pytest.MonkeyPatch) -> None:
            run = (
                data
                / ("labels" if kind == "labels" else "panel-inputs")
                / getattr(world, f"dev_{kind}").name
            )
            edit(run / "manifest.json", change)
            if kind == "inputs":
                rebind(run)
            repin(plan, key, run)

        return go

    def plain(go: Callable[[Path, dict[str, Any]], Any]) -> Break:
        def run(data: Path, plan: dict[str, Any], patch: pytest.MonkeyPatch) -> None:
            go(data, plan)

        return run

    def final(change: Callable[[dict], Any]) -> Break:
        return plain(lambda data, plan: edited(data, world, plan, change))

    def piece(change: Callable[[dict], Any]) -> Break:
        return plain(lambda data, plan: piece_edited(data, world, plan, change))

    def whole_edit(kind: str, change: Callable[[dict], Any]) -> Break:
        def go(data: Path, plan: dict[str, Any]) -> None:
            run = whole(data, world, kind)
            edit(run / "manifest.json", change)
            if kind == "inputs":
                rebind(run)

        return plain(go)

    def unbound(where: Callable[[Path], Path]) -> Break:
        return plain(
            lambda data, plan: edit(
                where(data) / "complete.json", lambda c: c.pop("manifest_sha256")
            )
        )

    def coded(dirty: Any) -> Break:
        return lambda data, plan, patch: patch.setattr(
            io, "code_version", lambda: {"git_sha": SHA, "dirty": dirty}
        )

    def arm(change: Callable[[dict[str, Any]], Any]) -> Break:
        return plain(lambda data, plan: change(plan["arms"][0]))

    def first_model(data: Path) -> Path:
        return final_of(data, world) / sorted(world.plan["arms"][0]["encoders"])[0]

    def second_whole_labels(data: Path, plan: dict[str, Any]) -> None:
        run = whole(data, world, "labels")
        shutil.copytree(run, run.with_name("20240101T000000Z-whole"))

    def half_written(data: Path, plan: dict[str, Any]) -> None:
        second_whole_labels(data, plan)
        (
            whole(data, world, "labels").with_name("20240101T000000Z-whole") / "complete.json"
        ).unlink()

    def no_whole(data: Path, plan: dict[str, Any]) -> None:
        shutil.rmtree(data / HELD)

    first_name, second_name = sorted(world.plan["arms"][0]["encoders"])

    def saved_changed(data: Path, plan: dict[str, Any]) -> None:
        piece = data / "panel-sequence" / world.pieces[0].name
        edit(
            piece / "manifest.json", lambda m: m["params"]["encoders"].__setitem__(first_name, ZERO)
        )
        digest = rebind(piece)

        def both(m: dict[str, Any]) -> None:
            m["params"]["sources"][0]["manifest_sha256"] = digest
            m["params"]["encoders"][first_name] = ZERO

        edited(data, world, plan, both)

    def prefix_changed(part: str, key: str) -> Break:
        def go(data: Path, plan: dict[str, Any], patch: pytest.MonkeyPatch) -> None:
            real = prefix.check

            def check(*args: Any) -> Any:
                found = real(*args)
                return {**found, part: {**found[part], key: ZERO}}

            patch.setattr(prefix, "check", check)

        return go

    def held_data_dir(data: Path, plan: dict[str, Any], patch: pytest.MonkeyPatch) -> Path:
        moved = data.parent / HELD / "data"
        shutil.copytree(data, moved)
        return moved

    return {
        "dirty code": (coded(True), "the test runs only from a clean commit"),
        "unknown code state": (coded(None), "the test runs only from a clean commit"),
        "no commit": (
            lambda data, plan, patch: patch.setattr(
                io, "code_version", lambda: {"git_sha": None, "dirty": False}
            ),
            "the test runs only from a clean commit: the running code is commit None",
        ),
        "installed copy": (
            lambda data, plan, patch: patch.setattr(io, "_source_tree", lambda package: None),
            "the test runs only from the source tree of the checkout at",
        ),
        "head not the commit": (
            lambda data, plan, patch: patch.setattr(frozen, "checkout", REAL_CHECKOUT),
            f"HEAD is [0-9a-f]{{40}}, not the running code's commit {SHA}",
        ),
        "code sha": (
            plain(lambda d, p: p.__setitem__("code_sha256", ZERO)),
            f"not the plan's code_sha256 {ZERO}",
        ),
        "dev labels coins": (
            dev("labels", set_param("symbols", []), "labels"),
            "development labels run .*: it records no list of coins",
        ),
        "dev labels start": (
            dev("labels", set_param("start", "2023-8-10"), "labels"),
            "development labels run .*: its start '2023-8-10' is not a date",
        ),
        "dev inputs grid": (
            dev("inputs", lambda m: m["params"].pop("labels_grid"), "inputs"),
            "development inputs run .*: it records no sha256 for its labels grid",
        ),
        "no merged runs": (final(set_param("sources", [])), "records no runs merged into it"),
        "merged run twice": (
            final(lambda m: m["params"]["sources"].append(m["params"]["sources"][0])),
            "records a merged run twice",
        ),
        "merged run merged": (piece(set_param("sources", [])), "is itself a merged run"),
        "merged run no models": (
            piece(lambda m: m["params"].pop("encoders")),
            f"the merged run {world.pieces[0].name}: it records no models",
        ),
        "merged model twice": (
            piece(lambda m: m["params"]["encoders"].__setitem__(second_name, ZERO)),
            f"the merged run {world.pieces[1].name}: {second_name} is recorded by another",
        ),
        "merged models": (
            piece(lambda m: m["params"]["encoders"].clear()),
            "the models its merged runs record, by name and sha256, are not the ones it records",
        ),
        "arm saved sha": (
            plain(saved_changed),
            "is not the model the final run's manifest records",
        ),
        "arm files sha": (
            plain(
                lambda d, p: edit(
                    final_of(d, world) / "complete.json",
                    lambda c: c["files"].__setitem__(first_name, ZERO),
                )
            ),
            "is not the model the final run's complete.json records",
        ),
        "prefix labels manifest": (
            prefix_changed("manifests", "labels"),
            "the prefix check read the labels manifest with sha256 0+, not",
        ),
        "prefix dev inputs manifest": (
            prefix_changed("manifests", "dev_inputs"),
            "the prefix check read the dev_inputs manifest",
        ),
        "prefix inputs table": (
            prefix_changed("tables", "inputs"),
            "the prefix check read the inputs table",
        ),
        "prefix dev inputs table": (
            prefix_changed("tables", "dev_inputs"),
            "the prefix check read the dev_inputs table",
        ),
        "data dir under held-back": (held_data_dir, "is under a held-back folder"),
        "labels sha": (
            plain(lambda d, p: p["runs"]["labels"].__setitem__("manifest_sha256", ZERO)),
            "development labels run .*: the sha256 of its manifest.json is not the plan's",
        ),
        "inputs sha": (
            plain(lambda d, p: p["runs"]["inputs"].__setitem__("manifest_sha256", ZERO)),
            "development inputs run .*: the sha256 of its manifest.json is not the plan's",
        ),
        "final sha": (
            plain(lambda d, p: p["runs"]["final"].__setitem__("manifest_sha256", ZERO)),
            "final sequence run .*: the sha256 of its manifest.json is not the plan's",
        ),
        "inputs unbound": (
            unbound(lambda d: d / "panel-inputs" / world.dev_inputs.name),
            "records no manifest sha256",
        ),
        "final unbound": (unbound(lambda d: final_of(d, world)), "records no manifest sha256"),
        "commit not approved": (
            plain(lambda d, p: p.__setitem__("approved", [OTHER])),
            "development labels run .* was made by commit .*, which the test plan does not approve",
        ),
        "labels dirty": (
            dev("labels", set_producer("dirty", True), "labels"),
            "development labels run .* was made by code with uncommitted changes",
        ),
        "labels no commit": (
            dev("labels", set_producer("git_sha", None), "labels"),
            "development labels run .* records no commit",
        ),
        "labels raised limit": (
            dev("labels", set_param("limit", str(LIMIT)), "labels"),
            "was made with a raised limit",
        ),
        "inputs dirty": (
            dev("inputs", set_producer("dirty", True), "inputs"),
            "development inputs run .* uncommitted changes",
        ),
        "inputs from other labels": (
            dev("inputs", set_param("labels_manifest_sha256", ZERO), "inputs"),
            "was not made from the pinned labels run",
        ),
        "final dirty": (final(set_producer("dirty", True)), "final sequence run .* uncommitted"),
        "final not approved": (
            final(set_producer("git_sha", OTHER)),
            "final sequence run .* does not approve",
        ),
        "pieces unbound": (
            final(set_param("pieces_bound", False)),
            "does not record every run merged into it as bound",
        ),
        "pieces commit not recorded": (
            final(lambda m: m["params"]["pieces_producer"].pop("git_sha")),
            "the runs merged into it records no commit",
        ),
        "pieces not approved": (
            final(lambda m: m["params"]["pieces_producer"].__setitem__("git_sha", OTHER)),
            "the runs merged into it was made by commit .* does not approve",
        ),
        "piece manifest": (
            plain(
                lambda d, p: edit(
                    d / "panel-sequence" / world.pieces[0].name / "manifest.json",
                    lambda m: m.__setitem__("note", 1),
                )
            ),
            "its sha256 differs from complete.json's",
        ),
        "piece other manifest": (
            final(lambda m: m["params"]["sources"][0].__setitem__("manifest_sha256", ZERO)),
            "the merged run .* is not the one merged",
        ),
        "piece missing": (
            plain(lambda d, p: shutil.rmtree(d / "panel-sequence" / world.pieces[1].name)),
            "the merged run .* no run at",
        ),
        "piece dirty": (piece(set_producer("dirty", True)), "the merged run .* uncommitted"),
        "plan sha": (
            plain(lambda d, p: p["runs"]["final"].__setitem__("plan_sha256", ZERO)),
            "its plan.toml is not the plan the test plan pins",
        ),
        "ladder sha": (
            plain(lambda d, p: p["runs"]["final"].__setitem__("ladder_sha256", ZERO)),
            "ladder.toml is not the comparison's ladder plan",
        ),
        "plan file changed": (
            plain(lambda d, p: put(final_of(d, world) / "plan.toml", "changed")),
            "plan.toml: its sha256 differs from complete.json's",
        ),
        "not fit-only": (
            plain(lambda d, p: replanned(d, world, p, "plan.toml", "folds = [2]", "folds = [1]")),
            "its plan is not the fit-only fold",
        ),
        "ladder end": (
            plain(
                lambda d, p: replanned(
                    d, world, p, "ladder.toml", "end = 2023-12-31", "end = 2023-12-30"
                )
            ),
            "its ladder plan ends on 2023-12-30",
        ),
        "final inputs table": (
            final(lambda m: m["params"]["inputs"].__setitem__("sha256", ZERO)),
            "records the inputs run",
        ),
        "piece labels grid": (
            piece(lambda m: m["params"]["labels_grid"].__setitem__("sha256", ZERO)),
            f"{world.pieces[0].name} records the labels grid",
        ),
        "arm the core refuses": (
            arm(lambda a: a.__setitem__("variant", "full")),
            "the arms: arm jepa: variant full",
        ),
        "pair the core refuses": (
            plain(lambda d, p: p["pairs"].append(["jepa", "nope"])),
            "the paired evaluation: pair jepa - nope names an unknown arm",
        ),
        "arm seeds": (
            arm(lambda a: a.__setitem__("seeds", [3])),
            "not the final run's models for its seeds",
        ),
        "arm model sha": (
            arm(lambda a: a["encoders"].__setitem__(sorted(a["encoders"])[0], ZERO)),
            "is not the model the final run's manifest records",
        ),
        "model file": (
            plain(lambda d, p: first_model(d).write_bytes(b"changed")),
            "its sha256 differs from the plan's",
        ),
        "no whole-span runs": (plain(no_whole), "run the data step first"),
        "second whole-span labels": (plain(second_whole_labels), "takes exactly one"),
        "half-written whole-span labels": (plain(half_written), "did not finish"),
        "whole labels commit": (
            whole_edit("labels", set_producer("git_sha", OTHER)),
            "whole-span labels run .* not the running commit",
        ),
        "whole labels dirty": (
            whole_edit("labels", set_producer("dirty", True)),
            "whole-span labels run .* uncommitted",
        ),
        "whole labels limit": (
            whole_edit("labels", set_param("limit", "2024-03-01")),
            "whole-span labels run .* records the limit 2024-03-01",
        ),
        "whole labels end": (
            whole_edit("labels", set_param("end", "2024-01-04")),
            "its grid ends on 2024-01-04",
        ),
        "whole labels symbols": (
            whole_edit("labels", set_param("symbols", ["BTCUSDT"])),
            "its symbols is not the development labels run's",
        ),
        "whole inputs unbound": (
            unbound(lambda d: whole(d, world, "inputs")),
            "records no manifest sha256",
        ),
        "whole inputs labels": (
            whole_edit("inputs", set_param("labels_manifest_sha256", ZERO)),
            "does not record the whole-span labels run's manifest",
        ),
        "whole inputs commit": (
            whole_edit("inputs", set_producer("git_sha", OTHER)),
            "whole-span inputs run .* not the running commit",
        ),
        "prefix": (plain(lambda d, p: rewritten_inputs(d, world)), "vol_1h differs at row"),
    }


NAMES = [
    "dirty code",
    "unknown code state",
    "no commit",
    "installed copy",
    "head not the commit",
    "code sha",
    "dev labels coins",
    "dev labels start",
    "dev inputs grid",
    "no merged runs",
    "merged run twice",
    "merged run merged",
    "merged run no models",
    "merged model twice",
    "merged models",
    "arm saved sha",
    "arm files sha",
    "prefix labels manifest",
    "prefix dev inputs manifest",
    "prefix inputs table",
    "prefix dev inputs table",
    "data dir under held-back",
    "labels sha",
    "inputs sha",
    "final sha",
    "inputs unbound",
    "final unbound",
    "commit not approved",
    "labels dirty",
    "labels no commit",
    "labels raised limit",
    "inputs dirty",
    "inputs from other labels",
    "final dirty",
    "final not approved",
    "pieces unbound",
    "pieces commit not recorded",
    "pieces not approved",
    "piece manifest",
    "piece other manifest",
    "piece missing",
    "piece dirty",
    "plan sha",
    "ladder sha",
    "plan file changed",
    "not fit-only",
    "ladder end",
    "final inputs table",
    "piece labels grid",
    "arm the core refuses",
    "pair the core refuses",
    "arm seeds",
    "arm model sha",
    "model file",
    "no whole-span runs",
    "second whole-span labels",
    "half-written whole-span labels",
    "whole labels commit",
    "whole labels dirty",
    "whole labels limit",
    "whole labels end",
    "whole labels symbols",
    "whole inputs unbound",
    "whole inputs labels",
    "whole inputs commit",
    "prefix",
]


def test_every_guard_has_a_case(world: World) -> None:
    assert sorted(guards(world)) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_each_provenance_check_refuses_its_broken_record(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    broken, problem = guards(world)[name]
    data = copy(world, tmp_path)
    plan = changed(world.plan)
    data = broken(data, plan, monkeypatch) or data
    path = put(tmp_path / "test.toml", toml(plan))
    opened: list[str] = []
    real_check, real_read = prefix.check, frozen.read_tables

    def checking(*args: Any) -> Any:
        opened.append("prefix")
        return real_check(*args)

    def reading(*args: Any) -> Any:
        opened.append("tables")
        return real_read(*args)

    monkeypatch.setattr(prefix, "check", checking)
    monkeypatch.setattr(frozen, "read_tables", reading)
    with pytest.raises(ValueError, match=problem):
        frozen.run(data, path)
    assert opened == (["prefix"] if name.startswith("prefix") else [])
    assert not (data / HELD / frozen.DOMAIN).exists()


def refused(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    assert main(argv) == 1, argv
    return capsys.readouterr().err


def external_plan(folder: Path, scores: Path) -> Path:
    text = (
        LADDER
        + f'\n[external]\nname = "x"\npath = {json.dumps(scores.as_posix())}\nscheme = "forward"\n'
    )
    return put(folder / "ladder.toml", text)


def test_ladder_refuses_the_whole_span_inputs_and_the_test_runs_scores(
    world: World, tested: Done, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = ["--data-dir", str(tmp_path / "out")]
    err = refused(["ladder", "--inputs", str(world.made["inputs"]), *out], capsys)
    assert "2024-01-05 is in the held-back test span" in err
    plan = external_plan(tmp_path, tested.run / "arm-jepa.parquet")
    err = refused(
        [
            "ladder",
            "--inputs",
            str(world.dev_inputs),
            "--plan",
            str(plan),
            "--data-dir",
            str(world.base),
        ],
        capsys,
    )
    assert "hour reaches the held-back test span" in err
    assert not (tmp_path / "out").exists() and not (world.base / "panel-ladder").exists()


def test_sequence_refuses_the_whole_span_inputs(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    err = refused(
        ["sequence", "--inputs", str(world.made["inputs"]), "--data-dir", str(tmp_path / "out")],
        capsys,
    )
    assert "2024-01-05 is in the held-back test span" in err
    assert not (tmp_path / "out").exists()


def test_classical_refuses_the_whole_span_inputs(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    err = refused(
        ["classical", "--inputs", str(world.made["inputs"]), "--data-dir", str(tmp_path / "out")],
        capsys,
    )
    assert "2024-01-05 is in the held-back test span" in err
    assert not (tmp_path / "out").exists()


def test_scores_fails_on_the_test_run_for_want_of_a_sequence_plan(
    tested: Done, tmp_path: Path
) -> None:
    argv = ["scores", "--sequence", str(tested.run), "--model", "jepa", "--variant", "vol"]
    with pytest.raises(FileNotFoundError, match=r"plan\.toml"):
        main([*argv, "--data-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_merge_refuses_the_test_run(
    world: World, tested: Done, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    err = refused(
        ["merge", "--runs", str(tested.run), str(world.final), "--data-dir", str(tmp_path / "out")],
        capsys,
    )
    assert "not a sequence run" in err
    assert not (tmp_path / "out").exists()


COMPARE = """
ladder = "ladder.toml"

[[arms]]
name = "x"
kind = "scores"
model = "jepa"
variant = "vol"

[[arms]]
name = "y"
kind = "classical"
export = "vol"

[[pairs]]
first = "x"
second = "y"
primary = true
side = "long+short"
horizon = 24
"""


def test_compare_refuses_the_test_run_as_an_arm(
    world: World, tested: Done, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    put(tmp_path / "ladder.toml", LADDER)
    plan = put(tmp_path / "compare.toml", COMPARE)
    argv = ["compare", "--plan", str(plan), "--arm", f"x={tested.run}", "--arm", f"y={tested.run}"]
    for inputs_run in (world.made["inputs"], world.dev_inputs):
        err = refused(
            [*argv, "--inputs", str(inputs_run), "--data-dir", str(tmp_path / "out")], capsys
        )
        assert f"arm x: {tested.run.name} is a panel-test run, not a panel-scores run" in err
    assert not (tmp_path / "out").exists()


def test_the_data_stages_at_the_default_limit_refuse_the_whole_span_runs(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    held_root = world.base / HELD / "archives" / "binance"
    out = ["--data-dir", str(tmp_path / "out")]
    days = ["--symbols", *SYMBOLS, "--start", str(DEV), "--end", str(END)]
    errors = [
        refused(["fetch", *days, "--dest", str(held_root)], capsys),
        refused(["labels", *days, "--archive", str(held_root), *out], capsys),
        refused(
            ["inputs", "--labels", str(world.made["labels"]), "--archive", str(held_root), *out],
            capsys,
        ),
    ]
    assert all("held-back test span" in err for err in errors), errors
    assert world.fake.requests == [] and not (tmp_path / "out").exists()
