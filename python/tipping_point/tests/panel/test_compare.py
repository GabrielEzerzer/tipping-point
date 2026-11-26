# The paired evaluation on synthetic panels: equal to the ladder's own comparison, one set of draws
# for every arm and side, sequence and window arms end to end, repeat runs, and every refusal of a
# file that does not provably come from the evaluation's runs and plans or does not hold its rows.

import json
import math
import shutil
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tipping_point.ising import io
from tipping_point.panel import classical, compare, inputs, labels, ladder, sequence, span, store
from tipping_point.panel.__main__ import main

HOUR = labels.HOUR_NS
COINS = ["AAAUSDT", "BBBUSDT"]
LABELS = "20260101T000000Z-t"
Config = classical.Config

LADDER = """
start = 2022-01-01
end = 2022-12-31
horizons = [24, 1]
sides = ["long", "short", "any"]
without = ["BBBUSDT"]

[model]
c = 0.01
clip = 5.0

[folds]
gap_days = 31
forward_first = 2

[bootstrap]
block_days = 14
draws = 50
seed = 1

[[rungs]]
name = "full"
groups = ["vol", "price", "flow", "lev"]
own_only = ["lev"]

[[rungs]]
name = "vol"
groups = ["vol"]

[[comparisons]]
first = "full"
second = "vol"
"""

PLAN = """
ladder = "ladder.toml"
variants = ["vol"]
rungs = ["full", "vol"]
models = ["jepa", "twin", "supervised"]
schemes = ["loqo", "forward"]
seeds = [3]
folds = []
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

PAIRED = """
ladder = "ladder.toml"

[[arms]]
name = "full"
kind = "classical"
export = "full"

[[arms]]
name = "vol"
kind = "classical"
export = "vol"

[[pairs]]
first = "full"
second = "vol"
primary = true
side = "long"
horizon = 24
"""


def folder(path: Path, texts: dict[str, str]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (path / name).write_text(text, encoding="utf-8")
    return path


def world(days: int = 365, seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    hours = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(days * 24, dtype=np.int64)
    n = len(hours) * len(COINS)
    values = {name: rng.normal(size=n) for name in inputs.OWN + inputs.MARKET}
    values["ls"][rng.random(n) < 0.01] = np.nan
    grid: dict[str, np.ndarray] = {
        "ts_ns": np.repeat(hours, len(COINS)),
        "symbol": np.tile(np.array(COINS, dtype=object), len(hours)),
    }
    for h in labels.HORIZONS:
        lift = values["vol_1h"] + 0.5 * np.roll(values["vol_24h"], 6 * len(COINS))
        grid[f"long_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 - lift))
        grid[f"short_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 + lift))
        grid[f"censored_{h}"] = rng.random(n) < 0.01
    grid["skip"] = rng.random(n) < 0.02
    return values, grid


def write_labels(data: Path, grid: dict, name: str = LABELS) -> str:
    (data / "labels" / name / "grid").mkdir(parents=True)
    pq.write_table(pa.table(grid), data / "labels" / name / "grid" / "0.parquet")
    return name


def write_inputs(data: Path, values: dict, grid: dict, name: str, tag: str) -> Path:
    grid_file = data / "labels" / name / "grid" / "0.parquet"
    table = pa.table(
        {"ts_ns": grid["ts_ns"], "symbol": grid["symbol"].astype(str), **values},
        schema=inputs.SCHEMA,
    )
    params = {
        "labels_run": name,
        "labels_grid": {"file": "grid/0.parquet", "sha256": store.sha256(grid_file)},
        "start": "2021-10-01",
        "dev_start": "2022-01-01",
        "end": "2022-12-31",
    }
    return store.write(data, "panel-inputs", tag, {"inputs": table}, params, {})


def fixed(data: Path, run_inputs: Path, plan: Path, tag: str, config: Config) -> Path:
    run_dir, _ = classical.run(run_inputs, data, plan, tag, config)
    return run_dir


def params(run: Path) -> dict[str, Any]:
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))["params"]


def record(run: Path, name: str) -> dict[str, Any]:
    return json.loads((run / name).read_text(encoding="utf-8"))


def tamper(run: Path, change: dict[str, Any]) -> None:
    found = record(run, "manifest.json")
    found["params"].update(change)
    (run / "manifest.json").write_text(json.dumps(found), encoding="utf-8")


def rewrite(run: Path, name: str, change: Callable[[dict[str, Any]], Any]) -> None:
    found = record(run, name)
    change(found)
    (run / name).write_text(json.dumps(found), encoding="utf-8")
    done = record(run, "complete.json")
    if name == "manifest.json" and "manifest_sha256" in done:
        done["manifest_sha256"] = store.sha256(run / name)
        (run / "complete.json").write_text(json.dumps(done), encoding="utf-8")


def unbind(run: Path) -> None:
    rewrite(run, "complete.json", lambda r: r.pop("manifest_sha256"))


def edit(run: Path, name: str, change: dict[str, Any]) -> None:
    rewrite(run, name, lambda r: (r["params"] if name == "manifest.json" else r).update(change))


def relabel(run: Path) -> None:
    rewrite(run, "manifest.json", lambda r: r.update(run_id=run.name))


def producer(run: Path, change: dict[str, Any]) -> None:
    rewrite(run, "manifest.json", lambda r: r["producer"].update(change))


def forge(run: Path, name: str, table: pa.Table) -> None:
    pq.write_table(table, run / name)
    rewrite(run, "complete.json", lambda r: r["files"].update({name: store.sha256(run / name)}))


def copied(run: Path, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    dest = root / str(sum(1 for _ in root.iterdir())) / run.name
    shutil.copytree(run, dest)
    return dest


def same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def tag_runs(data: Path, tag: str) -> list[Path]:
    return (
        list((data / "panel-compare").glob(f"*-{tag}")) if (data / "panel-compare").is_dir() else []
    )


@pytest.fixture(scope="module", autouse=True)
def committed() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(io, "code_version", lambda: {"git_sha": "c" * 40, "dirty": False})
        yield


@pytest.fixture(scope="module")
def panel(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    data = tmp_path_factory.mktemp("panel")
    values, grid = world()
    write_labels(data, grid)
    run_inputs = write_inputs(data, values, grid, LABELS, "inputs")
    plan = folder(data / "plan", {"ladder.toml": LADDER, "sequence.toml": PLAN}) / "sequence.toml"
    full = fixed(data, run_inputs, plan, "full", Config("rung", "full", None, 0.01))
    vol = fixed(data, run_inputs, plan, "vol", Config("rung", "vol", None, 0.01))
    paired = folder(data / "paired", {"ladder.toml": LADDER, "compare.toml": PAIRED})
    return {
        "data": data,
        "values": values,
        "grid": grid,
        "inputs": run_inputs,
        "plan": plan,
        "full": full,
        "vol": vol,
        "paired": paired / "compare.toml",
    }


SMALL = """
ladder = "ladder.toml"

[[arms]]
name = "jepa"
kind = "scores"
model = "jepa"
variant = "vol"

[[arms]]
name = "window"
kind = "classical"
export = "vol"

[[pairs]]
first = "jepa"
second = "window"
primary = true
side = "any"
horizon = 24
"""


@pytest.fixture(scope="module")
def small(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    data = tmp_path_factory.mktemp("small")
    values, grid = world(days=200)
    write_labels(data, grid)
    run_inputs = write_inputs(data, values, grid, LABELS, "inputs")
    text = PLAN.replace('schemes = ["loqo", "forward"]', 'schemes = ["forward"]')
    text = text.replace("folds = []", "folds = [2]").replace("seeds = [3]", "seeds = [3, 4]")
    plan = folder(data / "plan", {"ladder.toml": LADDER, "sequence.toml": text}) / "sequence.toml"
    seq, _ = sequence.run(run_inputs, data, plan, "sequence")
    scores, _ = sequence.scores_run(seq, data, "jepa", "vol", "jepa")
    window = fixed(data, run_inputs, plan, "window", Config("window", "vol", "last", 0.01))
    chosen, _ = classical.run(run_inputs, data, plan, "chosen")
    path = folder(data / "paired", {"ladder.toml": LADDER, "compare.toml": SMALL}) / "compare.toml"
    return {
        "data": data,
        "inputs": run_inputs,
        "plan": plan,
        "seq": seq,
        "scores": scores,
        "window": window,
        "chosen": chosen,
        "paired": path,
    }


def run_args(p: dict[str, Any], plan: Path, tag: str, **arms: Path) -> list[str]:
    given = [part for name, run in arms.items() for part in ("--arm", f"{name}={run}")]
    command = ["compare", "--inputs", p["inputs"].name, "--plan", str(plan), *given]
    return [*command, "--data-dir", str(p["data"]), "--tag", tag]


def plan_file(root: Path, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    where = root / str(sum(1 for _ in root.iterdir()))
    return folder(where, {"ladder.toml": LADDER, "compare.toml": text}) / "compare.toml"


def test_the_shipped_plan_names_one_primary_pair_on_the_ladders_plan() -> None:
    plan, records, (text, ladder_text) = compare.snapshot()
    assert plan.ladder == ladder.load()
    assert plan.primary == ("jepa", "classical")
    assert (plan.side, plan.horizon) == ("long+short", 24)
    kinds = {a.name: (a.kind, a.model, a.variant, a.export, dict(a.pins)) for a in plan.arms}
    assert kinds == {
        "jepa": ("scores", "jepa", "full", None, {}),
        "twin": ("scores", "twin", "full", None, {}),
        "supervised": ("scores", "supervised", "full", None, {}),
        "classical": ("classical", None, None, "full", {}),
        "jepa-vol": ("scores", "jepa", "vol", None, {}),
        "classical-vol": ("classical", None, None, "vol", {}),
        "classic": ("classical", None, None, "classic", {}),
    }
    assert plan.pairs == (
        ("jepa", "classical"),
        ("jepa", "twin"),
        ("jepa", "supervised"),
        ("twin", "classical"),
        ("supervised", "classical"),
        ("jepa-vol", "classical-vol"),
        ("jepa", "classic"),
    )
    assert records["ladder_plan"]["file"] == "ladder.toml" and text and ladder_text
    assert set(classical.names(sequence.load())) >= {a.export for a in plan.arms if a.export}


def test_the_pairs_equal_the_ladders_own_comparison_of_two_rungs(panel: dict[str, Any]) -> None:
    data = panel["data"]
    ladder_run, _ = ladder.run(panel["inputs"], data, panel["plan"].with_name("ladder.toml"), "l")
    arms = {"full": panel["full"], "vol": panel["vol"]}
    reverse = {"vol": panel["vol"], "full": panel["full"]}
    assert main(run_args(panel, panel["paired"], "one", **arms)) == 0
    assert main(run_args(panel, panel["paired"], "two", **reverse)) == 0
    (one,), (two,) = tag_runs(data, "one"), tag_runs(data, "two")
    tables = ("scores", "folds", "comparisons", "rows")
    for name in tables:
        assert (one / f"{name}.parquet").read_bytes() == (two / f"{name}.parquet").read_bytes()
    assert params(one)["arms"] == params(two)["arms"]
    theirs = pq.read_table(ladder_run / "comparisons.parquet").to_pylist()
    mine = pq.read_table(one / "comparisons.parquet").to_pylist()
    assert len(theirs) == 2 * 2 * 3 and len(mine) == 2 * 2 * 4
    for row in mine:
        assert row["primary_pair"] is True
        assert row["primary"] is (
            (row["scheme"], row["side"], row["horizon"]) == ("forward", "long", 24)
        )
        if row["side"] == "long+short":
            continue
        match = [
            r
            for r in theirs
            if all(r[k] == row[k] for k in ("horizon", "scheme", "side", "first", "second"))
        ]
        assert len(match) == 1
        for key, value in match[0].items():
            assert same(row[key], value), (key, row, value)
        if row["complete_draws"] == 1.0:
            assert all(same(row[f"{k}_complete"], row[k]) for k in compare.COMPLETE)
    keys = ("horizon", "scheme", "side", "name", "coins", "fold")
    their_folds = {
        tuple(r[k] for k in keys): r
        for r in pq.read_table(ladder_run / "folds.parquet").to_pylist()
    }
    my_folds = pq.read_table(one / "folds.parquet").to_pylist()
    assert {r["name"] for r in my_folds} == {"full", "vol"} and my_folds
    for row in my_folds:
        assert same(row, their_folds[tuple(row[k] for k in keys)])
    their_scores = {
        tuple(r[k] for k in ("horizon", "scheme", "side", "name", "coins")): r
        for r in pq.read_table(ladder_run / "scores.parquet").to_pylist()
    }
    for row in pq.read_table(one / "scores.parquet").to_pylist():
        assert same(
            row, their_scores[tuple(row[k] for k in ("horizon", "scheme", "side", "name", "coins"))]
        )
    assert pq.read_table(one / "rows.parquet").equals(pq.read_table(ladder_run / "rows.parquet"))
    recorded = params(one)
    assert recorded["primary"] == {
        "first": "full",
        "second": "vol",
        "scheme": "forward",
        "side": "long",
        "horizon": 24,
    }
    assert recorded["schemes"] == ["loqo", "forward"] and recorded["pairs"] == [["full", "vol"]]
    assert recorded["inputs_run"] == panel["inputs"].name and recorded["labels_run"] == LABELS
    assert recorded["inputs"]["sha256"] == store.sha256(panel["inputs"] / "inputs.parquet")
    assert recorded["ladder_plan"]["sha256"] == store.sha256(one / "ladder.toml")
    assert recorded["plan"]["sha256"] == store.sha256(one / "plan.toml")
    for name, run in arms.items():
        arm = recorded["arms"][name]
        source = params(run)
        complete = record(run, "complete.json")
        assert arm["run"] == run.name and arm["kind"] == "classical" and arm["export"] == name
        assert (arm["model"], arm["variant"], arm["readout"], arm["c"]) == (
            "rung",
            name,
            None,
            0.01,
        )
        assert arm["fixed"] == source["fixed"] and arm["selection"] is None
        assert arm["chosen"] is False
        assert arm["made_from"] == {
            "run": run.name,
            **{k: source[k] for k in ("inputs", "labels_grid", "ladder_plan")},
        }
        assert arm["code"] == {
            "git_sha": "c" * 40,
            "dirty": False,
            "uv_lock_sha256": source["code"]["uv_lock_sha256"],
        }
        for scheme in ("loqo", "forward"):
            file = f"{name}-{scheme}.parquet"
            assert arm["files"][scheme] == {"file": file, "sha256": complete["files"][file]}
    report = (one / "report.md").read_text(encoding="utf-8")
    primary = report.split("## Primary")[1].split("## Secondary")[0]
    assert primary.count("| full - vol | forward | 24 h | long |") == 1
    assert primary.count("| full - vol (complete draws) | forward | 24 h | long |") == 1
    assert "| full - vol | forward | 24 h | long |" not in report.split("## Secondary")[1]
    arms_table = report.split("## Arms")[1]
    planned = record(panel["full"], "complete.json")["files"]["plan.toml"][:12]
    assert arms_table.count(f"| fixed | - | 0.01 | {planned} | cccccccccccc (clean) |") == 2
    assert "Every arm's runs were made by the same commit and uv.lock." in arms_table


def test_every_arm_takes_the_same_draws(panel: dict[str, Any], tmp_path: Path) -> None:
    text = PAIRED.replace(
        'export = "vol"',
        'export = "vol"\n\n[[arms]]\nname = "again"\nkind = "classical"\nexport = "full"',
    )
    text = text.replace('first = "full"\nsecond = "vol"', 'first = "vol"\nsecond = "full"')
    text += '\n[[pairs]]\nfirst = "full"\nsecond = "again"\n'
    plan = folder(tmp_path / "plan", {"ladder.toml": LADDER, "compare.toml": text}) / "compare.toml"
    arms = [("full", str(panel["full"])), ("vol", str(panel["vol"])), ("again", str(panel["full"]))]
    swapped, _ = compare.run(arms, panel["inputs"], panel["data"], plan, "swapped")
    ordered, _ = compare.run(arms[:2], panel["inputs"], panel["data"], panel["paired"], "ordered")
    rows = [
        *pq.read_table(swapped / "comparisons.parquet").to_pylist(),
        *pq.read_table(ordered / "comparisons.parquet").to_pylist(),
    ]
    keyed = {(r["scheme"], r["horizon"], r["side"], r["first"], r["second"]): r for r in rows}
    assert len(keyed) == len(rows) == 3 * 2 * 2 * 4
    checked = 0
    for (scheme, h, side, first, second), row in keyed.items():
        if (first, second) == ("full", "again"):
            assert row["estimate"] == row["se"] == row["above_zero"] == 0.0
            assert row["q025"] == row["q05"] == row["q975"] == 0.0
            assert all(v == 0.0 for v in row["fold_differences"])
        if (first, second) == ("vol", "full"):
            other = keyed[(scheme, h, side, "full", "vol")]
            assert row["estimate"] == -other["estimate"]
            assert row["fold_differences"] == [-v for v in other["fold_differences"]]
            assert row["se"] == other["se"] and row["draws"] == other["draws"]
            assert row["q025"] == pytest.approx(-other["q975"], abs=1e-15)
            assert row["q975"] == pytest.approx(-other["q025"], abs=1e-15)
            assert row["above_zero"] + other["above_zero"] == pytest.approx(1.0)
            checked += 1
    assert checked == 2 * 2 * 4


def test_the_composite_side_pairs_long_and_short_draw_by_draw(
    panel: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = plan_file(tmp_path, PAIRED.replace('side = "long"', 'side = "long+short"'))
    calls: list[tuple[Any, ...]] = []
    seen: dict[tuple[str, int], dict[str, Any]] = {}
    draws, evaluate = ladder.draws, ladder.evaluate

    def spy(h: ladder.Horizon, base: ladder.Plan, horizon: int, *args: Any) -> Any:
        before = len(calls)
        out = evaluate(h, base, horizon, *args)
        assert calls[before:] == [(base.draws, base.seed, horizon)]
        seen[(h.scheme, horizon)] = args[2]
        return out

    monkeypatch.setattr(ladder, "draws", lambda *a: calls.append(a[1:]) or draws(*a))
    monkeypatch.setattr(ladder, "evaluate", spy)
    arms = [("full", str(panel["full"])), ("vol", str(panel["vol"]))]
    run_dir, _ = compare.run(arms, panel["inputs"], panel["data"], plan, "composite")
    rows = pq.read_table(run_dir / "comparisons.parquet").to_pylist()
    assert len(seen) == len(calls) == 2 * 2
    composite = [r for r in rows if r["side"] == "long+short"]
    assert len(composite) == 2 * 2
    for row in composite:
        kept = seen[(row["scheme"], row["horizon"])]
        boot, folds, whole = kept["boot"], kept["folds"], kept["complete"]
        sides = {
            r["side"]: r
            for r in rows
            if (r["scheme"], r["horizon"]) == (row["scheme"], row["horizon"])
        }
        assert row["estimate"] == (sides["long"]["estimate"] + sides["short"]["estimate"]) / 2
        d = (
            (boot[("full", "long")] - boot[("vol", "long")])
            + (boot[("full", "short")] - boot[("vol", "short")])
        ) / 2
        d = d[np.isfinite(d)]
        assert row["se"] == float(np.std(d, ddof=1)) and row["draws"] == d.size
        assert [row["q025"], row["q05"], row["q975"]] == list(np.quantile(d, ladder.QUANTILES))
        assert row["above_zero"] == float(np.mean(d > 0))
        a = (folds[("full", "long")] + folds[("full", "short")]) / 2
        b = (folds[("vol", "long")] + folds[("vol", "short")]) / 2
        assert row["fold_differences"] == [float(v) for v in a - b]
        assert row["complete_draws"] == float((whole["long"] & whole["short"]).mean())
        assert row["primary"] is ((row["scheme"], row["horizon"]) == ("forward", 24))
        unpaired = (boot[("full", "long")] - boot[("vol", "long")]) + np.roll(
            boot[("full", "short")] - boot[("vol", "short")], 1
        )
        assert np.std(unpaired / 2, ddof=1) != row["se"]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "| full - vol | forward | 24 h | long+short |" in report.split("## Secondary")[0]


def test_complete_draw_statistics_leave_out_draws_that_lose_a_quarter(tmp_path: Path) -> None:
    base = ladder.load(tmp_path / "ladder.toml", LADDER)
    hours = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(365 * 24, dtype=np.int64)
    fold, block = ladder.folds(hours, ladder.bounds(base), base.block_days)
    first = base.first_fold("forward")
    mine = fold >= first
    ts, fold, block = hours[mine], fold[mine], block[mine]
    rng = np.random.default_rng(4)
    long, short = np.zeros(ts.size, bool), np.zeros(ts.size, bool)
    for f in np.unique(fold):
        quarter = fold == f
        long[np.flatnonzero(quarter & (block == block[quarter].max()))[:5]] = True
        short[np.flatnonzero(quarter & (block == block[quarter].min()))[:5]] = True
    y = {"long": long, "short": short, "any": long | short}
    scores = {n: {s: rng.normal(size=ts.size) for s in y} for n in ("a", "b")}
    symbol = np.full(ts.size, "AAAUSDT", dtype=object)
    h = ladder.Horizon("forward", first, ts, symbol, y, fold, block, scores)
    _, _, rows = compare.evaluated(h, base, 24, ("a", "b"), (("a", "b"),))
    by = {r["side"]: r for r in rows}
    for side in ("long", "short", "long+short"):
        row = by[side]
        assert 0 < row["complete_draws"] < 1
        assert row["draws_complete"] == round(row["complete_draws"] * base.draws)
        assert row["draws_complete"] < row["draws"] <= base.draws
        assert row["se_complete"] != row["se"]
    sides = (by["long"]["complete_draws"], by["short"]["complete_draws"])
    assert by["long+short"]["complete_draws"] < min(sides)


def test_the_composite_scores_only_the_quarters_both_sides_score(tmp_path: Path) -> None:
    base = ladder.load(tmp_path / "ladder.toml", LADDER)
    hours = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(365 * 24, dtype=np.int64)
    fold, block = ladder.folds(hours, ladder.bounds(base), base.block_days)
    first = base.first_fold("forward")
    mine = fold >= first
    ts, fold, block = hours[mine], fold[mine], block[mine]
    quarters = np.unique(fold)
    assert len(quarters) == 2
    rng = np.random.default_rng(5)
    long = rng.random(ts.size) < 0.05
    short = (rng.random(ts.size) < 0.05) & (fold != quarters[0])
    y = {"long": long, "short": short, "any": long | short}
    lift = {"a": np.where(fold == quarters[0], 3.0, 0.3), "b": np.zeros(ts.size)}
    scores = {n: {s: rng.normal(size=ts.size) + lift[n] * y[s] for s in y} for n in lift}
    symbol = np.full(ts.size, "AAAUSDT", dtype=object)
    h = ladder.Horizon("forward", first, ts, symbol, y, fold, block, scores)
    kept: dict[str, Any] = {}
    ladder.evaluate(h, base, 24, ("a", "b"), (("a", "b"),), kept)
    _, _, rows = compare.evaluated(h, base, 24, ("a", "b"), (("a", "b"),))
    by = {r["side"]: r for r in rows}
    counts = {side: (r["folds"], r["quarters_scheduled"]) for side, r in by.items()}
    assert counts == {"long": (2, 2), "short": (1, 2), "any": (2, 2), "long+short": (1, 2)}
    assert by["short"]["complete_draws"] == 1.0
    composite = by["long+short"]
    (only,) = [d for d in composite["fold_differences"] if math.isfinite(d)]
    assert composite["estimate"] == pytest.approx(only, rel=1e-12)
    halves = (by["long"]["estimate"] + by["short"]["estimate"]) / 2
    assert abs(composite["estimate"] - halves) > 0.05
    booted = kept["booted"]
    long_d = booted[("a", "long")][:, 1] - booted[("b", "long")][:, 1]
    short_d = booted[("a", "short")][:, 1] - booted[("b", "short")][:, 1]
    d = (long_d + short_d) / 2
    d = d[np.isfinite(d)]
    assert composite["draws"] == d.size
    assert composite["se"] == pytest.approx(float(np.std(d, ddof=1)), rel=1e-12)
    assert composite["q05"] == pytest.approx(float(np.quantile(d, 0.05)), rel=1e-12)
    both = np.isfinite(booted[("a", "long")][:, 1]) & np.isfinite(booted[("a", "short")][:, 1])
    assert composite["complete_draws"] == float(both.mean())
    apart = {"long": long & (fold == quarters[0]), "short": short}
    apart["any"] = apart["long"] | apart["short"]
    noise = {n: {s: rng.normal(size=ts.size) for s in apart} for n in lift}
    h = ladder.Horizon("forward", first, ts, symbol, apart, fold, block, noise)
    _, _, rows = compare.evaluated(h, base, 24, ("a", "b"), (("a", "b"),))
    (none,) = [r for r in rows if r["side"] == "long+short"]
    assert (none["folds"], none["draws"], none["draws_complete"]) == (0, 0, 0)
    assert all(math.isnan(none[k]) for k in ("estimate", "se", "q05", "complete_draws"))


def test_a_sequence_arm_and_a_window_arm_compare_end_to_end(small: dict[str, Any]) -> None:
    data, scores, seq = small["data"], small["scores"], small["seq"]
    arms = {"jepa": Path(scores.name), "window": small["window"]}
    assert main(run_args(small, small["paired"], "seq", **arms)) == 0
    (run_dir,) = tag_runs(data, "seq")
    rows = pq.read_table(run_dir / "comparisons.parquet").to_pylist()
    assert len(rows) == 2 * 4 and {r["scheme"] for r in rows} == {"forward"}
    (primary,) = [r for r in rows if r["primary"]]
    assert (primary["side"], primary["horizon"], primary["folds"]) == ("any", 24, 1)
    recorded = params(run_dir)["arms"]
    jepa = recorded["jepa"]
    assert jepa["sequence_run"] == seq.name and jepa["made_from"]["run"] == seq.name
    assert (jepa["model"], jepa["variant"], jepa["seeds"]) == ("jepa", "vol", [3, 4])
    assert jepa["readout"] == params(scores)["readout"] and jepa["c"] == params(scores)["c"]
    assert jepa["selection"] == params(scores)["selection"] and jepa["chosen"] is True
    clean = {"git_sha": "c" * 40, "dirty": False}
    assert (
        jepa["sequence_code"]
        == jepa["trained_by"]
        == {
            **clean,
            "uv_lock_sha256": params(seq)["code"]["uv_lock_sha256"],
        }
    )
    assert recorded["window"]["model"] == "window" and recorded["window"]["chosen"] is False
    assert jepa["bound"] is recorded["window"]["bound"] is params(run_dir)["inputs_bound"] is True
    assert (primary["folds"], primary["quarters_scheduled"]) == (1, 2)
    text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "scores 1 of the 2 quarters scheduled for forward" in text.split("## Secondary")[0]
    report = text.split("## Arms")[1]
    assert "| jepa | vol | chosen by the rule |" in report and "| window | vol | fixed |" in report
    assert report.count(" | yes |") == 2 and "The inputs run is bound." in report
    file = (scores / "forward.parquet").as_posix()
    external = f'\n[external]\nname = "jepa"\npath = "{file}"\nscheme = "forward"\n'
    (data / "ladder-ext.toml").write_text(LADDER + external, encoding="utf-8")
    ladder_run, _ = ladder.run(small["inputs"], data, data / "ladder-ext.toml", "ext")
    keys = ("horizon", "scheme", "side", "name", "coins", "fold")
    theirs = {
        tuple(r[k] for k in keys): r
        for r in pq.read_table(ladder_run / "folds.parquet").to_pylist()
    }
    mine = [r for r in pq.read_table(run_dir / "folds.parquet").to_pylist() if r["name"] == "jepa"]
    assert mine and all(same(r, theirs[tuple(r[k] for k in keys)]) for r in mine)


def test_a_scores_arm_that_is_not_its_sequence_runs_models_is_refused(
    small: dict[str, Any], tmp_path: Path
) -> None:
    data, scores, seq, path = small["data"], small["scores"], small["seq"], small["paired"]
    saved, used = params(seq)["encoders"], params(scores)["encoders"]
    assert len(used) == 2

    def sequence_copy(suffix: str, change: Callable[[Path], Any] = lambda d: None) -> Path:
        dest = data / "panel-sequence" / f"{seq.name}-{suffix}"
        shutil.copytree(seq, dest)
        relabel(dest)
        change(dest)
        return dest

    def refused(change: Callable[[Path], Any], message: str, plan_path: Path = path) -> None:
        copy = copied(scores, tmp_path / "copies")
        change(copy)
        with pytest.raises(ValueError, match=message):
            compare.run(
                [("jepa", str(copy)), ("window", str(small["window"]))],
                small["inputs"],
                data,
                plan_path,
                "x",
            )

    def via(run: Path) -> Callable[[Path], Any]:
        return lambda d: edit(d, "manifest.json", {"sequence_run": run.name})

    gone = "20260101T000000Z-gone"
    refused(lambda d: edit(d, "manifest.json", {"sequence_run": gone}), f"no run at .*{gone}")
    for value in (f"../panel-classical/{small['window'].name}", "..", 5, None):
        refused(lambda d, v=value: edit(d, "manifest.json", {"sequence_run": v}), "is not a run id")
    moved = sequence_copy("moved", lambda d: (d / "complete.json").unlink())
    refused(via(moved), "the sequence run of .* has no complete.json")
    unclean = sequence_copy("unclean", lambda d: edit(d, "complete.json", {"clean": False}))
    refused(via(unclean), "does not mark .* clean")
    named = data / "panel-sequence" / f"{seq.name}-named"
    shutil.copytree(seq, named)
    refused(via(named), f"{named.name} records run id {seq.name}, not its folder's")
    kind = sequence_copy(
        "kind", lambda d: rewrite(d, "manifest.json", lambda r: r.update(domain="panel-classical"))
    )
    refused(via(kind), "is a panel-classical run, not a panel-sequence run")
    refused(
        lambda d: edit(d, "manifest.json", {"encoders": {k: "0" * 64 for k in used}}),
        "scored are not those of sequence run",
    )
    twin = {
        k.replace("-jepa.pt", "-twin.pt"): saved[k.replace("-jepa.pt", "-twin.pt")] for k in used
    }
    refused(lambda d: edit(d, "manifest.json", {"encoders": twin}), "are not every jepa on vol")
    first = sorted(used)[0]
    partial = {k: v for k, v in used.items() if k != first}
    refused(lambda d: edit(d, "manifest.json", {"encoders": partial}), "0 others, 1 missing")
    extra = first.replace("-jepa.pt", "-supervised.pt")
    more = {**used, extra: saved[extra]}
    refused(lambda d: edit(d, "manifest.json", {"encoders": more}), "1 others, 0 missing")
    full = path.read_text(encoding="utf-8").replace('variant = "vol"', 'variant = "full"')
    relabelled = plan_file(tmp_path / "plans", full)
    refused(
        lambda d: edit(d, "manifest.json", {"variant": "full"}),
        "not every jepa on full",
        relabelled,
    )
    missing = sequence_copy("missing", lambda d: (d / first).unlink())
    refused(via(missing), f"{first.split('/')[1]} is missing")
    bent = sequence_copy("bent", lambda d: (d / first).write_bytes(b"x" + (d / first).read_bytes()))
    refused(via(bent), f"{first.split('/')[1]}: its sha256 differs from complete.json's")
    stale = sequence_copy(
        "stale", lambda d: (d / "ladder.toml").write_text(LADDER + "\n", encoding="utf-8")
    )
    refused(via(stale), "ladder.toml: its sha256 differs from complete.json's")
    planned = sequence_copy(
        "planned", lambda d: (d / "plan.toml").write_text("x = 1\n", encoding="utf-8")
    )
    refused(via(planned), "plan.toml: its sha256 differs from complete.json's")
    seeded = LADDER.replace("seed = 1", "seed = 2")

    def reseed(d: Path) -> None:
        (d / "ladder.toml").write_bytes(seeded.encode("utf-8"))
        digest = store.sha256(d / "ladder.toml")
        rewrite(d, "complete.json", lambda r: r["files"].update({"ladder.toml": digest}))
        edit(d, "manifest.json", {"ladder_plan": {"file": "ladder.toml", "sha256": digest}})

    refused(via(sequence_copy("seeded", reseed)), "ladder.toml is not the comparison's ladder plan")
    dirty = sequence_copy("dirty", lambda d: producer(d, {"dirty": True}))
    refused(via(dirty), f"{dirty.name}, was made by code with uncommitted changes")
    unknown = sequence_copy("unknown", lambda d: producer(d, {"git_sha": None}))
    refused(via(unknown), f"{unknown.name}, records no commit")
    pieces = {"name": "tipping_point.panel", "git_sha": "e" * 40, "dirty": True}
    merged = sequence_copy(
        "merged", lambda d: edit(d, "manifest.json", {"sources": [], "pieces_producer": pieces})
    )
    refused(via(merged), "the runs merged into .* was made by code with uncommitted changes")
    refused(lambda d: producer(d, {"dirty": None}), "was made by code with uncommitted changes")
    refused(lambda d: producer(d, {"git_sha": ""}), "records no commit for its code")
    refused(
        lambda d: edit(d, "manifest.json", {"model": "twin"}), "scores twin on vol, not jepa on vol"
    )
    refused(
        lambda d: edit(d, "manifest.json", {"variant": "full"}),
        "scores jepa on full, not jepa on vol",
    )
    other = sequence_copy(
        "other",
        lambda d: edit(
            d, "manifest.json", {"inputs": {"file": "inputs.parquet", "sha256": "0" * 64}}
        ),
    )
    refused(via(other), f"{other.name} read another inputs table")
    twin_plan = plan_file(
        tmp_path / "plans",
        path.read_text(encoding="utf-8").replace('model = "jepa"', 'model = "twin"'),
    )
    refused(lambda d: None, "scores jepa on vol, not twin on vol", twin_plan)
    with pytest.raises(ValueError, match="is a panel-classical run, not a panel-scores run"):
        compare.run(
            [("jepa", str(small["window"])), ("window", str(small["window"]))],
            small["inputs"],
            data,
            path,
            "x",
        )
    with pytest.raises(ValueError, match="is a panel-scores run, not a panel-classical run"):
        compare.run(
            [("jepa", str(scores)), ("window", str(scores))], small["inputs"], data, path, "x"
        )
    assert not tag_runs(data, "x")
    clean = {"name": "tipping_point.panel", "git_sha": "e" * 40, "dirty": False}
    trained = sequence_copy(
        "trained", lambda d: edit(d, "manifest.json", {"sources": [], "pieces_producer": clean})
    )
    copy = copied(scores, tmp_path / "copies")
    via(trained)(copy)
    run_dir, _ = compare.run(
        [("jepa", str(copy)), ("window", str(small["window"]))],
        small["inputs"],
        data,
        path,
        "merged",
    )
    jepa = params(run_dir)["arms"]["jepa"]
    assert (
        jepa["trained_by"]["git_sha"] == "e" * 40 and jepa["sequence_code"]["git_sha"] == "c" * 40
    )
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert f"jepa (trained by) {'e' * 40}, clean" in report


def test_pinned_configurations_and_a_classical_arm_chosen_by_the_rule(
    small: dict[str, Any], tmp_path: Path
) -> None:
    data, scores, chosen = small["data"], small["scores"], small["chosen"]
    readout, c = params(scores)["readout"], params(scores)["c"]
    config = params(chosen)["exported"]["vol"]
    planned = record(small["seq"], "complete.json")["files"]["plan.toml"]
    fitted = record(chosen, "complete.json")["files"]["plan.toml"]
    pins = f'readout = "{readout}"\nc = {c!r}\nseeds = [4, 3]\nplan_sha256 = "{planned}"\n'
    window = f'model = "{config["model"]}"\nchosen = true\nplan_sha256 = "{fitted}"\n'
    if config["readout"]:
        window += f'readout = "{config["readout"]}"\nc = {config["c"]!r}\n'
    text = SMALL.replace('variant = "vol"\n', f'variant = "vol"\n{pins}').replace(
        'export = "vol"\n', f'export = "vol"\n{window}'
    )
    arms = [("jepa", str(scores)), ("window", str(chosen))]
    run_dir, _ = compare.run(arms, small["inputs"], data, plan_file(tmp_path, text), "pinned")
    recorded = params(run_dir)["arms"]["window"]
    selection = params(chosen)["selection"]
    assert recorded["chosen"] is True and recorded["fixed"] is None
    assert recorded["selection"] == {"metric": selection["metric"], **selection["chosen"]["vol"]}
    assert (recorded["model"], recorded["readout"], recorded["c"]) == (
        config["model"],
        config["readout"],
        config["c"],
    )
    assert params(run_dir)["arms"]["jepa"]["plan_sha256"] == planned
    assert recorded["plan_sha256"] == fitted
    report = (run_dir / "report.md").read_text(encoding="utf-8").split("## Arms")[1]
    assert report.count("| chosen by the rule |") == 2
    assert f"| {planned[:12]} |" in report and f"| {fitted[:12]} |" in report
    other = "0" * 64
    for old, new, message in (
        (
            f'seeds = [4, 3]\nplan_sha256 = "{planned}"',
            f'seeds = [4, 3]\nplan_sha256 = "{other}"',
            f"jepa: {small['scores'].name} has plan_sha256 {planned}, not the plan's {other}",
        ),
        (
            f'chosen = true\nplan_sha256 = "{fitted}"',
            f'chosen = true\nplan_sha256 = "{other}"',
            f"window: {chosen.name} has plan_sha256 {fitted}, not the plan's {other}",
        ),
        (
            f'readout = "{readout}"\nc = {c!r}\nseeds',
            f'readout = "other"\nc = {c!r}\nseeds',
            f"has readout {readout}, not the plan's other",
        ),
        (f"c = {c!r}\nseeds", "c = 123.0\nseeds", f"has c {c}, not the plan's 123.0"),
        ("seeds = [4, 3]", "seeds = [3]", "has seeds \\[3, 4\\], not the plan's \\[3\\]"),
        ("chosen = true", "chosen = false", "has chosen True, not the plan's False"),
        (
            f'model = "{config["model"]}"',
            'model = "rung"' if config["model"] == "window" else 'model = "window"',
            "has model",
        ),
    ):
        assert text.count(old) == 1, old
        with pytest.raises(ValueError, match=message):
            compare.run(
                arms, small["inputs"], data, plan_file(tmp_path, text.replace(old, new)), "x"
            )
    fixed_arms = [("jepa", str(scores)), ("window", str(small["window"]))]
    only = SMALL.replace('export = "vol"\n', 'export = "vol"\nchosen = true\n')
    with pytest.raises(ValueError, match="has chosen False, not the plan's True"):
        compare.run(fixed_arms, small["inputs"], data, plan_file(tmp_path, only), "x")
    assert not tag_runs(data, "x")


def test_an_arm_from_another_inputs_run_labels_grid_or_ladder_plan_is_refused(
    panel: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, values, grid = panel["data"], panel["values"], panel["grid"]
    changed = {k: v.copy() for k, v in values.items()}
    at = int(np.flatnonzero(np.isfinite(changed["vol_1h"]))[500])
    changed["vol_1h"][at] += 1.0
    other_inputs = write_inputs(data, changed, grid, LABELS, "changed")
    from_other = fixed(
        data, other_inputs, panel["plan"], "from-changed", Config("rung", "vol", None, 0.01)
    )
    flipped = dict(grid, long_24=grid["long_24"].copy())
    flipped["long_24"][1000] = ~flipped["long_24"][1000]
    write_labels(data, flipped, "20260101T000000Z-flipped")
    other_grid = write_inputs(data, values, flipped, "20260101T000000Z-flipped", "flipped")
    from_grid = fixed(
        data, other_grid, panel["plan"], "from-flipped", Config("rung", "vol", None, 0.01)
    )
    seeded = LADDER.replace("seed = 1", "seed = 2")
    other_plan = (
        folder(tmp_path / "seeded", {"ladder.toml": seeded, "sequence.toml": PLAN})
        / "sequence.toml"
    )
    from_plan = fixed(
        data, panel["inputs"], other_plan, "from-seeded", Config("rung", "vol", None, 0.01)
    )
    monkeypatch.setattr(ladder, "external_table", lambda *a: pytest.fail("read an arm's scores"))
    monkeypatch.setattr(compare, "score_bytes", lambda *a: pytest.fail("read an arm's score file"))
    ours = params(panel["vol"])

    def refused(run: Path, change: dict[str, Any], message: str) -> None:
        copy = copied(run, tmp_path / "copies")
        edit(copy, "manifest.json", change)
        arms = [("full", str(panel["full"])), ("vol", str(copy))]
        with pytest.raises(ValueError, match=message):
            compare.run(arms, panel["inputs"], data, panel["paired"], "x")

    refused(from_other, {}, f"made from inputs run {other_inputs.name}, not {panel['inputs'].name}")
    refused(
        from_other,
        {"inputs_run": panel["inputs"].name},
        "read another inputs table than inputs run",
    )
    refused(from_grid, {}, f"made from inputs run {other_grid.name}")
    refused(
        from_grid,
        {"inputs_run": panel["inputs"].name},
        "made from labels run 20260101T000000Z-flipped",
    )
    refused(
        from_grid,
        {"inputs_run": panel["inputs"].name, "labels_run": LABELS, "inputs": ours["inputs"]},
        f"read another labels grid than labels run {LABELS}",
    )
    refused(from_plan, {}, "ladder.toml is not the comparison's ladder plan")
    refused(
        panel["vol"],
        {"ladder_plan": {"file": "ladder.toml", "sha256": "0" * 64}},
        "manifest records another ladder.toml",
    )
    assert not tag_runs(data, "x")


def test_an_arm_that_is_not_its_run_or_not_the_plans_arm_is_refused(
    panel: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = panel["data"]
    monkeypatch.setattr(ladder, "external_table", lambda *a: pytest.fail("read an arm's scores"))

    def refused(change: Callable[[Path], Any], message: str) -> None:
        copy = copied(panel["vol"], tmp_path / "copies")
        change(copy)
        with pytest.raises(ValueError, match=message):
            compare.run(
                [("full", str(panel["full"])), ("vol", str(copy))],
                panel["inputs"],
                data,
                panel["paired"],
                "x",
            )

    refused(
        lambda d: edit(d, "complete.json", {"clean": False}),
        "arm vol: complete.json does not mark .* clean",
    )
    refused(lambda d: (d / "complete.json").unlink(), "arm vol: .* has no complete.json")
    refused(
        lambda d: rewrite(d, "manifest.json", lambda r: r.update(run_id="20260101T000000Z-vol")),
        "arm vol: .* records run id 20260101T000000Z-vol, not its folder's",
    )
    refused(lambda d: producer(d, {"dirty": True}), "arm vol: .* uncommitted changes")
    refused(lambda d: producer(d, {"git_sha": None}), "arm vol: .* records no commit")
    table = pq.read_table(panel["vol"] / "vol-forward.parquet")
    nudged = table.set_column(4, "score", pa.array(table["score"].to_numpy() + 1e-12))
    refused(
        lambda d: pq.write_table(nudged, d / "vol-forward.parquet"),
        "arm vol: .*vol-forward.parquet: its sha256 differs from complete.json's",
    )
    refused(
        lambda d: (d / "vol-forward.parquet").unlink(), "arm vol: .*vol-forward.parquet is missing"
    )
    refused(lambda d: (d / "plan.toml").unlink(), "arm vol: .*plan.toml is missing")
    refused(
        lambda d: (d / "ladder.toml").write_text(LADDER + "\n", encoding="utf-8"),
        "arm vol: .*ladder.toml: its sha256 differs from complete.json's",
    )
    refused(
        lambda d: (d / "plan.toml").write_text(PLAN + "\n", encoding="utf-8"),
        "arm vol: .*plan.toml: its sha256 differs from complete.json's",
    )
    refused(
        lambda d: edit(d, "manifest.json", {"plan": {"file": "sequence.toml", "sha256": "0" * 64}}),
        "arm vol: .* manifest records another plan.toml",
    )
    refused(
        lambda d: edit(d, "manifest.json", {"exported": {"full": params(d)["exported"]["vol"]}}),
        "arm vol: .* exported no vol",
    )
    refused(_domain, "is a panel-ladder run, not a panel-classical run")
    refused(lambda d: _drop(d, "vol-forward.parquet"), "holds no forward scores")
    arms = {"full": str(panel["full"]), "vol": str(panel["vol"])}
    for given, message in (
        ([*arms.items(), ("full", arms["full"])], "arm full given more than once"),
        ([("full", arms["full"])], "no run given for arm vol"),
        ([*arms.items(), ("other", arms["vol"])], "the plan has no arm other"),
    ):
        with pytest.raises(ValueError, match=message):
            compare.run(given, panel["inputs"], data, panel["paired"], "x")
    with pytest.raises(ValueError, match="no inputs run at"):
        compare.run(list(arms.items()), tmp_path / "none", data, panel["paired"], "x")
    assert main(run_args(panel, panel["paired"], "x", full=panel["full"])) == 1
    assert not tag_runs(data, "x")


def test_arms_made_by_different_clean_code_are_compared_and_reported(
    panel: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = panel["data"]
    monkeypatch.setattr(io, "code_version", lambda: {"git_sha": "d" * 40, "dirty": False})
    later = fixed(data, panel["inputs"], panel["plan"], "later", Config("rung", "vol", None, 0.01))
    arms = [("full", str(panel["full"])), ("vol", str(later))]
    run_dir, _ = compare.run(arms, panel["inputs"], data, panel["paired"], "later")
    recorded = params(run_dir)["arms"]
    assert recorded["full"]["code"]["git_sha"] == "c" * 40
    assert recorded["vol"]["code"]["git_sha"] == "d" * 40
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "were not all made by the same code" in report
    assert f"vol {'d' * 40}, clean" in report and f"full {'c' * 40}, clean" in report
    assert "dddddddddddd (clean)" in report
    states = [compare._state(v) for v in (False, True, None, "no")]
    assert states == ["clean", "uncommitted changes", "unknown", "unknown"]


def test_a_manifest_changed_after_its_run_was_written_is_refused(
    panel: dict[str, Any], small: dict[str, Any], tmp_path: Path
) -> None:
    data, values, grid = panel["data"], panel["values"], panel["grid"]
    for run in (panel["full"], panel["vol"], panel["inputs"], small["seq"], small["scores"]):
        assert record(run, "complete.json")["manifest_sha256"] == store.sha256(
            run / "manifest.json"
        )
    flipped = dict(grid, long_24=grid["long_24"].copy())
    flipped["long_24"][1000] = ~flipped["long_24"][1000]
    write_labels(data, flipped, "20260101T000000Z-bound")
    other = write_inputs(data, values, flipped, "20260101T000000Z-bound", "bound")
    fitted = fixed(data, other, panel["plan"], "from-bound", Config("rung", "vol", None, 0.01))
    ours = params(panel["vol"])
    forged = copied(fitted, tmp_path / "copies")
    tamper(forged, {k: ours[k] for k in ("inputs_run", "labels_run", "inputs", "labels_grid")})

    def arms(vol: Path) -> list[tuple[str, str]]:
        return [("full", str(panel["full"])), ("vol", str(vol))]

    with pytest.raises(ValueError, match=r"arm vol: .*manifest\.json: its sha256 differs"):
        compare.run(arms(forged), panel["inputs"], data, panel["paired"], "x")
    moved = copied(panel["inputs"], tmp_path / "inputs")
    tamper(moved, {"note": 1})
    with pytest.raises(ValueError, match=r"inputs run: .*manifest\.json: its sha256 differs"):
        compare.run(arms(panel["vol"]), moved, data, panel["paired"], "x")
    seq, path, data_small = small["seq"], small["paired"], small["data"]

    def sequence_copy(suffix: str, change: Callable[[Path], Any]) -> Path:
        dest = data_small / "panel-sequence" / f"{seq.name}-{suffix}"
        shutil.copytree(seq, dest)
        relabel(dest)
        change(dest)
        scores = copied(small["scores"], tmp_path / "scores")
        edit(scores, "manifest.json", {"sequence_run": dest.name})
        return scores

    changed = sequence_copy("changed", lambda d: tamper(d, {"threads": 2}))
    with pytest.raises(
        ValueError, match=r"the sequence run of .*manifest\.json: its sha256 differs"
    ):
        compare.run(
            [("jepa", str(changed)), ("window", str(small["window"]))],
            small["inputs"],
            data_small,
            path,
            "x",
        )
    assert not tag_runs(data, "x") and not tag_runs(data_small, "x")
    old = copied(panel["vol"], tmp_path / "copies")
    unbind(old)
    unbound_inputs = copied(panel["inputs"], tmp_path / "inputs")
    unbind(unbound_inputs)
    run_dir, _ = compare.run(arms(old), unbound_inputs, data, panel["paired"], "unbound")
    recorded = params(run_dir)
    assert recorded["arms"]["full"]["bound"] is True and recorded["arms"]["vol"]["bound"] is False
    assert recorded["inputs_bound"] is False
    report = (run_dir / "report.md").read_text(encoding="utf-8").split("## Arms")[1]
    assert report.count(" | yes |") == report.count(" | no |") == 1
    assert "The inputs run is not bound." in report
    unbound = sequence_copy("unbound", unbind)
    run_dir, _ = compare.run(
        [("jepa", str(unbound)), ("window", str(small["window"]))],
        small["inputs"],
        data_small,
        path,
        "unbound",
    )
    recorded = params(run_dir)["arms"]
    assert recorded["jepa"]["bound"] is False and recorded["window"]["bound"] is True


def test_an_inputs_run_with_a_missing_file_is_refused_by_name(
    panel: dict[str, Any], tmp_path: Path
) -> None:
    arms = [("full", str(panel["full"])), ("vol", str(panel["vol"]))]

    def refused(change: Callable[[Path], Any], message: str) -> None:
        copy = copied(panel["inputs"], tmp_path / "inputs")
        change(copy)
        with pytest.raises(ValueError, match=message):
            compare.run(arms, copy, panel["data"], panel["paired"], "x")

    refused(lambda d: (d / "complete.json").unlink(), "inputs run: .*complete.json is missing")
    refused(
        lambda d: edit(d, "manifest.json", {"labels_grid": {"file": "grid/9.parquet"}}),
        "inputs run: .*9.parquet is missing",
    )
    refused(lambda d: (d / "inputs.parquet").unlink(), "inputs run: .*inputs.parquet is missing")
    assert not tag_runs(panel["data"], "x")


def _domain(run: Path) -> None:
    rewrite(run, "manifest.json", lambda r: r.update(domain="panel-ladder"))


def _drop(run: Path, name: str) -> None:
    rewrite(run, "complete.json", lambda r: r["files"].pop(name))


def test_rows_that_are_not_exactly_the_ladders_are_refused(
    panel: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = panel["data"]
    name = "vol-forward.parquet"
    table = pq.read_table(panel["vol"] / name)
    at = np.flatnonzero(
        (table["horizon"].to_numpy() == 24)
        & (table["side"].to_numpy(zero_copy_only=False) == "long")
    )
    row = table.slice(int(at[10]), 1)

    def column(t: pa.Table, key: str, values: Any) -> pa.Table:
        return t.set_column(
            t.schema.get_field_index(key), key, pa.array(values, t.schema.field(key).type)
        )

    def nth(key: str, value: Any) -> pa.Table:
        values = table[key].to_numpy(zero_copy_only=False).copy()
        values[at[10]] = value
        return column(table, key, values)

    held = labels.ns(date(2024, 1, 2))
    cases = [
        (
            pa.concat_tables([table.slice(0, int(at[10])), table.slice(int(at[10]) + 1)]),
            ValueError,
            "miss or give a non-finite score for 1 scored rows \\(24 h, long\\)",
        ),
        (
            pa.concat_tables([table, column(row, "coin", ["ZZZUSDT"])]),
            ValueError,
            "24 h long, not the ladder's \\d+: 1 rows that are not the ladder's",
        ),
        (
            pa.concat_tables([table, column(row, "horizon", [6])]),
            ValueError,
            "1 rows at a horizon or side the ladder plan lacks",
        ),
        (pa.concat_tables([table, row]), ValueError, "give a row twice \\(24 h, long\\)"),
        (
            nth("fold", int(table["fold"][int(at[10])].as_py()) + 1),
            ValueError,
            "give 1 scored rows a fold other than the ladder's",
        ),
        (nth("score", np.nan), ValueError, "non-finite score for 1 scored rows"),
        (nth("score", np.inf), ValueError, "non-finite score for 1 scored rows"),
        (nth("hour", held), span.HeldBackError, "held-back test span"),
    ]
    copies = []
    for forged, kind, message in cases:
        copy = copied(panel["vol"], tmp_path)
        copies.append(copy)
        forge(copy, name, forged)
        with pytest.raises(kind, match=message):
            compare.run(
                [("full", str(panel["full"])), ("vol", str(copy))],
                panel["inputs"],
                data,
                panel["paired"],
                "x",
            )
    hashed: list[bytes] = []
    digest = compare._digest
    monkeypatch.setattr(compare, "_digest", lambda b: hashed.append(b) or digest(b))
    with pytest.raises(span.HeldBackError):
        compare.run(
            [("full", str(panel["full"])), ("vol", str(copies[-1]))],
            panel["inputs"],
            data,
            panel["paired"],
            "x",
        )
    assert hashed and (copies[-1] / name).read_bytes() not in hashed
    assert not tag_runs(data, "x")


def test_the_file_checked_is_the_file_scored(
    panel: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = panel["data"]
    copy = copied(panel["vol"], tmp_path)
    arms = [("full", str(panel["full"])), ("vol", str(copy))]
    plain, _ = compare.run(arms, panel["inputs"], data, panel["paired"], "plain")
    table = pq.read_table(copy / "vol-forward.parquet")
    other = np.random.default_rng(5).normal(size=table.num_rows)
    read = compare.score_bytes

    def swapping(name: str, path: Path, digest: str) -> bytes:
        checked = read(name, path, digest)
        if path == copy / "vol-forward.parquet":
            pq.write_table(table.set_column(4, "score", pa.array(other)), path)
        return checked

    monkeypatch.setattr(compare, "score_bytes", swapping)
    swapped, _ = compare.run(arms, panel["inputs"], data, panel["paired"], "swapped-file")
    assert pq.read_table(copy / "vol-forward.parquet")["score"].to_pylist() == list(other)
    for name in ("scores", "folds", "comparisons", "rows"):
        assert (plain / f"{name}.parquet").read_bytes() == (
            swapped / f"{name}.parquet"
        ).read_bytes()
    monkeypatch.setattr(compare, "score_bytes", read)
    late = table.set_column(
        1, "hour", pa.array(np.full(table.num_rows, labels.ns(date(2024, 1, 2))))
    )
    sink = pa.BufferOutputStream()
    pq.write_table(late, sink)
    given, real = sink.getvalue().to_pybytes(), compare._read
    monkeypatch.setattr(
        compare, "_read", lambda p, w: given if p.name == "vol-forward.parquet" else real(p, w)
    )
    pq.write_table(table, copy / "vol-forward.parquet")
    with pytest.raises(span.HeldBackError):
        compare.run(arms, panel["inputs"], data, panel["paired"], "x")


def test_a_plan_without_one_primary_pair_or_with_an_unknown_arm_is_refused(
    panel: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sequence, "read_columns", lambda *args: pytest.fail("read the inputs"))
    pair = '[[pairs]]\nfirst = "vol"\nsecond = "full"\n'
    scores = PAIRED.replace(
        'kind = "classical"\nexport = "vol"', 'kind = "scores"\nmodel = "jepa"\nvariant = "vol"'
    )
    cases = {
        "none": (
            PAIRED.replace("primary = true", "primary = false").replace(
                'side = "long"\nhorizon = 24\n', ""
            ),
            "0 primary pairs",
        ),
        "two": (PAIRED + pair + 'primary = true\nside = "any"\nhorizon = 1\n', "2 primary pairs"),
        "unknown": (
            PAIRED + pair.replace('"vol"', '"jepa"'),
            "pair jepa - full names an unknown arm",
        ),
        "twice": (PAIRED + pair, "pair vol - full listed twice"),
        "itself": (
            PAIRED + pair.replace('"vol"', '"full"'),
            "pair full - full compares an arm with itself",
        ),
        "side": (
            PAIRED + pair + 'side = "any"\n',
            "only the primary pair names a side and horizon",
        ),
        "primary side": (PAIRED.replace('side = "long"', 'side = "both"'), "primary side both"),
        "horizon": (PAIRED.replace("horizon = 24", "horizon = 6"), "primary horizon 6"),
        "true horizon": (PAIRED.replace("horizon = 24", "horizon = true"), "primary horizon True"),
        "float horizon": (PAIRED.replace("horizon = 24", "horizon = 24.0"), "primary horizon 24.0"),
        "kind": (
            PAIRED.replace('kind = "classical"\nexport = "vol"', 'kind = "ladder"'),
            "arm vol: kind ladder",
        ),
        "model": (
            PAIRED.replace('export = "vol"', 'export = "vol"\nmodel = "jepa"'),
            "arm vol: model must be rung or window",
        ),
        "named": (PAIRED.replace('name = "vol"', 'name = "full"'), "arm full named twice"),
        "variant": (scores.replace('variant = "vol"', "variant = 1"), "arm vol: variant must be"),
        "scores model": (scores.replace('model = "jepa"', 'model = "lstm"'), "arm vol: model lstm"),
        "no variant": (scores.replace('variant = "vol"\n', ""), "arm vol: no variant"),
        "export": (PAIRED.replace('export = "vol"', 'export = ""'), "arm vol: export must be"),
        "no export": (PAIRED.replace('export = "vol"\n', ""), "arm vol: no export"),
        "arm key": (
            PAIRED.replace('export = "vol"', 'export = "vol"\nseed = 1'),
            "arm vol: unknown key seed",
        ),
        "scores key": (
            scores.replace('variant = "vol"', 'variant = "vol"\nchosen = true'),
            "arm vol: unknown key chosen",
        ),
        "top key": ("note = 1\n" + PAIRED, "unknown key note"),
        "pair key": (
            PAIRED.replace("horizon = 24", "horizon = 24\nweight = 2"),
            "unknown key weight",
        ),
        "primary flag": (
            PAIRED.replace("primary = true", 'primary = "yes"'),
            "primary must be true or false",
        ),
        "c": (PAIRED.replace('export = "vol"', 'export = "vol"\nc = true'), "arm vol: c must be"),
        "negative c": (
            PAIRED.replace('export = "vol"', 'export = "vol"\nc = -1'),
            "arm vol: c must be",
        ),
        "readout": (
            PAIRED.replace('export = "vol"', 'export = "vol"\nreadout = 3'),
            "arm vol: readout must be",
        ),
        "chosen": (
            PAIRED.replace('export = "vol"', 'export = "vol"\nchosen = "yes"'),
            "arm vol: chosen must be",
        ),
        "seeds": (
            scores.replace('variant = "vol"', 'variant = "vol"\nseeds = [1, 1]'),
            "arm vol: seeds must be",
        ),
        "bool seeds": (
            scores.replace('variant = "vol"', 'variant = "vol"\nseeds = [true]'),
            "arm vol: seeds must be",
        ),
        "plan sha256": (
            PAIRED.replace('export = "vol"', 'export = "vol"\nplan_sha256 = "' + "A" * 64 + '"'),
            "arm vol: plan_sha256 must be",
        ),
        "short plan sha256": (
            scores.replace('variant = "vol"', 'variant = "vol"\nplan_sha256 = "abc"'),
            "arm vol: plan_sha256 must be",
        ),
        "arms": (
            PAIRED.split("[[arms]]")[0] + "arms = 1\n" + "[[pairs]]" + PAIRED.split("[[pairs]]")[1],
            "arms must be a list",
        ),
        "no ladder": (PAIRED.replace('ladder = "ladder.toml"', ""), "ladder must name"),
        "empty ladder": (
            PAIRED.replace('ladder = "ladder.toml"', 'ladder = ""'),
            "ladder must name",
        ),
        "ladder file": (
            PAIRED.replace('ladder = "ladder.toml"', 'ladder = "none.toml"'),
            "ladder: no file",
        ),
    }
    arms = [("full", str(panel["full"])), ("vol", str(panel["vol"]))]
    for name, (text, message) in cases.items():
        plan = (
            folder(tmp_path / name, {"ladder.toml": LADDER, "compare.toml": text}) / "compare.toml"
        )
        with pytest.raises(ValueError, match=message):
            compare.run(arms, panel["inputs"], panel["data"], plan, "x")
    late = LADDER.replace("end = 2022-12-31", "end = 2024-01-01")
    plan = folder(tmp_path / "late", {"ladder.toml": late, "compare.toml": PAIRED}) / "compare.toml"
    with pytest.raises(span.HeldBackError):
        compare.run(arms, panel["inputs"], panel["data"], plan, "x")
    assert not tag_runs(panel["data"], "x")
