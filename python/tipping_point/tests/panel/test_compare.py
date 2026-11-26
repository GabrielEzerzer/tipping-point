# The paired evaluation on synthetic panels: equal to the ladder's own comparison, one set of draws
# for every arm, a sequence arm end to end, repeat runs, and every refusal of a file that does not
# come from the evaluation's inputs run, labels grid and ladder plan or does not hold its rows.

import json
import math
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
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


def edit(run: Path, name: str, change: dict[str, Any]) -> None:
    record = json.loads((run / name).read_text(encoding="utf-8"))
    target = record["params"] if name == "manifest.json" else record
    target |= change
    (run / name).write_text(json.dumps(record), encoding="utf-8")


def forge(run: Path, name: str, table: pa.Table) -> None:
    pq.write_table(table, run / name)
    record = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    record["files"][name] = store.sha256(run / name)
    (run / "complete.json").write_text(json.dumps(record), encoding="utf-8")


def same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


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


def run_args(p: dict[str, Any], plan: Path, tag: str, **arms: Path) -> list[str]:
    given = [part for name, run in arms.items() for part in ("--arm", f"{name}={run}")]
    command = ["compare", "--inputs", p["inputs"].name, "--plan", str(plan), *given]
    return [*command, "--data-dir", str(p["data"]), "--tag", tag]


def test_the_shipped_plan_names_one_primary_pair_on_the_ladders_plan() -> None:
    plan, records, (text, ladder_text) = compare.snapshot()
    assert plan.ladder == ladder.load()
    assert plan.primary == ("jepa", "classical") and (plan.side, plan.horizon) == ("any", 24)
    kinds = {a.name: (a.kind, a.model, a.variant, a.export) for a in plan.arms}
    assert kinds == {
        "jepa": ("scores", "jepa", "full", None),
        "twin": ("scores", "twin", "full", None),
        "supervised": ("scores", "supervised", "full", None),
        "classical": ("classical", None, None, "full"),
        "jepa-vol": ("scores", "jepa", "vol", None),
        "classical-vol": ("classical", None, None, "vol"),
        "classic": ("classical", None, None, "classic"),
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
    assert main(run_args(panel, panel["paired"], "one", **arms)) == 0
    assert main(run_args(panel, panel["paired"], "two", **arms)) == 0
    one, two = (next((data / "panel-compare").glob(f"*-{t}")) for t in ("one", "two"))
    tables = ("scores", "folds", "comparisons", "rows")
    for name in tables:
        assert (one / f"{name}.parquet").read_bytes() == (two / f"{name}.parquet").read_bytes()
    theirs = pq.read_table(ladder_run / "comparisons.parquet").to_pylist()
    mine = pq.read_table(one / "comparisons.parquet").to_pylist()
    assert len(mine) == len(theirs) == 2 * 2 * 3
    for row in mine:
        match = [
            r
            for r in theirs
            if all(r[k] == row[k] for k in ("horizon", "scheme", "side", "first", "second"))
        ]
        assert len(match) == 1
        for key, value in match[0].items():
            assert same(row[key], value), (key, row, value)
        assert row["primary_pair"] is True
        assert row["primary"] is (
            (row["scheme"], row["side"], row["horizon"]) == ("forward", "long", 24)
        )
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
        complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
        assert arm["run"] == run.name and arm["kind"] == "classical" and arm["export"] == name
        assert (arm["model"], arm["variant"], arm["readout"], arm["c"]) == (
            "rung",
            name,
            None,
            0.01,
        )
        assert arm["fixed"] == source["fixed"] and arm["selection"] is None
        assert arm["made_from"] == {
            "run": run.name,
            **{k: source[k] for k in ("inputs", "labels_grid", "ladder_plan")},
        }
        assert set(arm["code"]) == {"git_sha", "dirty", "uv_lock_sha256"}
        assert arm["code"]["uv_lock_sha256"] == source["code"]["uv_lock_sha256"]
        for scheme in ("loqo", "forward"):
            file = f"{name}-{scheme}.parquet"
            assert arm["files"][scheme] == {"file": file, "sha256": complete["files"][file]}
    report = (one / "report.md").read_text(encoding="utf-8")
    primary = report.split("## Primary")[1].split("## Secondary")[0]
    assert primary.count("| full - vol | forward | 24 h | long |") == 1
    assert "| full - vol | forward | 24 h | long |" not in report.split("## Secondary")[1]


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
    assert len(keyed) == len(rows) == 3 * 2 * 2 * 3
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
    assert checked == 2 * 2 * 3


def test_a_sequence_arm_and_a_window_arm_compare_end_to_end(tmp_path: Path) -> None:
    data = tmp_path / "data"
    values, grid = world(days=200)
    write_labels(data, grid)
    run_inputs = write_inputs(data, values, grid, LABELS, "inputs")
    text = PLAN.replace('schemes = ["loqo", "forward"]', 'schemes = ["forward"]')
    text = text.replace("folds = []", "folds = [2]")
    plan = folder(data / "plan", {"ladder.toml": LADDER, "sequence.toml": text}) / "sequence.toml"
    seq, _ = sequence.run(run_inputs, data, plan, "sequence")
    scores, _ = sequence.scores_run(seq, data, "jepa", "vol", "jepa")
    window = fixed(data, run_inputs, plan, "window", Config("window", "vol", "last", 0.01))
    paired = """
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
    path = folder(data / "paired", {"ladder.toml": LADDER, "compare.toml": paired}) / "compare.toml"
    p = {"data": data, "inputs": run_inputs}
    assert main(run_args(p, path, "seq", jepa=Path(scores.name), window=window)) == 0
    (run_dir,) = (data / "panel-compare").glob("*-seq")
    rows = pq.read_table(run_dir / "comparisons.parquet").to_pylist()
    assert len(rows) == 2 * 3 and {r["scheme"] for r in rows} == {"forward"}
    (primary,) = [r for r in rows if r["primary"]]
    assert (primary["side"], primary["horizon"], primary["folds"]) == ("any", 24, 1)
    recorded = params(run_dir)["arms"]
    assert recorded["jepa"]["sequence_run"] == seq.name
    assert recorded["jepa"]["made_from"]["run"] == seq.name
    assert (recorded["jepa"]["model"], recorded["jepa"]["variant"]) == ("jepa", "vol")
    assert recorded["jepa"]["readout"] == params(scores)["readout"]
    assert recorded["jepa"]["selection"] == params(scores)["selection"]
    assert recorded["window"]["model"] == "window"
    file = (scores / "forward.parquet").as_posix()
    external = f'\n[external]\nname = "jepa"\npath = "{file}"\nscheme = "forward"\n'
    (data / "ladder-ext.toml").write_text(LADDER + external, encoding="utf-8")
    ladder_run, _ = ladder.run(run_inputs, data, data / "ladder-ext.toml", "ext")
    keys = ("horizon", "scheme", "side", "name", "coins", "fold")
    theirs = {
        tuple(r[k] for k in keys): r
        for r in pq.read_table(ladder_run / "folds.parquet").to_pylist()
    }
    mine = [r for r in pq.read_table(run_dir / "folds.parquet").to_pylist() if r["name"] == "jepa"]
    assert mine and all(same(r, theirs[tuple(r[k] for k in keys)]) for r in mine)

    def refused(change: Any, message: str, plan_path: Path = path) -> None:
        copy = data / "copies" / str(len(list(data.glob("copies/*"))))
        shutil.copytree(scores, copy)
        change(copy)
        with pytest.raises(ValueError, match=message):
            compare.run(
                [("jepa", str(copy)), ("window", str(window))], run_inputs, data, plan_path, "x"
            )

    gone = "20260101T000000Z-gone"
    refused(lambda d: edit(d, "manifest.json", {"sequence_run": gone}), f"no run at .*{gone}")
    moved = data / "panel-sequence" / f"{seq.name}-moved"
    shutil.copytree(seq, moved)
    (moved / "complete.json").unlink()
    refused(
        lambda d: edit(d, "manifest.json", {"sequence_run": moved.name}),
        "the sequence run of .* has no complete.json",
    )
    unclean = data / "panel-sequence" / f"{seq.name}-unclean"
    shutil.copytree(seq, unclean)
    edit(unclean, "complete.json", {"clean": False})
    refused(
        lambda d: edit(d, "manifest.json", {"sequence_run": unclean.name}), "does not mark .* clean"
    )
    used = params(scores)["encoders"]
    refused(
        lambda d: edit(d, "manifest.json", {"encoders": {k: "0" * 64 for k in used}}),
        "scored are not those of sequence run",
    )
    refused(
        lambda d: edit(d, "manifest.json", {"model": "twin"}), "scores twin on vol, not jepa on vol"
    )
    refused(
        lambda d: edit(d, "manifest.json", {"variant": "full"}),
        "scores jepa on full, not jepa on vol",
    )
    other = data / "panel-sequence" / f"{seq.name}-other"
    shutil.copytree(seq, other)
    edit(other, "manifest.json", {"inputs": {"file": "inputs.parquet", "sha256": "0" * 64}})
    refused(
        lambda d: edit(d, "manifest.json", {"sequence_run": other.name}),
        f"{other.name} read another inputs table",
    )
    twin = path.read_text(encoding="utf-8").replace('model = "jepa"', 'model = "twin"')
    wrong = folder(data / "twin", {"ladder.toml": LADDER, "compare.toml": twin}) / "compare.toml"
    refused(lambda d: None, "scores jepa on vol, not twin on vol", wrong)
    with pytest.raises(ValueError, match="is a panel-classical run, not a panel-scores run"):
        compare.run([("jepa", str(window)), ("window", str(window))], run_inputs, data, path, "x")
    with pytest.raises(ValueError, match="is a panel-scores run, not a panel-classical run"):
        compare.run([("jepa", str(scores)), ("window", str(scores))], run_inputs, data, path, "x")
    assert len(list((data / "panel-compare").iterdir())) == 1


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
    monkeypatch.setattr(ladder, "external_table", lambda path: pytest.fail("read an arm's scores"))
    ours = params(panel["vol"])

    def refused(run: Path, change: dict[str, Any], message: str) -> None:
        copy = tmp_path / "copies" / f"{run.name}-{len(change)}"
        shutil.copytree(run, copy)
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
    refused(from_plan, {}, "was made under another ladder plan")
    assert not list((data / "panel-compare").glob("*-x"))


def test_an_arm_that_is_not_its_run_or_not_the_plans_arm_is_refused(
    panel: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = panel["data"]
    monkeypatch.setattr(ladder, "external_table", lambda path: pytest.fail("read an arm's scores"))

    def refused(change: Any, message: str) -> None:
        copy = tmp_path / "copies" / str(len(list((tmp_path).glob("copies/*"))))
        shutil.copytree(panel["vol"], copy)
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
    table = pq.read_table(panel["vol"] / "vol-forward.parquet")
    nudged = table.set_column(4, "score", pa.array(table["score"].to_numpy() + 1e-12))
    refused(
        lambda d: pq.write_table(nudged, d / "vol-forward.parquet"),
        "arm vol: .*vol-forward.parquet: its sha256 differs from complete.json's",
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
    assert main(run_args(panel, panel["paired"], "x", full=panel["full"])) == 1
    assert not list((data / "panel-compare").glob("*-x"))


def _domain(run: Path) -> None:
    record = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    record["domain"] = "panel-ladder"
    (run / "manifest.json").write_text(json.dumps(record), encoding="utf-8")


def _drop(run: Path, name: str) -> None:
    record = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    del record["files"][name]
    (run / "complete.json").write_text(json.dumps(record), encoding="utf-8")


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
    for i, (forged, kind, message) in enumerate(cases):
        copy = tmp_path / str(i)
        shutil.copytree(panel["vol"], copy)
        forge(copy, name, forged)
        with pytest.raises(kind, match=message):
            compare.run(
                [("full", str(panel["full"])), ("vol", str(copy))],
                panel["inputs"],
                data,
                panel["paired"],
                "x",
            )
    hashed: list[Path] = []
    sha = store.sha256
    monkeypatch.setattr(store, "sha256", lambda p: hashed.append(p) or sha(p))
    with pytest.raises(span.HeldBackError):
        compare.run(
            [("full", str(panel["full"])), ("vol", str(tmp_path / str(len(cases) - 1)))],
            panel["inputs"],
            data,
            panel["paired"],
            "x",
        )
    assert hashed == []
    assert not list((data / "panel-compare").glob("*-x"))


def test_a_plan_without_one_primary_pair_or_with_an_unknown_arm_is_refused(
    panel: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sequence, "read_columns", lambda *args: pytest.fail("read the inputs"))
    pair = '[[pairs]]\nfirst = "vol"\nsecond = "full"\n'
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
        "horizon": (PAIRED.replace("horizon = 24", "horizon = 6"), "primary horizon 6"),
        "kind": (
            PAIRED.replace('kind = "classical"\nexport = "vol"', 'kind = "ladder"'),
            "arm vol: kind ladder",
        ),
        "model": (
            PAIRED.replace('export = "vol"', 'export = "vol"\nmodel = "jepa"'),
            "arm vol: a classical arm takes",
        ),
        "named": (PAIRED.replace('name = "vol"', 'name = "full"'), "arm full named twice"),
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
    assert not list((panel["data"] / "panel-compare").glob("*-x"))
