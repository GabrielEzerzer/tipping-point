# harness/panel_forward.py on synthetic panels: its rows, each number able to fail (next-day value
# carried or shuffled, gain over the input row, kept or dropped inputs, slow or fast embeddings,
# both rules), no label read, the rebuilt tensor, determinism and refusals, on a tiny run.

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import panel_forward
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.panel import encoder, inputs, labels, ladder, sequence, span, store
from tipping_point.panel.__main__ import main as panel_main
from torch import nn

HOUR = labels.HOUR_NS
COINS = ["AAAUSDT", "BBBUSDT"]
LADDER = """
start = 2022-01-01
end = 2022-12-31
horizons = [24]
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
draws = 10
seed = 1

[[rungs]]
name = "full"
groups = ["vol", "price", "flow", "lev"]
own_only = ["lev"]

[[rungs]]
name = "vol"
groups = ["vol"]
"""
PLAN = """
ladder = "ladder.toml"
variants = ["full", "vol"]
rungs = ["full", "vol"]
models = ["jepa", "twin"]
schemes = ["forward"]
seeds = [3]
folds = [1]
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
readouts = ["last"]
c = [1.0]
"""
ROWS = 300


def plan_files(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ladder.toml").write_text(LADDER, encoding="utf-8")
    (folder / "sequence.toml").write_text(PLAN, encoding="utf-8")
    return folder / "sequence.toml"


def world(days: int = 365, seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    hours = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(days * 24, dtype=np.int64)
    n = len(hours) * len(COINS)
    values = {name: rng.normal(size=n) for name in inputs.OWN + inputs.MARKET}
    values["oi_24h"][rng.random(n) < 0.05] = np.nan
    values["funding"][rng.random(n) < 0.05] = np.nan
    grid: dict[str, np.ndarray] = {
        "ts_ns": np.repeat(hours, len(COINS)),
        "symbol": np.tile(np.array(COINS, dtype=object), len(hours)),
    }
    for h in labels.HORIZONS:
        lift = values["vol_1h"]
        grid[f"long_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 - lift))
        grid[f"short_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 + lift))
        grid[f"censored_{h}"] = rng.random(n) < 0.01
    grid["skip"] = rng.random(n) < 0.02
    return values, grid


def write_grid(data: Path, grid: dict) -> Path:
    labels_dir = data / "labels" / "20260101T000000Z-t"
    (labels_dir / "grid").mkdir(parents=True, exist_ok=True)
    path = labels_dir / "grid" / "0.parquet"
    pq.write_table(pa.table(grid), path)
    return path


def write_inputs(data: Path, values: dict, grid: dict) -> Path:
    sha = store.sha256(write_grid(data, grid))
    table = pa.table(
        {"ts_ns": grid["ts_ns"], "symbol": grid["symbol"].astype(str), **values},
        schema=inputs.SCHEMA,
    )
    params = {
        "labels_run": "20260101T000000Z-t",
        "labels_grid": {"file": "grid/0.parquet", "sha256": sha},
        "start": "2021-10-01",
        "dev_start": "2022-01-01",
        "end": "2022-12-31",
    }
    return store.write(data, "panel-inputs", "inputs", {"inputs": table}, params, {})


def rebound(run: Path, change: dict) -> None:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    manifest["params"] |= change
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    complete["manifest_sha256"] = store.sha256(run / "manifest.json")
    (run / "complete.json").write_text(json.dumps(complete), encoding="utf-8")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    data = tmp_path_factory.mktemp("store")
    values, grid = world(days=182)
    run_inputs = write_inputs(data, values, grid)
    plan = plan_files(data / "plans")
    args = ["sequence", "--inputs", run_inputs.name, "--plan", str(plan), "--data-dir", str(data)]
    assert panel_main(args) == 0
    (run,) = (data / "panel-sequence").glob("*-sequence")
    return {"data": data, "inputs": run_inputs, "sequence": run}


def forward_args(built: dict[str, Path], out: Path, **change: Path) -> list[str]:
    paths = built | change
    return [
        "--sequence",
        str(paths["sequence"]),
        "--inputs",
        str(paths["inputs"]),
        "--data-dir",
        str(paths["data"]),
        "--out",
        str(out),
        "--rows",
        str(ROWS),
    ]


@pytest.fixture(scope="module")
def checked(built: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("checked") / "out"
    assert panel_forward.main(forward_args(built, out)) == 0
    return out


def tiny_plan(tmp_path: Path) -> sequence.Plan:
    return sequence.load(plan_files(tmp_path / "plans"))


def test_fit_rows_stay_in_training_and_evaluation_rows_in_the_left_out_quarter(
    tmp_path: Path,
) -> None:
    plan = tiny_plan(tmp_path)
    hours = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(365 * 24, dtype=np.int64)
    train, fit, evaluation = panel_forward.row_hours(hours, plan, 2)
    assert fit.any() and evaluation.any() and not (fit & evaluation).any()
    for j in np.flatnonzero(fit):
        assert train[j - plan.pad : j + panel_forward.AHEAD + 1].all()
    assert not fit[: plan.pad].any()
    quarter = (hours >= labels.ns(date(2022, 7, 1))) & (hours < labels.ns(date(2022, 10, 1)))
    assert np.array_equal(evaluation, quarter)
    _, _, last = panel_forward.row_hours(hours, plan, 3)
    assert last[-panel_forward.AHEAD - 1] and not last[-panel_forward.AHEAD :].any()
    allowed = np.tile(fit, (2, 1))
    allowed[1, : len(fit) // 2] = False
    one = panel_forward.sample(allowed, 50, sequence.stream("full", 2, 3, "fit"))
    again = panel_forward.sample(allowed, 50, sequence.stream("full", 2, 3, "fit"))
    other = panel_forward.sample(allowed, 50, sequence.stream("full", 2, 4, "fit"))
    assert len(one[0]) == 50 and allowed[one].all()
    assert all(np.array_equal(a, b) for a, b in zip(one, again, strict=True))
    assert not np.array_equal(one[1], other[1])
    everything = panel_forward.sample(allowed, 10**9, sequence.stream("full", 2, 3, "fit"))
    assert len(everything[0]) == allowed.sum() and allowed[everything].all()


class Pick(encoder.Encoder):
    def __init__(self, channels: list[int], lag: int = 0, shuffle: bool = False) -> None:
        nn.Module.__init__(self)
        self.channels, self.lag, self.shuffle = channels, lag, shuffle

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        out = x[:, 1 - self.lag :: 2][:, :, self.channels]
        if self.shuffle:
            out = out[torch.randperm(len(out), generator=torch.Generator().manual_seed(0))]
        return out


COLUMNS = (
    "vol_1h",
    "vol_6h",
    "vol_24h",
    "vol_7d",
    "vol_28d",
    "near_24h",
    "vsurge_1h",
    "volume_24h",
    "ret_1h",
    "ret_6h",
    "aret_24h",
    "dd_7d",
    "flow_1h",
    "flow_24h",
    "funding",
    "ls",
    "oi_vs_28d",
)


def channels(*names: str) -> list[int]:
    return [COLUMNS.index(n) for n in names]


CARRIER = channels(
    "vol_28d", "vol_1h", "dd_7d", "funding", "ret_1h", "ret_6h", "flow_1h", "vsurge_1h"
)
BLIND = channels(
    "vol_24h", "ret_1h", "ret_6h", "flow_1h", "vsurge_1h", "vol_6h", "aret_24h", "volume_24h"
)


def setup(raw: np.ndarray, plan: sequence.Plan) -> dict:
    nhours = raw.shape[1]
    train = np.arange(nhours) < 2000
    present = ~np.isnan(raw).all(axis=-1)
    x, z = panel_forward.prepared(raw, train, COLUMNS, plan)
    fit_hours = sequence.eligible(train, plan.pad, panel_forward.AHEAD)
    eval_hours = (np.arange(nhours) >= 2200) & (np.arange(nhours) < 2900)
    seeds = np.random.SeedSequence(1), np.random.SeedSequence(2)
    fit, ev, later = panel_forward.drawn(present, fit_hours, eval_hours, 1500, *seeds)
    alone, rows = panel_forward.inputs_alone(x, z, plan, fit, ev)

    def measured(net: encoder.Encoder) -> list[dict]:
        return panel_forward.measure(net, x, z, plan, fit, ev, alone, later)

    return {"measure": measured, "alone": rows, "fit": fit, "eval": ev}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict:
    plan = tiny_plan(tmp_path_factory.mktemp("plan"))
    rng = np.random.default_rng(7)
    ncoins, nhours = 2, 3000
    raw = rng.normal(size=(ncoins, nhours, len(COLUMNS)))

    def column(name: str) -> np.ndarray:
        return raw[..., COLUMNS.index(name)]

    def noise(n: int) -> np.ndarray:
        return 0.1 * rng.normal(size=(ncoins, n))

    signal, step = column("vol_28d"), column("dd_7d")
    column("vol_24h")[:, 24:] = signal[:, :-24] + noise(nhours - 24)
    column("near_24h")[:, 25:] = signal[:, :-25] + noise(nhours - 25)
    for t in range(1, nhours):
        column("vol_7d")[:, t] = 0.998 * column("vol_7d")[:, t - 1] + rng.normal(size=ncoins)
    for t in range(24, nhours):
        column("funding")[:, t] = 0.5 * column("funding")[:, t - 24] + step[:, t - 24]
    column("funding")[:, 24:] += noise(nhours - 24)
    return setup(raw, plan) | {"raw": raw, "plan": plan}


def last(rows: list[dict], kind: str, target: str, readout: str = "last") -> float:
    (value,) = [
        r["value"]
        for r in rows
        if (r["readout"], r["kind"], r["target"]) == (readout, kind, target)
    ]
    return value


def test_forward_r2_tells_an_embedding_carrying_the_next_day_from_one_with_shuffled_rows(
    synthetic: dict,
) -> None:
    carrier = synthetic["measure"](Pick(CARRIER))
    shuffled = synthetic["measure"](Pick(CARRIER, shuffle=True))
    blind = synthetic["measure"](Pick(BLIND))
    assert last(carrier, "forward", "vol_24h") > 0.95
    assert last(shuffled, "forward", "vol_24h") < 0.05
    assert last(carrier, "forward", "funding_change") > 0.9
    assert last(blind, "forward", "funding_change") < 0.05
    for target in ("near_24h", "aret_24h", "volume_24h", "flow_24h", "ls_change"):
        assert abs(last(carrier, "forward", target)) < 0.05
    targets = {r["target"] for r in carrier if r["kind"] == "forward"}
    assert targets == set(panel_forward.FORWARD) - {"oi_24h"}


def test_the_gain_over_the_inputs_tells_what_the_last_input_row_does_not_give(
    synthetic: dict,
) -> None:
    alone = {r["target"]: r["value"] for r in synthetic["alone"]}
    assert {r["readout"] for r in synthetic["alone"]} == {"none"}
    assert alone["vol_24h"] > 0.95 and abs(alone["near_24h"]) < 0.05
    current = synthetic["measure"](Pick(CARRIER))
    lagged = synthetic["measure"](Pick(CARRIER, lag=1))
    assert last(current, "forward", "vol_24h") > 0.95
    assert abs(last(current, "forward_gain", "vol_24h")) < 0.05
    assert abs(last(current, "forward_gain", "near_24h")) < 0.05
    assert last(lagged, "forward", "near_24h") > 0.95
    assert last(lagged, "forward_gain", "near_24h") > 0.9
    assert abs(last(lagged, "forward_gain", "vol_24h")) < 0.05


def test_retention_tells_a_kept_input_from_a_dropped_one(synthetic: dict) -> None:
    kept = synthetic["measure"](Pick(CARRIER))
    dropped = synthetic["measure"](Pick(BLIND))
    for target in ("vol_1h", "funding"):
        assert last(kept, "retention", target) > 0.99
        assert last(dropped, "retention", target) < 0.05
    assert last(dropped, "retention", "ret_1h") > 0.99
    retained = {r["target"] for r in kept if r["kind"] == "retention"}
    assert retained == set(panel_forward.RETAINED)


def test_persistence_tells_a_slow_embedding_from_a_fast_one(synthetic: dict) -> None:
    slow = synthetic["measure"](Pick(channels("vol_7d") * 8))
    fast = synthetic["measure"](Pick(channels("ret_1h") * 8))
    assert last(slow, "persistence", "last") > 0.8
    assert abs(last(fast, "persistence", "last")) < 0.15


def test_a_coin_with_a_wholly_missing_stretch_does_not_change_persistence(synthetic: dict) -> None:
    raw, plan = synthetic["raw"], synthetic["plan"]
    gappy = np.full((1, *raw.shape[1:]), np.nan)
    gappy[0, 2300:2324] = raw[0, 2300:2324]
    slow = Pick(channels("vol_7d") * 8)
    two = setup(raw, plan)
    three = setup(np.concatenate([raw, gappy]), plan)
    (expected,) = [r for r in two["measure"](slow) if r["kind"] == "persistence"]
    (found,) = [r for r in three["measure"](slow) if r["kind"] == "persistence"]
    assert found["value"] == pytest.approx(expected["value"], abs=1e-12)
    assert found["eval_rows"] == expected["eval_rows"] == 2 * 700
    assert all(np.array_equal(a, b) for a, b in zip(two["fit"], three["fit"], strict=True))
    assert (three["eval"][0] == 2).sum() == 24 and (two["eval"][0] < 2).all()


QUARTERS = range(2, 8)
SEEDS = (1, 2)


def rule_rows(
    vol: list[float], oi: list[float], variant: str = "full", kind: str = "forward"
) -> list[dict]:
    rows = []
    for target, diffs in (("vol_24h", vol), ("oi_24h", oi)):
        for f, d in enumerate(diffs, start=2):
            for seed, wobble in zip(SEEDS, (0.05, -0.05), strict=True):
                for model, value in (("jepa", 0.1 + d + wobble), ("twin", 0.1)):
                    rows.append(
                        {"variant": variant, "readout": "last", "kind": kind}
                        | {"target": target, "fold": f, "seed": seed, "model": model}
                        | {"value": value}
                    )
    return rows


def verdict(
    kind: str,
    vol: list[float],
    oi: list[float],
    variant: str = "full",
    folds: range = QUARTERS,
) -> str:
    return panel_forward.rule(rule_rows(vol, oi, variant, kind), kind, folds, SEEDS)[0]


def test_each_rule_passes_fails_or_is_not_judged_on_its_stated_conditions() -> None:
    six = [0.01] * 6
    five = [0.01] * 5 + [-0.005]
    four = [0.01] * 4 + [-0.005] * 2
    sinking = [0.01] * 5 + [-0.2]
    for kind in panel_forward.RULES:
        assert verdict(kind, six, six) == verdict(kind, five, five) == "PASS"
        assert verdict(kind, four, six) == verdict(kind, six, four) == "FAIL"
        assert verdict(kind, six, sinking) == "FAIL"
        assert verdict(kind, [-0.01] * 6, [-0.01] * 6) == "FAIL"
        assert verdict(kind, six, six, variant="vol") == "NOT JUDGED"
        assert verdict(kind, six, [0.01] * 3 + [-0.005]) == "NOT JUDGED"
        assert verdict(kind, [0.01, -0.005, 0.01], [0.01, 0.01, -0.001]) == "NOT JUDGED"
        assert verdict(kind, six, six, folds=range(1, 8)) == "NOT JUDGED"
        assert verdict(kind, six, six, folds=range(0)) == "NOT JUDGED"
        assert verdict(kind, six[:4], [0.01] * 3 + [-0.005], folds=range(2, 6)) == "PASS"
        assert verdict(kind, six[:4], [0.01] * 2 + [-0.005] * 2, folds=range(2, 6)) == "FAIL"
        lost_seed = [
            r
            for r in rule_rows(six, six, kind=kind)
            if (r["fold"], r["seed"], r["model"], r["target"]) != (4, 2, "twin", "oi_24h")
        ]
        found, parts = panel_forward.rule(lost_seed, kind, QUARTERS, SEEDS)
        assert found == "NOT JUDGED"
        assert parts[1] == "oi_24h: not judged, quarters lacking a seed or model: q5"
    mixed = rule_rows(six, six, kind="forward") + rule_rows(four, five, kind="forward_gain")
    assert panel_forward.rule(mixed, "forward", QUARTERS, SEEDS)[0] == "PASS"
    found, parts = panel_forward.rule(mixed, "forward_gain", QUARTERS, SEEDS)
    assert found == "FAIL"
    assert parts == [
        "vol_24h: mean difference +0.0050, positive in 4 of 6 quarters",
        "oi_24h: mean difference +0.0075, positive in 5 of 6 quarters",
    ]


def test_the_check_writes_its_table_report_and_manifest_and_repeats(
    built: dict[str, Path], checked: Path, tmp_path: Path
) -> None:
    assert panel_forward.main(forward_args(built, tmp_path / "two")) == 0
    one = pq.read_table(checked / "forward.parquet")
    two = pq.read_table(tmp_path / "two" / "forward.parquet")
    assert one.schema == panel_forward.TABLE and one.equals(two)
    rows = one.to_pylist()
    full_cell = 9 + 2 * (3 * (2 * 9 + 9) + 1)
    vol_cell = 3 + 2 * (3 * (2 * 3 + 3) + 1)
    assert len(rows) == full_cell + vol_cell
    full = [r for r in rows if r["variant"] == "full"]
    assert {r["model"] for r in rows} == {"jepa", "twin", "inputs"}
    assert {r["fold"] for r in rows} == {1}
    for kind in ("forward", "forward_gain", "forward_inputs"):
        found = {r["target"] for r in full if r["kind"] == kind}
        assert found == set(panel_forward.FORWARD), kind
    metrics = {(r["kind"], r["metric"]) for r in rows}
    assert {m for k, m in metrics if k == "forward_gain"} == {"r2_gain"}
    assert {m for k, m in metrics if k == "persistence"} == {"correlation"}
    assert {r["target"] for r in full if r["kind"] == "retention"} == set(panel_forward.RETAINED)
    vol = {r["target"] for r in rows if r["variant"] == "vol" and r["kind"] == "forward"}
    assert vol == {"vol_24h", "volume_24h", "near_24h"}
    assert all(np.isfinite(r["value"]) for r in rows)
    probed = [r for r in rows if r["kind"] != "persistence"]
    assert all(0 < r["fit_rows"] <= ROWS and 0 < r["eval_rows"] <= ROWS for r in probed)
    assert all(r["fit_rows"] < ROWS for r in full if r["target"] in ("oi_24h", "funding_change"))
    inputs_r2 = {
        (r["fold"], r["target"]): r["value"] for r in full if r["kind"] == "forward_inputs"
    }
    joined = {
        (r["fold"], r["model"], r["readout"], r["target"]): r["value"]
        for r in full
        if r["kind"] == "forward_gain"
    }
    assert len(joined) == 2 * 3 * 9 and all(
        np.isfinite(v + inputs_r2[(k[0], k[3])]) for k, v in joined.items()
    )
    run = built["sequence"]
    manifest = json.loads((checked / "manifest.json").read_text(encoding="utf-8"))
    params = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["params"]
    assert manifest["code"]["git_sha"] and manifest["arguments"]["rows"] == ROWS
    files = manifest["files"]
    encoders = {k.removeprefix("sequence/") for k in files if k.startswith("sequence/encoders/")}
    assert encoders == set(params["encoders"])
    roots = {"sequence": run, "inputs": built["inputs"], "labels": built["data"] / "labels"}
    for key, digest in files.items():
        root, _, name = key.partition("/")
        assert store.sha256(roots[root] / name) == digest, key
    assert {"sequence/manifest.json", "sequence/plan.toml", "inputs/inputs.parquet"} <= set(files)
    report = (checked / "report.md").read_text(encoding="utf-8").rstrip().splitlines()
    plan = sequence.load(run / "plan.toml", run / "ladder.toml")
    assert list(panel_forward.quarters(plan)) == [1, 2, 3]
    assert any("over the 3 forward left-out quarters" in x for x in report)
    for line, (number, kind) in zip(report[-3::2], enumerate(panel_forward.RULES, 1), strict=True):
        found, parts = panel_forward.rule(rows, kind, range(1, 4), plan.seeds)
        assert found == "NOT JUDGED" and parts[0].endswith("q3, q4")
        assert line.startswith(f"Rule {number},") and f"(kind `{kind}`)" in line, line
        assert line.endswith(f"**{found}** ({'; '.join(parts)}).")
    d = panel_forward.differences(rows, "full", "last", "forward_gain", "oi_24h")
    line = next(x for x in report if x.startswith("| full | last | forward_gain | oi_24h |"))
    assert line.endswith(f"| {sum(v > 0 for v in d.values())} of 1 |")
    groups = []
    for group in panel_forward.RETENTION_GROUPS.values():
        found = [
            r["value"]
            for r in full
            if (r["model"], r["readout"], r["kind"]) == ("jepa", "last", "retention")
            and r["target"] in group
        ]
        assert len(found) == 3
        groups.append(f"{np.mean(found):.4f}")
    assert f"| full | last | jepa | {' | '.join(groups)} |" in report


def test_the_rebuilt_tensor_is_the_one_the_run_trained_on(built: dict[str, Path]) -> None:
    run = built["sequence"]
    plan = sequence.load(run / "plan.toml", run / "ladder.toml")
    table = pq.read_table(built["inputs"] / "inputs.parquet")
    values = {n: table[n].to_numpy() for n in table.column_names if n not in ("ts_ns", "symbol")}
    labelled = pq.read_table(next((built["data"] / "labels").rglob("0.parquet")))
    grid = {n: labelled[n].to_numpy(zero_copy_only=False) for n in labelled.column_names}
    hours, coins = sequence.layout(grid)
    columns = plan.rung("full").columns
    raw = sequence.by_coin(np.column_stack([values[c] for c in columns]), len(coins))
    train, _, _ = panel_forward.row_hours(hours, plan, 1)
    x, _ = panel_forward.prepared(raw, train, columns, plan)
    kept = np.any([r.keep for r in sequence.common_rows(values, grid, plan).values()], axis=0)
    fold, _ = ladder.folds(grid["ts_ns"], ladder.bounds(plan.ladder), plan.ladder.block_days)
    index = np.flatnonzero(kept & (fold == 1))
    diagnostics = pq.read_table(run / "diagnostics.parquet").to_pylist()
    for model in panel_forward.MODELS:
        (recorded,) = [
            d
            for d in diagnostics
            if (d["variant"], d["scheme"], d["fold"], d["seed"], d["model"])
            == ("full", "forward", 1, 3, model)
        ]
        data = (run / sequence.encoder_name("full", "forward", 1, 3, model)).read_bytes()
        net = panel_forward.restored(x.shape[-1], plan, data)
        z = sequence.features(net, x, index % len(coins), index // len(coins), plan)
        found = sequence.describe(z[:, : plan.shape.width])
        for key in ("z_sd_min", "z_sd_median", "z_effective_rank"):
            assert found[key] == pytest.approx(recorded[key], rel=1e-6), (model, key)


def test_no_label_is_read(built: dict[str, Path], checked: Path, tmp_path: Path) -> None:
    copy = tmp_path / "store"
    for name in ("labels", "panel-inputs", "panel-sequence"):
        shutil.copytree(built["data"] / name, copy / name)
    inputs_run = copy / "panel-inputs" / built["inputs"].name
    sequence_run = copy / "panel-sequence" / built["sequence"].name
    grid_path = next((copy / "labels").rglob("0.parquet"))
    grid = pq.read_table(grid_path)
    rng = np.random.default_rng(99)
    garbage = {
        name: grid[name] if name in ("ts_ns", "symbol") else pa.array(rng.random(len(grid)) < 0.5)
        for name in grid.column_names
    }
    pq.write_table(pa.table(garbage), grid_path)
    record = {"labels_grid": {"file": "grid/0.parquet", "sha256": store.sha256(grid_path)}}
    rebound(inputs_run, record)
    rebound(sequence_run, record)
    paths = {"data": copy, "inputs": inputs_run, "sequence": sequence_run}
    assert panel_forward.main(forward_args(built, tmp_path / "garbage", **paths)) == 0
    clean = pq.read_table(checked / "forward.parquet")
    assert pq.read_table(tmp_path / "garbage" / "forward.parquet").equals(clean)
    original = pq.read_table(next((built["data"] / "labels").rglob("0.parquet")))
    assert not pq.read_table(grid_path)["long_24"].equals(original["long_24"])


def test_the_check_refuses_changed_runs_held_back_days_and_the_repo(
    built: dict[str, Path], tmp_path: Path
) -> None:
    def copied(key: str, folder: str) -> Path:
        target = tmp_path / folder / built[key].name
        shutil.copytree(built[key], target)
        return target

    def refused(out: str, **paths: Path) -> None:
        given = built | paths
        panel_forward.run(given["sequence"], given["inputs"], given["data"], tmp_path / out)

    unclean = copied("sequence", "unclean")
    complete = json.loads((unclean / "complete.json").read_text(encoding="utf-8"))
    complete["clean"] = False
    (unclean / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    assert panel_forward.main(forward_args(built, tmp_path / "o1", sequence=unclean)) == 1
    with pytest.raises(ValueError, match="clean"):
        refused("o1", sequence=unclean)

    swapped = copied("sequence", "swapped")
    folder = swapped / "encoders"
    twin = (folder / "full-forward-fold1-seed3-twin.pt").read_bytes()
    (folder / "full-forward-fold1-seed3-jepa.pt").write_bytes(twin)
    with pytest.raises(ValueError, match="sha256 differs"):
        refused("o2", sequence=swapped)

    late = copied("inputs", "late")
    rebound(late, {"end": "2024-01-01"})
    with pytest.raises(span.HeldBackError):
        refused("o3", inputs=late)

    grid = pq.read_table(next((built["data"] / "labels").rglob("0.parquet")))
    move = labels.ns(date(2024, 1, 1)) - labels.ns(date(2022, 1, 1)) - 100 * 24 * HOUR
    moved = grid.set_column(0, "ts_ns", pa.array(grid["ts_ns"].to_numpy() + move))
    write_grid(tmp_path / "shifted", moved.to_pydict())
    with pytest.raises(span.HeldBackError):
        refused("o4", data=tmp_path / "shifted")

    renamed = copied("inputs", "renamed")
    manifest = json.loads((renamed / "manifest.json").read_text(encoding="utf-8"))
    manifest["run_id"] = "other"
    (renamed / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not the inputs run"):
        refused("o5", inputs=renamed)

    inside = panel_forward.REPO / "harness" / "out" / "panel-forward-test"
    assert panel_forward.main(forward_args(built, inside)) == 1
    assert not inside.exists()
    (tmp_path / "o6").mkdir()
    (tmp_path / "o6" / "x").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="already holds"):
        refused("o6")
    for n in range(1, 6):
        assert not (tmp_path / f"o{n}").exists()
