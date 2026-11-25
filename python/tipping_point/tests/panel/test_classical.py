# The classical comparators on synthetic panels: rungs against the ladder, window statistics
# against a slow loop and later hours, fold isolation, the twin's probe on an identity encoder,
# the selection rule, exports the ladder reads, repeat runs and the refusals.

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.ews.scoring import auc
from tipping_point.panel import classical, inputs, labels, ladder, sequence, span, store
from tipping_point.panel.__main__ import main
from torch import nn

HOUR = labels.HOUR_NS
COINS = ["AAAUSDT", "BBBUSDT"]
Config = classical.Config

LADDER = """
start = 2022-01-01
end = 2022-12-31
horizons = [24, 1]
sides = ["long", "short", "any"]
without = []

[model]
c = 0.01
clip = 5.0

[folds]
gap_days = 31
forward_first = 2

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

[[rungs]]
name = "classic"
groups = ["vol", "classic"]
"""

PLAN = """
ladder = "ladder.toml"
variants = ["vol"]
rungs = ["full", "vol", "classic"]
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


def plans(folder: Path, plan: str = PLAN, base: str = LADDER) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ladder.toml").write_text(base, encoding="utf-8")
    path = folder / "sequence.toml"
    path.write_text(plan, encoding="utf-8")
    return path


def tiny(folder: Path, plan: str = PLAN) -> sequence.Plan:
    return sequence.load(plans(folder, plan))


def world(days: int = 365, seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    hours = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(days * 24, dtype=np.int64)
    n = len(hours) * len(COINS)
    values = {name: rng.normal(size=n) for name in inputs.OWN + inputs.MARKET}
    values["ls"][rng.random(n) < 0.01] = np.nan
    values["vol_6h"][rng.random(n) < 0.02] = np.nan
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


def write_runs(data: Path, values: dict, grid: dict, end: str = "2022-12-31") -> Path:
    labels_dir = data / "labels" / "20260101T000000Z-t"
    (labels_dir / "grid").mkdir(parents=True)
    grid_file = labels_dir / "grid" / "0.parquet"
    pq.write_table(pa.table(grid), grid_file)
    table = pa.table(
        {"ts_ns": grid["ts_ns"], "symbol": grid["symbol"].astype(str), **values},
        schema=inputs.SCHEMA,
    )
    params = {
        "labels_run": labels_dir.name,
        "labels_grid": {"file": "grid/0.parquet", "sha256": store.sha256(grid_file)},
        "start": "2021-10-01",
        "dev_start": "2022-01-01",
        "end": end,
    }
    return store.write(data, "panel-inputs", "inputs", {"inputs": table}, params, {})


def keyed(rows: list[dict]) -> dict:
    names = ("model", "variant", "readout", "c", "scheme", "horizon", "side", "fold")
    return {tuple(r[k] for k in names): r for r in rows}


def windows(plan: sequence.Plan) -> list[Config]:
    return [c for c in classical.configs(plan) if c.model == "window"]


def rungs(plan: sequence.Plan) -> list[Config]:
    return [c for c in classical.configs(plan) if c.model == "rung"]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> dict:
    folder = tmp_path_factory.mktemp("fitted")
    values, grid = world()
    plan = tiny(folder)
    out = classical.classical(values, grid, plan, classical.configs(plan))
    return {"values": values, "grid": grid, "plan": plan, "out": out}


def test_the_shipped_plan_gives_the_shipped_grid() -> None:
    plan = sequence.load()
    assert classical.spans(plan) == (24, 72, 336)
    assert classical.names(plan) == ("full", "vol", "classic")
    found = classical.configs(plan)
    assert len(found) == 3 * 6 + 2 * 3 * 6 and len(set(found)) == len(found)
    assert {c.variant for c in found if c.model == "window"} == {"full", "vol"}
    assert {c.c for c in found} == set(plan.cs)


def test_the_rungs_at_the_ladders_c_are_the_ladders_scores(fitted: dict) -> None:
    values, grid, plan = fitted["values"], fitted["grid"], fitted["plan"]
    edges = ladder.bounds(plan.ladder)
    rows = sequence.common_rows(values, grid, plan)
    ts = grid["ts_ns"]
    checked = 0
    for h, r in rows.items():
        x = np.column_stack([values[c][r.keep] for c in plan.ladder.columns])
        for scheme in plan.schemes:
            args = (x, r.y, ts[r.keep], r.fold, edges)
            pairs = [(n, c) for n in plan.rungs for c in plan.cs]
            grid_scores = ladder.penalised(*args, plan.ladder, scheme, h, lambda line: None, pairs)
            for c in plan.cs:
                at = replace(plan.ladder, c=c)
                theirs = ladder.out_of_fold(*args, at, scheme, h, lambda line: None)
                for name in plan.rungs:
                    for side in plan.ladder.sides:
                        mine = grid_scores[(name, c)][side]
                        assert np.array_equal(mine, theirs[name][side]), (h, scheme, c, name)
                        checked += 1
            low, high = grid_scores[("full", 0.01)]["long"], grid_scores[("full", 1.0)]["long"]
            assert not np.array_equal(low, high)
    assert checked == len(rows) * len(plan.schemes) * len(plan.cs) * 3 * 3
    rung_rows = sequence.rung_folds(values, rows, ts, edges, plan, lambda line: None)
    mine = keyed([r for r in fitted["out"]["folds"] if r["model"] == "rung"])
    assert {k[4:] for k in mine} == {k[4:] for k in keyed(rung_rows)}
    for key, row in keyed(rung_rows).items():
        assert mine[key] == row, key
    assert {r["c"] for r in mine.values()} == set(plan.cs)


def standardised_panel(seed: int, nhours: int, plan: sequence.Plan) -> tuple:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(3, nhours, 4)) * 2
    raw[0, 10:14, 1] = np.nan
    raw[1, :, 2] = np.nan
    raw[2, 50:60] = np.nan
    raw[2, 70, 0] = 40.0
    mu, sd = sequence.statistics(raw, np.ones(nhours, bool))
    return raw, mu, sd


def slow(x: np.ndarray, coin: int, hour: int, plan: sequence.Plan) -> np.ndarray:
    window = [x[coin, plan.pad + hour - k] for k in range(plan.window - 1, -1, -1)]
    channels = x.shape[2]
    means, spreads = [], []
    for n in classical.spans(plan):
        rows = window[-n:]
        mean = np.zeros(channels)
        for row in rows:
            mean += row.astype(np.float64)
        mean /= n
        square = np.zeros(channels)
        for row in rows:
            square += (row.astype(np.float64) - mean) ** 2
        means.append(mean)
        spreads.append(np.sqrt(square / n))
    return np.concatenate([window[-1], *means, *spreads])


def test_the_window_statistics_are_a_slow_loop_over_the_window(tmp_path: Path) -> None:
    plan = tiny(tmp_path)
    raw, mu, sd = standardised_panel(1, 120, plan)
    x = sequence.standardised(raw, mu, sd, plan.ladder.clip, plan.pad)
    coin = np.array([0, 0, 0, 1, 2, 2, 2, 1, 0])
    hour = np.array([0, 5, 12, 35, 55, 71, 119, 2, 119])
    got = classical.statistics(x, coin, hour, plan)
    assert got.dtype == np.float32 and got.shape == (len(coin), 7 * 5)
    for i, (k, t) in enumerate(zip(coin, hour, strict=True)):
        want = slow(x, int(k), int(t), plan)
        assert np.allclose(got[i], want, rtol=0, atol=1e-6), (k, t)
        assert np.array_equal(got[i, :5], x[k, plan.pad + t])
    assert got[0, 4] == 0.0 and got[1, 5 + 4] > 0.0
    assert got[3, 2] == 0.0 and got[3, 4] == 0.25
    flat = classical.statistics(
        np.ones((1, plan.pad + 3, 2), np.float32), np.zeros(3, int), np.arange(3), plan
    )
    assert np.array_equal(flat, np.concatenate([np.ones((3, 8)), np.zeros((3, 6))], 1))


def test_nothing_after_hour_t_reaches_the_statistics_at_t(tmp_path: Path) -> None:
    plan = tiny(tmp_path)
    raw, mu, sd = standardised_panel(2, 200, plan)
    coin, hour = np.array([0, 1, 2]), np.array([120, 3, 60])

    def stats(r: np.ndarray) -> np.ndarray:
        x = sequence.standardised(r, mu, sd, plan.ladder.clip, plan.pad)
        return classical.statistics(x, coin, hour, plan)

    before = stats(raw)
    later = raw.copy()
    later[0, 121:] = 1e6
    later[1, 4:] = np.random.default_rng(3).normal(size=later[1, 4:].shape) * 50
    later[1, 4:7] = np.nan
    later[2, 61:] = np.nan
    assert np.array_equal(stats(later), before)
    for j, moves in ((120, True), (120 - plan.window + 1, True), (120 - plan.window, False)):
        moved = raw.copy()
        moved[0, j, 0] += 1.0
        assert np.array_equal(stats(moved)[0], before[0]) != moves, j
        assert np.array_equal(stats(moved)[1:], before[1:])


def perturbed(values: dict, grid: dict, start: int, end: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(9)
    region = (grid["ts_ns"] >= start) & (grid["ts_ns"] < end)
    moved = {}
    for name, v in values.items():
        moved[name] = np.where(region & np.isfinite(v), v + rng.normal(size=v.shape) * 10, v)
    flipped = dict(grid)
    for h in labels.HORIZONS:
        for side in ("long", "short"):
            flipped[f"{side}_{h}"] = np.where(region, ~grid[f"{side}_{h}"], grid[f"{side}_{h}"])
    return moved, flipped


def test_the_left_out_quarter_and_its_gaps_reach_no_window_fit(fitted: dict) -> None:
    values, grid, plan = fitted["values"], fitted["grid"], fitted["plan"]
    edges = ladder.bounds(plan.ladder)
    gap = plan.ladder.gap_days * 24 * HOUR
    f = 2
    moved, flipped = perturbed(values, grid, edges[f] - gap, edges[f + 1] + gap)
    chosen = windows(plan)
    second = classical.classical(moved, flipped, plan, chosen)
    first = fitted["out"]
    mine = [k for k in first["fitted"] if k.endswith(f"fold {f + 1}")]
    assert sorted(mine) == ["vol forward fold 3", "vol loqo fold 3"]
    for key in mine:
        assert first["fitted"][key] == second["fitted"][key], key
    others = [k for k in first["fitted"] if k.startswith("vol loqo") and k not in mine]
    assert others and all(first["fitted"][k] != second["fitted"][k] for k in others)
    before = keyed([r for r in first["folds"] if r["model"] == "window" and r["fold"] == f])
    after = keyed([r for r in second["folds"] if r["fold"] == f])
    assert before.keys() == after.keys()
    assert [r["auc"] for r in before.values()] != [r["auc"] for r in after.values()]
    p = classical.panel(values, grid, plan)
    seen = set()
    for scheme, g, h, test, _ in classical.window_scores(
        values, p, plan, "vol", chosen[:1], lambda line: None, [], {}
    ):
        assert np.array_equal(test, p.rows[h].fold == g)
        seen.add((scheme, g, h))
    assert len(seen) == (4 + 2) * 2


class Stand(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_the_twins_probe_on_an_identity_encoder_is_the_window_comparator(
    fitted: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    values, grid, plan = fitted["values"], fitted["grid"], fitted["plan"]
    channels = len(plan.rung("vol").columns) + 1
    stand = replace(
        plan, models=("twin",), shape=replace(plan.shape, width=channels), variants=("vol",)
    )

    def features(
        net: nn.Module, x: torch.Tensor, coin: np.ndarray, hour: np.ndarray, p: object
    ) -> np.ndarray:
        assert isinstance(net, Stand)
        return classical.statistics(x.numpy(), coin, hour, stand)

    monkeypatch.setattr(sequence.encoder, "encoder", lambda *args, **kwargs: Stand())
    monkeypatch.setattr(sequence, "features", features)
    twin = sequence.sequence(values, grid, stand)
    rows = [r for r in twin["folds"] if r["model"] == "twin"]
    window = keyed([r for r in fitted["out"]["folds"] if r["model"] == "window"])
    assert len(rows) == len(window) > 0
    for r in rows:
        key = ("window", r["variant"], r["readout"], r["c"], r["scheme"], r["horizon"])
        mine = window[(*key, r["side"], r["fold"])]
        assert (mine["auc"], mine["rows"], mine["positives"]) == (
            r["auc"],
            r["rows"],
            r["positives"],
        ), key


def fold_row(model: str, readout: str | None, c: float, fold: int, side: str, value: float, **kw):
    return {
        "model": model,
        "readout": readout,
        "c": c,
        "fold": fold,
        "side": side,
        "auc": value,
        "seed": 0,
        "variant": kw.get("variant", "full"),
        "scheme": kw.get("scheme", "forward"),
        "horizon": kw.get("horizon", 24),
    }


def test_the_choice_is_the_rule_over_each_names_candidates() -> None:
    rows = []
    for fold in (2, 3):
        for side in ("long", "short"):
            rows.append(fold_row("rung", None, 0.01, fold, side, 0.60))
            rows.append(fold_row("rung", None, 1.0, fold, side, 0.62))
            rows.append(fold_row("window", "last", 0.01, fold, side, 0.62))
            rows.append(fold_row("window", "means", 0.001, fold, side, 0.61))
            rows.append(fold_row("window", "spreads", 0.1, fold, side, 0.99, horizon=1))
            rows.append(fold_row("window", "spreads", 0.1, fold, side, 0.99, scheme="loqo"))
            rows.append(fold_row("window", "spreads", 0.1, fold, side, 0.99, variant="vol"))
            rows.append(fold_row("twin", "spreads", 0.1, fold, side, 0.99))
            rows.append(fold_row("rung", None, 0.1, fold, side, 0.70, variant="classic"))
            rows.append(fold_row("rung", None, 0.01, fold, side, 0.50, variant="classic"))
        rows.append(fold_row("window", "spreads", 0.1, fold, "any", 0.99))
    config, score, listed = classical.choose(rows, "full")
    assert config == Config("rung", "full", None, 1.0) and score == pytest.approx(0.62)
    assert [(d["model"], d["readout"], d["c"]) for d in listed] == [
        ("rung", None, 0.01),
        ("rung", None, 1.0),
        ("window", "last", 0.01),
        ("window", "means", 0.001),
        ("window", "spreads", 0.1),
    ]
    assert listed[-1]["score"] is None and listed[1]["score"] == score
    assert classical.choose(rows, "classic")[0] == Config("rung", "classic", None, 0.1)
    assert classical.choose(rows, "vol")[0] == Config("window", "vol", "spreads", 0.1)
    no_rung = [r for r in rows if r["model"] != "rung"]
    assert classical.choose(no_rung, "full")[0] == Config("window", "full", "last", 0.01)
    readouts = [r | {"readout": "spreads"} for r in no_rung if r["readout"] == "last"]
    readouts += [r | {"readout": "means", "c": 0.01} for r in no_rung if r["readout"] == "last"]
    assert classical.choose(readouts, "full")[0] == Config("window", "full", "means", 0.01)
    cs = [r | {"c": 0.1} for r in readouts] + [r | {"c": 0.001} for r in readouts]
    assert classical.choose(cs, "full")[0] == Config("window", "full", "means", 0.001)
    holes = [
        r | {"auc": float("nan")} if (r["model"], r["c"], r["fold"]) == ("rung", 1.0, 3) else r
        for r in rows
    ]
    holes.append(fold_row("rung", None, 0.01, 4, "long", 0.30))
    config, score, listed = classical.choose(holes, "full")
    assert config == Config("rung", "full", None, 1.0) and score == pytest.approx(0.62)
    assert listed[0]["score"] == pytest.approx((np.mean([0.60, 0.60, 0.30]) + 0.60) / 2)
    gone = [r | {"auc": float("nan")} if (r["c"], r["side"]) == (1.0, "short") else r for r in rows]
    config, _, listed = classical.choose(gone, "full")
    assert config == Config("window", "full", "last", 0.01) and listed[1]["score"] is None
    with pytest.raises(ValueError, match="no forward 24 h fold scores of the lev comparators"):
        classical.choose(rows, "lev")


def file_aucs(table: pa.Table, values: dict, grid: dict, plan: sequence.Plan) -> dict:
    rows = sequence.common_rows(values, grid, plan)
    symbol = grid["symbol"].astype(str)
    scheme = table["scheme"][0].as_py()
    out = {}
    for h, r in rows.items():
        at = {
            (k, t): i
            for i, (k, t) in enumerate(zip(symbol[r.keep], grid["ts_ns"][r.keep], strict=True))
        }
        for side in plan.ladder.sides:
            mask = (table["horizon"].to_numpy() == h) & (
                table["side"].to_numpy(zero_copy_only=False) == side
            )
            part = table.filter(pa.array(mask))
            index = np.array(
                [
                    at[(k, t)]
                    for k, t in zip(part["coin"].to_pylist(), part["hour"].to_pylist(), strict=True)
                ]
            )
            assert len(index) == int((r.fold >= plan.ladder.first_fold(scheme)).sum())
            s, fold = part["score"].to_numpy(), part["fold"].to_numpy()
            assert np.array_equal(fold, r.fold[index])
            for f in np.unique(fold):
                label = r.y[side][index[fold == f]]
                out[(scheme, h, side, int(f))] = auc(s[fold == f][label], s[fold == f][~label])
    return out


EXTERNAL = pa.schema(
    [
        ("coin", pa.string()),
        ("hour", pa.int64()),
        ("horizon", pa.int16()),
        ("side", pa.string()),
        ("score", pa.float64()),
        ("fold", pa.int16()),
        ("scheme", pa.string()),
    ]
)


def test_an_exported_configuration_scores_its_folds_table_rows(fitted: dict) -> None:
    values, grid, plan = fitted["values"], fitted["grid"], fitted["plan"]
    folds = keyed(fitted["out"]["folds"])
    for config in (Config("rung", "classic", None, 1.0), Config("window", "vol", "spreads", 1.0)):
        tables = classical.export(values, grid, plan, config)
        assert set(tables) == set(plan.schemes)
        for scheme, table in tables.items():
            assert table.schema == EXTERNAL
            got = file_aucs(table, values, grid, plan)
            assert len(got) == 2 * 3 * (4 if scheme == "loqo" else 2)
            for (s, h, side, f), value in got.items():
                row = folds[(config.model, config.variant, config.readout, config.c, s, h, side, f)]
                assert value == row["auc"], (config, s, h, side, f)
    with pytest.raises(ValueError, match="no classical configuration"):
        classical.export(values, grid, plan, Config("window", "full", "last", 0.01))


def run_args(data: Path, run_inputs: Path, plan: Path, tag: str, *extra: str) -> list[str]:
    command = ["classical", "--inputs", run_inputs.name, "--plan", str(plan)]
    return [*command, "--data-dir", str(data), "--tag", tag, *extra]


FORWARD = PLAN.replace('schemes = ["loqo", "forward"]', 'schemes = ["forward"]')


def test_the_command_chooses_writes_and_the_ladder_scores_the_choice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values, grid = world()
    run_inputs = write_runs(tmp_path, values, grid)
    path = plans(tmp_path / "plan", FORWARD)
    for tag in ("one", "two"):
        assert main(run_args(tmp_path, run_inputs, path, tag)) == 0
    one, two = (next((tmp_path / "panel-classical").glob(f"*-{t}")) for t in ("one", "two"))
    assert str(two) in capsys.readouterr().out
    complete = json.loads((one / "complete.json").read_text(encoding="utf-8"))
    params = json.loads((one / "manifest.json").read_text(encoding="utf-8"))["params"]
    exported = {f"{n}-forward.parquet" for n in ("full", "vol", "classic")}
    assert set(complete["files"]) == {"folds.parquet", "rows.parquet"} | exported | {
        "report.md",
        "plan.toml",
        "ladder.toml",
    }
    for name in [n for n in complete["files"] if n.endswith(".parquet")]:
        assert (one / name).read_bytes() == (two / name).read_bytes(), name
    plan = sequence.load(one / "plan.toml", one / "ladder.toml")
    folds = pq.read_table(one / "folds.parquet")
    assert folds.schema == classical.FOLDS
    rows = folds.to_pylist()
    assert {r["model"] for r in rows} == {"rung", "window"} and {r["seed"] for r in rows} == {0}
    assert all(r["readout"] is None for r in rows if r["model"] == "rung")
    selection = params["selection"]
    assert selection["metric"] == classical.METRIC
    for name in classical.names(plan):
        config, score, listed = classical.choose(rows, name)
        assert selection["chosen"][name] == {**config.__dict__, "score": score}
        assert selection["candidates"][name] == listed
        assert params["exported"][name] == config.__dict__
        assert len(listed) == len(plan.cs) * (1 + (name in plan.variants) * len(plan.readouts))
        table = pq.read_table(one / f"{name}-forward.parquet")
        assert table.schema == EXTERNAL
        store.refuse_held_back(one / f"{name}-forward.parquet", "hour")
        for (s, h, side, f), value in file_aucs(table, values, grid, plan).items():
            match = [
                r
                for r in rows
                if (r["model"], r["variant"], r["readout"], r["c"])
                == tuple(config.__dict__.values())
                and (r["scheme"], r["horizon"], r["side"], r["fold"]) == (s, h, side, f)
            ]
            assert len(match) == 1 and match[0]["auc"] == value
    assert params["inputs"]["sha256"] == store.sha256(run_inputs / "inputs.parquet")
    assert params["plan"]["sha256"] == complete["files"]["plan.toml"]
    assert params["ladder_plan"]["sha256"] == complete["files"]["ladder.toml"]
    assert params["runtime"]["pins"].keys() == set(sequence.PINS)
    fixed = ["--fixed", "rung", "full", "-", "0.01"]
    assert main(run_args(tmp_path, run_inputs, path, "fixed", *fixed)) == 0
    run = next((tmp_path / "panel-classical").glob("*-fixed"))
    params = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["params"]
    assert params["selection"] is None and params["fixed"] == params["exported"]["full"]
    file = run / "full-forward.parquet"
    external = f'\n[external]\nname = "ext"\npath = "{file.as_posix()}"\nscheme = "forward"\n'
    (tmp_path / "ladder-ext.toml").write_text(LADDER + external, encoding="utf-8")
    ladder_run, _ = ladder.run(run_inputs, tmp_path, tmp_path / "ladder-ext.toml", "ext")
    scored = pq.read_table(ladder_run / "folds.parquet").to_pylist()
    ext = {
        (r["horizon"], r["side"], r["fold"], r["coins"]): r for r in scored if r["name"] == "ext"
    }
    full = {
        (r["horizon"], r["side"], r["fold"], r["coins"]): r
        for r in scored
        if r["name"] == "full" and r["scheme"] == "forward"
    }
    assert ext.keys() == full.keys() and ext
    for key, row in ext.items():
        assert (row["auc"], row["rows"]) == (full[key]["auc"], full[key]["rows"]), key
    bad = ["--fixed", "window", "classic", "last", "0.01"]
    assert main(run_args(tmp_path, run_inputs, path, "bad", *bad)) == 1
    assert not list((tmp_path / "panel-classical").glob("*-bad"))


def test_the_command_refuses_before_reading_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values, grid = world(days=200)
    path = plans(tmp_path / "plan", FORWARD)
    run_inputs = write_runs(tmp_path / "a", values, grid)

    def complete(change: dict) -> None:
        record = json.loads((run_inputs / "complete.json").read_text(encoding="utf-8"))
        (run_inputs / "complete.json").write_text(json.dumps(record | change), "utf-8")

    def manifest(change: dict) -> None:
        record = json.loads((run_inputs / "manifest.json").read_text(encoding="utf-8"))
        record["params"] |= change
        (run_inputs / "manifest.json").write_text(json.dumps(record), "utf-8")

    saved = {
        n: (run_inputs / n).read_text(encoding="utf-8") for n in ("manifest.json", "complete.json")
    }
    cases = [
        (lambda: complete({"clean": False}), "does not mark the run clean"),
        (lambda: complete({"files": {"inputs.parquet": "0" * 64}}), "sha256 differs"),
        (
            lambda: manifest({"labels_grid": {"file": "grid/0.parquet", "sha256": "0" * 64}}),
            "not the grid",
        ),
        (lambda: manifest({"end": "2024-01-01"}), "held-back"),
    ]
    for change, message in cases:
        change()
        with pytest.raises(ValueError, match=message):
            classical.run(run_inputs, tmp_path / "a", path, "x")
        for n, text in saved.items():
            (run_inputs / n).write_text(text, encoding="utf-8")
    shift = labels.ns(date(2024, 1, 1)) - labels.ns(date(2022, 1, 1)) - 5 * 24 * HOUR
    moved = write_runs(tmp_path / "b", values, dict(grid, ts_ns=grid["ts_ns"] + shift))
    with pytest.raises(span.HeldBackError):
        classical.run(moved, tmp_path / "b", path, "x")
    late_grid = write_runs(tmp_path / "c", values, grid)
    grid_file = tmp_path / "c" / "labels" / "20260101T000000Z-t" / "grid" / "0.parquet"
    pq.write_table(pa.table(dict(grid, ts_ns=grid["ts_ns"] + shift)), grid_file)
    with pytest.raises(span.HeldBackError):
        classical.run(late_grid, tmp_path / "c", path, "x")
    hashed: list[Path] = []
    sha = store.sha256
    monkeypatch.setattr(store, "sha256", lambda p: hashed.append(p) or sha(p))
    with pytest.raises(span.HeldBackError):
        classical.run(moved, tmp_path / "b", path, "x")
    assert hashed == []
    monkeypatch.undo()
    for folder in ("a", "b", "c"):
        assert not (tmp_path / folder / "panel-classical").exists()
    late = LADDER.replace("end = 2022-12-31", "end = 2024-01-01")
    with pytest.raises(span.HeldBackError):
        classical.run(run_inputs, tmp_path / "a", plans(tmp_path / "late", FORWARD, late), "x")
    gappy = plans(tmp_path / "gappy", FORWARD.replace("folds = []", "folds = [3]"))
    with pytest.raises(ValueError, match=r"quarters \[3\] unscored"):
        classical.run(run_inputs, tmp_path / "a", gappy, "x")
    monkeypatch.setattr(sequence, "read", lambda *args: pytest.fail("read the inputs"))
    with pytest.raises(ValueError, match="no classical configuration"):
        classical.run(run_inputs, tmp_path / "a", path, "x", Config("window", "classic", "last", 1))


def test_a_plan_replaced_while_it_is_read_configures_and_records_one_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values, grid = world(days=200)
    run_inputs = write_runs(tmp_path, values, grid)
    path = plans(tmp_path / "plan", FORWARD)
    swaps = {
        path.resolve(): ("c = [0.01, 1.0]", "c = [0.01, 0.1]"),
        path.with_name("ladder.toml").resolve(): ("c = 0.01", "c = 0.1"),
    }
    reads: dict[Path, int] = {}
    read_text = Path.read_text

    def replaced(self: Path, encoding: str | None = None) -> str:
        text = read_text(self, encoding=encoding)
        key = self.resolve()
        reads[key] = reads.get(key, 0) + 1
        return text.replace(*swaps[key]) if key in swaps and reads[key] > 1 else text

    monkeypatch.setattr(Path, "read_text", replaced)
    run_dir, _ = classical.run(run_inputs, tmp_path, path, "replaced")
    monkeypatch.undo()
    folds = pq.read_table(run_dir / "folds.parquet").to_pylist()
    assert {r["c"] for r in folds} == {0.01, 1.0}
    params = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["params"]
    assert (run_dir / "plan.toml").read_text(encoding="utf-8") == FORWARD
    assert (run_dir / "ladder.toml").read_text(encoding="utf-8") == LADDER
    assert params["plan"]["sha256"] == store.sha256(run_dir / "plan.toml")
    assert params["ladder_plan"]["sha256"] == store.sha256(run_dir / "ladder.toml")
