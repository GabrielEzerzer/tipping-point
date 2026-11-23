# The ladder on planted panels: the shipped plan, quarters and their gap, the common row set, the
# pair-count AUC and its block bootstrap against direct calculations, a planted signal and a null,
# external scores, the command's run and its held-back refusal, and the docs section.

import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.optimize import minimize
from tipping_point.ews.scoring import auc
from tipping_point.panel import inputs, labels, ladder, span, store
from tipping_point.panel.__main__ import main

HOUR = labels.HOUR_NS
DAY = 24 * HOUR
COINS = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
ROOT = Path(__file__).resolve().parents[4]

PLAN = """
start = 2022-01-01
end = 2022-12-31
horizons = [24]
sides = ["long", "short", "any"]
without = ["AAAUSDT"]

[model]
c = 0.01
clip = 5.0

[folds]
gap_days = 28

[bootstrap]
block_days = 14
draws = 300
seed = 7

[[rungs]]
name = "vol"
groups = ["vol"]

[[rungs]]
name = "vol+price"
groups = ["vol", "price"]

[[rungs]]
name = "classic"
groups = ["vol", "classic"]

[[comparisons]]
first = "vol+price"
second = "vol"

[[comparisons]]
first = "classic"
second = "vol"
"""


def plan_file(tmp_path: Path, extra: str = "", text: str = PLAN) -> Path:
    path = tmp_path / "plan.toml"
    path.write_text(text + extra, encoding="utf-8")
    return path


def world(signal: float = 1.2, seed: int = 0, days: int = 365) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    hours = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(days * 24, dtype=np.int64)
    n = len(hours) * len(COINS)
    values = {name: rng.normal(size=n) for name in inputs.OWN + inputs.MARKET}
    grid: dict[str, np.ndarray] = {
        "ts_ns": np.repeat(hours, len(COINS)),
        "symbol": np.tile(np.array(COINS, dtype=object), len(hours)),
    }
    for h in labels.HORIZONS:
        lift = signal * values["ret_24h"]
        grid[f"long_{h}"] = rng.random(n) < 1 / (1 + np.exp(3.0 - lift))
        grid[f"short_{h}"] = rng.random(n) < 1 / (1 + np.exp(3.5 + lift))
        grid[f"censored_{h}"] = rng.random(n) < 0.01
    grid["skip"] = rng.random(n) < 0.02
    return values, grid


def test_the_shipped_plan() -> None:
    plan = ladder.load()
    assert plan.horizons == (24, 6, 1) and plan.sides == ("long", "short", "any")
    assert (plan.start, plan.end) == (span.DEV_START, span.END)
    assert plan.names == ("vol", "vol+price", "vol+price+flow", "full", "full+mlev", "classic")
    assert plan.comparisons == (
        ("full", "vol"),
        ("vol+price", "vol"),
        ("vol+price+flow", "vol+price"),
        ("full", "vol+price+flow"),
        ("full+mlev", "full"),
        ("classic", "vol"),
    )
    assert (plan.c, plan.clip, plan.gap_days) == (0.01, 5.0, 28)
    assert (plan.block_days, plan.draws) == (14, 2000) and plan.external is None
    rungs = {r.name: set(r.columns) for r in plan.rungs}
    assert rungs["vol"] == set(inputs.GROUPS["vol"]) | {f"m_{n}" for n in inputs.GROUPS["vol"]}
    assert "oi_24h" in rungs["full"] and "m_oi_24h" not in rungs["full"]
    assert rungs["full+mlev"] - rungs["full"] == {f"m_{n}" for n in inputs.GROUPS["lev"]}
    assert "m_var_tau_24h" in rungs["classic"] and "ret_1h" not in rungs["classic"]
    assert set(plan.columns) == set(inputs.OWN + inputs.MARKET)


def test_folds_are_calendar_quarters_and_training_keeps_28_days_away() -> None:
    plan = ladder.load()
    edges = ladder.bounds(plan)
    starts = [date(y, m, 1) for y in (2022, 2023) for m in (1, 4, 7, 10)] + [date(2024, 1, 1)]
    assert edges.tolist() == [labels.ns(d) for d in starts]
    april, july = labels.ns(date(2022, 4, 1)), labels.ns(date(2022, 7, 1))
    ts = np.array(
        [
            *(april - HOUR, april, july - HOUR, april - 28 * DAY - HOUR, april - 28 * DAY),
            *(july + 28 * DAY - HOUR, july + 28 * DAY),
        ],
        np.int64,
    )
    fold, block = ladder.folds(ts, edges, 14)
    assert fold.tolist() == [0, 1, 1, 0, 0, 2, 2]
    assert block.tolist()[:3] == [6, 0, 6]
    train = ladder.training(ts, fold, 1, edges, 28)
    assert train.tolist() == [False, False, False, True, False, False, True]
    assert ladder.blocks_per_fold(edges, 14).tolist() == [7] * 8


def test_every_rung_is_scored_on_one_common_row_set() -> None:
    plan = ladder.load()
    values, grid = world(days=3)
    n = len(grid["ts_ns"])
    for name in ("skip", "censored_24", "long_24", "short_24"):
        grid[name] = np.zeros(n, bool)
    grid["skip"][[0, 1]] = True
    grid["censored_24"][[1, 2]] = True
    grid["long_24"][[1, 2, 3, 10]] = True
    values["m_ls_rel"][3] = np.nan
    values["ls"][4] = np.nan
    values["vol_1h"][5] = np.nan
    keep, accounting = ladder.select(values, grid, plan, 24)
    assert np.flatnonzero(~keep).tolist() == [0, 1, 2, 3, 4, 5]
    removed = {row["condition"]: (row["rows"], row["long"]) for row in accounting}
    assert removed == {
        "grid": (n, 4),
        "outside the plan's span": (0, 0),
        "skipped": (2, 1),
        "censored at 24 h": (1, 1),
        "missing an input": (3, 1),
        "scored": (n - 6, 1),
    }


def test_pair_counts_give_the_mann_whitney_auc_unweighted_and_weighted() -> None:
    rng = np.random.default_rng(4)
    n, k = 3000, 7
    score = np.round(rng.normal(size=n), 1)
    label = rng.random(n) < 0.1
    block = rng.integers(0, k, n)
    counts = ladder.pair_counts(score, label, block, k)
    assert ladder.block_auc(*counts)[0] == pytest.approx(auc(score[label], score[~label]))
    weights = rng.integers(0, 4, size=(5, k)).astype(np.float64)
    for w, got in zip(weights, ladder.block_auc(*counts, weights), strict=True):
        repeat = w[block].astype(int)
        s, y = np.repeat(score, repeat), np.repeat(label, repeat)
        assert got == pytest.approx(auc(s[y], s[~y]))


def test_the_bootstrap_draws_whole_blocks_with_every_coins_rows() -> None:
    values, grid = world(days=90)
    plan = ladder.load()
    ts, symbol = grid["ts_ns"], grid["symbol"]
    fold, block = ladder.folds(ts, ladder.bounds(plan), 14)
    assert (fold == 0).all()
    score, label = values["ret_24h"], grid["long_24"]
    draws = ladder.draws(np.array([7]), 20, seed=3, horizon=24)[0]
    assert draws.shape == (20, 7) and (draws.sum(axis=1) == 7).all()
    assert np.array_equal(draws, ladder.draws(np.array([7]), 20, seed=3, horizon=24)[0])
    assert not np.array_equal(draws, ladder.draws(np.array([7]), 20, seed=4, horizon=24)[0])
    fast = ladder.block_auc(*ladder.pair_counts(score, label, block, 7), draws)
    for d in range(len(draws)):
        drawn = np.repeat(np.arange(7), draws[d].astype(int))
        rows = np.concatenate([np.flatnonzero(block == b) for b in drawn])
        for b in drawn:
            assert sorted(set(symbol[block == b])) == COINS
        s, y = score[rows], label[rows]
        assert fast[d] == pytest.approx(auc(s[y], s[~y]))


def test_the_newton_fit_minimises_the_l2_logistic_loss() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 4))
    y = (rng.random(800) < 1 / (1 + np.exp(-x @ [1.0, -0.5, 0.0, 0.3]))).astype(float)
    c = 0.01

    def loss(b: np.ndarray) -> float:
        z = x @ b[:-1] + b[-1]
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 / c * np.sum(b[:-1] ** 2))

    got = ladder.fit(x, y, c)
    best = minimize(loss, np.zeros(5), method="BFGS", options={"gtol": 1e-10}).x
    assert np.allclose(got, best, atol=1e-5)
    assert ladder.predict(x, got) == pytest.approx(1 / (1 + np.exp(-(x @ got[:-1] + got[-1]))))


def scored(out: dict, name: str, side: str, coins: str = "all") -> dict:
    (row,) = [
        s for s in out["scores"] if s["name"] == name and s["side"] == side and s["coins"] == coins
    ]
    return row


def compared(out: dict, first: str, side: str) -> dict:
    (row,) = [c for c in out["comparisons"] if c["first"] == first and c["side"] == side]
    return row


def test_a_planted_signal_one_rung_sees_and_a_null(tmp_path: Path) -> None:
    values, grid = world()
    out = ladder.ladder(values, grid, ladder.load(plan_file(tmp_path)))
    seen = compared(out, "vol+price", "long")
    assert seen["q025"] > 0 and seen["above_zero"] == 1.0
    assert seen["folds_above_zero"] == seen["folds"] == 4
    assert seen["estimate"] == pytest.approx(
        scored(out, "vol+price", "long")["fold_mean"] - scored(out, "vol", "long")["fold_mean"]
    )
    assert scored(out, "vol+price", "long")["fold_mean"] > 0.7
    null = compared(out, "classic", "long")
    assert null["q025"] < 0 < null["q975"]
    assert 0.45 < scored(out, "vol", "any")["pooled"] < 0.55
    assert scored(out, "vol+price", "any")["pooled"] > 0.6
    (oof,) = out["oof"]
    others = np.array(oof["symbol"].to_pylist()) != "AAAUSDT"
    assert scored(out, "vol+price", "long", "without")["rows"] == int(others.sum())
    assert oof.num_rows == out["accounting"][24][-1]["rows"]
    p = {s: oof[f"vol+price:{s}"].to_numpy() for s in ("long", "short", "any")}
    assert np.allclose(p["any"], 1 - (1 - p["long"]) * (1 - p["short"]))


def external_table(values: dict, grid: dict, sides: list[str], drop: int = 0) -> pa.Table:
    n = len(grid["ts_ns"]) - drop
    parts = [
        pa.table(
            {
                "coin": grid["symbol"][:n].astype(str),
                "hour": grid["ts_ns"][:n],
                "horizon": np.full(n, 24),
                "side": [side] * n,
                "score": (1 if side != "short" else -1) * values["ret_24h"][:n],
            }
        )
        for side in sides
    ]
    return pa.concat_tables(parts)


def test_external_scores_are_scored_and_compared_like_a_rung(tmp_path: Path) -> None:
    values, grid = world()
    extra = '\n[external]\nname = "planted"\npath = "planted.parquet"\n'
    extra += '\n[[comparisons]]\nfirst = "planted"\nsecond = "vol"\n'
    plan = ladder.load(plan_file(tmp_path, extra))
    assert plan.external == ("planted", (tmp_path / "planted.parquet").resolve())
    table = external_table(values, grid, ["long", "short", "any"])
    out = ladder.ladder(values, grid, plan, table)
    planted = compared(out, "planted", "long")
    assert planted["q025"] > 0
    assert (
        scored(out, "planted", "long")["fold_mean"]
        > scored(out, "vol+price", "long")["fold_mean"] - 0.02
    )
    assert scored(out, "planted", "short")["fold_mean"] > 0.7
    with pytest.raises(ValueError, match="miss"):
        ladder.ladder(values, grid, plan, external_table(values, grid, list(ladder.SIDES), 5))
    with pytest.raises(ValueError, match="miss"):
        ladder.ladder(values, grid, plan, external_table(values, grid, ["long", "short"]))


def write_runs(data: Path, values: dict, grid: dict, end: str = "2022-12-31") -> Path:
    labels_dir = data / "labels" / "20260101T000000Z-t"
    (labels_dir / "grid").mkdir(parents=True)
    grid_file = labels_dir / "grid" / "0.parquet"
    pq.write_table(pa.table(grid), grid_file)
    files = {"grid/0.parquet": store.sha256(grid_file)}
    (labels_dir / "complete.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    table = pa.table(
        {"ts_ns": grid["ts_ns"], "symbol": grid["symbol"].astype(str), **values},
        schema=inputs.SCHEMA,
    )
    params = {
        "labels_run": labels_dir.name,
        "labels_grid": {"file": "grid/0.parquet", "sha256": files["grid/0.parquet"]},
        "start": "2021-10-01",
        "dev_start": "2022-01-01",
        "end": end,
    }
    return store.write(data, "panel-inputs", "inputs", {"inputs": table}, params, {})


def test_the_command_writes_a_ladder_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    values, grid = world()
    run_inputs = write_runs(tmp_path, values, grid)
    plan = plan_file(tmp_path)
    where = ["--plan", str(plan), "--data-dir", str(tmp_path)]
    assert main(["ladder", "--inputs", run_inputs.name, *where]) == 0
    (run,) = (tmp_path / "panel-ladder").glob("*-ladder")
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    assert set(complete["files"]) == {
        f"{n}.parquet" for n in ("oof", "scores", "folds", "comparisons", "rows")
    } | {"report.md", "plan.toml"}
    assert (run / "plan.toml").read_text(encoding="utf-8") == plan.read_text(encoding="utf-8")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["params"]["inputs_run"] == run_inputs.name
    assert manifest["params"]["plan"]["sha256"] == store.sha256(plan)
    report = (run / "report.md").read_text(encoding="utf-8")
    for text in ("## 24 h", "| vol+price - vol | long |", "| missing an input |", "AAAUSDT"):
        assert text in report
    comparisons = pq.read_table(run / "comparisons.parquet")
    assert comparisons.num_rows == 2 * 3
    assert str(run) in capsys.readouterr().out


def test_the_ladder_refuses_the_held_back_span(tmp_path: Path) -> None:
    held = plan_file(tmp_path, text=PLAN.replace("end = 2022-12-31", "end = 2024-01-01"))
    with pytest.raises(span.HeldBackError):
        ladder.load(held)
    values, grid = world(days=10)
    early = write_runs(tmp_path / "a", values, grid)
    assert main(["ladder", "--inputs", str(early), "--plan", str(held)]) == 1
    late = write_runs(tmp_path / "b", values, grid, end="2024-01-01")
    plan = plan_file(tmp_path)
    with pytest.raises(span.HeldBackError):
        ladder.run(late, tmp_path / "b", plan, "x")
    grid = dict(grid, ts_ns=grid["ts_ns"] + (labels.ns(date(2024, 1, 1)) - grid["ts_ns"][0]))
    shifted = write_runs(tmp_path / "c", values, grid)
    with pytest.raises(span.HeldBackError):
        ladder.run(shifted, tmp_path / "c", plan, "x")
    for folder in ("a", "b", "c"):
        assert not (tmp_path / folder / "panel-ladder").exists()


def test_the_docs_section_names_every_input_rung_and_command() -> None:
    text = (ROOT / "docs" / "market.md").read_text(encoding="utf-8")
    section = text[text.index("## Coin panel") :]
    end = section.find("\n## ", 3)
    section = section if end < 0 else section[:end]
    for name in inputs.OWN:
        assert f"`{name}`" in section, name
    for rung in ladder.load().rungs:
        assert f"`{rung.name}`" in section, rung.name
    for command in ("panel inputs", "panel ladder"):
        assert command in section
