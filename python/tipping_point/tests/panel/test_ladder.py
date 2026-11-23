# The ladder on planted panels: the plan and its refusals, quarters and their gap in both schemes,
# the common row set, the pair-count AUC and its block bootstrap against direct calculations, the
# solver, a planted signal and a null, external scores, the command and its refusals, and the docs.

import dataclasses
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
gap_days = 31
forward_first = 2

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
    assert (plan.c, plan.clip, plan.gap_days, plan.forward_first) == (0.01, 5.0, 31, 2)
    assert plan.gap_days * DAY >= ladder.needed_gap_ns(plan.horizons) > 30 * DAY
    assert (plan.block_days, plan.draws) == (14, 2000) and plan.external is None
    rungs = {r.name: set(r.columns) for r in plan.rungs}
    assert rungs["vol"] == set(inputs.GROUPS["vol"]) | {f"m_{n}" for n in inputs.GROUPS["vol"]}
    assert "oi_24h" in rungs["full"] and "m_oi_24h" not in rungs["full"]
    assert rungs["full+mlev"] - rungs["full"] == {f"m_{n}" for n in inputs.GROUPS["lev"]}
    assert "m_var_tau_24h" in rungs["classic"] and "ret_1h" not in rungs["classic"]
    assert set(plan.columns) == set(inputs.OWN + inputs.MARKET)


def test_each_sensitivity_plan_differs_from_the_plan_in_one_thing() -> None:
    plan = ladder.load()
    plans = {
        n: ladder.load(ladder.PLAN.with_name(f"ladder-{n}.toml")) for n in ("gap92", "block28")
    }
    assert plans["gap92"] == dataclasses.replace(plan, gap_days=92)
    assert plans["block28"] == dataclasses.replace(plan, block_days=28)
    public = ladder.load(ladder.PLAN.with_name("ladder-public.toml"))
    rungs = tuple(r for r in plan.rungs if "lev" not in r.groups)
    assert [r.name for r in rungs] == ["vol", "vol+price", "vol+price+flow", "classic"]
    comparisons = tuple(c for c in plan.comparisons if set(c) <= {r.name for r in rungs})
    assert public == dataclasses.replace(plan, rungs=rungs, comparisons=comparisons)
    assert not {"funding", "ls_rel", "m_ls_rel"} & set(public.columns)


def test_folds_are_calendar_quarters_and_training_keeps_the_gap_away() -> None:
    plan = ladder.load()
    edges = ladder.bounds(plan)
    starts = [date(y, m, 1) for y in (2022, 2023) for m in (1, 4, 7, 10)] + [date(2024, 1, 1)]
    assert edges.tolist() == [labels.ns(d) for d in starts]
    april, july = labels.ns(date(2022, 4, 1)), labels.ns(date(2022, 7, 1))
    gap = plan.gap_days * DAY
    ts = np.array(
        [
            *(april - HOUR, april, july - HOUR, april - gap - HOUR, april - gap),
            *(july + gap - HOUR, july + gap),
        ],
        np.int64,
    )
    fold, block = ladder.folds(ts, edges, 14)
    assert fold.tolist() == [0, 1, 1, 0, 0, 2, 2]
    assert block.tolist()[:3] == [6, 0, 6]
    train = ladder.training(ts, fold, 1, edges, plan.gap_days)
    assert train.tolist() == [False, False, False, True, False, False, True]
    forward = ladder.training(ts, fold, 1, edges, plan.gap_days, "forward")
    assert forward.tolist() == [False, False, False, True, False, False, False]
    assert ladder.blocks_per_fold(edges, 14).tolist() == [7] * 8


def quiet(_: str) -> None:
    pass


def year(plan: ladder.Plan, seed: int = 3) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    ts = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(365 * 24, dtype=np.int64)
    x = rng.normal(size=(len(ts), len(plan.columns)))
    lift = 1.5 * x[:, 0]
    y = {
        "long": rng.random(len(ts)) < 1 / (1 + np.exp(2.0 - lift)),
        "short": rng.random(len(ts)) < 1 / (1 + np.exp(2.0 + lift)),
    }
    fold, _ = ladder.folds(ts, ladder.bounds(plan), plan.block_days)
    return ts, x, y, fold


def test_a_cascade_after_a_quarter_reaches_no_model_of_that_quarter_through_a_28_day_input(
    tmp_path: Path,
) -> None:
    plan = ladder.load(plan_file(tmp_path))
    edges = ladder.bounds(plan)
    ts, x, y, fold = year(plan)
    raw = np.random.default_rng(9).normal(size=len(ts))
    window = 28 * 24 + 1

    def fed(r: np.ndarray) -> np.ndarray:
        out = x.copy()
        out[:, 1] = np.convolve(r, np.ones(window), "full")[: len(r)] / window
        return out

    after = (ts > edges[2]) & (ts <= edges[2] + 6 * HOUR)
    cascade = raw.copy()
    cascade[after] = 40.0
    hit = {
        side: v | ((ts >= edges[2] - 24 * HOUR) & (ts <= edges[2] + 6 * HOUR))
        for side, v in y.items()
    }
    quarter = fold == 1
    for gap_days, moves in ((plan.gap_days, False), (28, True)):
        shorter = dataclasses.replace(plan, gap_days=gap_days)
        base = ladder.out_of_fold(fed(raw), y, ts, fold, edges, shorter, "loqo", quiet)
        got = ladder.out_of_fold(fed(cascade), hit, ts, fold, edges, shorter, "loqo", quiet)
        for rung in plan.names:
            for side in ("long", "short"):
                same = np.array_equal(got[rung][side][quarter], base[rung][side][quarter])
                assert same != moves, (gap_days, rung, side)


def test_the_forward_scheme_fits_each_quarter_on_earlier_rows_only(tmp_path: Path) -> None:
    plan = ladder.load(plan_file(tmp_path))
    edges = ladder.bounds(plan)
    ts, x, y, fold = year(plan)
    base = ladder.out_of_fold(x, y, ts, fold, edges, plan, "forward", quiet)
    scored = fold >= plan.forward_first
    assert plan.forward_first == 2 and len(base["vol"]["long"]) == int(scored.sum())
    rng = np.random.default_rng(5)
    for f in range(plan.forward_first, len(edges) - 1):
        late = ts >= edges[f] - plan.gap_days * DAY
        moved = x.copy()
        planted = late & (fold != f)
        moved[planted] = 50.0 * np.sign(rng.normal(size=(int(planted.sum()), x.shape[1])))
        flipped = {side: np.where(late, ~v, v) for side, v in y.items()}
        got = ladder.out_of_fold(moved, flipped, ts, fold, edges, plan, "forward", quiet)
        mine = fold[scored] == f
        for rung in plan.names:
            for side in ("long", "short"):
                assert np.array_equal(got[rung][side][mine], base[rung][side][mine]), (f, rung)
    alone = dataclasses.replace(plan, gap_days=200)
    with pytest.raises(ValueError, match="no training rows"):
        ladder.out_of_fold(x, y, ts, fold, edges, alone, "forward", quiet)
    broken = x.copy()
    broken[np.flatnonzero(fold == 3)[0], 0] = np.nan
    with pytest.raises(RuntimeError, match="no finite score"):
        ladder.out_of_fold(broken, y, ts, fold, edges, plan, "forward", quiet)


def test_the_plan_refuses_invalid_values(tmp_path: Path) -> None:
    cases = [
        ("c = 0.01", "c = 0.0", "c = 0.0"),
        ("clip = 5.0", "clip = -1.0", "clip"),
        ("gap_days = 31", "gap_days = -1", "negative"),
        ("gap_days = 31", "gap_days = 30", "shorter than the 30.07 days"),
        ("block_days = 14", "block_days = 0", "block_days"),
        ("draws = 300", "draws = 0", "draws"),
        ('groups = ["vol", "price"]', 'groups = ["vol", "vol"]', "listed twice"),
        ("forward_first = 2", "forward_first = 0", "forward_first"),
        ("forward_first = 2", "forward_first = 4", "forward_first"),
    ]
    for old, new, message in cases:
        with pytest.raises(ValueError, match=message):
            ladder.load(plan_file(tmp_path, text=PLAN.replace(old, new)))
    with pytest.raises(ValueError, match="empty rung list"):
        ladder.load(plan_file(tmp_path, text=PLAN[: PLAN.index("[[rungs]]")]))
    extra = '\n[external]\nname = "e"\npath = "e.parquet"\nscheme = "both"\n'
    with pytest.raises(ValueError, match="external scheme"):
        ladder.load(plan_file(tmp_path, extra))


def test_the_left_out_quarter_is_fitted_without_its_rows_or_its_gap(tmp_path: Path) -> None:
    plan = ladder.load(plan_file(tmp_path))
    edges = ladder.bounds(plan)
    ts, x, y, fold = year(plan)
    rng = np.random.default_rng(3)
    base = ladder.out_of_fold(x, y, ts, fold, edges, plan, "loqo", quiet)
    gap = plan.gap_days * DAY
    left_out = fold == 1
    near = (ts >= edges[1] - gap) & (ts < edges[2] + gap) & ~left_out
    planted = near | (left_out & (np.arange(len(ts)) % 2 == 0))
    moved = x.copy()
    moved[planted] = 50.0 * np.sign(rng.normal(size=(int(planted.sum()), x.shape[1])))
    flipped = {side: np.where(near | left_out, ~v, v) for side, v in y.items()}
    got = ladder.out_of_fold(moved, flipped, ts, fold, edges, plan, "loqo", quiet)
    kept = left_out & ~planted
    assert near.sum() > 0 and kept.sum() > 1000
    for rung in plan.names:
        for side in ("long", "short"):
            assert np.array_equal(got[rung][side][kept], base[rung][side][kept]), (rung, side)


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


def test_the_bootstrap_resamples_whole_blocks_in_every_quarter(tmp_path: Path) -> None:
    text = PLAN.replace("end = 2022-12-31", "end = 2022-06-30").replace(
        "draws = 300", "draws = 200"
    )
    plan = ladder.load(
        plan_file(tmp_path, text=text.replace("forward_first = 2", "forward_first = 1"))
    )
    edges = ladder.bounds(plan)
    rng = np.random.default_rng(6)
    ts = labels.ns(date(2022, 1, 1)) + HOUR * np.arange(181 * 24, dtype=np.int64)
    fold, block = ladder.folds(ts, edges, plan.block_days)
    symbol = rng.choice(np.array(COINS, dtype=object), len(ts))
    rare = (fold == 1) & (block == 3)
    long = np.where(fold == 0, rng.random(len(ts)) < 0.05, rare & (rng.random(len(ts)) < 0.2))
    short = rng.random(len(ts)) < 0.05
    y = {"long": long, "short": short, "any": long | short}
    scores = {
        name: {side: rng.normal(size=len(ts)) + lift * y[side] for side in ladder.SIDES}
        for name, lift in (("vol", 0.3), ("vol+price", 0.8), ("classic", 0.0))
    }
    h = ladder.Horizon("loqo", 0, ts, symbol, y, fold, block, scores)
    _, _, comparisons = ladder.evaluate(h, plan, 24)
    weights = ladder.draws(
        ladder.blocks_per_fold(edges, plan.block_days), plan.draws, plan.seed, 24
    )
    assert not np.array_equal(weights[0], weights[1])
    (got,) = [c for c in comparisons if c["first"] == "vol+price" and c["side"] == "long"]
    means = {}
    dropped = np.zeros(plan.draws, bool)
    for name in ("vol+price", "vol"):
        s = scores[name]["long"]
        means[name] = np.empty(plan.draws)
        for d in range(plan.draws):
            per_quarter = []
            for f in range(2):
                drawn = np.repeat(np.arange(len(weights[f][d])), weights[f][d].astype(int))
                rows = np.concatenate([np.flatnonzero((fold == f) & (block == b)) for b in drawn])
                a, z = s[rows], long[rows]
                if z.any() and not z.all():
                    per_quarter.append(auc(a[z], a[~z]))
                else:
                    dropped[d] = True
            means[name][d] = np.mean(per_quarter)
    difference = means["vol+price"] - means["vol"]
    assert 0 < dropped.mean() < 1
    assert got["complete_draws"] == pytest.approx(1 - dropped.mean())
    assert got["draws"] == plan.draws
    assert got["se"] == pytest.approx(np.std(difference, ddof=1))
    assert [got["q025"], got["q05"], got["q975"]] == pytest.approx(
        np.quantile(difference, ladder.QUANTILES).tolist()
    )
    assert got["above_zero"] == pytest.approx(np.mean(difference > 0))
    per_fold = np.array(got["fold_differences"])
    assert got["se_quarters"] == pytest.approx(np.std(per_fold, ddof=1) / np.sqrt(2))


def test_the_solver_on_rare_positives_a_constant_column_and_no_positives() -> None:
    rng = np.random.default_rng(2)
    n, c = 50_000, 0.01
    x = rng.normal(size=(n, 3))
    x[:, 2] = 4.0
    y = np.zeros(n)
    y[rng.choice(n, 10, replace=False)] = 1.0
    z, _ = ladder.standardise(x, x, 5.0)
    assert (z[:, 2] == 0).all()
    beta = ladder.fit(z, y, c)
    p = ladder.predict(z, beta)
    assert np.isfinite(beta).all() and abs(beta[2]) < 1e-12
    design = np.hstack([z, np.ones((n, 1))])
    ridge = np.array([1 / c, 1 / c, 1 / c, 0.0])
    assert np.abs(design.T @ (p - y) + ridge * beta).max() < 1e-4
    assert p.sum() == pytest.approx(y.sum(), rel=1e-5)
    none = ladder.predict(z, ladder.fit(z, np.zeros(n), c))
    assert np.isfinite(none).all() and (none == 0).all()


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


def scored(out: dict, name: str, side: str, coins: str = "all", scheme: str = "loqo") -> dict:
    (row,) = [
        s
        for s in out["scores"]
        if s["name"] == name and s["side"] == side and s["coins"] == coins and s["scheme"] == scheme
    ]
    return row


def compared(out: dict, first: str, side: str, scheme: str = "loqo") -> dict:
    (row,) = [
        c
        for c in out["comparisons"]
        if c["first"] == first and c["side"] == side and c["scheme"] == scheme
    ]
    return row


def test_a_planted_signal_one_rung_sees_and_a_null(tmp_path: Path) -> None:
    values, grid = world()
    itself = '\n[[comparisons]]\nfirst = "vol"\nsecond = "vol"\n'
    out = ladder.ladder(values, grid, ladder.load(plan_file(tmp_path, itself)))
    for scheme in ladder.SCHEMES:
        for side in ladder.SIDES:
            zero = compared(out, "vol", side, scheme)
            for key in ("estimate", "se", "se_quarters", "q025", "q05", "q975"):
                assert zero[key] == 0.0, (scheme, side, key)
    forward = compared(out, "vol+price", "long", "forward")
    assert forward["folds"] == 2 and forward["q025"] > 0
    assert {r["fold"] for r in out["folds"] if r["scheme"] == "forward"} == {2, 3}
    weighted = scored(out, "vol+price", "long")
    quarters = [
        r
        for r in out["folds"]
        if r["scheme"] == "loqo"
        and r["name"] == "vol+price"
        and r["side"] == "long"
        and r["coins"] == "all"
    ]
    assert weighted["fold_mean_weighted"] == pytest.approx(
        sum(r["auc"] * r["positives"] for r in quarters) / sum(r["positives"] for r in quarters)
    )
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
    oof, later = out["oof"]
    assert set(later["scheme"].to_pylist()) == {"forward"}
    assert later["fold"].to_numpy().min() == 2 and oof["fold"].to_numpy().min() == 0
    others = np.array(oof["symbol"].to_pylist()) != "AAAUSDT"
    assert scored(out, "vol+price", "long", "without")["rows"] == int(others.sum())
    assert oof.num_rows == out["accounting"][24][-1]["rows"]
    p = {s: oof[f"vol+price:{s}"].to_numpy() for s in ("long", "short", "any")}
    assert np.allclose(p["any"], 1 - (1 - p["long"]) * (1 - p["short"]))
    assert seen["complete_draws"] == compared(out, "vol+price", "short")["complete_draws"] == 1.0
    q4 = np.flatnonzero(grid["short_24"] & (grid["ts_ns"] >= labels.ns(date(2022, 10, 1))))
    rare = grid["short_24"].copy()
    rare[q4[2:]] = False
    out = ladder.ladder(values, dict(grid, short_24=rare), ladder.load(plan_file(tmp_path)))
    complete = compared(out, "vol+price", "short")["complete_draws"]
    assert 0.0 < complete < 1.0
    assert compared(out, "classic", "short")["complete_draws"] == complete
    assert compared(out, "vol+price", "long")["complete_draws"] == 1.0


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
    extra = '\n[external]\nname = "planted"\npath = "planted.parquet"\nscheme = "loqo"\n'
    extra += '\n[[comparisons]]\nfirst = "planted"\nsecond = "vol"\n'
    plan = ladder.load(plan_file(tmp_path, extra))
    assert plan.external == ("planted", (tmp_path / "planted.parquet").resolve(), "loqo")
    table = external_table(values, grid, ["long", "short", "any"])
    out = ladder.ladder(values, grid, plan, table)
    assert {c["scheme"] for c in out["comparisons"] if c["first"] == "planted"} == {"loqo"}
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


def test_external_scores_join_on_coin_and_hour_whatever_the_table_looks_like() -> None:
    values, grid = world(days=20)
    ts, symbol = grid["ts_ns"], grid["symbol"]
    table = external_table(values, grid, list(ladder.SIDES))
    base = ladder.external_scores(table, ts, symbol, 24, ladder.SIDES)
    shuffled = table.take(np.random.default_rng(1).permutation(table.num_rows))
    stamped = table.set_column(1, "hour", table["hour"].cast(pa.timestamp("ns")))
    wide = table.set_column(0, "coin", table["coin"].cast(pa.large_string()))
    wide = wide.set_column(3, "side", wide["side"].cast(pa.large_string()))
    for other in (shuffled, stamped, wide):
        got = ladder.external_scores(other, ts, symbol, 24, ladder.SIDES)
        for side in ladder.SIDES:
            assert np.array_equal(got[side], base[side]), side
    twice = pa.concat_tables([table, table.slice(7, 1)])
    with pytest.raises(ValueError, match="twice"):
        ladder.external_scores(twice, ts, symbol, 24, ladder.SIDES)
    for bad in (np.inf, -np.inf):
        score = table["score"].to_numpy().copy()
        score[3] = bad
        with pytest.raises(ValueError, match="non-finite"):
            ladder.external_scores(
                table.set_column(4, "score", pa.array(score)), ts, symbol, 24, ladder.SIDES
            )


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
    assert "### forward" in report
    comparisons = pq.read_table(run / "comparisons.parquet")
    assert comparisons.num_rows == 2 * 3 * 2
    assert set(pq.read_table(run / "oof.parquet")["scheme"].to_pylist()) == set(ladder.SCHEMES)
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
    hours = np.array([labels.ns(date(2022, 1, 1)), labels.ns(date(2024, 1, 1))], np.int64)
    pq.write_table(pa.table({"hour": hours}), tmp_path / "late.parquet")
    stamps = pa.array(hours, pa.int64()).cast(pa.timestamp("ns"))
    pq.write_table(pa.table({"hour": stamps}), tmp_path / "stamped.parquet")
    for name in ("late", "stamped"):
        extra = f'\n[external]\nname = "{name}"\npath = "{name}.parquet"\nscheme = "loqo"\n'
        with pytest.raises(span.HeldBackError):
            ladder.run(early, tmp_path / "a", plan_file(tmp_path, extra), "x")
    complete = json.loads((early / "complete.json").read_text(encoding="utf-8"))
    (early / "complete.json").write_text(json.dumps(complete | {"clean": False}), "utf-8")
    with pytest.raises(ValueError, match="clean"):
        ladder.run(early, tmp_path / "a", plan, "x")
    complete["files"]["inputs.parquet"] = "0" * 64
    (early / "complete.json").write_text(json.dumps(complete), "utf-8")
    with pytest.raises(ValueError, match="sha256"):
        ladder.run(early, tmp_path / "a", plan, "x")
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
