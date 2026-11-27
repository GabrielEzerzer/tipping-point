# The held-back scoring core on synthetic panels: cut plans that reproduce the development forward
# scores, no label or input after the cut reaching a fit, the refusals, no choosing function on the
# path, the decision with its support settled first, and the test span's quarters and blocks.

import hashlib
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from tipping_point.panel import classical, heldback, inputs, labels, ladder, sequence, span

HOUR = labels.HOUR_NS
COINS = ["AAAUSDT", "BBBUSDT"]
END = date(2022, 12, 31)
CUTS = {2: date(2022, 7, 1), 3: date(2022, 10, 1)}
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
draws = 50
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
variants = ["vol"]
rungs = ["full", "vol"]
models = ["jepa", "supervised"]
schemes = ["forward"]
seeds = [3]
folds = [2, 3]
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

PAIRS = [("jepa", "window"), ("jepa", "full"), ("full", "vol")]
PRIMARY = ("jepa", "window")


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
        lift = values["vol_1h"]
        grid[f"long_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 - lift))
        grid[f"short_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 + lift))
        grid[f"censored_{h}"] = rng.random(n) < 0.01
    grid["skip"] = rng.random(n) < 0.02
    return values, grid


@pytest.fixture(scope="module")
def dev(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("heldback")
    (root / "ladder.toml").write_text(LADDER, encoding="utf-8")
    (root / "sequence.toml").write_text(PLAN, encoding="utf-8")
    plan = sequence.load(root / "sequence.toml")
    values, grid = world()
    encoders = sequence.sequence(values, grid, plan)["encoders"]
    return {"plan": plan, "values": values, "grid": grid, "encoders": encoders, "root": root}


def arms(p: dict[str, Any], f: int) -> list[heldback.Arm]:
    final = heldback.final_fold(p["plan"])
    saved = p["encoders"]

    def name(fold: int, model: str) -> str:
        return sequence.encoder_name("vol", "forward", fold, 3, model)

    record = {
        name(final, m): hashlib.sha256(saved[name(f, m)]).hexdigest()
        for m in ("jepa", "supervised")
    }

    def read(key: str) -> bytes:
        return saved[key.replace(f"-fold{final}-", f"-fold{f}-")]

    return [
        heldback.SequenceArm("jepa", "jepa", "vol", "means", 0.01, (3,), record, read),
        heldback.SequenceArm("supervised", "supervised", "vol", None, None, (3,), record, read),
        heldback.ClassicalArm("window", Config("window", "vol", "means", 1.0)),
        heldback.RungArm("full", "full"),
        heldback.RungArm("vol", "vol"),
    ]


def development(p: dict[str, Any]) -> dict[str, pa.Table]:
    values, grid, plan = p["values"], p["grid"], p["plan"]
    fetch = p["encoders"].__getitem__
    out = {
        "jepa": sequence.external(values, grid, plan, ("jepa", "vol", "means", 0.01), fetch),
        "supervised": sequence.external(
            values, grid, plan, ("supervised", "vol", None, None), fetch
        ),
        "window": classical.export(values, grid, plan, Config("window", "vol", "means", 1.0)),
    }
    tables = {name: t["forward"] for name, t in out.items()}
    parts: dict[str, list[pa.Table]] = {"full": [], "vol": []}
    for h in plan.ladder.horizons:
        _, schemes = ladder.horizon_rows(values, grid, plan.ladder, h, None, lambda line: None)
        (fwd,) = [s for s in schemes if s.scheme == "forward"]
        for name in parts:
            for side in plan.ladder.sides:
                n = len(fwd.ts)
                parts[name].append(
                    pa.table(
                        {
                            "coin": pa.array(fwd.symbol.astype(str)),
                            "hour": pa.array(fwd.ts, pa.int64()),
                            "horizon": pa.array(np.full(n, h), pa.int16()),
                            "side": pa.array(np.full(n, side)),
                            "score": pa.array(fwd.scores[name][side], pa.float64()),
                            "fold": pa.array(fwd.fold, pa.int16()),
                            "scheme": pa.array(np.full(n, "forward")),
                        }
                    )
                )
    return tables | {name: pa.concat_tables(t) for name, t in parts.items()}


def ordered(table: pa.Table) -> pa.Table:
    keys = [(k, "ascending") for k in ("horizon", "side", "coin", "hour")]
    return table.select(["coin", "hour", "horizon", "side", "score"]).sort_by(keys)


@pytest.mark.parametrize(("f", "end"), [(2, date(2022, 9, 30)), (3, END)])
def test_a_cut_at_a_development_quarter_reproduces_its_forward_scores(
    dev: dict[str, Any], f: int, end: date
) -> None:
    tables, used = heldback.scored(
        dev["values"], dev["grid"], dev["plan"], end, arms(dev, f), CUTS[f]
    )
    reference = development(dev)
    final = heldback.final_fold(dev["plan"])
    assert set(used) == {"jepa", "supervised"}
    for model, files in used.items():
        assert list(files) == [sequence.encoder_name("vol", "forward", final, 3, model)]
    for name, table in tables.items():
        assert set(table["fold"].to_pylist()) == {1}, name
        assert set(table["scheme"].to_pylist()) == {"forward"}, name
        ref = reference[name]
        ref = ref.filter(pa.array(ref["fold"].to_numpy() == f))
        got, want = ordered(table), ordered(ref)
        assert got.num_rows == want.num_rows > 0, name
        assert got.equals(want), name


def scrambled(grid: dict, cut: date) -> dict:
    rng = np.random.default_rng(5)
    after = grid["ts_ns"] >= labels.ns(cut)
    out = dict(grid)
    for h in labels.HORIZONS:
        for side in ("long", "short"):
            out[f"{side}_{h}"] = np.where(after, rng.random(after.size) < 0.5, grid[f"{side}_{h}"])
    return out


def test_no_label_at_or_after_the_cut_reaches_a_score(dev: dict[str, Any]) -> None:
    cut = CUTS[3]
    given = (dev["plan"], END, arms(dev, 3), cut)
    first, _ = heldback.scored(dev["values"], dev["grid"], *given)
    second, _ = heldback.scored(dev["values"], scrambled(dev["grid"], cut), *given)
    for name in first:
        assert first[name].equals(second[name]), name


def moved(values: dict, grid: dict, cut: date) -> dict:
    rng = np.random.default_rng(9)
    after = grid["ts_ns"] >= labels.ns(cut)
    return {
        name: np.where(after & np.isfinite(v), v + 10 * rng.normal(size=v.shape), v)
        for name, v in values.items()
    }


def test_no_input_at_or_after_the_cut_reaches_a_fit_or_a_scale(
    dev: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Any, ...]] = []
    fit, standardise, statistics = ladder.fit, ladder.standardise, sequence.statistics

    def fit_(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
        beta = fit(x, y, c)
        seen.append(("fit", beta.tobytes()))
        return beta

    def standardise_(train: np.ndarray, test: np.ndarray, clip: float) -> Any:
        a, b = standardise(train, test, clip)
        seen.append(("scale", train.mean(axis=0).tobytes(), train.std(axis=0).tobytes()))
        return a, b

    def statistics_(raw: np.ndarray, train: np.ndarray) -> Any:
        mu, sd = statistics(raw, train)
        seen.append(("statistics", mu.tobytes(), sd.tobytes()))
        return mu, sd

    monkeypatch.setattr(ladder, "fit", fit_)
    monkeypatch.setattr(ladder, "standardise", standardise_)
    monkeypatch.setattr(sequence, "statistics", statistics_)
    cut = CUTS[3]
    given = (dev["plan"], END, arms(dev, 3), cut)
    first_scores, _ = heldback.scored(dev["values"], dev["grid"], *given)
    first = list(seen)
    seen.clear()
    second_scores, _ = heldback.scored(moved(dev["values"], dev["grid"], cut), dev["grid"], *given)
    assert {k[0] for k in first} == {"fit", "scale", "statistics"}
    assert first == seen
    assert all(not first_scores[n].equals(second_scores[n]) for n in first_scores)


def test_the_fit_plan_refuses_loqo_other_folds_and_a_file_naming_cuts(
    dev: dict[str, Any], tmp_path: Path
) -> None:
    fit = heldback.fit_sequence(dev["plan"], END, CUTS[3])
    heldback.fit_only(fit)
    assert ladder.bounds(fit.ladder).tolist() == [
        labels.ns(d) for d in (date(2022, 1, 1), CUTS[3], date(2023, 1, 1))
    ]
    with pytest.raises(ValueError, match="fits only forward"):
        fit.ladder.first_fold("loqo")
    loqo = replace(fit, schemes=("loqo",))
    with pytest.raises(ValueError, match="fits only forward"):
        sequence.external(
            dev["values"], dev["grid"], loqo, ("supervised", "vol", None, None), lambda k: b""
        )
    with pytest.raises(ValueError, match="fits only forward"):
        classical.export(dev["values"], dev["grid"], loqo, Config("rung", "vol", None, 0.01))
    for bad in (
        loqo,
        replace(fit, schemes=("loqo", "forward")),
        replace(fit, folds=(0,)),
        replace(fit, folds=(0, 1)),
        replace(fit, folds=(1, 2)),
        replace(fit, ladder=replace(fit.ladder, forward_first=0)),
        dev["plan"],
    ):
        with pytest.raises(ValueError, match="a fit plan"):
            heldback.fit_only(bad)
    with pytest.raises(ValueError, match="cuts"):
        replace(fit.ladder, cuts=(date(2022, 1, 1), date(2023, 1, 1)))
    with pytest.raises(ValueError, match="cuts"):
        replace(fit.ladder, end=date(2022, 11, 30))
    for text in (
        "cuts = [2022-01-01, 2022-10-01, 2023-01-01]\n" + LADDER,
        LADDER.replace("forward_first = 2", "forward_first = 2\ncuts = [2022-01-01, 2023-01-01]"),
    ):
        path = tmp_path / "ladder.toml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="cuts"):
            ladder.load(path)
    with pytest.raises(ValueError, match="cut"):
        heldback.fit_plan(dev["plan"].ladder, END, date(2022, 1, 1))
    with pytest.raises(ValueError, match="already cut"):
        heldback.fit_plan(fit.ladder, END, CUTS[3])


def test_a_model_whose_sha256_differs_from_the_record_is_refused(dev: dict[str, Any]) -> None:
    arm = arms(dev, 3)[0]
    assert isinstance(arm, heldback.SequenceArm)
    (key,) = list(arm.encoders)[:1]
    given = (dev["values"], dev["grid"], dev["plan"], END)
    changed = replace(arm, encoders={**arm.encoders, key: "0" * 64})
    with pytest.raises(ValueError, match="sha256 differs"):
        heldback.scored(*given, [changed], CUTS[3])
    unrecorded = replace(arm, encoders={k: v for k, v in arm.encoders.items() if k != key})
    with pytest.raises(ValueError, match="records no"):
        heldback.scored(*given, [unrecorded], CUTS[3])
    other = replace(arm, read=lambda k: b"not the model")
    with pytest.raises(ValueError, match="sha256 differs"):
        heldback.scored(*given, [other], CUTS[3])


def rule(**change: Any) -> heldback.Rule:
    base: dict[str, Any] = {
        "bound": "q05",
        "draws": "all",
        "min_quarters": 2,
        "min_complete": 0.0,
        "min_blocks": 1,
        "min_supported": 1,
    }
    return heldback.Rule(**(base | change))


def test_no_choosing_function_is_reached_and_support_comes_first(
    dev: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"{name} was called")

        return call

    for module, name in (
        (sequence, "choose"),
        (sequence, "ranked"),
        (sequence, "best"),
        (sequence, "scores_run"),
        (classical, "run"),
        (classical, "choose"),
    ):
        monkeypatch.setattr(module, name, refused(f"{module.__name__}.{name}"))
    given = (dev["values"], dev["grid"], dev["plan"], END)
    out = heldback.run(*given, arms(dev, 2), PAIRS, PRIMARY, "long+short", 24, rule(), CUTS[2])
    decision = out["decision"]
    assert decision["outcome"] in ("supported", "not supported")
    assert decision["read"] == "q05" and decision["row"]["q05"] == decision["lower"]
    (row,) = [c for c in out["comparisons"] if c["primary"]]
    assert row["complete_draws"] == out["support"]["complete_draws"]
    assert row["folds"] == out["support"]["scoreable"] == 2
    other = [
        *arms(dev, 2)[:2],
        heldback.ClassicalArm("window", Config("window", "vol", "last", 0.01)),
    ]
    pairs = [("jepa", "window")]
    again = heldback.run(
        *given, [*other, *arms(dev, 2)[3:]], pairs, PRIMARY, "long+short", 24, rule(), CUTS[2]
    )
    (changed,) = [c for c in again["comparisons"] if c["primary"]]
    assert changed["estimate"] != row["estimate"]
    assert again["support"] == out["support"]
    monkeypatch.setattr(heldback, "scored", refused("heldback.scored"))
    early = heldback.run(
        *given, arms(dev, 2), PAIRS, PRIMARY, "long+short", 24, rule(min_quarters=3), CUTS[2]
    )
    assert early["decision"]["outcome"] == "inconclusive"
    assert set(early) == {"support", "decision"}


def placed(grid: dict, h: int = 24) -> dict:
    rng = np.random.default_rng(3)
    ts = grid["ts_ns"]
    out = dict(grid)
    out["skip"] = np.zeros(ts.size, bool)
    out[f"censored_{h}"] = np.zeros(ts.size, bool)
    q4 = labels.ns(date(2022, 10, 1))
    first_block = (ts >= q4) & (ts < q4 + 14 * 24 * HOUR)
    spread = rng.random(ts.size) < 0.05
    out[f"long_{h}"] = spread & ((ts < q4) | first_block)
    out[f"short_{h}"] = spread
    return out


def test_the_support_counts_quarters_blocks_and_complete_draws_from_labels(
    dev: dict[str, Any],
) -> None:
    evaluation = heldback.evaluation_plan(dev["plan"].ladder, END, CUTS[2])
    grid = placed(dev["grid"])
    found = heldback.support(dev["values"], grid, evaluation, "long", 24, rule())
    q = found["quarters"]
    assert [x["quarter"] for x in q] == ["2022-07-01", "2022-10-01"]
    assert [x["blocks"] for x in q] == [7, 7]
    assert [x["long_blocks_with_positive"] for x in q] == [7, 1]
    assert found["scoreable"] == 2 and found["supported"] == 2 and found["causes"] == []
    lost = (6 / 7) ** 7
    assert 0 < found["complete_draws"] < 1
    assert abs((1 - found["complete_draws"]) - lost) < 0.2
    both = heldback.support(dev["values"], grid, evaluation, "long+short", 24, rule())
    assert [x["short_blocks_with_positive"] for x in both["quarters"]] == [7, 7]
    assert both["complete_draws"] == found["complete_draws"]
    short = heldback.support(dev["values"], grid, evaluation, "short", 24, rule())
    assert short["complete_draws"] == 1.0


@pytest.mark.parametrize(
    ("change", "cause"),
    [
        ({"min_quarters": 3}, "scoreable quarters"),
        ({"min_complete": 1.0}, "complete draws"),
        ({"min_blocks": 2, "min_supported": 2}, "blocks"),
    ],
)
@pytest.mark.parametrize("side", ["long", "long+short"])
def test_each_cause_alone_makes_the_outcome_inconclusive(
    dev: dict[str, Any], change: dict[str, Any], cause: str, side: str
) -> None:
    evaluation = heldback.evaluation_plan(dev["plan"].ladder, END, CUTS[2])
    grid = placed(dev["grid"])
    strict = rule(min_supported=2, **{k: v for k, v in change.items() if k != "min_supported"})
    found = heldback.support(dev["values"], grid, evaluation, side, 24, strict)
    assert len(found["causes"]) == 1 and cause in found["causes"][0]
    decision = heldback.decide(found, [{"primary": True}], PRIMARY, strict)
    assert decision["outcome"] == "inconclusive"
    assert decision["causes"] == found["causes"] and decision["support"] == found


def primary_row(found: dict[str, Any], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {k: 0.0 for k in heldback.ROW}
    return row | {
        "first": PRIMARY[0],
        "second": PRIMARY[1],
        "side": found["side"],
        "horizon": found["horizon"],
        "scheme": "forward",
        "primary": True,
        "complete_draws": found["complete_draws"],
        **values,
    }


def test_the_decision_reads_the_rules_bound_on_sufficient_support(dev: dict[str, Any]) -> None:
    evaluation = heldback.evaluation_plan(dev["plan"].ladder, END, CUTS[2])
    grid = placed(dev["grid"])
    bounds = {"q05": 0.01, "q025": -0.01, "q05_complete": -0.02, "q025_complete": 0.03}
    expected = {
        ("q05", "all"): "supported",
        ("q025", "all"): "not supported",
        ("q05", "complete"): "not supported",
        ("q025", "complete"): "supported",
    }
    for (bound, draws), outcome in expected.items():
        r = rule(bound=bound, draws=draws)
        found = heldback.support(dev["values"], grid, evaluation, "long", 24, r)
        row = primary_row(found, **bounds)
        decision = heldback.decide(found, [row, {**row, "primary": False}], PRIMARY, r)
        key = bound + ("_complete" if draws == "complete" else "")
        assert decision["outcome"] == outcome
        assert (decision["read"], decision["lower"]) == (key, bounds[key])
        assert decision["support"] == found
        for value in (-1.0, 1.0):
            other = heldback.decide(found, [primary_row(found, **{key: value})], PRIMARY, r)
            assert other["support"] == found
        zero = heldback.decide(found, [primary_row(found, **{key: 0.0})], PRIMARY, r)
        assert zero["outcome"] == "not supported"
    r = rule()
    found = heldback.support(dev["values"], grid, evaluation, "long", 24, r)
    with pytest.raises(ValueError, match="another rule"):
        heldback.decide(found, [primary_row(found)], PRIMARY, rule(min_blocks=2))
    with pytest.raises(ValueError, match="not the support"):
        heldback.decide(found, [primary_row(found, side="short")], PRIMARY, r)
    with pytest.raises(RuntimeError, match="complete draws"):
        heldback.decide(found, [primary_row(found, complete_draws=0.5)], PRIMARY, r)
    with pytest.raises(ValueError, match="not a number"):
        heldback.decide(found, [primary_row(found, q05=float("nan"))], PRIMARY, r)
    for bad in ({"bound": "q10"}, {"draws": "some"}, {"min_quarters": -1}, {"min_complete": 2.0}):
        with pytest.raises(ValueError, match="the decision rule"):
            rule(**bad)


def test_the_test_span_has_eleven_quarters_of_the_blocks_the_bootstrap_expects() -> None:
    dev = ladder.load()
    end = date(2026, 9, 30)
    evaluation = heldback.evaluation_plan(dev, end)
    edges = ladder.bounds(evaluation)
    starts = [date(y, m, 1) for y in (2024, 2025, 2026) for m in (1, 4, 7, 10)]
    assert edges.tolist() == [labels.ns(d) for d in starts]
    assert ladder.blocks_per_fold(edges, evaluation.block_days).tolist() == [7] * 11
    assert evaluation.first_fold("forward") == 0 and evaluation.cuts is None
    days = np.diff(edges) // (24 * HOUR)
    last = days - (ladder.blocks_per_fold(edges, evaluation.block_days) - 1) * 14
    assert set(last.tolist()) <= {6, 7, 8}
    fit = heldback.fit_plan(dev, end)
    cut = labels.ns(span.HELD_BACK)
    assert ladder.bounds(fit).tolist() == [labels.ns(dev.start), cut, labels.ns(date(2026, 10, 1))]
    ts = labels.ns(dev.start) + HOUR * np.arange(0, 5 * 365 * 24, 7, dtype=np.int64)
    fold, _ = ladder.folds(ts, ladder.bounds(fit), fit.block_days)
    train = ladder.training(ts, fold, 1, ladder.bounds(fit), fit.gap_days, "forward")
    assert np.array_equal(train, ts < cut - fit.gap_days * 24 * HOUR)
    assert np.array_equal(fold == 1, (ts >= cut) & (ts < labels.ns(date(2026, 10, 1))))
    text = ladder.PLAN.read_text(encoding="utf-8")
    later = text.replace("start = 2022-01-01", "start = 2024-01-01").replace(
        "end = 2023-12-31", "end = 2026-09-30"
    )
    with pytest.raises(span.HeldBackError):
        ladder.load(ladder.PLAN, later)
    with pytest.raises(ValueError, match="forward_first"):
        ladder.load(ladder.PLAN, text.replace("forward_first = 2", "forward_first = 0"))
