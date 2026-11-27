# The held-back scoring core on synthetic panels: cut plans that reproduce the development forward
# scores, no label or input after the cut reaching a fit, the refusals, no choosing function on the
# path, the decision with its support settled first, and the test span's quarters and blocks.

import hashlib
import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from tipping_point.panel import classical, heldback, inputs, labels, ladder, sequence, span

HOUR = labels.HOUR_NS
DAY = 24 * HOUR
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
models = ["jepa", "twin", "supervised"]
schemes = ["forward"]
seeds = [3, 4]
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
def plan(tmp_path_factory: pytest.TempPathFactory) -> sequence.Plan:
    root = tmp_path_factory.mktemp("heldback")
    (root / "ladder.toml").write_text(LADDER, encoding="utf-8")
    (root / "sequence.toml").write_text(PLAN, encoding="utf-8")
    return sequence.load(root / "sequence.toml")


@pytest.fixture(scope="module")
def panel() -> tuple[dict, dict]:
    return world()


@pytest.fixture(scope="module")
def dev(plan: sequence.Plan, panel: tuple[dict, dict]) -> dict[str, Any]:
    values, grid = panel
    encoders = sequence.sequence(values, grid, plan)["encoders"]
    return {"plan": plan, "values": values, "grid": grid, "encoders": encoders}


def ending(plan: sequence.Plan, cut: date) -> sequence.Plan:
    return replace(plan, ladder=replace(plan.ladder, end=cut - timedelta(days=1)))


def evaluation(plan: sequence.Plan, cut: date, end: date = END, **change: Any) -> ladder.Plan:
    return heldback.evaluation_plan(replace(ending(plan, cut).ladder, **change), end)


def pinned(record: dict[str, str], read: Any) -> list[heldback.Arm]:
    def arm(name: str, model: str, readout: str | None, c: float | None, seeds: tuple) -> Any:
        return heldback.SequenceArm(name, model, "vol", readout, c, seeds, record, read)

    return [
        arm("jepa", "jepa", "means", 0.01, (3, 4)),
        arm("jepa3", "jepa", "means", 0.01, (3,)),
        arm("twin", "twin", "last", 1.0, (3, 4)),
        arm("supervised", "supervised", None, None, (3, 4)),
        heldback.ClassicalArm("window", Config("window", "vol", "means", 1.0)),
        heldback.RungArm("full", "full"),
        heldback.RungArm("vol", "vol"),
    ]


def arms(p: dict[str, Any]) -> list[heldback.Arm]:
    saved = p["encoders"]
    return pinned({k: hashlib.sha256(v).hexdigest() for k, v in saved.items()}, saved.__getitem__)


def unread(key: str) -> bytes:
    raise AssertionError(f"{key} was read")


def development(p: dict[str, Any]) -> dict[str, pa.Table]:
    values, grid, plan = p["values"], p["grid"], p["plan"]
    fetch = p["encoders"].__getitem__
    seq = {
        "jepa": (plan, ("jepa", "vol", "means", 0.01)),
        "jepa3": (replace(plan, seeds=(3,)), ("jepa", "vol", "means", 0.01)),
        "twin": (plan, ("twin", "vol", "last", 1.0)),
        "supervised": (plan, ("supervised", "vol", None, None)),
    }
    out = {name: sequence.external(values, grid, q, c, fetch) for name, (q, c) in seq.items()}
    out["window"] = classical.export(values, grid, plan, Config("window", "vol", "means", 1.0))
    tables = {name: t["forward"] for name, t in out.items()}
    parts: dict[str, list[pa.Table]] = {"full": [], "vol": []}
    for h in plan.ladder.horizons:
        _, schemes = ladder.horizon_rows(values, grid, plan.ladder, h, None, lambda line: None)
        (fwd,) = [s for s in schemes if s.scheme == "forward"]
        everything = np.ones(len(fwd.ts), bool)
        for name in parts:
            for side in plan.ladder.sides:
                parts[name].append(
                    sequence.scores_part(
                        fwd.symbol.astype(str),
                        fwd.ts,
                        fwd.fold,
                        everything,
                        h,
                        side,
                        fwd.scores[name],
                        "forward",
                    )
                )
    return tables | {name: pa.concat_tables(t) for name, t in parts.items()}


def ordered(table: pa.Table) -> pa.Table:
    keys = [(k, "ascending") for k in ("horizon", "side", "coin", "hour")]
    return table.select(["coin", "hour", "horizon", "side", "score"]).sort_by(keys)


@pytest.mark.parametrize(("f", "end"), [(2, date(2022, 9, 30)), (3, END)])
def test_a_development_plan_ending_before_a_quarter_reproduces_its_forward_scores(
    dev: dict[str, Any], f: int, end: date
) -> None:
    before = ending(dev["plan"], CUTS[f])
    assert heldback.final_fold(before) == f
    tables, used = heldback.scored(dev["values"], dev["grid"], before, end, arms(dev))
    reference = development(dev)
    given = {a.name: a for a in arms(dev) if isinstance(a, heldback.SequenceArm)}
    assert set(used) == set(given)
    for name, files in used.items():
        a = given[name]
        expected = [sequence.encoder_name("vol", "forward", f, s, a.model) for s in a.seeds]
        assert sorted(files) == sorted(expected)
    for name, table in tables.items():
        assert set(table["fold"].to_pylist()) == {1}, name
        assert set(table["scheme"].to_pylist()) == {"forward"}, name
        ref = reference[name]
        ref = ref.filter(pa.array(ref["fold"].to_numpy() == f))
        got, want = ordered(table), ordered(ref)
        assert got.num_rows == want.num_rows > 0, name
        assert got.equals(want), name
    assert not ordered(tables["jepa"]).equals(ordered(tables["jepa3"]))


def test_the_cut_is_the_day_after_the_development_plans_last(plan: sequence.Plan) -> None:
    shipped = ladder.load()
    end = date(2026, 9, 30)
    for last in (shipped.end, date(2023, 9, 30), date(2023, 6, 30)):
        dev = replace(plan, ladder=replace(shipped, end=last))
        fit = heldback.fit_sequence(dev, end)
        cut = ladder.bounds(dev.ladder)[-1]
        assert ladder.bounds(fit.ladder)[1] == cut == labels.ns(last + timedelta(days=1))
        assert ladder.bounds(heldback.evaluation_plan(dev.ladder, end))[0] == cut
        assert heldback.final_fold(dev) == len(ladder.bounds(dev.ladder)) - 1
    assert heldback.final_fold(replace(plan, ladder=shipped)) == 8
    assert heldback.fit_plan(shipped, end).cuts == (
        shipped.start,
        span.HELD_BACK,
        date(2026, 10, 1),
    )
    with pytest.raises(ValueError, match="falls before"):
        heldback.fit_plan(shipped, shipped.end)
    with pytest.raises(ValueError, match="falls before"):
        heldback.evaluation_plan(shipped, shipped.end)


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
    given = (ending(dev["plan"], cut), END, arms(dev))
    first, _ = heldback.scored(dev["values"], dev["grid"], *given)
    second, _ = heldback.scored(dev["values"], scrambled(dev["grid"], cut), *given)
    for name in first:
        assert first[name].equals(second[name]), name


def moved(values: dict, grid: dict, cut: date) -> tuple[dict, dict]:
    rng = np.random.default_rng(9)
    after = grid["ts_ns"] >= labels.ns(cut)
    out = {}
    for name, v in values.items():
        shifted = np.where(after & np.isfinite(v), v + 10 * rng.normal(size=v.shape), v)
        out[name] = np.where(after & (rng.random(v.shape) < 0.002), np.nan, shifted)
    flags = dict(grid)
    for name in ("skip", *(f"censored_{h}" for h in labels.HORIZONS)):
        flags[name] = np.where(after, rng.random(after.size) < 0.05, grid[name])
    return out, flags


def test_no_input_flag_or_gap_at_or_after_the_cut_reaches_a_fit_or_a_scale(
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
    given = (ending(dev["plan"], cut), END, arms(dev))
    first_scores, _ = heldback.scored(dev["values"], dev["grid"], *given)
    first = list(seen)
    seen.clear()
    values, grid = moved(dev["values"], dev["grid"], cut)
    assert (np.isnan(values["vol_1h"]) & ~np.isnan(dev["values"]["vol_1h"])).any()
    second_scores, _ = heldback.scored(values, grid, *given)
    assert {k[0] for k in first} == {"fit", "scale", "statistics"}
    assert first == seen
    assert all(first_scores[n].num_rows != second_scores[n].num_rows for n in first_scores)


def test_the_fit_plan_refuses_loqo_other_folds_and_cuts_it_was_not_given(
    plan: sequence.Plan, panel: tuple[dict, dict], tmp_path: Path
) -> None:
    values, grid = panel
    fit = heldback.fit_sequence(ending(plan, CUTS[3]), END)
    heldback.fit_only(fit)
    assert ladder.bounds(fit.ladder).tolist() == [
        labels.ns(d) for d in (date(2022, 1, 1), CUTS[3], date(2023, 1, 1))
    ]
    with pytest.raises(ValueError, match="fits only forward"):
        fit.ladder.first_fold("loqo")
    loqo = replace(fit, schemes=("loqo",))
    with pytest.raises(ValueError, match="fits only forward"):
        sequence.external(values, grid, loqo, ("supervised", "vol", None, None), unread)
    with pytest.raises(ValueError, match="fits only forward"):
        classical.export(values, grid, loqo, Config("rung", "vol", None, 0.01))
    four = (date(2022, 1, 1), CUTS[2], CUTS[3], date(2023, 1, 1))
    for bad in (
        loqo,
        replace(fit, schemes=("loqo", "forward")),
        replace(fit, folds=(0,)),
        replace(fit, folds=(0, 1)),
        replace(fit, folds=(1, 2)),
        replace(fit, ladder=replace(fit.ladder, forward_first=0)),
        replace(fit, ladder=replace(fit.ladder, cuts=four)),
        plan,
    ):
        with pytest.raises(ValueError, match="a fit plan"):
            heldback.fit_only(bad)
    for cuts in (
        (date(2022, 1, 1), date(2023, 1, 1)),
        (date(2022, 1, 1), CUTS[3], CUTS[2], date(2023, 1, 1)),
        (date(2022, 2, 1), CUTS[3], date(2023, 1, 1)),
        (date(2022, 1, 1), CUTS[3], CUTS[3], date(2023, 1, 1)),
    ):
        with pytest.raises(ValueError, match="cuts"):
            replace(fit.ladder, cuts=cuts)
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
    with pytest.raises(ValueError, match="falls before"):
        heldback.fit_plan(ending(plan, CUTS[3]).ladder, date(2022, 9, 30))
    with pytest.raises(ValueError, match="already cut"):
        heldback.fit_plan(fit.ladder, END)


def test_a_model_whose_sha256_differs_from_the_record_is_refused(dev: dict[str, Any]) -> None:
    arm = arms(dev)[0]
    assert isinstance(arm, heldback.SequenceArm)
    key = sequence.encoder_name("vol", "forward", 3, 3, "jepa")
    given = (dev["values"], dev["grid"], ending(dev["plan"], CUTS[3]), END)
    changed = replace(arm, encoders={**arm.encoders, key: "0" * 64})
    with pytest.raises(ValueError, match="sha256 differs"):
        heldback.scored(*given, [changed])
    unrecorded = replace(arm, encoders={k: v for k, v in arm.encoders.items() if k != key})
    with pytest.raises(ValueError, match="records no"):
        heldback.scored(*given, [unrecorded])
    other = replace(arm, read=lambda k: b"not the model")
    with pytest.raises(ValueError, match="sha256 differs"):
        heldback.scored(*given, [other])


def test_the_fetch_refuses_a_model_the_arm_does_not_pin(plan: sequence.Plan) -> None:
    arm = pinned({}, unread)[0]
    assert isinstance(arm, heldback.SequenceArm)
    fetch = heldback.fetcher(arm, 3, {})
    for key in (
        sequence.encoder_name("vol", "forward", 1, 5, "jepa"),
        sequence.encoder_name("vol", "forward", 3, 3, "jepa"),
        sequence.encoder_name("vol", "forward", 1, 3, "twin"),
        "anything",
    ):
        with pytest.raises(ValueError, match="not one of its pinned models"):
            fetch(key)


def sequence_arm(**change: Any) -> heldback.SequenceArm:
    arm = pinned({}, unread)[0]
    assert isinstance(arm, heldback.SequenceArm)
    return replace(arm, **change)


@pytest.mark.parametrize(
    ("bad", "problem"),
    [
        (sequence_arm(seeds=(3, 3)), "seeds"),
        (sequence_arm(seeds=()), "seeds"),
        (sequence_arm(seeds=(-1,)), "seeds"),
        (sequence_arm(seeds=(True,)), "seeds"),
        (sequence_arm(c=-1.0), "c -1.0"),
        (sequence_arm(c=math.inf), "c inf"),
        (sequence_arm(c=0.0), "c 0.0"),
        (sequence_arm(c=math.nan), "c nan"),
        (sequence_arm(c=0.5), "c 0.5"),
        (sequence_arm(c=None), "c None"),
        (sequence_arm(model="other"), "model other"),
        (sequence_arm(readout="median"), "readout median"),
        (sequence_arm(readout=None), "readout None"),
        (sequence_arm(variant="full"), "variant full"),
        (sequence_arm(model="supervised"), "takes no readout or c"),
        (sequence_arm(model="supervised", readout=None), "takes no readout or c"),
        (sequence_arm(model="supervised", c=None), "takes no readout or c"),
        (heldback.ClassicalArm("jepa", Config("window", "vol", "means", 0.5)), "classical"),
        (heldback.RungArm("jepa", "price"), "no rung price"),
        ("jepa", "not a sequence, classical or rung arm"),
    ],
)
def test_an_arm_whose_pins_do_not_fit_the_plan_is_refused_before_any_fit(
    plan: sequence.Plan,
    panel: tuple[dict, dict],
    monkeypatch: pytest.MonkeyPatch,
    bad: Any,
    problem: str,
) -> None:
    values, grid = panel
    good = [heldback.RungArm("vol", "vol")]
    heldback.arms_check([sequence_arm(), *good], plan)
    with pytest.raises(ValueError, match=problem):
        heldback.arms_check([bad, *good], plan)

    def fitted(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an arm was fitted")

    monkeypatch.setattr(heldback, "rung_tables", fitted)
    monkeypatch.setattr(heldback, "support", fitted)
    before = ending(plan, CUTS[3])
    with pytest.raises(ValueError, match=problem):
        heldback.scored(values, grid, before, END, [*good, bad])
    with pytest.raises(ValueError, match=problem):
        heldback.run(values, grid, before, END, [*good, bad], [], ("a", "b"), "long", 24, rule())


def test_arms_must_be_given_and_distinct_before_the_support(
    plan: sequence.Plan, panel: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    values, grid = panel

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the support was settled")

    monkeypatch.setattr(heldback, "support", refused)
    before = ending(plan, CUTS[3])
    same = [heldback.RungArm("vol", "vol"), heldback.RungArm("vol", "full")]
    for given in ([], same):
        with pytest.raises(ValueError, match="given and distinct"):
            heldback.run(values, grid, before, END, given, [], ("vol", "x"), "long", 24, rule())


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
    given = (dev["values"], dev["grid"], ending(dev["plan"], CUTS[2]), END)
    chosen = [a for a in arms(dev) if a.name in ("jepa", "window", "full", "vol")]
    out = heldback.run(*given, chosen, PAIRS, PRIMARY, "long+short", 24, rule())
    decision = out["decision"]
    assert decision["outcome"] in ("supported", "not supported")
    assert decision["read"] == "q05" and decision["row"]["q05"] == decision["lower"]
    (row,) = [c for c in out["comparisons"] if c["primary"]]
    assert row["complete_draws"] == out["support"]["complete_draws"]
    assert row["folds"] == out["support"]["scoreable"] == 2
    other = [
        chosen[0],
        heldback.ClassicalArm("window", Config("window", "vol", "last", 0.01)),
    ]
    again = heldback.run(*given, other, [PRIMARY], PRIMARY, "long+short", 24, rule())
    (changed,) = [c for c in again["comparisons"] if c["primary"]]
    assert changed["estimate"] != row["estimate"]
    assert again["support"] == out["support"]
    monkeypatch.setattr(heldback, "scored", refused("heldback.scored"))
    early = heldback.run(*given, chosen, PAIRS, PRIMARY, "long+short", 24, rule(min_quarters=3))
    assert early["decision"]["outcome"] == "inconclusive"
    assert set(early) == {"support", "decision"}


def fake(
    values: dict, grid: dict, plan: ladder.Plan, names: Sequence[str] = ("a", "b")
) -> dict[str, pa.Table]:
    rng = np.random.default_rng(11)
    out = {}
    for name in names:
        parts = []
        for h in plan.horizons:
            keep, _ = ladder.select(values, grid, plan, h)
            n = int(keep.sum())
            sides = {side: rng.normal(size=n) for side in plan.sides}
            for side in plan.sides:
                parts.append(
                    sequence.scores_part(
                        grid["symbol"][keep].astype(str),
                        grid["ts_ns"][keep],
                        np.ones(n, np.int16),
                        np.ones(n, bool),
                        h,
                        side,
                        sides,
                        "forward",
                    )
                )
        out[name] = pa.concat_tables(parts)
    return out


def agree(
    values: dict, grid: dict, plan: ladder.Plan, side: str, r: heldback.Rule
) -> tuple[dict[str, Any], dict[str, Any]]:
    found = heldback.support(values, grid, plan, side, 24, r)
    out = heldback.paired(
        values, grid, plan, fake(values, grid, plan), [("a", "b")], ("a", "b"), side, 24
    )
    (row,) = [c for c in out["comparisons"] if c["primary"]]
    numbered = found["numbered_draws"]
    assert (row["folds"], row["quarters_scheduled"], row["draws"], row["draws_complete"]) == (
        found["scoreable"],
        len(found["quarters"]),
        numbered["all"],
        numbered["complete"],
    )
    complete = found["complete_draws"]
    assert row["complete_draws"] == complete or (
        math.isnan(complete) and math.isnan(row["complete_draws"])
    )
    return found, row


def placed(grid: dict, h: int = 24) -> dict:
    rng = np.random.default_rng(3)
    ts = grid["ts_ns"]
    out = dict(grid)
    out["skip"] = np.zeros(ts.size, bool)
    out[f"censored_{h}"] = np.zeros(ts.size, bool)
    q4 = labels.ns(date(2022, 10, 1))
    first_block = (ts >= q4) & (ts < q4 + 14 * DAY)
    spread = rng.random(ts.size) < 0.05
    out[f"long_{h}"] = spread & ((ts < q4) | first_block)
    out[f"short_{h}"] = spread
    return out


def test_the_support_counts_quarters_blocks_and_complete_draws_from_labels(
    plan: sequence.Plan, panel: tuple[dict, dict]
) -> None:
    values, grid = panel
    q = evaluation(plan, CUTS[2])
    grid = placed(grid)
    found, _ = agree(values, grid, q, "long", rule())
    quarters = found["quarters"]
    assert [x["quarter"] for x in quarters] == ["2022-07-01", "2022-10-01"]
    assert [x["blocks"] for x in quarters] == [7, 7]
    assert [x["long_blocks_with_positive"] for x in quarters] == [7, 1]
    assert found["scoreable"] == 2 and found["supported"] == 2 and found["causes"] == []
    lost = (6 / 7) ** 7
    assert 0 < found["complete_draws"] < 1
    assert abs((1 - found["complete_draws"]) - lost) < 0.2
    both, _ = agree(values, grid, q, "long+short", rule())
    assert [x["short_blocks_with_positive"] for x in both["quarters"]] == [7, 7]
    assert both["complete_draws"] == found["complete_draws"]
    short, _ = agree(values, grid, q, "short", rule())
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
    plan: sequence.Plan, panel: tuple[dict, dict], change: dict[str, Any], cause: str, side: str
) -> None:
    values, grid = panel
    q = evaluation(plan, CUTS[2])
    strict = rule(min_supported=2, **{k: v for k, v in change.items() if k != "min_supported"})
    found = heldback.support(values, placed(grid), q, side, 24, strict)
    assert len(found["causes"]) == 1 and cause in found["causes"][0]
    decision = heldback.decide(found, [{"primary": True}], PRIMARY, strict)
    assert decision["outcome"] == "inconclusive"
    assert decision["causes"] == found["causes"] and decision["support"] == found


def apart(grid: dict, i: int, j: int, start: date) -> dict:
    rng = np.random.default_rng(4)
    ts = grid["ts_ns"]
    out = dict(grid)
    out["skip"] = np.zeros(ts.size, bool)
    out["censored_24"] = np.zeros(ts.size, bool)
    begin = labels.ns(start)
    inside = (ts >= begin) & (ts < labels.ns(date(2022, 10, 1)))
    block = (ts - begin) // (14 * DAY)
    some = rng.random(ts.size) < 0.3
    out["long_24"] = np.where(inside, some & (block == i), grid["long_24"])
    out["short_24"] = np.where(inside, some & (block == j), grid["short_24"])
    return out


def test_a_bound_no_draw_can_give_makes_the_outcome_inconclusive_before_any_score(
    plan: sequence.Plan, panel: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    values, grid = panel
    q = evaluation(plan, CUTS[2], date(2022, 9, 30), draws=4)
    nblocks = ladder.blocks_per_fold(ladder.bounds(q), q.block_days)
    (w,) = ladder.draws(nblocks, q.draws, q.seed, 24)
    found_pairs = [
        (i, j)
        for i in range(7)
        for j in range(7)
        if i != j and w[:, i].any() and w[:, j].any() and not (w[:, i] * w[:, j]).any()
    ]
    i, j = found_pairs[0]
    grid = apart(grid, i, j, CUTS[2])
    loose = {"min_quarters": 1, "min_blocks": 1, "min_supported": 1, "min_complete": 0.0}
    for draws, key in (("all", "q05"), ("complete", "q05_complete")):
        r = rule(draws=draws, **loose)
        found, row = agree(values, grid, q, "long+short", r)
        assert found["causes"] == [f"no draw gives the {key} a number"]
        assert found["numbered_draws"] == {"all": 0, "complete": 0}
        assert found["scoreable"] == 1 and found["complete_draws"] == 0.0
        assert math.isnan(row[key])
        decision = heldback.decide(found, [row], ("a", "b"), r)
        assert decision["outcome"] == "inconclusive"
        with pytest.raises(ValueError, match="not a number"):
            heldback.decide(found | {"causes": []}, [row], ("a", "b"), r)
    long, row = agree(values, grid, q, "long", rule(**loose))
    assert long["causes"] == [] and 0 < long["numbered_draws"]["all"] < 4
    assert math.isfinite(row["q05"])

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an arm was scored")

    monkeypatch.setattr(heldback, "scored", refused)
    before = replace(ending(plan, CUTS[2]), ladder=replace(ending(plan, CUTS[2]).ladder, draws=4))
    rungs = [heldback.RungArm("full", "full"), heldback.RungArm("vol", "vol")]
    for draws in ("all", "complete"):
        out = heldback.run(
            values,
            grid,
            before,
            date(2022, 9, 30),
            rungs,
            [("full", "vol")],
            ("full", "vol"),
            "long+short",
            24,
            rule(draws=draws, **loose),
        )
        assert out["decision"]["outcome"] == "inconclusive" and set(out) == {"support", "decision"}


def one_sided(grid: dict) -> dict:
    out = dict(grid)
    q4 = grid["ts_ns"] >= labels.ns(CUTS[3])
    out["short_24"] = np.where(q4, False, grid["short_24"])
    return out


def test_a_quarter_one_side_cannot_score_drops_out_of_long_and_short(
    plan: sequence.Plan, panel: tuple[dict, dict]
) -> None:
    values, grid = panel
    grid = one_sided(grid)
    q = evaluation(plan, CUTS[2])
    r = rule(min_quarters=1, min_blocks=0, min_supported=1)
    both, row = agree(values, grid, q, "long+short", r)
    assert [x["scoreable"] for x in both["quarters"]] == [True, False]
    assert [x["supported"] for x in both["quarters"]] == [True, False]
    assert both["scoreable"] == 1 and both["supported"] == 1 and both["causes"] == []
    assert row["folds"] == 1 and row["quarters_scheduled"] == 2
    long, _ = agree(values, grid, q, "long", r)
    assert long["scoreable"] == 2 and long["supported"] == 2
    rungs = [heldback.RungArm("full", "full"), heldback.RungArm("vol", "vol")]
    out = heldback.run(
        values,
        grid,
        ending(plan, CUTS[2]),
        END,
        rungs,
        [("full", "vol")],
        ("full", "vol"),
        "long+short",
        24,
        r,
    )
    assert out["decision"]["outcome"] in ("supported", "not supported")
    assert out["decision"]["row"]["folds"] == 1
    assert out["decision"]["row"]["quarters_scheduled"] == 2


def test_no_scoreable_quarter_names_every_cause_and_scores_nothing(
    plan: sequence.Plan, panel: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    values, grid = panel
    after = grid["ts_ns"] >= labels.ns(CUTS[2])
    grid = dict(grid)
    for side in ("long", "short"):
        grid[f"{side}_24"] = np.where(after, False, grid[f"{side}_24"])
    q = evaluation(plan, CUTS[2])
    r = rule(min_quarters=1)
    found, row = agree(values, grid, q, "long+short", r)
    assert found["scoreable"] == 0 and found["supported"] == 0
    assert math.isnan(found["complete_draws"])
    assert len(found["causes"]) == 4
    assert heldback.decide(found, [row], ("a", "b"), r)["outcome"] == "inconclusive"

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an arm was scored")

    monkeypatch.setattr(heldback, "scored", refused)
    rungs = [heldback.RungArm("full", "full"), heldback.RungArm("vol", "vol")]
    out = heldback.run(
        values,
        grid,
        ending(plan, CUTS[2]),
        END,
        rungs,
        [("full", "vol")],
        ("full", "vol"),
        "long",
        24,
        r,
    )
    assert out["decision"]["outcome"] == "inconclusive"


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
        "folds": found["scoreable"],
        "quarters_scheduled": len(found["quarters"]),
        "draws": found["numbered_draws"]["all"],
        "draws_complete": found["numbered_draws"]["complete"],
        **values,
    }


def test_the_decision_reads_the_rules_bound_on_sufficient_support(
    plan: sequence.Plan, panel: tuple[dict, dict]
) -> None:
    values, grid = panel
    q = evaluation(plan, CUTS[2])
    grid = placed(grid)
    bounds = {"q05": 0.01, "q025": -0.01, "q05_complete": -0.02, "q025_complete": 0.03}
    expected = {
        ("q05", "all"): "supported",
        ("q025", "all"): "not supported",
        ("q05", "complete"): "not supported",
        ("q025", "complete"): "supported",
    }
    for (bound, draws), outcome in expected.items():
        r = rule(bound=bound, draws=draws)
        found = heldback.support(values, grid, q, "long", 24, r)
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
    for bad in ({"bound": "q10"}, {"draws": "some"}, {"min_quarters": -1}, {"min_complete": 2.0}):
        with pytest.raises(ValueError, match="the decision rule"):
            rule(**bad)


def test_the_decision_refuses_a_row_that_is_not_the_supports(
    plan: sequence.Plan, panel: tuple[dict, dict]
) -> None:
    values, grid = panel
    r = rule()
    found = heldback.support(values, placed(grid), evaluation(plan, CUTS[2]), "long", 24, r)
    row = primary_row(found)
    heldback.decide(found, [row], PRIMARY, r)
    with pytest.raises(ValueError, match="another rule"):
        heldback.decide(found, [row], PRIMARY, rule(min_blocks=2))
    for rows in ([], [{**row, "primary": False}], [row, row]):
        with pytest.raises(ValueError, match="primary comparison rows"):
            heldback.decide(found, rows, PRIMARY, r)
    for change in (
        {"side": "short"},
        {"scheme": "loqo"},
        {"horizon": 1},
        {"first": "window", "second": "jepa"},
    ):
        with pytest.raises(ValueError, match="not the support"):
            heldback.decide(found, [primary_row(found, **change)], PRIMARY, r)
    with pytest.raises(RuntimeError, match="complete draws"):
        heldback.decide(found, [primary_row(found, complete_draws=0.5)], PRIMARY, r)
    for key in ("folds", "quarters_scheduled", "draws", "draws_complete"):
        with pytest.raises(RuntimeError, match="not the support"):
            heldback.decide(found, [primary_row(found, **{key: row[key] + 1})], PRIMARY, r)
    with pytest.raises(ValueError, match="not a number"):
        heldback.decide(found, [primary_row(found, q05=float("nan"))], PRIMARY, r)


@pytest.mark.parametrize(
    ("pairs", "primary", "side", "horizon", "problem"),
    [
        ([("a", "x")], ("a", "x"), "long", 24, "unknown arm"),
        ([("a", "a")], ("a", "a"), "long", 24, "with itself"),
        ([("a", "b"), ("b", "a")], ("a", "b"), "long", 24, "listed twice"),
        ([("a", "b"), ("a", "b")], ("a", "b"), "long", 24, "listed twice"),
        ([("a", "b")], ("a", "c"), "long", 24, "not one of the pairs"),
        ([("a", "b")], ("b", "a"), "long", 24, "not one of the pairs"),
        ([("a", "b")], ("a", "b"), "any", 24, "side any"),
        ([("a", "b")], ("a", "b"), "up", 24, "side up"),
        ([("a", "b")], ("a", "b"), "long", 4, "horizon 4"),
        ([("a", "b")], ("a", "b"), "long", True, "horizon True"),
        ([("a", "b")], ("a", "b"), "long", 24.0, "horizon 24.0"),
    ],
)
def test_the_paired_evaluation_refuses_what_it_cannot_pair(
    plan: sequence.Plan,
    pairs: list[tuple[str, str]],
    primary: tuple[str, str],
    side: str,
    horizon: Any,
    problem: str,
) -> None:
    q = evaluation(plan, CUTS[2])
    names = ["a", "b", "c"]
    heldback.check(names, [("a", "b"), ("a", "c")], ("a", "b"), "long+short", 1, q)
    with pytest.raises(ValueError, match=problem):
        heldback.check(names, pairs, primary, side, horizon, q)


def test_the_support_and_the_pairing_take_only_an_evaluation_plan(
    plan: sequence.Plan, panel: tuple[dict, dict]
) -> None:
    values, grid = panel
    q = evaluation(plan, CUTS[2])
    tables = fake(values, grid, q)
    no_short = replace(q, sides=("long", "any"))
    with pytest.raises(ValueError, match="side long"):
        heldback.check(["a", "b"], [("a", "b")], ("a", "b"), "long+short", 24, no_short)
    cases: list[tuple[str, Any, str]] = [("any", 24, "side any"), ("long", 1.0, "horizon 1.0")]
    for side, horizon, problem in cases:
        with pytest.raises(ValueError, match=problem):
            heldback.support(values, grid, q, side, horizon, rule())
    fit = heldback.fit_plan(ending(plan, CUTS[2]).ladder, END)
    for bad in (fit, plan.ladder, replace(q, forward_first=1)):
        with pytest.raises(ValueError, match="an evaluation plan"):
            heldback.support(values, grid, bad, "long", 24, rule())
        with pytest.raises(ValueError, match="an evaluation plan"):
            heldback.paired(values, grid, bad, tables, [("a", "b")], ("a", "b"), "long", 24)


def test_the_refold_takes_only_fit_fold_scores_inside_the_span(
    plan: sequence.Plan, panel: tuple[dict, dict]
) -> None:
    values, grid = panel
    q = evaluation(plan, CUTS[2])
    table = fake(values, grid, q, ("a",))["a"]
    folded = heldback.refold(table, q)
    hours = table["hour"].to_numpy()
    assert folded["fold"].to_numpy().tolist() == (hours >= labels.ns(CUTS[3])).astype(int).tolist()
    fold = table.schema.get_field_index("fold")
    hour = table.schema.get_field_index("hour")
    second = table.set_column(fold, "fold", pa.array(np.full(table.num_rows, 2), pa.int16()))
    with pytest.raises(ValueError, match="fit fold"):
        heldback.refold(second, q)
    for moved_to in (labels.ns(CUTS[2]) - HOUR, labels.ns(date(2023, 1, 1))):
        shifted = hours.copy()
        shifted[0] = moved_to
        outside = table.set_column(hour, "hour", pa.array(shifted, pa.int64()))
        with pytest.raises(ValueError, match="outside the evaluation span"):
            heldback.refold(outside, q)
    gaps = pa.array([None, *hours[1:].tolist()], pa.int64())
    with pytest.raises(ValueError, match="no hour"):
        heldback.refold(table.set_column(hour, "hour", gaps), q)


def test_the_test_span_has_eleven_quarters_of_the_blocks_the_bootstrap_expects() -> None:
    dev = ladder.load()
    end = date(2026, 9, 30)
    q = heldback.evaluation_plan(dev, end)
    edges = ladder.bounds(q)
    starts = [date(y, m, 1) for y in (2024, 2025, 2026) for m in (1, 4, 7, 10)]
    assert edges.tolist() == [labels.ns(d) for d in starts]
    assert ladder.blocks_per_fold(edges, q.block_days).tolist() == [7] * 11
    assert q.first_fold("forward") == 0 and q.cuts is None
    heldback.evaluation_only(q)
    days = np.diff(edges) // DAY
    last = days - (ladder.blocks_per_fold(edges, q.block_days) - 1) * 14
    assert set(last.tolist()) <= {6, 7, 8}
    fit = heldback.fit_plan(dev, end)
    cut = labels.ns(span.HELD_BACK)
    assert ladder.bounds(fit).tolist() == [labels.ns(dev.start), cut, labels.ns(date(2026, 10, 1))]
    ts = labels.ns(dev.start) + HOUR * np.arange(0, 5 * 365 * 24, 7, dtype=np.int64)
    fold, _ = ladder.folds(ts, ladder.bounds(fit), fit.block_days)
    train = ladder.training(ts, fold, 1, ladder.bounds(fit), fit.gap_days, "forward")
    assert np.array_equal(train, ts < cut - fit.gap_days * DAY)
    assert np.array_equal(fold == 1, (ts >= cut) & (ts < labels.ns(date(2026, 10, 1))))
    text = ladder.PLAN.read_text(encoding="utf-8")
    later = text.replace("start = 2022-01-01", "start = 2024-01-01").replace(
        "end = 2023-12-31", "end = 2026-09-30"
    )
    with pytest.raises(span.HeldBackError):
        ladder.load(ladder.PLAN, later)
    with pytest.raises(ValueError, match="forward_first"):
        ladder.load(ladder.PLAN, text.replace("forward_first = 2", "forward_first = 0"))
