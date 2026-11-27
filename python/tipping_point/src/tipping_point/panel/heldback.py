# The held-back test's scoring core (docs/market.md, "Coin panel"): a fit plan that fits each
# pinned arm once on development hours and scores the test span, an evaluation plan that pairs the
# arms over its calendar quarters, and a decision whose block support is settled before any score.

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pyarrow as pa

from . import classical, compare, ladder, sequence, span

SCHEME = compare.PRIMARY_SCHEME
FIT_FOLD = 1
SIDES = ("long", "short", compare.COMPOSITE)
BOUNDS = ("q05", "q025")
DRAWS = ("all", "complete")
LOG = Callable[[str], None]


@dataclass(frozen=True)
class SequenceArm:
    name: str
    model: str
    variant: str
    readout: str | None
    c: float | None
    seeds: tuple[int, ...]
    encoders: Mapping[str, str]
    read: Callable[[str], bytes]


@dataclass(frozen=True)
class ClassicalArm:
    name: str
    config: classical.Config


@dataclass(frozen=True)
class RungArm:
    name: str
    rung: str


Arm = SequenceArm | ClassicalArm | RungArm


@dataclass(frozen=True)
class Rule:
    bound: str
    draws: str
    min_quarters: int
    min_complete: float
    min_blocks: int
    min_supported: int

    def __post_init__(self) -> None:
        problems = []
        if self.bound not in BOUNDS:
            problems.append(f"bound {self.bound} (one of {BOUNDS})")
        if self.draws not in DRAWS:
            problems.append(f"draws {self.draws} (one of {DRAWS})")
        for name in ("min_quarters", "min_blocks", "min_supported"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                problems.append(f"{name} = {value!r} (a whole number, not negative)")
        share = self.min_complete
        if type(share) not in (int, float) or not 0 <= share <= 1:
            problems.append(f"min_complete = {share!r} (a share from 0 to 1)")
        if problems:
            raise ValueError(f"the decision rule: {', '.join(problems)}")


def _spans(dev: ladder.Plan, end: date, cut: date) -> None:
    if dev.cuts is not None:
        raise ValueError("the development plan is already cut")
    if not dev.start < cut <= end:
        raise ValueError(f"the cut {cut} must fall after {dev.start} and on or before {end}")


def fit_plan(dev: ladder.Plan, end: date, cut: date = span.HELD_BACK) -> ladder.Plan:
    _spans(dev, end, cut)
    return dataclasses.replace(
        dev,
        end=end,
        forward_first=FIT_FOLD,
        external=None,
        cuts=(dev.start, cut, end + timedelta(days=1)),
    )


def evaluation_plan(dev: ladder.Plan, end: date, cut: date = span.HELD_BACK) -> ladder.Plan:
    _spans(dev, end, cut)
    return dataclasses.replace(dev, start=cut, end=end, forward_first=0, external=None)


def fit_sequence(dev: sequence.Plan, end: date, cut: date = span.HELD_BACK) -> sequence.Plan:
    return dataclasses.replace(
        dev, ladder=fit_plan(dev.ladder, end, cut), schemes=(SCHEME,), folds=(FIT_FOLD,)
    )


def fit_only(plan: sequence.Plan) -> None:
    base = plan.ladder
    cut = base.cuts is not None and len(base.cuts) == 3 and base.forward_first == FIT_FOLD
    if not cut or plan.schemes != (SCHEME,) or plan.folds != (FIT_FOLD,):
        raise ValueError(
            f"a fit plan is cut once, fits {SCHEME} only and fold {FIT_FOLD} only, not schemes "
            f"{list(plan.schemes)} and folds {list(plan.folds)}"
        )


def final_fold(dev: sequence.Plan) -> int:
    return len(ladder.bounds(dev.ladder)) - 1


def fetcher(arm: SequenceArm, final: int, used: dict[str, str]) -> Callable[[str], bytes]:
    names = {
        sequence.encoder_name(arm.variant, SCHEME, FIT_FOLD, s, arm.model): sequence.encoder_name(
            arm.variant, SCHEME, final, s, arm.model
        )
        for s in arm.seeds
    }

    def fetch(key: str) -> bytes:
        if key not in names:
            raise ValueError(f"arm {arm.name}: {key} is not one of its pinned models")
        name = names[key]
        if name not in arm.encoders:
            raise ValueError(f"arm {arm.name}: the final run records no {name}")
        data = arm.read(name)
        digest = hashlib.sha256(data).hexdigest()
        if digest != arm.encoders[name]:
            raise ValueError(f"arm {arm.name}: {name}: its sha256 differs from the final run's")
        used[name] = digest
        return data

    return fetch


def rung_tables(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    plan: ladder.Plan,
    arms: Sequence[RungArm],
    log: LOG,
) -> dict[str, pa.Table]:
    edges = ladder.bounds(plan)
    configs = list(dict.fromkeys((arm.rung, plan.c) for arm in arms))
    parts: dict[str, list[pa.Table]] = {arm.name: [] for arm in arms}
    for h in plan.horizons:
        keep, _ = ladder.select(values, grid, plan, h)
        ts, symbol = grid["ts_ns"][keep], grid["symbol"][keep].astype(str)
        y = {side: grid[f"{side}_{h}"][keep] for side in ("long", "short")}
        fold, _ = ladder.folds(ts, edges, plan.block_days)
        x = np.column_stack([values[name][keep] for name in plan.columns])
        got = ladder.penalised(x, y, ts, fold, edges, plan, SCHEME, h, log, configs)
        mine = fold >= plan.first_fold(SCHEME)
        for arm in arms:
            for side in plan.sides:
                parts[arm.name].append(
                    sequence.scores_part(
                        symbol, ts, fold, mine, h, side, got[(arm.rung, plan.c)], SCHEME
                    )
                )
    return {name: pa.concat_tables(tables) for name, tables in parts.items()}


def scored(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    dev: sequence.Plan,
    end: date,
    arms: Sequence[Arm],
    cut: date = span.HELD_BACK,
    log: LOG = lambda line: None,
) -> tuple[dict[str, pa.Table], dict[str, dict[str, str]]]:
    plan = fit_sequence(dev, end, cut)
    fit_only(plan)
    names = [arm.name for arm in arms]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"arms {names} must be given and distinct")
    rungs = [arm for arm in arms if isinstance(arm, RungArm)]
    unknown = [arm.rung for arm in rungs if arm.rung not in {r.name for r in plan.ladder.rungs}]
    if unknown:
        raise ValueError(f"the ladder plan has no rung {', '.join(unknown)}")
    tables = rung_tables(values, grid, plan.ladder, rungs, log) if rungs else {}
    used: dict[str, dict[str, str]] = {}
    final = final_fold(dev)
    for arm in arms:
        if isinstance(arm, SequenceArm):
            used[arm.name] = {}
            pinned = dataclasses.replace(plan, seeds=tuple(arm.seeds))
            choice = (arm.model, arm.variant, arm.readout, arm.c)
            fetch = fetcher(arm, final, used[arm.name])
            tables[arm.name] = sequence.external(values, grid, pinned, choice, fetch, log)[SCHEME]
        elif isinstance(arm, ClassicalArm):
            tables[arm.name] = classical.export(values, grid, plan, arm.config, log)[SCHEME]
    return {name: tables[name] for name in names}, used


def refold(table: pa.Table, evaluation: ladder.Plan) -> pa.Table:
    if not (table["fold"].cast(pa.int64()).to_numpy() == FIT_FOLD).all():
        raise ValueError(f"scores to pair must all come from fit fold {FIT_FOLD}")
    hours = ladder.hour_ns(table["hour"]).to_numpy()
    edges = ladder.bounds(evaluation)
    if hours.size and (hours.min() < edges[0] or hours.max() >= edges[-1]):
        raise ValueError("scores to pair hold an hour outside the evaluation span")
    fold, _ = ladder.folds(hours, edges, evaluation.block_days)
    index = table.schema.get_field_index("fold")
    return table.set_column(index, "fold", pa.array(fold, pa.int16()))


def check(
    names: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    primary: tuple[str, str],
    side: str,
    horizon: int,
    evaluation: ladder.Plan,
) -> None:
    problems = [f"pair {a} - {b} names an unknown arm" for a, b in pairs if {a, b} - set(names)]
    problems += [f"pair {a} - {b} compares an arm with itself" for a, b in pairs if a == b]
    seen = [frozenset(p) for p in pairs]
    if len(set(seen)) != len(seen):
        problems.append("a pair is listed twice")
    if primary not in pairs:
        problems.append(f"primary {primary[0]} - {primary[1]} is not one of the pairs")
    wanted = ("long", "short") if side == compare.COMPOSITE else (side,)
    if side not in SIDES or not set(wanted) <= set(evaluation.sides):
        problems.append(f"side {side} (one of {SIDES} the plan holds)")
    if horizon not in evaluation.horizons:
        problems.append(f"horizon {horizon} (one of {list(evaluation.horizons)})")
    if problems:
        raise ValueError(f"the paired evaluation: {', '.join(problems)}")


def paired(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    evaluation: ladder.Plan,
    tables: Mapping[str, pa.Table],
    pairs: Sequence[tuple[str, str]],
    primary: tuple[str, str],
    side: str,
    horizon: int,
    log: LOG = lambda line: None,
) -> dict[str, Any]:
    check(list(tables), pairs, primary, side, horizon, evaluation)
    arms = tuple(compare.Arm(name, "scores") for name in tables)
    plan = compare.Plan(evaluation, arms, tuple(pairs), primary, side, horizon)
    held = {name: {SCHEME: refold(table, evaluation)} for name, table in tables.items()}
    return compare.compare(values, grid, plan, held, [SCHEME], log)


def _quarter(edge: int) -> str:
    return str(np.datetime64(int(edge), "ns"))[:10]


def support(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    evaluation: ladder.Plan,
    side: str,
    horizon: int,
    rule: Rule,
) -> dict[str, Any]:
    keep, accounting = ladder.select(values, grid, evaluation, horizon)
    ts = grid["ts_ns"][keep]
    edges = ladder.bounds(evaluation)
    fold, block = ladder.folds(ts, edges, evaluation.block_days)
    nblocks = ladder.blocks_per_fold(edges, evaluation.block_days)
    weights = ladder.draws(nblocks, evaluation.draws, evaluation.seed, horizon)
    sides = ("long", "short") if side == compare.COMPOSITE else (side,)
    scoreable = np.ones(len(nblocks), bool)
    supported = np.ones(len(nblocks), bool)
    held: dict[str, np.ndarray] = {}
    quarters: list[dict[str, Any]] = [
        {"quarter": _quarter(edges[f]), "blocks": int(nblocks[f]), "rows": int((fold == f).sum())}
        for f in range(len(nblocks))
    ]
    for s in sides:
        label = grid[f"{s}_{horizon}"][keep]
        held[s] = np.zeros((evaluation.draws, len(nblocks)), bool)
        for f, k in enumerate(nblocks):
            inside = fold == f
            p, n = ladder.block_counts(label[inside], block[inside], int(k))
            blocks = int((p > 0).sum())
            scoreable[f] &= bool(p.sum() > 0 and n.sum() > 0)
            supported[f] &= blocks >= rule.min_blocks
            held[s][:, f] = (weights[f] @ p) * (weights[f] @ n) > 0
            quarters[f] |= {f"{s}_positives": int(p.sum()), f"{s}_blocks_with_positive": blocks}
    supported &= scoreable
    whole = scoreable.any() & np.all([held[s][:, scoreable].all(axis=1) for s in sides], axis=0)
    complete = float(whole.mean()) if scoreable.any() else math.nan
    for f, q in enumerate(quarters):
        q |= {"scoreable": bool(scoreable[f]), "supported": bool(supported[f])}
    causes = []
    if scoreable.sum() < rule.min_quarters:
        causes.append(f"{int(scoreable.sum())} scoreable quarters, fewer than {rule.min_quarters}")
    if not complete >= rule.min_complete:
        causes.append(f"a share of complete draws of {complete}, below {rule.min_complete}")
    if supported.sum() < rule.min_supported:
        causes.append(
            f"{int(supported.sum())} quarters with a positive in at least {rule.min_blocks} "
            f"blocks, fewer than {rule.min_supported}"
        )
    return {
        "side": side,
        "horizon": horizon,
        "rule": asdict(rule),
        "quarters": quarters,
        "scoreable": int(scoreable.sum()),
        "supported": int(supported.sum()),
        "complete_draws": complete,
        "draws": evaluation.draws,
        "rows": accounting,
        "causes": causes,
    }


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


ROW = (
    "estimate",
    "q025",
    "q05",
    "q975",
    "above_zero",
    "draws",
    "q025_complete",
    "q05_complete",
    "q975_complete",
    "above_zero_complete",
    "draws_complete",
    "complete_draws",
    "folds",
    "quarters_scheduled",
    "fold_differences",
)


def decide(
    found: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    primary: tuple[str, str],
    rule: Rule,
) -> dict[str, Any]:
    if found["rule"] != asdict(rule):
        raise ValueError("the support was settled under another rule")
    out: dict[str, Any] = {"support": dict(found), "primary": list(primary)}
    if found["causes"]:
        return out | {"outcome": "inconclusive", "causes": list(found["causes"])}
    rows = [c for c in comparisons if c["primary"]]
    if len(rows) != 1:
        raise ValueError(f"{len(rows)} primary comparison rows, not one")
    (row,) = rows
    named = (row["first"], row["second"], row["side"], row["horizon"], row["scheme"])
    if named != (*primary, found["side"], found["horizon"], SCHEME):
        raise ValueError(f"the primary row is {named}, not the support's pair, side and horizon")
    if not _same(float(row["complete_draws"]), float(found["complete_draws"])):
        raise RuntimeError("the comparison's share of complete draws is not the support's")
    key = rule.bound + ("_complete" if rule.draws == "complete" else "")
    lower = float(row[key])
    if not math.isfinite(lower):
        raise ValueError(f"the primary row's {key} is not a number")
    return out | {
        "outcome": "supported" if lower > 0 else "not supported",
        "causes": [],
        "read": key,
        "lower": lower,
        "row": {k: row[k] for k in ROW},
    }


def run(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    dev: sequence.Plan,
    end: date,
    arms: Sequence[Arm],
    pairs: Sequence[tuple[str, str]],
    primary: tuple[str, str],
    side: str,
    horizon: int,
    rule: Rule,
    cut: date = span.HELD_BACK,
    log: LOG = lambda line: None,
) -> dict[str, Any]:
    evaluation = evaluation_plan(dev.ladder, end, cut)
    check([arm.name for arm in arms], pairs, primary, side, horizon, evaluation)
    found = support(values, grid, evaluation, side, horizon, rule)
    if found["causes"]:
        return {"support": found, "decision": decide(found, (), primary, rule)}
    tables, used = scored(values, grid, dev, end, arms, cut, log)
    out = paired(values, grid, evaluation, tables, pairs, primary, side, horizon, log)
    decision = decide(found, out["comparisons"], primary, rule)
    return out | {"support": found, "decision": decision, "tables": tables, "encoders": used}
