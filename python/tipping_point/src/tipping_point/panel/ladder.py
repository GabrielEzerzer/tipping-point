# The ladder: nested sets of panel inputs fitted out of fold by calendar quarter, each quarter left
# out and forward only, scored on one common row set from a plan file (ladder.toml). Comparisons
# carry a joint block bootstrap whose AUCs come from per-block pair counts (docs/market.md).

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..ews.scoring import auc
from ..ising import io
from . import inputs, labels, span, store

PLAN = Path(__file__).with_name("ladder.toml")
SIDES = ("long", "short", "any")
SCHEMES = ("loqo", "forward")
DAY_NS = 24 * labels.HOUR_NS
QUANTILES = (0.025, 0.05, 0.975)
NEWTON_STEPS = 100
NEWTON_TOL = 1e-9


@dataclass(frozen=True)
class Rung:
    name: str
    groups: tuple[str, ...]
    own_only: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        out: list[str] = []
        for group in self.groups:
            out += inputs.GROUPS[group]
            if group not in self.own_only:
                out += [f"m_{name}" for name in inputs.GROUPS[group]]
        return tuple(out)


@dataclass(frozen=True)
class Plan:
    start: date
    end: date
    horizons: tuple[int, ...]
    sides: tuple[str, ...]
    without: tuple[str, ...]
    c: float
    clip: float
    gap_days: int
    forward_first: int
    block_days: int
    draws: int
    seed: int
    rungs: tuple[Rung, ...]
    comparisons: tuple[tuple[str, str], ...]
    external: tuple[str, Path, str] | None = None
    cuts: tuple[date, ...] | None = None

    def __post_init__(self) -> None:
        if self.cuts is None:
            return
        ordered = all(a < b for a, b in pairwise(self.cuts))
        if not (
            len(self.cuts) >= 3
            and ordered
            and self.cuts[0] == self.start
            and self.cuts[-1] == self.end + timedelta(days=1)
        ):
            raise ValueError(
                f"cuts {list(self.cuts)} must rise from the span's first day to the day after its "
                f"last"
            )

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c for rung in self.rungs for c in rung.columns))

    @property
    def names(self) -> tuple[str, ...]:
        extra = (self.external[0],) if self.external else ()
        return (*(rung.name for rung in self.rungs), *extra)

    def scored_names(self, scheme: str) -> tuple[str, ...]:
        extra = (self.external[0],) if self.external and self.external[2] == scheme else ()
        return (*(rung.name for rung in self.rungs), *extra)

    def first_fold(self, scheme: str) -> int:
        if self.cuts is not None and scheme != "forward":
            raise ValueError(f"a plan cut by code fits only forward, not {scheme}")
        return self.forward_first if scheme == "forward" else 0


def needed_gap_ns(horizons: Sequence[int]) -> int:
    longest = max(horizons, default=0) * labels.HOUR_NS
    return inputs.LOOKBACK_NS + longest + labels.LEAD_NS + labels.PAST_HOURS * labels.HOUR_NS


def load(path: Path = PLAN, text: str | None = None) -> Plan:
    raw = tomllib.loads(path.read_text(encoding="utf-8") if text is None else text)
    if "cuts" in raw or "cuts" in raw.get("folds", {}):
        raise ValueError(f"{path}: unknown or invalid: cuts (only code cuts a plan)")
    external = raw.get("external")
    plan = Plan(
        start=raw["start"],
        end=raw["end"],
        horizons=tuple(raw["horizons"]),
        sides=tuple(raw["sides"]),
        without=tuple(raw.get("without", ())),
        c=float(raw["model"]["c"]),
        clip=float(raw["model"]["clip"]),
        gap_days=int(raw["folds"]["gap_days"]),
        forward_first=int(raw["folds"]["forward_first"]),
        block_days=int(raw["bootstrap"]["block_days"]),
        draws=int(raw["bootstrap"]["draws"]),
        seed=int(raw["bootstrap"]["seed"]),
        rungs=tuple(
            Rung(r["name"], tuple(r["groups"]), tuple(r.get("own_only", ())))
            for r in raw.get("rungs", ())
        ),
        comparisons=tuple((c["first"], c["second"]) for c in raw.get("comparisons", ())),
        external=None
        if external is None
        else (external["name"], (path.parent / external["path"]).resolve(), external["scheme"]),
    )
    span.refuse(plan.start, plan.end)
    problems = [f"group {g}" for r in plan.rungs for g in r.groups if g not in inputs.GROUPS]
    problems += [
        f"group listed twice in {r.name}" for r in plan.rungs if len(set(r.groups)) < len(r.groups)
    ]
    problems += [
        f"own-only {g} of {r.name}" for r in plan.rungs for g in r.own_only if g not in r.groups
    ]
    problems += [f"side {s}" for s in plan.sides if s not in SIDES]
    problems += [f"horizon {h}" for h in plan.horizons if h not in labels.HORIZONS]
    problems += [f"comparison name {n}" for c in plan.comparisons for n in c if n not in plan.names]
    if plan.external is not None and plan.external[2] not in SCHEMES:
        problems.append(f"external scheme {plan.external[2]}")
    if not plan.rungs:
        problems.append("an empty rung list")
    problems += [f"no groups in {r.name}" for r in plan.rungs if not r.groups]
    for name, listed in (("horizons", plan.horizons), ("sides", plan.sides)):
        if not listed or len(set(listed)) < len(listed):
            problems.append(f"{name} = {list(listed)} (must be non-empty and unique)")
    for name, value in (("c", plan.c), ("clip", plan.clip)):
        if not (np.isfinite(value) and value > 0):
            problems.append(f"{name} = {value} (must be finite and above 0)")
    for name, value in (("block_days", plan.block_days), ("draws", plan.draws)):
        if value <= 0:
            problems.append(f"{name} = {value} (must be above 0)")
    if plan.gap_days < 0:
        problems.append(f"gap_days = {plan.gap_days} (must not be negative)")
    elif plan.gap_days * DAY_NS < needed_gap_ns(plan.horizons):
        days = needed_gap_ns(plan.horizons) / DAY_NS
        problems.append(
            f"gap_days = {plan.gap_days}, shorter than the {days:.2f} days that the inputs' "
            "look-back, the longest horizon and the labels' lead and look-back need"
        )
    if len(set(plan.names)) != len(plan.names) or not plan.start <= plan.end:
        problems.append("names must be unique and the span must not end before it starts")
    elif not 1 <= plan.forward_first < len(bounds(plan)) - 1:
        problems.append(
            f"forward_first = {plan.forward_first} (must name a quarter after the first)"
        )
    if problems:
        raise ValueError(f"{path}: unknown or invalid: {', '.join(problems)}")
    return plan


def bounds(plan: Plan) -> np.ndarray:
    if plan.cuts is not None:
        return np.array([labels.ns(d) for d in plan.cuts], np.int64)
    edges = [plan.start]
    q = date(plan.start.year, 3 * ((plan.start.month - 1) // 3) + 1, 1)
    while True:
        q = date(q.year + (q.month == 10), (q.month + 2) % 12 + 1, 1)
        if q > plan.end:
            break
        edges.append(q)
    edges.append(plan.end + timedelta(days=1))
    return np.array([labels.ns(d) for d in edges], np.int64)


def folds(ts: np.ndarray, edges: np.ndarray, block_days: int) -> tuple[np.ndarray, np.ndarray]:
    fold = np.searchsorted(edges, ts, "right") - 1
    block = (ts - edges[fold]) // (block_days * DAY_NS)
    return fold, block


def blocks_per_fold(edges: np.ndarray, block_days: int) -> np.ndarray:
    return -(-np.diff(edges) // (block_days * DAY_NS))


def training(
    ts: np.ndarray, fold: np.ndarray, f: int, edges: np.ndarray, gap_days: int, scheme: str = "loqo"
) -> np.ndarray:
    gap = gap_days * DAY_NS
    if scheme == "forward":
        return ts < edges[f] - gap
    return (fold != f) & ((ts < edges[f] - gap) | (ts >= edges[f + 1] + gap))


def _probability(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-z))


def fit(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    n, d = x.shape
    design = np.hstack([x, np.ones((n, 1))])
    ridge = np.full(d + 1, 1.0 / c)
    ridge[-1] = 0.0
    beta = np.zeros(d + 1)
    share = float(np.mean(y))
    if share in (0.0, 1.0):
        raise ValueError("a fit needs positive and negative rows")
    beta[-1] = np.log(share / (1 - share))

    def loss(b: np.ndarray) -> float:
        z = design @ b
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(ridge * b * b))

    current = loss(beta)
    for _ in range(NEWTON_STEPS):
        p = _probability(design @ beta)
        grad = design.T @ (p - y) + ridge * beta
        hessian = (design * (p * (1 - p))[:, None]).T @ design + np.diag(ridge)
        step = np.linalg.solve(hessian, grad)
        decrement = float(grad @ step)
        if decrement / 2 <= NEWTON_TOL:
            break
        t = 1.0
        while True:
            trial = beta - t * step
            value = loss(trial)
            if value <= current - 1e-4 * t * decrement or t < 1e-10:
                break
            t /= 2
        beta, current = trial, value
    return beta


def predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return _probability(x @ beta[:-1] + beta[-1])


def standardise(train: np.ndarray, test: np.ndarray, clip: float) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return np.clip((train - mu) / sd, -clip, clip), np.clip((test - mu) / sd, -clip, clip)


def block_counts(
    label: np.ndarray, block: np.ndarray, nblocks: int
) -> tuple[np.ndarray, np.ndarray]:
    p = np.bincount(block[label], minlength=nblocks).astype(np.float64)
    n = np.bincount(block[~label], minlength=nblocks).astype(np.float64)
    return p, n


def pair_counts(
    score: np.ndarray, label: np.ndarray, block: np.ndarray, nblocks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive, negative = label, ~label
    p, n = block_counts(label, block, nblocks)
    counts = np.zeros((nblocks, nblocks))
    sp, bp = score[positive], block[positive]
    for b in range(nblocks):
        neg = np.sort(score[negative & (block == b)])
        below = np.searchsorted(neg, sp, "left")
        tied = np.searchsorted(neg, sp, "right") - below
        counts[:, b] = np.bincount(bp, weights=below + 0.5 * tied, minlength=nblocks)
    return counts, p, n


def block_auc(
    counts: np.ndarray, p: np.ndarray, n: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    w = np.ones((1, len(p))) if weights is None else weights
    pairs = np.einsum("da,ab,db->d", w, counts, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((w @ p) * (w @ n) > 0, pairs / ((w @ p) * (w @ n)), np.nan)


def draws(nblocks: np.ndarray, count: int, seed: int, horizon: int) -> list[np.ndarray]:
    out = []
    for f, k in enumerate(nblocks):
        rng = np.random.default_rng([seed, horizon, f])
        out.append(rng.multinomial(int(k), np.full(int(k), 1.0 / k), size=count).astype(np.float64))
    return out


def within_coin(
    score: np.ndarray, label: np.ndarray, symbol: np.ndarray, fold: np.ndarray
) -> float:
    total = weight = 0.0
    for s in np.unique(symbol):
        for f in np.unique(fold):
            cell = (symbol == s) & (fold == f)
            positives = int(label[cell].sum())
            if positives and positives < cell.sum():
                total += positives * auc(score[cell & label], score[cell & ~label])
                weight += positives
    return total / weight if weight else float("nan")


@dataclass
class Horizon:
    scheme: str
    first: int
    ts: np.ndarray
    symbol: np.ndarray
    labels: dict[str, np.ndarray]
    fold: np.ndarray
    block: np.ndarray
    scores: dict[str, dict[str, np.ndarray]]


def select(
    values: Mapping[str, np.ndarray], grid: Mapping[str, np.ndarray], plan: Plan, horizon: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    ts = grid["ts_ns"]
    long, short = grid[f"long_{horizon}"], grid[f"short_{horizon}"]
    missing = np.zeros(len(ts), bool)
    for name in plan.columns:
        missing |= ~np.isfinite(values[name])
    edges = (labels.ns(plan.start), labels.ns(plan.end + timedelta(days=1)))
    steps = [
        ("outside the plan's span", (ts < edges[0]) | (ts >= edges[1])),
        ("skipped", grid["skip"]),
        (f"censored at {horizon} h", grid[f"censored_{horizon}"]),
        ("missing an input", missing),
    ]
    keep = np.ones(len(ts), bool)
    out = [_count("grid", keep, long, short)]
    for name, removed in steps:
        out.append(_count(name, keep & removed, long, short))
        keep &= ~removed
    out.append(_count("scored", keep, long, short))
    return keep, out


def _count(name: str, mask: np.ndarray, long: np.ndarray, short: np.ndarray) -> dict[str, Any]:
    return {
        "condition": name,
        "rows": int(mask.sum()),
        "long": int((mask & long).sum()),
        "short": int((mask & short).sum()),
        "positives": int((mask & (long | short)).sum()),
    }


def out_of_fold(
    x: np.ndarray,
    y: Mapping[str, np.ndarray],
    ts: np.ndarray,
    fold: np.ndarray,
    edges: np.ndarray,
    plan: Plan,
    scheme: str,
    horizon: int,
    log: Callable[[str], None],
) -> dict[str, dict[str, np.ndarray]]:
    configs = [(r.name, plan.c) for r in plan.rungs]
    scored = penalised(x, y, ts, fold, edges, plan, scheme, horizon, log, configs)
    return {name: sides for (name, _), sides in scored.items()}


def penalised(
    x: np.ndarray,
    y: Mapping[str, np.ndarray],
    ts: np.ndarray,
    fold: np.ndarray,
    edges: np.ndarray,
    plan: Plan,
    scheme: str,
    horizon: int,
    log: Callable[[str], None],
    configs: Sequence[tuple[str, float]],
) -> dict[tuple[str, float], dict[str, np.ndarray]]:
    index = {name: i for i, name in enumerate(plan.columns)}
    rungs = {r.name: r for r in plan.rungs}
    first = plan.first_fold(scheme)
    out = {k: {s: np.full(len(ts), np.nan) for s in ("long", "short")} for k in configs}
    for f in range(first, len(edges) - 1):
        test = fold == f
        if not test.any():
            continue
        train = training(ts, fold, f, edges, plan.gap_days, scheme)
        if not train.any():
            raise ValueError(f"{scheme} fold {f}: a scored quarter has no training rows")
        for side in ("long", "short"):
            share = float(np.mean(y[side][train]))
            if share in (0.0, 1.0):
                quarter = str(np.datetime64(int(edges[f]), "ns"))[:10]
                raise ValueError(
                    f"{scheme}, quarter from {quarter}, {horizon} h, {side}: the training rows "
                    f"hold {'no' if share == 0.0 else 'only'} positives"
                )
        z_train, z_test = standardise(x[train], x[test], plan.clip)
        for name, c in configs:
            cols = [index[column] for column in rungs[name].columns]
            for side in ("long", "short"):
                beta = fit(z_train[:, cols], y[side][train].astype(np.float64), c)
                out[(name, c)][side][test] = predict(z_test[:, cols], beta)
        log(f"  {scheme} fold {f + 1}: {int(train.sum())} training rows, {int(test.sum())} scored")
    scored = fold >= first
    for (name, _), sides in out.items():
        for side in ("long", "short"):
            sides[side] = sides[side][scored]
            if not np.isfinite(sides[side]).all():
                raise RuntimeError(f"{scheme} {name} {side}: a scored row has no finite score")
        sides["any"] = 1 - (1 - sides["long"]) * (1 - sides["short"])
    return out


def _text(kind: pa.DataType) -> bool:
    if pa.types.is_dictionary(kind):
        return _text(kind.value_type)
    return pa.types.is_string(kind) or pa.types.is_large_string(kind)


def _hourly(kind: pa.DataType) -> bool:
    return pa.types.is_integer(kind) or pa.types.is_timestamp(kind)


EXTERNAL: dict[str, Callable[[pa.DataType], bool]] = {
    "coin": _text,
    "hour": _hourly,
    "horizon": pa.types.is_integer,
    "side": _text,
    "score": pa.types.is_floating,
    "fold": pa.types.is_integer,
    "scheme": _text,
}


def external_columns(schema: pa.Schema) -> None:
    for name, allowed in EXTERNAL.items():
        if name not in schema.names:
            raise ValueError(f"the external scores have no {name} column")
        if not allowed(schema.field(name).type):
            raise ValueError(
                f"the external scores' {name} column is {schema.field(name).type}, not a type "
                f"the ladder takes"
            )


def hour_ns(hours: pa.ChunkedArray) -> pa.ChunkedArray:
    if pa.types.is_timestamp(hours.type):
        hours = hours.cast(pa.timestamp("ns", hours.type.tz))
    return hours.cast(pa.int64())


def external_table(path: Path, data: bytes | None = None) -> pa.Table:
    def source() -> Any:
        return path if data is None else pa.BufferReader(data)

    external_columns(pq.read_schema(source()))
    hours = pq.read_table(source(), columns=["hour"])["hour"]
    if hours.null_count:
        raise ValueError(f"the external scores leave {hours.null_count} hours null")
    ns = hour_ns(hours).to_numpy()
    if ns.size and int(ns.max()) >= labels.ns(span.HELD_BACK):
        raise span.HeldBackError("the external scores hold an hour in the held-back test span")
    return pq.read_table(source())


def external_scores(
    table: pa.Table,
    ts: np.ndarray,
    symbol: np.ndarray,
    fold: np.ndarray,
    scheme: str,
    horizon: int,
    sides: Sequence[str],
) -> dict[str, np.ndarray]:
    external_columns(table.schema)
    schemes = table["scheme"].cast(pa.string()).unique().to_pylist()
    if schemes != [scheme]:
        raise ValueError(f"the external scores were fitted under {schemes}, not {scheme}")
    found = pa.table(
        {
            "coin": table["coin"].cast(pa.string()),
            "hour": hour_ns(table["hour"]),
            "horizon": table["horizon"].cast(pa.int64()),
            "side": table["side"].cast(pa.string()),
            "score": table["score"].cast(pa.float64()),
            "fold": table["fold"].cast(pa.int64()),
        }
    )
    rows = pa.table({"coin": symbol.astype(str), "hour": ts, "row": np.arange(len(ts))})
    at_horizon = found["horizon"].to_numpy() == horizon
    named = found["side"].to_numpy(zero_copy_only=False)
    out = {}
    for side in sides:
        mine = found.filter(pa.array(at_horizon & (named == side)))
        mine = mine.select(["coin", "hour", "score", "fold"])
        joined = rows.join(mine, keys=["coin", "hour"], join_type="left outer")
        if joined.num_rows != len(ts):
            raise ValueError(f"the external scores give a row twice ({horizon} h, {side})")
        joined = joined.sort_by("row")
        score = joined["score"].to_numpy(zero_copy_only=False).astype(np.float64)
        if not np.isfinite(score).all():
            raise ValueError(
                f"the external scores miss or give a non-finite score for "
                f"{int((~np.isfinite(score)).sum())} scored rows ({horizon} h, {side})"
            )
        given = joined["fold"].to_numpy(zero_copy_only=False)
        if not np.array_equal(given, fold):
            raise ValueError(
                f"the external scores give {int((given != fold).sum())} scored rows a fold other "
                f"than the ladder's ({horizon} h, {side})"
            )
        out[side] = score
    return out


def horizon_rows(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    plan: Plan,
    horizon: int,
    external: pa.Table | None,
    log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], list[Horizon]]:
    keep, accounting = select(values, grid, plan, horizon)
    ts, symbol = grid["ts_ns"][keep], grid["symbol"][keep]
    y = {"long": grid[f"long_{horizon}"][keep], "short": grid[f"short_{horizon}"][keep]}
    y["any"] = y["long"] | y["short"]
    edges = bounds(plan)
    fold, block = folds(ts, edges, plan.block_days)
    x = np.column_stack([values[name][keep] for name in plan.columns])
    out = []
    for scheme in SCHEMES:
        first = plan.first_fold(scheme)
        mine = fold >= first
        scores = out_of_fold(x, y, ts, fold, edges, plan, scheme, horizon, log)
        if external is not None and plan.external is not None and plan.external[2] == scheme:
            scores[plan.external[0]] = external_scores(
                external, ts[mine], symbol[mine], fold[mine], scheme, horizon, plan.sides
            )
        labelled = {side: v[mine] for side, v in y.items()}
        out.append(
            Horizon(
                scheme, first, ts[mine], symbol[mine], labelled, fold[mine], block[mine], scores
            )
        )
    return accounting, out


def evaluate(
    h: Horizon,
    plan: Plan,
    horizon: int,
    names: Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    kept: dict[str, Any] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    edges = bounds(plan)
    nblocks = blocks_per_fold(edges, plan.block_days)
    weights = draws(nblocks, plan.draws, plan.seed, horizon)
    quarters = range(h.first, len(nblocks))
    names = plan.scored_names(h.scheme) if names is None else tuple(names)
    pairs = plan.comparisons if pairs is None else tuple(pairs)
    subsets = {"all": np.ones(len(h.ts), bool), "without": ~np.isin(h.symbol, plan.without)}
    score_rows, fold_rows, primary, boot, drawn = [], [], {}, {}, {}
    complete: dict[str, float] = {}
    whole: dict[str, np.ndarray] = {}
    key = {"horizon": horizon, "scheme": h.scheme}
    for name in names:
        for side in plan.sides:
            s, label = h.scores[name][side], h.labels[side]
            for subset, mask in subsets.items():
                per_fold = np.full(len(quarters), np.nan)
                positives = np.zeros(len(quarters))
                booted = np.zeros((plan.draws, len(quarters)))
                for j, f in enumerate(quarters):
                    inside = mask & (h.fold == f)
                    counts = pair_counts(s[inside], label[inside], h.block[inside], int(nblocks[f]))
                    per_fold[j] = block_auc(*counts)[0]
                    positives[j] = label[inside].sum()
                    if subset == "all":
                        booted[:, j] = block_auc(*counts, weights[f])
                    fold_rows.append(
                        {
                            **key,
                            "side": side,
                            "name": name,
                            "coins": subset,
                            "fold": f,
                            "auc": float(per_fold[j]),
                            "rows": int(inside.sum()),
                            "positives": int(positives[j]),
                        }
                    )
                finite = np.isfinite(per_fold)
                if subset == "all":
                    primary[(name, side)] = per_fold
                    boot[(name, side)] = nanmean_rows(booted)
                    drawn[(name, side)] = booted
                    if name == names[0]:
                        whole[side] = np.isfinite(booted[:, finite]).all(axis=1) & finite.any()
                        complete[side] = float(whole[side].mean()) if finite.any() else np.nan
                weight = positives[finite].sum()
                score_rows.append(
                    {
                        **key,
                        "side": side,
                        "name": name,
                        "coins": subset,
                        "fold_mean": float(per_fold[finite].mean()) if finite.any() else np.nan,
                        "fold_mean_weighted": float(per_fold[finite] @ positives[finite] / weight)
                        if weight
                        else np.nan,
                        "folds": int(finite.sum()),
                        "pooled": auc(s[mask & label], s[mask & ~label]),
                        "within_coin": within_coin(
                            s[mask], label[mask], h.symbol[mask], h.fold[mask]
                        ),
                        "rows": int(mask.sum()),
                        "positives": int(label[mask].sum()),
                    }
                )
    comparisons = []
    for first, second in pairs:
        if first not in names or second not in names:
            continue
        for side in plan.sides:
            a, b = primary[(first, side)], primary[(second, side)]
            comparisons.append(
                {
                    **key,
                    "side": side,
                    "first": first,
                    "second": second,
                    **comparison(
                        float(np.nanmean(a) - np.nanmean(b)),
                        a,
                        b,
                        boot[(first, side)] - boot[(second, side)],
                        complete[side],
                    ),
                }
            )
    if kept is not None:
        kept |= {"folds": primary, "boot": boot, "booted": drawn, "complete": whole}
    return score_rows, fold_rows, comparisons


def comparison(
    estimate: float, a: np.ndarray, b: np.ndarray, difference: np.ndarray, complete: float
) -> dict[str, Any]:
    difference = difference[np.isfinite(difference)]
    both = np.isfinite(a) & np.isfinite(b)
    per_fold = a - b
    q = np.quantile(difference, QUANTILES) if difference.size else [np.nan] * 3
    spread = per_fold[both]
    return {
        "estimate": estimate,
        "se": float(np.std(difference, ddof=1)) if difference.size > 1 else np.nan,
        "se_quarters": float(np.std(spread, ddof=1) / np.sqrt(spread.size))
        if spread.size > 1
        else np.nan,
        "q025": float(q[0]),
        "q05": float(q[1]),
        "q975": float(q[2]),
        "above_zero": float(np.mean(difference > 0)) if difference.size else np.nan,
        "draws": int(difference.size),
        "complete_draws": complete,
        "fold_differences": [float(v) for v in per_fold],
        "folds_above_zero": int(np.sum(spread > 0)),
        "folds": int(both.sum()),
    }


def nanmean_rows(values: np.ndarray) -> np.ndarray:
    count = np.isfinite(values).sum(axis=1)
    total = np.where(np.isfinite(values), values, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def ladder(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    plan: Plan,
    external: pa.Table | None = None,
    log: Callable[[str], None] = lambda line: None,
) -> dict[str, Any]:
    if plan.external is not None and external is None:
        raise ValueError("the plan names external scores but none were given")
    out: dict[str, Any] = {
        "accounting": {},
        "scores": [],
        "folds": [],
        "comparisons": [],
        "oof": [],
    }
    for horizon in plan.horizons:
        log(f"{horizon} h")
        accounting, schemes = horizon_rows(values, grid, plan, horizon, external, log)
        out["accounting"][horizon] = accounting
        for h in schemes:
            scores, fold_rows, comparisons = evaluate(h, plan, horizon)
            out["scores"] += scores
            out["folds"] += fold_rows
            out["comparisons"] += comparisons
            oof = {
                "scheme": [h.scheme] * len(h.ts),
                "ts_ns": h.ts,
                "symbol": h.symbol.astype(str),
                "horizon": np.full(len(h.ts), horizon),
                "fold": h.fold,
                "long": h.labels["long"],
                "short": h.labels["short"],
            }
            for name in plan.scored_names(h.scheme):
                for side in plan.sides:
                    oof[f"{name}:{side}"] = h.scores[name][side]
            out["oof"].append(pa.table(oof))
    return out


def run(
    inputs_dir: Path,
    data_dir: Path,
    plan_path: Path,
    tag: str,
    log: Callable[[str], None] = lambda line: None,
) -> tuple[Path, dict[str, Any]]:
    began = time.perf_counter()
    plan = load(plan_path)
    manifest = io.read_json(inputs_dir / "manifest.json")
    params = manifest["params"]
    span.refuse(*(date.fromisoformat(params[k]) for k in ("start", "dev_start", "end")))
    complete = io.read_json(inputs_dir / "complete.json")
    if complete.get("clean") is not True:
        raise ValueError(f"{inputs_dir}: complete.json does not mark the run clean")
    labels_dir = data_dir / "labels" / params["labels_run"]
    grid_path = labels_dir / params["labels_grid"]["file"]
    table_path = inputs_dir / "inputs.parquet"
    store.refuse_held_back(grid_path)
    store.refuse_held_back(table_path)
    if plan.external is not None:
        store.refuse_held_back(plan.external[1], "hour")
    if store.sha256(grid_path) != params["labels_grid"]["sha256"]:
        raise ValueError(f"{grid_path}: not the grid the inputs run was made from")
    if store.sha256(table_path) != complete["files"].get(table_path.name):
        raise ValueError(f"{table_path}: its sha256 differs from complete.json's")
    grid_table = pq.read_table(grid_path)
    table = pq.read_table(table_path, columns=["ts_ns", "symbol", *plan.columns])
    if not (
        table["ts_ns"].equals(grid_table["ts_ns"]) and table["symbol"].equals(grid_table["symbol"])
    ):
        raise ValueError("the inputs' rows are not the grid's rows")
    external = None
    if plan.external is not None:
        external = external_table(plan.external[1])
    values = {n: table[n].to_numpy() for n in plan.columns}
    grid = {n: grid_table[n].to_numpy(zero_copy_only=False) for n in grid_table.column_names}
    out = ladder(values, grid, plan, external, log)
    seconds = time.perf_counter() - began
    plan_text = plan_path.read_text(encoding="utf-8")
    run_params = {
        "inputs_run": manifest["run_id"],
        "labels_run": params["labels_run"],
        "plan": {"file": plan_path.name, "sha256": store.sha256(plan_path)},
        "external": None
        if plan.external is None
        else {
            "name": plan.external[0],
            "file": plan.external[1].name,
            "sha256": store.sha256(plan.external[1]),
            "scheme": plan.external[2],
        },
        "code": {"uv_lock_sha256": labels.uv_lock()},
    }
    tables = {
        "oof": pa.concat_tables(out["oof"], promote_options="default"),
        "scores": pa.Table.from_pylist(out["scores"]),
        "folds": pa.Table.from_pylist(out["folds"]),
        "comparisons": pa.Table.from_pylist(out["comparisons"]),
        "rows": pa.Table.from_pylist(
            [{"horizon": h, **row} for h, rows in out["accounting"].items() for row in rows]
        ),
    }
    checks = {
        "seconds": round(seconds, 1),
        "accounting": {str(k): v for k, v in out["accounting"].items()},
    }
    text = report(out, plan, manifest["run_id"], params["labels_run"], plan_path.name, seconds)
    texts = {"report.md": text, "plan.toml": plan_text}
    run_dir = store.write(data_dir, "panel-ladder", tag, tables, run_params, checks, texts)
    return run_dir, out | {"seconds": seconds}


def _f(value: float, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def report(
    out: Mapping[str, Any],
    plan: Plan,
    inputs_run: str,
    labels_run: str,
    plan_name: str,
    seconds: float,
) -> str:
    gap = needed_gap_ns(plan.horizons) / DAY_NS
    lines = [
        "# Panel ladder",
        "",
        f"Inputs run `{inputs_run}`, labels run `{labels_run}`, plan `{plan_name}`; "
        f"{plan.start} to {plan.end}. Each rung is two L2 logistic regressions (C = {plan.c}), one "
        f"for long_H and one for short_H, on inputs standardised by the training rows and clipped "
        f"to +-{plan.clip:g}; the any score is 1 - (1 - p_long)(1 - p_short). Folds are the "
        f"calendar quarters of the span, scored two ways. In loqo each quarter is left out in "
        f"turn and training drops rows within {plan.gap_days} days of it on either side. In "
        f"forward a quarter is scored by a model fitted only on rows more than {plan.gap_days} "
        f"days before it, from quarter {plan.forward_first + 1} on, which previews scoring the "
        f"held-back test span. The gap covers the inputs' longest look-back, the longest horizon "
        f"and the labels' lead and 24-hour look-back ({gap:.2f} days). Every rung of a horizon and "
        f"scheme is scored on the same rows.",
        "",
        "Scores are AUCs and rank rows; they say nothing about calibration. Fold mean is the mean "
        "over quarters of the AUC inside each quarter (the primary score), weighted is the same "
        "mean weighted by each quarter's positives, pooled is the AUC of all out-of-fold scores "
        "together, and within coin is the AUC inside each coin and quarter weighted by its "
        "positives. The second value of each pair leaves "
        f"out {', '.join(plan.without)}. The eight-quarter loqo score is retrospective on a fixed "
        f"coin cohort.",
        "",
        f"Comparisons are first minus second in the primary score. The interval comes from a joint "
        f"block bootstrap: within each quarter, {plan.block_days}-day blocks of calendar time are "
        f"drawn with replacement, every coin's rows of a drawn block taken together, "
        f"{plan.draws} draws, seed {plan.seed}, the same draws for every rung and scheme. It is "
        f"conditional on the fitted models and the observed quarters. q2.5, q5 and q97.5 are "
        f"percentiles of the bootstrap differences; > 0 is the share of draws above zero; se q is "
        f"the standard deviation of the per-quarter differences over the square root of their "
        f"number, the between-quarter view; folds > 0 counts quarters where the first beats the "
        f"second. A draw that takes no positive or no negative in a quarter cannot score that "
        f"quarter, so its fold mean averages over the quarters it does score; complete is the "
        f"share of draws that score every quarter, the same for every comparison of a side and "
        f"scheme.",
        "",
        f"Wall time {_f(seconds / 60)} minutes.",
    ]
    rungs = {r.name: r for r in plan.rungs}
    lines += ["", "## Rungs", "", "| rung | groups | inputs |", "|---|---|---|"]
    for name in plan.names:
        if name in rungs:
            r = rungs[name]
            groups = ", ".join(g + (" (own only)" if g in r.own_only else "") for g in r.groups)
            lines.append(f"| {name} | {groups} | {len(r.columns)} |")
        else:
            assert plan.external is not None
            lines.append(f"| {name} | external out-of-fold scores ({plan.external[2]}) | - |")
    for horizon in plan.horizons:
        lines += ["", f"## {horizon} h", "", "Rows removed, in order:", ""]
        lines += ["| condition | rows | long | short | positives |", "|---|---|---|---|---|"]
        for row in out["accounting"][horizon]:
            lines.append(
                f"| {row['condition']} | {row['rows']} | {row['long']} | {row['short']} | "
                f"{row['positives']} |"
            )
        for scheme in SCHEMES:
            scores = [s for s in out["scores"] if s["horizon"] == horizon and s["scheme"] == scheme]
            lines += [
                "",
                f"### {scheme}",
                "",
                "| rung | side | fold mean | weighted | pooled | within coin | folds |",
                "|---|---|---|---|---|---|---|",
            ]
            for side in plan.sides:
                for name in plan.scored_names(scheme):
                    pair = {
                        s["coins"]: s for s in scores if s["side"] == side and s["name"] == name
                    }
                    a, w = pair["all"], pair["without"]
                    cells = [
                        f"{_f(a[k])} / {_f(w[k])}"
                        for k in ("fold_mean", "fold_mean_weighted", "pooled", "within_coin")
                    ]
                    lines.append(f"| {name} | {side} | {' | '.join(cells)} | {a['folds']} |")
            lines += [
                "",
                "| comparison | side | estimate | se | se q | q2.5 | q5 | q97.5 | > 0 | complete "
                "| folds > 0 |",
                "|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for c in out["comparisons"]:
                if c["horizon"] == horizon and c["scheme"] == scheme:
                    lines.append(
                        f"| {c['first']} - {c['second']} | {c['side']} | "
                        f"{_f(c['estimate'], True)} | {_f(c['se'])} | {_f(c['se_quarters'])} | "
                        f"{_f(c['q025'], True)} | {_f(c['q05'], True)} | {_f(c['q975'], True)} | "
                        f"{_f(c['above_zero'])} | {_f(c['complete_draws'])} | "
                        f"{c['folds_above_zero']} of {c['folds']} |"
                    )
    return "\n".join(lines) + "\n"
