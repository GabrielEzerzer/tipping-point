# The classical comparators of the sequence models (docs/market.md, "Coin panel"): the ladder's
# rungs over the probes' penalty grid and a logistic on plain statistics of the same 336-hour
# window, on the sequence plan's folds and rows, one configuration per name by the models' rule.

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..ews.scoring import auc
from . import labels, ladder, sequence, store

MODELS = ("rung", "window")
FOLDS = pa.schema(
    [
        ("horizon", pa.int64()),
        ("side", pa.string()),
        ("model", pa.string()),
        ("readout", pa.string()),
        ("c", pa.float64()),
        ("fold", pa.int64()),
        ("scheme", pa.string()),
        ("seed", pa.int64()),
        ("variant", pa.string()),
        ("auc", pa.float64()),
        ("rows", pa.int64()),
        ("positives", pa.int64()),
    ]
)
METRIC = "mean of the long and short 24 h forward fold-mean AUCs"


@dataclass(frozen=True)
class Config:
    model: str
    variant: str
    readout: str | None
    c: float


def configs(plan: sequence.Plan) -> list[Config]:
    out = [Config("rung", name, None, c) for name in plan.rungs for c in plan.cs]
    out += [
        Config("window", variant, readout, c)
        for variant in plan.variants
        for readout in plan.readouts
        for c in plan.cs
    ]
    return out


def names(plan: sequence.Plan) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*plan.rungs, *plan.variants)))


def check(config: Config, plan: sequence.Plan) -> None:
    if config not in configs(plan):
        raise ValueError(f"the plan has no classical configuration {asdict(config)}")


def selectable(plan: sequence.Plan) -> None:
    has = (
        "forward" in plan.schemes
        and 24 in plan.ladder.horizons
        and {"long", "short"} <= set(plan.ladder.sides)
    )
    if not has:
        raise ValueError(
            "choosing a configuration needs the forward scheme, the 24 h horizon and the long "
            "and short sides; give --fixed for a plan without them"
        )


def spans(plan: sequence.Plan) -> tuple[int, ...]:
    return (*(n * plan.shape.patch for n in sequence.SPANS), plan.window)


def statistics(
    x: np.ndarray, coin: np.ndarray, hour: np.ndarray, plan: sequence.Plan
) -> np.ndarray:
    nhours, channels = x.shape[1] - plan.pad, x.shape[2]
    out = np.empty((len(coin), sequence.READOUTS["spreads"] * channels), np.float32)
    for k in np.unique(coin):
        rows = np.flatnonzero(coin == k)
        t = hour[rows]
        series = x[k].astype(np.float64)
        means, spreads = [], []
        for n in spans(plan):
            start = plan.pad + 1 - n
            total = np.zeros((nhours, channels))
            for j in range(start, start + n):
                total += series[j : j + nhours]
            mean = total / n
            squares = np.zeros((nhours, channels))
            for j in range(start, start + n):
                squares += (series[j : j + nhours] - mean) ** 2
            means.append(mean[t])
            spreads.append(np.sqrt(squares[t] / n))
        out[rows] = np.concatenate([series[plan.pad + t], *means, *spreads], axis=1)
    return out


@dataclass
class Panel:
    hours: np.ndarray
    ncoins: int
    ts: np.ndarray
    symbol: np.ndarray
    edges: np.ndarray
    rows: dict[int, sequence.Rows]
    row_fold: np.ndarray
    kept: np.ndarray


def scorable(plan: sequence.Plan) -> None:
    sequence.check_folds(plan)
    if sequence.fit_only(plan):
        raise ValueError("the plan holds only the fit-only fold, which scores nothing")


def panel(
    values: Mapping[str, np.ndarray], grid: Mapping[str, np.ndarray], plan: sequence.Plan
) -> Panel:
    scorable(plan)
    hours, coins = sequence.layout(grid)
    edges = ladder.bounds(plan.ladder)
    rows = sequence.common_rows(values, grid, plan)
    ts = grid["ts_ns"]
    row_fold, _ = ladder.folds(ts, edges, plan.ladder.block_days)
    kept = np.any([r.keep for r in rows.values()], axis=0)
    for scheme in plan.schemes:
        first = plan.ladder.first_fold(scheme)
        quarters = sorted({int(f) for r in rows.values() for f in np.unique(r.fold) if f >= first})
        missing = [f + 1 for f in quarters if f not in plan.folds]
        if missing:
            raise ValueError(f"the plan's folds leave {scheme} quarters {missing} unscored")
    symbol = grid["symbol"].astype(str)
    return Panel(hours, len(coins), ts, symbol, edges, rows, row_fold, kept)


def rung_scores(
    values: Mapping[str, np.ndarray],
    p: Panel,
    plan: sequence.Plan,
    chosen: Sequence[Config],
    log: sequence.LOG,
) -> Iterator[tuple[int, str, dict[Config, dict[str, np.ndarray]]]]:
    pairs = [(config.variant, config.c) for config in chosen]
    for h, r in p.rows.items():
        x = np.column_stack([values[c][r.keep] for c in plan.ladder.columns])
        for scheme in plan.schemes:
            log(f"  rungs, {scheme}, {h} h")
            scored = ladder.penalised(
                x, r.y, p.ts[r.keep], r.fold, p.edges, plan.ladder, scheme, h, log, pairs
            )
            yield h, scheme, {config: scored[(config.variant, config.c)] for config in chosen}


def window_scores(
    values: Mapping[str, np.ndarray],
    p: Panel,
    plan: sequence.Plan,
    variant: str,
    chosen: Sequence[Config],
    log: sequence.LOG,
    fitted: dict[tuple[str, int], str],
) -> Iterator[tuple[str, int, int, np.ndarray, dict[Config, dict[str, np.ndarray]]]]:
    raw = sequence.by_coin(
        np.column_stack([values[c] for c in plan.rung(variant).columns]), p.ncoins
    )
    channels = raw.shape[-1] + 1
    pairs = [(config.readout or "", config.c) for config in chosen]
    for scheme in plan.schemes:
        for f in sorted(plan.folds):
            if f < plan.ladder.first_fold(scheme):
                continue
            train_hours = sequence.training(p.hours, plan, f, scheme)
            fit_hours = sequence.eligible(train_hours, plan.pad, plan.reach)
            if not fit_hours.any():
                raise ValueError(f"window {variant} {scheme} fold {f + 1}: no training row")
            log(f"  window {variant}, {scheme} fold {f + 1}")
            fit_rows = np.repeat(fit_hours, p.ncoins)
            mu, sd = sequence.statistics(raw, train_hours)
            x = sequence.standardised(raw, mu, sd, plan.ladder.clip, plan.pad)
            need_train, need_test = p.kept & fit_rows, p.kept & (p.row_fold == f)
            at_train, at_test = np.cumsum(need_train) - 1, np.cumsum(need_test) - 1
            train_coin, train_hour = sequence._coin_hour(need_train, p.ncoins)
            test_coin, test_hour = sequence._coin_hour(need_test, p.ncoins)
            both = statistics(
                x,
                np.concatenate([train_coin, test_coin]),
                np.concatenate([train_hour, test_hour]),
                plan,
            )
            del x
            train_f, test_f = both[: len(train_coin)], both[len(train_coin) :]
            digest = hashlib.sha256(mu.tobytes() + sd.tobytes())
            digest.update(train_f.tobytes())
            for h, r in p.rows.items():
                test = r.fold == f
                train = fit_rows[r.keep]
                if not test.any():
                    continue
                if not train.any():
                    raise ValueError(f"window {variant} {scheme} fold {f + 1} {h} h: no probe row")
                index = np.flatnonzero(r.keep)
                a, b = at_train[index[train]], at_test[index[test]]
                y_train = {side: r.y[side][train] for side in ("long", "short")}
                for side, ys in y_train.items():
                    share = float(np.mean(ys))
                    if share in (0.0, 1.0):
                        raise ValueError(
                            f"window {variant} {scheme} fold {f + 1} {h} h {side}: the training "
                            f"rows hold {'no' if share == 0.0 else 'only'} positives"
                        )
                probed = sequence.probe(
                    train_f[a], test_f[b], y_train, plan, digest, pairs, channels
                )
                yield scheme, f, h, test, {c: probed[k] for c, k in zip(chosen, pairs, strict=True)}
            fitted[(scheme, f)] = digest.hexdigest()


def _row(
    h: int, side: str, config: Config, scheme: str, f: int, s: np.ndarray, label: np.ndarray
) -> dict[str, Any]:
    return {
        "horizon": h,
        "side": side,
        "model": config.model,
        "readout": config.readout,
        "c": config.c,
        "fold": f,
        "scheme": scheme,
        "seed": 0,
        "variant": config.variant,
        "auc": auc(s[label], s[~label]),
        "rows": len(s),
        "positives": int(label.sum()),
    }


def order(plan: sequence.Plan) -> Any:
    known = names(plan)

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            MODELS.index(row["model"]),
            known.index(row["variant"]),
            -1 if row["readout"] is None else plan.readouts.index(row["readout"]),
            row["c"],
            plan.schemes.index(row["scheme"]),
            plan.ladder.horizons.index(row["horizon"]),
            plan.ladder.sides.index(row["side"]),
            row["fold"],
        )

    return key


def classical(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    plan: sequence.Plan,
    chosen: Sequence[Config],
    log: sequence.LOG = lambda line: None,
) -> dict[str, Any]:
    p = panel(values, grid, plan)
    folds: list[dict[str, Any]] = []
    fitted: dict[str, str] = {}
    rungs = [c for c in chosen if c.model == "rung"]
    if rungs:
        for h, scheme, scored in rung_scores(values, p, plan, rungs, log):
            r = p.rows[h]
            first = plan.ladder.first_fold(scheme)
            fold = r.fold[r.fold >= first]
            for config, sides in scored.items():
                for side in plan.ladder.sides:
                    label = r.y[side][r.fold >= first]
                    for f in plan.folds:
                        test = fold == f
                        if test.any():
                            folds.append(
                                _row(h, side, config, scheme, f, sides[side][test], label[test])
                            )
    for variant in plan.variants:
        mine = [c for c in chosen if c.model == "window" and c.variant == variant]
        if not mine:
            continue
        found: dict[tuple[str, int], str] = {}
        for scheme, f, h, test, got in window_scores(values, p, plan, variant, mine, log, found):
            r = p.rows[h]
            for config, sides in got.items():
                for side in plan.ladder.sides:
                    folds.append(_row(h, side, config, scheme, f, sides[side], r.y[side][test]))
        fitted |= {f"{variant} {scheme} fold {f + 1}": v for (scheme, f), v in found.items()}
    folds.sort(key=order(plan))
    return {
        "folds": folds,
        "accounting": {h: r.accounting for h, r in p.rows.items()},
        "fitted": fitted,
    }


def choose(
    folds: Sequence[Mapping[str, Any]], name: str
) -> tuple[Config, float, list[dict[str, Any]]]:
    readouts = list(sequence.READOUTS)

    def key(r: Mapping[str, Any]) -> tuple[int, int, float] | None:
        if r["variant"] != name or r["model"] not in MODELS:
            return None
        index = -1 if r["readout"] is None else readouts.index(r["readout"])
        return MODELS.index(r["model"]), index, r["c"]

    def config(k: tuple[int, int, float]) -> Config:
        return Config(MODELS[k[0]], name, None if k[1] < 0 else readouts[k[1]], k[2])

    scored: dict[Any, dict[str, dict[int, list[float]]]] = {}
    for r in folds:
        k = key(r)
        mine = (r["scheme"], r["horizon"]) == ("forward", 24) and r["side"] in ("long", "short")
        if mine and k is not None:
            per = scored.setdefault(k, {}).setdefault(r["side"], {})
            per.setdefault(r["fold"], []).append(r["auc"])
    covered = {
        k: tuple(
            tuple(sorted(f for f, v in sides.get(side, {}).items() if np.isfinite(np.mean(v))))
            for side in ("long", "short")
        )
        for k, sides in scored.items()
    }
    if len(set(covered.values())) > 1:
        seen = sorted(
            {
                (tuple(f + 1 for f in long), tuple(f + 1 for f in short))
                for long, short in covered.values()
            }
        )
        raise ValueError(
            f"the {name} candidates are not scored on the same folds (long, short): {seen}"
        )
    candidates = sequence.ranked(folds, key)
    found = sequence.best(candidates)
    if found is None:
        raise ValueError(f"no forward 24 h fold scores of the {name} comparators to choose from")
    listed = [
        asdict(config(k)) | {"score": score if np.isfinite(score) else None}
        for k, score in candidates
    ]
    return config(found[0]), found[1], listed


def finite(scores: np.ndarray, scheme: str, h: int, side: str) -> None:
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{scheme} {h} h {side}: a score is not finite")


def export(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    plan: sequence.Plan,
    config: Config,
    log: sequence.LOG = lambda line: None,
) -> dict[str, pa.Table]:
    check(config, plan)
    p = panel(values, grid, plan)
    parts: dict[str, list[pa.Table]] = {scheme: [] for scheme in plan.schemes}
    if config.model == "rung":
        for h, scheme, scored in rung_scores(values, p, plan, [config], log):
            r = p.rows[h]
            mine = r.fold >= plan.ladder.first_fold(scheme)
            for side in plan.ladder.sides:
                finite(scored[config][side], scheme, h, side)
                parts[scheme].append(
                    sequence.scores_part(
                        p.symbol[r.keep],
                        p.ts[r.keep],
                        r.fold,
                        mine,
                        h,
                        side,
                        scored[config],
                        scheme,
                    )
                )
    else:
        total = {
            (scheme, h): np.full((int(r.keep.sum()), 2), np.nan)
            for scheme in plan.schemes
            for h, r in p.rows.items()
        }
        for scheme, _, h, test, got in window_scores(
            values, p, plan, config.variant, [config], log, {}
        ):
            total[(scheme, h)][test] = np.column_stack([got[config][s] for s in ("long", "short")])
        for scheme in plan.schemes:
            for h, r in p.rows.items():
                mine = r.fold >= plan.ladder.first_fold(scheme)
                v = total[(scheme, h)][mine]
                if not np.isfinite(v).all():
                    raise RuntimeError(f"{scheme} {h} h: a scored row has no finite score")
                sides = {"long": v[:, 0], "short": v[:, 1]}
                sides["any"] = sequence.either(sides["long"], sides["short"])
                for side in plan.ladder.sides:
                    finite(sides[side], scheme, h, side)
                    parts[scheme].append(
                        sequence.scores_part(
                            p.symbol[r.keep], p.ts[r.keep], r.fold, mine, h, side, sides, scheme
                        )
                    )
    return {scheme: pa.concat_tables(tables) for scheme, tables in parts.items()}


def run(
    inputs_dir: Path,
    data_dir: Path,
    plan_path: Path,
    tag: str,
    fixed: Config | None = None,
    log: sequence.LOG = lambda line: None,
) -> tuple[Path, dict[str, Any]]:
    began = time.perf_counter()
    plan, records, (plan_text, ladder_text) = sequence.snapshot(plan_path)
    scorable(plan)
    if fixed is None:
        selectable(plan)
    else:
        check(fixed, plan)
    values, grid, source = sequence.read(inputs_dir, data_dir, plan)
    out = classical(values, grid, plan, configs(plan) if fixed is None else [fixed], log)
    selection: dict[str, Any] | None = None
    if fixed is None:
        exported: dict[str, Config] = {}
        selection = {"metric": METRIC, "chosen": {}}
        for name in names(plan):
            config, score, candidates = choose(out["folds"], name)
            exported[name] = config
            selection["chosen"][name] = asdict(config) | {"score": score}
            selection.setdefault("candidates", {})[name] = candidates
            log(f"  {name}: chose {asdict(config)}, {score:.4f}")
    else:
        exported = {fixed.variant: fixed}
    tables = {
        "folds": pa.Table.from_pylist(out["folds"], schema=FOLDS),
        "rows": pa.Table.from_pylist(
            [{"horizon": h, **row} for h, rs in out["accounting"].items() for row in rs]
        ),
    }
    for name, config in exported.items():
        for scheme, table in export(values, grid, plan, config, log).items():
            tables[f"{name}-{scheme}"] = table
    seconds = time.perf_counter() - began
    run_params = {
        **source,
        **records,
        "numpy": np.__version__,
        "runtime": sequence.runtime(),
        "code": {"uv_lock_sha256": labels.uv_lock()},
        "selection": selection,
        "fixed": None if fixed is None else asdict(fixed),
        "exported": {name: asdict(config) for name, config in exported.items()},
        "fitted": out["fitted"],
    }
    texts = {
        "report.md": report(out, plan, source, exported, selection, seconds),
        "plan.toml": plan_text,
        "ladder.toml": ladder_text,
    }
    checks = {"seconds": round(seconds, 1)}
    run_dir = store.write(data_dir, "panel-classical", tag, tables, run_params, checks, texts)
    return run_dir, out | {"seconds": seconds, "selection": selection, "exported": exported}


def _f(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{value:.3f}"


def report(
    out: Mapping[str, Any],
    plan: sequence.Plan,
    source: Mapping[str, Any],
    exported: Mapping[str, Config],
    selection: Mapping[str, Any] | None,
    seconds: float,
) -> str:
    folds = sorted(plan.folds)
    lines = [
        "# Panel classical comparators",
        "",
        f"Inputs run `{source['inputs_run']}`, labels run `{source['labels_run']}`; schemes "
        f"{', '.join(plan.schemes)}, left-out quarters {', '.join(str(f + 1) for f in folds)}. "
        f"Rungs are the ladder's logistic on one row of the inputs table, on the ladder's training "
        f"rows; windows are the same logistic on plain statistics of the {plan.window}-hour window "
        f"a sequence model reads, on the sequence probes' training rows and scales. AUC inside "
        f"each left-out quarter on the ladder's common rows.",
        "",
        f"Wall time {seconds / 60:.1f} minutes.",
    ]
    lines += ["", "## Exported", "", "| name | model | variant | readout | c | rule value |"]
    lines.append("|---|---|---|---|---|---|")
    for name, config in exported.items():
        value = selection["chosen"][name]["score"] if selection else float("nan")
        lines.append(
            f"| {name} | {config.model} | {config.variant} | {config.readout or '-'} | "
            f"{config.c:g} | {_f(value)} |"
        )
    for h in plan.ladder.horizons:
        lines += ["", f"## {h} h", ""]
        lines.append(
            "| model | variant | readout | c | scheme | side | "
            + " | ".join(f"q{f + 1}" for f in folds)
            + " | mean |"
        )
        lines.append("|---" * (7 + len(folds)) + "|")
        keyed: dict[tuple[Any, ...], dict[int, float]] = {}
        for row in out["folds"]:
            if row["horizon"] == h:
                key = tuple(row[k] for k in ("model", "variant", "readout", "c", "scheme", "side"))
                keyed.setdefault(key, {})[row["fold"]] = row["auc"]
        for (model, variant, name, c, scheme, side), per in keyed.items():
            cells = [per.get(f, float("nan")) for f in folds]
            mean = float(np.nanmean(cells)) if np.isfinite(cells).any() else float("nan")
            lines.append(
                f"| {model} | {variant} | {name or '-'} | {c:g} | {scheme} | {side} | "
                + " | ".join(_f(v) for v in cells)
                + f" | {_f(mean)} |"
            )
    return "\n".join(lines) + "\n"
