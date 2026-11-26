# Paired evaluation of exported score files (docs/market.md, "Coin panel"): each arm is checked
# against the inputs run, labels grid and ladder plan it came from, aligned to the ladder's rows and
# scored with the ladder's own block-bootstrap draws, one pair named primary in a plan file.

from __future__ import annotations

import hashlib
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..ising import io
from . import labels, ladder, sequence, store

PLAN = Path(__file__).with_name("compare.toml")
KINDS = {"scores": "panel-scores", "classical": "panel-classical"}
PRIMARY_SCHEME = "forward"
LOG = Callable[[str], None]


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str
    model: str | None = None
    variant: str | None = None
    export: str | None = None

    def file(self, scheme: str) -> str:
        return f"{scheme}.parquet" if self.kind == "scores" else f"{self.export}-{scheme}.parquet"


@dataclass(frozen=True)
class Plan:
    ladder: ladder.Plan
    arms: tuple[Arm, ...]
    pairs: tuple[tuple[str, str], ...]
    primary: tuple[str, str]
    side: str
    horizon: int

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(arm.name for arm in self.arms)


def _arm(raw: Mapping[str, Any], problems: list[str]) -> Arm | None:
    name, kind = raw.get("name"), raw.get("kind")
    fields = set(raw) - {"name", "kind"}
    if not isinstance(name, str) or not name:
        problems.append("an arm without a name")
        return None
    if kind == "scores":
        model = raw.get("model")
        if fields != {"model", "variant"} or model not in sequence.MODELS:
            problems.append(
                f"arm {name}: a scores arm takes a model of {sequence.MODELS} and a variant"
            )
            return None
        return Arm(name, kind, model=str(model), variant=str(raw["variant"]))
    if kind == "classical":
        if fields != {"export"} or not isinstance(raw["export"], str):
            problems.append(f"arm {name}: a classical arm takes the name it was exported under")
            return None
        return Arm(name, kind, export=raw["export"])
    problems.append(f"arm {name}: kind {kind} (scores or classical)")
    return None


def parse(path: Path, ladder_path: Path, texts: tuple[str, str]) -> Plan:
    raw = tomllib.loads(texts[0])
    base = ladder.load(ladder_path, texts[1])
    problems: list[str] = []
    arms = [a for a in (_arm(r, problems) for r in raw.get("arms", ())) if a is not None]
    names = [a.name for a in arms]
    if not arms and not problems:
        problems.append("no arms")
    problems += [f"arm {n} named twice" for n in sorted({n for n in names if names.count(n) > 1})]
    pairs: list[tuple[str, str]] = []
    primaries: list[Mapping[str, Any]] = []
    for p in raw.get("pairs", ()):
        first, second = p.get("first"), p.get("second")
        if first not in names or second not in names:
            problems.append(f"pair {first} - {second} names an unknown arm")
            continue
        if first == second:
            problems.append(f"pair {first} - {second} compares an arm with itself")
        elif {first, second} in [set(q) for q in pairs]:
            problems.append(f"pair {first} - {second} listed twice")
        pairs.append((first, second))
        extra = set(p) - {"first", "second", "primary", "side", "horizon"}
        if extra:
            problems.append(f"pair {first} - {second}: unknown {', '.join(sorted(extra))}")
        if p.get("primary", False) is True:
            primaries.append(p)
        elif p.get("primary", False) is not False:
            problems.append(f"pair {first} - {second}: primary must be true or false")
        elif "side" in p or "horizon" in p:
            problems.append(
                f"pair {first} - {second}: only the primary pair names a side and horizon"
            )
    if not pairs and not problems:
        problems.append("no pairs")
    if len(primaries) != 1:
        problems.append(f"{len(primaries)} primary pairs (exactly one must be primary)")
    side, horizon, primary = "", 0, ("", "")
    if len(primaries) == 1:
        found = primaries[0]
        primary = (found["first"], found["second"])
        side, horizon = found.get("side"), found.get("horizon")
        if side not in base.sides:
            problems.append(f"primary side {side} (one of the ladder plan's {list(base.sides)})")
        if not isinstance(horizon, int) or horizon not in base.horizons:
            problems.append(
                f"primary horizon {horizon} (one of the ladder plan's {list(base.horizons)})"
            )
    if problems:
        raise ValueError(f"{path}: unknown or invalid: {', '.join(problems)}")
    assert isinstance(horizon, int)
    return Plan(base, tuple(arms), tuple(pairs), primary, str(side), horizon)


def snapshot(path: Path = PLAN) -> tuple[Plan, dict[str, Any], tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    ladder_path = (path.parent / str(tomllib.loads(text)["ladder"])).resolve()
    texts = (text, ladder_path.read_text(encoding="utf-8"))
    plan = parse(path, ladder_path, texts)
    records = [
        {"file": p.name, "sha256": hashlib.sha256(t.encode("utf-8")).hexdigest()}
        for p, t in zip((path, ladder_path), texts, strict=True)
    ]
    return plan, {"plan": records[0], "ladder_plan": records[1]}, texts


def given(plan: Plan, arms: Sequence[tuple[str, str]], data_dir: Path) -> dict[str, Path]:
    names = [name for name, _ in arms]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"arm {', '.join(twice)} given more than once")
    unknown = sorted(set(names) - set(plan.names))
    if unknown:
        raise ValueError(f"the plan has no arm {', '.join(unknown)}")
    missing = [n for n in plan.names if n not in names]
    if missing:
        raise ValueError(f"no run given for arm {', '.join(missing)}")
    kinds = {arm.name: arm.kind for arm in plan.arms}
    out = {}
    for name, run in arms:
        path = Path(run)
        out[name] = path if path.is_dir() else data_dir / KINDS[kinds[name]] / run
    return out


def finished(run_dir: Path, what: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not (run_dir / "manifest.json").is_file():
        raise ValueError(f"{what}: no run at {run_dir}")
    if not (run_dir / "complete.json").is_file():
        raise ValueError(f"{what}: {run_dir.name} has no complete.json")
    complete = io.read_json(run_dir / "complete.json")
    if complete.get("clean") is not True:
        raise ValueError(f"{what}: complete.json does not mark {run_dir.name} clean")
    return io.read_json(run_dir / "manifest.json"), complete


def _code(manifest: Mapping[str, Any]) -> dict[str, Any]:
    producer = manifest.get("producer") or {}
    return {
        "git_sha": producer.get("git_sha"),
        "dirty": producer.get("dirty"),
        "uv_lock_sha256": (manifest["params"].get("code") or {}).get("uv_lock_sha256"),
    }


def provenance(arm: Arm, run_dir: Path, data_dir: Path, made: Mapping[str, str]) -> dict[str, Any]:
    what = f"arm {arm.name}"
    manifest, complete = finished(run_dir, what)
    domain = manifest.get("domain")
    if domain != KINDS[arm.kind]:
        raise ValueError(f"{what}: {run_dir.name} is a {domain} run, not a {KINDS[arm.kind]} run")
    params = manifest["params"]
    record: dict[str, Any] = {"run": manifest["run_id"], "kind": arm.kind}
    if arm.kind == "scores":
        model, variant = params.get("model"), params.get("variant")
        if (model, variant) != (arm.model, arm.variant):
            raise ValueError(
                f"{what}: {run_dir.name} scores {model} on {variant}, not {arm.model} on "
                f"{arm.variant}"
            )
        seq_dir = data_dir / "panel-sequence" / str(params.get("sequence_run"))
        seq_manifest, seq_complete = finished(
            seq_dir, f"{what}: the sequence run of {run_dir.name}"
        )
        if seq_manifest.get("domain") != "panel-sequence":
            raise ValueError(f"{what}: {seq_dir.name} is not a sequence run")
        source = seq_manifest["params"]
        saved, used = source.get("encoders") or {}, params.get("encoders") or {}
        if not used or any(
            saved.get(k) != v or seq_complete["files"].get(k) != v for k, v in used.items()
        ):
            raise ValueError(
                f"{what}: the models {run_dir.name} scored are not those of sequence run "
                f"{seq_dir.name}"
            )
        record |= {
            "model": model,
            "variant": variant,
            "readout": params.get("readout"),
            "c": params.get("c"),
            "selection": params.get("selection"),
            "sequence_run": seq_manifest["run_id"],
            "sequence_code": _code(seq_manifest),
        }
        chain = [(run_dir.name, params), (seq_dir.name, source)]
    else:
        exported = params.get("exported") or {}
        if arm.export not in exported:
            raise ValueError(f"{what}: {run_dir.name} exported no {arm.export}")
        config = exported[arm.export]
        selection = params.get("selection")
        record |= {
            "export": arm.export,
            "model": config["model"],
            "variant": config["variant"],
            "readout": config["readout"],
            "c": config["c"],
            "selection": None
            if selection is None
            else {"metric": selection["metric"], **selection["chosen"][arm.export]},
            "fixed": params.get("fixed"),
        }
        chain = [(run_dir.name, params)]
    for name, p in chain:
        for key in ("inputs_run", "labels_run"):
            if p.get(key) != made[key]:
                raise ValueError(
                    f"{what}: {name} was made from {key.replace('_', ' ')} {p.get(key)}, not "
                    f"{made[key]}"
                )
    holder, held = chain[-1]
    record["made_from"] = {
        "run": holder,
        **{k: held.get(k) for k in ("inputs", "labels_grid", "ladder_plan")},
    }
    record["code"] = _code(manifest)
    record["files"] = {
        s: {"file": arm.file(s), "sha256": complete["files"][arm.file(s)]}
        for s in ladder.SCHEMES
        if arm.file(s) in complete["files"]
    }
    if PRIMARY_SCHEME not in record["files"]:
        raise ValueError(
            f"{what}: {run_dir.name} holds no {PRIMARY_SCHEME} scores ({arm.file(PRIMARY_SCHEME)})"
        )
    return record


def _sha(record: Mapping[str, Any] | None) -> Any:
    return (record or {}).get("sha256")


def same_sources(
    arm: Arm, record: Mapping[str, Any], source: Mapping[str, Any], ladder_sha: str
) -> None:
    held = record["made_from"]
    what = f"arm {arm.name}: {held['run']}"
    if _sha(held["inputs"]) != source["inputs"]["sha256"]:
        raise ValueError(f"{what} read another inputs table than inputs run {source['inputs_run']}")
    if _sha(held["labels_grid"]) != source["labels_grid"]["sha256"]:
        raise ValueError(f"{what} read another labels grid than labels run {source['labels_run']}")
    if _sha(held["ladder_plan"]) != ladder_sha:
        raise ValueError(f"{what} was made under another ladder plan")


def aligned(
    table: pa.Table,
    name: str,
    ts: np.ndarray,
    symbol: np.ndarray,
    fold: np.ndarray,
    scheme: str,
    horizon: int,
    sides: Sequence[str],
) -> dict[str, np.ndarray]:
    what = f"arm {name}, {scheme}"
    try:
        scores = ladder.external_scores(table, ts, symbol, fold, scheme, horizon, sides)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from exc
    hours = table["horizon"].cast(pa.int64()).to_numpy()
    named = table["side"].cast(pa.string()).to_numpy(zero_copy_only=False)
    for side in sides:
        given = int(np.sum((hours == horizon) & (named == side)))
        if given != len(ts):
            raise ValueError(
                f"{what}: {given} rows at {horizon} h {side}, not the ladder's {len(ts)}: "
                f"{given - len(ts)} rows that are not the ladder's"
            )
    return scores


def compare(
    values: Mapping[str, np.ndarray],
    grid: Mapping[str, np.ndarray],
    plan: Plan,
    tables: Mapping[str, Mapping[str, pa.Table]],
    schemes: Sequence[str],
    log: LOG = lambda line: None,
) -> dict[str, Any]:
    base = plan.ladder
    edges = ladder.bounds(base)
    out: dict[str, Any] = {"accounting": {}, "scores": [], "folds": [], "comparisons": []}
    expected = {scheme: 0 for scheme in schemes}
    for horizon in base.horizons:
        keep, accounting = ladder.select(values, grid, base, horizon)
        out["accounting"][horizon] = accounting
        ts, symbol = grid["ts_ns"][keep], grid["symbol"][keep]
        y = {"long": grid[f"long_{horizon}"][keep], "short": grid[f"short_{horizon}"][keep]}
        y["any"] = y["long"] | y["short"]
        fold, block = ladder.folds(ts, edges, base.block_days)
        for scheme in schemes:
            first = base.first_fold(scheme)
            mine = fold >= first
            expected[scheme] += int(mine.sum()) * len(base.sides)
            scores = {
                name: aligned(
                    tables[name][scheme],
                    name,
                    ts[mine],
                    symbol[mine],
                    fold[mine],
                    scheme,
                    horizon,
                    base.sides,
                )
                for name in plan.names
            }
            h = ladder.Horizon(
                scheme,
                first,
                ts[mine],
                symbol[mine],
                {side: v[mine] for side, v in y.items()},
                fold[mine],
                block[mine],
                scores,
            )
            score_rows, fold_rows, pairs = ladder.evaluate(h, base, horizon, plan.names, plan.pairs)
            for row in pairs:
                row["primary_pair"] = (row["first"], row["second"]) == plan.primary
                row["primary"] = row["primary_pair"] and (scheme, row["side"], horizon) == (
                    PRIMARY_SCHEME,
                    plan.side,
                    plan.horizon,
                )
            out["scores"] += score_rows
            out["folds"] += fold_rows
            out["comparisons"] += pairs
            log(f"  {horizon} h, {scheme}: {int(mine.sum())} rows")
    for name in plan.names:
        for scheme in schemes:
            rows = tables[name][scheme].num_rows
            if rows != expected[scheme]:
                raise ValueError(
                    f"arm {name}, {scheme}: {rows} rows, not the ladder's {expected[scheme]}: "
                    f"{rows - expected[scheme]} rows at a horizon or side the ladder plan lacks"
                )
    if sum(row["primary"] for row in out["comparisons"]) != 1:
        raise RuntimeError("the primary pair, side and horizon were not scored once")
    return out


def run(
    arms: Sequence[tuple[str, str]],
    inputs_dir: Path,
    data_dir: Path,
    plan_path: Path,
    tag: str,
    log: LOG = lambda line: None,
) -> tuple[Path, dict[str, Any]]:
    began = time.perf_counter()
    plan, records, (plan_text, ladder_text) = snapshot(plan_path)
    dirs = given(plan, arms, data_dir)
    inputs_manifest = io.read_json(inputs_dir / "manifest.json")
    made = {
        "inputs_run": inputs_manifest["run_id"],
        "labels_run": inputs_manifest["params"]["labels_run"],
    }
    found = {arm.name: provenance(arm, dirs[arm.name], data_dir, made) for arm in plan.arms}
    schemes = [s for s in ladder.SCHEMES if all(s in r["files"] for r in found.values())]
    for record in found.values():
        record["files"] = {s: record["files"][s] for s in schemes}
    paths = {
        name: {s: dirs[name] / r["files"][s]["file"] for s in schemes} for name, r in found.items()
    }
    for per in paths.values():
        for path in per.values():
            store.refuse_held_back(path, "hour")
    values, grid, source = sequence.read_columns(inputs_dir, data_dir, plan.ladder.columns)
    for arm in plan.arms:
        same_sources(arm, found[arm.name], source, records["ladder_plan"]["sha256"])
    for name, per in paths.items():
        for scheme, path in per.items():
            if store.sha256(path) != found[name]["files"][scheme]["sha256"]:
                raise ValueError(f"arm {name}: {path}: its sha256 differs from complete.json's")
    tables = {
        name: {s: ladder.external_table(path) for s, path in per.items()}
        for name, per in paths.items()
    }
    out = compare(values, grid, plan, tables, schemes, log)
    seconds = time.perf_counter() - began
    primary = {
        "first": plan.primary[0],
        "second": plan.primary[1],
        "scheme": PRIMARY_SCHEME,
        "side": plan.side,
        "horizon": plan.horizon,
    }
    run_params = {
        **source,
        **records,
        "schemes": schemes,
        "primary": primary,
        "pairs": [list(p) for p in plan.pairs],
        "arms": found,
        "code": {"uv_lock_sha256": labels.uv_lock()},
    }
    result = {
        "scores": pa.Table.from_pylist(out["scores"]),
        "folds": pa.Table.from_pylist(out["folds"]),
        "comparisons": pa.Table.from_pylist(out["comparisons"]),
        "rows": pa.Table.from_pylist(
            [{"horizon": h, **row} for h, rows in out["accounting"].items() for row in rows]
        ),
    }
    texts = {
        "report.md": report(out, plan, source, records, found, seconds),
        "plan.toml": plan_text,
        "ladder.toml": ladder_text,
    }
    checks = {"seconds": round(seconds, 1)}
    run_dir = store.write(data_dir, "panel-compare", tag, result, run_params, checks, texts)
    return run_dir, out | {"seconds": seconds, "arms": found, "schemes": schemes}


def _f(value: float, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _row(c: Mapping[str, Any]) -> str:
    return (
        f"| {c['first']} - {c['second']} | {c['scheme']} | {c['horizon']} h | {c['side']} | "
        f"{_f(c['estimate'], True)} | {_f(c['se'])} | {_f(c['se_quarters'])} | "
        f"{_f(c['q025'], True)} | {_f(c['q05'], True)} | {_f(c['q975'], True)} | "
        f"{_f(c['above_zero'])} | {_f(c['complete_draws'])} | "
        f"{c['folds_above_zero']} of {c['folds']} |"
    )


def code_states(found: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[Any, ...]]:
    out = {}
    for name, record in found.items():
        for label, key in (("", "code"), (" (sequence run)", "sequence_code")):
            if key in record:
                code = record[key]
                out[f"{name}{label}"] = (code["git_sha"], code["dirty"], code["uv_lock_sha256"])
    return out


def report(
    out: Mapping[str, Any],
    plan: Plan,
    source: Mapping[str, Any],
    records: Mapping[str, Any],
    found: Mapping[str, Mapping[str, Any]],
    seconds: float,
) -> str:
    base = plan.ladder
    lines = [
        "# Panel comparison",
        "",
        f"Inputs run `{source['inputs_run']}`, labels run `{source['labels_run']}`, comparison "
        f"plan `{records['plan']['file']}`, ladder plan `{records['ladder_plan']['file']}`. Every "
        f"arm's out-of-fold scores are scored on the ladder's rows of each scheme, horizon and "
        f"side; each pair is first minus second in the mean over quarters of the AUC inside each "
        f"quarter. The interval is the ladder's joint block bootstrap ({base.block_days}-day "
        f"blocks, {base.draws} draws, seed {base.seed}) with the same draws for every arm, "
        f"conditional on the fitted models and the observed quarters. Columns as in the ladder's "
        f"report.",
        "",
        f"Wall time {_f(seconds / 60)} minutes.",
    ]
    header = [
        "| pair | scheme | horizon | side | estimate | se | se q | q2.5 | q5 | q97.5 | > 0 | "
        "complete | folds > 0 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += ["", "## Primary", ""]
    lines += [
        f"{plan.primary[0]} - {plan.primary[1]}, {PRIMARY_SCHEME}, {plan.side}, {plan.horizon} h, "
        "as this plan names it. A development comparison between configurations chosen on the "
        "development span is not evidence.",
        "",
        *header,
    ]
    lines += [_row(c) for c in out["comparisons"] if c["primary"]]
    lines += ["", "## Secondary", "", *header]
    lines += [_row(c) for c in out["comparisons"] if not c["primary"]]
    lines += ["", "## Arms", ""]
    lines += [
        "| arm | run | kind | model | variant | readout | c | commit | uv.lock |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in found.items():
        code = r["code"]
        commit = f"{str(code['git_sha'])[:12]}{' (uncommitted changes)' if code['dirty'] else ''}"
        c = "-" if r["c"] is None else f"{r['c']:g}"
        lines.append(
            f"| {name} | `{r['run']}` | {r['kind']} | {r['model']} | {r['variant']} | "
            f"{r['readout'] or '-'} | {c} | {commit} | {str(code['uv_lock_sha256'])[:12]} |"
        )
    states = code_states(found)
    lines.append("")
    if len(set(states.values())) > 1:
        lines.append(
            "The arms' runs were not all made by the same code (commit, uncommitted changes, "
            "uv.lock): "
            + "; ".join(
                f"{k} {v[0]}, {'dirty' if v[1] else 'clean'}, {v[2]}" for k, v in states.items()
            )
            + "."
        )
    else:
        lines.append("Every arm's runs were made by the same commit and uv.lock.")
    if any(v[1] is not False for v in states.values()):
        lines.append("Some were made by code with uncommitted changes or no recorded commit.")
    lines += ["", "## Fold-mean AUC", ""]
    lines += ["| arm | scheme | horizon | side | fold mean | folds |", "|---|---|---|---|---|---|"]
    for s in out["scores"]:
        if s["coins"] == "all":
            lines.append(
                f"| {s['name']} | {s['scheme']} | {s['horizon']} h | {s['side']} | "
                f"{_f(s['fold_mean'])} | {s['folds']} |"
            )
    return "\n".join(lines) + "\n"
