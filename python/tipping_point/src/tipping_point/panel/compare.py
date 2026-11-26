# Paired evaluation of exported score files (docs/market.md, "Coin panel"): each arm is checked
# against the inputs run, labels grid and ladder plan it came from, aligned to the ladder's rows and
# scored with the ladder's own block-bootstrap draws, one pair named primary in a plan file.

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..ising import io
from . import labels, ladder, sequence, store

PLAN = Path(__file__).with_name("compare.toml")
KINDS = {"scores": "panel-scores", "classical": "panel-classical"}
PRIMARY_SCHEME = "forward"
COMPOSITE = "long+short"
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
SHA256 = re.compile(r"[0-9a-f]{64}")
ARM_KEYS = {
    "scores": ({"model", "variant"}, {"readout", "c", "seeds", "plan_sha256"}),
    "classical": ({"export"}, {"model", "readout", "c", "chosen", "plan_sha256"}),
}
COMPLETE = ("se", "q025", "q05", "q975", "above_zero", "draws")
LOG = Callable[[str], None]


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str
    model: str | None = None
    variant: str | None = None
    export: str | None = None
    pins: Mapping[str, Any] = field(default_factory=dict)

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


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _integer(value: Any) -> bool:
    return type(value) is int


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _pin(kind: str, key: str, value: Any) -> str | None:
    if key == "readout" and not _text(value):
        return "a readout name"
    if key == "c" and not _number(value):
        return "a finite number above 0"
    if key == "seeds" and not (
        isinstance(value, list)
        and value
        and all(_integer(s) and s >= 0 for s in value)
        and len(set(value)) == len(value)
    ):
        return "a list of distinct seeds"
    if key == "model" and kind == "classical" and value not in ("rung", "window"):
        return "rung or window"
    if key == "chosen" and not isinstance(value, bool):
        return "true or false"
    if key == "plan_sha256" and not (isinstance(value, str) and SHA256.fullmatch(value)):
        return "a sha256 in 64 lowercase hex digits"
    return None


def _arm(raw: Any, problems: list[str]) -> Arm | None:
    if not isinstance(raw, Mapping):
        problems.append("an arm that is not a table")
        return None
    name, kind = raw.get("name"), raw.get("kind")
    if not _text(name):
        problems.append("an arm without a name")
        return None
    if kind not in ARM_KEYS:
        problems.append(f"arm {name}: kind {kind} (scores or classical)")
        return None
    required, optional = ARM_KEYS[kind]
    fields = set(raw) - {"name", "kind"}
    before = len(problems)
    problems += [f"arm {name}: unknown key {k}" for k in sorted(fields - required - optional)]
    problems += [f"arm {name}: no {k}" for k in sorted(required - fields)]
    if kind == "scores":
        if "model" in raw and raw["model"] not in sequence.MODELS:
            problems.append(f"arm {name}: model {raw['model']} (one of {sequence.MODELS})")
        if "variant" in raw and not _text(raw["variant"]):
            problems.append(f"arm {name}: variant must be a rung's name")
    elif "export" in raw and not _text(raw["export"]):
        problems.append(f"arm {name}: export must be the name the classical run exported")
    pins = {k: raw[k] for k in sorted(fields & optional)}
    for key, value in pins.items():
        wanted = _pin(kind, key, value)
        if wanted:
            problems.append(f"arm {name}: {key} must be {wanted}")
    if len(problems) > before:
        return None
    if "c" in pins:
        pins["c"] = float(pins["c"])
    if kind == "scores":
        return Arm(str(name), kind, model=raw["model"], variant=raw["variant"], pins=pins)
    return Arm(str(name), kind, export=raw["export"], pins=pins)


def _tables(raw: Mapping[str, Any], key: str, problems: list[str]) -> list[Any]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        problems.append(f"{key} must be a list of tables")
        return []
    return value


def parse(path: Path, ladder_path: Path, texts: tuple[str, str]) -> Plan:
    raw = tomllib.loads(texts[0])
    base = ladder.load(ladder_path, texts[1])
    problems: list[str] = [
        f"unknown key {k}" for k in sorted(set(raw) - {"ladder", "arms", "pairs"})
    ]
    arms = [a for a in (_arm(r, problems) for r in _tables(raw, "arms", problems)) if a]
    names = [a.name for a in arms]
    if not arms and not problems:
        problems.append("no arms")
    problems += [f"arm {n} named twice" for n in sorted({n for n in names if names.count(n) > 1})]
    sides = list(base.sides)
    if {"long", "short"} <= set(base.sides):
        sides.append(COMPOSITE)
    pairs: list[tuple[str, str]] = []
    primaries: list[Mapping[str, Any]] = []
    for p in _tables(raw, "pairs", problems):
        if not isinstance(p, Mapping):
            problems.append("a pair that is not a table")
            continue
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
            problems.append(f"pair {first} - {second}: unknown key {', '.join(sorted(extra))}")
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
        if side not in sides:
            problems.append(f"primary side {side} (one of {sides})")
        if not _integer(horizon) or horizon not in base.horizons:
            problems.append(
                f"primary horizon {horizon} (an integer, one of the ladder plan's "
                f"{list(base.horizons)})"
            )
    if problems:
        raise ValueError(f"{path}: unknown or invalid: {', '.join(problems)}")
    assert isinstance(horizon, int)
    return Plan(base, tuple(arms), tuple(pairs), primary, str(side), horizon)


def snapshot(path: Path = PLAN) -> tuple[Plan, dict[str, Any], tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    named = tomllib.loads(text).get("ladder")
    if not _text(named):
        raise ValueError(f"{path}: unknown or invalid: ladder must name the ladder plan's file")
    ladder_path = (path.parent / str(named)).resolve()
    if not ladder_path.is_file():
        raise ValueError(f"{path}: unknown or invalid: ladder: no file {ladder_path}")
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
        out[name] = absolute(path if path.is_dir() else data_dir / KINDS[kinds[name]] / run)
    return out


def absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _read(path: Path, what: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"{what}: {path} is missing")
    return path.read_bytes()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bound_manifest(
    run_dir: Path, complete: Mapping[str, Any], what: str
) -> tuple[dict[str, Any], bool]:
    data = _read(run_dir / "manifest.json", what)
    recorded = complete.get("manifest_sha256")
    if recorded is not None and _digest(data) != recorded:
        raise ValueError(
            f"{what}: {run_dir / 'manifest.json'}: its sha256 differs from complete.json's"
        )
    return json.loads(data.decode("utf-8")), recorded is not None


Held = dict[Path, tuple[dict[str, Any], dict[str, Any], bool]]


def unbound_refused(run_dir: Path, what: str) -> ValueError:
    return ValueError(
        f"{what}: {run_dir.name} records no manifest sha256 in its complete.json (written before "
        f"runs were bound); --unbound accepts it for a development comparison, never a test"
    )


def finished(
    run_dir: Path, what: str, domain: str, unbound: bool, seen: Held
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    key = run_dir.resolve()
    if key not in seen:
        if not (run_dir / "manifest.json").is_file():
            raise ValueError(f"{what}: no run at {run_dir}")
        if not (run_dir / "complete.json").is_file():
            raise ValueError(f"{what}: {run_dir.name} has no complete.json")
        complete = io.read_json(run_dir / "complete.json")
        seen[key] = (complete, *bound_manifest(run_dir, complete, what))
    complete, manifest, bound = seen[key]
    if complete.get("clean") is not True:
        raise ValueError(f"{what}: complete.json does not mark {run_dir.name} clean")
    if manifest.get("run_id") != run_dir.name:
        raise ValueError(
            f"{what}: {run_dir.name} records run id {manifest.get('run_id')}, not its folder's"
        )
    if manifest.get("domain") != domain:
        raise ValueError(
            f"{what}: {run_dir.name} is a {manifest.get('domain')} run, not a {domain} run"
        )
    if not (bound or unbound):
        raise unbound_refused(run_dir, what)
    return manifest, complete, bound


def clean_code(producer: Any, what: str) -> None:
    producer = producer if isinstance(producer, Mapping) else {}
    if not producer.get("git_sha"):
        raise ValueError(f"{what} records no commit for its code")
    if producer.get("dirty") is not False:
        raise ValueError(f"{what} was made by code with uncommitted changes")


def _code(producer: Mapping[str, Any], params: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "git_sha": producer.get("git_sha"),
        "dirty": producer.get("dirty"),
        "uv_lock_sha256": None
        if params is None
        else (params.get("code") or {}).get("uv_lock_sha256"),
    }


def plans_of(
    run_dir: Path,
    complete: Mapping[str, Any],
    params: Mapping[str, Any],
    ladder_sha: str,
    what: str,
) -> tuple[str, str]:
    files = complete["files"]
    texts = []
    for name, key in (("plan.toml", "plan"), ("ladder.toml", "ladder_plan")):
        data = _read(run_dir / name, what)
        if name not in files or _digest(data) != files[name]:
            raise ValueError(f"{what}: {run_dir / name}: its sha256 differs from complete.json's")
        if (params.get(key) or {}).get("sha256") != files[name]:
            raise ValueError(f"{what}: {run_dir.name}'s manifest records another {name}")
        texts.append(data.decode("utf-8"))
    if files["ladder.toml"] != ladder_sha:
        raise ValueError(
            f"{what}: {run_dir.name}'s ladder.toml is not the comparison's ladder plan"
        )
    return texts[0], texts[1]


def sequence_dir(value: Any, data_dir: Path, what: str) -> Path:
    base = (data_dir / "panel-sequence").resolve()
    if not (isinstance(value, str) and RUN_ID.fullmatch(value)):
        raise ValueError(f"{what}: sequence run {value!r} is not a run id")
    path = base / value
    if path.resolve().parent != base:
        raise ValueError(f"{what}: sequence run {value!r} lies outside {base}")
    return path


def scores_chain(
    arm: Arm,
    run_dir: Path,
    params: Mapping[str, Any],
    data_dir: Path,
    ladder_sha: str,
    what: str,
    unbound: bool,
    seen: Held,
    scored: Sequence[str],
) -> tuple[dict[str, Any], Path, dict[str, Any], bool]:
    model, variant = params.get("model"), params.get("variant")
    if (model, variant) != (arm.model, arm.variant):
        raise ValueError(
            f"{what}: {run_dir.name} scores {model} on {variant}, not {arm.model} on {arm.variant}"
        )
    assert isinstance(model, str) and isinstance(variant, str)
    seq_dir = sequence_dir(params.get("sequence_run"), data_dir, what)
    link = f"{what}: the sequence run of {run_dir.name}"
    seq_manifest, seq_complete, bound = finished(seq_dir, link, "panel-sequence", unbound, seen)
    source = seq_manifest["params"]
    clean_code(seq_manifest.get("producer"), f"{link}, {seq_dir.name},")
    trained = seq_manifest.get("producer")
    merged = "sources" in source or "pieces_producer" in source
    if merged:
        trained = source.get("pieces_producer")
        clean_code(trained, f"{link}: the runs merged into {seq_dir.name}")
        if source.get("pieces_bound") is not True:
            bound = False
            if not unbound:
                raise ValueError(
                    f"{link}: the runs merged into {seq_dir.name} were not all bound; "
                    f"--unbound accepts it for a development comparison, never a test"
                )
    plan_text, ladder_text = plans_of(seq_dir, seq_complete, source, ladder_sha, link)
    plan = sequence.load(seq_dir / "plan.toml", seq_dir / "ladder.toml", (plan_text, ladder_text))
    if model not in plan.models or variant not in plan.variants:
        raise ValueError(f"{link}: {seq_dir.name}'s plan fits no {model} on {variant}")
    saved, used = source.get("encoders") or {}, params.get("encoders") or {}
    models_used(
        plan, model, variant, set(used), scored, f"{what}: the models {run_dir.name} scored"
    )
    for key, digest in sorted(used.items()):
        if saved.get(key) != digest or seq_complete["files"].get(key) != digest:
            raise ValueError(
                f"{what}: the models {run_dir.name} scored are not those of sequence run "
                f"{seq_dir.name}"
            )
        if _digest(_read(seq_dir / key, link)) != digest:
            raise ValueError(f"{link}: {seq_dir / key}: its sha256 differs from complete.json's")
    record = {
        "model": model,
        "variant": variant,
        "readout": params.get("readout"),
        "c": params.get("c"),
        "seeds": sorted(plan.seeds),
        "chosen": None if params.get("selection") is None else True,
        "selection": params.get("selection"),
        "plan_sha256": seq_complete["files"]["plan.toml"],
        "sequence_run": seq_manifest["run_id"],
        "sequence_code": _code(seq_manifest.get("producer") or {}, None if merged else source),
        "trained_by": _code(trained or {}, source),
    }
    return record, seq_dir, source, bound


def models_used(
    plan: sequence.Plan, model: str, variant: str, used: set[str], scored: Sequence[str], what: str
) -> None:
    single = dataclasses.replace(plan, variants=(variant,), models=(model,))
    others = used - sequence.cells(single)
    partial, empty = [], []
    for scheme in scored:
        quarters = [f for f in plan.folds if f >= plan.ladder.first_fold(scheme)]
        found = 0
        for f in quarters if scheme in plan.schemes else []:
            seeds = {sequence.encoder_name(variant, scheme, f, s, model) for s in plan.seeds}
            held = len(seeds & used)
            found += held
            if 0 < held < len(seeds):
                partial.append(f"{scheme} quarter {f + 1}")
        if not found:
            empty.append(scheme)
    problems = [f"{len(others)} not {model} on {variant} of the plan"] if others else []
    problems += [f"not every seed for {', '.join(partial)}"] if partial else []
    problems += [f"none for {', '.join(empty)}"] if empty else []
    if problems:
        raise ValueError(
            f"{what} are not whole quarters of every seed of the sequence run's {model} on "
            f"{variant}: {'; '.join(problems)}"
        )


def classical_chain(
    arm: Arm,
    run_dir: Path,
    params: Mapping[str, Any],
    complete: Mapping[str, Any],
    ladder_sha: str,
    what: str,
) -> dict[str, Any]:
    exported = params.get("exported") or {}
    if arm.export not in exported:
        raise ValueError(f"{what}: {run_dir.name} exported no {arm.export}")
    plans_of(run_dir, complete, params, ladder_sha, what)
    config = exported[arm.export]
    selection = params.get("selection")
    return {
        "export": arm.export,
        "model": config["model"],
        "variant": config["variant"],
        "readout": config["readout"],
        "c": config["c"],
        "chosen": selection is not None,
        "selection": None
        if selection is None
        else {"metric": selection["metric"], **selection["chosen"][arm.export]},
        "fixed": params.get("fixed"),
        "plan_sha256": complete["files"]["plan.toml"],
    }


def provenance(
    arm: Arm,
    run_dir: Path,
    data_dir: Path,
    made: Mapping[str, str],
    ladder_sha: str,
    unbound: bool,
    seen: Held,
) -> dict[str, Any]:
    what = f"arm {arm.name}"
    manifest, complete, bound = finished(run_dir, what, KINDS[arm.kind], unbound, seen)
    clean_code(manifest.get("producer"), f"{what}: {run_dir.name}")
    params = manifest["params"]
    record: dict[str, Any] = {"run": manifest["run_id"], "kind": arm.kind}
    chain = [(run_dir.name, params)]
    if arm.kind == "scores":
        scored = [s for s in ladder.SCHEMES if arm.file(s) in complete["files"]]
        found, seq_dir, source, seq_bound = scores_chain(
            arm, run_dir, params, data_dir, ladder_sha, what, unbound, seen, scored
        )
        record |= found
        bound = bound and seq_bound
        chain.append((seq_dir.name, source))
    else:
        record |= classical_chain(arm, run_dir, params, complete, ladder_sha, what)
    record["bound"] = bound
    for name, p in chain:
        for key in ("inputs_run", "labels_run"):
            if p.get(key) != made[key]:
                raise ValueError(
                    f"{what}: {name} was made from {key.replace('_', ' ')} {p.get(key)}, not "
                    f"{made[key]}"
                )
    for key, want in arm.pins.items():
        got = record.get(key)
        if (sorted(got) if key == "seeds" and isinstance(got, list) else got) != (
            sorted(want) if key == "seeds" else want
        ):
            raise ValueError(f"{what}: {run_dir.name} has {key} {got}, not the plan's {want}")
    holder, held = chain[-1]
    record["made_from"] = {
        "run": holder,
        **{k: held.get(k) for k in ("inputs", "labels_grid", "ladder_plan")},
    }
    record["code"] = _code(manifest.get("producer") or {}, params)
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


def score_bytes(name: str, path: Path, digest: str) -> bytes:
    data = _read(path, f"arm {name}")
    store.refuse_held_back(path, "hour", data)
    if _digest(data) != digest:
        raise ValueError(f"arm {name}: {path}: its sha256 differs from complete.json's")
    return data


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


def on_complete(difference: np.ndarray, whole: np.ndarray) -> dict[str, Any]:
    nan = np.full(1, np.nan)
    stats = ladder.comparison(np.nan, nan, nan, difference[whole], np.nan)
    return {f"{k}_complete": stats[k] for k in COMPLETE}


def on_common(
    folds: Mapping[tuple[str, str], np.ndarray],
    booted: Mapping[tuple[str, str], np.ndarray],
    first: str,
    second: str,
    side: str,
    common: np.ndarray,
) -> tuple[float, np.ndarray]:
    a, b = (np.where(common, folds[(n, side)], np.nan) for n in (first, second))
    estimate = float(np.nanmean(a) - np.nanmean(b)) if common.any() else np.nan
    drawn = [
        ladder.nanmean_rows(np.where(common, booted[(n, side)], np.nan)) for n in (first, second)
    ]
    return estimate, drawn[0] - drawn[1]


def evaluated(
    h: ladder.Horizon,
    base: ladder.Plan,
    horizon: int,
    names: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> tuple[list[dict], list[dict], list[dict]]:
    kept: dict[str, Any] = {}
    score_rows, fold_rows, rows = ladder.evaluate(h, base, horizon, names, pairs, kept)
    folds, boot, booted, whole = kept["folds"], kept["boot"], kept["booted"], kept["complete"]
    out, sides = [], ("long", "short")
    for first, second in pairs:
        mine = [r for r in rows if (r["first"], r["second"]) == (first, second)]
        for row in mine:
            side = row["side"]
            difference = boot[(first, side)] - boot[(second, side)]
            scheduled = {"quarters_scheduled": len(folds[(first, side)])}
            out.append(row | scheduled | on_complete(difference, whole[side]))
        if not set(sides) <= set(base.sides):
            continue
        a = (folds[(first, "long")] + folds[(first, "short")]) / 2
        b = (folds[(second, "long")] + folds[(second, "short")]) / 2
        common = np.isfinite(a - b)
        long, short = (on_common(folds, booted, first, second, side, common) for side in sides)
        difference = (long[1] + short[1]) / 2
        estimate = (long[0] + short[0]) / 2
        both = (
            common.any()
            & np.isfinite(booted[(first, "long")][:, common]).all(axis=1)
            & np.isfinite(booted[(first, "short")][:, common]).all(axis=1)
        )
        row = {
            "horizon": horizon,
            "scheme": h.scheme,
            "side": COMPOSITE,
            "first": first,
            "second": second,
            **ladder.comparison(
                estimate, a, b, difference, float(np.mean(both)) if common.any() else np.nan
            ),
            "quarters_scheduled": len(common),
        }
        out.append(row | on_complete(difference, both))
    return score_rows, fold_rows, out


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
            score_rows, fold_rows, pairs = evaluated(h, base, horizon, plan.names, plan.pairs)
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
    unbound: bool = False,
) -> tuple[Path, dict[str, Any]]:
    began = time.perf_counter()
    plan, records, (plan_text, ladder_text) = snapshot(plan_path)
    ladder_sha = records["ladder_plan"]["sha256"]
    dirs = given(plan, arms, data_dir)
    inputs_dir = absolute(inputs_dir)
    if not (inputs_dir / "manifest.json").is_file():
        raise ValueError(f"no inputs run at {inputs_dir}")
    seen: Held = {}
    inputs_manifest, inputs_complete, inputs_bound = finished(
        inputs_dir, "inputs run", "panel-inputs", unbound, seen
    )
    clean_code(inputs_manifest.get("producer"), f"inputs run: {inputs_dir.name}")
    inputs_code = _code(inputs_manifest["producer"], inputs_manifest["params"])
    made = {
        "inputs_run": inputs_manifest["run_id"],
        "labels_run": inputs_manifest["params"].get("labels_run"),
    }
    found = {
        arm.name: provenance(arm, dirs[arm.name], data_dir, made, ladder_sha, unbound, seen)
        for arm in plan.arms
    }
    schemes = [s for s in ladder.SCHEMES if all(s in r["files"] for r in found.values())]
    for record in found.values():
        record["files"] = {s: record["files"][s] for s in schemes}
    held_in = inputs_manifest["params"]
    for path in (
        data_dir / "labels" / held_in["labels_run"] / held_in["labels_grid"]["file"],
        inputs_dir / "inputs.parquet",
    ):
        if not path.is_file():
            raise ValueError(f"inputs run: {path} is missing")
    values, grid, source = sequence.read_columns(
        inputs_dir, data_dir, plan.ladder.columns, (inputs_manifest, inputs_complete)
    )
    for arm in plan.arms:
        same_sources(arm, found[arm.name], source, ladder_sha)
    held = {
        name: {
            s: score_bytes(name, dirs[name] / r["files"][s]["file"], r["files"][s]["sha256"])
            for s in schemes
        }
        for name, r in found.items()
    }
    tables = {
        name: {
            s: ladder.external_table(dirs[name] / found[name]["files"][s]["file"], data)
            for s, data in per.items()
        }
        for name, per in held.items()
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
        "unbound_allowed": unbound,
        "inputs_bound": inputs_bound,
        "inputs_code": inputs_code,
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
        "report.md": report(
            out, plan, source, records, found, seconds, inputs_bound, unbound, inputs_code
        ),
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


def _row(c: Mapping[str, Any], suffix: str = "", label: str = "") -> str:
    return (
        f"| {c['first']} - {c['second']}{label} | {c['scheme']} | {c['horizon']} h | {c['side']} | "
        f"{_f(c['estimate'], True)} | {_f(c['se' + suffix])} | {_f(c['se_quarters'])} | "
        f"{_f(c['q025' + suffix], True)} | {_f(c['q05' + suffix], True)} | "
        f"{_f(c['q975' + suffix], True)} | {_f(c['above_zero' + suffix])} | "
        f"{_f(c['complete_draws'])} | {c['draws' + suffix]} | "
        f"{c['folds_above_zero']} of {c['folds']} |"
    )


def _state(dirty: Any) -> str:
    if dirty is False:
        return "clean"
    return "uncommitted changes" if dirty is True else "unknown"


def code_states(found: Mapping[str, Mapping[str, Any]]) -> dict[str, tuple[Any, ...]]:
    out = {}
    for name, record in found.items():
        for label, key in (
            ("", "code"),
            (" (sequence run)", "sequence_code"),
            (" (trained by)", "trained_by"),
        ):
            if key in record:
                code = record[key]
                out[f"{name}{label}"] = (code["git_sha"], code["dirty"], code["uv_lock_sha256"])
    return out


def _lock(value: Any) -> str:
    return "-" if value is None else str(value)


def _configuration(record: Mapping[str, Any]) -> str:
    return {True: "chosen by the rule", False: "fixed"}.get(record["chosen"], "none to choose")


def report(
    out: Mapping[str, Any],
    plan: Plan,
    source: Mapping[str, Any],
    records: Mapping[str, Any],
    found: Mapping[str, Mapping[str, Any]],
    seconds: float,
    inputs_bound: bool,
    unbound: bool = False,
    inputs_code: Mapping[str, Any] | None = None,
) -> str:
    base = plan.ladder
    lines = [
        "# Panel comparison",
        "",
        f"Inputs run `{source['inputs_run']}`, labels run `{source['labels_run']}`, comparison "
        f"plan `{records['plan']['file']}`, ladder plan `{records['ladder_plan']['file']}`. Every "
        f"arm's out-of-fold scores are scored on the ladder's rows of each scheme, horizon and "
        f"side; each pair is first minus second in the mean over quarters of the AUC inside each "
        f"quarter. `{COMPOSITE}` is the mean of the long and short sides, draw by draw, on the "
        f"quarters both sides score. The "
        f"interval is the ladder's joint block bootstrap ({base.block_days}-day blocks within "
        f"quarters, {base.draws} draws, seed {base.seed}) with the same draws for every arm and "
        f"side, conditional on the fitted models and the observed quarters, so it carries no "
        f"quarter-to-quarter or refitting variation; `se q` is the standard error over quarters. "
        f"`> 0` is the share of draws above zero, not a p-value. `complete` is the share of draws "
        f"in which every quarter scored in the original has both classes; `draws` counts the draws "
        f"used.",
        "",
        f"Wall time {_f(seconds / 60)} minutes.",
    ]
    header = [
        "| pair | scheme | horizon | side | estimate | se | se q | q2.5 | q5 | q97.5 | > 0 | "
        "complete | draws | folds > 0 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += ["", "## Primary", ""]
    if unbound:
        lines += [
            "This comparison ran with --unbound, which accepts runs whose manifests are not "
            "bound, so it is not a test.",
            "",
        ]
    lines += [
        f"{plan.primary[0]} - {plan.primary[1]}, {PRIMARY_SCHEME}, {plan.side}, {plan.horizon} h, "
        "as this plan names it, then the same row on complete draws only. A development "
        "comparison between configurations chosen on the development span is not evidence.",
        "",
        *header,
    ]
    (top,) = [c for c in out["comparisons"] if c["primary"]]
    lines += [_row(top), _row(top, "_complete", " (complete draws)"), ""]
    lines.append(
        f"The primary row scores {top['folds']} of the {top['quarters_scheduled']} quarters "
        f"scheduled for {PRIMARY_SCHEME}."
    )
    lines += ["", "## Secondary", "", *header]
    lines += [_row(c) for c in out["comparisons"] if not c["primary"]]
    lines += ["", "## Arms", ""]
    lines += [
        "| arm | run | kind | model | variant | configuration | readout | c | plan | commit | "
        "uv.lock | bound |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in found.items():
        code = r["code"]
        c = "-" if r["c"] is None else f"{r['c']:g}"
        lines.append(
            f"| {name} | `{r['run']}` | {r['kind']} | {r['model']} | {r['variant']} | "
            f"{_configuration(r)} | {r['readout'] or '-'} | {c} | {r['plan_sha256'][:12]} | "
            f"{str(code['git_sha'])[:12]} ({_state(code['dirty'])}) | "
            f"{_lock(code['uv_lock_sha256'])[:12]} | {'yes' if r['bound'] else 'no'} |"
        )
    states = code_states(found)
    lines += [
        "",
        "`plan` is the sha256 of the plan.toml the arm's source run was made under. `bound` says "
        "whether every run in the arm's chain records its manifest's sha256 in complete.json; a "
        "run written before that record existed is not bound, and its manifest could have changed "
        "unseen. The inputs run is " + ("bound." if inputs_bound else "not bound."),
        "",
    ]
    if inputs_code is not None:
        lines += [
            f"The inputs run was made by commit {inputs_code['git_sha']} "
            f"({_state(inputs_code['dirty'])}), uv.lock {_lock(inputs_code['uv_lock_sha256'])}.",
            "",
        ]
    commits = {v[:2] for v in states.values()}
    locks = {v[2] for v in states.values() if v[2] is not None}
    if len(commits) > 1 or len(locks) > 1:
        lines.append(
            "The arms' runs were not all made by the same code (commit, state, uv.lock): "
            + "; ".join(f"{k} {v[0]}, {_state(v[1])}, {_lock(v[2])}" for k, v in states.items())
            + "."
        )
    else:
        lines.append("Every arm's runs were made by the same commit and uv.lock.")
    lines += ["", "## Fold-mean AUC", ""]
    lines += ["| arm | scheme | horizon | side | fold mean | folds |", "|---|---|---|---|---|---|"]
    for s in out["scores"]:
        if s["coins"] == "all":
            lines.append(
                f"| {s['name']} | {s['scheme']} | {s['horizon']} h | {s['side']} | "
                f"{_f(s['fold_mean'])} | {s['folds']} |"
            )
    return "\n".join(lines) + "\n"
