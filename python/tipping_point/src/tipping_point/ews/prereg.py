"""Pre-registration files: paper/prereg/<id>-v<k>.toml, read with the standard library's tomllib.

load      parse and check the structure (placeholders such as "TBD" or "<...>" are allowed)
freeze    refuse a file with placeholders, or one that is untracked or differs from HEAD, then tag
          the commit prereg/<id>-v<k> (an annotated tag: it carries its own timestamp)
frozen    the file's sha256 must equal that of the blob at its tag; scoring refuses otherwise
stamp     what a result's manifest records: the prereg's id, version, sha256, tag and commit
write_result  <data dir>/ews/<run_id>/ in #22's layout: manifest.json (stamped), one Parquet
          file per table, complete.json last
log_trial append one evaluation to <data dir>/ews/trials.jsonl; every evaluation counts as a
          trial, sensitivity cells included, and the count goes in the paper

An amendment is a new file, <id>-v<k+1>.toml, whose `supersedes` names the version it replaces
and whose `reason` says why; results scored under an earlier version stay bound to it.

A file with a [latents] section scores a JEPA's latent columns (#28). It is checked against its own
schema (_LATENT_SCHEMA): the checkpoints it scores, each with its untrained twin, the (observable,
indicator) pairs judged by AUC, the statistics judged by the dynamic test, and the [baseline] whose
runs it reads. As a draft it may leave as a placeholder a direction, an extremum or a reference's
tolerance, which the file's experiment sets before the freeze; the freeze's strict check refuses
every placeholder, whichever key holds it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .. import __version__, engine
from ..ising import io
from .trend import RAW

PREREG_DIR = Path("paper") / "prereg"
INDICATORS = {"variance", "ac", "skewness"}
OBSERVABLES = {"abs_m", "energy"}
CADENCES = {"dense", "matched"}
DIRECTIONS = {"greater", "less", "two-sided", "match"}
ROLES = {"positive-control", "descriptive", "reference"}
QUANTITIES = {"l2_var_abs_m", "mean_abs_m", "skewness_abs_m", "tau_int_abs_m", "one_minus_ac1"}
SURROGATES = {"phase", "ar1"}
STATISTICS = {"peak"}  # fss.dynamic.statistic: what the dynamic test reads off each ramp
# in a file with a [latents] section, which has its own observables, roles and statistics
LATENT_COLUMNS = {"surprise", *(f"pc_{k}" for k in range(1, 9))}
LATENT_INDICATORS = INDICATORS | RAW
LATENT_ROLES = ROLES | {"pipeline-check"}
LATENT_STATISTICS = ("l2_var", "mean", "one_minus_ac1")  # a quantity is <statistic>_<column>
CHECKPOINT_ROLES = {"primary", "ema-only", "held-out"}  # primary: the whole grid; else the headline
DYNAMIC_ROLES = {"pipeline-check", "test"}
EXTREMA = {"max", "min"}
# jepa.probes.TARGETS and COLUMNS but raw, which this package cannot import
PROBE_TARGETS, PROBE_CONTROLS = {"abs_m", "energy", "l2var"}, {"iid", "shuffle_within_T"}
# "TBD", "<words>" as in TEMPLATE, or a choice still open, "<min|max, set before the freeze>"
_PLACEHOLDER = re.compile(r"TBD|<[a-z][^<>|]*>|<[a-z]+(\|[a-z]+)+,[^<>]*>")
_SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class PreregError(ValueError):
    """A pre-registration that is malformed, unfrozen or modified since its freeze."""


@dataclass(frozen=True)
class Prereg:
    path: Path
    doc: dict[str, Any]
    sha256: str

    @property
    def id(self) -> str:
        return self.doc["id"]

    @property
    def version(self) -> int:
        return self.doc["version"]

    @property
    def name(self) -> str:
        return f"{self.id}-v{self.version}"

    @property
    def tag(self) -> str:
        return f"prereg/{self.name}"


# key -> (type, required); a nested dict is a table checked recursively
_NUMBER = (int, float)
_SCHEMA: dict[str, Any] = {
    "id": str,
    "version": int,
    "supersedes": str,
    "reason": str,
    "question": str,
    "hypotheses": [{"name": str, "statement": str, "prediction": str, "direction": str}],
    "data": {
        "domain": str,
        "L": [int],
        "seeds": [int],
        "replicas": int,
        "exclusions": [{"run_id": str, "reason": str}],
        "tau_c": {"source": str, "pilot_run": str, "pilot_sha256": str},
        "ramps": {"kind": str, "x": _NUMBER},
        "nulls": {"kind": str, "eps_L": [_NUMBER]},
        "staircase": {"kind": str, "temperature": _NUMBER},
    },
    "indicators": {
        "names": [str],
        "observables": [str],
        "cadences": [str],
        "thin": _NUMBER,
        "lag": int,
        "detrend": str,
        "direction": {},
    },
    "event": {"definition": str, "horizons": [_NUMBER], "segment": _NUMBER},
    "scoring": {
        "positives": str,
        "negatives": str,
        "auc": str,
        "alpha": _NUMBER,
        "ci_level": _NUMBER,
        "bootstrap": int,
        "surrogates": {"primary": str, "secondary": str, "n": int, "cells": str},
        "alarms": {
            "indicators": [str],
            "far": _NUMBER,
            "report_per": _NUMBER,
            "band": [_NUMBER],
            "every": _NUMBER,
        },
    },
    "fss": {
        "static": [{"quantity": str, "role": str}],
        "dynamic": {
            "statistic": str,
            "indicators": [str],
            "observable": str,
            "cadence": str,
            "search": [_NUMBER],
            "expected_inverse_nu": _NUMBER,
            "tolerance": _NUMBER,
            "min_draw_share": _NUMBER,
        },
    },
    "disclosure": {"exploratory": str, "decisions": str, "confirmatory": str, "links": [str]},
    "sensitivity": {
        "window": [_NUMBER],
        "bandwidth": [_NUMBER],
        "headline": {"window": _NUMBER, "bandwidth": _NUMBER},
    },
    "trials": {"budget": int, "counted": str},
    "deviations": {"policy": str, "log": [{"date": str, "change": str, "reason": str}]},
}
_LATENT_SCHEMA: dict[str, Any] = _SCHEMA | {
    "baseline": {"prereg": str, "result": str},
    "latents": {
        "table": str,
        "columns": [str],
        "training_seeds": [int],
        "validation_seed": int,
        "steps": int,
        "extraction": str,
        "spares": [int],
        "checkpoints": [
            {
                "role": str,
                "model_seed": int,
                "training": str,
                "trained_sha256": str,
                "twin_sha256": str,
            }
        ],
    },
    "indicators": _SCHEMA["indicators"] | {"pairs": [[str]]},
    "scoring": _SCHEMA["scoring"]
    | {
        "surrogates": _SCHEMA["scoring"]["surrogates"] | {"n_ablation": int},
        "alarms": {  # every judged pair's: indicators.pairs
            "far": _NUMBER,
            "report_per": _NUMBER,
            "band": [_NUMBER],
            "every": _NUMBER,
        },
        "margins": {"pairing": str, "seeds": str, "sign_test": bool, "ablations": str},
    },
    "fss": {
        "static": _SCHEMA["fss"]["static"],
        "dynamic": {
            "statistics": [{"observable": str, "indicator": str, "extremum": str, "role": str}],
            "cadence": str,
            "search": [_NUMBER],
            "expected_inverse_nu": _NUMBER,
            "tolerance": _NUMBER,
            "edge_rule": str,
            "min_draw_share": _NUMBER,
        },
    },
    "probes": {
        "targets": [str],
        "controls": [str],
        "fit": str,
        "test": str,
        "ridge_alpha": _NUMBER,
        "selectivity": str,
        "per": str,
    },
}


def _check(value: Any, schema: Any, where: str, problems: list[str]) -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            problems.append(f"{where}: expected a table")
            return
        for key, sub in schema.items():
            if key not in value:
                problems.append(f"{where}.{key}: missing".lstrip("."))
            else:
                _check(value[key], sub, f"{where}.{key}".lstrip("."), problems)
    elif isinstance(schema, list):
        if not isinstance(value, list):
            problems.append(f"{where}: expected an array")
            return
        for i, item in enumerate(value):
            _check(item, schema[0], f"{where}[{i}]", problems)
    elif (isinstance(value, bool) and schema is not bool) or not isinstance(value, schema):
        # TOML booleans are not numbers
        names = " or ".join(
            t.__name__ for t in (schema if isinstance(schema, tuple) else (schema,))
        )
        problems.append(f"{where}: expected {names}, got {value!r}")


def _strings(value: Any, where: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _strings(v, f"{where}.{k}".lstrip("."))]
    if isinstance(value, list):
        return [s for i, v in enumerate(value) for s in _strings(v, f"{where}[{i}]")]
    return [(where, value)] if isinstance(value, str) else []


def _placeholder(value: Any) -> bool:
    """A value still to be set: a latent draft may leave one where it has to choose, and strict
    refuses it anywhere."""
    return isinstance(value, str) and _PLACEHOLDER.search(value) is not None


def _latent_problems(doc: dict[str, Any]) -> list[str]:
    """What a file with a [latents] section can have wrong beyond the checks every file takes."""
    found: list[str] = []
    ind, latents, dyn = doc["indicators"], doc["latents"], doc["fss"]["dynamic"]
    columns = set(latents["columns"])
    if latents["table"] != "latents":
        found.append("latents.table: 'latents', the folder extract writes under each run")
    if not columns or not columns <= LATENT_COLUMNS:
        found.append(f"latents.columns: a non-empty subset of {sorted(LATENT_COLUMNS)}")
    if not set(ind["observables"]) <= columns:
        found.append("indicators.observables: must be latents.columns")
    pairs = [tuple(pair) for pair in ind["pairs"]]
    if (
        not pairs
        or len(set(pairs)) != len(pairs)
        or any(
            len(pair) != 2 or pair[0] not in ind["observables"] or pair[1] not in ind["names"]
            for pair in pairs
        )
    ):
        found.append("indicators.pairs: distinct [observable, indicator] pairs of the scored ones")
    for observable, own in ind.get("direction_by_observable", {}).items():
        if observable not in ind["observables"] or not isinstance(own, dict):
            found.append(f"indicators.direction_by_observable.{observable}: a scored observable")
            continue
        for name, value in own.items():
            if value not in ("greater", "less") and not _placeholder(value):
                where = f"indicators.direction_by_observable.{observable}.{name}"
                found.append(f"{where}: 'greater' or 'less', the expected trend")
    entries = latents["checkpoints"]
    for i, entry in enumerate(entries):
        if entry["role"] not in CHECKPOINT_ROLES:
            found.append(f"latents.checkpoints[{i}].role: one of {sorted(CHECKPOINT_ROLES)}")
    if not any(entry["role"] == "primary" for entry in entries):
        found.append("latents.checkpoints: at least one primary")
    seeds = [entry["model_seed"] for entry in entries]
    if len(set(seeds)) != len(seeds):
        found.append("latents.checkpoints: one entry per model_seed")
    named = [entry["training"] for entry in entries if not _placeholder(entry["training"])]
    if len(set(named)) != len(named):
        found.append("latents.checkpoints: one entry per training")
    trained = {*latents["training_seeds"], latents["validation_seed"]}
    if trained & set(doc["data"]["seeds"]):
        found.append(
            "latents.training_seeds: a simulator seed this file scores is never trained on"
        )
    if doc["scoring"]["surrogates"]["n_ablation"] < 1:
        found.append("scoring.surrogates.n_ablation: the surrogates per ablation checkpoint")
    if not _SLUG.fullmatch(doc["baseline"]["prereg"]):
        found.append("baseline.prereg: the <id>-v<k> whose runs this file reads")
    if not dyn["statistics"]:
        found.append("fss.dynamic.statistics: at least one")
    for i, entry in enumerate(dyn["statistics"]):
        where = f"fss.dynamic.statistics[{i}]"
        if entry["observable"] not in columns:
            found.append(f"{where}.observable: one of latents.columns")
        if entry["indicator"] not in LATENT_INDICATORS:
            found.append(f"{where}.indicator: one of {sorted(LATENT_INDICATORS)}")
        if entry["extremum"] not in EXTREMA and not _placeholder(entry["extremum"]):
            found.append(f"{where}.extremum: one of {sorted(EXTREMA)}")
        if entry["role"] not in DYNAMIC_ROLES:
            found.append(f"{where}.role: one of {sorted(DYNAMIC_ROLES)}")
    for i, row in enumerate(doc["fss"]["static"]):
        tolerance = row.get("tolerance", 0.0)
        number = isinstance(tolerance, _NUMBER) and not isinstance(tolerance, bool)
        if not number and not _placeholder(tolerance):
            found.append(f"fss.static[{i}].tolerance: a number")
    probes = doc["probes"]
    if not probes["targets"] or not set(probes["targets"]) <= PROBE_TARGETS:
        found.append(f"probes.targets: a non-empty subset of {sorted(PROBE_TARGETS)}")
    if set(probes["controls"]) != PROBE_CONTROLS:
        found.append(f"probes.controls: {sorted(PROBE_CONTROLS)}")
    if probes["ridge_alpha"] <= 0:
        found.append("probes.ridge_alpha: positive, per sample")
    return found


def problems(doc: dict[str, Any], strict: bool = False) -> list[str]:
    """Everything wrong with a pre-registration document; strict also refuses placeholders."""
    found: list[str] = []
    latent = "latents" in doc
    _check(doc, _LATENT_SCHEMA if latent else _SCHEMA, "", found)
    if found:
        return found
    if not _SLUG.fullmatch(doc["id"]) and not _PLACEHOLDER.search(doc["id"]):
        found.append(f"id: {doc['id']!r} is not a lowercase slug")
    if doc["version"] < 1:
        found.append("version: must be >= 1")
    if doc["supersedes"] and not doc["reason"].strip():
        found.append("reason: an amendment must say why it supersedes " + doc["supersedes"])
    if not doc["hypotheses"]:
        found.append("hypotheses: at least one")
    for i, h in enumerate(doc["hypotheses"]):
        if h["direction"] not in DIRECTIONS and not (latent and _placeholder(h["direction"])):
            found.append(f"hypotheses[{i}].direction: one of {sorted(DIRECTIONS)}")
    ind, data, fss = doc["indicators"], doc["data"], doc["fss"]
    for key, allowed in (
        ("names", LATENT_INDICATORS if latent else INDICATORS),
        ("observables", LATENT_COLUMNS if latent else OBSERVABLES),
        ("cadences", {"matched"} if latent else CADENCES),  # a latents table holds saved frames
    ):
        if not ind[key] or not set(ind[key]) <= allowed:
            found.append(f"indicators.{key}: a non-empty subset of {sorted(allowed)}")
    if ind["detrend"] not in {"gaussian", "none"}:
        found.append("indicators.detrend: 'gaussian' or 'none'")
    if len(data["L"]) < 2:
        found.append("data.L: at least two sizes; a finite-size fit needs them")
    if not data["seeds"]:
        found.append("data.seeds: at least one")
    directions = ind["direction"]
    for name in ind["names"]:
        value = directions.get(name)
        if value not in ("greater", "less") and not (latent and _placeholder(value)):
            found.append(f"indicators.direction.{name}: 'greater' or 'less', the expected trend")
    if (data["ramps"]["kind"], data["nulls"]["kind"]) != ("ramp", "fixed"):
        found.append("data: ramps are kind 'ramp' and nulls kind 'fixed'")
    window = data["ramps"].get("eps_L")  # optional: without it, any window is scored
    window_ok = (
        isinstance(window, list)
        and len(window) == 2
        and all(isinstance(e, _NUMBER) and not isinstance(e, bool) for e in window)
        and window[0] > window[1]
    )
    if window is not None and not window_ok:
        found.append("data.ramps.eps_L: [start, end] in eps_L, start above end (a cooling ramp)")
    if data["tau_c"]["source"] != "pilot":
        found.append("data.tau_c.source: 'pilot', tau_c as measured per L (never fitted)")
    quantities, roles = QUANTITIES, ROLES
    if latent:
        columns = doc["latents"]["columns"]
        quantities = {f"{stat}_{column}" for stat in LATENT_STATISTICS for column in columns}
        roles = LATENT_ROLES
    for i, row in enumerate(fss["static"]):
        if row["quantity"] not in quantities or row["role"] not in roles:
            found.append(
                f"fss.static[{i}]: quantity in {sorted(quantities)}, role in {sorted(roles)}"
            )
        if (
            row["role"] in ("positive-control", "pipeline-check")
            and not {"expected", "tolerance"} <= row.keys()
        ):
            judged = row["role"].replace("-", " ")
            found.append(f"fss.static[{i}]: a {judged} needs expected and tolerance")
        if row["role"] == "reference" and "tolerance" not in row:
            found.append(f"fss.static[{i}]: a reference needs its own tolerance")
    dyn = fss["dynamic"]
    if latent:
        found += _latent_problems(doc)
    else:
        if dyn["statistic"] not in STATISTICS:
            found.append(f"fss.dynamic.statistic: one of {sorted(STATISTICS)}")
        if not set(dyn["indicators"]) <= set(ind["names"]):
            found.append("fss.dynamic.indicators: must be scored indicators")
        if dyn["observable"] not in ind["observables"]:
            found.append("fss.dynamic.observable: must be a scored observable")
        if not set(doc["scoring"]["alarms"]["indicators"]) <= set(ind["names"]):
            found.append("scoring.alarms.indicators: must be scored indicators")
    if dyn["cadence"] not in ind["cadences"]:
        found.append("fss.dynamic.cadence: one of indicators.cadences, the verdict's cadence")
    if not 0 < dyn["min_draw_share"] <= 1:
        found.append("fss.dynamic.min_draw_share: a share of the bootstrap draws, in (0, 1]")
    search = dyn["search"]
    if len(search) != 2 or search[0] <= search[1]:
        found.append("fss.dynamic.search: [from eps_L, to eps_L], the hot edge first")
    elif (
        window_ok
        and isinstance(window, list)  # window_ok says so; pyright needs it said here
        and not (window[0] >= search[0] and window[1] <= search[1])
    ):
        found.append("fss.dynamic.search: must lie inside data.ramps.eps_L, the window ramps run")
    sens = doc["sensitivity"]
    if (
        sens["headline"]["window"] not in sens["window"]
        or sens["headline"]["bandwidth"] not in sens["bandwidth"]
    ):
        found.append("sensitivity.headline: must be a cell of the grid")
    if len(doc["scoring"]["alarms"]["band"]) != 2:
        found.append("scoring.alarms.band: [from eps_L, to eps_L]")
    tests = doc["scoring"]["surrogates"]
    if tests["cells"] not in {"headline", "all"}:
        found.append("scoring.surrogates.cells: 'headline' or 'all'")
    for key in ("primary", "secondary"):
        if tests[key] not in SURROGATES:
            found.append(f"scoring.surrogates.{key}: one of {sorted(SURROGATES)}")
    if ind["detrend"] == "none" and "ar1" in (tests["primary"], tests["secondary"]):
        found.append(
            "scoring.surrogates: ar1 needs detrending; fitted to a trending series it draws "
            "near-random walks with no trend"
        )
    if strict:
        found += [
            f"{where}: placeholder {text!r}"
            for where, text in _strings(doc)
            if _PLACEHOLDER.search(text)
        ]
        if not _SHA256.fullmatch(data["tau_c"]["pilot_sha256"]):
            found.append("data.tau_c.pilot_sha256: the pilot record's sha256 (64 hex digits)")
        pins = data["staircase"].get("tau_sha256")
        unpinned = [
            L
            for L in data["L"]
            if not (isinstance(pins, dict) and _SHA256.fullmatch(str(pins.get(str(L)))))
        ]
        if unpinned:
            found.append(
                "data.staircase.tau_sha256: per L, the sha256 of the tau.json the staircases' "
                f"dwell is read from (64 hex digits); none for L = {unpinned}"
            )
        if doc["trials"]["budget"] < cells(doc):
            found.append(f"trials.budget: one scoring run evaluates {cells(doc)} cells")
        for i, entry in enumerate(doc["latents"]["checkpoints"] if latent else ()):
            for key in ("trained_sha256", "twin_sha256"):
                if not _SHA256.fullmatch(entry[key]):
                    found.append(f"latents.checkpoints[{i}].{key}: a sha256 (64 hex digits)")
    return found


def load(path: Path) -> Prereg:
    path = Path(path)
    raw = path.read_bytes()
    try:
        doc = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PreregError(f"{path}: {exc}") from exc
    found = problems(doc)
    if found:
        raise PreregError(f"{path}:\n  " + "\n  ".join(found))
    return Prereg(path, doc, hashlib.sha256(raw).hexdigest())


def _git(root: Path, *args: str, binary: bool = False) -> Any:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if done.returncode != 0:
        raise PreregError(f"git {' '.join(args)}: {done.stderr.decode(errors='replace').strip()}")
    return done.stdout if binary else done.stdout.decode().strip()


def repo_root(path: Path) -> Path:
    start = path if path.is_dir() else path.parent
    return Path(_git(start, "rev-parse", "--show-toplevel"))


def latest(root: Path, prereg_id: str) -> Path:
    """The highest version of paper/prereg/<id>-v<k>.toml."""
    pattern = re.compile(re.escape(prereg_id) + r"-v(\d+)\.toml")
    versions = {
        int(m.group(1)): p
        for p in (root / PREREG_DIR).glob(f"{prereg_id}-v*.toml")
        if (m := pattern.fullmatch(p.name))
    }
    if not versions:
        raise PreregError(f"no {PREREG_DIR.as_posix()}/{prereg_id}-v<k>.toml under {root}")
    return versions[max(versions)]


def freeze(root: Path, prereg_id: str) -> str:
    """Tag the commit holding the latest version of a pre-registration; returns the tag."""
    path = latest(root, prereg_id)
    prereg = load(path)
    relative = path.relative_to(root).as_posix()
    if prereg.name != path.stem:
        raise PreregError(
            f"{relative}: holds {prereg.name}, so it must be named {prereg.name}.toml"
        )
    found = problems(prereg.doc, strict=True)
    if found:
        raise PreregError(f"{relative} is not ready to freeze:\n  " + "\n  ".join(found))
    if not _git(root, "ls-files", "--", relative):
        raise PreregError(f"{relative} is not committed")
    if _git(root, "status", "--porcelain", "--", relative):
        raise PreregError(f"{relative} differs from HEAD: commit it before the freeze")
    if _git(root, "tag", "--list", prereg.tag):
        raise PreregError(f"{prereg.tag} exists: a frozen version is never re-tagged; amend it")
    baseline_commit(prereg)  # the runs it reads must already be fixed
    message = f"Pre-registration {prereg.name}, sha256 {prereg.sha256}"
    _git(root, "tag", "-a", prereg.tag, "-m", message, "HEAD")
    return prereg.tag


@dataclass(frozen=True)
class Frozen:
    tag: str
    commit: str


def frozen(prereg: Prereg) -> Frozen:
    """The prereg's tag and commit, if the file is byte for byte the blob frozen at its tag."""
    root = repo_root(prereg.path)
    relative = prereg.path.resolve().relative_to(root.resolve()).as_posix()
    if not _git(root, "tag", "--list", prereg.tag):
        command = f"python -m tipping_point.ews prereg freeze {prereg.id}"
        raise PreregError(f"{prereg.name} is not frozen: run `{command}`")
    blob = _git(root, "cat-file", "blob", f"{prereg.tag}:{relative}", binary=True)
    if hashlib.sha256(blob).hexdigest() != prereg.sha256:
        raise PreregError(f"{relative} differs from the version frozen at {prereg.tag}")
    return Frozen(prereg.tag, _git(root, "rev-parse", f"{prereg.tag}^{{commit}}"))


def baseline_commit(prereg: Prereg) -> str | None:
    """The commit of the tag of the version a file's [baseline] names, whose runs it reads and which
    therefore made them; None for a file with no [baseline]."""
    baseline = prereg.doc.get("baseline")
    if baseline is None:
        return None
    root, tag = repo_root(prereg.path), f"prereg/{baseline['prereg']}"
    if not _git(root, "tag", "--list", tag):
        raise PreregError(f"{prereg.name} reads the runs of {baseline['prereg']}: no tag {tag}")
    return _git(root, "rev-parse", f"{tag}^{{commit}}")


def stamp(prereg: Prereg, freeze_ref: Frozen) -> dict[str, Any]:
    return {
        "id": prereg.id,
        "version": prereg.version,
        "sha256": prereg.sha256,
        "tag": freeze_ref.tag,
        "commit": freeze_ref.commit,
    }


def trials_path(data_dir: Path) -> Path:
    return Path(data_dir) / "ews" / "trials.jsonl"


def log_trial(data_dir: Path, record: dict[str, Any]) -> None:
    """Append one evaluation, flushed and synced before returning; the log only ever grows."""
    path = trials_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"logged_ns": time.time_ns(), **record}, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def trials(data_dir: Path, prereg: Prereg) -> int:
    """Evaluations logged so far under this prereg's id and version."""
    path = trials_path(data_dir)
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        count += (record["prereg"]["id"], record["prereg"]["version"]) == (
            prereg.id,
            prereg.version,
        )
    return count


def cells(doc: dict[str, Any]) -> int:
    """AUC evaluations one scoring run makes: one trial each. A file with a [latents] section makes
    them per checkpoint and per pair: each primary checkpoint and its twin on the whole grid, where
    an indicator in RAW has no bandwidth and so one cell per window, each ablation checkpoint and
    its twin at the headline cell alone."""
    ind, sens = doc["indicators"], doc["sensitivity"]
    sizes = len(doc["data"]["L"]) * len(doc["event"]["horizons"]) * len(ind["cadences"])
    grid = len(sens["window"]) * len(sens["bandwidth"])
    if "latents" not in doc:
        return len(ind["names"]) * len(ind["observables"]) * sizes * grid
    raw = sum(name in RAW for _, name in ind["pairs"])
    primary = (len(ind["pairs"]) - raw) * grid + raw * len(sens["window"])
    roles = [entry["role"] for entry in doc["latents"]["checkpoints"]]
    return 2 * sizes * sum(primary if role == "primary" else len(ind["pairs"]) for role in roles)


def write_result(
    data_dir: Path,
    prereg: Prereg,
    freeze_ref: Frozen,
    tables: dict[str, list[dict[str, Any]]],
    inputs: list[dict[str, Any]],
    params: dict[str, Any],
) -> Path:
    """A scoring run's output in the shared layout: manifest.json first, each non-empty table as
    <table>/00000000000000000000.parquet keyed by `cell`, and complete.json once all are synced."""
    run_id = engine.run_id(prereg.name)
    run_dir = Path(data_dir) / "ews" / run_id
    run_dir.mkdir(parents=True)
    filled = {name: rows for name, rows in tables.items() if rows}
    manifest = {
        "format": 1,
        "domain": "ews",
        "run_id": run_id,
        "producer": {"name": "tipping_point.ews", "version": __version__, **io.code_version()},
        "started_ns": time.time_ns(),
        "key": "cell",
        "tables": list(filled),
        "params": {"prereg": stamp(prereg, freeze_ref), "inputs": inputs, **params},
    }
    io.write_json(run_dir / "manifest.json", manifest)
    files: dict[str, str] = {}
    for name, rows in filled.items():
        table = pa.Table.from_pylist([{"cell": i, **row} for i, row in enumerate(rows)])
        path = run_dir / name / f"{0:020d}.parquet"
        path.parent.mkdir()
        io.write_file(path, lambda f, table=table: pq.write_table(table, f, compression="zstd"))
        files[path.relative_to(run_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "ended_ns": time.time_ns(),
        "last_key": max((len(rows) - 1 for rows in filled.values()), default=-1),
        "rows": {name: len(rows) for name, rows in filled.items()},
        "clean": True,
        "files": files,
    }
    io.write_json(run_dir / "complete.json", record)
    return run_dir
