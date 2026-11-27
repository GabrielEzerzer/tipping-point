# The held-back test (docs/market.md, "Coin panel"): a frozen plan, panel/test.toml, pins the runs,
# commits, arms and decision rule; the data step builds the whole-span runs, and the run step checks
# every record before it settles the support, then scores the test span once per plan.

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import subprocess
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .. import _git
from ..ising import io
from ..market import archive
from . import classical, compare, heldback, labels, prefix, sequence, span, store

PLAN = Path(__file__).with_name("test.toml")
REPO = Path(__file__).resolve().parents[5]
PLAN_FILE = "python/tipping_point/src/tipping_point/panel/test.toml"
CODE_PATHS = (
    "python/tipping_point/src",
    "python/tipping_point/pyproject.toml",
    "pyproject.toml",
    "uv.lock",
)
MISSING = re.compile(r"/([A-Z0-9]+)-(?:1m|metrics|fundingRate)-(\d{4})-(\d{2})(?:-(\d{2}))?\.zip$")
DOMAIN = "panel-test"
BAD = ("refused", "failed", "unreadable")
COMMIT = re.compile(r"[0-9a-f]{40}")
NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
LOG = Callable[[str], None]
TOP = {
    "end",
    "code_sha256",
    "approved",
    "runs",
    "arms",
    "pairs",
    "primary",
    "side",
    "horizon",
    "rule",
}
RUNS = {"labels": {"run", "manifest_sha256"}, "inputs": {"run", "manifest_sha256"}}
FINAL = {"run", "manifest_sha256", "plan_sha256", "ladder_sha256"}
RULE = {f.name for f in dataclasses.fields(heldback.Rule)}
ARM = {
    "sequence": ({"name", "kind", "model", "variant", "seeds", "encoders"}, {"readout", "c"}),
    "classical": ({"name", "kind", "model", "variant", "c"}, {"readout"}),
    "rung": ({"name", "kind", "rung"}, set()),
}


@dataclass(frozen=True)
class Pin:
    run: str
    manifest_sha256: str


@dataclass(frozen=True)
class Test:
    path: Path
    text: str
    sha256: str
    end: date
    code_sha256: str
    approved: tuple[str, ...]
    labels: Pin
    inputs: Pin
    final: Pin
    plan_sha256: str
    ladder_sha256: str
    arms: tuple[heldback.Arm, ...]
    pairs: tuple[tuple[str, str], ...]
    primary: tuple[str, str]
    side: str
    horizon: int
    rule: heldback.Rule

    @property
    def limit(self) -> date:
        return heldback.limit_after(self.end)


def _unread(name: str) -> bytes:
    raise ValueError(f"{name}: a model is read only from the checked final run")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _whole(value: Any) -> bool:
    return type(value) is int


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _keys(raw: Any, need: set[str], what: str, optional: set[str] | None = None) -> list[str]:
    if not isinstance(raw, dict):
        return [f"{what} is not a table"]
    problems = [f"{what}: unknown key {k}" for k in sorted(set(raw) - need - (optional or set()))]
    return problems + [f"{what}: missing key {k}" for k in sorted(need - set(raw))]


def _pin(raw: dict[str, Any], what: str, problems: list[str]) -> Pin:
    if not (isinstance(raw.get("run"), str) and compare.RUN_ID.fullmatch(raw["run"])):
        problems.append(f"{what}: run {raw.get('run')!r} is not a run id")
    for key in sorted(set(raw) - {"run"}):
        if not (isinstance(raw[key], str) and compare.SHA256.fullmatch(raw[key])):
            problems.append(f"{what}: {key} {raw[key]!r} is not a sha256")
    return Pin(str(raw.get("run")), str(raw.get("manifest_sha256")))


def _pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_text(v) for v in value)


def _sequence_arm(raw: dict[str, Any], what: str, problems: list[str]) -> heldback.Arm:
    model, readout, c, seeds = raw["model"], raw.get("readout"), raw.get("c"), raw["seeds"]
    if model not in sequence.MODELS:
        problems.append(f"{what}: model {model!r} (one of {list(sequence.MODELS)})")
    if model == "supervised":
        if "readout" in raw or "c" in raw:
            problems.append(f"{what}: the supervised model takes no readout or c")
    elif readout not in sequence.READOUTS or not (c is not None and _number(c) and c > 0):
        problems.append(
            f"{what}: readout {readout!r} (one of {list(sequence.READOUTS)}) and c {c!r} (a "
            "number above 0) are needed"
        )
    if not (
        isinstance(seeds, list)
        and seeds
        and all(_whole(s) and s >= 0 for s in seeds)
        and len(set(seeds)) == len(seeds)
    ):
        problems.append(f"{what}: seeds {seeds!r} (given, distinct, whole and not negative)")
    encoders = raw["encoders"]
    if not (
        isinstance(encoders, dict)
        and encoders
        and all(isinstance(v, str) and compare.SHA256.fullmatch(v) for v in encoders.values())
    ):
        problems.append(f"{what}: encoders must map each model file to its sha256")
        encoders = {}
    return heldback.SequenceArm(
        raw["name"],
        str(model),
        str(raw["variant"]),
        readout,
        float(c) if c is not None and _number(c) else None,
        tuple(seeds) if isinstance(seeds, list) else (),
        dict(encoders),
        _unread,
    )


def _arm(raw: Any, problems: list[str]) -> heldback.Arm | None:
    kind = raw.get("kind") if isinstance(raw, dict) else None
    what = f"arm {raw.get('name') if isinstance(raw, dict) else raw!r}"
    if not isinstance(kind, str) or kind not in ARM:
        problems.append(f"{what}: kind {kind!r} (one of {list(ARM)})")
        return None
    need, optional = ARM[kind]
    found = _keys(raw, need, what, optional)
    if found:
        problems += found
        return None
    if not (isinstance(raw["name"], str) and NAME.fullmatch(raw["name"])):
        problems.append(f"{what}: a name is letters, digits, _ and - only")
        return None
    texts = [k for k in ("model", "variant", "readout", "rung") if k in raw and not _text(raw[k])]
    if texts:
        problems.append(f"{what}: {', '.join(texts)} must be text")
        return None
    if kind == "rung":
        return heldback.RungArm(raw["name"], raw["rung"])
    if kind == "sequence":
        return _sequence_arm(raw, what, problems)
    model, readout, c = raw["model"], raw.get("readout"), raw["c"]
    if (model, "readout" in raw) not in (("rung", False), ("window", True)) or not (
        _number(c) and c > 0
    ):
        problems.append(
            f"{what}: a classical arm is model rung with no readout or model window with one, "
            f"and c a number above 0"
        )
        return None
    return heldback.ClassicalArm(raw["name"], classical.Config(model, raw["variant"], readout, c))


def load(path: Path = PLAN) -> Test:
    if not path.is_file():
        raise ValueError(
            f"the test plan is not frozen: there is no {path}, and the test runs only from it"
        )
    data = path.read_bytes()
    text = data.decode("utf-8")
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from None
    problems = _keys(raw, TOP, "the plan")
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    end = raw["end"]
    if type(end) is not date or end < span.HELD_BACK:
        problems.append(f"end {end!r} (a date from {span.HELD_BACK} on)")
    code = raw["code_sha256"]
    if not (isinstance(code, str) and compare.SHA256.fullmatch(code)):
        problems.append(f"code_sha256 {code!r} is not a sha256")
    approved = raw["approved"]
    if not (
        isinstance(approved, list)
        and approved
        and all(isinstance(a, str) and COMMIT.fullmatch(a) for a in approved)
        and len(set(approved)) == len(approved)
    ):
        problems.append("approved must list distinct full commit sha1s")
    runs = raw["runs"]
    pins: dict[str, Pin] = {}
    found = _keys(runs, set(RUNS) | {"final"}, "runs")
    problems += found
    if not found:
        for key, need in (*RUNS.items(), ("final", FINAL)):
            what = f"runs.{key}"
            listed = _keys(runs[key], need, what)
            problems += listed
            if not listed:
                pins[key] = _pin(runs[key], what, problems)
    arms: list[heldback.Arm] = []
    if not (isinstance(raw["arms"], list) and raw["arms"]):
        problems.append("arms must list at least one arm")
    else:
        arms = [a for a in (_arm(r, problems) for r in raw["arms"]) if a is not None]
    folded = [a.name.casefold() for a in arms]
    if len(set(folded)) != len(folded):
        problems.append("arm names must differ, ignoring case")
    pairs, primary = raw["pairs"], raw["primary"]
    if not (isinstance(pairs, list) and pairs and all(_pair(p) for p in pairs)):
        problems.append("pairs must list pairs of arm names")
        pairs = []
    if not _pair(primary):
        problems.append("primary must be a pair of arm names")
        primary = ["", ""]
    if not _text(raw["side"]):
        problems.append("side must be text")
    if not _whole(raw["horizon"]):
        problems.append(f"horizon {raw['horizon']!r} must be a whole number")
    rule = None
    found = _keys(raw["rule"], RULE, "rule")
    problems += found
    if not found:
        try:
            rule = heldback.Rule(**raw["rule"])
        except ValueError as exc:
            problems.append(str(exc))
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    assert rule is not None
    final = runs["final"]
    return Test(
        path=path,
        text=text,
        sha256=compare._digest(data),
        end=end,
        code_sha256=code,
        approved=tuple(approved),
        labels=pins["labels"],
        inputs=pins["inputs"],
        final=pins["final"],
        plan_sha256=final["plan_sha256"],
        ladder_sha256=final["ladder_sha256"],
        arms=tuple(arms),
        pairs=tuple((a, b) for a, b in pairs),
        primary=(primary[0], primary[1]),
        side=raw["side"],
        horizon=raw["horizon"],
        rule=rule,
    )


def _git_at(root: Path, *args: str) -> str:
    env = {key: value for key, value in os.environ.items() if key not in _git._REDIRECTS}
    try:
        out = subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"the running code's check needs git and the repository at {root}: {exc}"
        ) from None
    return out.stdout.decode("utf-8")


def code_sha256(root: Path = REPO) -> str:
    out = _git_at(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", *CODE_PATHS)
    lines = sorted(line for line in out.split("\0") if line and _path(line) != PLAN_FILE)
    for path in CODE_PATHS:
        if not any(_path(line) == path or _path(line).startswith(path + "/") for line in lines):
            raise ValueError(
                f"{root}: HEAD tracks no {path}, which the running code's sha256 covers"
            )
    return compare._digest("\n".join(lines).encode("utf-8"))


def _path(line: str) -> str:
    return line.split("\t", 1)[1]


def checkout(commit: str, root: Path = REPO) -> None:
    head = _git_at(root, "rev-parse", "HEAD").strip()
    if head != commit:
        raise ValueError(f"{root}: HEAD is {head}, not the running code's commit {commit}")
    if _git_at(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(
            f"{root}: the checkout has uncommitted changes or untracked files; the test runs "
            f"only from a clean commit"
        )
    ignored = _git_at(
        root, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", CODE_PATHS[0]
    )
    loadable = sorted(p for p in ignored.split("\0") if p.endswith((".py", ".pyd", ".so")))
    if loadable:
        raise ValueError(
            f"{root}: ignored files Python could import sit in the source tree, outside the code "
            f"sha256: {', '.join(loadable)}"
        )
    if not _git_at(root, "ls-tree", "--full-tree", "--name-only", "HEAD", "--", PLAN_FILE):
        raise ValueError(
            f"{root}: HEAD does not track {PLAN_FILE}; the test runs only from a committed plan"
        )


def running(test: Test) -> dict[str, Any]:
    found = io.code_version()
    if not found.get("git_sha") or found.get("dirty") is not False:
        raise ValueError(
            f"the test runs only from a clean commit: the running code is commit "
            f"{found.get('git_sha')} with dirty {found.get('dirty')}"
        )
    package = Path(__file__).resolve().parents[1]
    source = io._source_tree(package)
    if source is None or source.resolve() != REPO / "python" / "tipping_point" / "pyproject.toml":
        raise ValueError(
            f"the test runs only from the source tree of the checkout at {REPO}, and the "
            f"imported package {package} is not it"
        )
    checkout(found["git_sha"])
    digest = code_sha256()
    if digest != test.code_sha256:
        raise ValueError(
            f"the running code's sha256 is {digest}, not the plan's code_sha256 "
            f"{test.code_sha256}: the test runs only the code the plan names"
        )
    return {
        "git_sha": found["git_sha"],
        "dirty": False,
        "code_sha256": digest,
        "uv_lock_sha256": labels.uv_lock(),
    }


def approved(producer: Any, test: Test, what: str) -> str:
    compare.clean_code(producer, what)
    sha = producer["git_sha"]
    if sha not in test.approved:
        raise ValueError(f"{what} was made by commit {sha}, which the test plan does not approve")
    return sha


def earlier(folder: Path, sha: str) -> list[dict[str, Any]]:
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"{folder} is not a folder: the test runs are kept in a folder there")
    found: list[dict[str, Any]] = []
    for entry in sorted(folder.iterdir()) if folder.is_dir() else []:
        try:
            recorded = io.read_json(entry / "manifest.json")["params"]["plan_sha256"]
            done = (entry / "complete.json").is_file()
            outcome = (
                io.read_json(entry / "complete.json")["checks"].get("outcome") if done else None
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise ValueError(
                f"{entry}: its manifest.json or complete.json cannot be read for the plan it ran "
                f"and its outcome, so it cannot be told from a run of this plan"
            ) from None
        if recorded == sha:
            state = "finished" if done else "did not finish"
            raise ValueError(
                f"{entry.name} already ran this plan (sha256 {sha}) and {state}: one test per plan"
            )
        found.append(
            {"run": entry.name, "plan_sha256": recorded, "finished": done, "outcome": outcome}
        )
    return found


def _sha_of(record: Any) -> Any:
    return record.get("sha256") if isinstance(record, Mapping) else None


def _iso(value: Any) -> bool:
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


@dataclass(frozen=True)
class Development:
    labels: Path
    inputs: Path
    labels_params: dict[str, Any]
    inputs_params: dict[str, Any]
    inputs_sha256: str
    grid_sha256: str
    commits: dict[str, str]


def development(test: Test, data_dir: Path, seen: compare.Held) -> Development:
    span.held_back_root(span.HELD_BACK, data_dir)
    found: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    commits = {}
    for key, domain, pin in (
        ("labels", "labels", test.labels),
        ("inputs", "panel-inputs", test.inputs),
    ):
        run_dir = data_dir / domain / pin.run
        what = f"the development {key} run {pin.run}"
        manifest, complete, _, digest = compare.finished(
            run_dir, what, domain, key == "labels", seen
        )
        if digest != pin.manifest_sha256:
            raise ValueError(f"{what}: the sha256 of its manifest.json is not the plan's")
        commits[key] = approved(manifest.get("producer"), test, what)
        if "limit" in manifest["params"]:
            raise ValueError(f"{what} was made with a raised limit, so it is not development data")
        found[key] = (run_dir, manifest, complete)
    lp = found["labels"][1]["params"]
    what = f"the development labels run {test.labels.run}"
    symbols = lp.get("symbols")
    if not (isinstance(symbols, list) and symbols and all(_text(s) for s in symbols)):
        raise ValueError(f"{what}: it records no list of coins")
    for key in ("start", "dev_start"):
        if not _iso(lp.get(key)):
            raise ValueError(f"{what}: its {key} {lp.get(key)!r} is not a date")
    params = found["inputs"][1]["params"]
    what = f"the development inputs run {test.inputs.run}"
    if (params.get("labels_run"), params.get("labels_manifest_sha256")) != (
        test.labels.run,
        test.labels.manifest_sha256,
    ):
        raise ValueError(f"{what} was not made from the pinned labels run")
    grid = _sha_of(params.get("labels_grid"))
    table = found["inputs"][2]["files"].get("inputs.parquet")
    for name, digest in (("labels grid", grid), ("inputs table", table)):
        if not (isinstance(digest, str) and compare.SHA256.fullmatch(digest)):
            raise ValueError(f"{what}: it records no sha256 for its {name}")
    return Development(
        labels=found["labels"][0],
        inputs=found["inputs"][0],
        labels_params=lp,
        inputs_params=params,
        inputs_sha256=table,
        grid_sha256=grid,
        commits=commits,
    )


@dataclass(frozen=True)
class Final:
    run: Path
    params: dict[str, Any]
    files: dict[str, str]
    plan: sequence.Plan
    commits: dict[str, str]
    pieces: list[dict[str, Any]]


def _merged(params: Mapping[str, Any]) -> bool:
    return "sources" in params or "pieces_producer" in params


def final(test: Test, data_dir: Path, dev: Development, seen: compare.Held) -> Final:
    what = f"the final sequence run {test.final.run}"
    seq_dir = compare.sequence_dir(test.final.run, data_dir, what)
    manifest, complete, _, digest = compare.finished(seq_dir, what, "panel-sequence", False, seen)
    if digest != test.final.manifest_sha256:
        raise ValueError(f"{what}: the sha256 of its manifest.json is not the plan's")
    params = manifest["params"]
    commits = {"final": approved(manifest.get("producer"), test, what)}
    chain = [(seq_dir.name, params)]
    pieces: list[dict[str, Any]] = []
    if _merged(params):
        if params.get("pieces_bound") is not True:
            raise ValueError(f"{what}: it does not record every run merged into it as bound")
        commits["pieces"] = approved(
            params.get("pieces_producer"), test, f"{what}: the runs merged into it"
        )
        sources = params.get("sources")
        if not (isinstance(sources, list) and sources):
            raise ValueError(f"{what}: it records no runs merged into it")
        named = [s.get("run") if isinstance(s, Mapping) else None for s in sources]
        if len({repr(n) for n in named}) != len(named):
            raise ValueError(f"{what}: it records a merged run twice")
        models: dict[str, Any] = {}
        for source in sources:
            source = source if isinstance(source, Mapping) else {}
            link = f"{what}: the merged run {source.get('run')}"
            piece = compare.sequence_dir(source.get("run"), data_dir, link)
            pm, _, _, pd = compare.finished(piece, link, "panel-sequence", False, seen)
            if pd != source.get("manifest_sha256"):
                raise ValueError(f"{link}: the sha256 of its manifest.json is not the one merged")
            commit = approved(pm.get("producer"), test, link)
            if _merged(pm["params"]):
                raise ValueError(f"{link}: it is itself a merged run")
            saved = pm["params"].get("encoders")
            if not isinstance(saved, Mapping):
                raise ValueError(f"{link}: it records no models")
            for name, model in saved.items():
                if name in models:
                    raise ValueError(f"{link}: {name} is recorded by another merged run too")
                models[name] = model
            chain.append((piece.name, pm["params"]))
            pieces.append({"run": piece.name, "manifest_sha256": pd, "commit": commit})
        if models != params.get("encoders"):
            raise ValueError(
                f"{what}: the models its merged runs record, by name and sha256, are not the ones "
                f"it records"
            )
    texts = compare.plans_of(seq_dir, complete, params, test.ladder_sha256, what)
    if complete["files"]["plan.toml"] != test.plan_sha256:
        raise ValueError(f"{what}: its plan.toml is not the plan the test plan pins")
    plan = sequence.load(seq_dir / "plan.toml", seq_dir / "ladder.toml", texts)
    if not sequence.fit_only(plan):
        raise ValueError(f"{what}: its plan is not the fit-only fold of a forward plan")
    if plan.ladder.end != span.END:
        raise ValueError(f"{what}: its ladder plan ends on {plan.ladder.end}, not {span.END}")
    want = {
        "inputs run": (test.inputs.run, dev.inputs_sha256),
        "labels grid": (test.labels.run, dev.grid_sha256),
    }
    for name, p in chain:
        got = {
            "inputs run": (p.get("inputs_run"), _sha_of(p.get("inputs"))),
            "labels grid": (p.get("labels_run"), _sha_of(p.get("labels_grid"))),
        }
        for key, value in want.items():
            if got[key] != value:
                raise ValueError(
                    f"{what}: {name} records the {key} {got[key][0]} with sha256 {got[key][1]}, "
                    f"not the pinned development inputs run's {value[0]} with {value[1]}"
                )
    return Final(seq_dir, params, complete["files"], plan, commits, pieces)


def _kept(models: dict[str, bytes], what: str) -> Callable[[str], bytes]:
    def read(name: str) -> bytes:
        if name not in models:
            raise ValueError(f"{what}: {name} is not one of its checked models")
        return models[name]

    return read


def bound_arms(test: Test, last: Final) -> list[heldback.Arm]:
    fold = heldback.final_fold(last.plan)
    saved = last.params.get("encoders") or {}
    out: list[heldback.Arm] = []
    for arm in test.arms:
        if isinstance(arm, heldback.SequenceArm):
            what = f"arm {arm.name}"
            names = {
                sequence.encoder_name(arm.variant, heldback.SCHEME, fold, s, arm.model)
                for s in arm.seeds
            }
            if set(arm.encoders) != names:
                raise ValueError(
                    f"{what} pins {sorted(arm.encoders)}, not the final run's models for its "
                    f"seeds, {sorted(names)}"
                )
            models: dict[str, bytes] = {}
            for name, digest in sorted(arm.encoders.items()):
                if saved.get(name) != digest:
                    raise ValueError(
                        f"{what}: {name} is not the model the final run's manifest records"
                    )
                if last.files.get(name) != digest:
                    raise ValueError(
                        f"{what}: {name} is not the model the final run's complete.json records"
                    )
                data = compare._read(last.run / name, what)
                if compare._digest(data) != digest:
                    raise ValueError(
                        f"{what}: {last.run / name}: its sha256 differs from the plan's"
                    )
                models[name] = data
            arm = dataclasses.replace(arm, read=_kept(models, what))
        out.append(arm)
    return out


@dataclass(frozen=True)
class Pinned:
    dev: Development
    last: Final
    evaluation: Any
    arms: list[heldback.Arm]


def pinned(test: Test, data_dir: Path, seen: compare.Held) -> Pinned:
    dev = development(test, data_dir, seen)
    last = final(test, data_dir, dev, seen)
    evaluation = heldback.evaluation_plan(last.plan.ladder, test.end)
    heldback.arms_check(test.arms, last.plan)
    names = [a.name for a in test.arms]
    heldback.check(names, test.pairs, test.primary, test.side, test.horizon, evaluation)
    return Pinned(dev, last, evaluation, bound_arms(test, last))


def existing(folder: Path, what: str) -> Path | None:
    runs = sorted(folder.iterdir()) if folder.is_dir() else []
    for run in runs:
        if not (run / "complete.json").is_file():
            raise ValueError(
                f"{run} has no complete.json: a run is finished only once complete.json is "
                f"written last, so this {what} run did not finish; remove it and run the data "
                f"step again"
            )
    if len(runs) > 1:
        raise ValueError(
            f"{folder} holds {len(runs)} {what} runs ({', '.join(r.name for r in runs)}); the "
            f"test takes exactly one, so remove all but one"
        )
    return runs[0] if runs else None


def missing_files(manifest: Mapping[str, Any], end: date, what: str) -> list[dict[str, Any]]:
    urls = manifest.get("missing_sources")
    if not isinstance(urls, list):
        raise ValueError(f"{what}: it records no list of missing archive files")
    days: dict[str, list[tuple[date, date]]] = {s: [] for s in manifest["params"]["symbols"]}
    for url in urls:
        m = MISSING.search(url) if isinstance(url, str) else None
        if m is None or m[1] not in days:
            raise ValueError(f"{what}: {url!r} is not an archive file of one of its coins")
        year, month, day = int(m[2]), int(m[3]), m[4]
        first = date(year, month, int(day) if day else 1)
        days[m[1]].append((first, span.last_day(year, m[3], day or "")))
    rows = []
    for symbol, found in days.items():
        late = [(a, min(b, end)) for a, b in found if b >= span.HELD_BACK]
        rows.append(
            {
                "symbol": symbol,
                "missing": len(found),
                "missing_in_test": len(late),
                "first": min(a for a, _ in late).isoformat() if late else None,
                "last": max(b for _, b in late).isoformat() if late else None,
            }
        )
    return rows


def whole_labels(
    run_dir: Path, test: Test, dev: Development, commit: str, seen: compare.Held
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    what = f"the whole-span labels run {run_dir.name}"
    manifest, _, _, digest = compare.finished(run_dir, what, "labels", True, seen)
    _current(manifest, test, commit, what)
    params = manifest["params"]
    if params.get("end") != test.end.isoformat():
        raise ValueError(f"{what}: its grid ends on {params.get('end')}, not on {test.end}")
    for key in ("symbols", "start", "dev_start"):
        if params.get(key) != dev.labels_params[key]:
            raise ValueError(f"{what}: its {key} is not the development labels run's")
    return manifest, digest, missing_files(manifest, test.end, what)


def _files_read(entries: Any, what: str) -> dict[str, str]:
    if not isinstance(entries, list):
        raise ValueError(f"{what}: it records no list of archive files read")
    files: dict[str, str] = {}
    for entry in entries:
        url = entry.get("url") if isinstance(entry, dict) else None
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if not (
            isinstance(url, str)
            and isinstance(digest, str)
            and compare.SHA256.fullmatch(digest)
            and url not in files
        ):
            raise ValueError(f"{what}: {entry!r} is not one archive file read with its sha256")
        files[url] = digest
    return files


def same_archive(labels: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    read = _files_read(labels.get("sources"), "the whole-span labels run")
    took = _files_read(inputs["params"].get("sources"), "the whole-span inputs run")
    absent = set(labels["missing_sources"])
    problems = []
    for url in sorted(read.keys() | took.keys()):
        if url in read and url in took:
            if read[url] != took[url]:
                problems.append(f"{url} read by labels as {read[url]}, by inputs as {took[url]}")
        elif url in took:
            how = "missing for labels" if url in absent else "not read by labels"
            problems.append(f"{url} {how} and read by inputs")
        else:
            problems.append(f"{url} read by labels and not by inputs")
    if problems:
        raise ValueError(
            f"the whole-span labels and inputs runs read different archive files ("
            f"{'; '.join(problems)}): remove both whole-span runs and run the data step again"
        )


def whole_inputs(
    run_dir: Path,
    labels_dir: Path,
    labels: Mapping[str, Any],
    labels_sha: str,
    test: Test,
    commit: str,
    seen: compare.Held,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    what = f"the whole-span inputs run {run_dir.name}"
    manifest, complete, _, digest = compare.finished(run_dir, what, "panel-inputs", False, seen)
    _current(manifest, test, commit, what)
    params = manifest["params"]
    if (params.get("labels_run"), params.get("labels_manifest_sha256")) != (
        labels_dir.name,
        labels_sha,
    ):
        raise ValueError(f"{what}: it does not record the whole-span labels run's manifest")
    same_archive(labels, manifest)
    return manifest, complete, digest


def _current(manifest: Mapping[str, Any], test: Test, commit: str, what: str) -> None:
    producer = manifest.get("producer")
    compare.clean_code(producer, what)
    assert isinstance(producer, Mapping)
    if producer["git_sha"] != commit:
        raise ValueError(
            f"{what} was made by commit {producer['git_sha']}, not the running commit {commit}; "
            "the data step and the run step run at one commit"
        )
    if manifest["params"].get("limit") != test.limit.isoformat():
        raise ValueError(
            f"{what}: it records the limit {manifest['params'].get('limit')}, not {test.limit}"
        )


def prefixed(
    found: Mapping[str, Any],
    test: Test,
    dev: Development,
    whole: Mapping[str, str],
    complete: Mapping[str, Any],
) -> None:
    want = {
        "manifests": {
            "labels": whole["labels"],
            "inputs": whole["inputs"],
            "dev_labels": test.labels.manifest_sha256,
            "dev_inputs": test.inputs.manifest_sha256,
        },
        "tables": {
            "inputs": complete["files"].get("inputs.parquet"),
            "dev_inputs": dev.inputs_sha256,
        },
    }
    for part, expected in want.items():
        got = found[part]
        for key, digest in expected.items():
            if got.get(key) != digest:
                raise ValueError(
                    f"the prefix check read the {key} {part[:-1]} with sha256 {got.get(key)}, not "
                    f"the {digest} checked against the plan and the records"
                )


def downloads(done: Mapping[str, Any], root: Path) -> None:
    bad = {
        status: sorted(name for name, entry in done.items() if entry["status"] == status)
        for status in BAD
    }
    if not any(bad.values()):
        return
    parts = [
        f"{len(bad[s])} {s} (the first {bad[s][0]}): the download was not written, so a re-run "
        f"fetches {'it' if len(bad[s]) == 1 else 'them'} again"
        for s in ("refused", "failed")
        if bad[s]
    ]
    if bad["unreadable"]:
        paths = [root / name for name in bad["unreadable"]]
        parts.append(
            f"{len(paths)} unreadable: each matches its checksum and is kept, so a re-run reads it "
            f"again and fails the same way; remove each file and its checksum file to fetch it "
            f"again: " + ", ".join(f"{p} and {archive._checksum_path(p)}" for p in paths)
        )
    raise ValueError("no labels run was made: " + "; ".join(parts))


def data(
    data_dir: Path, get: archive.Get, path: Path = PLAN, log: LOG = lambda line: None
) -> dict[str, Any]:
    test = load(path)
    commit = running(test)["git_sha"]
    data_dir = Path(data_dir).resolve()
    seen: compare.Held = {}
    dev = pinned(test, data_dir, seen).dev
    held = data_dir / span.HELD_BACK_DIR
    root = held / "archives" / "binance"
    labels_dir = existing(held / "labels", "labels")
    inputs_dir = existing(held / "panel-inputs", "inputs")
    if labels_dir is None and inputs_dir is not None:
        raise ValueError(f"{inputs_dir} is a whole-span inputs run with no whole-span labels run")
    p = dev.labels_params
    symbols, start, dev_start = (
        p["symbols"],
        date.fromisoformat(p["start"]),
        date.fromisoformat(p["dev_start"]),
    )
    made = []
    if labels_dir is None:
        downloads(heldback.fetch_whole(symbols, start, test.end, root, get, log), root)
        labels_dir = heldback.labels_whole(root, held, symbols, start, dev_start, test.end, log)
        made.append(labels_dir)
    labels, labels_sha, missing = whole_labels(labels_dir, test, dev, commit, seen)
    if inputs_dir is None:
        inputs_dir = heldback.inputs_whole(labels_dir, root, held, test.end, log)
        made.append(inputs_dir)
    _, complete, inputs_sha = whole_inputs(
        inputs_dir, labels_dir, labels, labels_sha, test, commit, seen
    )
    checked = prefix.check(labels_dir, inputs_dir, dev.labels, dev.inputs)
    prefixed(checked, test, dev, {"labels": labels_sha, "inputs": inputs_sha}, complete)
    return {
        "labels": labels_dir,
        "inputs": inputs_dir,
        "made": made,
        "prefix": checked,
        "missing": missing,
    }


def read_tables(
    labels_dir: Path,
    inputs_dir: Path,
    manifest: Mapping[str, Any],
    complete: Mapping[str, Any],
    columns: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    record = manifest["params"]["labels_grid"]
    parts = Path(record["file"]).parts
    if len(parts) != 2 or parts[0] != "grid":
        raise ValueError(f"{record['file']} is not a grid table inside {labels_dir}")
    grid_data = compare._read(labels_dir / record["file"], "the whole-span labels run")
    inputs_data = compare._read(inputs_dir / "inputs.parquet", "the whole-span inputs run")
    for data, want, what in (
        (grid_data, record["sha256"], "grid"),
        (inputs_data, complete["files"].get("inputs.parquet"), "inputs table"),
    ):
        if compare._digest(data) != want:
            raise ValueError(f"the whole-span {what}: its sha256 differs from the one recorded")
    grid_table = pq.read_table(pa.BufferReader(grid_data))
    table = pq.read_table(pa.BufferReader(inputs_data), columns=["ts_ns", "symbol", *columns])
    if not (
        table["ts_ns"].equals(grid_table["ts_ns"]) and table["symbol"].equals(grid_table["symbol"])
    ):
        raise ValueError("the whole-span inputs' rows are not its grid's rows")
    values = {n: table[n].to_numpy() for n in columns}
    grid = {n: grid_table[n].to_numpy(zero_copy_only=False) for n in grid_table.column_names}
    read = {
        "labels_grid": {"run": labels_dir.name, **record},
        "inputs": {
            "run": inputs_dir.name,
            "file": "inputs.parquet",
            "sha256": compare._digest(inputs_data),
        },
    }
    return values, grid, read


@dataclass(frozen=True)
class Checked:
    test: Test
    code: dict[str, Any]
    held: Path
    earlier: list[dict[str, Any]]
    plan: sequence.Plan
    arms: list[heldback.Arm]
    evaluation: Any
    values: dict[str, np.ndarray]
    grid: dict[str, np.ndarray]
    params: dict[str, Any]


def checked(data_dir: Path, path: Path = PLAN) -> Checked:
    test = load(path)
    code = running(test)
    data_dir = Path(data_dir).resolve()
    span.held_back_root(span.HELD_BACK, data_dir)
    held = data_dir / span.HELD_BACK_DIR
    before = earlier(held / DOMAIN, test.sha256)
    seen: compare.Held = {}
    p = pinned(test, data_dir, seen)
    dev, last, plan = p.dev, p.last, p.last.plan
    labels_dir = existing(held / "labels", "labels")
    inputs_dir = existing(held / "panel-inputs", "inputs")
    if labels_dir is None or inputs_dir is None:
        raise ValueError(
            f"{held} holds no whole-span labels and inputs runs: run the data step first"
        )
    labels, labels_sha, missing = whole_labels(labels_dir, test, dev, code["git_sha"], seen)
    manifest, complete, inputs_sha = whole_inputs(
        inputs_dir, labels_dir, labels, labels_sha, test, code["git_sha"], seen
    )
    found = prefix.check(labels_dir, inputs_dir, dev.labels, dev.inputs)
    prefixed(found, test, dev, {"labels": labels_sha, "inputs": inputs_sha}, complete)
    values, grid, read = read_tables(
        labels_dir, inputs_dir, manifest, complete, plan.ladder.columns
    )
    heldback.covered(grid, plan.ladder.start, test.end)
    params = {
        "plan_sha256": test.sha256,
        "plan_file": test.path.name,
        "end": test.end.isoformat(),
        "limit": test.limit.isoformat(),
        "code": code,
        "threads": plan.threads,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "runtime": sequence.runtime(),
        "approved": list(test.approved),
        "runs": {
            "labels": {**asdict(test.labels), "commit": dev.commits["labels"]},
            "inputs": {**asdict(test.inputs), "commit": dev.commits["inputs"]},
            "final": {
                **asdict(test.final),
                "plan_sha256": test.plan_sha256,
                "ladder_sha256": test.ladder_sha256,
                "commits": last.commits,
                "pieces": last.pieces,
            },
            "whole_labels": {
                "run": labels_dir.name,
                "manifest_sha256": labels_sha,
                "commit": code["git_sha"],
            },
            "whole_inputs": {
                "run": inputs_dir.name,
                "manifest_sha256": inputs_sha,
                "commit": code["git_sha"],
            },
        },
        "read": read,
        "prefix": found,
        "missing": missing,
        "earlier": before,
        "arms": [_described(a) for a in test.arms],
        "pairs": [list(p) for p in test.pairs],
        "primary": list(test.primary),
        "side": test.side,
        "horizon": test.horizon,
        "rule": asdict(test.rule),
    }
    return Checked(test, code, held, before, plan, p.arms, p.evaluation, values, grid, params)


def _described(arm: heldback.Arm) -> dict[str, Any]:
    if isinstance(arm, heldback.SequenceArm):
        out = asdict(dataclasses.replace(arm, read=_unread))
        out.pop("read")
        return {"kind": "sequence", **out, "seeds": list(arm.seeds)}
    if isinstance(arm, heldback.ClassicalArm):
        return {"kind": "classical", "name": arm.name, **asdict(arm.config)}
    return {"kind": "rung", "name": arm.name, "rung": arm.rung}


def finite(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")


def _json(record: Any) -> str:
    return json.dumps(finite(record), indent=2, default=_plain, allow_nan=False) + "\n"


def run(
    data_dir: Path, path: Path = PLAN, log: LOG = lambda line: None
) -> tuple[Path, dict[str, Any]]:
    began = time.perf_counter()
    c = checked(data_dir, path)
    test = c.test
    names = ["comparisons", "scores", "folds", "rows", *(f"arm-{a.name}" for a in c.arms)]
    run_dir = store.begin(c.held, DOMAIN, "test", names, c.params)
    files: dict[str, str] = {}
    store.put(run_dir, files, texts={"test.toml": test.text})
    found = finite(
        heldback.support(c.values, c.grid, c.evaluation, test.side, test.horizon, test.rule)
    )
    store.put(run_dir, files, texts={"support.json": _json(found)})
    log(f"support: {found['scoreable']} scoreable quarters, causes {found['causes']}")
    out: dict[str, Any] = {"support": found}
    rows: dict[str, int] = {}
    if found["causes"]:
        out["decision"] = heldback.decide(found, (), test.primary, test.rule)
    else:
        torch.set_num_threads(c.plan.threads)
        tables, used = heldback.scored(c.values, c.grid, c.plan, test.end, c.arms, log)
        out |= heldback.paired(
            c.values,
            c.grid,
            c.evaluation,
            tables,
            test.pairs,
            test.primary,
            test.side,
            test.horizon,
            log,
        )
        out["decision"] = heldback.decide(found, out["comparisons"], test.primary, test.rule)
        out |= {"tables": tables, "encoders": used}
        written = {
            "comparisons": pa.Table.from_pylist(out["comparisons"]),
            "scores": pa.Table.from_pylist(out["scores"]),
            "folds": pa.Table.from_pylist(out["folds"]),
            "rows": pa.Table.from_pylist(
                [{"horizon": h, **row} for h, part in out["accounting"].items() for row in part]
            ),
            **{f"arm-{name}": table for name, table in tables.items()},
        }
        store.put(run_dir, files, tables=written)
        rows = {name: table.num_rows for name, table in written.items()}
    seconds = time.perf_counter() - began
    texts = {
        "decision.json": _json(out["decision"]),
        "report.md": report(out, c, run_dir.name, seconds),
    }
    store.put(run_dir, files, texts=texts)
    checks = {
        "outcome": out["decision"]["outcome"],
        "encoders_used": out.get("encoders", {}),
        "seconds": round(seconds, 1),
    }
    store.end(run_dir, rows, files, checks)
    return run_dir, out | {"seconds": seconds}


def _f(value: Any, signed: bool = False) -> str:
    return "-" if value is None else compare._f(float(value), signed)


def report(out: Mapping[str, Any], c: Checked, run_id: str, seconds: float) -> str:
    test, decision, found = c.test, out["decision"], out["support"]
    rule = test.rule
    p = c.params
    first, second = test.primary
    read = f"{rule.bound} on {'complete draws only' if rule.draws == 'complete' else 'all draws'}"
    scored = "comparisons" in out
    fitted = (
        f"every arm was fitted once on the development hours before {span.HELD_BACK} and scores "
        f"the test span, which is folded by calendar quarter"
        if scored
        else "the test span is folded by calendar quarter, and no arm was fitted or scored"
    )
    lines = [
        "# Held-back test",
        "",
        f"Run `{run_id}` of the test plan `{p['plan_file']}` (sha256 {test.sha256}). Test span "
        f"{span.HELD_BACK} to {test.end}; {fitted}.",
        "",
        "## Outcome",
        "",
        f"**{decision['outcome']}**",
        "",
    ]
    if decision["causes"]:
        lines += ["The support fell short of the rule, so no arm was scored:", ""]
        lines += [f"- {cause}" for cause in decision["causes"]]
        lines.append("")
    else:
        lines += [
            f"The primary pair {first} - {second}, {test.side}, {test.horizon} h, gives "
            f"{decision['read']} = {_f(decision['lower'], True)}. The rule reads it as supported "
            f"only when it is above zero.",
            "",
        ]
    lines += [
        "## Rule",
        "",
        f"The rule as read from the plan: the lower bound is {read}; the support needs at least "
        f"{rule.min_quarters} scoreable quarters, a share of complete draws of at least "
        f"{rule.min_complete}, and at least {rule.min_supported} quarters with a positive in at "
        f"least {rule.min_blocks} blocks. It was settled from the labels and from which rows have "
        f"every input, with no model output, before any arm was scored, and written to "
        f"support.json first.",
        "",
        f"Scoreable quarters {found['scoreable']}, supported {found['supported']}, share of "
        f"complete draws {_f(found['complete_draws'])} of {found['draws']}.",
        "",
    ]
    sides = ("long", "short") if test.side == compare.COMPOSITE else (test.side,)
    head = "| quarter | blocks | rows | " + " | ".join(
        f"{s} positives | {s} blocks with a positive" for s in sides
    )
    lines += [head + " | scoreable | supported |", "|---" * (5 + 2 * len(sides)) + "|"]
    for q in found["quarters"]:
        counts = " | ".join(
            f"{q[f'{s}_positives']} | {q[f'{s}_blocks_with_positive']}" for s in sides
        )
        lines.append(
            f"| {q['quarter']} | {q['blocks']} | {q['rows']} | {counts} | "
            f"{'yes' if q['scoreable'] else 'no'} | {'yes' if q['supported'] else 'no'} |"
        )
    if "comparisons" in out:
        header = [
            "| pair | scheme | horizon | side | estimate | se | se q | q2.5 | q5 | q97.5 | > 0 | "
            "complete | draws | folds > 0 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        (top,) = [row for row in out["comparisons"] if row["primary"]]
        lines += ["", "## Primary", "", *header, compare._row(top)]
        lines += [compare._row(top, "_complete", " (complete draws)"), "", "## Other pairs", ""]
        others = [compare._row(row) for row in out["comparisons"] if not row["primary"]]
        lines += [*header, *others] if others else ["None."]
    lines += [
        "",
        "## Missing archive files",
        "",
        "Files the archive did not publish when the data step fetched it, from the whole-span "
        "labels run's record. Without a day's klines or metrics the cascade windows over that "
        "day are not evaluated, so the labels censor each hour whose look-back or horizon "
        "reaches them, and inputs read from those bars can be missing. A funding file changes "
        "no label, and the inputs run refuses one missing before the coin's last bar.",
        "",
        f"| coin | missing in all | missing from {span.HELD_BACK} | first | last |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {m['symbol']} | {m['missing']} | {m['missing_in_test']} | {m['first'] or '-'} | "
        f"{m['last'] or '-'} |"
        for m in p["missing"]
    ]
    lines += ["", "## Earlier test runs", ""]
    if c.earlier:
        lines += ["| run | plan sha256 | finished | outcome |", "|---|---|---|---|"]
        lines += [
            f"| `{e['run']}` | {e['plan_sha256']} | {'yes' if e['finished'] else 'no'} | "
            f"{e['outcome'] or '-'} |"
            for e in c.earlier
        ]
    else:
        lines.append("None.")
    runs = p["runs"]
    code = p["code"]
    lines += [
        "",
        "## Provenance",
        "",
        f"- Running code: commit {code['git_sha']} (clean), code sha256 {code['code_sha256']} "
        f"(the plan's), uv.lock {code['uv_lock_sha256']}, torch threads {p['threads']}. "
        + (
            "This code refitted every classical and rung arm and every probe."
            if scored
            else "No arm or probe was refitted."
        ),
        f"- Installed packages, recorded and not checked against uv.lock: torch {p['torch']}, "
        f"numpy {p['numpy']}, pyarrow {p['pyarrow']}; CPU {p['runtime']['cpu']}, capability "
        f"{p['runtime']['capability']}.",
    ]
    for key, label in (("labels", "labels"), ("inputs", "inputs")):
        r = runs[key]
        lines.append(
            f"- Development {label} run `{r['run']}`, manifest sha256 {r['manifest_sha256']}, "
            f"commit {r['commit']}."
        )
    f = runs["final"]
    lines.append(
        f"- Final sequence run `{f['run']}`, manifest sha256 {f['manifest_sha256']}, plan.toml "
        f"{f['plan_sha256']}, ladder.toml {f['ladder_sha256']}, commits "
        + ", ".join(f"{k} {v}" for k, v in f["commits"].items())
        + "."
    )
    lines += [
        f"  - Merged from `{piece['run']}`, manifest sha256 {piece['manifest_sha256']}, commit "
        f"{piece['commit']}."
        for piece in f["pieces"]
    ]
    for key, label in (("whole_labels", "labels"), ("whole_inputs", "inputs")):
        r = runs[key]
        lines.append(
            f"- Whole-span {label} run `{r['run']}`, manifest sha256 {r['manifest_sha256']}, "
            f"commit {r['commit']}, limit {p['limit']}."
        )
    rows = p["prefix"]["rows"]
    lines.append(
        f"- Prefix check: before {span.HELD_BACK} the whole-span tables equal the development "
        f"tables bit for bit in {rows['thresholds']} threshold rows, {rows['inputs']} inputs rows "
        f"and {rows['grid']} grid rows, outside each horizon's last hours of {span.END.year}."
    )
    lines.append(f"- Approved commits: {', '.join(test.approved)}.")
    for name, used in (out.get("encoders") or {}).items():
        lines += [f"- Arm {name} read {key}, sha256 {digest}." for key, digest in used.items()]
    lines += ["", f"Wall time {_f(seconds / 60)} minutes."]
    return "\n".join(lines) + "\n"
