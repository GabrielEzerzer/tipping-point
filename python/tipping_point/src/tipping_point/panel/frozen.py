# The held-back test (docs/market.md, "Coin panel"): a frozen plan, panel/test.toml, pins the runs,
# commits, arms and decision rule; the data step builds the whole-span runs, and the run step checks
# every record before it settles the support, then scores the test span once per plan.

from __future__ import annotations

import dataclasses
import json
import math
import re
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

from ..ising import io
from ..market import archive
from . import classical, compare, heldback, labels, prefix, sequence, span, store

PLAN = Path(__file__).with_name("test.toml")
DOMAIN = "panel-test"
BAD = ("refused", "failed", "unreadable")
COMMIT = re.compile(r"[0-9a-f]{40}")
NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
LOG = Callable[[str], None]
TOP = {"end", "approved", "runs", "arms", "pairs", "primary", "side", "horizon", "rule"}
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
        None if c is None else float(c),
        tuple(seeds) if isinstance(seeds, list) else (),
        dict(encoders),
        _unread,
    )


def _arm(raw: Any, problems: list[str]) -> heldback.Arm | None:
    kind = raw.get("kind") if isinstance(raw, dict) else None
    what = f"arm {raw.get('name') if isinstance(raw, dict) else raw!r}"
    if kind not in ARM:
        problems.append(f"{what}: kind {kind!r} (one of {list(ARM)})")
        return None
    need, optional = ARM[kind]
    found = _keys(raw, need, what, optional)
    if found:
        problems += found
        return None
    if not (isinstance(raw["name"], str) and NAME.fullmatch(raw["name"])):
        problems.append(f"{what}: a name is letters, digits, _ and - only")
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


def running() -> dict[str, Any]:
    found = io.code_version()
    if not found.get("git_sha") or found.get("dirty") is not False:
        raise ValueError(
            f"the test runs only from a clean commit: the running code is commit "
            f"{found.get('git_sha')} with dirty {found.get('dirty')}"
        )
    return {"git_sha": found["git_sha"], "dirty": False, "uv_lock_sha256": labels.uv_lock()}


def approved(producer: Any, test: Test, what: str) -> str:
    compare.clean_code(producer, what)
    sha = producer["git_sha"]
    if sha not in test.approved:
        raise ValueError(f"{what} was made by commit {sha}, which the test plan does not approve")
    return sha


def earlier(folder: Path, sha: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for entry in sorted(folder.iterdir()) if folder.is_dir() else []:
        try:
            manifest = io.read_json(entry / "manifest.json")
            recorded = manifest["params"]["plan_sha256"]
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError(
                f"{entry}: no readable manifest.json names the plan this test run ran, so it "
                f"cannot be told from a run of this plan"
            ) from None
        done = (entry / "complete.json").is_file()
        if recorded == sha:
            state = "finished" if done else "did not finish"
            raise ValueError(
                f"{entry.name} already ran this plan (sha256 {sha}) and {state}: one test per plan"
            )
        outcome = io.read_json(entry / "complete.json")["checks"].get("outcome") if done else None
        found.append(
            {"run": entry.name, "plan_sha256": recorded, "finished": done, "outcome": outcome}
        )
    return found


@dataclass(frozen=True)
class Development:
    labels: Path
    inputs: Path
    labels_params: dict[str, Any]
    inputs_params: dict[str, Any]
    inputs_sha256: str
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
    params = found["inputs"][1]["params"]
    if (params.get("labels_run"), params.get("labels_manifest_sha256")) != (
        test.labels.run,
        test.labels.manifest_sha256,
    ):
        raise ValueError(
            f"the development inputs run {test.inputs.run} was not made from the pinned labels run"
        )
    return Development(
        labels=found["labels"][0],
        inputs=found["inputs"][0],
        labels_params=found["labels"][1]["params"],
        inputs_params=params,
        inputs_sha256=found["inputs"][2]["files"]["inputs.parquet"],
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
    if "sources" in params or "pieces_producer" in params:
        if params.get("pieces_bound") is not True:
            raise ValueError(f"{what}: it does not record every run merged into it as bound")
        commits["pieces"] = approved(
            params.get("pieces_producer"), test, f"{what}: the runs merged into it"
        )
        sources = params.get("sources")
        if not (isinstance(sources, list) and sources):
            raise ValueError(f"{what}: it records no runs merged into it")
        for source in sources:
            source = source if isinstance(source, Mapping) else {}
            link = f"{what}: the merged run {source.get('run')}"
            piece = compare.sequence_dir(source.get("run"), data_dir, link)
            pm, _, _, pd = compare.finished(piece, link, "panel-sequence", False, seen)
            if pd != source.get("manifest_sha256"):
                raise ValueError(f"{link}: the sha256 of its manifest.json is not the one merged")
            approved(pm.get("producer"), test, link)
            chain.append((piece.name, pm["params"]))
            pieces.append({"run": piece.name, "manifest_sha256": pd})
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
        "labels grid": (test.labels.run, compare._sha(dev.inputs_params.get("labels_grid"))),
    }
    for name, p in chain:
        got = {
            "inputs run": (p.get("inputs_run"), compare._sha(p.get("inputs"))),
            "labels grid": (p.get("labels_run"), compare._sha(p.get("labels_grid"))),
        }
        for key, value in want.items():
            if got[key] != value:
                raise ValueError(
                    f"{what}: {name} records the {key} {got[key][0]} with sha256 {got[key][1]}, "
                    f"not the pinned development inputs run's {value[0]} with {value[1]}"
                )
    return Final(seq_dir, params, complete["files"], plan, commits, pieces)


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
            for name, digest in sorted(arm.encoders.items()):
                if saved.get(name) != digest or last.files.get(name) != digest:
                    raise ValueError(f"{what}: {name} is not the model the final run records")
                if compare._digest(compare._read(last.run / name, what)) != digest:
                    raise ValueError(
                        f"{what}: {last.run / name}: its sha256 differs from the plan's"
                    )

            def read(name: str, folder: Path = last.run, what: str = what) -> bytes:
                return compare._read(folder / name, what)

            arm = dataclasses.replace(arm, read=read)
        out.append(arm)
    return out


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


def whole_labels(
    run_dir: Path, test: Test, dev: Development, commit: str, seen: compare.Held
) -> str:
    what = f"the whole-span labels run {run_dir.name}"
    manifest, _, _, digest = compare.finished(run_dir, what, "labels", True, seen)
    _current(manifest, test, commit, what)
    params = manifest["params"]
    if params.get("end") != test.end.isoformat():
        raise ValueError(f"{what}: its grid ends on {params.get('end')}, not on {test.end}")
    for key in ("symbols", "start", "dev_start"):
        if params.get(key) != dev.labels_params.get(key):
            raise ValueError(f"{what}: its {key} is not the development labels run's")
    return digest


def whole_inputs(
    run_dir: Path, labels_dir: Path, labels_sha: str, test: Test, commit: str, seen: compare.Held
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


def data(
    data_dir: Path, get: archive.Get, path: Path = PLAN, log: LOG = lambda line: None
) -> dict[str, Any]:
    test = load(path)
    commit = running()["git_sha"]
    data_dir = Path(data_dir).resolve()
    seen: compare.Held = {}
    dev = development(test, data_dir, seen)
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
        done = heldback.fetch_whole(symbols, start, test.end, root, get, log)
        bad = sorted(name for name, entry in done.items() if entry["status"] in BAD)
        if bad:
            raise ValueError(
                f"{len(bad)} archive files were refused, failed or unreadable, the first {bad[0]}; "
                f"no labels run was made, and a re-run fetches them again"
            )
        labels_dir = heldback.labels_whole(root, held, symbols, start, dev_start, test.end, log)
        made.append(labels_dir)
    labels_sha = whole_labels(labels_dir, test, dev, commit, seen)
    if inputs_dir is None:
        inputs_dir = heldback.inputs_whole(labels_dir, root, held, test.end, log)
        made.append(inputs_dir)
    whole_inputs(inputs_dir, labels_dir, labels_sha, test, commit, seen)
    checked = prefix.check(labels_dir, inputs_dir, dev.labels, dev.inputs)
    return {"labels": labels_dir, "inputs": inputs_dir, "made": made, "prefix": checked}


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
    code = running()
    data_dir = Path(data_dir).resolve()
    span.held_back_root(span.HELD_BACK, data_dir)
    held = data_dir / span.HELD_BACK_DIR
    before = earlier(held / DOMAIN, test.sha256)
    seen: compare.Held = {}
    dev = development(test, data_dir, seen)
    last = final(test, data_dir, dev, seen)
    plan = last.plan
    evaluation = heldback.evaluation_plan(plan.ladder, test.end)
    heldback.arms_check(test.arms, plan)
    names = [a.name for a in test.arms]
    heldback.check(names, test.pairs, test.primary, test.side, test.horizon, evaluation)
    arms = bound_arms(test, last)
    labels_dir = existing(held / "labels", "labels")
    inputs_dir = existing(held / "panel-inputs", "inputs")
    if labels_dir is None or inputs_dir is None:
        raise ValueError(
            f"{held} holds no whole-span labels and inputs runs: run the data step first"
        )
    labels_sha = whole_labels(labels_dir, test, dev, code["git_sha"], seen)
    manifest, complete, inputs_sha = whole_inputs(
        inputs_dir, labels_dir, labels_sha, test, code["git_sha"], seen
    )
    found = prefix.check(labels_dir, inputs_dir, dev.labels, dev.inputs)
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
            "whole_labels": {"run": labels_dir.name, "manifest_sha256": labels_sha},
            "whole_inputs": {"run": inputs_dir.name, "manifest_sha256": inputs_sha},
        },
        "read": read,
        "prefix": found,
        "earlier": before,
        "arms": [_described(a) for a in test.arms],
        "pairs": [list(p) for p in test.pairs],
        "primary": list(test.primary),
        "side": test.side,
        "horizon": test.horizon,
        "rule": asdict(test.rule),
    }
    return Checked(test, code, held, before, plan, arms, evaluation, values, grid, params)


def _described(arm: heldback.Arm) -> dict[str, Any]:
    if isinstance(arm, heldback.SequenceArm):
        out = asdict(dataclasses.replace(arm, read=_unread))
        out.pop("read")
        return {"kind": "sequence", **out, "seeds": list(arm.seeds)}
    if isinstance(arm, heldback.ClassicalArm):
        return {"kind": "classical", "name": arm.name, **asdict(arm.config)}
    return {"kind": "rung", "name": arm.name, "rung": arm.rung}


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")


def _json(record: Any) -> str:
    return json.dumps(record, indent=2, default=_plain) + "\n"


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
    found = heldback.support(c.values, c.grid, c.evaluation, test.side, test.horizon, test.rule)
    store.put(run_dir, files, texts={"support.json": _json(found)})
    log(f"support: {found['scoreable']} scoreable quarters, causes {found['causes']}")
    out: dict[str, Any] = {"support": found}
    rows: dict[str, int] = {}
    if found["causes"]:
        out["decision"] = heldback.decide(found, (), test.primary, test.rule)
    else:
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
    return compare._f(float(value), signed)


def report(out: Mapping[str, Any], c: Checked, run_id: str, seconds: float) -> str:
    test, decision, found = c.test, out["decision"], out["support"]
    rule = test.rule
    p = c.params
    first, second = test.primary
    read = f"{rule.bound} on {'complete draws only' if rule.draws == 'complete' else 'all draws'}"
    lines = [
        "# Held-back test",
        "",
        f"Run `{run_id}` of the test plan `{p['plan_file']}` (sha256 {test.sha256}). Test span "
        f"{span.HELD_BACK} to {test.end}; every arm was fitted once on the development hours "
        f"before {span.HELD_BACK} and scores the test span, which is folded by calendar quarter.",
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
        f"least {rule.min_blocks} blocks. It was settled from the labels alone, before any arm "
        f"was scored, and written to support.json first.",
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
        f"- Running code: commit {code['git_sha']} (clean), uv.lock {code['uv_lock_sha256']}.",
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
        f"  - Merged from `{piece['run']}`, manifest sha256 {piece['manifest_sha256']}."
        for piece in f["pieces"]
    ]
    for key, label in (("whole_labels", "labels"), ("whole_inputs", "inputs")):
        r = runs[key]
        lines.append(
            f"- Whole-span {label} run `{r['run']}`, manifest sha256 {r['manifest_sha256']}, "
            f"limit {p['limit']}."
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
