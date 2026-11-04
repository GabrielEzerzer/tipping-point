"""Checks of notebooks/kaggle_ising_runs.py, run by its --self-test or directly:

    uv run --no-sync python notebooks/kaggle_ising_runs.py --self-test
    uv run --no-sync python notebooks/kaggle_ising_runs_selftest.py

Plan validation, the files a run reads from another and the CLI's verdict on every run's arguments;
dependency resolution; the queues (longest first, dependencies, pinning, a failure, a failed
extraction, the deadline, the output cap) with a stand-in for the simulator; the previous session's
runs carried forward and reused or resumed; inputs found and checked against their sha256, and a
final.pt against the complete.json beside it; the wheel and COMMIT.txt checks; run verification,
judged by exit codes and files only; the rates STATUS gives, by checkpoint and L; the kernel folder
against what `kaggle kernels push` checks; invalid plans refused at every door; pilot-28.json,
primary-28.json and the first-read plans, their stage ids, inputs and kernel sources, on stand-in
mounts; v1-confirmatory.json against what the frozen file's `ews score` selects and accepts, its
runs made by the tag's own CLI as manifests, with nothing simulated; ablations-28.json against
primary-28.json and #28's six ablations, each training's command taken by e598e62's own CLI with
nothing trained, and each planted fault red by name.

Last, the whole path on real children: the written kernel runs two local sessions of a tiny plan
(a simulator run, a real `jepa train` stopped by a simulated deadline, `jepa resume`, then a real
`jepa extract` of its final.pt and of its twin, each run checked with `jepa verify`), then a real
extraction fails with another behind it on its queue, then one reads its final.pt out of a kernel
output, all in a venv that prints Kaggle's sitecustomize errors on stderr from every child. No wheel
is built: the venv holds a copy of the package stamped clean, so the outcome does not depend on this
checkout's state.

About five minutes on a laptop CPU. One line per check, then exit 1 if any failed. The gate's
self-test leg (harness/gate.py) runs this and reads its closing "N/M checks passed" and exit code.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import copy
import hashlib
import inspect
import io
import itertools
import json
import os
import re
import runpy
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tomllib
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

PLANS = Path(__file__).resolve().parent / "plans"
COMMIT, OTHER = "c" * 40, "d" * 40
TORCH = "2.14.1+cpu"
RECORD = "20261025T194846Z-tauc-L16-32-64-128"
BASE: dict[str, Any] = {
    "kernel": "tp-self-test",
    "runs": [
        {"name": "a", "command": "pilot", "args": ["--L", 8]},
        {
            "name": "b",
            "command": "staircase",
            "args": ["--L", 8, "--tau", {"run": "a", "file": "tau.json"}],
        },
    ],
}
NOISE = [  # what every child printed on Kaggle's image (tp-tauc-l64-l128's log)
    "Error in sitecustomize; set PYTHONVERBOSE for traceback:",
    "ModuleNotFoundError: No module named 'wrapt'",
]
# written out, not read from the runner or the wheel: a slip in either must turn these red (#126)
CONFIRMATORY_SEEDS = range(6, 11)
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    shown = f": {detail}" if detail != "" and not ok else ""
    print(f"{'PASS' if ok else 'FAIL'}  {name}{shown}", flush=True)


def job_a() -> dict:
    return json.loads((PLANS / "job-a.json").read_text(encoding="utf-8"))


def ctx_at(k: ModuleType, root: Path, device: str = "cpu") -> Any:
    ctx = k.Ctx(
        input=root / "input",
        work=root / "working",
        tmp=root / "tmp",
        device=device,
        poll_s=0.01,
        bench_sweeps=10,
    )
    for folder in (ctx.input, ctx.out / "ising", ctx.out / "logs", ctx.cwd, ctx.helper.parent):
        folder.mkdir(parents=True, exist_ok=True)
    return ctx


def refused(k: ModuleType, plan: dict) -> str:
    try:
        k.load_plan(plan)
    except k.PlanError as exc:
        return str(exc)
    return ""


def unwritten_reads(k: ModuleType, plan: dict) -> list[str]:
    """Each file a run reads from an earlier run that the earlier run's command never writes, and
    each earlier run read with no file, which hands the CLI a folder where it wants a file. Only a
    `jepa extract` reads a run that way: its simulator run is a folder. load_plan takes any path
    inside the folder and no path at all, so such a run first fails on Kaggle as it starts."""
    writes = {"pilot": {"tau.json"}, "jepa train": {k.FINAL, k.TWIN}}
    written = {r["name"]: writes.get(r["command"], ()) for r in plan["runs"]}
    return [
        f"{r['name']} reads {arg['file']} of {arg.get('run')}"
        for r in plan["runs"]
        for arg in r["args"]
        if isinstance(arg, dict)
        and "file" in arg
        and arg["file"] not in written.get(arg.get("run"), ())
    ] + [
        f"{r['name']} reads {arg['run']} with no file"
        for r in plan["runs"]
        if r["command"] != "jepa extract"
        for arg in r["args"]
        if isinstance(arg, dict) and "run" in arg and "file" not in arg
    ]


def plans(k: ModuleType) -> None:
    for path in sorted(PLANS.glob("*.json")):
        plan = json.loads(path.read_text(encoding="utf-8"))
        why = refused(k, plan)
        check(f"plan {path.name} loads", not why, why)
        wrong = unwritten_reads(k, plan)
        check(f"plan {path.name}: each file a run reads, its run writes", not wrong, wrong)
    misread = job_a()
    next(r for r in misread["runs"] if r["name"] == "qs-L64-s12")["args"][-1]["file"] = "taus.json"
    wrong = unwritten_reads(k, misread)
    check(
        "... and a file its run never writes is caught",
        wrong == ["qs-L64-s12 reads taus.json of grid-L64"],
        wrong,
    )
    misread = job_a()
    next(r for r in misread["runs"] if r["name"] == "qs-L64-s12")["args"][-1] = {"run": "grid-L64"}
    wrong = unwritten_reads(k, misread)
    check(
        "... and a run read with no file is caught: --tau would get a folder",
        wrong == ["qs-L64-s12 reads grid-L64 with no file"],
        wrong,
    )
    extraction = copy.deepcopy(JEPA_PLAN)
    extraction["runs"].insert(1, {"name": "sim", "command": "null", "args": []})
    extraction["runs"][2]["args"][1] = {"run": "sim"}
    wrong = refused(k, extraction) or unwritten_reads(k, extraction)
    check("... unless a jepa extract reads it: its simulator run is a folder", not wrong, wrong)
    runs = {r.name: r for r in k.make_runs(job_a())}
    pilots = [r for r in runs.values() if r.command == "pilot"]
    stairs = [r for r in runs.values() if r.command == "staircase"]
    check(
        "job A: three grid pilots with no dependency",
        len(pilots) == 3 and not any(r.deps for r in pilots),
    )
    check(
        "job A: six staircases, each waiting on grid-L64 alone",
        len(stairs) == 6 and all(r.deps == ["grid-L64"] for r in stairs),
    )
    seeds = sorted(int(r.args[r.args.index("--seed") + 1]) for r in stairs)
    check("job A: staircase seeds 11-16", seeds == [11, 12, 13, 14, 15, 16], seeds)


def refusals(k: ModuleType) -> None:
    check("the minimal plan loads", not refused(k, copy.deepcopy(BASE)))
    entry = {"run": RECORD, "file": "pilot.json", "sha256": "0" * 64}

    def twice(p: dict) -> None:
        p["runs"].append(dict(p["runs"][0]))

    def later(p: dict) -> None:
        p["runs"].reverse()

    def itself(p: dict) -> None:
        p["runs"][1]["args"][3] = {"run": "b", "file": "tau.json"}

    def unknown_run(p: dict) -> None:
        p["runs"][1]["args"][3] = {"run": "zz"}

    def outside(p: dict) -> None:
        p["runs"][1]["args"][3]["file"] = "../tau.json"

    def bad_ref(p: dict) -> None:
        p["runs"][1]["args"][3] = {"run": "a", "path": "tau.json"}

    cases = [
        ("a name used twice", twice, "used twice"),
        ("an unknown command", lambda p: p["runs"][0].update(command="tauc"), "command 'tauc'"),
        ("a reference to a later run", later, "not an earlier run"),
        ("a reference to itself", itself, "not an earlier run"),
        ("a reference to no run", unknown_run, "'zz' is not an earlier run"),
        ("a reference of another shape", bad_ref, "reference"),
        ("a path out of the run folder", outside, "relative path"),
        ("--device", lambda p: p["runs"][0]["args"].extend(["--device", "cpu"]), "runner sets it"),
        ("--data-dir=...", lambda p: p["runs"][0]["args"].append("--data-dir=x"), "runner sets it"),
        ("a negative gpu", lambda p: p["runs"][0].update(gpu=-1), "gpu -1"),
        ("a boolean gpu", lambda p: p["runs"][0].update(gpu=True), "gpu True"),
        ("a boolean argument", lambda p: p["runs"][0]["args"].append(True), "a string, a number"),
        ("an unknown run key", lambda p: p["runs"][0].update(gpus=1), "unknown keys"),
        ("an unknown plan key", lambda p: p.update(dataset=["o/d"]), "unknown keys"),
        ("a kernel that is no slug", lambda p: p.update(kernel="TP Job"), "kernel 'TP Job'"),
        ("no runs", lambda p: p.update(runs=[]), "non-empty"),
        (
            "an input the plan does not declare",
            lambda p: p["runs"][1]["args"].extend(["--pilot", {"input": "record"}]),
            "not in the plan's inputs",
        ),
        (
            "inputs with nowhere to look",
            lambda p: p.update(inputs={"record": entry}),
            "kernel source",
        ),
        (
            "an input without its sha256",
            lambda p: p.update(
                kernel_sources=["o/k"], inputs={"record": dict(entry, sha256="abc")}
            ),
            "sha256",
        ),
    ]
    for what, mutate, needle in cases:
        plan = copy.deepcopy(BASE)
        mutate(plan)
        why = refused(k, plan)
        check(f"refused: {what}", needle in why, why or "accepted")
    plan = copy.deepcopy(BASE)
    plan["runs"][0]["command"], plan["runs"][1]["gpu"] = "tauc", -1
    why = refused(k, plan)
    check("every problem is listed at once", "tauc" in why and "gpu -1" in why, why)


def cli(k: ModuleType) -> None:
    errors = k.cli_errors(job_a())
    check("job A: the CLI itself takes every run's arguments", errors == [], errors)
    plan = copy.deepcopy(BASE)
    plan["runs"][0]["args"] += ["--sweepz", 5]  # argparse takes a prefix: --sweep is --sweeps
    errors = k.cli_errors(plan)
    check(
        "the CLI's parser catches a misspelt flag",
        len(errors) == 1 and "unrecognized" in errors[0],
        errors,
    )
    plan = copy.deepcopy(BASE)
    plan["runs"][1]["args"] += ["--init", "warm"]
    errors = k.cli_errors(plan)
    check(
        "... and a bad choice in a run that waits on a file",
        "invalid choice" in str(errors),
        errors,
    )


def shrinking(k: ModuleType) -> None:
    args = {r["name"]: r["args"] for r in job_a()["runs"]}
    small = k.shrink(args["grid-L64"], 8)
    expected = [
        "--L",
        8,
        "--replicas",
        16,
        "--seed",
        6,
        "--therm",
        625,
        "--sweeps",
        3125,
        "--checkpoint-every",
        156,
    ]
    check("shrink 8: grid-L64 at L = 8, sweep counts / 64", small == expected, small)
    check("shrink 8: L never below 4", k.shrink(args["grid-L16"], 8)[1] == 4)
    stair = args["qs-L64-s11"]
    check("shrink: seeds and references untouched", k.shrink(stair, 8)[5:] == stair[5:])
    check("shrink 1 changes nothing", k.shrink(stair, 1) == stair)
    cut = [r["name"] for r in k.only(job_a(), ["grid-L16", "qs-L64-s11"])["runs"]]
    check(
        "--only keeps the named runs and what they read",
        cut == ["grid-L16", "grid-L64", "qs-L64-s11"],
        cut,
    )
    try:
        k.only(job_a(), ["qs-L64-s99"])
        check("--only refuses a run the plan lacks", False)
    except k.PlanError as exc:
        check("--only refuses a run the plan lacks", "qs-L64-s99" in str(exc), exc)


def drive(
    k: ModuleType,
    plan: dict,
    gpus: int,
    estimates: dict[str, float] | None = None,
    outcome: dict[str, tuple[str, str]] | None = None,
    hold: str | None = None,
    budget: Any = None,
    stop_after: float | None = None,
) -> tuple[Any, list[tuple[str, str, int]]]:
    """The plan through a Scheduler whose runs are stand-ins: each takes a moment and ends as
    `outcome` says; `hold` runs until the session is stopped and is then interrupted."""
    runs = k.make_runs(plan)
    for r in runs:
        r.estimate_s = (estimates or {}).get(r.name, 1.0)
    events: list[tuple[str, str, int]] = []
    lock = threading.Lock()

    def execute(gpu: int, run: Any) -> tuple[str, str]:
        with lock:
            events.append(("start", run.name, gpu))
        if run.name == hold:
            while sched.halt is None:
                time.sleep(0.005)
            state = ("interrupted", "stopped at the session deadline; resumable")
        else:
            time.sleep(0.02)
            state = (outcome or {}).get(run.name, ("complete", ""))
        with lock:
            events.append(("end", run.name, gpu))
        return state

    sched = k.Scheduler(runs, gpus, execute, budget=budget)
    if stop_after is not None:
        threading.Timer(stop_after, sched.stop, args=("session deadline",)).start()
    sched.run(lambda: None, 0.01)
    return sched, events


def failed_extractions(k: ModuleType) -> None:
    """One table that fails must not cost the tables behind it, since GPU quota is the scarce
    thing. A failed training or simulator run still stops its queue."""

    def extraction(name: str, *reads: str, gpu: int | None = None) -> dict:
        refs = [{"run": r, "file": k.FINAL} for r in reads]
        return {"name": name, "command": "jepa extract", "gpu": gpu, "args": refs}

    training = {"name": "m", "command": "jepa train", "gpu": 0, "args": []}
    plan = {
        "kernel": "tp-self-test",
        "runs": [
            training,
            extraction("x1", "m", gpu=1),
            extraction("x2", "m", gpu=1),
            extraction("x3", "m", gpu=1),
            extraction("y", "x2", gpu=1),
            extraction("z", "y", gpu=1),
            extraction("x4", "m", gpu=1),
        ],
    }
    sched, events = drive(k, plan, 2, outcome={"x2": ("failed", "exit 1: boom")})
    by = {r.name: r for r in sched.runs}
    states = {name: r.state for name, r in by.items()}
    failed_at = events.index(("end", "x2", 1))
    check(
        "a failed extraction leaves its queue running: the tables behind it start and complete",
        all(states[n] == "complete" for n in ("x1", "x3", "x4"))
        and all(events.index(("start", n, 1)) > failed_at for n in ("x3", "x4")),
        events,
    )
    check("... and its queue is not marked stopped", sched.stopped == {}, sched.stopped)
    check(
        "... it blocks only what reads it, directly or in turn",
        states["y"] == states["z"] == "blocked"
        and by["y"].note == "depends on x2, which failed"
        and by["z"].note == "depends on y, which is blocked",
        [(r.name, r.state, r.note) for r in sched.runs],
    )
    status, detail = k.verdict(sched.runs, sched.stopped)
    check(
        "... STATUS failed names it and what it blocked, and no stopped queue",
        status == "failed"
        and "x2 FAILED on gpu1" in detail
        and "out/logs/x2.log" in detail
        and "blocked by a failure: y, z" in detail
        and "queue stopped" not in detail
        and "4/7 complete" in detail,
        detail,
    )
    plan = {
        "kernel": "tp-self-test",
        "runs": [extraction("x1"), extraction("x2"), extraction("x3")],
    }
    sched, _ = drive(k, plan, 1, outcome={"x1": ("failed", "exit 1: boom")})
    states = {r.name: r.state for r in sched.runs}
    check(
        "... on one GPU, with nothing pinned, too",
        states == {"x1": "failed", "x2": "complete", "x3": "complete"} and sched.stopped == {},
        states,
    )

    for command in ("jepa train", *k.ISING_COMMANDS):
        plan = {
            "kernel": "tp-self-test",
            "runs": [
                {"name": "s", "command": command, "gpu": 0, "args": []},
                {"name": "t", "command": command, "gpu": 0, "args": []},
                extraction("after", gpu=0),
                extraction("reader", "s", gpu=1),
                extraction("other", gpu=1),
            ],
        }
        sched, _ = drive(k, plan, 2, outcome={"s": ("failed", "exit 1: boom")})
        by = {r.name: r for r in sched.runs}
        pinned = "pinned to gpu0, whose queue stopped after s failed"
        check(
            f"a failed {command} still stops its queue, and the other queue goes on",
            sched.stopped == {0: "s failed"}
            and by["t"].state == by["after"].state == "not started"
            and by["t"].note == by["after"].note == pinned
            and by["reader"].state == "blocked"
            and by["other"].state == "complete",
            [(r.name, r.state, r.note) for r in sched.runs],
        )
    plan = {"kernel": "tp-self-test", "runs": [dict(training, gpu=None), extraction("x1")]}
    sched, _ = drive(k, plan, 1, outcome={"m": ("failed", "exit 1: boom")})
    by = {r.name: r for r in sched.runs}
    check(
        "... on one GPU an independent extraction behind it is not started",
        by["x1"].state == "not started" and by["x1"].note == "every queue stopped",
        (by["x1"].state, by["x1"].note),
    )


def queues(k: ModuleType) -> None:
    sizes = {"grid-L64": 30.0, "grid-L32": 20.0, "grid-L16": 10.0}
    sched, events = drive(k, job_a(), 2, sizes)
    starts = [name for kind, name, _ in events if kind == "start"]
    end64 = events.index(next(e for e in events if e[:2] == ("end", "grid-L64")))
    late = all(
        events.index(e) > end64 for e in events if e[0] == "start" and e[1].startswith("qs-")
    )
    check(
        "longest first: grid-L64 and grid-L32 start the two queues",
        set(starts[:2]) == {"grid-L64", "grid-L32"},
        starts,
    )
    check("no staircase starts before grid-L64 is complete", late, events)
    status, detail = k.verdict(sched.runs, sched.stopped)
    check(
        "all nine complete: STATUS complete",
        status == "complete" and "9/9 complete" in detail,
        detail,
    )

    sched, _ = drive(k, job_a(), 2, sizes, outcome={"grid-L64": ("failed", "exit 1: boom")})
    by = {r.name: r for r in sched.runs}
    gpu64 = by["grid-L64"].gpu_used
    check(
        "a failure stops its own queue", sched.stopped == {gpu64: "grid-L64 failed"}, sched.stopped
    )
    check(
        "... while the other queue finishes its runs",
        by["grid-L16"].state == by["grid-L32"].state == "complete"
        and by["grid-L16"].gpu_used != gpu64,
    )
    check(
        "... and the runs reading it are blocked",
        all(by[f"qs-L64-s{s}"].state == "blocked" for s in range(11, 17)),
    )
    status, detail = k.verdict(sched.runs, sched.stopped)
    check(
        "STATUS failed names the run, its log and the stopped queue",
        status == "failed"
        and "grid-L64 FAILED" in detail
        and "out/logs/grid-L64.log" in detail
        and f"queue stopped: gpu{gpu64}" in detail
        and "blocked by a failure: qs-L64-s11" in detail,
        detail,
    )

    pinned = {
        "kernel": "tp-self-test",
        "runs": [
            {"name": n, "command": "pilot", "args": [], **({"gpu": g} if g is not None else {})}
            for n, g in (("x", 1), ("y", 1), ("z", None), ("w", None))
        ],
    }
    sched, _ = drive(k, pinned, 2)
    by = {r.name: r for r in sched.runs}
    check(
        "pinned runs run on their GPU",
        by["x"].gpu_used == by["y"].gpu_used == 1,
        [by[n].gpu_used for n in by],
    )
    pinned = {
        "kernel": "tp-self-test",
        "runs": [
            {"name": "p", "command": "pilot", "args": [], "gpu": 0},
            {"name": "q", "command": "pilot", "args": [], "gpu": 0},
            {"name": "r", "command": "pilot", "args": []},
        ],
    }
    sched, _ = drive(k, pinned, 2, {"p": 3.0, "q": 2.0}, outcome={"p": ("failed", "exit 1")})
    by = {r.name: r for r in sched.runs}
    check(
        "a run pinned to a stopped queue is not started; the rest go on",
        by["q"].state == "not started"
        and "pinned to gpu0" in by["q"].note
        and by["r"].state == "complete",
        [(r.name, r.state, r.note) for r in sched.runs],
    )

    sched, _ = drive(k, job_a(), 2, sizes, hold="grid-L64", stop_after=0.3)
    by = {r.name: r for r in sched.runs}
    status, detail = k.verdict(sched.runs, sched.stopped)
    check(
        "the deadline: the running run is interrupted, its dependants not started",
        by["grid-L64"].state == "interrupted"
        and all(by[f"qs-L64-s{s}"].state == "not started" for s in range(11, 17))
        and "depends on grid-L64, which is interrupted" in by["qs-L64-s11"].note,
        [(r.name, r.state, r.note) for r in sched.runs],
    )
    check(
        "STATUS incomplete says how to continue",
        status == "incomplete" and "grid-L64 interrupted" in detail and "--resume" in detail,
        detail,
    )

    calls = []

    def cap() -> str | None:
        calls.append(1)
        return (
            None
            if len(calls) <= 3
            else "output cap: 18.1 GB under /kaggle/working, the cap is 18 GB"
        )

    sched, _ = drive(k, job_a(), 2, sizes, budget=cap)
    status, detail = k.verdict(sched.runs, sched.stopped)
    started = [r.name for r in sched.runs if r.state == "complete"]
    check(
        "the output cap: no run starts past it",
        sorted(started) == ["grid-L16", "grid-L32", "grid-L64"],
        started,
    )
    check(
        "STATUS says so",
        status == "incomplete" and "not started (output cap: 18.1 GB" in detail,
        detail,
    )


def resolution(k: ModuleType, root: Path) -> None:
    ctx = ctx_at(k, root)
    session = k.Session(ctx, job_a(), {}, COMMIT, TORCH, 2)
    stair, grid = session.by_name["qs-L64-s11"], session.by_name["grid-L64"]
    tail, key = session.resolve(stair, pending_ok=True)
    check(
        "a staircase waiting on grid-L64 holds a placeholder", tail[-1] == key[-1] == k.PLACEHOLDER
    )
    try:
        session.resolve(stair)
        check("... and cannot start", False)
    except k.RunError as exc:
        check("... and cannot start", "grid-L64 is pending" in str(exc), exc)
    grid.state, grid.run_id = "complete", "20261025T000000Z-L64-pilot-s6"
    grid.folder = f"ising/{grid.run_id}"
    folder = ctx.out / "ising" / grid.run_id
    folder.mkdir()
    try:
        session.resolve(stair)
        check("a missing tau.json stops the run that reads it", False)
    except k.RunError as exc:
        check(
            "a missing tau.json stops the run that reads it",
            "tau.json does not exist" in str(exc),
            exc,
        )
    (folder / "tau.json").write_text("{}", encoding="utf-8")
    tail, key = session.resolve(stair)
    check(
        "--tau resolves to grid-L64's folder when the run starts",
        tail[-1] == str(folder / "tau.json"),
        tail,
    )
    check("the key names that folder by run id", key[-1] == f"ising/{grid.run_id}/tau.json", key)


def made(folder: Path, done: bool, git_sha: str = COMMIT, checkpoint: bool = False) -> None:
    folder.mkdir(parents=True)
    manifest = {
        "producer": {"git_sha": git_sha, "dirty": False},
        "env": {"torch": TORCH, "device": "cpu"},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if done:
        (folder / "complete.json").write_text(
            json.dumps({"clean": True, "files": {}, "checks": {}})
        )
    if checkpoint:
        (folder / "checkpoint.pt").write_bytes(b"state")


def previous(k: ModuleType, root: Path) -> None:
    ctx = ctx_at(k, root)
    plan = job_a()
    ids = {f"grid-L{L}": f"20261025T00000{i}Z-L{L}-pilot-s6" for i, L in enumerate((16, 32, 64))}
    keys = {r["name"]: [r["command"], *map(str, r["args"])] for r in plan["runs"]}
    out = ctx.input / plan["kernel"] / "out"
    made(out / "ising" / ids["grid-L16"], done=True)
    made(out / "ising" / ids["grid-L32"], done=False, checkpoint=True)
    made(out / "ising" / ids["grid-L64"], done=True, git_sha=OTHER)
    made(out / "ising" / "20261025T000009Z-gone", done=True)
    records = [
        {
            "name": "grid-L16",
            "run_id": ids["grid-L16"],
            "key": keys["grid-L16"],
            "state": "complete",
        },
        {
            "name": "grid-L32",
            "run_id": ids["grid-L32"],
            "key": keys["grid-L32"],
            "state": "interrupted",
            "step": 30000,
            "wall_s": 20.0,
        },
        {
            "name": "grid-L64",
            "run_id": ids["grid-L64"],
            "key": keys["grid-L64"],
            "state": "complete",
        },
        {"name": "gone", "run_id": "20261025T000009Z-gone", "key": [], "state": "complete"},
    ]
    state = {"kernel": plan["kernel"], "updated_utc": "2026-10-25T01:00:00+00:00", "runs": records}
    (out / k.STATE).write_text(json.dumps(state), encoding="utf-8")
    older = ctx.input / "older" / k.STATE
    older.parent.mkdir()
    older.write_text(json.dumps(dict(state, updated_utc="2026-10-24T01:00:00+00:00", runs=[])))
    prior = k.carry_forward(ctx, plan)
    copied = [(ctx.out / "ising" / i).is_dir() for i in ids.values()]
    check(
        "the newest state of this kernel is the one read",
        prior is not None and prior["path"] == str(out / k.STATE),
    )
    check("its runs are copied into this output first", all(copied), copied)
    check("... only the plan's", not (ctx.out / "ising" / "20261025T000009Z-gone").exists())
    session = k.Session(ctx, plan, {}, COMMIT, TORCH, 2, prior)
    by = session.by_name
    check(
        "a carried run keeps its folder until the session decides",
        by["grid-L32"].run_id == ids["grid-L32"],
    )
    for name in ids:
        by[name].key = keys[name]
    folder, how = session.adopt(by["grid-L16"])
    check(
        "a finished run of the same commit is reused",
        how == "reuse" and folder == ctx.out / "ising" / ids["grid-L16"],
    )
    folder, how = session.adopt(by["grid-L32"])
    check(
        "an unfinished one is resumed from its checkpoint",
        how == "resume" and by["grid-L32"].step == 30000,
    )
    folder, how = session.adopt(by["grid-L64"])
    gone = not (ctx.out / "ising" / ids["grid-L64"]).exists()
    check(
        "one made by another commit starts afresh, its copy removed",
        how is None and gone and by["grid-L64"].run_id is None,
    )
    by["grid-L16"].key = [*keys["grid-L16"], "--T", "2.0"]
    folder, how = session.adopt(by["grid-L16"])
    check("one made from other arguments starts afresh", how is None and folder is None)
    check("a run new to the plan starts afresh", session.adopt(by["qs-L64-s11"]) == (None, None))


def inputs(k: ModuleType, root: Path) -> None:
    ctx = ctx_at(k, root)
    record = b'{"kernel_commit": "655949a"}'
    digest = hashlib.sha256(record).hexdigest()
    plan = copy.deepcopy(BASE)
    plan.update(
        kernel_sources=["gabrielezerzer/tp-tauc-l64-l128"], datasets=["gabrielezerzer/tp-data"]
    )
    plan["inputs"] = {"record": {"run": RECORD, "file": "pilot.json", "sha256": digest}}
    plan["runs"].append(
        {"name": "n", "command": "null", "args": ["--L", 16, "--pilot", {"input": "record"}]}
    )
    check("a plan with an input loads", not refused(k, plan), refused(k, plan))
    folder = ctx.input / "tp-tauc-l64-l128" / "out" / "ising" / RECORD
    folder.mkdir(parents=True)
    (folder / "pilot.json").write_bytes(record)
    (ctx.input / "tp-data").mkdir()
    session = k.Session(ctx, plan, {}, COMMIT, TORCH, 2)
    session.locate_inputs()
    check(
        "an input in an attached kernel output is found, its sha256 checked",
        session.inputs["record"] == folder / "pilot.json",
    )
    tail, key = session.resolve(session.by_name["n"])
    check(
        "--pilot resolves to it",
        tail[-1] == str(folder / "pilot.json") and key[-1] == f"input/{RECORD}/pilot.json",
        key,
    )
    wrong = copy.deepcopy(plan)
    wrong["inputs"]["record"]["sha256"] = "f" * 64
    try:
        k.Session(ctx, wrong, {}, COMMIT, TORCH, 2).locate_inputs()
        check("a sha256 other than the plan's is refused before any run", False)
    except k.Refused as exc:
        check("a sha256 other than the plan's is refused before any run", digest in str(exc), exc)
    shutil.rmtree(ctx.input / "tp-tauc-l64-l128" / "out")
    with zipfile.ZipFile(ctx.input / "tp-data" / f"{RECORD}.zip", "w") as z:
        z.writestr("pilot.json", record)
    session = k.Session(ctx, plan, {}, COMMIT, TORCH, 2)
    session.locate_inputs()
    check(
        "... or unpacked from a <run id>.zip in a dataset",
        session.inputs["record"].read_bytes() == record,
    )
    (ctx.input / "tp-data" / f"{RECORD}.zip").unlink()
    try:
        k.Session(ctx, plan, {}, COMMIT, TORCH, 2).locate_inputs()
        check("a missing input is refused", False)
    except k.Refused as exc:
        check("a missing input is refused", "found none" in str(exc), exc)


def input_checkpoints(k: ModuleType, root: Path) -> None:
    """A final.pt input with no complete.json beside it is refused at session start, since jepa
    extract refuses it in every run that reads it (#128). Its untrained twin needs none."""
    ctx = ctx_at(k, root)
    trained, untrained = b"trained", b"untrained"

    def entry(run_id: str, file: str, payload: bytes) -> dict:
        return {"run": run_id, "file": file, "sha256": hashlib.sha256(payload).hexdigest()}

    def located(entries: dict, *slugs: str) -> tuple[dict, str]:
        """What locate_inputs finds in these attached sources, and its refusal."""
        plan = copy.deepcopy(BASE)
        plan.update(kernel_sources=[f"o/{slug}" for slug in slugs], inputs=entries)
        session = k.Session(ctx, plan, {}, COMMIT, TORCH, 2)
        try:
            session.locate_inputs()
        except k.Refused as exc:
            return session.inputs, str(exc)
        return session.inputs, ""

    def training(slug: str, done: bool) -> Path:
        """A training's folder in a kernel output, complete.json beside final.pt once it is done."""
        folder = ctx.input / slug / "out" / "jepa" / M1_RUN
        (folder / "ckpt").mkdir(parents=True)
        (folder / k.FINAL).write_bytes(trained)
        (folder / k.TWIN).write_bytes(untrained)
        if done:
            (folder / "complete.json").write_text("{}", encoding="utf-8")
        return folder

    final = {"final": entry(M1_RUN, k.FINAL, trained)}
    twin = {"twin": entry(M1_RUN, k.TWIN, untrained)}
    bare = training("tp-bare", done=False)
    _, why = located(final, "tp-bare")
    check(
        "a final.pt of the right sha256 with no complete.json beside it is refused before any run",
        f"input final: {bare / k.FINAL} has no complete.json" in why,
        why or "accepted",
    )
    found, why = located(twin, "tp-bare")
    check("... and the untrained twin beside it needs none", found == {"twin": bare / k.TWIN}, why)
    held = training("tp-held", done=True)
    found, why = located({**final, **twin}, "tp-held")
    check(
        "a final.pt with its complete.json is found, and the twin with it",
        found == {"final": held / k.FINAL, "twin": held / k.TWIN},
        why or found,
    )
    found, why = located(final, "tp-bare", "tp-held")
    check(
        "... over a copy with none, in a source listed first",
        found == {"final": held / k.FINAL},
        why or found,
    )

    # out of a <run id>.zip, each its own run id since a session unpacks a run id once
    zipped = {"20261101T051047Z-L16-m-bare": False, "20261101T051047Z-L16-m-held": True}
    archives = ctx.input / "tp-zips"
    archives.mkdir()
    for run_id, done in zipped.items():
        with zipfile.ZipFile(archives / f"{run_id}.zip", "w") as z:
            z.writestr("manifest.json", "{}")
            z.writestr(k.FINAL, trained)
            if done:
                z.writestr("complete.json", "{}")
    no_file, with_file = zipped
    _, why = located({"final": entry(no_file, k.FINAL, trained)}, "tp-zips")
    check(
        "a final.pt unpacked from a <run id>.zip with no complete.json is refused",
        "has no complete.json" in why,
        why or "accepted",
    )
    found, why = located({"final": entry(with_file, k.FINAL, trained)}, "tp-zips")
    check(
        "... and found when the zip holds one",
        "final" in found and (found["final"].parent / "complete.json").is_file(),
        why or found,
    )


def wheel(k: ModuleType, root: Path) -> None:
    ctx = ctx_at(k, root)
    folder = ctx.input / k.WHEEL_DATASET
    folder.mkdir()
    (folder / "tipping_point-0.1.0-py3-none-any.whl").write_bytes(b"PK")
    sha = "655949aa79ea32adfc103635d4cfb303ae24b94f"

    def outcome(text: str | None) -> str:
        note = folder / "COMMIT.txt"
        if text is None:
            note.unlink(missing_ok=True)
        else:
            note.write_text(text, encoding="utf-8")
        try:
            return k.find_wheel(ctx)[1]["commit"]
        except k.Refused as exc:
            return f"refused: {exc}"

    check("no COMMIT.txt: refused", outcome(None).startswith("refused: expected one COMMIT.txt"))
    check(
        "a COMMIT.txt marked dirty: refused",
        outcome(f"{sha}-dirty\n").startswith("refused: COMMIT.txt marks"),
    )
    check(
        "a COMMIT.txt naming no commit: refused",
        outcome("main\n").startswith("refused: COMMIT.txt names no"),
    )
    check("a clean commit: accepted", outcome(f"{sha}\n") == sha)
    (folder / "tipping_point-0.2.0-py3-none-any.whl").write_bytes(b"PK")
    check(
        "two wheels: refused", outcome(f"{sha}\n").startswith("refused: expected one tipping_point")
    )
    cuda = ctx_at(k, root / "cuda", device="cuda")
    env = {
        "torch": "2.14.1+cu126",
        "cuda_available": True,
        "devices": [{"index": 0}],
        "code_version": {"git_sha": sha, "dirty": False},
    }

    def env_says(**changes: Any) -> str:
        try:
            k.check_env(cuda, {**env, **changes}, sha)
        except k.Refused as exc:
            return str(exc)
        return "accepted"

    check(
        "the installed package names COMMIT.txt's clean commit: accepted", env_says() == "accepted"
    )
    check(
        "... another commit: refused",
        "names commit" in env_says(code_version={"git_sha": OTHER, "dirty": False}),
    )
    check(
        "... the same, dirty: refused",
        "dirty" in env_says(code_version={"git_sha": sha, "dirty": True}),
    )
    check(
        "... a stamp-less wheel: refused",
        "names commit None" in env_says(code_version={"git_sha": None, "dirty": None}),
    )
    check("another torch: refused", "not torch==" in env_says(torch="2.14.1+cpu"))
    check("no GPU: refused", "no CUDA device" in env_says(devices=[]))


def verification(k: ModuleType, root: Path) -> None:
    folder = root / "20261025T000000Z-L8-qs-s1"
    (folder / "stats").mkdir(parents=True)
    chunk = folder / "stats" / f"{1:020d}.parquet"
    chunk.write_bytes(b"rows")
    thermalised = {
        "gate": "thermalised",
        "temperature": 2.2179487179487176,
        "passed": True,
        "detail": "flat",
    }

    def write(
        producer: dict | None = None,
        clean: bool = True,
        passed: bool = True,
        gate: str = "thermalised",
        command: str = "staircase",
    ) -> tuple[list[str], list[str]]:
        """verdicts' (what fails the run, what the scorer drops and counts) for a run of this
        command whose complete.json holds one check, of this gate."""
        manifest = {"producer": producer or {"git_sha": COMMIT, "dirty": False}}
        done = {
            "clean": clean,
            "files": {f"stats/{chunk.name}": hashlib.sha256(b"rows").hexdigest()},
            "checks": {gate: [dict(thermalised, gate=gate, passed=passed)]},
        }
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (folder / "complete.json").write_text(json.dumps(done), encoding="utf-8")
        return k.verdicts(folder, COMMIT, command)

    check("a clean run of the wheel's commit passes", write() == ([], []))
    check(
        "another commit fails",
        "is not the wheel's" in str(write({"git_sha": OTHER, "dirty": False})[0]),
    )
    check(
        "a dirty manifest fails",
        "dirty is True" in str(write({"git_sha": COMMIT, "dirty": True})[0]),
    )
    check("complete.json not clean fails", "clean is False" in str(write(clean=False)[0]))
    miss = "check thermalised failed at T = 2.2179: flat"
    said = write(passed=False, gate="independence")
    check(
        "a failed check fails, with its temperature",
        said == (["check independence failed at T = 2.2179: flat"], []),
        said,
    )
    said = write(passed=False)
    check(
        "... but a simulator run's failed thermalised check is noted, not failed: the scorer drops "
        "that plateau or null block and counts it",
        said == ([], [miss]),
        said,
    )
    said = write(passed=False, command="jepa train")
    check("... and it fails any other run", said == ([miss], []), said)
    (folder / "complete.json").unlink()
    check(
        "no complete.json fails",
        k.verdicts(folder, COMMIT, "staircase") == (["complete.json is missing or unreadable"], []),
    )
    write()
    check(
        "jepa verify: exit 0 is clean whatever else it printed",
        k.file_problems(0, [*NOISE, f"{folder}: ok"], [folder]) == [],
    )
    said = k.file_problems(1, [*NOISE, f"{folder}: sha256 differs: x"], [folder])
    check(
        "jepa verify: a failure names the folder's problem, never the noise",
        said == [f"{folder}: sha256 differs: x"],
        said,
    )
    check(
        "jepa verify: a failure with nothing to say still fails",
        k.file_problems(1, NOISE, [folder]) == ["jepa verify exited 1"],
    )
    argv = [sys.executable, "-I", "-m", "tipping_point.jepa", "verify", str(folder)]
    ran = subprocess.run(argv, capture_output=True, text=True, check=False)
    said = k.file_problems(ran.returncode, ran.stdout.splitlines(), [folder])
    check(
        "the real jepa verify accepts the intact folder",
        ran.returncode == 0 and said == [],
        ran.stdout + ran.stderr,
    )
    chunk.write_bytes(b"changed")
    ran = subprocess.run(argv, capture_output=True, text=True, check=False)
    said = k.file_problems(ran.returncode, ran.stdout.splitlines(), [folder])
    check(
        "... and names a changed file", ran.returncode == 1 and "sha256 differs" in str(said), said
    )

    # The session's verdict on a finished null run whose one failed check is thermalised: complete,
    # the miss noted, so its queue goes on, in the session that made it and on each resume; one
    # that failed another check as well still fails and stops its queue
    ctx = ctx_at(k, root / "session")
    ctx.py = sys.executable  # what runs jepa verify
    specs = [
        {"name": name, "command": "null", "args": ["--L", 8, "--seed", seed, "--epsilon-L", 2]}
        for name, seed in (("a", 7), ("b", 8), ("c", 9))
    ]
    plan = {"kernel": "tp-self-test", "runs": specs}
    ids = {"a": "20261102T000007Z-L8-null-e2-s7", "b": "20261102T000008Z-L8-null-e2-s8"}
    prior = {
        "runs": [
            {
                "name": s["name"],
                "folder": f"ising/{ids[s['name']]}",
                "key": [s["command"], *map(str, s["args"])],
                "state": "complete",
            }
            for s in specs[:2]
        ]
    }

    def finished(name: str, **passed: bool) -> Path:
        """The run's folder as the simulator leaves it, with one check per gate named."""
        run_dir = ctx.out / "ising" / ids[name]
        if not run_dir.is_dir():
            made(run_dir, done=False)
        checks = {gate: [dict(thermalised, gate=gate, passed=ok)] for gate, ok in passed.items()}
        done = {"clean": True, "files": {}, "checks": checks}
        (run_dir / "complete.json").write_text(json.dumps(done), encoding="utf-8")
        return run_dir

    def resumed() -> Any:
        """The next session on one GPU: a and b reused in turn, the deadline before c starts."""
        session = k.Session(ctx, plan, {}, COMMIT, TORCH, 1, prior)
        takes: list[int] = []

        def deadline() -> str | None:
            takes.append(1)
            return "session deadline" if len(takes) > 2 else None

        session.scheduler.budget = deadline
        session.scheduler.run(session.tick, ctx.poll_s)
        return session

    fresh = k.Session(ctx, plan, {}, COMMIT, TORCH, 1)
    run = fresh.by_name["a"]
    state, note = fresh.settle(run, finished("a", thermalised=False), 0, collections.deque(), "")
    check(
        "a simulator run whose only failed check is thermalised: complete, the miss noted, in the "
        "session that made it",
        (state, run.problems, run.dropped) == ("complete", [], [miss]),
        (state, note, run.problems, run.dropped),
    )
    finished("b", thermalised=True)
    session = resumed()
    by = session.by_name
    check(
        "... and on resume: reused, complete again, its folder kept",
        by["a"].state == "complete"
        and by["a"].reused
        and by["a"].dropped == [miss]
        and (ctx.out / "ising" / ids["a"] / "complete.json").is_file(),
        [(r.name, r.state, r.note, r.dropped) for r in session.runs],
    )
    check(
        "... its queue goes on: the run behind it on the one GPU completes",
        by["b"].state == "complete" and session.scheduler.stopped == {},
        (session.scheduler.stopped, [(r.name, r.state, r.note) for r in session.runs]),
    )
    status, detail = k.verdict(session.runs, session.scheduler.stopped)
    check(
        "... STATUS incomplete names it as the scorer's to drop, and says how to continue",
        status == "incomplete"
        and "thermalised missed in a plateau or null block, which the scorer drops and counts: a;"
        in detail
        and "--resume" in detail,
        detail,
    )
    finished("a", thermalised=False, independence=False)
    session = resumed()
    by = session.by_name
    status, detail = k.verdict(session.runs, session.scheduler.stopped)
    check(
        "a simulator run that failed another check as well: failed, its folder kept, its queue "
        "stopped as before",
        by["a"].state == "failed"
        and "check independence failed" in by["a"].note
        and (ctx.out / "ising" / ids["a"] / "complete.json").is_file()
        and session.scheduler.stopped == {0: "a failed"}
        and by["b"].state == by["c"].state == "not started"
        and status == "failed",
        (detail, [(r.name, r.state, r.note) for r in session.runs]),
    )


def kernel_folder(k: ModuleType, root: Path) -> None:
    script = k.write_kernel(k.read_plan(PLANS / "job-a.json"), root / "a")
    meta = json.loads((root / "a" / "kernel-metadata.json").read_text(encoding="utf-8"))
    loaded = runpy.run_path(str(script), run_name="tp_kernel_check")
    check("the written kernel embeds the plan", json.loads(loaded["PLAN_JSON"]) == job_a())
    check(
        "its metadata: private, a GPU, the internet, the wheel, no kernel source",
        meta["id"] == "gabrielezerzer/tp-ising-job-a"
        and meta["title"] == "tp-ising-job-a"
        and meta["code_file"] == "tp-ising-job-a.py"
        and meta["kernel_type"] == "script"
        and (meta["is_private"], meta["enable_gpu"], meta["enable_internet"]) == (True, True, True)
        and meta["dataset_sources"] == ["gabrielezerzer/tipping-point-wheel"]
        and meta["kernel_sources"] == [],
        meta,
    )
    plan = job_a()
    plan.update(
        kernel_sources=["gabrielezerzer/tp-tauc-l64-l128"], datasets=["gabrielezerzer/tp-data"]
    )
    k.write_kernel(plan, root / "b", resume=True)
    meta = json.loads((root / "b" / "kernel-metadata.json").read_text(encoding="utf-8"))
    check(
        "--resume attaches the kernel's own output beside the plan's sources",
        meta["kernel_sources"]
        == ["gabrielezerzer/tp-tauc-l64-l128", "gabrielezerzer/tp-ising-job-a"]
        and meta["dataset_sources"]
        == ["gabrielezerzer/tipping-point-wheel", "gabrielezerzer/tp-data"],
        meta,
    )


TRAINING = "20261026T000000Z-L16-m3"
STAIRS = ("20261024T133416Z-L16-qs-s11", "20261024T133934Z-L16-qs-s12")
NULL = "20261023T010039Z-L16-null-e2-s1"
JEPA_PLAN: dict[str, Any] = {
    "kernel": "tp-self-test",
    "datasets": ["gabrielezerzer/tp-data"],
    "runs": [
        {
            "name": "m3",
            "command": "jepa train",
            "stage": list(STAIRS),
            "args": [
                "--L",
                16,
                "--train-seeds",
                11,
                "--val-seed",
                12,
                "--seed",
                3,
                "--pilot",
                "pilot.json",
                "--steps",
                20,
            ],
        },
        {
            "name": "x-final",
            "command": "jepa extract",
            "args": [{"run": "m3", "file": "final.pt"}, {"dataset_run": NULL}],
        },
        {
            "name": "x-twin",
            "command": "jepa extract",
            "args": [
                {"run": "m3", "file": "ckpt/step_0000000.pt"},
                {"dataset_run": NULL},
                "--select",
                "tc-plus",
            ],
        },
    ],
}


def dataset_run(folder: Path, seed: int, payload: bytes = b"rows") -> None:
    """A simulator run as `jepa verify` reads it: a manifest, one chunk, complete.json."""
    (folder / "configs").mkdir(parents=True)
    (folder / "configs" / f"{1:020d}.parquet").write_bytes(payload)
    manifest = {"producer": {"git_sha": OTHER, "dirty": False}, "params": {"seed": seed}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    files = {f"configs/{1:020d}.parquet": hashlib.sha256(b"rows").hexdigest()}
    done = {"clean": True, "files": files, "checks": {}}
    (folder / "complete.json").write_text(json.dumps(done), encoding="utf-8")


def seed_spellings(seed: int) -> list[list]:
    """Each way a plan can give `ising` this seed (#129): --seed N and --seed=N, any prefix of the
    flag for either, since argparse takes one, and after another --seed, since it keeps the last.
    `ising null` and `ising ramp` take even --s: no other flag of theirs starts with it."""
    flags = ("--seed", "--see", "--se", "--s")
    spelt = [form for f in flags for form in ([f, seed], [f, str(seed)], [f"{f}={seed}"])]
    return [*spelt, ["--seed", 1, "--seed", seed], ["--seed=1", f"--see={seed}"]]


def jepa_entries(k: ModuleType, root: Path) -> None:
    from tipping_point.ising.__main__ import _parser as ising_parser
    from tipping_point.jepa import train

    names = (k.TWIN, k.FINAL, k.LEDGER)
    check(
        "the checkpoint and ledger names are the library's",
        names == (train.TWIN.as_posix(), str(train.FINAL), train.LEDGER),
        names,
    )
    check("a plan of jepa entries loads", not refused(k, copy.deepcopy(JEPA_PLAN)))

    def extract_on(target: object, extra: list | None = None) -> str:
        plan = copy.deepcopy(JEPA_PLAN)
        plan["runs"] += extra or []
        plan["runs"][1]["args"][1] = target
        return refused(k, plan)

    turned = {
        seed: why
        for seed in CONFIRMATORY_SEEDS
        if (why := extract_on({"dataset_run": f"20261101T000000Z-L16-null-e2-s{seed}"}))
    }
    check("accepted: jepa extract on a dataset run at any of seeds 6-10", not turned, turned)
    untagged = {"dataset_run": "20261101T000000Z-train"}
    why = extract_on(untagged)
    check("accepted: jepa extract on a dataset run whose id names no seed", not why, why)
    plan = copy.deepcopy(JEPA_PLAN)
    plan["runs"].insert(1, {"name": "sim", "command": "null", "args": []})
    plan["runs"][2]["args"][1] = {"run": "sim"}
    turned = {}
    for seed in CONFIRMATORY_SEEDS:
        for given in seed_spellings(seed):
            plan["runs"][1]["args"] = ["--L", 16, *given]
            if why := refused(k, plan):
                turned[str(given)] = why
    check(
        "accepted: jepa extract on this plan's own run at any of seeds 6-10, in any spelling",
        not turned,
        turned,
    )

    def cli_reads(given: list) -> int | None:
        """The seed `ising null`'s own parser reads from these arguments; None if it refuses."""
        argv = ["null", "--L", "16", "--epsilon-L", "2", "--pilot", "pilot.json", *map(str, given)]
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                return ising_parser().parse_args(argv).seed
            except SystemExit:
                return None

    misread = [(g, cli_reads(g)) for g in seed_spellings(7) if cli_reads(g) != 7]
    check("... each of those spellings being one the CLI itself reads so", not misread, misread)
    for what, change, needle in (
        ("a stage entry that is no run id", {"stage": ["L16-qs-s11"]}, "stage: a list of run ids"),
        ("a staged run listed twice", {"stage": [STAIRS[0], STAIRS[0]]}, "listed twice"),
        ("a command the runner does not run", {"command": "jepa probe"}, "command 'jepa probe'"),
    ):
        plan = copy.deepcopy(JEPA_PLAN)
        plan["runs"][0].update(change)
        check(f"refused: {what}", needle in refused(k, plan), refused(k, plan))
    plan = copy.deepcopy(JEPA_PLAN)
    del plan["datasets"]
    why = refused(k, plan)
    check("refused: dataset runs with nowhere to look", "dataset or a kernel source" in why, why)
    plan = dict(copy.deepcopy(JEPA_PLAN), wheel="no slash")
    check("refused: a wheel that names no dataset", "wheel 'no slash'" in refused(k, plan))

    errors = k.cli_errors(JEPA_PLAN)
    check("the jepa CLI itself takes every entry's arguments", errors == [], errors)
    plan = copy.deepcopy(JEPA_PLAN)
    plan["runs"][0]["args"] += ["--stepz", 5]
    errors = k.cli_errors(plan)
    check("... and catches a misspelt training flag", "unrecognized" in str(errors), errors)
    plan["runs"][0]["args"] = plan["runs"][0]["args"][:-2]
    plan["runs"][1]["args"] = plan["runs"][1]["args"][:1]
    errors = k.cli_errors(plan)
    check("... and an extraction missing its run", "required" in str(errors), errors)

    ctx = ctx_at(k, root)
    ctx.py = sys.executable
    for run_id, seed in ((STAIRS[0], 11), (STAIRS[1], 12), (NULL, 1)):
        dataset_run(ctx.input / "tp-data" / run_id, seed)
    session = k.Session(ctx, copy.deepcopy(JEPA_PLAN), {}, COMMIT, TORCH, 2)
    session.locate_datasets()
    session.stage()
    staged = ctx.out / "ising" / STAIRS[0] / "configs" / f"{1:020d}.parquet"
    check(
        "every dataset run the plan reads is staged and verified",
        all((ctx.out / "ising" / r / "complete.json").is_file() for r in (*STAIRS, NULL)),
    )
    check("... hard linked when the input shares the filesystem", os.stat(staged).st_nlink >= 2)
    shutil.rmtree(ctx.out / "ising" / STAIRS[1])
    (ctx.input / "tp-data" / STAIRS[1] / "configs" / f"{1:020d}.parquet").write_bytes(b"other")
    try:
        damaged = k.Session(ctx, copy.deepcopy(JEPA_PLAN), {}, COMMIT, TORCH, 2)
        damaged.locate_datasets()
        damaged.stage()
        check("a damaged input is refused by jepa verify's exit code", False)
    except k.Refused as exc:
        check(
            "a damaged input is refused by jepa verify's exit code",
            "sha256 differs" in str(exc),
            exc,
        )

    m3, final, twin = (session.by_name[n] for n in ("m3", "x-final", "x-twin"))
    m3.state, m3.folder, m3.run_id = "complete", f"jepa/{TRAINING}", TRAINING
    (ctx.out / "jepa" / TRAINING / "ckpt").mkdir(parents=True)
    (ctx.out / "jepa" / TRAINING / "final.pt").write_bytes(b"trained")
    (ctx.out / "jepa" / TRAINING / k.TWIN).write_bytes(b"untrained")
    tail, key = session.resolve(final)
    expected = [str(ctx.out / "jepa" / TRAINING / "final.pt"), str(ctx.out / "ising" / NULL)]
    check("final.pt and the dataset run resolve to their folders", tail == expected, tail)
    check(
        "... named by folder in the key",
        key[1:] == [f"jepa/{TRAINING}/final.pt", f"dataset/{NULL}"],
        key,
    )
    tail, _ = session.resolve(twin)
    check(
        "the step-0 twin resolves too", tail[0] == str(ctx.out / "jepa" / TRAINING / k.TWIN), tail
    )
    argv = session.command_line(final, session.resolve(final)[0], None)
    check(
        "jepa extract runs with --device and no --data-dir",
        argv[3:6] == ["-m", "tipping_point.jepa", "extract"] and argv[-2:] == ["--device", "cpu"],
        argv,
    )
    argv = session.command_line(m3, session.resolve(m3)[0], None)
    check(
        "jepa train runs with --device and --data-dir",
        argv[-4:] == ["--device", "cpu", "--data-dir", str(ctx.out)],
        argv,
    )
    argv = session.command_line(m3, [], ctx.out / "jepa" / TRAINING)
    check(
        "an unfinished training resumes with jepa resume",
        argv[4:]
        == ["tipping_point.jepa", "resume", str(ctx.out / "jepa" / TRAINING), "--device", "cpu"],
        argv,
    )
    argv = session.command_line(
        final, session.resolve(final)[0], ctx.out / "ising" / NULL / "latents" / "x"
    )
    check("an unfinished extraction runs again", "resume" not in argv and "extract" in argv, argv)

    check("no latents table yet", session.latents(final) is None)
    table = ctx.out / "ising" / NULL / "latents" / f"{TRAINING}-step0000020"
    table.mkdir(parents=True)
    digest = hashlib.sha256(b"trained").hexdigest()
    manifest = {
        "producer": {"git_sha": COMMIT, "dirty": False},
        "env": {"torch": TORCH, "device": "cpu"},
        "params": {"checkpoint": {"sha256": digest}, "dataset": {"L": 16}},
    }
    (table / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    check(
        "the latents table is the one naming its checkpoint's sha256",
        session.latents(final) == table,
    )
    check("... not the twin's", session.latents(twin) is None)
    own = ctx.out / "ising" / NULL / "manifest.json"
    turned = {}
    for seed in CONFIRMATORY_SEEDS:
        own.write_text(json.dumps({"params": {"seed": seed}}), encoding="utf-8")
        try:
            session.resolve(final)
            if session.latents(final) != table:
                turned[seed] = "its latents table is not found"
        except k.RunError as exc:
            turned[seed] = str(exc)
    check(
        "a run at any of seeds 6-10, by its own manifest, resolves and finds its latents table",
        not turned,
        turned,
    )
    own.write_text(json.dumps({"params": {"seed": 1}}), encoding="utf-8")

    session.prior = {
        "x-final": {"name": "x-final", "folder": f"ising/{NULL}", "key": key},
        "m3": {"name": "m3", "folder": f"jepa/{TRAINING}", "key": ["jepa train"]},
    }
    final.key = key
    found, how = session.adopt(final)
    check(
        "an interrupted extraction is resumed from its table",
        (found, how) == (table, "resume"),
        (found, how),
    )
    (table / "complete.json").write_text("{}", encoding="utf-8")
    found, how = session.adopt(final)
    check("a finished one is reused", how == "reuse", how)
    made(ctx.out / "jepa" / "20261026T000001Z-L16-m4", done=False, checkpoint=True)
    session.prior["m3"]["folder"] = "jepa/20261026T000001Z-L16-m4"
    m3.key = ["jepa train"]
    found, how = session.adopt(m3)
    check(
        "an unfinished training is resumed",
        how == "resume" and found is not None and found.name.endswith("m4"),
        how,
    )
    m3.folder, m3.run_id = f"jepa/{TRAINING}", TRAINING

    ledger = ctx.out / "jepa" / k.LEDGER
    lines = [
        {"run_id": TRAINING, "from_ns": 0, "to_ns": 10 * 10**9, "step": 1000},  # start-up in it
        {"run_id": TRAINING, "from_ns": 10 * 10**9, "to_ns": 15 * 10**9, "step": 2000},
        {"run_id": TRAINING, "from_ns": 15 * 10**9, "to_ns": 16 * 10**9, "step": 2000},  # closing
        {"run_id": "other", "from_ns": 0, "to_ns": 1, "step": 5},
    ]
    ledger.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    (ctx.out / "jepa" / TRAINING / "complete.json").write_text("{}", encoding="utf-8")
    began = 7 * 10**9
    manifest["started_ns"] = began  # the table's clock starts once the model is loaded
    (table / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    done = {"rows": {"latents": 1000}, "ended_ns": began + 5 * 10**8}
    (table / "complete.json").write_text(json.dumps(done), encoding="utf-8")
    final.state, final.folder, final.wall_s = "complete", table.relative_to(ctx.out).as_posix(), 2.0
    final.resumed = False
    rates = session.measured()
    check(
        "a training's rate leaves out start-up and the closing line",
        rates["train"].get("m3", {}).get("s_per_step") == 0.005,
        rates,
    )
    group = rates["extract"].get("m3 final.pt L16", {})
    check(
        "an extraction's rate is per frame, by checkpoint: the readout by its table's own clock",
        group.get("readout_us_per_frame") == 500.0 and group.get("us_per_frame") == 2000.0,
        rates,
    )
    text = k.rates_text(rates)
    check(
        "STATUS carries both",
        "0.005 s/step" in text and "500.0 us/frame readout, 2000.0 with start-up" in text,
        text,
    )
    final.resumed = True
    group = session.measured()["extract"].get("m3 final.pt L16", {})
    check(
        "a table finished across sessions has no readout clock",
        group.get("readout_us_per_frame") is None,
        group,
    )

    (ctx.out / "ising" / STAIRS[0] / "latents").mkdir()
    session.unstage()  # STAIRS[1] is the damaged copy the refused session left: no latents
    kept = [(ctx.out / "ising" / r).is_dir() for r in (STAIRS[0], NULL, STAIRS[1])]
    check(
        "unstaging keeps the runs that gained latents and drops a plain copy",
        kept == [True, True, False],
        kept,
    )

    prev = root / "carry"
    pctx = ctx_at(k, prev)
    out = pctx.input / "tp-self-test" / "out"
    (out / "ising" / NULL / "latents" / "t").mkdir(parents=True)
    (out / "jepa" / TRAINING).mkdir(parents=True)
    (out / "jepa" / k.LEDGER).write_text("{}\n", encoding="utf-8")
    (out / "ising" / "20261025T000000Z-L64-pilot-s6").mkdir(parents=True)
    records = [
        {"name": "x-final", "folder": f"ising/{NULL}/latents/t", "state": "complete"},
        {"name": "m3", "folder": f"jepa/{TRAINING}", "state": "interrupted"},
    ]
    state = {"kernel": "tp-self-test", "updated_utc": "2026-10-26", "runs": records}
    (out / k.STATE).write_text(json.dumps(state), encoding="utf-8")
    prior = k.carry_forward(pctx, copy.deepcopy(JEPA_PLAN))
    check(
        "a latents table travels in its dataset run; a training and the ledger come too",
        (pctx.out / "ising" / NULL / "latents" / "t").is_dir()
        and (pctx.out / "jepa" / TRAINING).is_dir()
        and (pctx.out / "jepa" / k.LEDGER).is_file()
        and prior is not None
        and len(prior["runs"]) == 2,
        prior,
    )
    legacy = {
        "kernel": "tp-ising-job-a",
        "updated_utc": "2026-10-26",
        "runs": [{"name": "grid-L64", "run_id": "20261025T000000Z-L64-pilot-s6"}],
    }
    (out / k.STATE).write_text(json.dumps(legacy), encoding="utf-8")
    prior = k.carry_forward(pctx, job_a())
    check(
        "a record from before folders reads as ising/<run id>",
        prior is not None and prior["runs"][0]["folder"] == "ising/20261025T000000Z-L64-pilot-s6",
        prior,
    )

    wctx = ctx_at(k, root / "wheels")
    frozen = wctx.input / "tipping-point-wheel-v1"
    frozen.mkdir()
    (frozen / "tipping_point-0.1.0-py3-none-any.whl").write_bytes(b"PK")
    (frozen / "COMMIT.txt").write_text(
        "655949aa79ea32adfc103635d4cfb303ae24b94f\n", encoding="utf-8"
    )
    _, info = k.find_wheel(wctx, "gabrielezerzer/tipping-point-wheel-v1")
    check(
        "the plan's wheel dataset is the one installed", info["file"].startswith("tipping_point-")
    )
    (wctx.input / "tp-data").mkdir()
    (wctx.input / "tp-data" / "tipping_point-0.2.0-py3-none-any.whl").write_bytes(b"PK")
    try:
        k.find_wheel(wctx, "gabrielezerzer/tipping-point-wheel-v1")
        check("a second wheel anywhere among the inputs is refused", False)
    except k.Refused as exc:
        check(
            "a second wheel anywhere among the inputs is refused", "expected one" in str(exc), exc
        )
    plan = dict(copy.deepcopy(JEPA_PLAN), wheel="gabrielezerzer/tipping-point-wheel-v1")
    k.write_kernel(plan, root / "frozen")
    meta = json.loads((root / "frozen" / "kernel-metadata.json").read_text(encoding="utf-8"))
    check(
        "the kernel attaches the plan's wheel dataset",
        meta["dataset_sources"]
        == ["gabrielezerzer/tipping-point-wheel-v1", "gabrielezerzer/tp-data"],
        meta,
    )


def extraction_rates(k: ModuleType, root: Path) -> None:
    """Section 4 projects extraction per L, and a frame at L = 64 costs many at L = 16, so STATUS
    must not average them. The benchmark's shape: one checkpoint over several sizes."""
    ctx = ctx_at(k, root)
    trained, twin = {"run": "m", "file": k.FINAL}, {"run": "m", "file": k.TWIN}
    tables = (  # name, checkpoint, dataset run, L, frames, seconds of readout, of the whole process
        ("a", trained, "20261101T000001Z-L16-null-e2-s1", 16, 1000, 0.5, 2.0),
        ("b", trained, "20261101T000002Z-L16-ramp-x2-s1", 16, 3000, 1.5, 2.0),
        ("c", trained, "20261101T000003Z-L64-qs-s16", 64, 1000, 8.0, 10.0),
        ("d", twin, "20261101T000001Z-L16-null-e2-s1", 16, 2000, 3.0, 4.0),
    )
    plan = {
        "kernel": "tp-self-test",
        "runs": [
            {"name": "m", "command": "jepa train", "args": []},
            *(
                {"name": t[0], "command": "jepa extract", "args": [t[1], {"dataset_run": t[2]}]}
                for t in tables
            ),
        ],
    }
    session = k.Session(ctx, plan, {}, COMMIT, TORCH, 2)
    for name, _, run_id, L, frames, readout, whole in tables:
        folder = ctx.out / "ising" / run_id / "latents" / name
        folder.mkdir(parents=True)
        manifest = {"started_ns": 10**9, "params": {"dataset": {"L": L}}}
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        done = {"rows": {"latents": frames}, "ended_ns": 10**9 + int(readout * 1e9)}
        (folder / "complete.json").write_text(json.dumps(done), encoding="utf-8")
        run = session.by_name[name]
        run.state, run.folder = "complete", folder.relative_to(ctx.out).as_posix()
        run.wall_s = whole
    rates = session.measured()
    groups = rates["extract"]
    small, large = groups.get("m final.pt L16", {}), groups.get("m final.pt L64", {})
    check(
        "an extraction's rate is by checkpoint and L: one group for each pair",
        sorted(groups) == ["m ckpt/step_0000000.pt L16", "m final.pt L16", "m final.pt L64"],
        sorted(groups),
    )
    check(
        "... the tables at one L add up",
        (small.get("tables"), small.get("frames")) == (2, 4000)
        and small.get("readout_us_per_frame") == 500.0
        and small.get("us_per_frame") == 1000.0,
        small,
    )
    check(
        "... and L = 64 keeps its own rate, not an average with L = 16",
        large.get("readout_us_per_frame") == 8000.0 and large.get("us_per_frame") == 10000.0,
        large,
    )
    twin_rate = groups.get("m ckpt/step_0000000.pt L16", {}).get("readout_us_per_frame")
    check("... and the twin stays apart from final.pt", twin_rate == 1500.0, twin_rate)
    text = k.rates_text(rates)
    check(
        "STATUS gives the rate of each L",
        "extract m final.pt L16 500.0 us/frame readout" in text
        and "extract m final.pt L64 8000.0 us/frame readout" in text,
        text,
    )
    # a hand-made manifest may stop at any level above params.dataset.L; extract always writes it
    for name, manifest in (
        ("a", {"started_ns": 10**9}),
        ("b", {"started_ns": 10**9, "params": {}}),
        ("c", {"started_ns": 10**9, "params": {"dataset": {}}}),
    ):
        path = ctx.out / session.by_name[name].folder / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    rates = session.measured()
    check(
        "a table with no params.dataset.L groups without a size, not as LNone",
        sorted(rates["extract"]) == ["m ckpt/step_0000000.pt L16", "m final.pt"],
        sorted(rates["extract"]),
    )
    text = k.rates_text(rates)
    check(
        "... and STATUS gives its rate under that group",
        "extract m final.pt 2000.0 us/frame readout" in text and "LNone" not in text,
        text,
    )


def push_problems(folder: Path) -> list[str]:
    """What `kaggle kernels push` (CLI 2.2.5, KaggleApi.kernels_push) refuses in a folder before it
    sends anything, plus the T4 x2 shape a plan's gpu pins and one queue per T4 need."""
    problems = []
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    if len(meta.get("title") or "") < 5:
        problems.append("the title is under five characters")
    code = folder / str(meta.get("code_file") or "")
    if not code.is_file():
        problems.append(f"no source file {meta.get('code_file')!r}")
    else:
        ast.parse(code.read_text(encoding="utf-8"))
    owner, _, slug = str(meta.get("id")).partition("/")
    if not owner or not slug or "/" in slug:
        problems.append(f"id {meta.get('id')!r} is not <owner>/<slug>")
    if slug != meta.get("title"):  # the CLI warns of surprising behaviour when these differ
        problems.append(f"the title {meta.get('title')!r} does not resolve to the slug {slug!r}")
    if meta.get("language") not in ("python", "r", "rmarkdown"):
        problems.append(f"language {meta.get('language')!r}")
    if meta.get("kernel_type") not in ("script", "notebook"):
        problems.append(f"kernel_type {meta.get('kernel_type')!r}")
    for key in ("dataset_sources", "kernel_sources"):
        problems += [
            f"{key}: {s!r} is not <owner>/<slug>"
            for s in meta.get(key, [])
            if not re.fullmatch(r"[^/\s]+/[^/\s]+", s)
        ]
    if meta.get("machine_shape") != "NvidiaTeslaT4":  # the CLI's name for GPU T4 x2
        problems.append(f"machine_shape {meta.get('machine_shape')!r}: not T4 x2")
    if not (meta.get("enable_gpu") and meta.get("enable_internet") and meta.get("is_private")):
        problems.append("a kernel needs a GPU and the internet, and stays private")
    return problems


def leaks(folder: Path) -> list[str]:
    """Credentials, and data naming a person or a machine, in the files a push would send."""
    secret = re.compile(
        r"(?i)kaggle_key|kaggle_username|usersecrets|kaggle_secrets|github_token|ghp_|access_token"
    )
    email = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}")
    found = []
    for path in folder.iterdir():
        text = path.read_text(encoding="utf-8")
        found += [f"{path.name}: {m[0]}" for m in secret.finditer(text)]
        found += [f"{path.name}: {m[0]}" for m in email.finditer(text)]
        if str(Path.home()) in text:
            found.append(f"{path.name}: a local path")
    return found


def kernel_acceptance(k: ModuleType, root: Path) -> None:
    for name in (
        "job-a",
        "pilot-28",
        "primary-28",
        "first-read-m1",
        "first-read-m2",
        "v1-confirmatory",
        "ablations-28",
    ):
        source, folder = PLANS / f"{name}.json", root / name
        plan = json.loads(source.read_text(encoding="utf-8"))
        code = k.main(["--write-kernel", str(source), str(folder)])
        problems = push_problems(folder)
        check(
            f"{name}: --write-kernel writes a folder `kaggle kernels push` would take",
            code == 0 and not problems,
            problems,
        )
        check(
            f"{name}: the folder holds the kernel and its metadata alone",
            sorted(p.suffix for p in folder.iterdir()) == [".json", ".py"],
        )
        script = folder / f"{plan['kernel']}.py"
        loaded = runpy.run_path(str(script), run_name="tp_kernel_check")
        check(f"{name}: the plan is embedded whole", json.loads(loaded["PLAN_JSON"]) == plan)
        found = leaks(folder)
        check(f"{name}: no credentials and nothing identifying in the folder", not found, found)
    folder = root / "resumed"
    code = k.main(["--write-kernel", str(PLANS / "pilot-28.json"), str(folder), "--resume"])
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    check(
        "pilot-28 --resume: the same folder, with the kernel's own output attached last",
        code == 0
        and not push_problems(folder)
        and meta["kernel_sources"][-1] == f"{k.OWNER}/tp-jepa-pilot-28",
        meta,
    )
    install = inspect.getsource(k.ensure_env) + inspect.getsource(k.kernel_main)
    check(
        "the kernel's path installs the wheel: no clone, no token",
        "git" not in install and "token" not in install.lower() and "wheel" in install,
    )
    lock = tomllib.loads((PLANS.parents[1] / "uv.lock").read_text(encoding="utf-8"))
    locked = {(p["name"], p["version"]) for p in lock["package"]}
    pins = [tuple(pin.split("==")) for pin in (k.TORCH, *k.PINNED)]
    check("the kernel's pinned versions are uv.lock's", all(p in locked for p in pins), pins)


def invalid_plans(k: Any, root: Path) -> None:  # Any: it swaps the module's PLAN_JSON
    bad = copy.deepcopy(JEPA_PLAN)
    bad["runs"][1]["args"][0] = {"dataset_run": NULL}  # two simulator runs, no checkpoint
    path = root / "bad.json"
    root.mkdir(parents=True)
    path.write_text(json.dumps(bad), encoding="utf-8")
    dest = root / "never"
    code = k.main(["--write-kernel", str(path), str(dest)])
    check(
        "--write-kernel refuses an invalid plan: exit 2, no folder", code == 2 and not dest.exists()
    )
    check(
        "--dry-run refuses it too, before building anything", k.main(["--dry-run", str(path)]) == 2
    )

    saved, k.PLAN_JSON = k.PLAN_JSON, json.dumps(bad)
    try:
        ctx = ctx_at(k, root)
        code = k.kernel_main(ctx)
        said = (ctx.out / "STATUS").read_text(encoding="utf-8")
        check(
            "a kernel holding an invalid plan runs nothing: STATUS refused, exit 1",
            code == 1 and said.startswith("refused:") and "reads one checkpoint" in said,
            f"exit {code}: {said}",
        )
        k.PLAN_JSON = None
        code = k.kernel_main(ctx)
        said = (ctx.out / "STATUS").read_text(encoding="utf-8")
        check("a kernel with no plan at all: refused", said.startswith("refused: no plan"), said)
    finally:
        k.PLAN_JSON = saved

    def extract_args(args: list) -> str:
        plan = copy.deepcopy(JEPA_PLAN)
        plan["runs"][1]["args"] = args
        return refused(k, plan)

    both = {"dataset_run": NULL}
    trained = {"run": "m3", "file": "final.pt"}
    for what, args in (
        ("no simulator run", [trained]),
        ("no checkpoint", [both]),
        ("a checkpoint given as a plain path", ["final.pt", both]),
        ("a training with no file named", [{"run": "m3"}, both]),
        ("two simulator runs", [trained, both, {"dataset_run": STAIRS[0]}]),
    ):
        why = extract_args(args)
        check(f"refused: jepa extract with {what}", "one checkpoint" in why, why or "accepted")


PILOT_RECORD = "20261025T194846Z-tauc-L16-32-64-128"  # #54's record
PILOT_SHA256 = "7c5a4a296f2d8d1d55e223e4c775cd98267c43e6ae89f5e6ec594ac5c0581698"
DRAFT = "ising-latent-v1.toml"
DRAFT_RUN = "20261101T000000Z-ising-latent-v1-draft"  # its folder in tp-prereg-ising-latent
# the draft the trainings ran under; the frozen file's sha256 differs because its placeholders
# were filled
DRAFT_SHA256 = "4d9fd3ed79e02f3e0ec535dd02e2b4f8c6a716ceed5476a6a00fcc72e95552ad"
M1_RUN = "20261101T051047Z-L16-32-64-m1"  # primary model seed 1, in tp-jepa-primary-28's output
# The model each first read reads, in that output. Written out here as well as in its plan, so that
# a slip in a plan's pin turns red here and is not first refused on Kaggle.
FIRST_READS = {
    "m1": {
        "run": M1_RUN,
        "final": "3f251147a5072964176cab40122cbf855755d7a7dd77bf946eaaef661b43d149",
        "twin": "5e076a53e61d8c359d55c13738f8a2619e4af1002843de0c10b3ca33bc6d4224",
    },
    "m2": {
        "run": "20261101T051110Z-L16-32-64-m2",
        "final": "7638b436d5ee0cb39f68e6ce62f965b0c0c4f66ff9c402e80e93df0afad9aef4",
        "twin": "d22f7e494b6ee40bc6e319173e183c8c21709a3f5d149a0251b03fac84415176",
    },
}
DATASET_ID = re.compile(r"\d{8}T\d{6}Z-L(\d+)-(qs|ramp|null)-(?:[a-z0-9]+-)?s(\d+)")
KINDS = {"qs": "staircase", "ramp": "ramp", "null": "fixed"}


def flag_ints(args: list, flag: str) -> list[int]:
    """The integers a flag takes, up to the next flag."""
    values = args[args.index(flag) + 1 :]
    end = next((i for i, v in enumerate(values) if not isinstance(v, int)), len(values))
    return values[:end]


def stand_in_run(folder: Path, run_id: str) -> None:
    """A simulator run as the library's readers see it: its L, seed and kind from its id."""
    tag = DATASET_ID.fullmatch(run_id)
    assert tag is not None, run_id
    folder.mkdir(parents=True)
    params = {"L": int(tag[1]), "seed": int(tag[3]), "schedule": {"kind": KINDS[tag[2]]}}
    manifest = {"run_id": run_id, "params": params}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    done = {"clean": True, "files": {}, "checks": {}}
    (folder / "complete.json").write_text(json.dumps(done), encoding="utf-8")


def pilot_28(k: ModuleType, root: Path) -> None:
    from tipping_point.jepa.data import find_staircases

    plan = json.loads((PLANS / "pilot-28.json").read_text(encoding="utf-8"))
    runs = {r["name"]: r for r in plan["runs"]}
    trainings = {"timing": ([16, 32, 64], 99, 2000), "pilot": ([16, 32], 100, 10000)}
    errors = k.cli_errors(plan)
    check("pilot-28: the CLI itself takes every run's arguments", errors == [], errors)
    extractions = [r for r in plan["runs"] if r["command"] == "jepa extract"]
    check(
        "pilot-28: its two trainings run on GPU 0, every extraction on GPU 1",
        [r["name"] for r in plan["runs"] if r["command"] == "jepa train"] == list(trainings)
        and all(runs[n]["gpu"] == 0 for n in trainings)
        and all(r["gpu"] == 1 for r in extractions),
    )
    check(
        "pilot-28: its kernel sources are Job A's output and the tau_c kernel's",
        plan["kernel_sources"] == [f"{k.OWNER}/{job_a()['kernel']}", f"{k.OWNER}/tp-tauc-l64-l128"],
        plan["kernel_sources"],
    )
    record = plan["inputs"]["record"]
    check(
        "pilot-28: its --pilot is #54's record, pinned by sha256",
        (record["run"], record["file"], record["sha256"])
        == (PILOT_RECORD, "pilot.json", PILOT_SHA256),
        record,
    )
    for name, (sizes, seed, steps) in trainings.items():
        args = runs[name]["args"]
        seeds = [*flag_ints(args, "--train-seeds"), *flag_ints(args, "--val-seed")]
        check(
            f"pilot-28: {name} is the research pass's: its sizes, seeds 11-16, model seed, steps",
            flag_ints(args, "--L") == sizes
            and seeds == [11, 12, 13, 14, 15, 16]
            and flag_ints(args, "--seed") == [seed]
            and flag_ints(args, "--steps") == [steps]
            and {"input": "record"} in args,
            args,
        )
        tags = [DATASET_ID.fullmatch(s) for s in runs[name]["stage"]]
        staged = sorted((int(t[1]), int(t[3])) for t in tags if t and t[2] == "qs")
        check(
            f"pilot-28: {name}'s stage ids are one staircase for each of its (L, seed)",
            staged == sorted((L, s) for L in sizes for s in seeds)
            and len(staged) == len(runs[name]["stage"]),
            runs[name]["stage"],
        )
    check(
        "pilot-28: timing checkpoints every 500 steps, the pilot at the training's default",
        flag_ints(runs["timing"]["args"], "--checkpoint-every") == [500]
        and "--checkpoint-every" not in runs["pilot"]["args"],
        runs["timing"]["args"],
    )
    made = {
        tuple(flag_ints(r["args"], f)[0] for f in ("--L", "--seed"))
        for r in job_a()["runs"]
        if r["command"] == "staircase"
    }
    large = {
        (int(t[1]), int(t[3]))
        for t in map(DATASET_ID.fullmatch, runs["timing"]["stage"])
        if t and t[1] == "64"
    }
    check("pilot-28: its L = 64 stage ids are the staircases Job A makes", large == made, large)
    read = [(r["args"][0], r["args"][1]["dataset_run"], r["args"][2:]) for r in extractions]
    untrained = all(c["file"] == k.TWIN for c, _, _ in read if c["run"] == "timing")
    tagged = [DATASET_ID.fullmatch(d) for _, d, _ in read]
    wrong = [
        d
        for (_, d, rest), t in zip(read, tagged, strict=True)
        if t is None or (rest == ["--select", "tc-plus"]) != (t[2] == "qs")
    ]
    check("pilot-28: staircases are read as tc-plus, ramps and nulls whole", not wrong, wrong)
    seeds = {int(t[3]) for t in tagged if t}
    check(
        "pilot-28: extraction reads the exploratory seeds and the validation seed alone",
        seeds == {1, 2, 3, 4, 5, 16} and untrained,
        seeds,
    )
    pairs = collections.Counter((c["run"], c["file"], d) for c, d, _ in read if c["run"] == "pilot")
    exploratory = {d for c, d, _ in read if c["run"] == "pilot"}
    check(
        "pilot-28: the pilot's final.pt and twin each read all 60 exploratory runs, once",
        len(exploratory) == 60
        and all(pairs[("pilot", f, d)] == 1 for f in (k.TWIN, k.FINAL) for d in exploratory),
        len(exploratory),
    )
    benched = {d for c, d, _ in read if c["run"] == "timing"}
    check(
        "pilot-28: the benchmark reads a null and a slow ramp per L, and the L = 64 staircase",
        sorted(d.split("Z-", 1)[1] for d in benched)
        == [
            "L16-null-e2-s1",
            "L16-ramp-x0p02-s1",
            "L32-null-e2-s1",
            "L32-ramp-x0p02-s1",
            "L64-qs-s16",
        ],
        benched,
    )

    # the plan on the mounts Kaggle gives it, each run where its source holds it: only the record
    # is a stand-in, since its bytes are on Kaggle and the runner checks them there
    ctx = ctx_at(k, root)
    ctx.py = sys.executable
    ctx.helper.write_text(k.HELPER_SOURCE, encoding="utf-8")
    trial = copy.deepcopy(plan)
    folder = ctx.input / "tp-tauc-l64-l128" / "out" / "ising" / PILOT_RECORD
    folder.mkdir(parents=True)
    (folder / "pilot.json").write_text("{}", encoding="utf-8")
    trial["inputs"]["record"]["sha256"] = hashlib.sha256(b"{}").hexdigest()
    session = k.Session(ctx, trial, {}, COMMIT, TORCH, 2)
    dataset_ids = k.dataset_runs(session.runs)
    job_output = ctx.input / "tp-ising-job-a" / "out" / "ising"
    dataset = ctx.input / "tipping-point-ising-l16-l32"
    for run_id in dataset_ids:
        stand_in_run((job_output if "-L64-" in run_id else dataset) / run_id, run_id)
    session.locate_inputs()
    session.locate_datasets()
    check(
        "pilot-28: all 78 dataset runs are found among the attached sources",
        len(session.datasets) == 78 and set(session.datasets) == set(dataset_ids),
        len(session.datasets),
    )
    errors = session.capture(session.runs)
    check("pilot-28: the helper's capture, as on Kaggle, refuses nothing", errors == [], errors)
    session.stage()
    for name, (sizes, _, _) in trainings.items():
        found = find_staircases(ctx.out, sizes, [11, 12, 13, 14, 15, 16])
        check(
            f"pilot-28: staged, the training's own lookup finds {name}'s runs, and only one each",
            {r.run_id for r in found.values()} == set(runs[name]["stage"]),
            sorted(r.run_id for r in found.values()),
        )
    for name in trainings:
        training = session.by_name[name]
        training.state, training.folder = "complete", f"jepa/{name}"
        (ctx.out / training.folder / "ckpt").mkdir(parents=True)
        (ctx.out / training.folder / k.FINAL).write_bytes(b"trained")
        (ctx.out / training.folder / k.TWIN).write_bytes(b"untrained")
    tail, _ = session.resolve(session.by_name["bench-L64-qs-s16"])
    check(
        "pilot-28: an extraction resolves to its staged run, tc-plus kept",
        tail[0] == str(ctx.out / session.by_name["timing"].folder / k.TWIN)
        and tail[1] == str(ctx.out / "ising" / "20261025T223231Z-L64-qs-s16")
        and tail[2:] == ["--select", "tc-plus"],
        tail,
    )
    unresolved = []
    for r in session.runs:
        if r.command == "jepa extract":
            try:
                session.extract_paths(r)
            except k.RunError as exc:
                unresolved.append(str(exc))
    check(
        "pilot-28: every extraction names its checkpoint and its simulator run",
        not unresolved,
        unresolved,
    )
    every = flag_ints(runs["timing"]["args"], "--checkpoint-every")[0]
    ends = [*range(every, trainings["timing"][2] + 1, every), trainings["timing"][2]]  # + closing
    clock = [i * 10**9 for i in range(len(ends) + 1)]
    ledger = [
        {"run_id": "timing", "from_ns": a, "to_ns": b, "step": step}
        for (a, b), step in zip(itertools.pairwise(clock), ends, strict=True)
    ]
    (ctx.out / "jepa" / k.LEDGER).write_text(
        "".join(json.dumps(line) + "\n" for line in ledger), encoding="utf-8"
    )
    session.by_name["timing"].run_id = "timing"
    rate = session.measured()["train"].get("timing", {})
    check(
        "pilot-28: so timing's rate comes from 1,500 steps after warm-up, not from one interval",
        rate.get("steps") == 1500,
        rate,
    )
    session.unstage()
    left = [p.name for p in (ctx.out / "ising").iterdir()]
    check("pilot-28: staged copies that gain no latents leave the output", not left, left)


def noisy_env(k: ModuleType, root: Path) -> str:
    """An environment as the kernel's reads on Kaggle, with no wheel to build: a venv holding a copy
    of the package stamped with COMMIT and clean (so no checkout's state shows through), this
    environment's dependencies, and Kaggle's failing sitecustomize. Its python."""
    import tipping_point

    env = root / "env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(env)], check=True)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    where = k.kaggle_noise(python)
    package = Path(tipping_point.__file__).parent
    shutil.copytree(package, where / "tipping_point", ignore=shutil.ignore_patterns("__pycache__"))
    stamp = f"commit = {COMMIT!r}\ndirty = False\nkernel_sha256 = None\nbuilt_at = None\n"
    (where / "tipping_point" / "_build_info.py").write_text(stamp, encoding="utf-8")
    deps = site.getsitepackages()[-1]
    (where / "tp_deps.pth").write_text(
        f"import site; site.addsitedir({deps!r})\n", encoding="utf-8"
    )
    return str(python)


def real_runs(k: ModuleType, root: Path) -> None:
    root.mkdir(parents=True)
    python = noisy_env(k, root)
    said = subprocess.run(
        [python, "-I", "-c", "from tipping_point.ising import io; print(io.code_version())"],
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "the stand-in environment names the wheel's commit, clean",
        said.returncode == 0 and f"'git_sha': '{COMMIT}', 'dirty': False" in said.stdout,
        said.stdout + said.stderr,
    )
    check(
        "... and its children print Kaggle's noise on stderr and exit 0",
        said.returncode == 0 and all(line in said.stderr for line in NOISE),
        said.stderr,
    )
    wheels = root / "input" / k.WHEEL_DATASET
    wheels.mkdir(parents=True)
    (wheels / "tipping_point-0.1.0-py3-none-any.whl").write_bytes(b"PK")
    (wheels / "COMMIT.txt").write_text(f"{COMMIT}\n", encoding="utf-8")
    plan = k.tiny_jepa_plan(root)
    with open(root / "sessions.log", "w", encoding="utf-8") as sink:
        code = k.rehearse(plan, root, python, 2, 1, sink)
    tail = (root / "sessions.log").read_text(encoding="utf-8").splitlines()[-40:]
    check("the written kernel's two sessions come out as they should", code == 0, "\n".join(tail))

    out1, out2 = root / "input" / plan["kernel"] / "out", root / "working" / "out"
    first, last = (json.loads((o / k.STATE).read_text(encoding="utf-8")) for o in (out1, out2))
    by1, by2 = ({r["name"]: r for r in s["runs"]} for s in (first, last))
    training = by1["tiny"]["folder"]
    check(
        "a real jepa train, stopped by the deadline: interrupted, its checkpoint kept",
        by1["tiny"]["state"] == "interrupted"
        and (out1 / training / "checkpoint.pt").is_file()
        and not (out1 / training / "complete.json").exists(),
        by1["tiny"],
    )
    check(
        "... its dependants not started, STATUS incomplete and how to continue",
        by1["x-final"]["state"] == by1["x-twin"]["state"] == "not started"
        and (out1 / "STATUS").read_text(encoding="utf-8").startswith("incomplete:")
        and "--resume" in first["status_detail"],
        first["status_detail"],
    )
    status = (out2 / "STATUS").read_text(encoding="utf-8")
    check("session 2: STATUS complete, all four verified", status.startswith("complete:"), status)
    check(
        "... the finished simulator run reused, the training resumed in its folder",
        by2["quick"]["reused"]
        and by2["tiny"]["resumed"]
        and by2["tiny"]["exit"] == 0
        and by2["tiny"]["folder"] == training,
        by2["tiny"],
    )
    done = json.loads((out2 / training / "complete.json").read_text(encoding="utf-8"))
    args = next(r for r in plan["runs"] if r["name"] == "tiny")["args"]
    steps = flag_ints(args, "--steps")[0]
    check(
        "... resume lost and repeated no step: every step's row once, none stopped early",
        done["rows"]["train"] == steps
        and done["last_key"] == steps
        and done["stopped_early"] is False,
        done["rows"],
    )
    ledger = [
        json.loads(line)
        for line in (out2 / "jepa" / k.LEDGER).read_text(encoding="utf-8").splitlines()
    ]
    joins = sum(a["to_ns"] != b["from_ns"] for a, b in itertools.pairwise(ledger))
    check(
        "... the GPU-hour ledger came along and holds both sessions",
        joins == 1 and {x["run_id"] for x in ledger} == {by2["tiny"]["run_id"]},
        ledger,
    )
    check(
        "a finished pilot's folder holds the tau.json a staircase reads",
        (out2 / by2["quick"]["folder"] / "tau.json").is_file(),
        by2["quick"],
    )
    for name, untrained, rule in (("x-twin", True, "tc-plus"), ("x-final", False, "all")):
        table = out2 / by2[name]["folder"]
        manifest = json.loads((table / "manifest.json").read_text(encoding="utf-8"))
        rows = json.loads((table / "complete.json").read_text(encoding="utf-8"))["rows"]
        check(
            f"a real jepa extract, {name}: complete and verified, its checkpoint and rule named",
            by2[name]["state"] == "complete"
            and manifest["params"]["checkpoint"]["untrained"] is untrained
            and manifest["params"]["checkpoint"]["run_id"] == by2["tiny"]["run_id"]
            and manifest["params"]["selection"]["rule"] == rule
            and manifest["producer"]["git_sha"] == COMMIT
            and manifest["producer"]["dirty"] is False
            and rows["latents"] > 0,
            manifest["params"]["selection"],
        )
    rates = last["rates"]
    check(
        "STATUS gives the measured rates: seconds per step, microseconds per frame",
        rates["train"]["tiny"]["s_per_step"] > 0
        and rates["extract"][f"tiny {k.FINAL} L16"]["us_per_frame"] > 0
        and "s/step" in status
        and "us/frame" in status,
        status,
    )
    for path in (out2 / "logs" / "tiny.log", out2 / "logs" / "x-final.log"):
        text = path.read_text(encoding="utf-8")
        check(
            f"{path.name} holds Kaggle's noise: the runs above passed through it",
            all(line in text for line in NOISE),
            text[:300],
        )
    stage = next(r for r in plan["runs"] if r["name"] == "tiny")["stage"]
    check(
        "the training's staged runs have left the output, the extracted ones stayed",
        not any((out2 / "ising" / s).exists() for s in stage)
        and (out2 / by2["x-final"]["folder"]).is_dir(),
        sorted(p.name for p in (out2 / "ising").iterdir()),
    )

    ctx = ctx_at(k, root)
    ctx.py = python
    session = k.Session(ctx, plan, {}, COMMIT, last["env"]["torch"], 2, {"runs": last["runs"]})
    tiny = session.by_name["tiny"]
    folder = out2 / training
    state, note = session.settle(tiny, folder, 0, collections.deque(NOISE), "")
    check("exit 0 and files that verify: complete, whatever stderr said", state == "complete", note)
    state, note = session.settle(tiny, folder, 1, collections.deque(), "")
    check(
        "exit 1 with a complete.json in place: failed, the exit code decides",
        state == "failed" and note.startswith("exit 1"),
        note,
    )
    final = out2 / by2["x-final"]["folder"]
    chunk = sorted(final.glob("*.parquet"))[0]
    kept = chunk.read_bytes()
    chunk.write_bytes(kept + b"x")
    state, note = session.settle(session.by_name["x-final"], final, 0, collections.deque(), "")
    chunk.write_bytes(kept)
    check(
        "exit 0 and a changed file: failed, jepa verify decides",
        state == "failed" and "sha256 differs" in note,
        (state, note),
    )

    missing = copy.deepcopy(plan)
    entry = next(r for r in missing["runs"] if r["name"] == "tiny")
    entry["args"][entry["args"].index("--train-seeds") + 1] = 999
    entry["name"], entry["stage"] = "absent", []
    missing["runs"] = [entry]
    session = k.Session(ctx, missing, {}, COMMIT, last["env"]["torch"], 2)
    session.locate_inputs()
    state, note = session.execute(0, session.by_name["absent"])
    check(
        "a real child that exits 1: failed, with its own error, not the noise",
        state == "failed" and "no complete staircase run" in note,
        (state, note),
    )

    # x-final's table is complete, so `jepa extract` of it again exits 1: a real failed extraction,
    # with a real one behind it on the same queue and another that reads the failure
    specs = {r["name"]: r for r in plan["runs"]}
    again = dict(specs["x-final"], name="x-again", gpu=0)
    null = again["args"][1]
    reader = dict(again, name="x-reader", args=[{"run": "x-again", "file": k.FINAL}, null])
    behind = dict(again, name="x-behind", args=[{"run": "tiny", "file": k.TWIN}, null])
    queued = dict(plan, runs=[specs["tiny"], again, reader, behind])
    prior = {"runs": [by2["tiny"]]}
    session = k.Session(ctx, queued, {}, COMMIT, last["env"]["torch"], 1, prior)
    session.by_name["tiny"].state = "complete"
    session.scheduler.run(session.tick, 0.05)
    by = session.by_name
    check(
        "a real failed extraction: failed for its own reason, the queue not stopped",
        by["x-again"].state == "failed"
        and "a latents table is written once" in by["x-again"].note
        and session.scheduler.stopped == {},
        [(r.name, r.state, r.note) for r in session.runs],
    )
    check(
        "... the one reading it blocked, the real one behind it complete and verified",
        by["x-reader"].state == "blocked"
        and by["x-behind"].state == "complete"
        and (out2 / str(by["x-behind"].folder) / "complete.json").is_file(),
        [(r.name, r.state, r.note) for r in session.runs],
    )

    # a checkpoint read out of an input mount, as a first-read plan reads its model's: the
    # training's folder attached as a kernel output, where jepa extract needs complete.json beside
    # final.pt (#127)
    source = "tp-tiny-primary"
    attached = root / "input" / source / "out" / "jepa" / by2["tiny"]["run_id"]
    shutil.copytree(out2 / training, attached)
    digest = hashlib.sha256((attached / k.FINAL).read_bytes()).hexdigest()
    mounted = k.load_plan(
        dict(
            plan,
            kernel_sources=[f"{k.OWNER}/{source}"],
            inputs={"tiny_final": {"run": attached.name, "file": k.FINAL, "sha256": digest}},
            runs=[
                {
                    "name": "x-input",
                    "command": "jepa extract",
                    "args": [{"input": "tiny_final"}, *specs["x-twin"]["args"][1:]],
                }
            ],
        )
    )
    session = k.Session(ctx, mounted, {}, COMMIT, last["env"]["torch"], 1)
    session.locate_inputs()
    session.scheduler.run(session.tick, 0.05)
    run = session.by_name["x-input"]
    table = k.read_json(out2 / str(run.folder) / "manifest.json") or {}
    named = (table.get("params") or {}).get("checkpoint") or {}
    check(
        "a real jepa extract of an {'input'} checkpoint in a kernel output: complete and verified, "
        "its sha256 and step named, no local path, STATUS's rate by input",
        run.state == "complete"
        and (named.get("sha256"), named.get("step"), named.get("path")) == (digest, steps, None)
        and "extract tiny_final L16" in k.rates_text(session.measured()),
        (run.state, run.note, named),
    )


def primary_28(k: ModuleType, root: Path) -> None:
    from tipping_point.jepa.data import find_staircases

    plan = json.loads((PLANS / "primary-28.json").read_text(encoding="utf-8"))
    runs = {r["name"]: r for r in plan["runs"]}
    errors = k.cli_errors(plan)
    check("primary-28: the CLI itself takes every run's arguments", errors == [], errors)
    check(
        "primary-28: five trainings and nothing else, on model seeds 1-5",
        all(r["command"] == "jepa train" for r in plan["runs"])
        and [flag_ints(r["args"], "--seed") for r in plan["runs"]] == [[s] for s in range(1, 6)],
        list(runs),
    )
    check(
        "primary-28: one process per T4, the three first on GPU 0",
        {n: r["gpu"] for n, r in runs.items()} == {"m1": 0, "m2": 1, "m3": 0, "m4": 1, "m5": 0},
    )
    prereg = plan["inputs"]["prereg"]
    check(
        "primary-28: its --prereg is the draft it trained under, pinned by sha256",
        (prereg["run"], prereg["file"], prereg["sha256"]) == (DRAFT_RUN, DRAFT, DRAFT_SHA256),
        prereg,
    )
    record = plan["inputs"]["record"]
    check(
        "primary-28: its --pilot is #54's record, pinned by sha256",
        (record["run"], record["file"], record["sha256"])
        == (PILOT_RECORD, "pilot.json", PILOT_SHA256),
        record,
    )
    made = {
        tuple(flag_ints(r["args"], f)[0] for f in ("--L", "--seed"))
        for r in job_a()["runs"]
        if r["command"] == "staircase"
    }
    for name, run in runs.items():
        args = run["args"]
        seeds = [*flag_ints(args, "--train-seeds"), *flag_ints(args, "--val-seed")]
        check(
            f"primary-28: {name} trains all three sizes on seeds 11-16 for 25,000 steps",
            flag_ints(args, "--L") == [16, 32, 64]
            and seeds == [11, 12, 13, 14, 15, 16]
            and flag_ints(args, "--steps") == [25000]
            and {"input": "record"} in args
            and {"input": "prereg"} in args
            and "--warmup" not in args,  # jepa train's own: steps / 20
            args,
        )
        tags = [DATASET_ID.fullmatch(s) for s in run["stage"]]
        staged = sorted((int(t[1]), int(t[3])) for t in tags if t and t[2] == "qs")
        large = {pair for pair in staged if pair[0] == 64}
        check(
            f"primary-28: {name}'s stage ids are one staircase per (L, seed), L = 64 Job A's",
            staged == sorted((L, s) for L in (16, 32, 64) for s in seeds)
            and len(staged) == len(run["stage"])
            and large == made,
            run["stage"],
        )
    # the stand-ins below are built from these ids, so only another plan's list catches a wrong one
    pilot = json.loads((PLANS / "pilot-28.json").read_text(encoding="utf-8"))
    timing = next(r["stage"] for r in pilot["runs"] if r["name"] == "timing")
    apart = [name for name, run in runs.items() if sorted(run["stage"]) != sorted(timing)]
    check("primary-28: each model's stage ids are pilot-28's timing stage ids", not apart, apart)

    # on the mounts Kaggle gives it: the record's and the draft's bytes are stand-ins here, the
    # runner checks the real ones there, so only their sha256 change
    ctx = ctx_at(k, root)
    ctx.py = sys.executable
    ctx.helper.write_text(k.HELPER_SOURCE, encoding="utf-8")
    trial = copy.deepcopy(plan)
    folder = ctx.input / "tp-tauc-l64-l128" / "out" / "ising" / PILOT_RECORD
    folder.mkdir(parents=True)
    (folder / "pilot.json").write_text("{}", encoding="utf-8")
    trial["inputs"]["record"]["sha256"] = hashlib.sha256(b"{}").hexdigest()
    held = ctx.input / "tp-prereg-ising-latent" / trial["inputs"]["prereg"]["run"]
    held.mkdir(parents=True)
    (held / DRAFT).write_text("id = 'draft'", encoding="utf-8")
    trial["inputs"]["prereg"]["sha256"] = hashlib.sha256(b"id = 'draft'").hexdigest()
    session = k.Session(ctx, trial, {}, COMMIT, TORCH, 2)
    job_output = ctx.input / "tp-ising-job-a" / "out" / "ising"
    dataset = ctx.input / "tipping-point-ising-l16-l32"
    for run_id in k.dataset_runs(session.runs):
        stand_in_run((job_output if "-L64-" in run_id else dataset) / run_id, run_id)
    session.locate_inputs()
    session.locate_datasets()
    check("primary-28: the 18 staircases are found among the sources", len(session.datasets) == 18)
    errors = session.capture(session.runs)
    check("primary-28: the helper's capture, as on Kaggle, refuses nothing", errors == [], errors)
    session.stage()
    found = find_staircases(ctx.out, [16, 32, 64], [11, 12, 13, 14, 15, 16])
    check(
        "primary-28: staged, the training's own lookup finds its runs, and only one each",
        {r.run_id for r in found.values()} == set(runs["m1"]["stage"]),
    )
    tail, key = session.resolve(session.by_name["m3"])
    check(
        "primary-28: --prereg and --pilot resolve to the located files",
        tail[tail.index("--prereg") + 1] == str(held / DRAFT)
        and tail[tail.index("--pilot") + 1] == str(folder / "pilot.json")
        and f"input/{trial['inputs']['prereg']['run']}/{DRAFT}" in key,
        tail,
    )


def first_read_plan(model: str) -> dict:
    return json.loads((PLANS / f"first-read-{model}.json").read_text(encoding="utf-8"))


def first_read_inputs(k: ModuleType, model: str) -> dict:
    """The inputs a model's first read must name: its two files, pinned as FIRST_READS says."""
    pins = FIRST_READS[model]
    return {
        f"{model}_final": {"run": pins["run"], "file": k.FINAL, "sha256": pins["final"]},
        f"{model}_twin": {"run": pins["run"], "file": k.TWIN, "sha256": pins["twin"]},
    }


def first_read(k: ModuleType, root: Path, model: str) -> None:
    label, trained = f"first-read-{model}", FIRST_READS[model]["run"]
    plan = first_read_plan(model)
    pilot = json.loads((PLANS / "pilot-28.json").read_text(encoding="utf-8"))
    check(
        f"{label}: its kernel is its own, so a push is no new version of another read's",
        plan["kernel"] == f"tp-jepa-{label}",
        plan["kernel"],
    )
    errors = k.cli_errors(plan)
    check(f"{label}: the CLI itself takes every run's arguments", errors == [], errors)
    checkpoints = {k.FINAL: f"{model}_final", k.TWIN: f"{model}_twin"}
    mirrored = [
        {
            "name": r["name"],
            "command": r["command"],
            "args": [{"input": checkpoints[r["args"][0]["file"]]}, *r["args"][1:]],
        }
        for r in pilot["runs"]
        if r["name"].startswith("x-")
    ]
    check(
        f"{label}: pilot-28's own read and nothing else: its names, order and arguments, on "
        f"{model}'s two files, no run pinned so that each T4 takes the next",
        plan["runs"] == mirrored,
    )
    entries = plan["inputs"]
    check(
        f"{label}: its checkpoints are {model}'s final.pt and twin, pinned, from primary-28",
        plan["kernel_sources"] == [f"{k.OWNER}/tp-jepa-primary-28"]
        and plan["datasets"] == pilot["datasets"]
        and entries == first_read_inputs(k, model),
        entries,
    )
    check(
        f"{label}: its wheel is the project's wheel dataset, by name: Kaggle attaches the "
        "dataset's latest version",
        plan["wheel"] == f"{k.OWNER}/{k.WHEEL_DATASET}",
        plan["wheel"],
    )
    check(
        f"{label}: its description names model seed {model[1:]} and the commit that trained",
        f"Model seed {model[1:]}'s final.pt" in plan["description"]
        and "must hold e598e62, the commit that trained" in plan["description"],
        plan["description"],
    )
    read = [
        (r["args"][0]["input"], r["args"][1]["dataset_run"], r["args"][2:]) for r in plan["runs"]
    ]
    tagged = [DATASET_ID.fullmatch(d) for _, d, _ in read]
    wrong = [
        d
        for (_, d, rest), t in zip(read, tagged, strict=True)
        if t is None or rest != (["--select", "tc-plus"] if t[2] == "qs" else [])
    ]
    check(f"{label}: staircases are read as tc-plus, ramps and nulls whole", not wrong, wrong)
    check(
        f"{label}: it reads seeds 1-5 at L = 16 and 32, and nothing else",
        {int(t[3]) for t in tagged if t} == {1, 2, 3, 4, 5}
        and {int(t[1]) for t in tagged if t} == {16, 32},
    )
    pairs = collections.Counter((name, d) for name, d, _ in read)
    runs = {d for _, d in pairs}
    check(
        f"{label}: final.pt and the twin each read all 60 runs, once",
        len(runs) == 60 and all(pairs[(n, d)] == 1 for n in entries for d in runs),
        len(runs),
    )
    turned = {}
    for seed in CONFIRMATORY_SEEDS:
        variant = copy.deepcopy(plan)
        variant["runs"][-1]["args"][1]["dataset_run"] = f"20261025T031229Z-L32-ramp-x2-s{seed}"
        if why := refused(k, variant):
            turned[seed] = why
    check(f"{label}: a run at any of seeds 6-10 is accepted at load", not turned, turned)

    # on the mounts Kaggle gives it: the model's bytes are stand-ins here, the runner checks the
    # real ones
    ctx = ctx_at(k, root)
    ctx.py = sys.executable
    ctx.helper.write_text(k.HELPER_SOURCE, encoding="utf-8")
    trial = copy.deepcopy(plan)
    output = ctx.input / "tp-jepa-primary-28" / "out"
    held = output / "jepa" / trained
    for name, entry in trial["inputs"].items():
        (held / entry["file"]).parent.mkdir(parents=True, exist_ok=True)
        (held / entry["file"]).write_text(name, encoding="utf-8")
        entry["sha256"] = hashlib.sha256(name.encode()).hexdigest()
    named = {"name": "x-twin-L16-qs-s1", "folder": f"jepa/{model}"}  # a name this plan has as well
    state = {"kernel": "tp-jepa-primary-28", "runs": [named]}
    (output / k.STATE).write_text(json.dumps(state), encoding="utf-8")
    check(
        f"{label}: the primary kernel's own state, attached with its output, is no previous "
        "session of this kernel",
        k.carry_forward(ctx, trial) is None,
    )
    session = k.Session(ctx, trial, {}, COMMIT, TORCH, 2)
    dataset = ctx.input / "tipping-point-ising-l16-l32"
    for run_id in k.dataset_runs(session.runs):
        stand_in_run(dataset / run_id, run_id)
    try:
        session.locate_inputs()
        why = ""
    except k.Refused as exc:
        why = str(exc)
    check(
        f"{label}: with no complete.json beside {model}'s final.pt, the session is refused",
        f"input {model}_final:" in why and "has no complete.json" in why,
        why or "accepted",
    )
    (held / "complete.json").write_text("{}", encoding="utf-8")  # a finished one holds it (#128)
    session.locate_inputs()
    session.locate_datasets()
    check(
        f"{label}: {model}'s two files and the 60 runs are found among the attached sources",
        set(session.inputs) == set(entries) and set(session.datasets) == runs,
        len(session.datasets),
    )
    errors = session.capture(session.runs)
    check(f"{label}: the helper's capture, as on Kaggle, refuses nothing", errors == [], errors)
    session.stage()
    staircase = next(d for d in runs if d.endswith("-L16-qs-s1"))
    tail, key = session.resolve(session.by_name["x-final-L16-qs-s1"])
    check(
        f"{label}: an extraction resolves to {model}'s file and its staged run, tc-plus kept",
        tail == [str(held / k.FINAL), str(ctx.out / "ising" / staircase), "--select", "tc-plus"]
        and f"input/{trained}/{k.FINAL}" in key,
        tail,
    )
    unresolved = []
    for r in session.runs:
        try:
            session.extract_paths(r)
        except k.RunError as exc:
            unresolved.append(str(exc))
    check(
        f"{label}: every extraction names its checkpoint and its simulator run",
        not unresolved,
        unresolved,
    )
    for name in ("x-twin-L16-qs-s1", "x-final-L16-qs-s1"):
        table = ctx.out / "ising" / staircase / "latents" / name
        table.mkdir(parents=True)
        manifest = {"started_ns": 10**9, "params": {"dataset": {"L": 16}}}
        (table / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        done = {"rows": {"latents": 1000}, "ended_ns": 2 * 10**9}
        (table / "complete.json").write_text(json.dumps(done), encoding="utf-8")
        run = session.by_name[name]
        run.state, run.folder = "complete", table.relative_to(ctx.out).as_posix()
    text = k.rates_text(session.measured())
    check(
        f"{label}: STATUS keeps final.pt's rate and the twin's apart, by input",
        f"extract {model}_final L16" in text and f"extract {model}_twin L16" in text,
        text,
    )


def first_reads(k: ModuleType, root: Path) -> None:
    written = sorted(path.stem for path in PLANS.glob("first-read-*.json"))
    check(
        "every first-read plan has its pins in FIRST_READS, so the checks below reach it",
        written == [f"first-read-{model}" for model in sorted(FIRST_READS)],
        written,
    )
    # a plan copied whole, its run and hash both left at the model before, would pass every other
    # check against pins copied the same way, and read that model again with nothing to refuse it
    pinned = [pin for pins in FIRST_READS.values() for pin in pins.values()]
    check(
        "no two first reads share a run or a sha256, and each run's id names its model",
        len(set(pinned)) == len(pinned)
        and all(pins["run"].endswith(f"-{model}") for model, pins in FIRST_READS.items()),
        pinned,
    )
    for model in FIRST_READS:
        first_read(k, root / model, model)
    base, *others = FIRST_READS
    for model in others:
        plan, expected = first_read_plan(model), first_read_plan(base)
        expected["kernel"] = f"tp-jepa-first-read-{model}"
        expected["description"] = plan["description"]  # first_read reads it
        expected["inputs"] = first_read_inputs(k, model)
        for run in expected["runs"]:
            run["args"][0]["input"] = run["args"][0]["input"].replace(base, model)
        apart = sorted(key for key in {*plan, *expected} if plan.get(key) != expected.get(key))
        check(
            f"first-read-{model}: first-read-{base}'s plan but for its kernel name, description, "
            "its inputs' names, run and sha256, and the runs' references to those inputs",
            not apart,
            apart,
        )


V1_PLAN = PLANS / "v1-confirmatory.json"
V1_PREREG = PLANS.parents[1] / "paper" / "prereg" / "ising-ews-v1.toml"
V1_RECORD = (
    PLANS.parents[1] / "python" / "tipping_point" / "tests" / "fixtures" / "ising_pilot.json"
)
V1_KERNEL = "tp-ising-v1-confirmatory"
V1_WHEEL = "gabrielezerzer/tipping-point-wheel-v1"
V1_SOURCES = ["gabrielezerzer/tp-ising-job-a", "gabrielezerzer/tp-tauc-l64-l128"]
# the grid pilots whose tau.json the frozen file pins, in tp-ising-job-a's output: it names them in
# comments alone, so they are written out here
V1_GRID_PILOTS = {
    16: "20261025T220933Z-L16-pilot-s6",
    32: "20261025T220814Z-L32-pilot-s6",
    64: "20261025T220810Z-L64-pilot-s6",
}
V1_RESUME = (
    "uv run --no-sync python notebooks/kaggle_ising_runs.py --write-kernel "
    "notebooks/plans/v1-confirmatory.json ../.wf/kaggle/kernel-v1-confirmatory-resume --resume; "
    "kaggle kernels push -p ../.wf/kaggle/kernel-v1-confirmatory-resume"
)
V1_TAG_CLI = r'''
"""Each plan's runs through a copy of the tag's package: its own parser and schedules, with
run.start swapped for a stub that writes the manifest the tag would write and stops. Nothing is
simulated.
Usage: <this> <folder holding tipping_point> <job.json> <out.json>"""
import contextlib
import io as text_io
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import tipping_point
import torch
from tipping_point.ising import __main__ as cli
from tipping_point.ising import io, run

assert Path(tipping_point.__file__).is_relative_to(sys.argv[1]), tipping_point.__file__
made = []


def stub(spec, schedule, root, tag, device=None, pilot=None):
    run_id = f"20261102T{len(made):06d}Z-{tag}"  # engine.run_id's shape, unique within this process
    folder = Path(root) / "ising" / run_id
    folder.mkdir(parents=True)
    manifest = run._manifest(spec, schedule, run_id, torch.device(device), pilot)
    io.write_json(folder / "manifest.json", manifest)
    made.append(run_id)
    return folder


run.start = stub
results = {}
for label, runs in json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")).items():
    results[label] = {}
    for name, argv in runs.items():
        said = text_io.StringIO()
        try:
            with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
                cli.main(argv)
            results[label][name] = {"run_id": made[-1]}
        except SystemExit as exc:  # argparse's usage error, or the CLI's own refusal
            why = exc.code if isinstance(exc.code, str) else said.getvalue().strip()[-300:]
            results[label][name] = {"error": why}
Path(sys.argv[3]).write_text(json.dumps(results), encoding="utf-8")
'''


def v1_wanted(data: dict) -> dict[str, dict]:
    """Every run ews.ising.select needs, by name, with the flags its slot takes (README, note 13):
    one slow ramp and one staircase per (L, seed), one null run per (L, seed, eps_L). A ramp at
    another x fills no slot, so note 13's 90 runs, which hold the fast ramps, are not these."""
    start, end = data["ramps"]["eps_L"]
    record = {"input": "record"}
    slow = ["--x", data["ramps"]["x"], "--pilot", record, "--eps-start", start, "--eps-end", end]
    wanted = {}
    for L, seed in itertools.product(data["L"], data["seeds"]):
        base = ["--L", L, "--seed", seed, "--replicas", data["replicas"]]
        wanted[f"ramp-L{L}-s{seed}"] = {"command": "ramp", "args": [*base, *slow]}
        wanted[f"qs-L{L}-s{seed}"] = {
            "command": "staircase",
            "args": [*base, "--tau", {"input": f"tau_L{L}"}],
        }
        for eps in data["nulls"]["eps_L"]:
            wanted[f"null-L{L}-e{eps:g}-s{seed}"] = {
                "command": "null",
                "args": [*base, "--epsilon-L", eps, "--pilot", record],
            }
    return wanted


def v1_pins(data: dict) -> dict[str, dict]:
    """The inputs v1 pins, as a plan's `inputs` names them: the record and each dwell file."""
    pins = {
        "record": {
            "run": data["tau_c"]["pilot_run"],
            "file": "pilot.json",
            "sha256": data["tau_c"]["pilot_sha256"],
        }
    }
    for L in data["L"]:
        pins[f"tau_L{L}"] = {
            "run": V1_GRID_PILOTS[L],
            "file": "tau.json",
            "sha256": data["staircase"]["tau_sha256"][str(L)],
        }
    return pins


def v1_plan_problems(plan: dict, data: dict) -> dict[str, list[str]]:
    """A plan's runs and inputs against the frozen file's data section, by what is wrong and which
    run it is wrong in. The scorer alone would let some through: it refuses too few replicas, not
    too many, and takes any --saves or --pre."""
    wanted, runs = v1_wanted(data), {r["name"]: r for r in plan["runs"]}
    found: dict[str, list[str]] = {
        "missing": [name for name in wanted if name not in runs],
        "extra": [name for name in runs if name not in wanted],
        "seed": [],
        "replicas": [],
        "inputs": [],
        "flags": [],
    }
    kinds = {"--seed": "seed", "--replicas": "replicas", "--tau": "inputs", "--pilot": "inputs"}
    for name in wanted:
        if name not in runs:
            continue
        got, want = (r["args"] for r in (runs[name], wanted[name]))
        flags = {str(flag): value for flag, value in zip(got[::2], got[1::2], strict=False)}
        needed = dict(zip(want[::2], want[1::2], strict=True))
        for flag in {*flags, *needed}:
            if flags.get(flag) != needed.get(flag):
                said = f"{name}: {flag} {flags.get(flag)!r}, not {needed.get(flag)!r}"
                found[kinds.get(flag, "flags")].append(said)
        spec = {key: value for key, value in runs[name].items() if key not in ("name", "args")}
        if spec != {"command": wanted[name]["command"]}:  # another command, or a GPU pin
            found["flags"].append(f"{name}: {spec}, not {wanted[name]['command']}")
    pins, given = v1_pins(data), plan.get("inputs", {})
    for name in {*pins, *given}:
        if given.get(name) != pins.get(name):
            found["inputs"].append(f"input {name}: {given.get(name)}, not {pins.get(name)}")
    return found


def v1_tag_package(root: Path, commit: str) -> Path:
    """The folder holding the tag's package as the wheel built from it holds it, out of git, stamped
    as hatch_build.py stamps it: main's CLI may move on, and the wheel cannot."""
    args = ["git", "-C", str(PLANS.parents[1]), "archive", commit]
    args.append("python/tipping_point/src/tipping_point")
    raw = subprocess.run(args, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for member in tar.getmembers():
            member.name = member.name.removeprefix("python/tipping_point/src/")
            if member.name.startswith("tipping_point"):
                tar.extract(member, root, filter="data")
    stamp = f"commit = {commit!r}\ndirty = False\n"
    (root / "tipping_point" / "_build_info.py").write_text(stamp, encoding="utf-8")
    return root


def v1_on_stand_ins(plan: dict, stand_ins: dict[str, str]) -> dict:
    """The plan pinning each dwell file's stand-in where it pins the real one: the runner checks the
    real bytes on Kaggle. Any other pin stays as it is, and finds nothing on the mounts."""
    trial = copy.deepcopy(plan)
    for entry in trial.get("inputs", {}).values():
        entry["sha256"] = stand_ins.get(entry["sha256"], entry["sha256"])
    return trial


def v1_scorer_problems(
    k: ModuleType,
    plans: dict[str, dict],
    ctx: Any,
    stand_ins: dict[str, str],
    doc: dict,
    commit: str,
    root: Path,
) -> dict[str, list[str]]:
    """What the scorer says of the runs the tag's own CLI makes from each plan: the session's lookup
    of its inputs on the stand-in mounts, the tag's parser and schedules writing each manifest, then
    what `ews score` asks of them before it reads a frame (ising.admit): one complete run per slot,
    at the file's size, seed and replicas, by the tag's commit, against the pinned record and the
    pinned dwell. A run that fills no slot is named too: the scorer itself only leaves it out."""
    from tipping_point.ews import ising

    job, said = {}, {label: [] for label in plans}
    dirs = {label: root / str(n) for n, label in enumerate(plans)}
    for label, plan in plans.items():
        trial = v1_on_stand_ins(plan, stand_ins)
        session = k.Session(ctx, trial, {}, COMMIT, TORCH, 2)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                session.locate_inputs()
        except k.Refused as exc:
            said[label].append(f"the session refuses: {exc}")
            continue
        job[label] = {
            r.name: [
                k.module_of(r.command)[1],
                *session.resolve(r)[0],
                *session.flags(r, dirs[label]),
            ]
            for r in session.runs
        }
    root.mkdir(parents=True, exist_ok=True)
    script, given, out = root / "tag_cli.py", root / "job.json", root / "made.json"
    script.write_text(V1_TAG_CLI, encoding="utf-8")
    given.write_text(json.dumps(job), encoding="utf-8")
    package = v1_tag_package(root / "tag", commit)
    argv = [sys.executable, "-I", str(script), str(package), str(given), str(out)]
    done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert done.returncode == 0, done.stderr[-2000:]
    record = doc["data"]["tau_c"]["pilot_run"]
    for label, made in json.loads(out.read_text(encoding="utf-8")).items():
        ids = {}
        for name, got in made.items():
            if "error" in got:
                said[label].append(f"{name}: the tag's CLI refuses it: {got['error']}")
                continue
            ids[got["run_id"]] = name
            folder = dirs[label] / "ising" / got["run_id"]
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            schedule = manifest["params"]["schedule"]
            blocks = schedule["params"]["temperatures"] if schedule["kind"] == "fixed" else []
            checks = [{"temperature": T, "passed": True} for T in blocks]
            done_json = {"clean": True, "files": {}, "checks": {"thermalised": checks}}
            (folder / "complete.json").write_text(json.dumps(done_json), encoding="utf-8")
        (dirs[label] / "ising" / record).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(V1_RECORD, dirs[label] / "ising" / record / "pilot.json")
        try:
            admitted = ising.admit(doc, dirs[label], lambda _: None, commit=commit)
        except ising.IsingDataError as exc:
            said[label].append(f"the scorer refuses: {exc}")
            continue
        scored = {r.run_id for r in admitted.scored}
        said[label] += [f"{n} fills no slot" for run_id, n in ids.items() if run_id not in scored]
    return said


def v1_wrong_plans(canonical: dict) -> list[tuple[str, dict, str, str, str]]:
    """Each way a plan could go wrong: what it is, the plan so changed, where v1_plan_problems must
    file it and the name it must give, and the name the scorer's refusal must give, "" where the
    scorer would let it through."""

    def changed(*changes: tuple[str, str, Any]) -> dict:
        plan = copy.deepcopy(canonical)
        for name, flag, value in changes:
            args = next(r for r in plan["runs"] if r["name"] == name)["args"]
            args[args.index(flag) + 1] = value
        return plan

    missing = copy.deepcopy(canonical)
    missing["runs"] = [r for r in missing["runs"] if r["name"] != "ramp-L64-s9"]
    fast = copy.deepcopy(canonical)
    fast_run = copy.deepcopy(next(r for r in fast["runs"] if r["name"] == "ramp-L16-s6"))
    fast_run["name"] = "ramp-x2-L16-s6"
    fast_run["args"][fast_run["args"].index("--x") + 1] = 2
    fast["runs"].append(fast_run)
    doubled = copy.deepcopy(canonical)
    twin = copy.deepcopy(next(r for r in doubled["runs"] if r["name"] == "null-L32-e4-s7"))
    twin["name"] = "null-L32-e4-s7-again"
    doubled["runs"].append(twin)
    record_pin = copy.deepcopy(canonical)
    record_pin["inputs"]["record"]["sha256"] = "0" * 64
    tau_pin = copy.deepcopy(canonical)
    tau_pin["inputs"]["tau_L64"]["sha256"] = "0" * 64
    saves = copy.deepcopy(canonical)
    next(r for r in saves["runs"] if r["name"] == "qs-L16-s7")["args"] += ["--saves", 100]
    return [
        ("a missing run", missing, "missing", "ramp-L64-s9", "(64, 'ramp', 9)"),
        ("an extra run no slot takes", fast, "extra", "ramp-x2-L16-s6", "ramp-x2-L16-s6 fills no"),
        (
            "an extra run filling a slot twice",
            doubled,
            "extra",
            "null-L32-e4-s7-again",
            "more than",
        ),
        (
            "a wrong seed",
            changed(("qs-L32-s7", "--seed", 11)),
            "seed",
            "qs-L32-s7",
            "(32, 'staircase', 7)",
        ),
        (
            "too few replicas",
            changed(("null-L16-e4-s8", "--replicas", 16)),
            "replicas",
            "null-L16-e4-s8",
            "fewer than 64 replicas",
        ),
        (
            "too many replicas",
            changed(("ramp-L32-s6", "--replicas", 128)),
            "replicas",
            "ramp-L32-s6",
            "",
        ),
        (
            "another size's dwell file",
            changed(("qs-L16-s6", "--tau", {"input": "tau_L32"})),
            "inputs",
            "qs-L16-s6",
            "staircases whose dwell",
        ),
        (
            "a number for the dwell",
            changed(("qs-L64-s10", "--tau", 50)),
            "inputs",
            "qs-L64-s10",
            "tau.json sha256 None",
        ),
        (
            "a record pinned by another sha256",
            record_pin,
            "inputs",
            "input record",
            "input record:",
        ),
        (
            "a dwell file pinned by another sha256",
            tau_pin,
            "inputs",
            "input tau_L64",
            "input tau_L64:",
        ),
        (
            "a ramp over another window",
            changed(("ramp-L64-s6", "--eps-end", -4)),
            "flags",
            "ramp-L64-s6",
            "its ramp runs from",
        ),
        (
            "a ramp at another x",
            changed(("ramp-L16-s10", "--x", 2)),
            "flags",
            "ramp-L16-s10",
            "'ramp', 10",
        ),
        ("a flag the scorer lets through", saves, "flags", "qs-L16-s7", ""),
    ]


def v1_confirmatory(k: Any, root: Path) -> None:  # Any: it swaps the module's PLAN_JSON
    from tipping_point.ews import prereg
    from tipping_point.ising import protocol

    pre = prereg.load(V1_PREREG)
    tag = prereg.frozen(pre)  # what `ews score` asks first: the frozen file, byte for byte
    data = pre.doc["data"]
    plan = json.loads(V1_PLAN.read_text(encoding="utf-8"))
    why = refused(k, plan)
    check("v1-confirmatory: the plan loads", not why, why)
    if why:
        return
    errors = k.cli_errors(plan)
    check("v1-confirmatory: the CLI itself takes every run's arguments", errors == [], errors)
    check(
        "v1-confirmatory: its kernel is its own, its wheel the v1 wheel, never the project's other",
        plan["kernel"] == V1_KERNEL
        and plan.get("wheel") == V1_WHEEL
        and k.DEFAULT_WHEEL not in [*plan.get("datasets", []), *plan.get("kernel_sources", [])],
        [plan["kernel"], plan.get("wheel"), plan.get("datasets")],
    )
    check(
        "v1-confirmatory: `commit` is the commit the freeze tag points at",
        plan.get("commit") == tag.commit,
        plan.get("commit"),
    )
    check(
        "v1-confirmatory: the record and the dwell files come from the kernels that wrote them",
        plan.get("kernel_sources") == V1_SOURCES and not plan.get("datasets"),
        plan.get("kernel_sources"),
    )
    wanted = v1_wanted(data)
    count = len(data["L"]) * len(data["seeds"]) * (2 + len(data["nulls"]["eps_L"]))
    found = v1_plan_problems(plan, data)
    check(
        f"v1-confirmatory: the scorer's selection needs {count} runs, per (L, seed) a slow ramp, a "
        "staircase and a null run per eps_L",
        len(wanted) == count == 75,
        len(wanted),
    )
    for kind, what in (
        ("missing", "no run the selection needs is missing"),
        ("extra", "no run beyond what the selection needs"),
        ("seed", "every run has its slot's seed"),
        ("replicas", f"every run has data.replicas, {data['replicas']}"),
        ("inputs", "every input is the file's pin, and each run reads its own"),
        ("flags", "every run has exactly its slot's flags, and no GPU pin"),
    ):
        check(f"v1-confirmatory: {what}", not found[kind], found[kind])

    # on the mounts Kaggle gives it: the record's bytes are the real ones, the repository's copy;
    # the dwell files sit on Kaggle alone, so each gets a stand-in, and the file's pin of it follows
    ctx = ctx_at(k, root / "session")
    ctx.py = sys.executable
    ctx.helper.write_text(k.HELPER_SOURCE, encoding="utf-8")
    holder = ctx.input / "tp-tauc-l64-l128" / "out" / "ising" / data["tau_c"]["pilot_run"]
    holder.mkdir(parents=True)
    shutil.copyfile(V1_RECORD, holder / "pilot.json")
    doc = copy.deepcopy(pre.doc)
    stand_ins, tau_files = {}, {}
    for L in data["L"]:
        points = [{"temperature": T, "tau": 60.0, "reliable": True} for T in protocol.GRID]
        text = json.dumps({"L": L, "seconds": 1.0, "points": points}, indent=2)
        folder = ctx.input / "tp-ising-job-a" / "out" / "ising" / V1_GRID_PILOTS[L]
        folder.mkdir(parents=True)
        tau_files[L] = folder / "tau.json"
        tau_files[L].write_text(text, encoding="utf-8", newline="\n")
        sha = hashlib.sha256(text.encode()).hexdigest()
        stand_ins[data["staircase"]["tau_sha256"][str(L)]] = sha
        doc["data"]["staircase"]["tau_sha256"][str(L)] = sha

    canonical = {
        "kernel": V1_KERNEL,
        "wheel": V1_WHEEL,
        "commit": tag.commit,
        "kernel_sources": V1_SOURCES,
        "inputs": v1_pins(data),
        "runs": [{"name": name, **run} for name, run in wanted.items()],
    }
    wrong = v1_wrong_plans(canonical)
    plans = {
        "the plan": plan,
        "the canonical plan": canonical,
        **{w[0]: w[1] for w in wrong if w[4]},
    }
    said = v1_scorer_problems(k, plans, ctx, stand_ins, doc, tag.commit, root / "made")
    check(
        f"v1-confirmatory: made by the tag's own CLI, `ews score` admits all {count} and scores "
        "every one, against the tag's commit, the pinned record and the pinned dwell",
        not said["the plan"],
        said["the plan"],
    )
    check(
        "... as it does the plan worked out from the frozen file, so these refusals are the plan's",
        not said["the canonical plan"],
        said["the canonical plan"],
    )
    for what, wrong_plan, kind, named, says in wrong:
        problems = v1_plan_problems(wrong_plan, data)
        check(
            f"... {what} goes red under {kind}, naming {named}",
            not refused(k, wrong_plan) and any(named in p for p in problems[kind]),
            problems,
        )
        if says:
            check(
                f"... {what}: the scorer's refusal names {says}",
                any(says in p for p in said[what]),
                said[what],
            )
    if any(problem.startswith("the session refuses") for problem in said["the plan"]):
        return  # red above, by name, and what follows needs its inputs found
    made = sorted((root / "made" / "0" / "ising").glob("*/manifest.json"))
    keys = {json.loads(m.read_text("utf-8"))["params"]["rng"]["seed"] for m in made}
    check("v1-confirmatory: every run draws its own stream", len(keys) == len(made), len(keys))

    trial = v1_on_stand_ins(plan, stand_ins)
    session = k.Session(ctx, trial, {}, COMMIT, TORCH, 2)
    session.locate_inputs()
    errors = session.capture(session.runs)
    check(
        "v1-confirmatory: the helper's capture, as on Kaggle, refuses nothing", errors == [], errors
    )
    check(
        f"v1-confirmatory: each run is one block of {data['replicas']} lattices at a size of v1",
        {tuple(r.shape or ()) for r in session.runs} == {(L, data["replicas"]) for L in data["L"]},
        sorted({tuple(r.shape or ()) for r in session.runs}),
    )
    ramp = next(r for r in session.runs if {"input": "record"} in r.args)
    tail, key = session.resolve(ramp)
    check(
        "v1-confirmatory: its key names the record by run and file, not by where it was mounted, "
        "so a resumed session reuses and continues the runs a session made",
        tail[tail.index("--pilot") + 1] == str(holder / "pilot.json")
        and f"input/{data['tau_c']['pilot_run']}/pilot.json" in key
        and str(ctx.input) not in " ".join(key),
        key,
    )
    resaved = json.dumps(json.loads(tau_files[32].read_text(encoding="utf-8")))
    tau_files[32].write_text(resaved, encoding="utf-8")
    again = k.Session(ctx, trial, {}, COMMIT, TORCH, 2)
    try:
        again.locate_inputs()
        why = ""
    except k.Refused as exc:
        why = str(exc)
    check(
        "v1-confirmatory: a re-serialised tau.json is refused, by name",
        "input tau_L32:" in why,
        why,
    )

    state = {"kernel": "tp-ising-job-a", "runs": [{"name": "qs-L16-s6", "folder": "ising/x"}]}
    (ctx.input / "tp-ising-job-a" / "out" / k.STATE).write_text(json.dumps(state), encoding="utf-8")
    check(
        "v1-confirmatory: Job A's own state, attached with its output, is no previous session",
        k.carry_forward(ctx, trial) is None,
    )
    folder = root / "resumed"
    code = k.main(["--write-kernel", str(V1_PLAN), str(folder), "--resume"])
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    check(
        "v1-confirmatory --resume: the v1 wheel alone, its two sources, then its own output last",
        code == 0
        and not push_problems(folder)
        and meta["dataset_sources"] == [V1_WHEEL]
        and meta["kernel_sources"] == [*V1_SOURCES, f"{k.OWNER}/{V1_KERNEL}"],
        meta,
    )
    check(
        "v1-confirmatory: the description holds the one line that resumes it",
        V1_RESUME in plan.get("description", ""),
        plan.get("description"),
    )

    # the plan's commit, read by the kernel before any quota goes on a wheel the scorer would refuse
    held = ctx.input / "tipping-point-wheel-v1"
    held.mkdir()
    (held / "tipping_point-0.1.0-py3-none-any.whl").write_bytes(b"PK")

    def session_says(commit: str) -> str:
        (held / "COMMIT.txt").write_text(f"{commit}\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            k.kernel_main(ctx)
        return (ctx.out / "STATUS").read_text(encoding="utf-8")

    saved, k.PLAN_JSON = k.PLAN_JSON, json.dumps(trial)
    try:
        other = session_says(OTHER)
        same = session_says(tag.commit)
    finally:
        k.PLAN_JSON = saved
    check(
        "v1-confirmatory: a wheel from any other commit is refused before the environment is built",
        other.startswith(f"refused: COMMIT.txt names {OTHER}; this plan's runs need {tag.commit}"),
        other,
    )
    check(
        "... and the tag's commit passes that line, on to the installed package's own stamp",
        same.startswith("refused: the installed package names commit"),
        same,
    )
    for bad in ("main", 7, tag.commit[:12]):
        why = refused(k, dict(copy.deepcopy(plan), commit=bad))
        check(f"refused: a plan's commit of {bad!r}", "40 lowercase hex digits" in why, why)


ABLATIONS_PLAN = PLANS / "ablations-28.json"
ABLATIONS_KERNEL = "tp-jepa-ablations-28"
# e598e62, the commit that trained primary-28 and that tipping-point-wheel holds. Written out here
# as well as in the plan, so that a slip in the plan's pin turns red here and is not first refused
# on Kaggle.
ABLATIONS_COMMIT = "e598e627998cb940458be3c26a02d1c4a3ea7f08"
# #28's research pass, section 4 "Runs", as #28 accepted it: each role of the draft's ablation
# checkpoints, its model seeds, and the flags it adds to a primary's training
ABLATIONS = {
    "ema-only": ((11, 12, 13), ["--var-weight", 0, "--cov-weight", 0]),
    "held-out": ((21, 22, 23), ["--held-out"]),
}
HELD_OUT_BAND = [2.0, 2.5]  # the research pass's band, written out: a slip in data.HELD_OUT is red
ABLATIONS_RESUME = (
    "uv run --no-sync python notebooks/kaggle_ising_runs.py --write-kernel "
    "notebooks/plans/ablations-28.json ../.wf/kaggle/kernel-ablations-28-resume --resume; "
    "kaggle kernels push -p ../.wf/kaggle/kernel-ablations-28-resume"
)
ABLATIONS_WHEEL_CLI = r'''
"""Each training's command through a copy of the wheel commit's package: its own parser and
_train, with data.load and train.start swapped for stubs that note what they are handed and stop.
The commit's own lookup finds the staged runs; nothing is loaded and nothing is trained.
Usage: <this> <folder holding tipping_point> <job.json> <out.json>"""
import contextlib
import io as text_io
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import tipping_point
from tipping_point.jepa import __main__ as cli
from tipping_point.jepa import train

assert Path(tipping_point.__file__).is_relative_to(sys.argv[1]), tipping_point.__file__
seen = {}


def load(data_dir, train_runs, val_runs, held_out=None, refused=frozenset(), expect=None):
    seen.update(train=list(train_runs), val=list(val_runs), held_out=held_out)
    seen.update(refused=sorted(refused))


def start(config, data, root, tag, device=None, pilot=None, prereg=None):
    seen.update(config=config.to_json(), tag=tag, pilot=pilot, prereg=prereg)
    return tag


cli.load, train.start = load, start
results = {}
for label, runs in json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")).items():
    results[label] = {}
    for name, argv in runs.items():
        seen.clear()
        said = text_io.StringIO()
        try:
            with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
                code = cli.main(argv)
        except SystemExit as exc:  # argparse's usage error
            code = exc.code
        why = said.getvalue().strip()[-300:] or f"exit {code}"
        results[label][name] = dict(seen) if code == 0 else {"error": why}
Path(sys.argv[3]).write_text(json.dumps(results), encoding="utf-8")
'''


def ablation_name(role: str, seed: int) -> str:
    return f"L16-32-64-{role}-m{seed}"


def ablations_canonical(primary: dict) -> dict:
    """The plan #28 asks for, worked out from primary-28.json: its m1 for each ablation, with the
    model seed and the flags the role adds and without its GPU pin, so that a T4 takes the next
    training as it comes free; primary-28's wheel, sources and inputs; and `commit` e598e62."""
    base = primary["runs"][0]
    runs = []
    for role, (seeds, extra) in ABLATIONS.items():
        for seed in seeds:
            args = copy.deepcopy(base["args"])
            args[args.index("--seed") + 1] = seed
            runs.append(
                {
                    "name": ablation_name(role, seed),
                    "command": base["command"],
                    "stage": list(base["stage"]),
                    "args": [*args, *extra],
                }
            )
    shared = ("wheel", "datasets", "kernel_sources", "inputs")
    return {
        "kernel": ABLATIONS_KERNEL,
        **{key: copy.deepcopy(primary[key]) for key in shared},
        "commit": ABLATIONS_COMMIT,
        "runs": runs,
    }


def flag_groups(args: list) -> list[tuple[str, list]]:
    """A run's arguments as argparse groups them: each flag with the values up to the next flag."""
    groups: list[tuple[str, list]] = []
    for arg in args:
        if isinstance(arg, str) and arg.startswith("--"):
            groups.append((arg, []))
        elif groups:
            groups[-1][1].append(arg)
        else:
            groups.append(("(before any flag)", [arg]))
    return groups


def ablations_problems(plan: dict, canonical: dict) -> dict[str, list[str]]:
    """A plan against the one #28 asks for, by what is wrong, each problem naming its run, input or
    key. Flags are compared as argparse reads them: their order is free, but a flag given twice is
    named."""
    runs, wanted = ({r["name"]: r for r in p["runs"]} for p in (plan, canonical))
    found: dict[str, list[str]] = {
        "missing": [name for name in wanted if name not in runs],
        "extra": [name for name in runs if name not in wanted],
        **{kind: [] for kind in ("seed", "steps", "flags", "stage", "gpus", "inputs", "plan")},
    }
    kinds = {"--seed": "seed", "--steps": "steps"}
    for name in (name for name in wanted if name in runs):
        got, want = runs[name], wanted[name]
        given = flag_groups(got.get("args", []))
        flags, needed = dict(given), dict(flag_groups(want["args"]))
        if len(flags) != len(given):
            found["flags"].append(f"{name}: a flag given twice")
        for flag in sorted({*flags, *needed}):
            if flags.get(flag) != needed.get(flag):
                said = f"{name}: {flag} {flags.get(flag)!r}, not {needed.get(flag)!r}"
                found[kinds.get(flag, "flags")].append(said)
        if got.get("stage") != want["stage"]:
            found["stage"].append(f"{name}: its stage ids are not primary-28's")
        if set(got) - {"gpu"} != set(want) or got["command"] != want["command"]:
            found["flags"].append(f"{name}: keys {sorted(got)}, command {got.get('command')!r}")
    # no run is pinned, so a T4 takes the next training as it comes free; a null gpu, as the runner
    # reads it, pins nothing
    found["gpus"] += [
        f"{r.get('name')}: pinned to gpu {r['gpu']!r}"
        for r in plan["runs"]
        if r.get("gpu") is not None
    ]
    pins, given_inputs = canonical["inputs"], plan.get("inputs", {})
    for name in sorted({*pins, *given_inputs}):
        if given_inputs.get(name) != pins.get(name):
            found["inputs"].append(f"input {name}: {given_inputs.get(name)}, not {pins.get(name)}")
    for key in ("kernel", "wheel", "commit", "datasets", "kernel_sources"):
        if plan.get(key) != canonical[key]:
            found["plan"].append(f"{key}: {plan.get(key)!r}, not {canonical[key]!r}")
    return found


def ablations_wrong_plans(canonical: dict) -> list[tuple[str, dict, str, str, str]]:
    """Each planted fault: what it is, the plan so changed, where ablations_problems must file it
    and the name it must give, and what the session or e598e62's CLI must then say of it on the
    mounts, "" where neither can tell: this section alone stands between such a plan and Kaggle."""

    def changed(name: str, change: Callable[[dict], object]) -> dict:
        plan = copy.deepcopy(canonical)
        change(next(r for r in plan["runs"] if r["name"] == name))
        return plan

    def drop(flag: str, values: int) -> Callable[[dict], None]:
        def change(run: dict) -> None:
            at = run["args"].index(flag)
            del run["args"][at : at + 1 + values]

        return change

    def value(flag: str, new: int) -> Callable[[dict], None]:
        def change(run: dict) -> None:
            run["args"][run["args"].index(flag) + 1] = new

        return change

    def seed_seven(run: dict) -> None:
        run["stage"][0] = "20261102T000000Z-L16-qs-s7"

    ema, held = ablation_name("ema-only", 12), ablation_name("held-out", 22)
    seed, steps = ablation_name("ema-only", 13), ablation_name("held-out", 21)
    pinned, stage = ablation_name("held-out", 23), ablation_name("ema-only", 11)
    sha = copy.deepcopy(canonical)
    sha["inputs"]["prereg"]["sha256"] = "0" * 64
    balanced = copy.deepcopy(canonical)  # the first version's pins: 0, 1, 0, ... in plan order
    for i, run in enumerate(balanced["runs"]):
        run["gpu"] = i % 2
    return [
        (
            "a missing --var-weight 0 on an EMA-only run",
            changed(ema, drop("--var-weight", 1)),
            "flags",
            ema,
            f"{ema}: var_weight 0.025, not 0.0",
        ),
        (
            "a missing --cov-weight 0 on an EMA-only run",
            changed(ema, drop("--cov-weight", 1)),
            "flags",
            ema,
            f"{ema}: cov_weight 0.001, not 0.0",
        ),
        (
            "a missing --held-out on a held-out run",
            changed(held, drop("--held-out", 0)),
            "flags",
            held,
            f"{held}: held_out None, not [2.0, 2.5]",
        ),
        (
            "a wrong model seed",
            changed(seed, value("--seed", 3)),
            "seed",
            seed,
            f"{seed}: seed 3, not 13",
        ),
        (
            "wrong steps",
            changed(steps, value("--steps", 50000)),
            "steps",
            steps,
            f"{steps}: steps 50000, not 25000",
        ),
        ("a wrong prereg sha", sha, "inputs", "input prereg", "the session refuses: input prereg:"),
        (
            "a run pinned to a GPU",
            changed(pinned, lambda run: run.update(gpu=0)),
            "gpus",
            pinned,
            "",
        ),
        ("every run pinned, three on each T4", balanced, "gpus", stage, ""),
        ("a staircase of seed 7 staged", changed(stage, seed_seven), "stage", stage, ""),
    ]


def ablations_commands(k: ModuleType, ctx: Any, plan: dict) -> dict[str, list[str]] | str:
    """Each training's command as the runner gives it on the mounts, its inputs found and checked
    against their sha256 and the runner's own flags added; or the session's refusal."""
    session = k.Session(ctx, plan, {}, COMMIT, TORCH, 2)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            session.locate_inputs()
    except k.Refused as exc:
        return f"the session refuses: {exc}"
    return {
        r.name: [k.module_of(r.command)[1], *session.resolve(r)[0], *session.flags(r, ctx.out)]
        for r in session.runs
    }


def ablations_made(job: dict[str, dict[str, list[str]]], root: Path) -> dict[str, dict] | str:
    """What e598e62's own CLI makes of each command (ABLATIONS_WHEEL_CLI), by label and run; or why
    it did not run."""
    root.mkdir(parents=True, exist_ok=True)
    script, given, out = root / "wheel_cli.py", root / "job.json", root / "made.json"
    script.write_text(ABLATIONS_WHEEL_CLI, encoding="utf-8")
    given.write_text(json.dumps(job), encoding="utf-8")
    package = v1_tag_package(root / "wheel", ABLATIONS_COMMIT)
    argv = [sys.executable, "-I", str(script), str(package), str(given), str(out)]
    done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        return done.stderr[-2000:] or f"exit {done.returncode}"
    return json.loads(out.read_text(encoding="utf-8"))


def ablations_expected(primary: dict) -> dict[str, dict]:
    """What e598e62's CLI must make of each ablation: what it makes of primary-28's m1, but for the
    model seed and the role's change, the VICReg weights at 0 or the band held out (and the CLI's
    -heldout tag)."""
    expected = {}
    for role, (seeds, _) in ABLATIONS.items():
        for seed in seeds:
            config = dict(primary["config"], seed=seed)
            if role == "ema-only":
                config.update(var_weight=0.0, cov_weight=0.0)
            held = role == "held-out"
            expected[ablation_name(role, seed)] = dict(
                primary,
                config=config,
                tag=f"L16-32-64-m{seed}" + ("-heldout" if held else ""),
                held_out=HELD_OUT_BAND if held else None,
            )
    return expected


def ablations_made_problems(made: dict[str, dict], expected: dict[str, dict]) -> list[str]:
    """Each difference between what e598e62's CLI made of a training and what #28 asks of it, naming
    the run and the field, a config field by its own name."""
    problems = [f"{name}: made, though #28 asks for none" for name in made if name not in expected]
    for name, want in expected.items():
        got = made.get(name, {"error": "the plan has no such training"})
        if "error" in got:
            problems.append(f"{name}: the wheel's CLI refuses it: {got['error']}")
            continue
        config = got.get("config", {})
        pairs = [(f, config.get(f), want["config"].get(f)) for f in {*config, *want["config"]}]
        pairs += [(key, got.get(key), want.get(key)) for key in {*got, *want} - {"config"}]
        problems += [
            f"{name}: {field} {value!r}, not {needed!r}"
            for field, value, needed in sorted(pairs, key=lambda pair: pair[0])
            if value != needed
        ]
    return problems


def ablations_28(k: Any, root: Path) -> None:  # Any: it swaps the module's PLAN_JSON
    draft_path = PLANS.parents[1] / "paper" / "prereg" / DRAFT
    plan = json.loads(ABLATIONS_PLAN.read_text(encoding="utf-8"))
    primary = json.loads((PLANS / "primary-28.json").read_text(encoding="utf-8"))
    why = refused(k, plan)
    check("ablations-28: the plan loads", not why, why)
    if why:
        return
    errors = k.cli_errors(plan)
    check("ablations-28: the CLI itself takes every run's arguments", errors == [], errors)
    roles: dict[str, list[int]] = collections.defaultdict(list)
    for entry in tomllib.loads(draft_path.read_text(encoding="utf-8"))["latents"]["checkpoints"]:
        roles[entry["role"]].append(entry["model_seed"])
    roles.pop("primary", None)
    check(
        "ablations-28: its roles and model seeds are the draft's ablation checkpoints",
        roles == {role: list(seeds) for role, (seeds, _) in ABLATIONS.items()},
        dict(roles),
    )
    canonical = ablations_canonical(primary)
    found = ablations_problems(plan, canonical)
    for kind, what in (
        ("missing", "no training #28 asks for is missing"),
        ("extra", "no training beyond the six"),
        ("seed", "each training has its role's model seed"),
        ("steps", "each training runs primary-28's 25,000 steps"),
        (
            "flags",
            "each training has primary-28's flags and exactly its role's: --var-weight 0 "
            "--cov-weight 0 for EMA-only, --held-out for held-out",
        ),
        ("stage", "each training stages primary-28's 18 staircases"),
        ("gpus", "no training is pinned to a GPU, so a T4 takes the next as it comes free"),
        ("inputs", "the record and the draft are primary-28's pins"),
        ("plan", "its kernel is its own, primary-28's wheel and sources, and `commit` e598e62"),
    ):
        check(f"ablations-28: {what}", not found[kind], found[kind])
    staged = sorted({s for r in plan["runs"] for s in r["stage"]})
    tags = [DATASET_ID.fullmatch(s) for s in staged]
    check(
        "ablations-28: it stages the 18 staircases of seeds 11-16 at L = 16, 32 and 64, so none "
        "of seeds 6-10",
        all(t is not None and t[2] == "qs" for t in tags)
        and sorted((int(t[1]), int(t[3])) for t in tags if t)
        == sorted((L, s) for L in (16, 32, 64) for s in range(11, 17)),
        staged,
    )
    check(
        "ablations-28: its --prereg is the draft primary-28 trained under, pinned by sha256 as "
        "primary-28 pins it",
        plan["inputs"]["prereg"] == primary["inputs"]["prereg"]
        and plan["inputs"]["prereg"]["sha256"] == DRAFT_SHA256,
        plan["inputs"]["prereg"],
    )
    text = plan.get("description", "")
    check(
        "ablations-28: the description holds the one line that resumes it",
        ABLATIONS_RESUME in text,
        text,
    )
    check(
        "ablations-28: the description says nothing is read from seeds 6-10, and names e598e62",
        "Nothing is read from seeds 6-10" in text and "e598e62" in text,
        text,
    )

    # on the mounts Kaggle gives it: the record's and the draft's bytes are the real ones, the
    # repository's record and the draft as e598e62 holds it, the bytes the trainings read, so the
    # plan's own pins are what is checked; the staircases are stand-ins
    ctx = ctx_at(k, root / "session")
    ctx.py = sys.executable
    ctx.helper.write_text(k.HELPER_SOURCE, encoding="utf-8")
    holder = ctx.input / "tp-tauc-l64-l128" / "out" / "ising" / PILOT_RECORD
    holder.mkdir(parents=True)
    shutil.copyfile(V1_RECORD, holder / "pilot.json")
    held = ctx.input / "tp-prereg-ising-latent" / DRAFT_RUN
    held.mkdir(parents=True)
    show = ["git", "-C", str(PLANS.parents[1]), "show", f"{ABLATIONS_COMMIT}:paper/prereg/{DRAFT}"]
    (held / DRAFT).write_bytes(subprocess.run(show, capture_output=True, check=True).stdout)
    job_output = ctx.input / "tp-ising-job-a" / "out" / "ising"
    dataset = ctx.input / "tipping-point-ising-l16-l32"
    for run_id in staged:
        stand_in_run((job_output if "-L64-" in run_id else dataset) / run_id, run_id)
    session = k.Session(ctx, plan, {}, COMMIT, TORCH, 2)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            session.locate_inputs()
            session.locate_datasets()
        why = ""
    except k.Refused as exc:
        why = str(exc)
    check(
        "ablations-28: on the mounts, the record and the draft are found by the plan's own sha256, "
        "and the 18 staircases among the sources",
        not why
        and set(session.inputs) == {"record", "prereg"}
        and sorted(session.datasets) == staged,
        why or sorted(session.datasets),
    )
    if why:
        return
    with contextlib.redirect_stdout(io.StringIO()):
        errors = session.capture(session.runs)
        session.stage()
    check("ablations-28: the helper's capture, as on Kaggle, refuses nothing", errors == [], errors)

    # every training's command as the runner gives it, through e598e62's own CLI
    wrong = ablations_wrong_plans(canonical)
    plans = {
        "the plan": plan,
        "the canonical plan": canonical,
        **{w[0]: w[1] for w in wrong if w[4]},
    }
    m1 = ablations_commands(k, ctx, primary)
    job = {"primary-28": {"m1": m1["m1"]} if isinstance(m1, dict) else {}}
    said: dict[str, list[str]] = {label: [] for label in plans}
    for label, each in plans.items():
        commands = ablations_commands(k, ctx, each)
        if isinstance(commands, str):
            said[label].append(commands)
        else:
            job[label] = commands
    made = ablations_made(job, root / "made")
    if isinstance(made, str):
        check("ablations-28: e598e62's own CLI takes the commands", False, made)
        return
    base = made["primary-28"].get("m1", {})
    config, val = base.get("config", {}), base.get("val", [])
    check(
        "ablations-28: e598e62's own CLI makes primary-28's m1 as it trained: tag L16-32-64-m1, "
        "25,000 steps, warm-up 1,250, VICReg at 0.025 and 0.001, no band held out, the draft read "
        "and seeds 6-10 refused, staircases 11-15 to train on and 16 to validate on",
        base.get("tag") == "L16-32-64-m1"
        and (config.get("seed"), config.get("steps"), config.get("warmup")) == (1, 25000, 1250)
        and (config.get("var_weight"), config.get("cov_weight")) == (0.025, 0.001)
        and "held_out" in base
        and base["held_out"] is None
        and base.get("prereg")
        == {
            "id": "ising-latent",
            "version": 1,
            "sha256": DRAFT_SHA256,
            "seeds": [*CONFIRMATORY_SEEDS],
        }
        and set(CONFIRMATORY_SEEDS) <= set(base.get("refused", []))
        and (base.get("pilot") or {}).get("sha256") == PILOT_SHA256
        and sorted([*base.get("train", []), *val]) == staged
        and len(val) == 3
        and all(v.endswith("-qs-s16") for v in val),
        base,
    )
    if "config" not in base:
        return
    expected = ablations_expected(base)
    for label in plans:
        if label in made:
            said[label] += ablations_made_problems(made[label], expected)
    check(
        "ablations-28: made by e598e62's own CLI, each training is primary-28's m1 but for its "
        "model seed and its role's change, on the same staircases, record and draft",
        not said["the plan"],
        said["the plan"],
    )
    check(
        "... as is the plan worked out from primary-28.json, so these refusals are the plan's",
        not said["the canonical plan"],
        said["the canonical plan"],
    )
    for what, wrong_plan, kind, named, says in wrong:
        problems = ablations_problems(wrong_plan, canonical)
        check(
            f"... {what} goes red under {kind}, naming {named}",
            not refused(k, wrong_plan) and any(named in p for p in problems[kind]),
            problems,
        )
        if says:
            check(
                f"... {what}: on the mounts, {says}",
                any(says in p for p in said[what]),
                said[what],
            )

    folder = root / "resumed"
    code = k.main(["--write-kernel", str(ABLATIONS_PLAN), str(folder), "--resume"])
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    check(
        "ablations-28 --resume: primary-28's wheel and datasets, its two sources, then its own "
        "output last",
        code == 0
        and not push_problems(folder)
        and meta["dataset_sources"] == [primary["wheel"], *primary["datasets"]]
        and meta["kernel_sources"] == [*primary["kernel_sources"], f"{k.OWNER}/{ABLATIONS_KERNEL}"],
        meta,
    )

    # the plan's commit, read by the kernel before any quota goes on another commit's wheel
    wheels = ctx.input / k.WHEEL_DATASET
    wheels.mkdir()
    (wheels / "tipping_point-0.1.0-py3-none-any.whl").write_bytes(b"PK")

    def session_says(commit: str) -> str:
        (wheels / "COMMIT.txt").write_text(f"{commit}\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            k.kernel_main(ctx)
        return (ctx.out / "STATUS").read_text(encoding="utf-8")

    saved, k.PLAN_JSON = k.PLAN_JSON, json.dumps(plan)
    try:
        other = session_says(OTHER)
        same = session_says(ABLATIONS_COMMIT)
    finally:
        k.PLAN_JSON = saved
    check(
        "ablations-28: a wheel from any other commit is refused before the environment is built",
        other.startswith(
            f"refused: COMMIT.txt names {OTHER}; this plan's runs need {ABLATIONS_COMMIT}"
        ),
        other,
    )
    check(
        "... and e598e62 passes that line, on to the installed package's own stamp",
        same.startswith("refused: the installed package names commit"),
        same,
    )


def run_all(k: ModuleType) -> int:
    with tempfile.TemporaryDirectory(prefix="tp-self-test-", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        plans(k)
        refusals(k)
        cli(k)
        shrinking(k)
        queues(k)
        failed_extractions(k)
        resolution(k, root / "resolve")
        previous(k, root / "previous")
        inputs(k, root / "inputs")
        input_checkpoints(k, root / "input-checkpoints")
        wheel(k, root / "wheel")
        verification(k, root / "verify")
        kernel_folder(k, root / "kernel")
        jepa_entries(k, root / "jepa")
        extraction_rates(k, root / "rates")
        kernel_acceptance(k, root / "acceptance")
        invalid_plans(k, root / "invalid")
        pilot_28(k, root / "pilot-28")
        primary_28(k, root / "primary-28")
        first_reads(k, root / "first-reads")
        v1_confirmatory(k, root / "v1-confirmatory")
        ablations_28(k, root / "ablations-28")
        real_runs(k, root / "real")
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    import kaggle_ising_runs

    sys.exit(run_all(kaggle_ising_runs))
