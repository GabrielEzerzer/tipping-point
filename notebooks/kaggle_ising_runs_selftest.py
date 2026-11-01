"""Checks of notebooks/kaggle_ising_runs.py, run by its --self-test or directly:

    uv run --no-sync python notebooks/kaggle_ising_runs.py --self-test
    uv run --no-sync python notebooks/kaggle_ising_runs_selftest.py

Plan validation, the files a run reads from another and the CLI's verdict on every run's arguments;
dependency resolution; the queues (longest first, dependencies, pinning, a failure, a failed
extraction, the deadline, the output cap) with a stand-in for the simulator; the previous session's
runs carried forward and reused or resumed; inputs found and checked against their sha256; the
wheel and COMMIT.txt checks; run verification, judged by exit codes and files only; the rates STATUS
gives, by checkpoint and L; the kernel folder against what `kaggle kernels push` checks; invalid
plans refused at every door; pilot-28.json and primary-28.json, their stage ids and kernel sources,
on stand-in mounts.

Last, the whole path on real children: the written kernel runs two local sessions of a tiny plan
(a simulator run, a real `jepa train` stopped by a simulated deadline, `jepa resume`, then a real
`jepa extract` of its final.pt and of its twin, each run checked with `jepa verify`), then a real
extraction fails with another behind it on its queue, all in a venv that prints Kaggle's
sitecustomize errors on stderr from every child. No wheel is built: the venv holds a copy of the
package stamped clean, so the outcome does not depend on this checkout's state.

About five minutes on a laptop CPU. One line per check, then exit 1 if any failed. The gate's
self-test leg (harness/gate.py) runs this and reads its closing "N/M checks passed" and exit code.
"""

from __future__ import annotations

import ast
import collections
import copy
import hashlib
import inspect
import itertools
import json
import os
import re
import runpy
import shutil
import site
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import zipfile
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
    """Each file a run reads from an earlier run that the earlier run's command never writes.
    load_plan takes any path inside the folder, so such a run first fails on Kaggle as it starts."""
    writes = {"pilot": {"tau.json"}, "jepa train": {k.FINAL, k.TWIN}}
    written = {r["name"]: writes.get(r["command"], ()) for r in plan["runs"]}
    return [
        f"{r['name']} reads {arg['file']} of {arg.get('run')}"
        for r in plan["runs"]
        for arg in r["args"]
        if isinstance(arg, dict)
        and "file" in arg
        and arg["file"] not in written.get(arg.get("run"), ())
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

    def write(producer: dict | None = None, clean: bool = True, passed: bool = True) -> list[str]:
        manifest = {"producer": producer or {"git_sha": COMMIT, "dirty": False}}
        done = {
            "clean": clean,
            "files": {f"stats/{chunk.name}": hashlib.sha256(b"rows").hexdigest()},
            "checks": {"thermalised": [dict(thermalised, passed=passed)]},
        }
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (folder / "complete.json").write_text(json.dumps(done), encoding="utf-8")
        return k.verdicts(folder, COMMIT)

    check("a clean run of the wheel's commit passes", write() == [])
    check(
        "another commit fails",
        "is not the wheel's" in str(write({"git_sha": OTHER, "dirty": False})),
    )
    check(
        "a dirty manifest fails", "dirty is True" in str(write({"git_sha": COMMIT, "dirty": True}))
    )
    check("complete.json not clean fails", "clean is False" in str(write(clean=False)))
    problems = write(passed=False)
    check(
        "a failed check fails, with its temperature",
        "check thermalised failed at T = 2.2179: flat" in str(problems),
        problems,
    )
    (folder / "complete.json").unlink()
    check(
        "no complete.json fails",
        k.verdicts(folder, COMMIT) == ["complete.json is missing or unreadable"],
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


def jepa_entries(k: ModuleType, root: Path) -> None:
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

    seven = {"dataset_run": "20261101T000000Z-L16-null-e2-s7"}
    check("refused: jepa extract on a seed 7 dataset run", "seed 7" in extract_on(seven))
    untagged = {"dataset_run": "20261101T000000Z-train"}
    check(
        "refused: jepa extract on a run whose seed it cannot read",
        "no seed" in extract_on(untagged),
    )
    made8 = [{"name": "sim8", "command": "null", "args": ["--L", 16, "--seed", 8]}]
    plan = copy.deepcopy(JEPA_PLAN)
    plan["runs"].insert(1, made8[0])
    plan["runs"][2]["args"][1] = {"run": "sim8"}
    check("refused: jepa extract on this plan's own seed 8 run", "seed 8" in refused(k, plan))
    plan["runs"][1]["args"] = ["--L", 16]  # no --seed: the CLI's seed 1
    check("... and accepted at the CLI's default seed 1", not refused(k, plan), refused(k, plan))
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
    session.refuse_seeds(final)
    check("seed 1 may be extracted", True)
    seven = ctx.out / "ising" / NULL / "manifest.json"
    seven.write_text(json.dumps({"params": {"seed": 7}}), encoding="utf-8")
    try:
        session.refuse_seeds(final)
        check("seed 7 is refused again from the run's own manifest", False)
    except k.RunError as exc:
        check("seed 7 is refused again from the run's own manifest", "seed 7" in str(exc), exc)
    seven.write_text(json.dumps({"params": {"seed": 1}}), encoding="utf-8")

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
    for name in ("job-a", "pilot-28", "primary-28"):
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
    bad["runs"][1]["args"][1] = {"dataset_run": "20261101T000000Z-L16-null-e2-s7"}
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
            code == 1 and said.startswith("refused:") and "seed 7" in said,
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
    refused_seeds = []
    for r in session.runs:
        if r.command == "jepa extract":
            try:
                session.refuse_seeds(r)
            except k.RunError as exc:
                refused_seeds.append(str(exc))
    check(
        "pilot-28: no extraction reads seeds 6-10, read from the runs' own manifests",
        not refused_seeds,
        refused_seeds,
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
        "a real pilot writes the tau.json a staircase reads",
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
    pinned = hashlib.sha256(
        (PLANS.parents[1] / "paper" / "prereg" / DRAFT).read_bytes()
    ).hexdigest()
    prereg = plan["inputs"]["prereg"]
    check(
        "primary-28: its --prereg is the committed draft, pinned by sha256",
        (prereg["run"], prereg["file"], prereg["sha256"]) == (DRAFT_RUN, DRAFT, pinned),
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
        wheel(k, root / "wheel")
        verification(k, root / "verify")
        kernel_folder(k, root / "kernel")
        jepa_entries(k, root / "jepa")
        extraction_rates(k, root / "rates")
        kernel_acceptance(k, root / "acceptance")
        invalid_plans(k, root / "invalid")
        pilot_28(k, root / "pilot-28")
        primary_28(k, root / "primary-28")
        real_runs(k, root / "real")
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    import kaggle_ising_runs

    sys.exit(run_all(kaggle_ising_runs))
