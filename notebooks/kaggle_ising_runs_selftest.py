"""In-process checks of notebooks/kaggle_ising_runs.py, run by its --self-test:

    uv run --no-sync python notebooks/kaggle_ising_runs.py --self-test

Plan validation and the CLI's verdict on every run's arguments; dependency resolution; the queues
(longest first, dependencies, pinning, a failure, the deadline, the output cap) with a stand-in for
the simulator; the previous session's runs carried forward and reused or resumed; inputs found and
checked against their sha256; the wheel and COMMIT.txt checks; run verification, judged by exit
codes and files only. Nothing is simulated; one `jepa verify` runs in a child process. One line
per check, then exit 1 if any failed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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


def plans(k: ModuleType) -> None:
    for path in sorted(PLANS.glob("*.json")):
        why = refused(k, json.loads(path.read_text(encoding="utf-8")))
        check(f"plan {path.name} loads", not why, why)
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
        "params": {"checkpoint": {"sha256": digest}},
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
    group = rates["extract"].get("m3 final.pt", {})
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
    group = session.measured()["extract"].get("m3 final.pt", {})
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


def run_all(k: ModuleType) -> int:
    with tempfile.TemporaryDirectory(prefix="tp-self-test-") as tmp:
        root = Path(tmp)
        plans(k)
        refusals(k)
        cli(k)
        shrinking(k)
        queues(k)
        resolution(k, root / "resolve")
        previous(k, root / "previous")
        inputs(k, root / "inputs")
        wheel(k, root / "wheel")
        verification(k, root / "verify")
        kernel_folder(k, root / "kernel")
        jepa_entries(k, root / "jepa")
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1
