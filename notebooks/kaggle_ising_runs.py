"""Tipping Point: an explicit plan of simulator runs and JEPA trainings and extractions on Kaggle's
GPUs. A Kaggle script kernel.

A plan is plain JSON: an ordered list of runs, each an `ising` or `jepa` subcommand with its
arguments. --write-kernel embeds a plan in a copy of this file and writes kernel-metadata.json
beside it (private, on GPU T4 x2), and `kaggle kernels push -p <dir>` runs it. The machinery is
tp-tauc-l64-l128's (the kernel that ran on Kaggle's T4 x2, kept under .wf/kaggle/): the pinned
environment from a wheel dataset with its COMMIT.txt check, the preflight benchmark, a queue per
GPU through CUDA_VISIBLE_DEVICES, the stop at 8 h 15 min, checkpoints and resume across sessions,
STATUS, and the exit-code policy.

    {"kernel": "tp-ising-job-a",                  the kernel slug: <owner>/<slug> and its title
     "description": "...",                        optional
     "wheel": "<owner>/<slug>",                   optional: gabrielezerzer/tipping-point-wheel
     "commit": "<40 hex digits>",                 optional: the only commit COMMIT.txt may name
     "datasets": ["<owner>/<slug>"],              optional: datasets to attach
     "kernel_sources": ["<owner>/<slug>"],        optional: other kernels' outputs to attach
     "inputs": {"record": {"run": "<run id>", "file": "pilot.json", "sha256": "<hex>"}},
     "runs": [
       {"name": "grid-L64", "command": "pilot", "args": ["--L", 64, "--seed", 6]},
       {"name": "qs-L64-s11", "command": "staircase", "gpu": 1,
        "args": ["--L", 64, "--seed", 11, "--tau", {"run": "grid-L64", "file": "tau.json"}]},
       {"name": "m1", "command": "jepa train", "gpu": 0, "stage": ["<run id>", ...],
        "args": ["--L", 16, "--train-seeds", 11, "--val-seed", 16, "--seed", 1,
                 "--pilot", {"input": "record"}]},
       {"name": "x1", "command": "jepa extract", "gpu": 1,
        "args": [{"run": "m1", "file": "final.pt"}, {"dataset_run": "<run id>"}]}]}

- command is pilot, staircase, ramp, null, jepa train or jepa extract; args holds its flags and
  values, one item each. The runner adds --device, and --data-dir where the command takes one.
- {"run": <name>, "file": <path>} is that earlier run's folder plus <path>, resolved when the run
  starts (run ids are timestamped); the run waits until that one is complete and verified. A
  training's folder holds final.pt and its step-0 twin ckpt/step_0000000.pt.
- {"input": <name>} is the file an `inputs` entry names, looked up in the attached datasets and
  kernel outputs (in a <run id> folder or a <run id>.zip) and checked against its sha256 before
  any run starts: a pilot record for --pilot, or a checkpoint for `jepa extract`. A final.pt with
  no complete.json beside it is refused then as well, since `jepa extract` would refuse it.
- {"dataset_run": <run id>} is a simulator run from an attached dataset or kernel output, and
  "stage" lists runs a training reads (it finds them by L and seed). Before any run starts, each is
  placed in out/ising/<run id> (inputs are read-only, and an extraction writes inside its run:
  copied, hard linked on one filesystem, or unpacked) and checked with `jepa verify`. A copy that
  gains no latents leaves the output at the end.
- gpu pins a run to one GPU. The others go to whichever GPU is free, longest first, by the CLI's
  own schedule length at the preflight's measured rate.

From the repository root:

    uv run --no-sync python notebooks/kaggle_ising_runs.py --write-kernel <plan> <dir> [--resume]
    uv run --no-sync python notebooks/kaggle_ising_runs.py --dry-run <plan> [--only <names>]
    uv run --no-sync python notebooks/kaggle_ising_runs.py --dry-run-jepa
    uv run --no-sync python notebooks/kaggle_ising_runs.py --self-test

The wheel dataset holds one wheel and its COMMIT.txt. From a clean checkout of the commit to run:

    uv build --package tipping-point --wheel --out-dir <dir>
    rm <dir>/.gitignore
    git rev-parse HEAD > <dir>/COMMIT.txt

then `kaggle datasets version -p <dir> -m <note>`, with a dataset-metadata.json naming the dataset.

--resume attaches the kernel's own previous output (kernel_sources). Push a version with it after
a session stopped at the deadline, and that version reuses the finished runs and continues the
unfinished ones (`ising resume`, `jepa resume`, or the same `jepa extract` again, which resumes).
Every session first copies the previous output's runs and the GPU-hour ledger into its own, so
each version's output holds the whole plan so far. So one version's 20 GB caps the whole chain: it
stops at 18 GB however it is split across versions. A plan whose output would exceed 18 GB must be
cut into separate plans, as #28 slice 4's latents are: plans/extract-s6.json to extract-s10.json,
one per confirmatory seed.

--dry-run really executes the plan on the CPU in a temporary folder, with sizes divided by
--shrink (8: L = 4 and 8 for Job A) and sweep counts by its square. It builds a wheel from HEAD in a
clean clone (the code the Kaggle wheel would hold), in an environment whose children print Kaggle's
sitecustomize errors, and runs two sessions of the written kernel. The first is stopped by a
simulated deadline at the first progress line after a run finishes. The second runs with the
first's output attached, as Kaggle attaches it, and a fresh `jepa verify` closes the rehearsal over
every run's folder. --only keeps the named runs and the runs they read: a staircase takes at least
101,000 sweeps whatever its L, since every plateau holds 500 + 2,000, so a busy CPU wants one.
--input SLUG=DIR attaches a folder as a dataset or kernel output. JEPA arguments are never shrunk:
--dry-run-jepa rehearses a JEPA plan instead, on tiny simulator runs it writes
(harness/jepa_ising.py's three-plateau staircase): a quick simulator run, a short training stopped
by the simulated deadline and resumed, then its final.pt and step-0 twin extracted.

On Kaggle simulator runs land in /kaggle/working/out/ising/<run id>/, trainings in
out/jepa/<run id>/ beside the GPU-hour ledger, latents inside their dataset run's folder, and each
run's output in out/logs/<name>.log (`kaggle kernels output` fetches out/). Before any run starts,
the wheel must be the only tipping_point wheel among the inputs, with a COMMIT.txt naming a commit
not marked dirty, and the installed package must name the same clean commit. After each run, the
runner checks: exit 0; complete.json present, clean, with every check passed; every file it lists
intact (`jepa verify`, which also checks a latents table's rows); and a manifest whose git_sha is
COMMIT.txt's, with dirty false. A simulator run's failed thermalised check is the one exception:
the scorer drops that plateau or null block and counts it, so the run is complete and STATUS
names it. Runs are judged by exit codes, files and manifests, never by output text, because
Kaggle's image prints sitecustomize errors from every child process. A failed simulator run or
training stops its GPU's queue and blocks the runs that read it; the other queues go on. A failed
extraction blocks only the runs that read it, and its queue's other tables go on. No run starts
once /kaggle/working holds 18 GB (Kaggle keeps 20 GB) or past 8 h 15 min.

Status: out/STATUS holds one line (running, complete, incomplete, failed or refused) with the
measured rates: seconds per training step after warm-up, from the training's own GPU-hour ledger,
and microseconds per extracted frame, by checkpoint and L. out/kaggle-ising-runs.json holds every
run's state, the rates, the plan, the environment and the preflight. The kernel exits non-zero only
when its output holds no run folder, since a failed version may not keep its output.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

PLAN_JSON: str | None = None  # --write-kernel writes the plan here

OWNER = "gabrielezerzer"
WHEEL_DATASET = "tipping-point-wheel"
DEFAULT_WHEEL = f"{OWNER}/{WHEEL_DATASET}"  # training jobs attach the latest wheel
PYTHON = "3.13"  # the wheel's Requires-Python
TORCH = "torch==2.14.1+cu126"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
PINNED = ("numpy==2.5.4", "pyarrow==26.0.0")  # uv.lock's versions
# Kaggle's image puts a sitecustomize that imports wrapt on the path of its /usr/bin/python3, which
# is the venv's interpreter, so -I does not skip it: without wrapt every child prints two error
# lines (tp-tauc-l64-l128's log). Runs never depend on it.
SITE_NOISE = ("wrapt==1.17.3",)
ISING_COMMANDS = ("pilot", "staircase", "ramp", "null")
COMMANDS = (*ISING_COMMANDS, "jepa train", "jepa extract")
RUNNER_FLAGS = ("--device", "--data-dir", "-h", "--help")
# what tipping_point.jepa.train names (TWIN, FINAL, LEDGER); the self-test holds them to it
TWIN, FINAL, LEDGER = "ckpt/step_0000000.pt", "final.pt", "gpu_hours.jsonl"
SESSION_S = 9 * 3600  # Kaggle's session cap
STOP_S = SESSION_S - 45 * 60  # no run starts, and running ones stop, past this
HEARTBEAT_S = 600
LOOP_OVERHEAD = 1.10  # the run loop's schedule logic and Parquet flushes over the bare kernel
RUN_OVERHEAD_S = 60  # start-up, and a pilot's closing tau.json pass
BENCH_SWEEPS = 2_000
NEW_FOLDER_S = 1800  # a fresh run makes its folder within this (a training loads its data first)
CAP_BYTES = 18 * 10**9  # Kaggle keeps at most 20 GB of output
STATE = "kaggle-ising-runs.json"
PLACEHOLDER = "<pending>"  # an argument still waiting on an earlier run
SIZE_FLAGS = ("--L",)
SWEEP_FLAGS = ("--therm", "--sweeps", "--checkpoint-every", "--warmup", "--pre", "--hold", "--rate")
NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
SLUG = re.compile(r"[a-z0-9][a-z0-9-]{4,49}")
SOURCE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
RUN_ID = re.compile(r"\d{8}T\d{6}Z-[A-Za-z0-9_-]+")
PROGRESS = re.compile(  # a simulator run's line, and a training's (`step 1,000/2,000 loss ...`)
    r"(?P<run>\d{8}T\d{6}Z-[A-Za-z0-9_-]+): step (?P<step>[\d,]+)/(?P<total>[\d,]+)"
    r"(?: \((?P<rate>[\d,]+) sweeps/s\))?"
)
T0 = time.time()

ENV_PROBE = r"""
import json, platform
import numpy, pyarrow, torch
import tipping_point
from tipping_point.ising import io, pilot
info = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "arch_list": torch.cuda.get_arch_list(),
    "numpy": numpy.__version__,
    "pyarrow": pyarrow.__version__,
    "tipping_point": tipping_point.__version__,
    "kernel_sha256": pilot.kernel_sha256(),
    "code_version": io.code_version(),
    "devices": [],
}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    info["devices"].append(
        {"index": i, "name": p.name, "sm": f"sm_{p.major}{p.minor}",
         "memory_gb": round(p.total_memory / 2**30, 1)}
    )
print(json.dumps(info))
"""

HELPER_SOURCE = r'''
"""kaggle_ising_runs' steps that need the tipping_point library, run in the pinned environment.
Usage: helper.py <command> <out.json> <in.json>, in.json holding the keyword arguments."""
import contextlib
import io as text_io
import json
import sys
import time
from pathlib import Path

import torch
from tipping_point.ising import __main__ as cli
from tipping_point.ising import io, kernel, run
from tipping_point.ising.exact import TC
from tipping_point.jepa import __main__ as jepa_cli


def bench(shapes, sweeps, device):
    """The bare kernel loop at Tc with the per-sweep totals a run records, for each [L, lattices]:
    the floor under the run loop's rate."""
    return [_bench(L, lattices, sweeps, device) for L, lattices in shapes]


def _bench(L, lattices, sweeps, device):
    dev = torch.device(device)
    torch.use_deterministic_algorithms(True)
    gen = torch.Generator(device=dev).manual_seed(1)
    s = torch.randint(0, 2, (lattices, L, L), generator=gen, device=dev, dtype=torch.int8) * 2 - 1
    black = kernel.checkerboard(L, dev)
    table = kernel.accept_table(torch.full((lattices,), TC, dtype=torch.float64, device=dev))
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)

    def timed(n):
        nonlocal s
        kept = []
        sync()
        clock = time.perf_counter()
        for _ in range(n):
            s = kernel.sweep(s, black, table, gen)
            kept.append(kernel.totals(s))
        sync()
        return time.perf_counter() - clock

    timed(min(100, sweeps))  # warm-up: kernel loading, the caching allocator
    seconds = timed(sweeps)
    return {
        "L": L,
        "lattices": lattices,
        "sweeps": sweeps,
        "device": device,
        "ms_per_sweep_with_totals": round(1e3 * seconds / sweeps, 4),
        "sweeps_per_s_with_totals": round(sweeps / seconds, 1),
    }


class _Started(Exception):
    pass


def capture(items):
    """What each `ising <argv>` would run, asked of the CLI itself: run.start is swapped for a stub
    that notes the run's size and length and stops, so nothing is simulated or written. An argv
    still waiting on an earlier run (partial), and every `jepa` one, is only parsed."""
    return [_capture(item["argv"], item["partial"], item.get("module")) for item in items]


def _capture(argv, partial, module):
    seen = {}

    def stub(spec, schedule, root, tag, device=None, pilot=None):
        seen.update(L=spec.L, lattices=spec.replicas * schedule.blocks, total=schedule.total)
        raise _Started

    said = text_io.StringIO()
    real, run.start = run.start, stub
    try:
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
            if module == "jepa":
                jepa_cli._parser().parse_args(argv)
                return {}
            if partial:
                args = cli._parser().parse_args(argv)
                # shortcut: a waiting run is taken as one block of --replicas, true of staircase,
                # ramp and null, the commands that take a file; upgrade if pilot ever takes one
                return {"L": args.L, "lattices": args.replicas, "total": None}
            cli.main(argv)
    except _Started:
        return seen
    except SystemExit as exc:  # argparse's usage error, or the CLI's own refusal
        lines = said.getvalue().strip().splitlines() or [f"exit {exc.code}"]
        return {"error": exc.code if isinstance(exc.code, str) else lines[-1]}
    except Exception as exc:
        return {"error": repr(exc)}
    finally:
        run.start = real
    return {"error": "the command ended without starting a run"}


def tau_json(run_dir, seconds):
    """What `ising pilot` writes after its run (cli._pilot), from the CLI's own _measure. A resumed
    pilot needs it: `ising resume` never writes tau.json."""
    run_dir = Path(run_dir)
    L = io.read_json(run_dir / "manifest.json")["params"]["L"]
    points = cli._measure(run_dir, L)
    io.write_json(run_dir / "tau.json", {"L": L, "seconds": seconds, "points": points})
    return {"points": len(points), "reliable": sum(bool(p["reliable"]) for p in points)}


COMMANDS = {"bench": bench, "capture": capture, "tau_json": tau_json}

if __name__ == "__main__":
    command, out, given = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    result = COMMANDS[command](**json.loads(given.read_text(encoding="utf-8")))
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
'''


def log(message: object) -> None:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    for line in str(message).splitlines() or [""]:
        print(f"[{stamp} UTC +{time.time() - T0:6.0f}s] {line}", flush=True)


def now_utc() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def write_text(path: Path, text: str) -> None:
    """Through a temporary file and a rename, so no reader sees half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_bytes(root: Path) -> int:
    total = 0
    for folder, _, files in os.walk(root):  # skips a folder removed while it walks
        for name in files:
            with contextlib.suppress(OSError):  # renamed or removed while counted
                total += os.stat(os.path.join(folder, name)).st_size
    return total


def guarded(what: str, fn: Callable[..., object], *args: object) -> None:
    """fn(*args), an error logged and not raised: bookkeeping never stops a queue."""
    try:
        fn(*args)
    except Exception as exc:
        log(f"FAILED {what}: {exc!r}\n{traceback.format_exc()}")


def mem_total_gb() -> float | None:
    with contextlib.suppress(OSError, ValueError, IndexError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 1)
    return None


def stream(argv, *, env=None, cwd=None, check=True, prefix=""):
    """Run a child, logging each output line with a timestamp; (exit code, lines)."""
    argv = [str(a) for a in argv]
    log(f"{prefix}$ {' '.join(argv)}")
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        cwd=cwd,
    )
    lines = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            lines.append(line)
            log(prefix + line)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    code = proc.wait()
    if check and code != 0:
        tail = "\n".join(lines[-30:])
        raise RuntimeError(f"exit {code} from {' '.join(argv[:8])} ...\n{tail}")
    return code, lines


# -- the plan ----------------------------------------------------------------------------------


class PlanError(ValueError):
    """A plan that cannot run as written; the message lists every problem."""


def safe_path(value: object) -> bool:
    """A relative POSIX path that stays inside the folder it is read against."""
    return (
        isinstance(value, str)
        and value != ""
        and not value.startswith("/")
        and "\\" not in value
        and ":" not in value
        and ".." not in PurePosixPath(value).parts
    )


def input_problems(name: object, entry: object) -> list[str]:
    if not (isinstance(name, str) and NAME.fullmatch(name)):
        return [f"input name {name!r}: 1-64 of A-Z, a-z, 0-9, '-' and '_'"]
    if not (isinstance(entry, dict) and set(entry) == {"run", "file", "sha256"}):
        return [f"input {name!r}: {{'run': <run id>, 'file': <path>, 'sha256': <hex>}}"]
    problems = []
    if not (isinstance(entry["run"], str) and RUN_ID.fullmatch(entry["run"])):
        problems.append(f"input {name!r}: run {entry['run']!r} is not a run id")
    if not safe_path(entry["file"]):
        problems.append(f"input {name!r}: file {entry['file']!r} is not a relative path")
    if not (isinstance(entry["sha256"], str) and SHA256.fullmatch(entry["sha256"])):
        problems.append(f"input {name!r}: sha256 must be 64 lowercase hex digits")
    return problems


def ref_problems(ref: dict, earlier: dict[str, dict], inputs: dict, sourced: bool) -> list[str]:
    if set(ref) in ({"run"}, {"run", "file"}):
        target = ref["run"]
        problems = []
        if not (isinstance(target, str) and target in earlier):
            problems.append(f"{target!r} is not an earlier run of this plan")
        if "file" in ref and not safe_path(ref["file"]):
            problems.append(f"file {ref['file']!r} is not a relative path inside the run folder")
        return problems
    if set(ref) == {"input"}:
        target = ref["input"]
        ok = isinstance(target, str) and target in inputs
        return [] if ok else [f"input {target!r} is not in the plan's inputs"]
    if set(ref) == {"dataset_run"}:
        target = ref["dataset_run"]
        problems = (
            []
            if isinstance(target, str) and RUN_ID.fullmatch(target)
            else [f"dataset_run {target!r} is not a run id"]
        )
        if not sourced:
            problems.append("a dataset_run needs a dataset or a kernel source to be found in")
        return problems
    return [
        f"reference {ref}: {{'run': <earlier run>, 'file': <path>}}, {{'input': <name>}} or "
        "{'dataset_run': <run id>}"
    ]


def extract_problems(args: list, earlier: dict[str, dict]) -> list[str]:
    """A `jepa extract` reads one checkpoint and one simulator run, each a reference."""
    problems = []
    refs = [a for a in args if isinstance(a, dict)]
    runs = [
        a
        for a in refs
        if "dataset_run" in a or earlier.get(str(a.get("run")), {}).get("command") in ISING_COMMANDS
    ]
    checkpoints = [
        a
        for a in refs
        if "input" in a
        or ("file" in a and earlier.get(str(a.get("run")), {}).get("command") == "jepa train")
    ]
    if len(runs) != 1 or len(checkpoints) != 1 or len(refs) != 2:
        problems.append(
            "jepa extract reads one checkpoint ({'run': <training>, 'file': <path>} or "
            "{'input': <name>}) and one simulator run ({'dataset_run': <run id>} or "
            "{'run': <simulator run>})"
        )
    return problems


def run_problems(run: dict, earlier: dict[str, dict], inputs: dict, sourced: bool) -> list[str]:
    problems = []
    unknown = set(run) - {"name", "command", "args", "gpu", "stage"}
    if unknown:
        problems.append(f"unknown keys {sorted(unknown)}")
    name = run.get("name")
    if not (isinstance(name, str) and NAME.fullmatch(name)):
        problems.append(f"name {name!r}: 1-64 of A-Z, a-z, 0-9, '-' and '_'")
    elif name in earlier:
        problems.append("the name is used twice")
    if run.get("command") not in COMMANDS:
        problems.append(f"command {run.get('command')!r}: one of {', '.join(COMMANDS)}")
    gpu = run.get("gpu")
    if gpu is not None and (isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0):
        problems.append(f"gpu {gpu!r}: a GPU index, 0 or more")
    stage = run.get("stage", [])
    if not (
        isinstance(stage, list) and all(isinstance(s, str) and RUN_ID.fullmatch(s) for s in stage)
    ):
        problems.append("stage: a list of run ids")
    elif len(set(stage)) != len(stage):
        problems.append("stage: a run id listed twice")
    elif stage and not sourced:
        problems.append("stage needs a dataset or a kernel source to find the runs in")
    args = run.get("args")
    if not isinstance(args, list):
        return [*problems, "args: a list of flags and values"]
    for arg in args:
        if isinstance(arg, dict):
            problems += ref_problems(arg, earlier, inputs, sourced)
        elif isinstance(arg, bool) or not isinstance(arg, str | int | float):
            problems.append(f"argument {arg!r}: a string, a number or a reference")
        elif isinstance(arg, float) and not math.isfinite(arg):
            problems.append(f"argument {arg!r}: not a finite number")
        elif isinstance(arg, str) and arg.split("=", 1)[0] in RUNNER_FLAGS:
            problems.append(f"argument {arg}: the runner sets it")
    if run.get("command") == "jepa extract":
        problems += extract_problems(args, earlier)
    return problems


def load_plan(plan: object) -> dict:
    """The plan, checked as a whole; PlanError lists every problem found."""
    if not isinstance(plan, dict):
        raise PlanError("a plan is a JSON object")
    problems = []
    keys = {
        "kernel",
        "description",
        "wheel",
        "commit",
        "datasets",
        "kernel_sources",
        "inputs",
        "runs",
    }
    if set(plan) - keys:
        problems.append(f"unknown keys {sorted(set(plan) - keys)}")
    if "commit" in plan and not (
        isinstance(plan["commit"], str) and COMMIT.fullmatch(plan["commit"])
    ):
        problems.append(f"commit {plan['commit']!r}: 40 lowercase hex digits")
    if not (isinstance(plan.get("kernel"), str) and SLUG.fullmatch(plan["kernel"])):
        problems.append(f"kernel {plan.get('kernel')!r}: a slug, 5-50 of a-z, 0-9 and '-'")
    if not isinstance(plan.get("description", ""), str):
        problems.append("description: a string")
    wheel = plan.get("wheel", DEFAULT_WHEEL)
    if not (isinstance(wheel, str) and SOURCE.fullmatch(wheel)):
        problems.append(f"wheel {wheel!r}: the '<owner>/<slug>' of a wheel dataset")
    sources = 0
    for key in ("datasets", "kernel_sources"):
        value = plan.get(key, [])
        if isinstance(value, list) and all(
            isinstance(s, str) and SOURCE.fullmatch(s) for s in value
        ):
            sources += len(value)
        else:
            problems.append(f"{key}: a list of '<owner>/<slug>'")
    inputs = plan.get("inputs", {})
    if not isinstance(inputs, dict):
        problems.append("inputs: an object of named files")
        inputs = {}
    for name, entry in inputs.items():
        problems += input_problems(name, entry)
    if inputs and not sources:
        problems.append("inputs need a dataset or a kernel source to be found in")
    runs = plan.get("runs")
    if not (isinstance(runs, list) and runs):
        problems.append("runs: a non-empty list")
        runs = []
    earlier: dict[str, dict] = {}
    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            problems.append(f"runs[{i}]: an object")
            continue
        name = run.get("name")
        where = f"run {name!r}" if isinstance(name, str) else f"runs[{i}]"
        problems += [f"{where}: {p}" for p in run_problems(run, earlier, inputs, sources > 0)]
        if isinstance(name, str):
            earlier[name] = run
    if problems:
        raise PlanError("\n".join(problems))
    return plan


@dataclass
class Run:
    """One run of the plan, and what became of it in this session."""

    name: str
    command: str
    args: list
    gpu: int | None
    index: int
    deps: list[str]
    stage: list[str] = field(default_factory=list)  # dataset runs a training reads
    state: str = "pending"  # running, complete, failed, interrupted, blocked, not started
    note: str = ""
    run_id: str | None = None  # the name of its folder
    folder: str | None = None  # under out/: ising/<id>, jepa/<id>, ising/<id>/latents/<table>
    key: list[str] | None = None  # command and arguments, references as folders: what reuse matches
    shape: list[int] | None = None  # [L, lattices]
    total: int | None = None  # sweeps
    step: int = 0
    sweeps_per_s: int | None = None
    estimate_s: float | None = None
    gpu_used: int | None = None
    exit: int | None = None
    wall_s: float = 0.0
    reused: bool = False
    resumed: bool = False
    problems: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # thermalised misses: the scorer's to drop
    output_tail: list[str] = field(default_factory=list)


def make_runs(plan: dict) -> list[Run]:
    runs = []
    for i, spec in enumerate(plan["runs"]):
        refs = [a["run"] for a in spec["args"] if isinstance(a, dict) and "run" in a]
        runs.append(
            Run(
                spec["name"],
                spec["command"],
                spec["args"],
                spec.get("gpu"),
                i,
                [*dict.fromkeys(refs)],
                stage=list(spec.get("stage", [])),
            )
        )
    return runs


def module_of(command: str) -> tuple[str, str]:
    """A plan command's CLI module and subcommand: `pilot` is ising's, `jepa train` jepa's."""
    module, _, sub = command.rpartition(" ")
    return module or "ising", sub


def dataset_runs(runs: list[Run]) -> list[str]:
    """Every simulator run the plan reads from its inputs: staged and dataset_run ones."""
    named = [s for r in runs for s in r.stage]
    named += [
        a["dataset_run"] for r in runs for a in r.args if isinstance(a, dict) and "dataset_run" in a
    ]
    return [*dict.fromkeys(named)]


def link_or_copy(src: str, dst: str) -> str:
    """A hard link where both sit on one filesystem, else a copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def shrink(args: list, factor: int) -> list:
    """A dry run's arguments: L divided by factor (even, at least 4) and sweep counts by its
    square, since tau_c grows about as L^2: each run keeps roughly its length in tau_c."""
    out = list(args)
    if factor == 1:
        return out
    for i in range(1, len(out)):
        flag, value = out[i - 1], out[i]
        if isinstance(value, bool) or not isinstance(value, int | str) or not str(value).isdigit():
            continue
        n = int(value)
        if flag in SIZE_FLAGS:
            out[i] = max(4, n // factor // 2 * 2)
        elif flag in SWEEP_FLAGS and n > 0:
            out[i] = max(1, n // factor**2)
    return out


# -- where a session runs ----------------------------------------------------------------------


@dataclass
class Ctx:
    """Where a session reads and writes, and how it runs."""

    input: Path  # attached datasets and kernel outputs, read-only
    work: Path  # what the version keeps, so what the output budget counts
    tmp: Path  # the environment, the helper and the children's working dir: never output
    device: str
    poll_s: float
    bench_sweeps: int
    py: str = ""  # the pinned environment's python, once there is one
    gpus: int | None = None  # None: as many as torch sees
    shrink: int = 1
    interrupt: bool = False  # dry run: a deadline at the first progress line after a run finishes
    stop_s: float = STOP_S
    cap_bytes: int = CAP_BYTES

    @property
    def out(self) -> Path:
        return self.work / "out"

    @property
    def cwd(self) -> Path:
        return self.tmp / "tp-cwd"

    @property
    def helper(self) -> Path:
        return self.tmp / "tp-helper" / "helper.py"


def kaggle_ctx() -> Ctx:
    return Ctx(
        input=Path("/kaggle/input"),
        work=Path("/kaggle/working"),
        tmp=Path("/tmp"),
        device="cuda",
        poll_s=10,
        bench_sweeps=BENCH_SWEEPS,
    )


def child_env(ctx: Ctx) -> dict[str, str]:
    return dict(os.environ, TP_DATA_DIR=str(ctx.out), CUBLAS_WORKSPACE_CONFIG=":4096:8")


def run_helper(ctx: Ctx, command: str, env: dict | None = None, **kwargs: Any) -> Any:
    """A helper command in the pinned environment; its result comes back through a file."""
    stamp = f"{command}-{time.time_ns()}-{threading.get_ident()}"
    given, out = ctx.helper.parent / f"{stamp}-in.json", ctx.helper.parent / f"{stamp}-out.json"
    given.write_text(json.dumps(kwargs), encoding="utf-8")
    argv = [ctx.py, "-I", "-u", ctx.helper, command, out, given]
    stream(argv, env=env or child_env(ctx), cwd=ctx.cwd, prefix="[helper] ")
    return json.loads(out.read_text(encoding="utf-8"))


# -- the wheel and the environment -------------------------------------------------------------


class Refused(Exception):
    """The session cannot start; no run has begun."""


def find_input(root: Path, slug: str) -> Path:
    direct = root / slug
    if direct.is_dir():
        return direct
    hits = sorted(p for p in root.glob(f"**/{slug}") if p.is_dir())
    if hits:
        return hits[0]
    listing = "\n".join(str(p) for p in sorted(root.glob("*/*"))[:60])
    raise Refused(f"input {slug!r} is not attached under {root}:\n{listing}")


def find_wheel(ctx: Ctx, source: str = DEFAULT_WHEEL) -> tuple[Path, dict]:
    """The wheel of the plan's wheel dataset, and the commit its COMMIT.txt names; refused when
    that is missing or dirty, or when the inputs hold any other tipping_point wheel."""
    root = find_input(ctx.input, source.split("/", 1)[1])
    wheels = sorted(ctx.input.rglob("tipping_point-*.whl"))
    if len(wheels) != 1 or not wheels[0].is_relative_to(root):
        raise Refused(
            f"expected one tipping_point wheel among the inputs, in {source}; "
            f"found {[str(w) for w in wheels]}"
        )
    notes = sorted(root.rglob("COMMIT.txt"))
    if len(notes) != 1:
        raise Refused(
            f"expected one COMMIT.txt beside the wheel in {root}, found {len(notes)}: "
            "nothing else names the commit the runs would record"
        )
    raw = notes[0].read_bytes()  # PowerShell 5.1's `>` writes UTF-16 with a byte-order mark (#134)
    text = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig").strip()
    if "dirty" in text.lower():
        raise Refused(f"COMMIT.txt marks the commit dirty ({text!r}): build from a clean checkout")
    if not COMMIT.fullmatch(text):
        raise Refused(f"COMMIT.txt names no commit: {text!r}")
    info = {"file": wheels[0].name, "sha256": sha256_file(wheels[0]), "commit": text}
    log(f"wheel {info['file']} sha256 {info['sha256'][:16]}..., COMMIT.txt {text!r}")
    return wheels[0], info


def ensure_env(ctx: Ctx, wheel: Path) -> dict:
    """The pinned environment under ctx.tmp, as tp-tauc-l64-l128 builds it, then a probe of it."""
    env_dir, uv_cache = ctx.tmp / "tp-env", ctx.tmp / "tp-uv-cache"
    for where in (ctx.tmp, ctx.work):  # the environment needs about 7 GB under /tmp
        log(f"free disk under {where}: {shutil.disk_usage(where).free / 2**30:.1f} GB")
    pip_env = dict(os.environ, PIP_ROOT_USER_ACTION="ignore", PIP_DISABLE_PIP_VERSION_CHECK="1")
    stream([sys.executable, "-m", "pip", "install", "--quiet", "uv"], env=pip_env, cwd=ctx.cwd)
    uv_env = dict(
        os.environ,
        UV_CACHE_DIR=str(uv_cache),
        UV_LINK_MODE="copy",
        UV_PYTHON_INSTALL_DIR=str(ctx.tmp / "tp-uv-python"),
    )
    uv = [sys.executable, "-m", "uv"]
    stream([*uv, "--version"], env=uv_env, cwd=ctx.cwd)
    stream([*uv, "venv", "--python", PYTHON, env_dir], env=uv_env, cwd=ctx.cwd)
    ctx.py = str(env_dir / "bin" / "python")
    install = [*uv, "pip", "install", "--python", ctx.py]
    stream([*install, TORCH, "--index-url", TORCH_INDEX], env=uv_env, cwd=ctx.cwd)
    stream([*install, *PINNED, *SITE_NOISE, wheel], env=uv_env, cwd=ctx.cwd)
    shutil.rmtree(uv_cache, ignore_errors=True)
    return probe(ctx)


def probe(ctx: Ctx) -> dict:
    """The pinned environment, from inside: versions, devices and the commit it names."""
    ctx.helper.parent.mkdir(parents=True, exist_ok=True)
    ctx.helper.write_text(HELPER_SOURCE, encoding="utf-8")
    _, lines = stream([ctx.py, "-I", "-c", ENV_PROBE], env=child_env(ctx), cwd=ctx.cwd)
    info = json.loads(next(line for line in reversed(lines) if line.startswith("{")))
    if shutil.which("nvidia-smi"):
        query = "--query-gpu=index,name,memory.total,driver_version"
        argv = ["nvidia-smi", query, "--format=csv,noheader"]
        _, info["nvidia_smi"] = stream(argv, check=False, cwd=ctx.cwd)
    info["cpu_count"] = os.cpu_count()
    info["mem_total_gb"] = mem_total_gb()
    info["disk_free_gb"] = round(shutil.disk_usage(ctx.work).free / 2**30, 1)
    return info


def check_env(ctx: Ctx, env: dict, commit: str) -> None:
    if ctx.device == "cuda":
        if env["torch"] != TORCH.split("==")[1]:
            raise Refused(f"torch is {env['torch']}, not {TORCH}")
        if not env["cuda_available"] or not env["devices"]:
            raise Refused("torch sees no CUDA device; attach a GPU (enable_gpu)")
    version = env["code_version"]
    if version.get("git_sha") != commit:
        raise Refused(
            f"the installed package names commit {version.get('git_sha')}, COMMIT.txt {commit}"
        )
    if version.get("dirty") is not False:
        raise Refused(f"the installed package's commit is marked dirty ({version.get('dirty')})")


def unzip_run(archive: Path, scratch: Path) -> Path | None:
    """One <run_id>.zip into scratch/<run_id>/ (tp-tauc-l64-l128's): a run's files at the archive
    root, as kaggle's --dir-mode zip makes it, or under a <run_id>/ folder."""
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        for name in names:
            if name.startswith("/") or ".." in PurePosixPath(name).parts or ":" in name:
                raise Refused(f"{archive.name} holds an unsafe path {name!r}")
        markers = sorted(
            (
                PurePosixPath(n)
                for n in names
                if PurePosixPath(n).name in ("manifest.json", "pilot.json")
            ),
            key=lambda p: len(p.parts),
        )
        if not markers:
            return None
        prefix = markers[0].parent.parts  # () when the run's files sit at the archive root
        run_id = prefix[-1] if prefix else archive.stem
        if not RUN_ID.fullmatch(run_id):
            raise Refused(f"{archive.name} unpacks to {run_id!r}, not a run id")
        target = scratch / run_id
        if target.exists():
            return target
        for name in names:
            parts = PurePosixPath(name).parts
            if parts[: len(prefix)] != prefix:
                continue
            dest = target.joinpath(*parts[len(prefix) :])
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, open(dest, "wb") as sink:
                shutil.copyfileobj(source, sink, 1 << 20)
    return target


def carry_forward(ctx: Ctx, plan: dict) -> dict | None:
    """The previous session's state, when its output is attached (kernel_sources), with the plan's
    runs it holds copied into this output first, so they stay in the chain whatever becomes of
    this session. The newest state of this kernel wins."""
    found = []
    for path in ctx.input.rglob(STATE):
        state = read_json(path)
        if state and state.get("kernel") == plan["kernel"] and isinstance(state.get("runs"), list):
            found.append((str(state.get("updated_utc") or ""), str(path), path, state))
    if not found:
        log("none attached")
        return None
    _, _, path, state = max(found, key=lambda f: (f[0], f[1]))
    names = {r["name"] for r in plan["runs"]}
    kept = []
    for record in state["runs"]:
        if not (isinstance(record, dict) and record.get("name") in names):
            continue
        legacy = f"ising/{record['run_id']}" if record.get("run_id") else None  # before folders
        folder = record.get("folder") or legacy
        if not folder:
            continue
        top = "/".join(PurePosixPath(folder).parts[:2])  # a latents table travels in its run
        src, dest = path.parent / top, ctx.out / top
        if not src.is_dir():
            log(f"  {record['name']}: {src} is not in the attached output; skipped")
            continue
        if not dest.exists():
            shutil.copytree(src, dest)
        kept.append(dict(record, folder=folder))
        log(f"  {record['name']}: {folder} ({record.get('state')}) copied in")
    ledger = path.parent / "jepa" / LEDGER
    if ledger.is_file():  # the GPU hours of every session so far
        (ctx.out / "jepa").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ledger, ctx.out / "jepa" / LEDGER)
    log(f"{path}: {state.get('status')}; {len(kept)} runs carried forward")
    return {"path": str(path), "status": state.get("status"), "runs": kept}


# -- verification ------------------------------------------------------------------------------


def checks_in(node: object) -> list[dict]:
    """Every verdict in complete.json's checks: the objects that carry "passed"."""
    if isinstance(node, dict):
        return [node] if "passed" in node else [c for v in node.values() for c in checks_in(v)]
    if isinstance(node, list):
        return [c for v in node for c in checks_in(v)]
    return []


def verdicts(run_dir: Path, commit: str, command: str) -> tuple[list[str], list[str]]:
    """What keeps a finished run from counting: its manifest must name the wheel's commit and say
    it is not dirty, and complete.json must say clean and hold no failed check. Second, what
    counts with a note: a simulator run's failed thermalised checks, since the scorer drops each
    such plateau or null block and counts it (ews.ising.plateaus and select)."""
    manifest, done = read_json(run_dir / "manifest.json"), read_json(run_dir / "complete.json")
    if manifest is None:
        return ["manifest.json is missing or unreadable"], []
    if done is None:
        return ["complete.json is missing or unreadable"], []
    producer = manifest.get("producer") or {}
    problems: list[str] = []
    dropped: list[str] = []
    if producer.get("git_sha") != commit:
        problems.append(f"manifest git_sha {producer.get('git_sha')} is not the wheel's {commit}")
    if producer.get("dirty") is not False:
        problems.append(f"manifest dirty is {producer.get('dirty')}")
    if done.get("clean") is not True:
        problems.append(f"complete.json clean is {done.get('clean')}")
    for check in checks_in(done.get("checks")):
        if check.get("passed") is not True:
            T, gate = check.get("temperature"), check.get("gate", "?")
            at = f" at T = {T:.4f}" if isinstance(T, int | float) else ""
            said = f"check {gate} failed{at}: {check.get('detail', '')}"
            if gate == "thermalised" and command in ISING_COMMANDS:
                dropped.append(said)
            else:
                problems.append(said)
    return problems, dropped


def file_problems(code: int, lines: list[str], dirs: list[Path]) -> list[str]:
    """`jepa verify`'s verdict is its exit code, never its other output: Kaggle's image prints
    sitecustomize errors from every child. The lines it writes for the folders explain a failure."""
    if code == 0:
        return []
    mine = [
        line
        for line in lines
        if any(line.startswith(f"{d}: ") for d in dirs) and not line.endswith(": ok")
    ]
    return mine or [f"jepa verify exited {code}"]


# -- the queues --------------------------------------------------------------------------------


class Scheduler:
    """One queue per GPU. A free GPU takes the longest pending run whose dependencies are complete
    (a pinned run only on its own GPU). A failed run blocks the runs that read it. A failed
    simulator run or training also stops its GPU's queue, while the other queues go on; a failed
    extraction does not, so the tables behind it still run. No run starts after `stop`, or once
    `budget` objects."""

    def __init__(
        self,
        runs: list[Run],
        gpus: int,
        execute: Callable[[int, Run], tuple[str, str]],
        budget: Callable[[], str | None] | None = None,
        ready: Callable[[list[Run]], None] | None = None,
        changed: Callable[[], None] | None = None,
    ) -> None:
        self.runs, self.gpus, self.execute = runs, gpus, execute
        self.by_name = {r.name: r for r in runs}
        self.budget = budget or (lambda: None)
        self.ready = ready or (lambda runs: None)
        self.changed = changed or (lambda: None)
        self.cond = threading.Condition()
        self.halt: str | None = None  # why no run may start any more
        self.stopped: dict[int, str] = {}  # GPU -> why its queue stopped

    def stop(self, reason: str) -> None:
        with self.cond:
            self.halt = self.halt or reason
            self.cond.notify_all()

    def run(self, tick: Callable[[], None], poll_s: float) -> None:
        threads = [
            threading.Thread(target=self._queue, args=(gpu,), name=f"gpu{gpu}", daemon=True)
            for gpu in range(self.gpus)
        ]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            time.sleep(poll_s)
            guarded("the session's clock", tick)
        with self.cond:
            self._settle(final=True)

    def _queue(self, gpu: int) -> None:
        while (run := self._take(gpu)) is not None:
            try:
                state, note = self.execute(gpu, run)
            except Exception as exc:  # recorded; the other queues go on
                state, note = "failed", f"runner error: {exc!r}"
                log(f"[gpu{gpu} {run.name}] FAILED in the runner: {exc!r}")
                log(traceback.format_exc())
            with self.cond:
                run.state, run.note = state, note
                # a failed table costs only the runs that read it: quota is what stopping wastes
                if state == "failed" and run.command != "jepa extract":
                    self.stopped[gpu] = f"{run.name} failed"
            if state == "complete":
                guarded("sizing the runs it frees", self.ready, self._unblocked(run))
            with self.cond:
                self.cond.notify_all()
            guarded("the session record", self.changed)

    def _unblocked(self, done: Run) -> list[Run]:
        with self.cond:
            return [
                r
                for r in self.runs
                if r.state == "pending"
                and done.name in r.deps
                and all(self.by_name[d].state == "complete" for d in r.deps)
            ]

    def _take(self, gpu: int) -> Run | None:
        with self.cond:
            while True:
                self._settle()
                if gpu in self.stopped or self.halt is not None:
                    return None
                mine = [r for r in self.runs if r.state == "pending" and r.gpu in (None, gpu)]
                if not mine:
                    return None
                ready = [
                    r for r in mine if all(self.by_name[d].state == "complete" for d in r.deps)
                ]
                if ready:
                    self.halt = self.budget()
                    if self.halt is not None:
                        log(f"no run starts from now on: {self.halt}")
                        return None
                    run = max(ready, key=lambda r: (r.estimate_s or 0.0, -r.index))
                    run.state, run.gpu_used = "running", gpu
                    return run
                self.cond.wait(timeout=5)

    def _settle(self, final: bool = False) -> None:
        """Mark the pending runs that can no longer start in this session. Plan order settles a
        run's dependencies before it."""
        for r in self.runs:
            if r.state != "pending":
                continue
            dead = next(
                (
                    self.by_name[d]
                    for d in r.deps
                    if self.by_name[d].state not in ("pending", "running", "complete")
                ),
                None,
            )
            if dead is not None:
                r.state = "blocked" if dead.state in ("failed", "blocked") else "not started"
                was = "failed" if dead.state == "failed" else f"is {dead.state}"
                r.note = f"depends on {dead.name}, which {was}"
            elif r.gpu is not None and r.gpu in self.stopped:
                r.state = "not started"
                r.note = f"pinned to gpu{r.gpu}, whose queue stopped after {self.stopped[r.gpu]}"
            elif final:
                r.state, r.note = "not started", self.halt or "every queue stopped"


def verdict(runs: list[Run], stopped: dict[int, str], final: bool = True) -> tuple[str, str]:
    """STATUS from the runs' states alone, which come from exit codes, files and manifests:
    complete, incomplete or failed once the session ends, running before."""
    by: dict[str, list[Run]] = collections.defaultdict(list)
    for r in runs:
        by[r.state].append(r)

    def names(group: list[Run]) -> str:
        return ", ".join(r.name for r in group)

    parts = []
    for r in by["failed"]:
        where = "" if r.gpu_used is None else f" on gpu{r.gpu_used}"
        parts.append(f"{r.name} FAILED{where}: {r.note} (log out/logs/{r.name}.log)")
    if stopped:
        parts.append(
            "queue stopped: " + ", ".join(f"gpu{g} after {w}" for g, w in sorted(stopped.items()))
        )
    if by["blocked"]:
        parts.append(f"blocked by a failure: {names(by['blocked'])}")
    for r in by["interrupted"]:
        parts.append(f"{r.name} interrupted at step {r.step:,} of {r.total or 0:,} (resumable)")
    waiting: dict[str, list[Run]] = collections.defaultdict(list)
    for r in by["not started"]:
        waiting[r.note].append(r)
    parts += [f"not started ({note}): {names(group)}" for note, group in waiting.items()]
    live = by["running"] + by["pending"]
    if not final and live:
        parts.append(f"running: {names(by['running']) or 'none'}; waiting: {len(by['pending'])}")
    done = by["complete"]
    extra = [
        f"{n} {what}"
        for n, what in (
            (sum(r.reused for r in done), "reused"),
            (sum(r.resumed for r in done), "resumed"),
        )
        if n
    ]
    parts.append(
        f"{len(done)}/{len(runs)} complete and verified"
        + (f" ({', '.join(extra)})" if extra else "")
    )
    if noted := [r for r in done if r.dropped]:
        parts.append(
            "thermalised missed in a plateau or null block, which the scorer drops and counts: "
            + names(noted)
        )
    if by["failed"] or by["blocked"]:
        status = "failed"
    elif live and not final:
        status = "running"
    elif live or by["interrupted"] or by["not started"]:
        status = "incomplete"
        parts.append("to continue, write the kernel with --resume and push it again")
    else:
        status = "complete"
    return status, "; ".join(parts)


# -- a session ---------------------------------------------------------------------------------


class RunError(Exception):
    """A run that cannot start: an argument does not resolve."""


class Session:
    """One session of a plan, on Kaggle or in a dry run: inputs, the previous session's runs, the
    preflight, the queues, and the record of it all in out/."""

    def __init__(
        self,
        ctx: Ctx,
        plan: dict,
        summary: dict,
        commit: str,
        torch_version: str,
        gpus: int,
        prior: dict | None = None,
    ) -> None:
        self.ctx, self.plan, self.summary = ctx, plan, summary
        self.commit, self.torch, self.gpus = commit, torch_version, gpus
        self.runs = make_runs(plan)
        self.by_name = {r.name: r for r in self.runs}
        self.env = child_env(ctx)
        self.inputs: dict[str, Path] = {}  # input name -> its checked file
        self.datasets: dict[str, Path] = {}  # dataset run id -> where the inputs hold it
        self.prior: dict[str, dict] = {r["name"]: r for r in (prior or {}).get("runs", [])}
        for r in self.runs:  # until this session decides, a run keeps the previous one's folder
            if r.name in self.prior:
                p = self.prior[r.name]
                r.folder, r.run_id = p["folder"], PurePosixPath(p["folder"]).name
                r.key, r.step = p.get("key"), int(p.get("step") or 0)
                r.wall_s = float(p.get("wall_s") or 0.0)
        self.rates: dict[tuple[int, int], float] = {}
        self.active: dict[str, subprocess.Popen] = {}
        self.active_lock = threading.Lock()  # a start against the deadline's kill sweep
        # one new run folder appears at a time in each place runs start in (an extraction
        # writes inside its dataset run, found by its checkpoint)
        self.start_locks = {"ising": threading.Lock(), "jepa": threading.Lock()}
        self.save_lock = threading.Lock()
        self.deadline_hit = False
        self.beat = time.time()
        self.scheduler = Scheduler(
            self.runs,
            gpus,
            self.execute,
            budget=self.budget,
            ready=self.capture_ready,
            changed=self.save,
        )

    def run(self) -> int:
        log("== inputs")
        self.locate_inputs()
        self.locate_datasets()
        log("== the plan as the CLI sets it up")
        errors = self.capture(self.runs)
        if errors:
            raise Refused("the CLI refuses: " + " | ".join(errors))
        far = [r.name for r in self.runs if r.gpu is not None and r.gpu >= self.gpus]
        if far:
            raise Refused(
                f"{', '.join(far)}: pinned to a GPU this session lacks ({self.gpus} here)"
            )
        log("== preflight: the bare kernel loop on each run's shape, GPU 0")
        self.preflight()
        log("== the dataset runs the plan reads, staged in out/ising")
        self.stage()
        self.save()
        log("== runs")
        try:
            self.scheduler.run(self.tick, self.ctx.poll_s)
        except Exception as exc:  # the runs made so far are in out/: record the error, keep them
            self.summary["runner_error"] = traceback.format_exc()[-6000:]
            log(f"FAILED in the runner: {exc!r}\n{self.summary['runner_error']}")
        return self.finish()

    def roots(self) -> list[Path]:
        """Where the plan's attached datasets and kernel outputs sit among the inputs."""
        sources = self.plan.get("datasets", []) + self.plan.get("kernel_sources", [])
        return [find_input(self.ctx.input, s.split("/", 1)[1]) for s in sources]

    def locate_datasets(self) -> None:
        """Where the inputs hold each dataset run the plan reads: its folder (as uploaded, or
        unpacked by Kaggle, possibly still named <run id>.zip) or its <run id>.zip."""
        wanted = dataset_runs(self.runs)
        if not wanted:
            return
        names = {*wanted, *(f"{w}.zip" for w in wanted)}
        hits: dict[str, list[Path]] = {}
        for root in self.roots():  # one walk per root, not one per run
            for path in root.rglob("*"):
                if path.name in names:
                    hits.setdefault(path.name.removesuffix(".zip"), []).append(path)
        for run_id in wanted:
            found = sorted(hits.get(run_id, []), key=lambda p: (p.is_file(), str(p)))
            usable = [p for p in found if p.is_file() or (p / "complete.json").is_file()]
            if not usable:
                raise Refused(f"dataset run {run_id} is in none of the attached inputs")
            self.datasets[run_id] = usable[0]
        log(f"{len(wanted)} dataset runs found among the attached inputs")

    def stage(self) -> None:
        """Each dataset run the plan reads, placed in out/ising/<run id>: inputs are read-only,
        a training finds its runs in its data dir, and an extraction writes inside its run.
        Copied, hard linked on one filesystem, or unpacked from its zip; then all checked with
        one `jepa verify`, judged by its exit code. A folder the previous session left (with
        its latents) stays as it is."""
        if not self.datasets:
            log("none")
            return
        ising = self.ctx.out / "ising"
        for run_id, src in self.datasets.items():
            dest = ising / run_id
            if dest.exists():
                continue
            if src.is_file():
                unpacked = unzip_run(src, ising)
                if unpacked is None or unpacked.name != run_id:
                    raise Refused(f"{src} does not unpack to the run {run_id}")
            else:
                shutil.copytree(src, dest, copy_function=link_or_copy)
        dirs = [ising / run_id for run_id in self.datasets]
        argv = [self.ctx.py, "-I", "-u", "-m", "tipping_point.jepa", "verify", *dirs]
        code, lines = stream(argv, env=self.env, cwd=self.ctx.cwd, check=False, prefix="[verify] ")
        problems = file_problems(code, lines, dirs)
        if problems:
            raise Refused("staged dataset runs fail jepa verify: " + "; ".join(problems[:5]))
        self.summary["staged"] = {run_id: str(src) for run_id, src in self.datasets.items()}

    def unstage(self) -> None:
        """Staged dataset runs that gained no latents leave the output: plain copies of inputs."""
        for run_id in self.datasets:
            dest = self.ctx.out / "ising" / run_id
            if dest.is_dir() and not (dest / "latents").exists():
                shutil.rmtree(dest, ignore_errors=True)
                log(f"unstaged {run_id}: a plain copy of an input")

    def locate_inputs(self) -> None:
        entries = self.plan.get("inputs", {})
        if not entries:
            log("none")
            return
        roots = self.roots()
        scratch = self.ctx.tmp / "tp-inputs"
        for name, entry in entries.items():
            found = []
            for root in roots:
                for hit in [*root.rglob(entry["run"]), *root.rglob(entry["run"] + ".zip")]:
                    folder = unzip_run(hit, scratch) if hit.is_file() else hit
                    if folder is not None and (folder / entry["file"]).is_file():
                        found.append(folder / entry["file"])
            digests = {str(p): sha256_file(p) for p in found}
            good = [p for p in found if digests[str(p)] == entry["sha256"]]
            if not good:
                where = ", ".join(map(str, roots))
                raise Refused(
                    f"input {name}: no {entry['run']}/{entry['file']} with sha256 "
                    f"{entry['sha256']} in {where}; found {digests or 'none'}"
                )
            # jepa extract refuses a trained checkpoint with no complete.json beside it (#128)
            # shortcut: presence only, not its file list, collapse or stopped_early; read those too
            # before a plan reads a training whose final verdict has not been seen
            usable = [p for p in good if p.name != FINAL or (p.parent / "complete.json").is_file()]
            if not usable:
                raise Refused(
                    f"input {name}: {good[0]} has no complete.json beside it, so jepa extract "
                    "would refuse it in every run that reads it"
                )
            self.inputs[name] = usable[0]
            log(f"input {name}: {usable[0]}, sha256 {entry['sha256'][:16]}... as the plan says")
        self.summary["inputs"] = {n: str(p) for n, p in self.inputs.items()}

    def resolve(self, run: Run, pending_ok: bool = False) -> tuple[list[str], list[str]]:
        """The run's arguments as the CLI takes them, and its key: references as folders."""
        tail, key = [], [run.command]
        args = shrink(run.args, self.ctx.shrink) if run.command in ISING_COMMANDS else run.args
        for arg in args:
            if not isinstance(arg, dict):
                tail.append(str(arg))
                key.append(str(arg))
            elif "input" in arg:
                entry = self.plan["inputs"][arg["input"]]
                tail.append(str(self.inputs[arg["input"]]))
                key.append(f"input/{entry['run']}/{entry['file']}")
            elif "dataset_run" in arg:
                folder = self.ctx.out / "ising" / arg["dataset_run"]
                if not folder.is_dir():
                    if not pending_ok:
                        raise RunError(f"{arg['dataset_run']} is not staged")
                    tail.append(PLACEHOLDER)
                    key.append(PLACEHOLDER)
                    continue
                tail.append(str(folder))
                key.append(f"dataset/{arg['dataset_run']}")
            else:
                dep = self.by_name[arg["run"]]
                if dep.state != "complete" or dep.folder is None:
                    if not pending_ok:
                        why = "has no folder" if dep.state == "complete" else f"is {dep.state}"
                        raise RunError(f"{dep.name} {why}, not complete")
                    tail.append(PLACEHOLDER)
                    key.append(PLACEHOLDER)
                    continue
                rel = dep.folder + (f"/{arg['file']}" if "file" in arg else "")
                if not (self.ctx.out / rel).exists():
                    raise RunError(f"{rel} does not exist ({dep.name}'s folder)")
                tail.append(str(self.ctx.out / rel))
                key.append(rel)
        return tail, key

    def helper(self, command: str, env: dict | None = None, **kwargs: Any) -> Any:
        return run_helper(self.ctx, command, env or self.env, **kwargs)

    def flags(self, run: Run, data_dir: Path) -> list[str]:
        """What the runner adds to a command: --device, and --data-dir where it takes one."""
        data = [] if run.command == "jepa extract" else ["--data-dir", str(data_dir)]
        return ["--device", self.ctx.device, *data]

    def capture(self, runs: list[Run]) -> list[str]:
        """Each simulator run's size and length as the CLI itself sets it up (the helper's
        capture); a run still waiting on an earlier one, and every jepa run, is only parsed.
        The CLI's refusals, by run."""
        items = []
        for r in runs:
            tail, _ = self.resolve(r, pending_ok=True)
            module, sub = module_of(r.command)
            argv = [sub, *tail, *self.flags(r, self.ctx.tmp / "tp-capture")]
            items.append({"argv": argv, "partial": PLACEHOLDER in tail, "module": module})
        errors = []
        for r, got in zip(runs, self.helper("capture", items=items), strict=True):
            if "error" in got:
                errors.append(f"{r.name}: {got['error']}")
            elif "L" in got:
                r.shape, r.total = [got["L"], got["lattices"]], got["total"]
                r.estimate_s = self.estimate(r)
        return errors

    def capture_ready(self, runs: list[Run]) -> None:
        """Runs whose last dependency just finished: their real length, for the order and log."""
        waiting = [r for r in runs if r.total is None and r.command in ISING_COMMANDS]
        if not waiting:
            return
        for error in self.capture(waiting):
            log(f"the CLI refuses {error}")
        for r in waiting:
            if r.estimate_s is not None and r.total is not None:
                log(f"{r.name}: {r.total:,} sweeps, about {r.estimate_s / 3600:.2f} h")

    def estimate(self, r: Run) -> float | None:
        """Seconds the run should take at the preflight's rate, start-up and closing pass in."""
        rate = self.rates.get((r.shape[0], r.shape[1])) if r.shape else None
        if not rate or r.total is None:
            return None
        prior = self.prior.get(r.name, {})
        left = 0 if prior.get("state") == "complete" else r.total - min(r.total, r.step)
        return left / rate * LOOP_OVERHEAD + RUN_OVERHEAD_S

    def preflight(self) -> None:
        shapes = sorted({(r.shape[0], r.shape[1]) for r in self.runs if r.shape})
        if not shapes:
            log("no simulator run to time: jepa runs report their own rates")
            return
        gpu0 = dict(self.env, CUDA_VISIBLE_DEVICES="0")
        sweeps, device = self.ctx.bench_sweeps, self.ctx.device
        bench = self.helper(
            "bench", gpu0, shapes=[list(s) for s in shapes], sweeps=sweeps, device=device
        )
        self.summary["preflight"] = bench
        self.rates = {(b["L"], b["lattices"]): b["sweeps_per_s_with_totals"] for b in bench}
        for r in self.runs:
            r.estimate_s = self.estimate(r)
        overheads = f"x{LOOP_OVERHEAD} for the run loop, +{RUN_OVERHEAD_S} s a run"
        log(f"budget at the preflight rates ({overheads}):")
        known = 0.0
        for r in self.runs:
            if r.command not in ISING_COMMANDS:
                continue
            L, lattices = r.shape or [0, 0]
            head = f"  {r.name:<18} {r.command:<9} L={L:<3} x{lattices:<4}"
            if r.estimate_s is None or r.total is None:
                log(f"{head} length known once {', '.join(r.deps) or 'it starts'} is done")
                continue
            known += r.estimate_s
            rate = self.rates[(L, lattices)]
            log(f"{head} {r.total:>11,} sweeps at {rate:,.0f}/s -> {r.estimate_s / 3600:.2f} h")
        left = self.ctx.stop_s - (time.time() - T0)
        log(
            f"  known so far: {known / 3600:.2f} GPU hours on {self.gpus} GPU(s), about "
            f"{known / self.gpus / 3600:.2f} h each; runs stop {left / 3600:.2f} h from now"
        )
        self.summary["projection"] = {"known_gpu_hours": round(known / 3600, 3), "gpus": self.gpus}

    def budget(self) -> str | None:
        used = tree_bytes(self.ctx.work)
        self.summary["output_bytes"] = used
        if used < self.ctx.cap_bytes:
            return None
        return (
            f"output cap: {used / 1e9:.1f} GB under {self.ctx.work}, "
            f"the cap is {self.ctx.cap_bytes / 1e9:g} GB"
        )

    def execute(self, gpu: int, run: Run) -> tuple[str, str]:
        """One run on one GPU: reuse the previous session's finished one, continue its unfinished
        one (`ising resume`, `jepa resume`, or the same extraction again), or start it."""
        prefix = f"[gpu{gpu} {run.name}] "
        try:
            tail, run.key = self.resolve(run)
        except RunError as exc:
            return "failed", str(exc)
        found, how = self.adopt(run)
        if found is not None and how == "reuse":
            where = found.relative_to(self.ctx.out).as_posix()
            log(f"{prefix}complete already: {where}, from the previous session")
            return self.verify(run, found, prefix, "reused from the previous session's output")
        return self.launch(gpu, run, self.command_line(run, tail, found), found, prefix)

    def command_line(self, run: Run, tail: list[str], unfinished: Path | None) -> list[str]:
        """The child's argv: the command, or the resume of its unfinished folder. An extraction
        resumes by running again, keeping the chunks it finished."""
        module, sub = module_of(run.command)
        base = [self.ctx.py, "-I", "-u", "-m", f"tipping_point.{module}"]
        if unfinished is not None and run.command != "jepa extract":
            device = ["--device", self.ctx.device] if module == "jepa" else []
            return [*base, "resume", str(unfinished), *device]
        return [*base, sub, *tail, *self.flags(run, self.ctx.out)]

    def extract_paths(self, run: Run) -> tuple[Path, Path]:
        """The checkpoint an extraction reads and the simulator run it writes in."""
        ckpt = target = None
        for arg in run.args:
            if not isinstance(arg, dict):
                continue
            if "dataset_run" in arg:
                target = self.ctx.out / "ising" / arg["dataset_run"]
            elif "input" in arg:
                ckpt = self.inputs[arg["input"]]
            elif self.by_name[arg["run"]].command in ISING_COMMANDS:
                target = self.ctx.out / str(self.by_name[arg["run"]].folder)
            else:
                ckpt = self.ctx.out / str(self.by_name[arg["run"]].folder) / arg.get("file", "")
        if ckpt is None or target is None:
            raise RunError("jepa extract reads a checkpoint and a simulator run: one of each")
        return ckpt, target

    def latents(self, run: Run) -> Path | None:
        """An extraction's latents table: the folder in its simulator run whose manifest names
        its checkpoint by sha256 (a run holds one table per checkpoint), else None."""
        ckpt, target = self.extract_paths(run)
        if not ckpt.is_file():
            return None
        digest = sha256_file(ckpt)
        for manifest in sorted((target / "latents").glob("*/manifest.json")):
            params = (read_json(manifest) or {}).get("params") or {}
            if (params.get("checkpoint") or {}).get("sha256") == digest:
                return manifest.parent
        return None

    def adopt(self, run: Run) -> tuple[Path | None, str | None]:
        """The previous session's folder for this run, carried into this output, when it was made
        from the same arguments by the same commit, torch and device: (folder, reuse or resume).
        An extraction's is the latents table that names its checkpoint."""
        prior = self.prior.get(run.name)
        if prior is None:
            return None, None
        extract = run.command == "jepa extract"
        folder = self.latents(run) if extract else self.ctx.out / prior["folder"]
        manifest = (read_json(folder / "manifest.json") if folder is not None else None) or {}
        env = manifest.get("env") or {}
        made = [
            (manifest.get("producer") or {}).get("git_sha"),
            env.get("torch"),
            env.get("device"),
        ]
        here = [self.commit, self.torch, self.ctx.device]
        state = (
            "manifest.json" if extract else "checkpoint.pt"
        )  # what an unfinished one resumes from
        if prior.get("key") != run.key:
            why = "it was made from other arguments"
        elif folder is None or not folder.is_dir():
            why = "its folder is not in this output"
        elif made != here:
            why = f"it was made by commit, torch and device {made}, not {here}"
        elif not ((folder / "complete.json").is_file() or (folder / state).is_file()):
            why = f"it holds neither complete.json nor {state}"
        else:
            why = None
        if why is not None or folder is None:
            log(f"[{run.name}] {prior['folder']} is not reused: {why}; starting afresh")
            if folder is not None:  # superseded: no place in this output
                shutil.rmtree(folder, ignore_errors=True)
            run.folder = run.run_id = None
            run.step, run.wall_s = 0, 0.0
            return None, None
        run.folder, run.run_id = folder.relative_to(self.ctx.out).as_posix(), folder.name
        if (folder / "complete.json").is_file():
            run.reused = True
            return folder, "reuse"
        run.resumed = True
        return folder, "resume"

    def launch(
        self, gpu: int, run: Run, argv: list[str], run_dir: Path | None, prefix: str
    ) -> tuple[str, str]:
        env = dict(self.env, CUDA_VISIBLE_DEVICES=str(gpu))
        domain = {"jepa train": "jepa", "jepa extract": ""}.get(run.command, "ising")
        parent = self.ctx.out / domain
        tail: collections.deque[str] = collections.deque(maxlen=60)
        clock = time.time()
        fresh = run_dir is None and domain != ""
        if run.command == "jepa extract" and run_dir is None:  # its run carries partial latents
            run.folder = self.extract_paths(run)[1].relative_to(self.ctx.out).as_posix()
        with self.start_locks[domain] if fresh else contextlib.nullcontext():
            before = set(os.listdir(parent)) if fresh else set()
            with self.active_lock:  # a run never starts after the deadline has stopped the others
                if self.deadline_hit:
                    return "not started", "session deadline"
                log(prefix + "$ " + " ".join(argv))
                proc = subprocess.Popen(
                    argv,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                    cwd=self.ctx.cwd,
                )
                self.active[run.name] = proc
            pump = threading.Thread(target=self.pump, args=(proc, run, tail, prefix), daemon=True)
            pump.start()
            if fresh:
                run_dir = self.new_folder(parent, before, proc)
                run.run_id = None if run_dir is None else run_dir.name
                run.folder = None if run_dir is None else f"{domain}/{run_dir.name}"
                if run_dir is not None:
                    self.save()
                    # the next start falls in a later second, since a run id names its second
                    time.sleep(1)
        try:
            if fresh and run_dir is None and proc.poll() is None:  # no folder after NEW_FOLDER_S
                log(f"{prefix}no run folder after {NEW_FOLDER_S} s; stopping it")
                proc.kill()
            code = proc.wait()
            pump.join()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            with self.active_lock:
                self.active.pop(run.name, None)
        run.exit = code
        run.wall_s += time.time() - clock
        log(f"{prefix}exit {code} after {time.time() - clock:.0f} s")
        return self.settle(run, run_dir, code, tail, prefix)

    def new_folder(self, parent: Path, before: set[str], proc: subprocess.Popen) -> Path | None:
        """The folder a fresh run makes: the one new run id in out/ising or out/jepa. Starts there
        take turns (that place's start lock), so it is this run's."""
        clock = time.time()
        while time.time() - clock < NEW_FOLDER_S:
            exited = proc.poll() is not None
            new = sorted(n for n in os.listdir(parent) if n not in before and RUN_ID.fullmatch(n))
            if new:
                if len(new) > 1:
                    log(f"WARNING: several new run folders {new}; taking the last")
                return parent / new[-1]
            if exited:
                return None
            time.sleep(0.2)
        return None

    def pump(self, proc: subprocess.Popen, run: Run, tail: collections.deque, prefix: str) -> None:
        """The child's output into its log, its progress into the run's record."""
        assert proc.stdout is not None
        try:
            with open(self.ctx.out / "logs" / f"{run.name}.log", "a", encoding="utf-8") as sink:
                for raw in proc.stdout:
                    line = raw.rstrip("\n")
                    tail.append(line)
                    sink.write(line + "\n")
                    sink.flush()
                    found = PROGRESS.search(line)
                    if found:
                        step, total, rate = (
                            int(found[k].replace(",", "")) if found[k] else None
                            for k in ("step", "total", "rate")
                        )
                        run.step, run.total, run.sweeps_per_s = step or 0, total, rate
                        if rate and step is not None and total is not None:
                            line += f"   [ETA {(total - step) / rate / 3600:.2f} h]"
                        if self.ctx.interrupt and any(r.state == "complete" for r in self.runs):
                            self.deadline("session deadline, simulated by the dry run")
                    log(prefix + line)
        except Exception as exc:  # keep draining: a full pipe would stall the child
            log(f"{prefix}output reader: {exc!r}")
            for _ in proc.stdout:
                pass

    def settle(
        self, run: Run, run_dir: Path | None, code: int, tail: collections.deque, prefix: str
    ) -> tuple[str, str]:
        """The run's state from its exit code and its files, never from its output text."""
        killed = self.deadline_hit and code != 0
        state = "checkpoint.pt"  # what an unfinished run resumes from
        if run.command == "jepa extract":  # its table, found by its checkpoint
            run_dir, state = self.latents(run), "manifest.json"
        if run_dir is not None and (run_dir / "complete.json").is_file() and (code == 0 or killed):
            return self.verify(run, run_dir, prefix)
        if killed:
            if run_dir is not None and (run_dir / state).is_file():
                return "interrupted", "stopped at the session deadline; resumable"
            return "not started", "session deadline (stopped before it saved any state)"
        run.output_tail = list(tail)[-40:]
        last = next((line for line in reversed(tail) if line.strip()), "no output")
        if run_dir is not None:
            made = ""
        elif run.command == "jepa extract":
            made = " with no finished latents table naming its checkpoint"
        else:
            made = " before it made a run folder"
        return "failed", f"exit {code}{made}: {last}"

    def verify(self, run: Run, run_dir: Path, prefix: str, note: str = "") -> tuple[str, str]:
        if run.command == "pilot" and not (run_dir / "tau.json").is_file():
            # `ising pilot` writes tau.json after complete.json; `ising resume` never does
            log(f"{prefix}no tau.json, as after a resume: writing it as `ising pilot` would")
            self.helper("tau_json", run_dir=str(run_dir), seconds=round(run.wall_s, 1))
        run.folder, run.run_id = run_dir.relative_to(self.ctx.out).as_posix(), run_dir.name
        problems, run.dropped = verdicts(run_dir, self.commit, run.command)
        argv = [self.ctx.py, "-I", "-u", "-m", "tipping_point.jepa", "verify", run_dir]
        code, lines = stream(
            argv, env=self.env, cwd=self.ctx.cwd, check=False, prefix=prefix + "[verify] "
        )
        problems += file_problems(code, lines, [run_dir])
        run.problems = problems
        if problems:
            log(f"{prefix}FAILED its checks: {problems}")
            return "failed", "checks: " + "; ".join(problems[:3])
        if run.total:
            run.step = run.total
        done = read_json(run_dir / "complete.json") or {}
        if run.command == "jepa train" and done.get("stopped_early"):  # extract refuses final.pt
            collapse = done.get("collapse") or {}
            early = f"stopped early at step {done.get('last_key')}"
            note = "; ".join(
                filter(None, [note, f"{early}, collapsed: {collapse.get('collapsed')}"])
            )
        if run.dropped:
            log(f"{prefix}thermalised missed, which the scorer drops and counts: {run.dropped}")
        log(f"{prefix}complete and verified: {run.folder}")
        return "complete", note

    def measured(self) -> dict[str, dict]:
        """The rates the plan's jepa runs measured. A training's seconds per step after warm-up,
        from its own ledger lines: only lines that continue the one before them in one process,
        so each session's first line, which holds start-up, data loading and the twin's PCA, is
        left out (and so is the closing line, which adds no step). An extraction's microseconds
        per frame (a latents row), by checkpoint and by the L its table's manifest records, since
        a frame's cost grows with L: the readout from the table's own clock (the manifest's
        started_ns, written once the model is loaded, to complete.json's ended_ns; tables finished
        in one session only), and the whole process, start-up included."""
        ledger = []
        path = self.ctx.out / "jepa" / LEDGER
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(ValueError):
                    ledger.append(json.loads(line))
        train: dict[str, dict] = {}
        extract: dict[str, dict] = {}
        for r in self.runs:
            if r.state != "complete" or r.folder is None:
                continue
            done = read_json(self.ctx.out / r.folder / "complete.json") or {}
            if r.command == "jepa train":
                mine = [x for x in ledger if x.get("run_id") == r.run_id]
                lines = sorted(mine, key=lambda x: x.get("to_ns", 0))
                steady = [
                    (b["to_ns"] - b["from_ns"], b["step"] - a["step"])
                    for a, b in itertools.pairwise(lines)
                    if b.get("from_ns") == a.get("to_ns") and b.get("step", 0) > a.get("step", 0)
                ]
                if steady:
                    seconds, steps = sum(t for t, _ in steady) / 1e9, sum(n for _, n in steady)
                    train[r.name] = {
                        "s_per_step": round(seconds / steps, 5),
                        "steps": steps,
                        "stopped_early": bool(done.get("stopped_early")),
                    }
            elif r.command == "jepa extract":
                ckpt = next(
                    (a for a in r.args if isinstance(a, dict) and ("file" in a or "input" in a)), {}
                )
                manifest = read_json(self.ctx.out / r.folder / "manifest.json") or {}
                size = ((manifest.get("params") or {}).get("dataset") or {}).get("L")
                group = extract.setdefault(
                    " ".join(
                        filter(
                            None,
                            (
                                ckpt.get("run", ckpt.get("input", "?")),
                                ckpt.get("file"),
                                f"L{size}" if size is not None else None,
                            ),
                        )
                    ),
                    {
                        "tables": 0,
                        "frames": 0,
                        "seconds": 0.0,
                        "readout_frames": 0,
                        "readout_s": 0.0,
                    },
                )
                rows = int((done.get("rows") or {}).get("latents") or 0)
                group["tables"] += 1
                group["frames"] += rows
                group["seconds"] += r.wall_s
                began = manifest.get("started_ns")
                if (
                    not r.resumed
                    and isinstance(began, int)
                    and isinstance(done.get("ended_ns"), int)
                ):
                    group["readout_frames"] += rows
                    group["readout_s"] += (done["ended_ns"] - began) / 1e9
        for group in extract.values():
            frames, readout = group["frames"], group["readout_frames"]
            group["us_per_frame"] = round(1e6 * group["seconds"] / frames, 2) if frames else None
            group["readout_us_per_frame"] = (
                round(1e6 * group["readout_s"] / readout, 2) if readout else None
            )
            group["seconds"], group["readout_s"] = (
                round(group["seconds"], 1),
                round(group["readout_s"], 2),
            )
        return {"train": train, "extract": extract}

    def deadline(self, reason: str) -> None:
        with self.active_lock:
            if self.deadline_hit:
                return
            self.deadline_hit = True
            log(f"SESSION DEADLINE ({reason}): every run stops now and keeps its last checkpoint")
            for name, proc in self.active.items():
                log(f"  stopping {name}")
                proc.kill()
        self.scheduler.stop(reason)

    def tick(self) -> None:
        if time.time() - T0 > self.ctx.stop_s:
            self.deadline("session deadline")
        if time.time() - self.beat > HEARTBEAT_S:
            self.beat = time.time()
            for r in self.runs:
                if r.state == "running" and r.total:
                    rate = f"{r.sweeps_per_s or 0:,} sweeps/s"
                    log(f"[gpu{r.gpu_used} {r.name}] at step {r.step:,} of {r.total:,}, {rate}")
            self.save()

    def save(self, final: bool = False) -> None:
        with self.save_lock:
            status, detail = verdict(self.runs, self.scheduler.stopped, final)
            rates = self.measured()
            if rates_text(rates):
                detail = f"{detail}; {rates_text(rates)}"
            self.summary.update(
                updated_utc=now_utc(),
                status=status,
                status_detail=detail,
                rates=rates,
                runs=[asdict(r) for r in self.runs],
            )
            write_text(self.ctx.out / STATE, json.dumps(self.summary, indent=2, default=str))
            write_text(self.ctx.out / "STATUS", f"{status}: {detail}\n")

    def finish(self) -> int:
        self.summary["ended_utc"] = now_utc()
        self.summary["seconds"] = round(time.time() - T0)
        self.save(final=True)
        status, detail = self.summary["status"], self.summary["status_detail"]
        banner = "=" * 100
        log(f"{banner}\nSTATUS {status.upper()}: {detail}\n{banner}")
        return exit_code(self.ctx)

    def shutdown(self) -> None:
        """Nothing keeps running on a GPU after an error."""
        self.scheduler.stop("shutdown")
        with self.active_lock:
            self.deadline_hit = True
            for proc in self.active.values():
                if proc.poll() is None:
                    proc.kill()


def rates_text(rates: dict[str, dict]) -> str:
    """The measured rates as STATUS gives them."""
    parts = []
    for name, t in rates["train"].items():
        early = ", stopped early" if t["stopped_early"] else ""
        parts.append(
            f"{name} {t['s_per_step']:.3f} s/step after warm-up ({t['steps']:,} steps{early})"
        )
    for ref, e in rates["extract"].items():
        if e["us_per_frame"] is None:
            continue
        readout = e["readout_us_per_frame"]
        head = f"{readout:.1f} us/frame readout, " if readout is not None else ""
        parts.append(
            f"extract {ref} {head}{e['us_per_frame']:.1f} with start-up "
            f"({e['tables']} tables, {e['frames']:,} frames)"
        )
    return "rates: " + "; ".join(parts) if parts else ""


def exit_code(ctx: Ctx) -> int:
    """Non-zero only when the output holds no run folder: a failed version may lose its output."""
    folders = [ctx.out / "ising", ctx.out / "jepa"]
    held = any(p.is_dir() for f in folders if f.is_dir() for p in f.iterdir())
    return 0 if held else 1


def refuse(ctx: Ctx, summary: dict, why: str) -> int:
    summary.update(status="refused", status_detail=why, updated_utc=now_utc(), ended_utc=now_utc())
    write_text(ctx.out / STATE, json.dumps(summary, indent=2, default=str))
    write_text(ctx.out / "STATUS", f"refused: {why}\n")
    banner = "=" * 100
    log(f"{banner}\nSTATUS REFUSED: {why}\n{banner}")
    return exit_code(ctx)


def kernel_main(ctx: Ctx) -> int:
    for folder in (
        ctx.out / "logs",
        ctx.out / "ising",
        ctx.out / "jepa",
        ctx.cwd,
        ctx.helper.parent,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    summary: dict = {"kernel": None, "started_utc": now_utc(), "device": ctx.device}
    session = None
    try:
        if PLAN_JSON is None:
            raise Refused("no plan is embedded: write the kernel folder with --write-kernel")
        plan = load_plan(json.loads(PLAN_JSON))
        summary.update(kernel=plan["kernel"], plan=plan)
        log(f"== {plan['kernel']}: {len(plan['runs'])} runs")
        log("== previous sessions (this kernel's output attached as an input)")
        prior = carry_forward(ctx, plan)
        if prior is not None:  # a refusal from here on still hands the chain on
            summary.update(prior={k: prior[k] for k in ("path", "status")}, runs=prior["runs"])
        log("== wheel and environment")
        wheel, summary["wheel"] = find_wheel(ctx, plan.get("wheel", DEFAULT_WHEEL))
        commit = summary["wheel"]["commit"]
        # a frozen prereg's scorer refuses any other commit's runs: say so before the quota goes
        if plan.get("commit", commit) != commit:
            raise Refused(f"COMMIT.txt names {commit}; this plan's runs need {plan['commit']}")
        env = summary["env"] = probe(ctx) if ctx.py else ensure_env(ctx, wheel)
        log(json.dumps(env, indent=2))
        check_env(ctx, env, commit)
        gpus = ctx.gpus or len(env["devices"])
        session = Session(ctx, plan, summary, commit, env["torch"], gpus, prior)
        return session.run()
    except (Refused, PlanError) as why:
        return refuse(ctx, summary, str(why))
    except Exception as exc:  # Session.run keeps errors once runs start: this is before any
        log(traceback.format_exc())
        return refuse(ctx, summary, f"the runner failed before any run started: {exc!r}")
    finally:
        if session is not None:
            session.shutdown()
            guarded("unstaging the plain copies of inputs", session.unstage)


# -- local: the kernel folder, the dry run, the self-test --------------------------------------


def cli_errors(plan: dict) -> list[str]:
    """The CLI's own verdict on every run's arguments, through the helper's capture run in this
    process. A run that takes a file is only parsed: the file comes into being on Kaggle."""
    helper: dict[str, Any] = {"__name__": "tp_kaggle_helper"}
    exec(compile(HELPER_SOURCE, "helper.py", "exec"), helper)
    with tempfile.TemporaryDirectory(prefix="tp-capture-") as scratch:
        items = []
        for spec in plan["runs"]:
            module, sub = module_of(spec["command"])
            args = [PLACEHOLDER if isinstance(a, dict) else str(a) for a in spec["args"]]
            data = [] if spec["command"] == "jepa extract" else ["--data-dir", scratch]
            argv = [sub, *args, "--device", "cpu", *data]
            items.append({"argv": argv, "partial": PLACEHOLDER in args, "module": module})
        results = helper["capture"](items)
    return [
        f"{spec['name']}: {got['error']}"
        for spec, got in zip(plan["runs"], results, strict=True)
        if "error" in got
    ]


def read_plan(path: Path) -> dict:
    return load_plan(json.loads(Path(path).read_text(encoding="utf-8")))


def only(plan: dict, names: list[str]) -> dict:
    """The plan cut down to the named runs and every run they read, in plan order."""
    by = {r["name"]: r for r in plan["runs"]}
    missing = [n for n in names if n not in by]
    if missing:
        raise PlanError(f"--only: the plan has no run named {', '.join(missing)}")
    keep: set[str] = set()
    todo = list(names)
    while todo:
        name = todo.pop()
        if name not in keep:
            keep.add(name)
            todo += [a["run"] for a in by[name]["args"] if isinstance(a, dict) and "run" in a]
    return dict(plan, runs=[r for r in plan["runs"] if r["name"] in keep])


def write_kernel(plan: dict, dest: Path, resume: bool = False) -> Path:
    """A pushable kernel folder: this script with the plan embedded, and kernel-metadata.json."""
    load_plan(plan)
    errors = cli_errors(plan)
    if errors:
        raise PlanError("the CLI refuses:\n" + "\n".join(errors))
    source = Path(__file__).read_text(encoding="utf-8")
    literal = repr(json.dumps(plan, indent=1))
    script, n = re.subn(
        r"^PLAN_JSON: str \| None = None\b",
        lambda _: f"PLAN_JSON: str | None = {literal}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(f"{__file__} has no PLAN_JSON line to fill")
    slug = plan["kernel"]
    own = [f"{OWNER}/{slug}"] if resume else []
    metadata = {
        "id": f"{OWNER}/{slug}",
        "title": slug,
        "code_file": f"{slug}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",  # the CLI's name for GPU T4 x2: one queue per T4
        "dataset_sources": [plan.get("wheel", DEFAULT_WHEEL), *plan.get("datasets", [])],
        "competition_sources": [],
        "kernel_sources": [*plan.get("kernel_sources", []), *own],
        "model_sources": [],
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{slug}.py").write_text(script, encoding="utf-8", newline="\n")
    text = json.dumps(metadata, indent=2) + "\n"
    (dest / "kernel-metadata.json").write_text(text, encoding="utf-8", newline="\n")
    return dest / f"{slug}.py"


def kaggle_noise(python: Path) -> Path:
    """Make a local environment's children print what every child prints on Kaggle's image (#54),
    by the same means: a sitecustomize that fails, which -I does not skip. Its site-packages."""
    site = subprocess.run(
        [str(python), "-I", "-c", "import site; print(site.getsitepackages()[-1])"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    failing = "raise ModuleNotFoundError(\"No module named 'wrapt'\")\n"
    (Path(site) / "sitecustomize.py").write_text(failing, encoding="utf-8")
    return Path(site)


def build_wheel_env(root: Path) -> str:
    """The dry run's tipping-point-wheel and pinned environment: the wheel built from HEAD in a
    clean clone, as the Kaggle wheel is built, with its COMMIT.txt, and a venv of the locked CPU
    dependencies plus that wheel (harness/wheel_provenance.py's). Its python."""
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "harness"))
    import wheel_provenance as wp

    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    clone = root / "src"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(clone)], check=True
    )
    subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", "--detach", head], check=True)
    wheels = root / "input" / WHEEL_DATASET
    wp._uv(
        "build",
        "--package",
        "tipping-point",
        "--wheel",
        "--out-dir",
        str(wheels),
        "--quiet",
        cwd=clone,
    )
    (wheel,) = wheels.glob("tipping_point-*.whl")
    stamp = wp._stamp(wheel)
    note = f"{stamp['commit']}{'-dirty' if stamp['dirty'] else ''}\n"
    (wheels / "COMMIT.txt").write_text(note, encoding="utf-8")
    log(f"wheel from HEAD {head[:12]} in a clean clone: {stamp}")
    requirements = root / "requirements.txt"
    wp._uv(
        *("export", "--frozen", "--extra", "cpu", "--no-dev", "--no-emit-workspace"),
        *("--no-hashes", "--no-header", "--no-annotate", "--quiet", "-o", str(requirements)),
    )
    python = wp._venv(root / "env", wheel, requirements)
    kaggle_noise(python)
    return str(python)


def dry_root(args: argparse.Namespace, prefix: str) -> Path:
    if args.root is not None and args.root.exists() and any(args.root.iterdir()):
        raise SystemExit(f"--root {args.root} is not empty: a dry run needs a folder of its own")
    return (args.root or Path(tempfile.mkdtemp(prefix=prefix))).resolve()


def dry_run(args: argparse.Namespace) -> int:
    plan = read_plan(Path(args.dry_run))
    if args.only:
        plan = only(plan, args.only.split(","))
    root = dry_root(args, "tp-dry-run-")
    log(
        f"dry run of {args.dry_run} ({', '.join(r['name'] for r in plan['runs'])}) under {root}: "
        f"L / {args.shrink}, sweep counts / {args.shrink**2}, {args.gpus} queues on the CPU"
    )
    python = build_wheel_env(root)
    for spec in args.input:  # what Kaggle would attach: SLUG=DIR
        slug, _, folder = spec.partition("=")
        shutil.copytree(folder, root / "input" / slug)
    return rehearse(plan, root, python, args.gpus, args.shrink)


def dry_run_jepa(args: argparse.Namespace) -> int:
    root = dry_root(args, "tp-dry-jepa-")
    log(f"JEPA dry run under {root}: tiny simulator runs, {args.gpus} queues on the CPU")
    python = build_wheel_env(root)
    return rehearse(tiny_jepa_plan(root), root, python, args.gpus, 1)


def tiny_jepa_plan(root: Path) -> dict:
    """A JEPA plan over tiny simulator runs written as an attached dataset (harness/jepa_ising.py's
    staircase: three plateaus of eight bursts at L = 16, four replicas, in seconds on a CPU; a null
    run beside them), with the repository's copy of the #50 pilot record as a kernel output. A
    quick simulator run finishes first, so session 1's simulated deadline falls on the training,
    which session 2 resumes; then its final.pt and its step-0 twin are extracted and verified."""
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "harness"))
    import jepa_ising
    from tipping_point.ising import protocol
    from tipping_point.ising import run as simulator

    scratch, dataset = root / "tiny-data", root / "input" / "tp-tiny-data"
    dataset.mkdir(parents=True)
    made = {}
    null = protocol.Fixed((2.5,), therm=50, sweeps=400, saves=4)
    for tag, seed, schedule in (
        ("L16-qs-s101", 101, jepa_ising.staircase()),
        ("L16-qs-s102", 102, jepa_ising.staircase()),
        ("L16-qs-s1", 1, jepa_ising.staircase()),
        ("L16-null-e2-s1", 1, null),
    ):
        spec = simulator.Spec(L=16, replicas=4, seed=seed, checkpoint_every=jepa_ising.FLUSH)
        folder = simulator.start(spec, schedule, scratch, tag, "cpu")
        os.replace(folder, dataset / folder.name)
        made[tag] = folder.name
    record_id = "20261022T182733Z-tauc-L16-32-64"  # the #50 record the fixture copies
    record = root / "input" / "tp-tauc-l64-l128" / "out" / "ising" / record_id / "pilot.json"
    record.parent.mkdir(parents=True)
    shutil.copyfile(jepa_ising.PILOT, record)
    train = ["--L", 16, "--train-seeds", 101, "--val-seed", 102, "--seed", 3]
    train += ["--pilot", {"input": "record"}, "--steps", 40, "--batch-sites", 2048]
    train += ["--checkpoint-every", 5, "--monitor-every", 5]
    return load_plan(
        {
            "kernel": "tp-jepa-dry-run",
            "description": "--dry-run-jepa's tiny plan",
            "datasets": ["local/tp-tiny-data"],
            "kernel_sources": [f"{OWNER}/tp-tauc-l64-l128"],
            "inputs": {
                "record": {"run": record_id, "file": "pilot.json", "sha256": sha256_file(record)}
            },
            "runs": [
                {
                    "name": "quick",
                    "command": "pilot",
                    "gpu": 1,
                    "args": ["--L", 8, "--T", 2.5, "--replicas", 4, "--therm", 20, "--sweeps", 200],
                },
                {
                    "name": "tiny",
                    "command": "jepa train",
                    "gpu": 0,
                    "stage": [made["L16-qs-s101"], made["L16-qs-s102"]],
                    "args": train,
                },
                {
                    "name": "x-twin",
                    "command": "jepa extract",
                    "gpu": 1,
                    "args": [
                        {"run": "tiny", "file": TWIN},
                        {"dataset_run": made["L16-qs-s1"]},
                        "--select",
                        "tc-plus",
                    ],
                },
                {
                    "name": "x-final",
                    "command": "jepa extract",
                    "gpu": 1,
                    "args": [
                        {"run": "tiny", "file": FINAL},
                        {"dataset_run": made["L16-null-e2-s1"]},
                    ],
                },
            ],
        }
    )


def rehearse(
    plan: dict, root: Path, python: str, gpus: int, shrink_by: int, sink: Any = None
) -> int:
    """Two local sessions of the written kernel: the first stopped by a simulated deadline at the
    first progress line after a run finishes, the second with the first's output attached. Their
    output goes to sink, or to this process's own."""
    script = write_kernel(plan, root / "kernel", resume=True)
    sessions = []
    for n in (1, 2):
        if n == 2:  # session 1's output, attached as kernel_sources attaches it
            os.replace(root / "working", root / "input" / plan["kernel"])
        log(f"== dry-run session {n}" + (", stopped by a simulated deadline" if n == 1 else ""))
        argv = [sys.executable, script, "--local-session", "--root", root, "--python", python]
        argv += ["--gpus", gpus, "--shrink", shrink_by, *(["--interrupt"] if n == 1 else [])]
        ran = subprocess.run([str(a) for a in argv], check=False, stdout=sink, stderr=sink)
        sessions.append((ran.returncode, read_json(root / "working" / "out" / STATE) or {}))
    return dry_run_report(sessions, root, python)


def dry_run_report(sessions: list[tuple[int, dict]], root: Path, python: str) -> int:
    print("\n" + "=" * 100)
    for n, (code, state) in enumerate(sessions, 1):
        print(
            f"session {n}: exit {code}; STATUS {state.get('status')}: {state.get('status_detail')}"
        )
        for r in state.get("runs", []):
            how = " (reused)" if r["reused"] else " (resumed)" if r["resumed"] else ""
            where = r.get("folder") or "-"
            print(f"  {r['name']:<14} {r['state']:<12}{how:<10} {where:<44} {r['note']}")
    (_, first), (code, last) = sessions
    runs = {r["name"]: r for r in last.get("runs", [])}
    folders = {r["name"]: r.get("folder") for r in last.get("runs", [])}
    reads = [
        f"{r['name']} read {folders[d]}"
        for r in last.get("runs", [])
        for d in r["deps"]
        if folders[d] and any(str(k).startswith(str(folders[d])) for k in r["key"] or [])
    ]
    needed = sum(len(r["deps"]) for r in last.get("runs", []))
    stopped = [r["name"] for r in first.get("runs", []) if r["state"] == "interrupted"]
    dirs = [str(root / "working" / "out" / where) for where in folders.values() if where]
    verify = [python, "-I", "-m", "tipping_point.jepa", "verify", *dirs]
    fresh = subprocess.run(verify, capture_output=True, text=True, check=False)
    findings = {
        "session 1 finished a run": any(r["state"] == "complete" for r in first.get("runs", [])),
        "session 1 left a resumable run": bool(stopped),
        "session 2 is complete": last.get("status") == "complete" and code == 0,
        "session 2 reused a finished run": any(r["reused"] for r in runs.values()),
        "session 2 resumed every interrupted run": all(runs[n]["resumed"] for n in stopped),
        "every reference read its run's folder": len(reads) == needed,
        f"a fresh jepa verify passes on all {len(dirs)} runs' folders": fresh.returncode == 0,
    }
    for what, ok in findings.items():
        print(f"{'PASS' if ok else 'FAIL'}  {what}")
    for line in reads:
        print(f"      {line}")
    print(f"dry run {'OK' if all(findings.values()) else 'FAILED'}; everything is under {root}")
    return 0 if all(findings.values()) else 1


def local_ctx(args: argparse.Namespace) -> Ctx:
    root = Path(args.root)
    return Ctx(
        input=root / "input",
        work=root / "working",
        tmp=root / "tmp",
        device="cpu",
        poll_s=0.2,
        bench_sweeps=50,
        py=args.python,
        gpus=args.gpus,
        shrink=args.shrink,
        interrupt=args.interrupt,
    )


def self_test() -> int:
    """Plan validation and refusal, dependency resolution, the queues, the previous session's
    runs, inputs, the wheel checks, run verification and the plans in notebooks/plans, then real
    jepa train, resume and extract runs through the written kernel. The checks live beside this
    file (kaggle_ising_runs_selftest.py), so the pushed kernel stays lean."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kaggle_ising_runs_selftest import run_all

    return run_all(sys.modules[__name__])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write-kernel", nargs=2, metavar=("PLAN", "DIR"), help="write a pushable kernel folder"
    )
    mode.add_argument(
        "--dry-run", metavar="PLAN", help="execute PLAN on the CPU in a temporary folder"
    )
    mode.add_argument(
        "--dry-run-jepa",
        action="store_true",
        help="rehearse a tiny JEPA plan on the CPU: train, resume, extract, verify",
    )
    mode.add_argument("--self-test", action="store_true", help="check the runner in-process")
    mode.add_argument("--local-session", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume", action="store_true", help="--write-kernel: attach this kernel's last output"
    )
    parser.add_argument(
        "--shrink", type=int, default=8, help="--dry-run: L / SHRINK, sweeps / SHRINK^2"
    )
    parser.add_argument(
        "--only", metavar="NAMES", help="--dry-run: these runs (comma-separated) and what they read"
    )
    parser.add_argument("--gpus", type=int, default=2, help="--dry-run: how many queues")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="SLUG=DIR",
        help="--dry-run: attach DIR as the dataset or kernel output SLUG",
    )
    parser.add_argument(
        "--root", type=Path, help="--dry-run: work here, not in a new temporary folder"
    )
    parser.add_argument("--python", help=argparse.SUPPRESS)
    parser.add_argument("--interrupt", action="store_true", help=argparse.SUPPRESS)
    args, unknown = parser.parse_known_args(argv)
    chosen = (
        args.write_kernel
        or args.dry_run
        or args.dry_run_jepa
        or args.self_test
        or args.local_session
    )
    if unknown and chosen:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    try:
        if args.write_kernel:
            plan, dest = args.write_kernel
            script = write_kernel(read_plan(Path(plan)), Path(dest), args.resume)
            print(f"wrote {script} and {script.parent / 'kernel-metadata.json'}")
            print(f"push it with: kaggle kernels push -p {script.parent}")
            return 0
        if args.dry_run:
            return dry_run(args)
        if args.dry_run_jepa:
            return dry_run_jepa(args)
    except PlanError as why:
        print(f"the plan is refused:\n{why}", file=sys.stderr)
        return 2
    if args.self_test:
        return self_test()
    if args.local_session:
        return kernel_main(local_ctx(args))
    return kernel_main(kaggle_ctx())


if __name__ == "__main__":
    code = main()
    if code:  # only on failure: a notebook runner can count SystemExit(0) as an error
        sys.exit(code)
