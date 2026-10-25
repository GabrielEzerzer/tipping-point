"""Tipping Point: an explicit plan of `ising` runs on Kaggle's GPUs. A Kaggle script kernel.

A plan is plain JSON: an ordered list of runs, each an `ising` subcommand with its arguments.
--write-kernel embeds a plan in a copy of this file and writes kernel-metadata.json beside it, and
`kaggle kernels push -p <dir>` runs it. The machinery is tp-tauc-l64-l128's (the kernel that ran
on Kaggle's T4 x2, kept under .wf/kaggle/): the pinned environment from the tipping-point-wheel
dataset with its COMMIT.txt check, the preflight benchmark, a queue per GPU through
CUDA_VISIBLE_DEVICES, the stop at 8 h 15 min, checkpoints and `ising resume` across sessions,
STATUS, and the exit-code policy.

    {"kernel": "tp-ising-job-a",                  the kernel slug: <owner>/<slug> and its title
     "description": "...",                        optional
     "datasets": ["<owner>/<slug>"],              optional: datasets to attach
     "kernel_sources": ["<owner>/<slug>"],        optional: other kernels' outputs to attach
     "inputs": {"record": {"run": "<run id>", "file": "pilot.json", "sha256": "<hex>"}},
     "runs": [
       {"name": "grid-L64", "command": "pilot", "args": ["--L", 64, "--seed", 6]},
       {"name": "qs-L64-s11", "command": "staircase", "gpu": 1,
        "args": ["--L", 64, "--seed", 11, "--tau", {"run": "grid-L64", "file": "tau.json"}]},
       {"name": "null-L16-s1", "command": "null",
        "args": ["--L", 16, "--epsilon-L", 2, "--pilot", {"input": "record"}]}]}

- command is pilot, staircase, ramp or null; args holds its flags and values, one item each. The
  runner adds --device and --data-dir.
- {"run": <name>, "file": <path>} is that earlier run's folder plus <path>, resolved when the run
  starts (run ids are timestamped). The run waits until that one is complete and verified.
- {"input": <name>} is the file an `inputs` entry names, looked up in the attached datasets and
  kernel outputs (in a <run id> folder or a <run id>.zip) and checked against its sha256 before
  any run starts. A pilot record for --pilot is the use.
- gpu pins a run to one GPU. The others go to whichever GPU is free, longest first, by the CLI's
  own schedule length at the preflight's measured rate.

From the repository root:

    uv run --no-sync python notebooks/kaggle_ising_runs.py --write-kernel <plan> <dir> [--resume]
    uv run --no-sync python notebooks/kaggle_ising_runs.py --dry-run <plan> [--only <names>]
    uv run --no-sync python notebooks/kaggle_ising_runs.py --self-test

--resume attaches the kernel's own previous output (kernel_sources). Push a version with it after
a session stopped at the deadline, and that version reuses the finished runs and continues the
unfinished ones with `ising resume`. Every session first copies the previous output's runs into
its own, so each version's output holds the whole plan so far.

--dry-run really executes the plan on the CPU in a temporary folder, with sizes divided by
--shrink (8: L = 4 and 8 for Job A) and sweep counts by its square. It builds a wheel from HEAD in a
clean clone (the code the Kaggle wheel would hold) and runs two sessions of the written kernel.
The first is stopped by a simulated deadline at the first progress line after a run finishes. The
second runs with the first's output attached, as Kaggle attaches it. --only keeps the named runs
and the runs they read: a staircase takes at least 101,000 sweeps whatever its L, since every
plateau holds 500 + 2,000, so a busy CPU wants one. --input SLUG=DIR attaches a folder as a
dataset or kernel output.

On Kaggle every run lands in /kaggle/working/out/ising/<run id>/, its output in
out/logs/<name>.log (`kaggle kernels output` fetches out/). Before any run starts, the wheel must
have a COMMIT.txt naming a commit not marked dirty, and the installed package must name the same
clean commit. After each run, the runner checks: exit 0; complete.json present, clean, with every
check passed; every file complete.json lists intact (`jepa verify`); and a manifest whose git_sha
is COMMIT.txt's, with dirty false. Runs are judged by exit codes, files and manifests, never by
output text, because Kaggle's image prints sitecustomize errors from every child process. A
failed run stops its GPU's queue and blocks the runs that read it; the other queues go on. No run
starts once /kaggle/working holds 18 GB (Kaggle keeps 20 GB) or past 8 h 15 min.

Status: out/STATUS holds one line (running, complete, incomplete, failed or refused), and
out/kaggle-ising-runs.json holds every run's state, the plan, the environment and the preflight.
The kernel exits non-zero only when its output holds no run folder, since a failed version may
not keep its output.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import hashlib
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
PYTHON = "3.13"  # the wheel's Requires-Python
TORCH = "torch==2.14.1+cu126"
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
PINNED = ("numpy==2.5.4", "pyarrow==26.0.0")  # uv.lock's versions
# Kaggle's image puts a sitecustomize that imports wrapt on the path of its /usr/bin/python3, which
# is the venv's interpreter, so -I does not skip it: without wrapt every child prints two error
# lines (tp-tauc-l64-l128's log). Runs never depend on it.
SITE_NOISE = ("wrapt==1.17.3",)
COMMANDS = ("pilot", "staircase", "ramp", "null")
RUNNER_FLAGS = ("--device", "--data-dir", "-h", "--help")
SESSION_S = 9 * 3600  # Kaggle's session cap
STOP_S = SESSION_S - 45 * 60  # no run starts, and running ones stop, past this
HEARTBEAT_S = 600
LOOP_OVERHEAD = 1.10  # the run loop's schedule logic and Parquet flushes over the bare kernel
RUN_OVERHEAD_S = 60  # start-up, and a pilot's closing tau.json pass
BENCH_SWEEPS = 2_000
NEW_FOLDER_S = 600  # a fresh run makes its folder within this, or it is stopped
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
PROGRESS = re.compile(
    r"(?P<run>\d{8}T\d{6}Z-[A-Za-z0-9_-]+): step (?P<step>[\d,]+)/(?P<total>[\d,]+) "
    r"\((?P<rate>[\d,]+) sweeps/s\)"
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
    still waiting on an earlier run (partial) is only parsed."""
    return [_capture(item["argv"], item["partial"]) for item in items]


def _capture(argv, partial):
    seen = {}

    def stub(spec, schedule, root, tag, device=None, pilot=None):
        seen.update(L=spec.L, lattices=spec.replicas * schedule.blocks, total=schedule.total)
        raise _Started

    said = text_io.StringIO()
    real, run.start = run.start, stub
    try:
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
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


def ref_problems(ref: dict, earlier: set[str], inputs: dict) -> list[str]:
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
    return [f"reference {ref}: {{'run': <earlier run>, 'file': <path>}} or {{'input': <name>}}"]


def run_problems(run: dict, earlier: set[str], inputs: dict) -> list[str]:
    problems = []
    unknown = set(run) - {"name", "command", "args", "gpu"}
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
    args = run.get("args")
    if not isinstance(args, list):
        return [*problems, "args: a list of flags and values"]
    for arg in args:
        if isinstance(arg, dict):
            problems += ref_problems(arg, earlier, inputs)
        elif isinstance(arg, bool) or not isinstance(arg, str | int | float):
            problems.append(f"argument {arg!r}: a string, a number or a reference")
        elif isinstance(arg, float) and not math.isfinite(arg):
            problems.append(f"argument {arg!r}: not a finite number")
        elif isinstance(arg, str) and arg.split("=", 1)[0] in RUNNER_FLAGS:
            problems.append(f"argument {arg}: the runner sets it")
    return problems


def load_plan(plan: object) -> dict:
    """The plan, checked as a whole; PlanError lists every problem found."""
    if not isinstance(plan, dict):
        raise PlanError("a plan is a JSON object")
    problems = []
    keys = {"kernel", "description", "datasets", "kernel_sources", "inputs", "runs"}
    if set(plan) - keys:
        problems.append(f"unknown keys {sorted(set(plan) - keys)}")
    if not (isinstance(plan.get("kernel"), str) and SLUG.fullmatch(plan["kernel"])):
        problems.append(f"kernel {plan.get('kernel')!r}: a slug, 5-50 of a-z, 0-9 and '-'")
    if not isinstance(plan.get("description", ""), str):
        problems.append("description: a string")
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
    earlier: set[str] = set()
    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            problems.append(f"runs[{i}]: an object")
            continue
        name = run.get("name")
        where = f"run {name!r}" if isinstance(name, str) else f"runs[{i}]"
        problems += [f"{where}: {p}" for p in run_problems(run, earlier, inputs)]
        if isinstance(name, str):
            earlier.add(name)
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
    state: str = "pending"  # running, complete, failed, interrupted, blocked, not started
    note: str = ""
    run_id: str | None = None
    key: list[str] | None = None  # command and arguments, references as run ids: what reuse matches
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
            )
        )
    return runs


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


def ising_argv(py: str, *args: object) -> list[str]:
    return [py, "-I", "-u", "-m", "tipping_point.ising", *map(str, args)]


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


def find_wheel(ctx: Ctx) -> tuple[Path, dict]:
    """The wheel, and the commit its COMMIT.txt names; refused when that is missing or dirty."""
    root = find_input(ctx.input, WHEEL_DATASET)
    wheels = sorted(root.rglob("tipping_point-*.whl"))
    if len(wheels) != 1:
        raise Refused(
            f"expected one tipping_point wheel in {root}, found {[w.name for w in wheels]}"
        )
    notes = sorted(root.rglob("COMMIT.txt"))
    if len(notes) != 1:
        raise Refused(
            f"expected one COMMIT.txt beside the wheel in {root}, found {len(notes)}: "
            "nothing else names the commit the runs would record"
        )
    text = notes[0].read_text(encoding="utf-8").strip()
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
        if not (isinstance(record, dict) and record.get("run_id") and record.get("name") in names):
            continue
        src, dest = path.parent / "ising" / record["run_id"], ctx.out / "ising" / record["run_id"]
        if not src.is_dir():
            log(f"  {record['name']}: {src} is not in the attached output; skipped")
            continue
        if not dest.exists():
            shutil.copytree(src, dest)
        kept.append(record)
        log(f"  {record['name']}: {record['run_id']} ({record.get('state')}) copied in")
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


def verdicts(run_dir: Path, commit: str) -> list[str]:
    """What keeps a finished run from counting: its manifest must name the wheel's commit and say
    it is not dirty, and complete.json must say clean and hold no failed check."""
    manifest, done = read_json(run_dir / "manifest.json"), read_json(run_dir / "complete.json")
    if manifest is None:
        return ["manifest.json is missing or unreadable"]
    if done is None:
        return ["complete.json is missing or unreadable"]
    producer = manifest.get("producer") or {}
    problems = []
    if producer.get("git_sha") != commit:
        problems.append(f"manifest git_sha {producer.get('git_sha')} is not the wheel's {commit}")
    if producer.get("dirty") is not False:
        problems.append(f"manifest dirty is {producer.get('dirty')}")
    if done.get("clean") is not True:
        problems.append(f"complete.json clean is {done.get('clean')}")
    for check in checks_in(done.get("checks")):
        if check.get("passed") is not True:
            T = check.get("temperature")
            at = f" at T = {T:.4f}" if isinstance(T, int | float) else ""
            problems.append(f"check {check.get('gate', '?')} failed{at}: {check.get('detail', '')}")
    return problems


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
    (a pinned run only on its own GPU). A failed run stops its GPU's queue and blocks the runs that
    read it, while the other queues go on. No run starts after `stop`, or once `budget` objects."""

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
                if state == "failed":
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
        self.prior: dict[str, dict] = {r["name"]: r for r in (prior or {}).get("runs", [])}
        for r in self.runs:  # until this session decides, a run keeps the previous one's folder
            if r.name in self.prior:
                p = self.prior[r.name]
                r.run_id, r.key, r.step = p["run_id"], p.get("key"), int(p.get("step") or 0)
                r.wall_s = float(p.get("wall_s") or 0.0)
        self.rates: dict[tuple[int, int], float] = {}
        self.active: dict[str, subprocess.Popen] = {}
        self.active_lock = threading.Lock()  # a start against the deadline's kill sweep
        self.start_lock = threading.Lock()  # one new run folder appears at a time
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
        self.save()
        log("== runs")
        try:
            self.scheduler.run(self.tick, self.ctx.poll_s)
        except Exception as exc:  # the runs made so far are in out/: record the error, keep them
            self.summary["runner_error"] = traceback.format_exc()[-6000:]
            log(f"FAILED in the runner: {exc!r}\n{self.summary['runner_error']}")
        return self.finish()

    def locate_inputs(self) -> None:
        entries = self.plan.get("inputs", {})
        if not entries:
            log("none")
            return
        sources = self.plan.get("datasets", []) + self.plan.get("kernel_sources", [])
        roots = [find_input(self.ctx.input, s.split("/", 1)[1]) for s in sources]
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
            self.inputs[name] = good[0]
            log(f"input {name}: {good[0]}, sha256 {entry['sha256'][:16]}... as the plan says")
        self.summary["inputs"] = {n: str(p) for n, p in self.inputs.items()}

    def resolve(self, run: Run, pending_ok: bool = False) -> tuple[list[str], list[str]]:
        """The run's arguments as the CLI takes them, and its key: references as run ids."""
        tail, key = [], [run.command]
        for arg in shrink(run.args, self.ctx.shrink):
            if not isinstance(arg, dict):
                tail.append(str(arg))
                key.append(str(arg))
            elif "input" in arg:
                entry = self.plan["inputs"][arg["input"]]
                tail.append(str(self.inputs[arg["input"]]))
                key.append(f"input/{entry['run']}/{entry['file']}")
            else:
                dep = self.by_name[arg["run"]]
                if dep.state != "complete" or dep.run_id is None:
                    if not pending_ok:
                        raise RunError(f"{dep.name} is {dep.state}, not complete")
                    tail.append(PLACEHOLDER)
                    key.append(PLACEHOLDER)
                    continue
                rel = f"ising/{dep.run_id}" + (f"/{arg['file']}" if "file" in arg else "")
                if not (self.ctx.out / rel).exists():
                    raise RunError(f"{rel} does not exist ({dep.name}'s folder)")
                tail.append(str(self.ctx.out / rel))
                key.append(rel)
        return tail, key

    def helper(self, command: str, env: dict | None = None, **kwargs: Any) -> Any:
        return run_helper(self.ctx, command, env or self.env, **kwargs)

    def capture(self, runs: list[Run]) -> list[str]:
        """Each run's size and length as the CLI itself sets it up (the helper's capture); a run
        still waiting on an earlier one is only parsed. The CLI's refusals, by run."""
        items = []
        for r in runs:
            tail, _ = self.resolve(r, pending_ok=True)
            scratch = str(self.ctx.tmp / "tp-capture")
            argv = [r.command, *tail, "--device", self.ctx.device, "--data-dir", scratch]
            items.append({"argv": argv, "partial": PLACEHOLDER in tail})
        errors = []
        for r, got in zip(runs, self.helper("capture", items=items), strict=True):
            if "error" in got:
                errors.append(f"{r.name}: {got['error']}")
                continue
            r.shape, r.total = [got["L"], got["lattices"]], got["total"]
            r.estimate_s = self.estimate(r)
        return errors

    def capture_ready(self, runs: list[Run]) -> None:
        """Runs whose last dependency just finished: their real length, for the order and log."""
        waiting = [r for r in runs if r.total is None]
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
        one with `ising resume`, or start it."""
        prefix = f"[gpu{gpu} {run.name}] "
        try:
            tail, run.key = self.resolve(run)
        except RunError as exc:
            return "failed", str(exc)
        run_dir, how = self.adopt(run)
        if run_dir is not None and how == "reuse":
            log(f"{prefix}complete already: {run_dir.name}, from the previous session")
            return self.verify(run, run_dir, prefix, "reused from the previous session's output")
        if run_dir is not None:
            argv = ising_argv(self.ctx.py, "resume", run_dir)
        else:
            flags = ["--device", self.ctx.device, "--data-dir", self.ctx.out]
            argv = ising_argv(self.ctx.py, run.command, *tail, *flags)
        return self.launch(gpu, run, argv, run_dir, prefix)

    def adopt(self, run: Run) -> tuple[Path | None, str | None]:
        """The previous session's folder for this run, carried into this output, when it was made
        from the same arguments by the same commit, torch and device: (folder, reuse or resume)."""
        prior = self.prior.get(run.name)
        if prior is None:
            return None, None
        folder = self.ctx.out / "ising" / prior["run_id"]
        manifest = read_json(folder / "manifest.json") or {}
        env = manifest.get("env") or {}
        made = [
            (manifest.get("producer") or {}).get("git_sha"),
            env.get("torch"),
            env.get("device"),
        ]
        here = [self.commit, self.torch, self.ctx.device]
        if prior.get("key") != run.key:
            why = "it was made from other arguments"
        elif not folder.is_dir():
            why = "its folder is not in this output"
        elif made != here:
            why = f"it was made by commit, torch and device {made}, not {here}"
        elif not ((folder / "complete.json").is_file() or (folder / "checkpoint.pt").is_file()):
            why = "it holds neither complete.json nor a checkpoint"
        else:
            why = None
        if why is not None:
            log(f"[{run.name}] {prior['run_id']} is not reused: {why}; starting afresh")
            shutil.rmtree(folder, ignore_errors=True)  # superseded: no place in this output
            run.run_id, run.step, run.wall_s = None, 0, 0.0
            return None, None
        if (folder / "complete.json").is_file():
            run.reused = True
            return folder, "reuse"
        run.resumed = True
        return folder, "resume"

    def launch(
        self, gpu: int, run: Run, argv: list[str], run_dir: Path | None, prefix: str
    ) -> tuple[str, str]:
        env = dict(self.env, CUDA_VISIBLE_DEVICES=str(gpu))
        ising = self.ctx.out / "ising"
        tail: collections.deque[str] = collections.deque(maxlen=60)
        clock = time.time()
        fresh = run_dir is None
        with self.start_lock if fresh else contextlib.nullcontext():
            before = set(os.listdir(ising))
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
                run_dir = self.new_folder(ising, before, proc)
                run.run_id = None if run_dir is None else run_dir.name
                if run_dir is not None:
                    self.save()
                    # the next start falls in a later second, since a run id names its second
                    time.sleep(1)
        try:
            if run_dir is None and proc.poll() is None:  # alive, yet no folder after NEW_FOLDER_S
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

    def new_folder(self, ising: Path, before: set[str], proc: subprocess.Popen) -> Path | None:
        """The folder a fresh run makes: the one new name in out/ising. Starts take turns (the
        start lock), so it is this run's."""
        clock = time.time()
        while time.time() - clock < NEW_FOLDER_S:
            exited = proc.poll() is not None
            new = sorted(n for n in os.listdir(ising) if n not in before and RUN_ID.fullmatch(n))
            if new:
                if len(new) > 1:
                    log(f"WARNING: several new run folders {new}; taking the last")
                return ising / new[-1]
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
                            int(found[k].replace(",", "")) for k in ("step", "total", "rate")
                        )
                        run.step, run.total, run.sweeps_per_s = step, total, rate
                        if rate:
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
        if run_dir is not None and (run_dir / "complete.json").is_file() and (code == 0 or killed):
            return self.verify(run, run_dir, prefix)
        if killed:
            if run_dir is not None and (run_dir / "checkpoint.pt").is_file():
                return "interrupted", "stopped at the session deadline; resumable"
            return "not started", "session deadline (stopped before its first checkpoint)"
        run.output_tail = list(tail)[-40:]
        last = next((line for line in reversed(tail) if line.strip()), "no output")
        made = "" if run_dir is not None else " before it made a run folder"
        return "failed", f"exit {code}{made}: {last}"

    def verify(self, run: Run, run_dir: Path, prefix: str, note: str = "") -> tuple[str, str]:
        if run.command == "pilot" and not (run_dir / "tau.json").is_file():
            # `ising pilot` writes tau.json after complete.json; `ising resume` never does
            log(f"{prefix}no tau.json, as after a resume: writing it as `ising pilot` would")
            self.helper("tau_json", run_dir=str(run_dir), seconds=round(run.wall_s, 1))
        problems = verdicts(run_dir, self.commit)
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
        log(f"{prefix}complete and verified: {run_dir.name}")
        return "complete", note

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
            self.summary.update(
                updated_utc=now_utc(),
                status=status,
                status_detail=detail,
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


def exit_code(ctx: Ctx) -> int:
    """Non-zero only when the output holds no run folder: a failed version may lose its output."""
    folder = ctx.out / "ising"
    return 0 if folder.is_dir() and any(p.is_dir() for p in folder.iterdir()) else 1


def refuse(ctx: Ctx, summary: dict, why: str) -> int:
    summary.update(status="refused", status_detail=why, updated_utc=now_utc(), ended_utc=now_utc())
    write_text(ctx.out / STATE, json.dumps(summary, indent=2, default=str))
    write_text(ctx.out / "STATUS", f"refused: {why}\n")
    banner = "=" * 100
    log(f"{banner}\nSTATUS REFUSED: {why}\n{banner}")
    return exit_code(ctx)


def kernel_main(ctx: Ctx) -> int:
    for folder in (ctx.out / "logs", ctx.out / "ising", ctx.cwd, ctx.helper.parent):
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
        wheel, summary["wheel"] = find_wheel(ctx)
        env = summary["env"] = probe(ctx) if ctx.py else ensure_env(ctx, wheel)
        log(json.dumps(env, indent=2))
        check_env(ctx, env, summary["wheel"]["commit"])
        gpus = ctx.gpus or len(env["devices"])
        commit = summary["wheel"]["commit"]
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


# -- local: the kernel folder, the dry run, the self-test --------------------------------------


def cli_errors(plan: dict) -> list[str]:
    """The CLI's own verdict on every run's arguments, through the helper's capture run in this
    process. A run that takes a file is only parsed: the file comes into being on Kaggle."""
    helper: dict[str, Any] = {"__name__": "tp_kaggle_helper"}
    exec(compile(HELPER_SOURCE, "helper.py", "exec"), helper)
    with tempfile.TemporaryDirectory(prefix="tp-capture-") as scratch:
        items = []
        for spec in plan["runs"]:
            args = [PLACEHOLDER if isinstance(a, dict) else str(a) for a in spec["args"]]
            argv = [spec["command"], *args, "--device", "cpu", "--data-dir", scratch]
            items.append({"argv": argv, "partial": PLACEHOLDER in args})
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
        "dataset_sources": [f"{OWNER}/{WHEEL_DATASET}", *plan.get("datasets", [])],
        "competition_sources": [],
        "kernel_sources": [*plan.get("kernel_sources", []), *own],
        "model_sources": [],
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{slug}.py").write_text(script, encoding="utf-8", newline="\n")
    text = json.dumps(metadata, indent=2) + "\n"
    (dest / "kernel-metadata.json").write_text(text, encoding="utf-8", newline="\n")
    return dest / f"{slug}.py"


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
    return str(wp._venv(root / "env", wheel, requirements))


def dry_run(args: argparse.Namespace) -> int:
    plan = read_plan(Path(args.dry_run))
    if args.only:
        plan = only(plan, args.only.split(","))
    if args.root is not None and args.root.exists() and any(args.root.iterdir()):
        raise SystemExit(f"--root {args.root} is not empty: a dry run needs a folder of its own")
    root = (args.root or Path(tempfile.mkdtemp(prefix="tp-dry-run-"))).resolve()
    log(
        f"dry run of {args.dry_run} ({', '.join(r['name'] for r in plan['runs'])}) under {root}: "
        f"L / {args.shrink}, sweep counts / {args.shrink**2}, {args.gpus} queues on the CPU"
    )
    python = build_wheel_env(root)
    script = write_kernel(plan, root / "kernel", resume=True)
    for spec in args.input:  # what Kaggle would attach: SLUG=DIR
        slug, _, folder = spec.partition("=")
        shutil.copytree(folder, root / "input" / slug)
    sessions = []
    for n in (1, 2):
        if n == 2:  # session 1's output, attached as kernel_sources attaches it
            os.replace(root / "working", root / "input" / plan["kernel"])
        log(f"== dry-run session {n}" + (", stopped by a simulated deadline" if n == 1 else ""))
        argv = [sys.executable, script, "--local-session", "--root", root, "--python", python]
        argv += ["--gpus", args.gpus, "--shrink", args.shrink, *(["--interrupt"] if n == 1 else [])]
        code = subprocess.run([str(a) for a in argv], check=False).returncode
        sessions.append((code, read_json(root / "working" / "out" / STATE) or {}))
    return dry_run_report(sessions, root)


def dry_run_report(sessions: list[tuple[int, dict]], root: Path) -> int:
    print("\n" + "=" * 100)
    for n, (code, state) in enumerate(sessions, 1):
        print(
            f"session {n}: exit {code}; STATUS {state.get('status')}: {state.get('status_detail')}"
        )
        for r in state.get("runs", []):
            how = " (reused)" if r["reused"] else " (resumed)" if r["resumed"] else ""
            print(
                f"  {r['name']:<14} {r['state']:<12}{how:<10} {r['run_id'] or '-':<34} {r['note']}"
            )
    (_, first), (code, last) = sessions
    runs = {r["name"]: r for r in last.get("runs", [])}
    folders = {r["name"]: r["run_id"] for r in last.get("runs", [])}
    reads = [
        f"{r['name']} read ising/{folders[d]}"
        for r in last.get("runs", [])
        for d in r["deps"]
        if any(str(k).startswith(f"ising/{folders[d]}") for k in r["key"] or [])
    ]
    needed = sum(len(r["deps"]) for r in last.get("runs", []))
    stopped = [r["name"] for r in first.get("runs", []) if r["state"] == "interrupted"]
    findings = {
        "session 1 finished a run": any(r["state"] == "complete" for r in first.get("runs", [])),
        "session 1 left a resumable run": bool(stopped),
        "session 2 is complete": last.get("status") == "complete" and code == 0,
        "session 2 reused a finished run": any(r["reused"] for r in runs.values()),
        "session 2 resumed every interrupted run": all(runs[n]["resumed"] for n in stopped),
        "every reference read its run's folder": len(reads) == needed,
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
    """Plan validation, dependency resolution, the queues, the previous session's runs, inputs,
    the wheel checks and run verification, in-process: nothing is simulated. The checks live
    beside this file (kaggle_ising_runs_selftest.py), so the pushed kernel stays lean."""
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
    chosen = args.write_kernel or args.dry_run or args.self_test or args.local_session
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
