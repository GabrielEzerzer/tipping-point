# The panel sequence run as pieces on Kaggle CPU kernels, one piece per kernel, and a dry run on a
# synthetic panel that proves the environment, the pinned commit, the merge and the download.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DESCRIPTION = """\
Tipping Point: the panel sequence run (python -m tipping_point.panel sequence) split into pieces,
one Kaggle CPU kernel each, and a dry run on a synthetic panel. A Kaggle script kernel.

    uv run --no-sync python notebooks/kaggle_panel_sequence.py write --commit <40 hex> --out <dir> \
        [--data-dir <store>]
    uv run --no-sync python notebooks/kaggle_panel_sequence.py local-dry <empty dir>

`write` makes one kernel folder per piece (tp-panel-seq-<variant>-s<seed>-f<folds>) and one for
the dry run (tp-panel-seq-dry), each this file with its job embedded and a kernel-metadata.json:
private, CPU, internet on (to fetch the code and the pinned packages). `kaggle kernels push -p
<folder>` starts one. A piece is the shipped plan (panel/sequence.toml) with its variants, seeds
and folds cut to one variant, one seed and three folds; the twelve pieces are every cell of the
full plan once, so `python -m tipping_point.panel merge`, whose parent plan is the shipped plan
by default, joins them into the full run and refuses any set that misses or adds a cell.

Every kernel clones the public repository at the job's commit and refuses to run unless that
checkout's HEAD is the commit and it is clean. It runs from that checkout, so each run's manifest
records the commit (producer.git_sha) and the sha256 of its uv.lock, in a Python 3.13 environment
of torch 2.14.1+cpu and uv.lock's numpy and pyarrow, with OMP_NUM_THREADS=2 and the plan's
threads = 2, one process per kernel. Every process starts with the instruction set pinned to AVX2
for torch's own kernels (ATEN_CPU_CAPABILITY=avx2), oneDNN (ONEDNN_MAX_CPU_ISA=AVX2), MKL
(MKL_ENABLE_INSTRUCTIONS=AVX2, MKL_CBWR=AVX2), numpy's OpenBLAS, which the ladder's logistic fits
use (OPENBLAS_CORETYPE=Haswell, its AVX2 core, and OPENBLAS_NUM_THREADS=2), and numpy's own
dispatch (NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR", its AVX512 targets); the kernel
refuses to run unless torch reports the AVX2 capability, OpenBLAS the Haswell core and two
threads, and numpy those targets off, and every sequence run records its CPU, capability and
pins. A piece reads the private dataset tp-panel-sequence-data, which holds only the files the
sequence run reads: the inputs run's manifest.json, complete.json and inputs.parquet, and the
labels run's grid file. `write` reads the sha256 of the inputs table and the grid file from the
inputs run's records in the local store and puts them in each piece's job; the kernel refuses
before training unless the dataset's files have those sha256 values and its manifest names the
expected inputs run and labels run, and before hashing them it refuses a manifest span or Parquet
time bounds that reach the held-back span, read from the metadata only. A piece writes its run
to out/panel-sequence/<run id>/ and its log to out/logs/, and marks itself complete only if the
run's saved models are exactly its plan's cells and it skipped nothing;
`kaggle kernels output <owner>/<slug> -p <dir>` fetches out/. out/STATUS holds one line: running,
complete or failed.

The dry run reads no dataset. On a synthetic panel whose labels are planted on one input, it runs
two pieces of a small plan, merges them against that plan, scores the merged run, and then times
a fixed number of JEPA and supervised steps at the shipped plan's shapes on 30 synthetic coins and
one ladder fit at the probe's widest readout, once with the OpenBLAS pins and once without
(out/summary.json). `local-dry` runs the same steps on this machine in the current environment,
for the speed ratio, and writes dry.json in its folder; it needs a clean checkout, since the merge
refuses runs made from uncommitted code.
"""
JOB_JSON: str | None = None

OWNER = "gabrielezerzer"
REPO = "https://github.com/GabrielEzerzer/tipping-point"
DATASET = "tp-panel-sequence-data"
INPUTS_RUN = "20261124T103955Z-inputs"
LABELS_RUN = "20261124T020153Z-panel-20220101-20231231"
GRID_FILE = "01640995200000000000.parquet"
PYTHON = "3.13"
TORCH_CPU = "torch==2.14.1+cpu"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PINNED = ("numpy==2.5.4", "pyarrow==26.0.0")
THREADS = "2"
VARIANTS = ("full", "vol")
SEEDS = (1, 2, 3)
FOLD_GROUPS = ((2, 3, 4), (5, 6, 7))
BENCH = {"coins": 30, "hours": 3000, "jepa_steps": 60, "supervised_steps": 60, "fit_rows": 100_000}
COMMIT = re.compile(r"[0-9a-f]{40}")
PINS = {
    "ATEN_CPU_CAPABILITY": "avx2",
    "ONEDNN_MAX_CPU_ISA": "AVX2",
    "MKL_ENABLE_INSTRUCTIONS": "AVX2",
    "MKL_CBWR": "AVX2",
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": THREADS,
    "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR",
}
CAPABILITY = "AVX2"
BLAS_PROBE = """import ctypes, json
from pathlib import Path
import numpy
libs = Path(numpy.__file__).parent.parent / "numpy.libs"
(path,) = [p for p in libs.iterdir() if "openblas" in p.name]
lib = ctypes.CDLL(str(path))
said = {}
for key, name, kind in (
    ("core", "scipy_openblas_get_corename64_", ctypes.c_char_p),
    ("config", "scipy_openblas_get_config64_", ctypes.c_char_p),
    ("threads", "scipy_openblas_get_num_threads64_", ctypes.c_int),
):
    fn = getattr(lib, name)
    fn.restype = kind
    value = fn()
    said[key] = value.decode() if isinstance(value, bytes) else value
print(json.dumps(said))
"""
NPY_PROBE = """import json, sys
from numpy._core import _multiarray_umath as m
off = sys.argv[1].split()
print(json.dumps({"dispatch": m.__cpu_dispatch__, "on": [f for f in off if m.__cpu_features__[f]]}))
"""
CELLS = """import json, sys
from pathlib import Path
from tipping_point.panel import sequence
print(json.dumps(sorted(sequence.cells(sequence.load(Path(sys.argv[1]))))))
"""
ISA_PROBE = """import torch
torch.set_num_threads(2)
x = torch.randn(4, 8, 64)
torch.nn.functional.conv1d(x, torch.randn(8, 8, 3))
a = torch.randn(256, 256)
print(float((a @ a).sum()))
"""

DRY_LADDER = """start = 2022-01-01
end = 2022-12-31
horizons = [24, 1]
sides = ["long", "short", "any"]
without = []

[model]
c = 0.01
clip = 5.0

[folds]
gap_days = 31
forward_first = 2

[bootstrap]
block_days = 14
draws = 10
seed = 1

[[rungs]]
name = "vol"
groups = ["vol"]
"""

DRY_PLAN = """ladder = "ladder.toml"
variants = ["vol"]
rungs = ["vol"]
models = ["jepa", "twin", "supervised"]
schemes = ["forward"]
seeds = [3]
folds = FOLDS
target = "fresh"
threads = 2

[window]
length = 36
future = 4
shifts = [2, 4]

[fresh]
changes = ["oi_vs_28d", "funding", "ls"]

[fresh.windowed]
vol_1h = 1
ret_1h = 1
flow_1h = 1
vsurge_1h = 1
vol_6h = 6
ret_6h = 6
vol_24h = 24
near_24h = 24
volume_24h = 24
ret_24h = 24
aret_24h = 24
flow_24h = 24
oi_24h = 24

[encoder]
patch = 2
width = 8
conv_blocks = 1
attention_blocks = 1
heads = 2
kernel = 3
expansion = 2

[predictor]
hidden = 8

[jepa]
steps = 20
batch = 16
lr = 1e-3
final_lr = 1e-4
warmup = 2
weight_decay = 0.05
curve_every = 5
momentum_start = 0.996
momentum_end = 1.0
prediction = 1.0
objective = "predictor"
variance = 0.5
covariance = 0.02

[supervised]
steps = 20
batch = 16
lr = 1e-3
final_lr = 1e-4
warmup = 2
weight_decay = 0.1
curve_every = 5
dropout = 0.2

[selection]
share = 0.2
every = 5
negatives = 200

[probe]
batch = 4096
readouts = ["last", "means", "spreads"]
c = [0.01, 1.0]
"""


def log(message: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(argv: list[Any], env: dict[str, str] | None = None, logfile: Path | None = None) -> str:
    log("$ " + " ".join(map(str, argv)))
    proc = subprocess.run(
        [str(a) for a in argv], env=env, capture_output=True, text=True, check=False
    )
    text = proc.stdout + proc.stderr
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as f:
            f.write(text)
    print(text[-4000:], flush=True)
    if proc.returncode:
        raise RuntimeError(f"exit {proc.returncode}: {' '.join(map(str, argv))}")
    return proc.stdout


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pieces() -> list[dict[str, Any]]:
    return [
        {"variants": [v], "seeds": [s], "folds": list(g)}
        for v in VARIANTS
        for s in SEEDS
        for g in FOLD_GROUPS
    ]


def slug(piece: dict[str, Any]) -> str:
    folds = piece["folds"]
    return f"tp-panel-seq-{piece['variants'][0]}-s{piece['seeds'][0]}-f{folds[0]}-{folds[-1]}"


def cut(text: str, piece: dict[str, Any]) -> str:
    for key in ("variants", "seeds", "folds"):
        line = f"{key} = {json.dumps(piece[key])}"
        text, n = re.subn(rf"^{key} = .*$", line, text, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"the plan has no single one-line {key} entry")
    return text


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {"cpu_count": os.cpu_count(), "python": sys.version.split()[0]}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        names = re.findall(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), flags=re.MULTILINE)
        info["cpu_model"] = names[0] if names else None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        found = re.search(r"^MemTotal:\s*(\d+)", meminfo.read_text(), flags=re.MULTILINE)
        info["mem_gb"] = round(int(found.group(1)) / 2**20, 1) if found else None
    return info


def child_env(src: Path | None) -> dict[str, str]:
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, **PINS)
    if src is not None:
        env["PYTHONPATH"] = str(src / "python" / "tipping_point" / "src")
    return env


def environment(tmp: Path, commit: str, logs: Path) -> tuple[str, Path, dict[str, Any]]:
    src, venv = tmp / "src", tmp / "env"
    pip_env = dict(os.environ, PIP_ROOT_USER_ACTION="ignore", PIP_DISABLE_PIP_VERSION_CHECK="1")
    run([sys.executable, "-m", "pip", "install", "--quiet", "uv"], pip_env, logs / "env.log")
    uv_env = dict(
        os.environ,
        UV_CACHE_DIR=str(tmp / "uv-cache"),
        UV_LINK_MODE="copy",
        UV_PYTHON_INSTALL_DIR=str(tmp / "uv-python"),
    )
    uv = [sys.executable, "-m", "uv"]
    run([*uv, "venv", "--python", PYTHON, venv], uv_env, logs / "env.log")
    py = str(venv / "bin" / "python")
    install = [*uv, "pip", "install", "--python", py]
    run([*install, TORCH_CPU, "--index-url", TORCH_CPU_INDEX], uv_env, logs / "env.log")
    run([*install, *PINNED], uv_env, logs / "env.log")
    run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", REPO, src])
    run(["git", "-C", src, "checkout", "--quiet", "--detach", commit])
    probe = (
        "import json, numpy, pyarrow, torch\n"
        "from tipping_point.ising import io\n"
        "print(json.dumps({'code': io.code_version(), 'torch': torch.__version__,"
        " 'numpy': numpy.__version__, 'pyarrow': pyarrow.__version__,"
        " 'threads': torch.get_num_threads(),"
        " 'capability': torch.backends.cpu.get_cpu_capability()}))\n"
    )
    info = json.loads(run([py, "-c", probe], child_env(src)).strip().splitlines()[-1])
    if info["code"] != {"git_sha": commit, "dirty": False}:
        raise RuntimeError(f"the checkout is {info['code']}, not commit {commit} clean")
    if info["capability"] != CAPABILITY:
        raise RuntimeError(f"torch runs at {info['capability']}, not the pinned {CAPABILITY}")
    verbose = dict(child_env(src), ONEDNN_VERBOSE="1", MKL_VERBOSE="1")
    said = run([py, "-c", ISA_PROBE], verbose).splitlines()
    info["isa_lines"] = [line for line in said if re.search(r"isa|MKL_VERBOSE oneMKL", line)][:4]
    info["openblas"] = json.loads(run([py, "-c", BLAS_PROBE], child_env(src)).splitlines()[-1])
    blas = (info["openblas"]["core"], str(info["openblas"]["threads"]))
    if blas != (PINS["OPENBLAS_CORETYPE"], THREADS):
        raise RuntimeError(f"OpenBLAS runs {info['openblas']}, not Haswell with {THREADS} threads")
    off = PINS["NPY_DISABLE_CPU_FEATURES"]
    info["numpy_cpu"] = json.loads(run([py, "-c", NPY_PROBE, off], child_env(src)).splitlines()[-1])
    if info["numpy_cpu"]["on"]:
        raise RuntimeError(f"numpy dispatches to {info['numpy_cpu']['on']}, which are pinned off")
    return py, src, info


def verify(run_dir: Path, commit: str | None, cells: set[str] | None = None) -> dict[str, Any]:
    complete = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bad = [n for n, d in complete["files"].items() if sha256(run_dir / n) != d]
    if complete.get("clean") is not True or bad:
        raise RuntimeError(f"{run_dir.name}: not clean, or files differ: {bad}")
    producer = manifest["producer"]
    if commit is not None and (producer["git_sha"], producer["dirty"]) != (commit, False):
        raise RuntimeError(f"{run_dir.name}: made by {producer}, not commit {commit} clean")
    runtime = manifest["params"].get("runtime")
    pinned = runtime is not None and (runtime["capability"], runtime["pins"]) == (CAPABILITY, PINS)
    if manifest["domain"] == "panel-sequence" and not pinned:
        raise RuntimeError(f"{run_dir.name}: ran at {runtime}, not {CAPABILITY} with {PINS}")
    encoders = manifest["params"].get("encoders", {})
    if cells is not None and set(encoders) != cells:
        raise RuntimeError(f"{run_dir.name}: its saved models are not exactly its plan's cells")
    files = complete["files"]
    if cells is not None and any(c not in files or files[c] != encoders[c] for c in cells):
        raise RuntimeError(f"{run_dir.name}: a saved model is not among its verified files")
    if cells is not None and complete["checks"]["skipped"]:
        raise RuntimeError(f"{run_dir.name}: skipped {complete['checks']['skipped']}")
    found = {"run": run_dir.name, "files": len(complete["files"]), "checks": complete["checks"]}
    return found | ({"runtime": runtime} if runtime else {})


def new_run(folder: Path, before: set[str]) -> Path:
    (made,) = [p for p in folder.iterdir() if p.name not in before]
    return made


def find_dataset(root: Path) -> Path:
    hits = sorted(p.parent for p in root.rglob("inputs.parquet"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one inputs.parquet under {root}, found {hits}")
    return hits[0]


def guard_dataset(flat: Path, names: list[str]) -> None:
    from datetime import date

    from tipping_point.panel import span, store

    params = json.loads((flat / "manifest.json").read_text(encoding="utf-8"))["params"]
    span.refuse(*(date.fromisoformat(params[k]) for k in ("start", "dev_start", "end")))
    for name in names:
        store.refuse_held_back(flat / name)


def check_dataset(flat: Path, data: dict[str, Any], guard: Any) -> None:
    manifest = json.loads((flat / "manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((flat / "complete.json").read_text(encoding="utf-8"))
    runs = (manifest.get("run_id"), manifest.get("params", {}).get("labels_run"))
    if runs != (data["inputs_run"], data["labels_run"]):
        raise RuntimeError(
            f"the dataset holds runs {runs}, not {data['inputs_run']} of {data['labels_run']}"
        )
    if complete.get("files", {}).get("inputs.parquet") != data["sha256"]["inputs.parquet"]:
        raise RuntimeError("the dataset's complete.json records another inputs table")
    guard(flat, sorted(data["sha256"]))
    for name, digest in data["sha256"].items():
        if sha256(flat / name) != digest:
            raise RuntimeError(f"the dataset's {name} has changed: its sha256 is not {digest}")


def piece_job(job: dict[str, Any], py: str, src: Path, tmp: Path, out: Path) -> dict[str, Any]:
    flat = find_dataset(Path("/kaggle/input"))
    script = src / "notebooks" / "kaggle_panel_sequence.py"

    def guard(folder: Path, names: list[str]) -> None:
        run([py, script, "guard", folder, *names], child_env(src))

    check_dataset(flat, job["data"], guard)
    store = tmp / "store"
    inputs_dir = store / "panel-inputs" / INPUTS_RUN
    grid_dir = store / "labels" / LABELS_RUN / "grid"
    inputs_dir.mkdir(parents=True)
    grid_dir.mkdir(parents=True)
    for name in ("manifest.json", "complete.json", "inputs.parquet"):
        (inputs_dir / name).symlink_to(flat / name)
    (grid_dir / GRID_FILE).symlink_to(flat / GRID_FILE)
    runs = out / "panel-sequence"
    runs.mkdir(parents=True)
    (store / "panel-sequence").symlink_to(runs)
    panel = src / "python" / "tipping_point" / "src" / "tipping_point" / "panel"
    plan_dir = tmp / "plan"
    plan_dir.mkdir()
    shipped = (panel / "sequence.toml").read_text(encoding="utf-8")
    (plan_dir / "sequence.toml").write_text(cut(shipped, job["piece"]), encoding="utf-8")
    shutil.copyfile(panel / "ladder.toml", plan_dir / "ladder.toml")
    began = time.time()
    argv = [py, "-m", "tipping_point.panel", "sequence", "--inputs", INPUTS_RUN]
    argv += ["--plan", plan_dir / "sequence.toml", "--data-dir", store, "--tag", job["kernel"]]
    run(argv, child_env(src), out / "logs" / "sequence.log")
    made = new_run(runs, set())
    said = run([py, "-c", CELLS, plan_dir / "sequence.toml"], child_env(src))
    cells = set(json.loads(said.strip().splitlines()[-1]))
    return {"seconds": round(time.time() - began, 1), **verify(made, job["commit"], cells)}


def write_synthetic(data: Path) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tipping_point.panel import inputs, labels, store

    rng = np.random.default_rng(0)
    coins = ["AAAUSDT", "BBBUSDT"]
    hours = labels.ns(labels.date(2022, 1, 1)) + labels.HOUR_NS * np.arange(365 * 24)
    n = len(hours) * len(coins)
    values = {name: rng.normal(size=n) for name in inputs.OWN + inputs.MARKET}
    grid: dict[str, Any] = {
        "ts_ns": np.repeat(hours, len(coins)).astype(np.int64),
        "symbol": np.tile(np.array(coins, dtype=object), len(hours)),
    }
    for h in labels.HORIZONS:
        lift = values["vol_1h"]
        grid[f"long_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 - lift))
        grid[f"short_{h}"] = rng.random(n) < 1 / (1 + np.exp(2.5 + lift))
        grid[f"censored_{h}"] = rng.random(n) < 0.01
    grid["skip"] = rng.random(n) < 0.02
    labels_dir = data / "labels" / "20260101T000000Z-synthetic"
    (labels_dir / "grid").mkdir(parents=True)
    pq.write_table(pa.table(grid), labels_dir / "grid" / "0.parquet")
    table = pa.table(
        {"ts_ns": grid["ts_ns"], "symbol": grid["symbol"].astype(str), **values},
        schema=inputs.SCHEMA,
    )
    params = {
        "labels_run": labels_dir.name,
        "labels_grid": {"file": "grid/0.parquet", "sha256": sha256(labels_dir / "grid/0.parquet")},
        "start": "2021-10-01",
        "dev_start": "2022-01-01",
        "end": "2022-12-31",
    }
    run_dir = store.write(data, "panel-inputs", "synthetic", {"inputs": table}, params, {})
    for name, folds in (("a", "[2]"), ("b", "[3]"), ("parent", "[2, 3]")):
        folder = data / "plans" / name
        folder.mkdir(parents=True)
        (folder / "ladder.toml").write_text(DRY_LADDER, encoding="utf-8")
        (folder / "sequence.toml").write_text(DRY_PLAN.replace("FOLDS", folds), encoding="utf-8")
    print(json.dumps({"inputs": run_dir.name}))


def bench() -> None:
    from dataclasses import replace

    import numpy as np
    import torch
    from tipping_point.panel import ladder, sequence

    plan = sequence.load()
    torch.set_num_threads(plan.threads)
    rng = np.random.default_rng(1)
    ncoins, nhours = BENCH["coins"], BENCH["hours"]
    columns = plan.rung("full").columns
    raw = rng.normal(size=(ncoins, nhours, len(columns)))
    train = np.ones(nhours, bool)
    mu, sd = sequence.statistics(raw, train)
    x = torch.from_numpy(sequence.standardised(raw, mu, sd, plan.ladder.clip, plan.pad))
    fit = sequence.eligible(train, plan.pad, plan.reach)
    rules = sequence.fresh(raw, columns, train, plan)
    jepa = replace(plan, jepa=replace(plan.jepa, steps=BENCH["jepa_steps"]))
    began = time.perf_counter()
    sequence.pretrain(x, fit, jepa, 0, np.random.SeedSequence(0), lambda line: None, rules)
    jepa_s = time.perf_counter() - began
    y = torch.from_numpy((rng.random((ncoins, nhours, len(plan.outputs))) < 0.05).astype("f4"))
    keep = torch.ones_like(y)
    t = replace(plan.supervised, steps=BENCH["supervised_steps"])
    rows = sequence.sample_rows(fit, ncoins)
    began = time.perf_counter()
    sequence.train_supervised(
        x, y, keep, rows, t, plan, 0, np.random.SeedSequence(1), lambda line: None
    )
    supervised_s = time.perf_counter() - began
    width = sequence.READOUTS["spreads"] * plan.shape.width
    a = rng.normal(size=(BENCH["fit_rows"], width))
    labels = rng.random(BENCH["fit_rows"]) < 1 / (1 + np.exp(3 - a[:, 0]))
    began = time.perf_counter()
    ladder.fit(a, labels.astype(np.float64), max(plan.cs))
    fit_s = time.perf_counter() - began
    print(
        json.dumps(
            {
                **BENCH,
                "jepa_batch": plan.jepa.batch,
                "supervised_batch": plan.supervised.batch,
                "channels": int(x.shape[-1]),
                "threads": torch.get_num_threads(),
                "jepa_s_per_step": jepa_s / BENCH["jepa_steps"],
                "supervised_s_per_step": supervised_s / BENCH["supervised_steps"],
                "fit_features": width,
                "fit_s": fit_s,
                "openblas": {
                    n: os.environ.get(n) for n in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS")
                },
            }
        )
    )


def dry_steps(py: str, script: Path, env: dict[str, str], data: Path, logs: Path) -> dict:
    made = json.loads(run([py, script, "synthetic", data], env).strip().splitlines()[-1])
    panel = [py, "-m", "tipping_point.panel"]
    runs = data / "panel-sequence"
    found: dict[str, Any] = {"inputs": made["inputs"]}
    for name in ("a", "b"):
        before = {p.name for p in runs.iterdir()} if runs.is_dir() else set()
        plan = data / "plans" / name / "sequence.toml"
        argv = [*panel, "sequence", "--inputs", made["inputs"], "--plan", plan]
        run([*argv, "--data-dir", data, "--tag", f"piece-{name}"], env, logs / "dry.log")
        found[name] = verify(new_run(runs, before), None)
    before = {p.name for p in runs.iterdir()}
    pair = [runs / found["a"]["run"], runs / found["b"]["run"]]
    parent = data / "plans" / "parent" / "sequence.toml"
    argv = [*panel, "merge", "--runs", *pair, "--plan", parent, "--data-dir", data]
    run(argv, env, logs / "dry.log")
    found["merged"] = verify(new_run(runs, before), None)
    argv = [*panel, "scores", "--sequence", found["merged"]["run"], "--model", "jepa"]
    run([*argv, "--variant", "vol", "--data-dir", data], env, logs / "dry.log")
    (scores,) = (data / "panel-scores").iterdir()
    found["scores"] = verify(scores, None)
    found["bench"] = json.loads(run([py, script, "bench"], env).strip().splitlines()[-1])
    free = {n: v for n, v in env.items() if not n.startswith("OPENBLAS_")}
    said = run([py, script, "bench"], free).strip().splitlines()[-1]
    found["bench_openblas_unpinned"] = json.loads(said)
    return found


def dry_job(job: dict[str, Any], py: str, src: Path, out: Path) -> dict[str, Any]:
    script = src / "notebooks" / "kaggle_panel_sequence.py"
    return dry_steps(py, script, child_env(src), out / "dry", out / "logs")


def kernel(job: dict[str, Any]) -> int:
    out, tmp = Path("/kaggle/working/out"), Path("/tmp/tp")
    out.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    (out / "STATUS").write_text("running\n", encoding="utf-8")
    summary: dict[str, Any] = {"job": job, "machine": machine(), "started": time.time()}
    try:
        if not COMMIT.fullmatch(job["commit"]):
            raise RuntimeError(f"the job names no commit: {job['commit']!r}")
        began = time.time()
        py, src, summary["environment"] = environment(tmp, job["commit"], out / "logs")
        summary["environment_seconds"] = round(time.time() - began, 1)
        if job["mode"] == "piece":
            summary["result"] = piece_job(job, py, src, tmp, out)
        else:
            summary["result"] = dry_job(job, py, src, out)
        status = "complete"
    except Exception as exc:
        summary["error"] = repr(exc)
        status = "failed"
        log(f"failed: {exc!r}")
    summary["seconds"] = round(time.time() - summary["started"], 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "STATUS").write_text(f"{status}\n", encoding="utf-8")
    log(f"{status} in {summary['seconds']} s")
    return 0


def write_kernel(job: dict[str, Any], dest: Path, datasets: list[str]) -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    literal = repr(json.dumps(job))
    script, n = re.subn(
        r"^JOB_JSON: str \| None = None$",
        lambda _: f"JOB_JSON: str | None = {literal}",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError("this file has no JOB_JSON line to fill")
    metadata = {
        "id": f"{OWNER}/{job['kernel']}",
        "title": job["kernel"],
        "code_file": f"{job['kernel']}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{job['kernel']}.py").write_text(script, encoding="utf-8", newline="\n")
    text = json.dumps(metadata, indent=2) + "\n"
    (dest / "kernel-metadata.json").write_text(text, encoding="utf-8", newline="\n")


def recorded_data(store: Path) -> dict[str, Any]:
    inputs_dir = store / "panel-inputs" / INPUTS_RUN
    try:
        manifest = json.loads((inputs_dir / "manifest.json").read_text(encoding="utf-8"))
        complete = json.loads((inputs_dir / "complete.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"no inputs run {INPUTS_RUN} under {store}: {exc}") from exc
    params = manifest["params"]
    if manifest["run_id"] != INPUTS_RUN or params["labels_run"] != LABELS_RUN:
        raise SystemExit(f"{inputs_dir} is not inputs run {INPUTS_RUN} of labels run {LABELS_RUN}")
    if params["labels_grid"]["file"] != f"grid/{GRID_FILE}" or complete.get("clean") is not True:
        raise SystemExit(f"{inputs_dir}: another grid file, or not marked clean")
    digests = {"inputs.parquet": complete["files"]["inputs.parquet"]}
    digests[GRID_FILE] = params["labels_grid"]["sha256"]
    return {"inputs_run": INPUTS_RUN, "labels_run": LABELS_RUN, "sha256": digests}


def write(commit: str, out: Path, store: Path) -> None:
    if not COMMIT.fullmatch(commit):
        raise SystemExit(f"--commit must be 40 hex digits, not {commit!r}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    data = recorded_data(store)
    dry = {"kernel": "tp-panel-seq-dry", "mode": "dry", "commit": commit}
    write_kernel(dry, out / dry["kernel"], [])
    for piece in pieces():
        job = {"kernel": slug(piece), "mode": "piece", "commit": commit, "piece": piece}
        job["data"] = data
        write_kernel(job, out / job["kernel"], [f"{OWNER}/{DATASET}"])
    print("\n".join(sorted(p.name for p in out.iterdir())))


def local_dry(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} is not empty")
    root.mkdir(parents=True, exist_ok=True)
    began = time.time()
    found = dry_steps(sys.executable, Path(__file__).resolve(), child_env(None), root, root)
    found["machine"] = machine()
    found["seconds"] = round(time.time() - began, 1)
    (root / "dry.json").write_text(json.dumps(found, indent=1), encoding="utf-8")
    print(json.dumps(found["bench"]))


def main(argv: list[str] | None = None) -> int:
    if JOB_JSON is not None:
        return kernel(json.loads(JOB_JSON))
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("write")
    p.add_argument("--commit", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, help="the local store (default $TP_DATA_DIR)")
    local = sub.add_parser("local-dry", help="the dry run's steps here; needs a clean checkout")
    local.add_argument("root", type=Path)
    sub.add_parser("synthetic").add_argument("data", type=Path)
    sub.add_parser("bench")
    guard = sub.add_parser("guard")
    guard.add_argument("flat", type=Path)
    guard.add_argument("names", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "write":
        from tipping_point import engine

        write(args.commit, args.out, args.data_dir or engine.data_dir())
    elif args.command == "local-dry":
        local_dry(args.root)
    elif args.command == "synthetic":
        write_synthetic(args.data)
    elif args.command == "guard":
        guard_dataset(args.flat, args.names)
    else:
        bench()
    return 0


if __name__ == "__main__":
    sys.exit(main())
