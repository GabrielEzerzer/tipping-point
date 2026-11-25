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

    uv run --no-sync python notebooks/kaggle_panel_sequence.py write --commit <40 hex> --out <dir>
    uv run --no-sync python notebooks/kaggle_panel_sequence.py local-dry <empty dir>

`write` makes one kernel folder per piece (tp-panel-seq-<variant>-s<seed>-f<folds>) and one for
the dry run (tp-panel-seq-dry), each this file with its job embedded and a kernel-metadata.json:
private, CPU, internet on (to fetch the code and the pinned packages). `kaggle kernels push -p
<folder>` starts one. A piece is the shipped plan (panel/sequence.toml) with its variants, seeds
and folds cut to one variant, one seed and three folds; the twelve pieces are every cell of the
full plan once, so `python -m tipping_point.panel merge` joins them into the full run.

Every kernel clones the public repository at the job's commit and refuses to run unless that
checkout's HEAD is the commit and it is clean. It runs from that checkout, so each run's manifest
records the commit (producer.git_sha) and the sha256 of its uv.lock, in a Python 3.13 environment
of torch 2.14.1+cpu and uv.lock's numpy and pyarrow, with OMP_NUM_THREADS=2 and the plan's
threads = 2, one process per kernel. A piece reads the private dataset tp-panel-sequence-data,
which holds only the files the sequence run reads: the inputs run's manifest.json, complete.json
and inputs.parquet, and the labels run's grid file. It writes its run to
out/panel-sequence/<run id>/ and its log to out/logs/; `kaggle kernels output <owner>/<slug> -p
<dir>` fetches out/. out/STATUS holds one line: running, complete or failed.

The dry run reads no dataset. On a synthetic panel whose labels are planted on one input, it runs
two pieces of a small plan, merges them, scores the merged run, and then times a fixed number of
JEPA and supervised steps at the shipped plan's shapes on 30 synthetic coins (out/dry.json).
`local-dry` runs the same steps on this machine in the current environment, for the speed ratio.
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
BENCH = {"coins": 30, "hours": 3000, "jepa_steps": 60, "supervised_steps": 60}
COMMIT = re.compile(r"[0-9a-f]{40}")

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
target = "future"
threads = 2

[window]
length = 36
future = 4
shifts = [2, 4]

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
    env = dict(os.environ, OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS)
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
        " 'threads': torch.get_num_threads()}))\n"
    )
    info = json.loads(run([py, "-c", probe], child_env(src)).strip().splitlines()[-1])
    if info["code"] != {"git_sha": commit, "dirty": False}:
        raise RuntimeError(f"the checkout is {info['code']}, not commit {commit} clean")
    return py, src, info


def verify(run_dir: Path, commit: str | None) -> dict[str, Any]:
    complete = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bad = [n for n, d in complete["files"].items() if sha256(run_dir / n) != d]
    if complete.get("clean") is not True or bad:
        raise RuntimeError(f"{run_dir.name}: not clean, or files differ: {bad}")
    producer = manifest["producer"]
    if commit is not None and (producer["git_sha"], producer["dirty"]) != (commit, False):
        raise RuntimeError(f"{run_dir.name}: made by {producer}, not commit {commit} clean")
    return {"run": run_dir.name, "files": len(complete["files"]), "checks": complete["checks"]}


def new_run(folder: Path, before: set[str]) -> Path:
    (made,) = [p for p in folder.iterdir() if p.name not in before]
    return made


def find_dataset(root: Path) -> Path:
    hits = sorted(p.parent for p in root.rglob("inputs.parquet"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one inputs.parquet under {root}, found {hits}")
    return hits[0]


def piece_job(job: dict[str, Any], py: str, src: Path, tmp: Path, out: Path) -> dict[str, Any]:
    flat = find_dataset(Path("/kaggle/input"))
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
    return {"seconds": round(time.time() - began, 1), **verify(made, job["commit"])}


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
    for name, folds in (("a", "[2]"), ("b", "[3]")):
        folder = data / "plans" / name
        folder.mkdir(parents=True)
        (folder / "ladder.toml").write_text(DRY_LADDER, encoding="utf-8")
        (folder / "sequence.toml").write_text(DRY_PLAN.replace("FOLDS", folds), encoding="utf-8")
    print(json.dumps({"inputs": run_dir.name}))


def bench() -> None:
    from dataclasses import replace

    import numpy as np
    import torch
    from tipping_point.panel import sequence

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
    jepa = replace(plan, jepa=replace(plan.jepa, steps=BENCH["jepa_steps"]))
    began = time.perf_counter()
    sequence.pretrain(x, fit, jepa, 0, np.random.SeedSequence(0), lambda line: None)
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
    run([*panel, "merge", "--runs", *pair, "--data-dir", data], env, logs / "dry.log")
    found["merged"] = verify(new_run(runs, before), None)
    argv = [*panel, "scores", "--sequence", found["merged"]["run"], "--model", "jepa"]
    run([*argv, "--variant", "vol", "--data-dir", data], env, logs / "dry.log")
    (scores,) = (data / "panel-scores").iterdir()
    found["scores"] = verify(scores, None)
    found["bench"] = json.loads(run([py, script, "bench"], env).strip().splitlines()[-1])
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


def write(commit: str, out: Path) -> None:
    if not COMMIT.fullmatch(commit):
        raise SystemExit(f"--commit must be 40 hex digits, not {commit!r}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    dry = {"kernel": "tp-panel-seq-dry", "mode": "dry", "commit": commit}
    write_kernel(dry, out / dry["kernel"], [])
    for piece in pieces():
        job = {"kernel": slug(piece), "mode": "piece", "commit": commit, "piece": piece}
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
    sub.add_parser("local-dry").add_argument("root", type=Path)
    sub.add_parser("synthetic").add_argument("data", type=Path)
    sub.add_parser("bench")
    args = parser.parse_args(argv)
    if args.command == "write":
        write(args.commit, args.out)
    elif args.command == "local-dry":
        local_dry(args.root)
    elif args.command == "synthetic":
        write_synthetic(args.data)
    else:
        bench()
    return 0


if __name__ == "__main__":
    sys.exit(main())
