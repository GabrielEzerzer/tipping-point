# %% [markdown]
# # Tipping Point: JEPA training on Kaggle (#27, #48)
#
# A Kaggle script kernel (or paste the cells into a notebook): one model seed per T4, everything
# kept in `/kaggle/working`, which saving the version turns into its output. Not executed by the
# repository's gate; the CUDA checks it relies on are #48's.
#
# Its runs record the pinned commit through git, since it installs the checkout itself. A kernel
# that installs a wheel instead needs one built with the commit stamp (README, #109).
#
# Inputs:
# - a private Kaggle Dataset of simulator runs, laid out as `ising/<run_id>/...` with their
#   `complete.json`, plus the #50 pilot run's `pilot.json`;
# - the Kaggle Secret `GITHUB_TOKEN`: a fine-grained token, read-only, this repository only;
# - optionally the previous version's output, attached as an input, to resume a killed session.
#
# Afterwards, locally: `kaggle kernels output <user>/<kernel> -p $TP_DATA_DIR`, then
# `python -m tipping_point.jepa verify <training>` before anything reads it.

# %%
import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/GabrielEzerzer/tipping-point.git"
COMMIT = "<pin the exact commit to train>"  # a commit sha, never a branch
DATASET = Path("/kaggle/input/tipping-point-ising")  # holds ising/<run_id>/
PILOT = DATASET / "ising" / "<pilot run>" / "pilot.json"
PREVIOUS = Path("/kaggle/input/tipping-point-jepa-output")  # last version's output, if any
WORK = Path("/kaggle/working")
CODE, DATA = WORK / "tipping-point", WORK / "data"  # DATA is TP_DATA_DIR
SIZES, TRAIN_SEEDS, VAL_SEED = [16, 32, 64], [101, 102], 103  # simulator seeds, never 1-5
MODEL_SEEDS = [1, 2]  # one per T4
STEPS, HELD_OUT = 50_000, False  # HELD_OUT: the [2.0, 2.5] ablation


def run(*argv: str, **kwargs: object) -> None:
    subprocess.run(argv, check=True, **kwargs)  # type: ignore[call-overload]


# %% [markdown]
# ## Code: the pinned commit, installed with the cu126 extra
#
# The token travels as an HTTP header in git's environment, never in a command line, a URL or
# `.git/config`, so no error message or log can print it.

# %%
from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]  # noqa: E402

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
git_env = dict(
    os.environ,
    GIT_CONFIG_COUNT="1",
    GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
    GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {basic}",
)
if not CODE.exists():
    run("git", "clone", "--quiet", REPO, str(CODE), env=git_env)
run("git", "-C", str(CODE), "fetch", "--quiet", "origin", COMMIT, env=git_env)
run("git", "-C", str(CODE), "checkout", "--quiet", "--detach", COMMIT)
run(sys.executable, "-m", "pip", "install", "--quiet", "uv")
run("uv", "sync", "--frozen", "--extra", "cu126", cwd=CODE)
PY = str(CODE / ".venv" / "bin" / "python")
run(PY, "-c", "import torch; print(torch.__version__, torch.cuda.get_arch_list())")
run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv")

# %% [markdown]
# ## Data: the dataset's runs under TP_DATA_DIR, every file checked against complete.json

# %%
env = dict(os.environ, TP_DATA_DIR=str(DATA), CUBLAS_WORKSPACE_CONFIG=":4096:8")
(DATA / "ising").mkdir(parents=True, exist_ok=True)
runs = sorted(p for p in (DATASET / "ising").iterdir() if (p / "manifest.json").is_file())
for source in runs:  # read-only input: training only reads it, so links suffice
    link = DATA / "ising" / source.name
    if not link.exists():
        link.symlink_to(source, target_is_directory=True)
complete = [str(DATA / "ising" / p.name) for p in runs if (p / "complete.json").is_file()]
run(PY, "-m", "tipping_point.jepa", "verify", *complete, env=env)

# %% [markdown]
# ## Resume: the previous version's trainings and GPU-hour ledger, copied back (input is
# read-only); an unfinished training continues from its checkpoint instead of starting over

# %%
(DATA / "jepa").mkdir(parents=True, exist_ok=True)
if (PREVIOUS / "data" / "jepa").is_dir():
    for path in (PREVIOUS / "data" / "jepa").iterdir():
        target = DATA / "jepa" / path.name
        if path.is_dir() and not target.exists():
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)  # the ledger: the previous sessions' hours


def unfinished(seed: int) -> Path | None:
    tag = f"m{seed}-heldout" if HELD_OUT else f"m{seed}"
    for training in sorted((DATA / "jepa").glob(f"*-L{'-'.join(map(str, SIZES))}-{tag}")):
        if (training / "checkpoint.pt").is_file() and not (training / "complete.json").exists():
            return training
    return None


# %% [markdown]
# ## Train: one process per T4, each pinned to its GPU, logs beside the output

# %%
processes = []
try:
    for gpu, seed in enumerate(MODEL_SEEDS):
        prior = unfinished(seed)
        if prior is not None:
            argv = [PY, "-m", "tipping_point.jepa", "resume", str(prior)]
        else:
            argv = [PY, "-m", "tipping_point.jepa", "train", "--L", *map(str, SIZES)]
            argv += ["--train-seeds", *map(str, TRAIN_SEEDS), "--val-seed", str(VAL_SEED)]
            argv += ["--seed", str(seed), "--steps", str(STEPS), "--pilot", str(PILOT)]
            argv += ["--held-out"] if HELD_OUT else []
        log = open(WORK / f"train-m{seed}.log", "a", encoding="utf-8")  # noqa: SIM115
        child = subprocess.Popen(
            argv,
            cwd=CODE,
            env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((child, log))
    codes = [child.wait() for child, _ in processes]
finally:  # a failed or interrupted cell never leaves a training running on a GPU
    for child, log in processes:
        if child.poll() is None:
            child.kill()
            child.wait()
        log.close()
print("exit codes:", codes)

# %% [markdown]
# ## Results: checked, and the week's GPU hours against the quota

# %%
finished = [str(p.parent) for p in sorted((DATA / "jepa").glob("*/complete.json"))]
if finished:
    run(PY, "-m", "tipping_point.jepa", "verify", *finished, env=env)
run(PY, "-m", "tipping_point.jepa", "budget", "--data-dir", str(DATA), env=env)
for name in (p.name for p in runs):  # the links point into read-only input: not output
    (DATA / "ising" / name).unlink()
