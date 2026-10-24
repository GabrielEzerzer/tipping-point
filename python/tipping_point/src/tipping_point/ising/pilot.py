"""The pilot record: tau_c(L), the integrated autocorrelation time of |m| at exactly Tc on one
kernel, and its fit tau_c = A L^z. Ramp rates, null-run lengths and the EWS scorer's time unit
read tau_c from it, never from a constant.

`python -m tipping_point.ising tauc <run dir> ...` builds it from complete `pilot` runs, one per
L, each holding a block at exactly Tc with a stats row every sweep, and writes
<data dir>/ising/<run_id>/pilot.json:

  {
    "kernel_commit": "<sha>",   commit the pilot runs came from; "<sha>-dirty" if any of their
                                trees had uncommitted edits
    "kernel_sha256": "<hex>",   sha256 of the kernel source they ran (kernel.py: the sweep and its
                                acceptance table), as their manifests record it
    "per_L": {                  one entry per size, keyed by str(L):
      "16": {
        "tau_c": ...,           Sokal tau_int of |m|, 1 + 2 sum rho(t) with window c = 5, sweeps
        "tau_c_ci": [lo, hi],   95% bootstrap over replicas; optional for readers
        "tau_c_binning": ...,   the binning cross-check of tau_c
        "tau_e": ...,           Sokal tau_int of the energy per site
        "n_over_tau": ...,      measured sweeps / tau_c, at least MIN_N_OVER_TAU = 300
        "replicas": ..., "sweeps": ..., "seed": ...   the pilot run's
      }, ...
    },
    "fit": {"A": ..., "z": ..., "A_ci": [lo, hi], "z_ci": [lo, hi]}
                                tau_c = A L^z, least squares in log-log over the sizes; 95% CIs
                                from refitting a bootstrap over each size's replicas
  }

`tau_c(record, L)` reads a measured size from per_L and any other size from the fit.

`read` refuses a stale record, one whose kernel_sha256 is not the running kernel's: its tau_c
clocks another kernel. A record without the field (written before #55) counts as stale. The
commit cannot decide it, since it moves with every merge whether the kernel changed or not.
`ramp` and `null` take --allow-stale-pilot to use one anyway, and their manifests carry `stamp`,
which records whether it was.

Subtracting each series' mean pulls the windowed estimate about 10 / (N / tau) low (2c over
N / tau, c = 5), so a size is recorded only from N >= 300 tau_c: at most ~3% short. The record's
L = 64 entry (343 tau_c) reads ~3% low and L = 16 (960 tau_c) ~1%; test_autocorr pins both.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import engine
from . import autocorr, io
from .exact import TC

DRAWS = 2000  # bootstrap resamples per size
MIN_N_OVER_TAU = 300  # measured sweeps per tau_c below which the estimate reads > 3% short
_QUANTILES = (0.025, 0.975)  # 95% intervals
_ENTRY_KEYS = ("tau_c", "tau_c_binning", "tau_e", "n_over_tau", "replicas", "sweeps", "seed")
_SHA256 = re.compile(r"[0-9a-f]{64}")
KERNEL = Path(__file__).with_name("kernel.py")  # the dynamics tau_c measures


def kernel_sha256() -> str:
    """sha256 of the running kernel's source: the kernel a record must have measured."""
    return hashlib.sha256(KERNEL.read_bytes()).hexdigest()


def series(columns: dict[str, np.ndarray], T: float, L: int) -> tuple[np.ndarray, np.ndarray]:
    """(replicas, steps) arrays of |m| and e per site, from a run's measured stats rows at T."""
    mine = (columns["phase"] == 1) & (columns["temperature"] == T)
    steps = len(np.unique(columns["step"][mine]))
    # rows are in (step, replica) order: reshape, then one row per replica
    m = np.abs(columns["magnetisation"][mine]).reshape(steps, -1).T / L**2
    e = columns["energy"][mine].reshape(steps, -1).T / L**2
    return m, e


def measure(
    run_dir: Path, draws: int = DRAWS, seed: int = 0
) -> tuple[int, dict[str, Any], np.ndarray, str, str]:
    """One size from a complete run with a block at exactly Tc: L, its per_L entry, the
    bootstrap taus of |m|, the run's commit and its kernel's sha256."""
    if not (run_dir / "complete.json").exists():
        raise ValueError(f"{run_dir.name} is not complete")
    manifest = io.read_json(run_dir / "manifest.json")
    L, producer = manifest["params"]["L"], manifest["producer"]
    kernel = manifest["params"]["algorithm"].get("kernel_sha256")
    if producer["git_sha"] is None:
        raise ValueError(f"{run_dir.name} records no git commit, so no kernel to name")
    if kernel is None:
        raise ValueError(f"{run_dir.name} records no kernel_sha256 (run before #55); re-run it")
    stats = io.read_table(run_dir, "stats")
    columns = {name: stats[name].to_numpy() for name in stats.column_names}
    at_tc = (columns["phase"] == 1) & (columns["temperature"] == TC)
    if not at_tc.any():
        raise ValueError(f"{run_dir.name} measured no block at exactly Tc = {TC!r}")
    if np.any(np.diff(np.unique(columns["step"][at_tc])) != 1):
        raise ValueError(f"{run_dir.name}: tau_c in sweeps needs a stats row every sweep")
    m, e = series(columns, TC, L)
    tau_m, tau_e = autocorr.estimate(m), autocorr.estimate(e)
    if not tau_m.reliable:
        raise ValueError(f"{run_dir.name}: tau of |m| at Tc is unreliable ({tau_m}); run longer")
    if m.shape[1] < MIN_N_OVER_TAU * tau_m.tau:
        raise ValueError(
            f"{run_dir.name}: {m.shape[1] / tau_m.tau:.0f} tau_c of |m| at Tc, need "
            f"{MIN_N_OVER_TAU}: the estimate reads ~10 / (N / tau) short; run longer"
        )
    boot = autocorr.bootstrap(m, draws, np.random.default_rng([seed, L]))
    entry = {
        "tau_c": tau_m.tau,
        "tau_c_ci": np.quantile(boot, _QUANTILES).tolist(),
        "tau_c_binning": tau_m.binning,
        "tau_e": tau_e.tau,
        "n_over_tau": m.shape[1] / tau_m.tau,
        "replicas": m.shape[0],
        "sweeps": m.shape[1],
        "seed": manifest["params"]["seed"],
    }
    commit = producer["git_sha"] + ("-dirty" if producer["dirty"] else "")
    return L, entry, boot, commit, kernel


def fit(L: Sequence[int], tau: np.ndarray) -> tuple[Any, Any]:
    """Least squares of log tau = log A + z log L along the last axis of tau: (A, z)."""
    x = np.log(np.asarray(L, dtype=np.float64))
    y = np.log(tau)
    dx = x - x.mean()
    z = (y - y.mean(axis=-1, keepdims=True)) @ dx / (dx @ dx)
    return np.exp(y.mean(axis=-1) - z * x.mean()), z


def record(run_dirs: Sequence[Path], draws: int = DRAWS, seed: int = 0) -> dict[str, Any]:
    """The pilot record of complete pilot runs at Tc, one per L, at least two sizes."""
    results = sorted((measure(Path(d), draws, seed) for d in run_dirs), key=lambda r: r[0])
    sizes = [L for L, *_ in results]
    if len(sizes) < 2 or len(set(sizes)) < len(sizes):
        raise ValueError(f"need one pilot run per L and at least two sizes, got L = {sizes}")
    commits, kernels = {r[3] for r in results}, {r[4] for r in results}
    if len(commits) > 1:
        raise ValueError(f"pilot runs come from different commits: {sorted(commits)}")
    if len(kernels) > 1:
        raise ValueError(f"pilot runs ran different kernels: {sorted(kernels)}")
    A, z = fit(sizes, np.array([r[1]["tau_c"] for r in results]))
    A_draws, z_draws = fit(sizes, np.stack([r[2] for r in results], axis=1))
    return {
        "kernel_commit": commits.pop(),
        "kernel_sha256": kernels.pop(),
        "per_L": {str(r[0]): r[1] for r in results},
        "fit": {
            "A": float(A),
            "z": float(z),
            "A_ci": np.quantile(A_draws, _QUANTILES).tolist(),
            "z_ci": np.quantile(z_draws, _QUANTILES).tolist(),
        },
    }


def write(record: dict[str, Any], root: Path) -> Path:
    """Write <root>/ising/<run_id>/pilot.json, the run_id naming its sizes; return the path."""
    sizes = "-".join(record["per_L"])
    run_dir = Path(root) / "ising" / engine.run_id(f"tauc-L{sizes}")
    run_dir.mkdir(parents=True)
    io.write_json(run_dir / "pilot.json", record)
    return run_dir / "pilot.json"


def read(path: Path, allow_stale: bool = False) -> dict[str, Any]:
    """A pilot record, refused unless it has every field above (tau_c_ci may be absent; a missing
    kernel_sha256 makes it stale), every tau_c, A and z is a finite positive number, and every
    size held at least 300 tau_c; refused when stale too, unless `allow_stale`."""
    record = io.read_json(Path(path))
    per_L, fit = record.get("per_L"), record.get("fit")
    if not (
        isinstance(record.get("kernel_commit"), str)
        and isinstance(per_L, dict)
        and per_L
        and isinstance(fit, dict)
    ):
        raise ValueError(f"{path}: not a pilot record (kernel_commit, per_L, fit)")
    for L, entry in per_L.items():
        if not (
            L.isdigit()
            and isinstance(entry, dict)
            and all(key in entry for key in _ENTRY_KEYS)
            and _positive(entry["tau_c"])
            and _positive(entry["n_over_tau"])
            and entry["n_over_tau"] >= MIN_N_OVER_TAU
        ):
            raise ValueError(f"{path}: bad per_L entry {L}: {entry}")
    if not (
        all(_positive(fit.get(key)) for key in ("A", "z"))
        and all(len(fit.get(key) or ()) == 2 for key in ("A_ci", "z_ci"))
    ):
        raise ValueError(f"{path}: bad fit {fit}")
    kernel = record.get("kernel_sha256")
    if kernel is not None and not (isinstance(kernel, str) and _SHA256.fullmatch(kernel)):
        raise ValueError(f"{path}: bad kernel_sha256 {kernel!r}")
    why = stale(record)
    if why and not allow_stale:
        raise ValueError(
            f"{path} is stale: {why}. Its tau_c may belong to another kernel: re-run the pilot "
            "and tauc, or pass --allow-stale-pilot to use it anyway"
        )
    return record


def stale(record: dict[str, Any]) -> str | None:
    """Why a record did not measure the running kernel, or None when it did."""
    kernel = record.get("kernel_sha256")
    if kernel is None:
        return "it records no kernel_sha256 (it was written before #55)"
    if kernel != kernel_sha256():
        return f"it measured kernel {kernel[:12]}, not the running kernel.py {kernel_sha256()[:12]}"
    return None


def stamp(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """What a run's manifest records of the pilot record it read: its run id (the folder it sits
    in, never a local path) and sha256, the kernel it measured, and whether that was stale
    (possible only with --allow-stale-pilot)."""
    return {
        "run_id": Path(path).resolve().parent.name,  # resolved: a bare "pilot.json" has a folder
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "kernel_commit": record["kernel_commit"],
        "kernel_sha256": record.get("kernel_sha256"),
        "stale": stale(record) is not None,
    }


def tau_c(record: dict[str, Any], L: int) -> float:
    """tau_c at size L: its measured entry, else the fit A L^z (beyond the pilot sizes, an
    extrapolation)."""
    entry = record["per_L"].get(str(L))
    if entry is not None:
        return float(entry["tau_c"])
    return float(record["fit"]["A"] * L ** record["fit"]["z"])


def _positive(value: object) -> bool:
    """A finite positive number (NaN fails both comparisons; a bool is not a number here)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and 0 < value < math.inf
