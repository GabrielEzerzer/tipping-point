"""Planted Ising runs for the EWS adapter and scorer tests: the simulator's real layout (io's
writers, protocol's schedules), with |m| and e drawn from processes whose answers are known
instead of simulated, so a scoring run takes seconds and its numbers can be checked.

The dataset (L = 16 and 32, seeds 1 and 2, 8 replicas):
  ramps       3.5 -> 1.5; |m| an AR(1) whose spread and memory grow as eps_L falls to 0
  nulls       fixed runs, blocks at eps_L = 1.5 and 2.5; the same AR(1) frozen at that eps_L
  staircases  plateaus at 3.0 and 2.269231; at the second, L^2 Var(|m|) ~ L^1.75,
              <|m|> ~ L^-0.125, constant skewness, and a memory of L^2/512 sweeps
  pilot.json  tau_c = 4 and 8 sweeps, in a pilot run folder
"""

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from tipping_point.ising import io, protocol
from tipping_point.ising.exact import TC

SIZES, SEEDS, REPLICAS = (16, 32), (1, 2), 8
TAU_C = {16: 4.0, 32: 8.0}
RATES = {16: 700, 32: 2800}  # sweeps per unit T: the segment holds the same samples at each L
NULL_EPS = (1.5, 2.5)
PLATEAU = 2.269231
PILOT_RUN = "20261023T000000Z-pilot"

Values = Callable[[np.ndarray, np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]]


def _totals(
    abs_m: np.ndarray, e: np.ndarray, L: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Integer totals the writer accepts: M even, |M| <= L^2, either sign; E = -2 L^2 (mod 4)."""
    area = L * L
    M = 2 * np.rint(np.clip(abs_m, 0.0, 1.0) * area / 2).astype(np.int64)
    M *= rng.choice(np.array([-1, 1]), size=M.shape)
    E = -2 * area + 4 * np.rint((np.clip(e, -2.0, 2.0) + 2.0) * area / 4).astype(np.int64)
    return M.astype(np.int32), E.astype(np.int32)


def write_run(
    root: Path,
    tag: str,
    schedule: protocol.Schedule,
    L: int,
    seed: int,
    values: Values,
    failed: tuple[float, ...] = (),
    complete: bool = True,
    verdicts: bool = True,
    pilot: dict[str, Any] | None = None,
) -> Path:
    """A run folder as the simulator leaves it, with per-site |m| and e from
    values(steps, temperatures (steps, lattices), rng) -> two (steps, lattices) arrays.
    `verdicts` = False leaves complete.json without thermalised checks, as before #50; `pilot`
    is the pilot-record stamp a ramp or null run's manifest carries (#55)."""
    rng = np.random.default_rng([seed, L, len(tag)])
    run_dir = root / "ising" / f"20261023T000000Z-{tag}"
    run_dir.mkdir(parents=True)
    lattices = REPLICAS * schedule.blocks
    manifest = {
        "format": 1,
        "domain": "ising",
        "run_id": run_dir.name,
        "producer": {"name": "planted", "version": "0", "git_sha": None, "dirty": None},
        "started_ns": 0,
        "key": "step",
        "tables": ["stats", "configs"] if schedule.frames else ["stats"],
        "params": {
            "L": L,
            "replicas": REPLICAS,
            "seed": seed,
            "blocks": schedule.blocks,
            "Tc": TC,
            "schedule": schedule.to_json(),
            **({"pilot": pilot} if pilot is not None else {}),
        },
    }
    io.write_json(run_dir / "manifest.json", manifest)
    steps = np.arange(1, schedule.total + 1)
    temps = np.array([schedule.temperatures(int(t)) for t in steps])
    per_lattice = np.repeat(temps, REPLICAS, axis=1)
    abs_m, e = values(steps, per_lattice, rng)
    M, E = _totals(abs_m, e, L, rng)

    def keys(rows: np.ndarray) -> dict[str, np.ndarray]:
        picked = steps[rows]
        return {
            "step": np.repeat(picked, lattices),
            "replica": np.tile(np.arange(lattices), len(picked)),
            "temperature": per_lattice[rows].ravel(),
            "segment": np.repeat([schedule.segment(int(t)) for t in picked], lattices),
        }

    stats = np.array([schedule.records_stats(int(t)) for t in steps])
    columns = keys(stats)
    columns["phase"] = np.repeat([schedule.phase(int(t)) for t in steps[stats]], lattices)
    columns["magnetisation"], columns["energy"] = M[stats].ravel(), E[stats].ravel()
    io.write_chunk(run_dir, "stats", L, columns)
    if schedule.frames:
        saved = np.array([int(t) in schedule.frames for t in steps])
        columns = keys(saved)
        burst, frame = np.array([schedule.frames[int(t)] for t in steps[saved]]).T
        columns["burst"], columns["frame"] = np.repeat(burst, lattices), np.repeat(frame, lattices)
        columns["magnetisation"], columns["energy"] = M[saved].ravel(), E[saved].ravel()
        columns["spins"] = np.zeros((saved.sum() * lattices, io.packed_width(L)), dtype=np.uint8)
        io.write_chunk(run_dir, "configs", L, columns)
    if complete:
        # complete.json's thermalised verdicts: one per staircase plateau, or per null block
        plateaus = []
        if not verdicts:
            pass
        elif schedule.kind == "staircase":
            plateaus = sorted({float(T) for T in temps[:, 0]})
        elif schedule.kind == "fixed":
            plateaus = [float(T) for T in temps[0]]
        checks = [
            {
                "gate": "thermalised",
                "temperature": T,
                "passed": T not in failed,
                "detail": "planted",
            }
            for T in plateaus
        ]
        io.write_complete(run_dir, schedule.total, {"thermalised": checks} if checks else {})
    return run_dir


def _ar1(phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance AR(1) down axis 0 with a coefficient per step and lattice."""
    x = np.empty(phi.shape)
    x[0] = rng.standard_normal(phi.shape[1])
    noise = rng.standard_normal(phi.shape)
    for t in range(1, len(phi)):
        x[t] = phi[t] * x[t - 1] + np.sqrt(1 - phi[t] ** 2) * noise[t]
    return x


def ramp_values(L: int) -> Values:
    """|m| near 0.3 whose spread and memory grow as eps_L falls towards 0, flat below."""

    def values(
        steps: np.ndarray, temps: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        eps = np.clip(L * (temps - TC) / TC, 0.0, 10.0)
        spread = 0.01 + 0.05 / (1.0 + 2.0 * eps)
        phi = 0.95 - 0.3 * eps / 10.0
        abs_m = 0.3 + spread * _ar1(phi, rng)
        return abs_m, -1.4 + 0.02 * rng.standard_normal(abs_m.shape)

    return values


def plateau_values(L: int) -> Values:
    """At the plateau nearest Tc: <|m|> = 0.8 L^-1/8, Var(|m|) = (0.06 L^-1/8)^2 so that
    L^2 Var ~ L^1.75, a fixed negative skewness (a mirrored log-normal), and a memory of
    L^2/512 sweeps; elsewhere plain AR(1)."""

    def values(
        steps: np.ndarray, temps: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        tau, s2 = L**2 / 512.0, 0.09  # s2: the log-normal's log-variance
        g = _ar1(np.full(temps.shape, np.exp(-1.0 / tau)), rng)
        lognormal = np.exp(np.sqrt(s2) * g)
        skewed = -(lognormal - np.exp(s2 / 2)) / np.sqrt((np.exp(s2) - 1) * np.exp(s2))
        near = np.isclose(temps, PLATEAU)
        abs_m = np.where(near, 0.8 * L**-0.125 + 0.06 * L**-0.125 * skewed, 0.2 + 0.02 * g)
        return abs_m, -1.4 + 0.02 * rng.standard_normal(abs_m.shape)

    return values


def ramp(L: int) -> protocol.Ramp:
    """The slow ramp at L; its size is passed when Ramp takes one (#50 spaces saves in eps_L)."""
    sized: dict[str, Any] = {"L": L} if "L" in inspect.signature(protocol.Ramp).parameters else {}
    return protocol.Ramp(RATES[L], pre=20, hold=20, x=0.02, **sized)


def burst_gap(schedule: protocol.Schedule, L: int, top: float = 3.0, bottom: float = 0.0) -> int:
    """Sweeps between the ramp's burst starts while the bath is between eps_L = top and bottom."""
    starts = sorted(t for t, (_, frame) in schedule.frames.items() if frame == 0)
    inside = [t for t in starts if bottom <= L * (schedule.temperatures(t)[0] - TC) / TC <= top]
    return int(np.median(np.diff(inside)))


def nulls(L: int, sweeps: int) -> protocol.Fixed:
    """Blocks at eps_L = 1.5 and 2.5, saving bursts as densely as the ramp does in its analysis
    band, so the matched cadence has the same frames per sweep in both."""
    temps = [TC * (1 + e / L) for e in NULL_EPS]
    return protocol.Fixed(temps, therm=20, sweeps=sweeps, saves=sweeps // burst_gap(ramp(L), L))


def staircase() -> protocol.Staircase:
    return protocol.Staircase(
        [2.0, 2.0], temperatures=(3.0, PLATEAU), warmup=0, min_therm=50, min_meas=2000, saves=100
    )


def pilot(root: Path, per_L: dict[int, float] | None = None) -> Path:
    """The #50 pilot record's shape, in a pilot run's folder."""
    folder = root / "ising" / PILOT_RUN
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "kernel_commit": "0" * 40,
        "per_L": {
            str(L): {"tau_c": tau, "tau_c_binning": tau, "tau_e": tau / 2, "n_over_tau": 1000.0,
                     "replicas": 16, "sweeps": 50_000, "seed": 1}
            for L, tau in (per_L or TAU_C).items()
        },
        "fit": {"A": 0.016, "z": 2.0, "A_ci": [0.015, 0.017], "z_ci": [1.9, 2.1]},
    }  # fmt: skip
    path = folder / "pilot.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path
