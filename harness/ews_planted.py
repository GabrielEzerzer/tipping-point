"""Planted Ising runs for the EWS adapter and scorer tests: the simulator's real layout (io's
writers, protocol's schedules), with |m| and e drawn from processes whose answers are known
instead of simulated, so a scoring run takes seconds and its numbers can be checked.

The dataset (L = 16 and 32, seeds 1 and 2, 8 replicas):
  ramps       3.5 -> 1.5; |m| an AR(1) whose spread and memory grow as eps_L falls to 0
  nulls       fixed runs, blocks at eps_L = 1.5 and 2.5; the same AR(1) frozen at that eps_L
  staircases  plateaus at 3.0 and 2.269231; at the second, L^2 Var(|m|) ~ L^1.75,
              <|m|> ~ L^-0.125, constant skewness, and a memory of L^2/512 sweeps
  pilot.json  tau_c = 4 and 8 sweeps, in a pilot run folder

The ising-ews v1 dataset (v1_dataset, below the first) is the freeze's: L = 16, 32 and 64, ramps
over the scaled window eps_L = 12 -> -8, and a spread of |m| that peaks where a mode plants it.
"""

import inspect
import json
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tipping_point.ising import io, protocol
from tipping_point.ising import pilot as tauc
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
    replicas: int = REPLICAS,
    rng: np.random.Generator | None = None,
    git_sha: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """A run folder as the simulator leaves it, with per-site |m| and e from
    values(steps, temperatures (steps, lattices), rng) -> two (steps, lattices) arrays.
    `verdicts` = False leaves complete.json without thermalised checks, as before #50; `pilot`
    is the pilot-record stamp a ramp or null run's manifest carries (#55). `git_sha` and `dirty`
    go where the simulator puts the commit it ran from (producer.git_sha, producer.dirty); `rng`
    replaces the default stream, which runs whose tags have the same length share."""
    rng = rng if rng is not None else np.random.default_rng([seed, L, len(tag)])
    run_dir = root / "ising" / f"20261023T000000Z-{tag}"
    run_dir.mkdir(parents=True)
    lattices = replicas * schedule.blocks
    manifest = {
        "format": 1,
        "domain": "ising",
        "run_id": run_dir.name,
        "producer": {"name": "planted", "version": "0", "git_sha": git_sha, "dirty": dirty},
        "started_ns": 0,
        "key": "step",
        "tables": ["stats", "configs"] if schedule.frames else ["stats"],
        "params": {
            "L": L,
            "replicas": replicas,
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
    per_lattice = np.repeat(temps, replicas, axis=1)
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


# ---------------------------------------------------------------- the ising-ews v1 dataset
#
# The freeze's scoring (#97, #98; decisions: https://github.com/GabrielEzerzer/tipping-point/
# issues/26#issuecomment-6402030372) on answers planted in advance. Each mode is a data dir of its
# own at L = 16, 32 and 64, whose slow ramps carry a spread of |m| that peaks
#   fss      at eps_L = 0.8 at every L: finite-size scaling, 1/nu = 1
#   fixed_T  at one temperature, Tc (1 + 0.8/16), so at eps_L = 0.8, 1.6 and 3.2: an indicator
#            merely monotone in T, 1/nu = 0
#   edge     as in fss, except that at L = 32 two of each seed's three replicas carry a spread
#            that rises through the whole search band [6, -4] (it peaks past the cold edge, at
#            eps_L = -4.15), so their extremum sits on the band's edge and is not located. Over
#            half of one L unlocated makes the verdict inconclusive, while the located peaks
#            alone give 1/nu = 1 and would pass: the edge rule, read at any one L, decides it.
#            Were every peak on an edge, 1/nu would have no value and the verdict would be
#            inconclusive with or without the rule.

V1_SIZES, V1_SEEDS, V1_REPLICAS = (16, 32, 64), (6, 7), 3
V1_TAU_C = {16: 1.0, 32: 1.5, 64: 2.0}  # sweeps: a different number of samples per tau_c at each L
V1_PILOT_RUN = "20261025T000000Z-tauc-L16-32-64"
V1_X = 0.02  # the slow ramp: 1/x = 50 measured tau_c per unit eps_L at every L
V1_EPS_L = (12.0, -8.0)  # every ramp's scaled window: T0 = Tc (1 + 12/L), T1 = Tc (1 - 8/L)
V1_NULL_EPS = (2.0, 4.0, 6.0)
V1_SEARCH = (6.0, -4.0)  # where the dynamic test looks for each replica's peak
V1_PRE = V1_HOLD = 40  # sweeps held at T0 before the ramp and at T1 after it
V1_NULL_TAU = 400  # measured tau_c per null block: a scored slice of up to 340 tau_c, and s(t)
MODES = ("fss", "fixed_T", "edge")
PEAK = 0.8  # eps_L of the spread's peak (fixed_T: at L = 16)
WIDTH = 0.15  # its half width at half height in eps_L, under the 0.4 eps_L headline window
EDGE_L, EDGE_REPLICAS = 32, 2  # edge: the size, and how many of each seed's replicas peak past
EDGE_PEAK = -4.15  # the band's cold edge, and where
SPREAD = (0.02, 0.12)  # sd of |m|: the background, and at the peak
_STREAMS = {"null": 1, "qs": 2, "fss": 3, "fixed_T": 4, "edge": 5}


def _past_the_band(mode: str, L: int, replica: int) -> bool:
    return mode == "edge" and L == EDGE_L and replica < EDGE_REPLICAS


def v1_peak(mode: str, L: int, replica: int = 0) -> float:
    """eps_L at which the planted spread of |m| peaks for a replica (its index within its seed)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if _past_the_band(mode, L, replica):
        return EDGE_PEAK
    return PEAK * L / V1_SIZES[0] if mode == "fixed_T" else PEAK


def v1_spread(mode: str, L: int, eps_L: np.ndarray | float, replica: int = 0) -> np.ndarray:
    """The planted sd of |m| at eps_L: the background plus a Lorentzian in the variance, centred
    on v1_peak. Its slow tails raise the variance long before the peak, as on the real ramps,
    where the variance warning comes before the alarm band opens (#98)."""
    u = (np.asarray(eps_L, dtype=np.float64) - v1_peak(mode, L, replica)) / WIDTH
    bg, top = SPREAD
    return np.sqrt(bg**2 + (top**2 - bg**2) / (1.0 + u * u))


def _two_point(L: int, sigma: Callable[[np.ndarray], np.ndarray]) -> Values:
    """|m| = 0.5 + sigma(eps_L) s, with s = +-1 drawn every sweep: two-point fluctuations, whose
    squares are sigma^2 exactly, so the spread is planted with the least sampling noise and a few
    replicas pin where it peaks. A dither of one step of |m| (2 / L^2) before the rounding to
    whole spins keeps the realised spread smooth in eps_L instead of a staircase. e per site is
    plain noise."""

    def values(
        steps: np.ndarray, temps: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        sign = rng.choice(np.array([-1.0, 1.0]), size=temps.shape)
        dither = rng.uniform(-1.0, 1.0, size=temps.shape) / L**2
        abs_m = 0.5 + sigma(L * (temps - TC) / TC) * sign + dither
        return abs_m, -1.4 + 0.02 * rng.standard_normal(temps.shape)

    return values


def peak_values(mode: str, L: int) -> Values:
    """A v1 ramp's |m|: two-point fluctuations whose spread is v1_spread, lattice j being replica
    j of its seed."""

    def sigma(eps: np.ndarray) -> np.ndarray:
        return np.stack([v1_spread(mode, L, eps[:, j], j) for j in range(eps.shape[1])], axis=1)

    return _two_point(L, sigma)


def background_values(L: int) -> Values:
    """A v1 null block's |m|: the same fluctuations, stationary at the background spread."""
    return _two_point(L, lambda eps: np.full(eps.shape, SPREAD[0]))


def _fit() -> tuple[float, float]:
    """The planted record's tau_c = A L^z, by the record's own least squares."""
    sizes = sorted(V1_TAU_C)
    A, z = tauc.fit(sizes, np.array([V1_TAU_C[L] for L in sizes]))
    return float(A), float(z)


def v1_ramp(L: int, eps_L: tuple[float, float] = V1_EPS_L) -> protocol.Ramp:
    """The v1 slow ramp at L, as `ramp --x 0.02 --eps-start 12 --eps-end -8` lays it out: held at
    T0 = Tc (1 + eps_L[0] / L), then cooled to T1 = Tc (1 + eps_L[1] / L) at x = 0.02 of the tau_c
    the record measured at L (#56), tau_c L / (x Tc) sweeps per unit T, its params recording the
    record's fit, that the rate came from a measurement, and the window."""
    T0, T1 = (TC * (1 + e / L) for e in eps_L)
    A, z = _fit()
    return protocol.Ramp(
        protocol.rate_for(L, V1_X, V1_TAU_C[L]),
        L,
        T0=T0,
        T1=T1,
        pre=V1_PRE,
        hold=V1_HOLD,
        x=V1_X,
        A=A,
        z=z,
        rate_source="measured",
        eps_L=eps_L,
    )


def v1_nulls(L: int) -> protocol.Fixed:
    """A null run with a block at each null eps_L, measured for V1_NULL_TAU tau_c and saving
    bursts as densely as the ramp does in its analysis band, so that the matched cadence has the
    same frames per sweep in both."""
    sweeps = math.ceil(V1_NULL_TAU * V1_TAU_C[L])
    temps = [TC * (1 + e / L) for e in V1_NULL_EPS]
    gap = burst_gap(v1_ramp(L), L)
    return protocol.Fixed(temps, therm=20, sweeps=sweeps, saves=sweeps // gap, tau_c=V1_TAU_C[L])


def v1_staircase() -> protocol.Staircase:
    """staircase() with 500 measured sweeps per plateau instead of 2,000: the static fits only
    have to run, since no v1 test reads them."""
    return protocol.Staircase(
        [2.0, 2.0], temperatures=(3.0, PLATEAU), warmup=0, min_therm=50, min_meas=500, saves=40
    )


def v1_record(root: Path) -> Path:
    """The planted tau_c record in a pilot run's folder: the #50 record's shape, measured at every
    v1 size and on the running kernel (kernel_sha256), so it is not stale."""
    A, z = _fit()
    record = {
        "kernel_commit": "0" * 40,
        "kernel_sha256": tauc.kernel_sha256(),
        "per_L": {
            str(L): {"tau_c": tau, "tau_c_ci": [0.9 * tau, 1.1 * tau], "tau_c_binning": tau,
                     "tau_e": tau / 2, "n_over_tau": 1000.0, "replicas": 64,
                     "sweeps": round(1000 * tau), "seed": 1}
            for L, tau in V1_TAU_C.items()
        },
        "fit": {"A": A, "z": z, "A_ci": [0.9 * A, 1.1 * A], "z_ci": [z - 0.1, z + 0.1]},
    }  # fmt: skip
    folder = root / "ising" / V1_PILOT_RUN
    folder.mkdir(parents=True, exist_ok=True)
    io.write_json(folder / "pilot.json", record)
    return folder / "pilot.json"


def _stamp(root: Path) -> dict[str, Any]:
    """The stamp of root's v1 record, as a ramp or null run's manifest carries it (#55)."""
    path = root / "ising" / V1_PILOT_RUN / "pilot.json"
    return tauc.stamp(path, io.read_json(path))


def v1_ramp_run(
    root: Path,
    mode: str,
    L: int,
    seed: int,
    eps_L: tuple[float, float] = V1_EPS_L,
    git_sha: str | None = None,
    dirty: bool | None = None,
) -> Path:
    """One v1 ramp run in root, stamped with root's v1 record; eps_L sets its window."""
    window = "" if tuple(eps_L) == V1_EPS_L else "-e{:g}to{:g}".format(*eps_L).replace(".", "p")
    return write_run(
        root,
        f"L{L}-ramp-x0p02{window}-s{seed}",
        v1_ramp(L, eps_L),
        L,
        seed,
        peak_values(mode, L),
        pilot=_stamp(root),
        replicas=V1_REPLICAS,
        rng=np.random.default_rng([seed, L, _STREAMS[mode]]),
        git_sha=git_sha,
        dirty=dirty,
    )


@dataclass(frozen=True)
class V1:
    root: Path  # the data dir the scorer reads
    mode: str
    pilot: Path  # the record's pilot.json
    runs: dict[tuple[str, int, int], Path]  # ("null" | "qs" | "ramp", L, seed) -> run folder


def v1_dataset(
    root: Path,
    mode: str,
    git_sha: str | None = None,
    dirty: bool | None = None,
    shared: V1 | None = None,
) -> V1:
    """One mode's data dir: the record, and per L and seed a ramp, a null run and a staircase.
    Ramps and null runs carry the record's stamp (#55), and every run says it was made from
    git_sha, dirty or not, where the simulator says so. No mode changes the record, the null runs
    or the staircases: with `shared`, they are copied from that dataset and stamped with this
    one's commit instead of written again."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    runs: dict[tuple[str, int, int], Path] = {}
    if shared is None:
        record = v1_record(root)
        made: dict[str, Any] = {"replicas": V1_REPLICAS, "git_sha": git_sha, "dirty": dirty}
        for L in V1_SIZES:
            for seed in V1_SEEDS:
                runs["null", L, seed] = write_run(
                    root, f"L{L}-null-s{seed}", v1_nulls(L), L, seed, background_values(L),
                    pilot=_stamp(root), rng=np.random.default_rng([seed, L, _STREAMS["null"]]),
                    **made,
                )  # fmt: skip
                runs["qs", L, seed] = write_run(
                    root, f"L{L}-qs-s{seed}", v1_staircase(), L, seed, plateau_values(L),
                    rng=np.random.default_rng([seed, L, _STREAMS["qs"]]), **made,
                )  # fmt: skip
    else:
        copied = v1_copy(shared, root, ramps=False)
        record, runs = copied.pilot, dict(copied.runs)
        for run_dir in runs.values():
            restamp(run_dir, git_sha, dirty)
    for L in V1_SIZES:
        for seed in V1_SEEDS:
            runs["ramp", L, seed] = v1_ramp_run(root, mode, L, seed, git_sha=git_sha, dirty=dirty)
    return V1(root, mode, record, dict(sorted(runs.items())))


def _link_tree(source: Path, target: Path) -> None:
    """A folder copied as hard links to its files, or as copies where links are refused."""

    def link(src: str, dst: str) -> None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(source, target, copy_function=link)


def v1_copy(data: V1, root: Path, ramps: bool = True) -> V1:
    """data in another data dir (without its ramps, if not `ramps`), its files hard-linked where
    the file system allows, for a test to change, add or remove runs in. Nothing here writes into
    a file: io.write_file replaces it, so the two dirs never share a change."""
    _link_tree(data.pilot.parent, root / data.pilot.parent.relative_to(data.root))
    runs = {}
    for key, run_dir in data.runs.items():
        if ramps or key[0] != "ramp":
            runs[key] = root / run_dir.relative_to(data.root)
            _link_tree(run_dir, runs[key])
    return V1(root, data.mode, root / data.pilot.relative_to(data.root), runs)


def restamp(run_dir: Path, git_sha: str | None, dirty: bool | None) -> None:
    """Change the commit a planted run says it was made from (producer.git_sha and
    producer.dirty in its manifest). complete.json hashes the tables only, so the run stays
    complete and its digest unchanged."""
    manifest = io.read_json(run_dir / "manifest.json")
    manifest["producer"] |= {"git_sha": git_sha, "dirty": dirty}
    io.write_json(run_dir / "manifest.json", manifest, replace=True)


def restamp_pilot(run_dir: Path, stamp: dict[str, Any] | None) -> None:
    """Change the pilot record a planted ramp or null run says it was made against
    (params.pilot, as pilot.stamp writes it; None, as the simulator writes for a run made
    against no record). As with restamp, the run stays complete."""
    manifest = io.read_json(run_dir / "manifest.json")
    manifest["params"]["pilot"] = stamp
    io.write_json(run_dir / "manifest.json", manifest, replace=True)
