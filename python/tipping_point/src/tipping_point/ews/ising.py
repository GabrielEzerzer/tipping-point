"""Ising adapter: simulator runs in, (batch, time) arrays of |m|, e per site and, where a checkpoint
has been read over the run, its latent columns out, and the Ising experiment of a pre-registration
scored from them.

Runs follow tipping_point.ising.io's layout. This module reads the manifest (L, Tc, seed,
replicas, blocks, schedule), `stats` (step, replica, temperature, segment, phase, magnetisation,
energy) and `configs` (the same per saved frame, with burst and frame), never the spins. A run
without complete.json is refused; a staircase plateau that failed complete.json's thermalised
check is dropped and counted. Schedules of kind ramp, fixed and staircase are read; each block of
a fixed run holds one temperature, and a null block is matched on its eps_L.

tau_c(L), the time unit, comes from the #50 pilot record <data dir>/ising/<pilot run>/pilot.json:
  {"kernel_commit": str, "kernel_sha256": str (absent before #55),
   "per_L": {"16": {"tau_c", "tau_c_binning", "tau_e", "n_over_tau", "replicas", "sweeps",
                    "seed", optional "tau_c_ci": [lo, hi]}, ...},
   "fit": {"A", "z", "A_ci", "z_ci"}}
Only the measured per_L values are read (other keys are ignored); the fit, and any tau_c it would
extrapolate to an unmeasured L, never enter scoring. Nothing here holds a tau_c of its own:
without the record, or for an L it did not measure, scoring stops. A stale record, one that did
not measure the running kernel (tipping_point.ising.pilot.stale, the rule ramp and null runs are
held to), is refused unless the caller allows it.

Cadences: `dense` is every stats row, thinned to one sample per `thin` tau_c; `matched` is every
saved frame (bursts of consecutive sweeps), exactly what the JEPA sees. Windows, bandwidths and
segments are converted from tau_c and eps_L into samples per cadence and L.

Latents (#28): <run>/latents/<training>-step<step:07d>/ is what `jepa extract` writes, and it is
read here with a plain Parquet reader, since jepa.extract imports this module. It holds the saved
frames its manifest's selection keeps, so its columns exist at the matched cadence only: series()
joins them on the six key columns, whose equality it asserts, and refuses a table without
complete.json. A prereg with a [latents] section is scored once per checkpoint it names, a trained
checkpoint and its untrained twin alike, and every row of every table says which in its `checkpoint`
column: the table's folder name, or "" for a row of the classic observables, which read no table.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..ising import autocorr, io
from ..ising.pilot import stale as _stale
from . import fss, indicators, scoring, surrogates
from .prereg import SELECTIONS
from .trend import RAW, Pipeline, trailing_score

PILOT = "pilot.json"
OBSERVABLES = ("abs_m", "energy")
_STATS = ("step", "replica", "temperature", "segment", "phase", "magnetisation", "energy")
KEY = ("step", "replica", "temperature", "segment", "burst", "frame")
_CONFIGS = (*KEY, "magnetisation", "energy")
LATENTS = "latents"
PRIMARY = "primary"  # latents.checkpoints.role of the checkpoints scored on the whole grid
REPORTED = "reported"  # the verdict of a row judged on the twins alone, on a trained pass
_CELL_TABLES = ("auc", "surrogates", "alarms", "peaks")  # what _cells fills
Row = dict[str, Any]
Peaks = dict[tuple[str, str], dict[int, list[np.ndarray]]]  # per-L, per-seed peak eps_L


class IsingDataError(RuntimeError):
    """Input the scorer cannot use: missing, incomplete, ambiguous or inconsistent."""


# ---------------------------------------------------------------- reading


@dataclass(frozen=True)
class Pilot:
    path: Path
    sha256: str
    kernel_commit: str
    kernel_sha256: str | None  # None in a record written before #55
    tau_c: dict[int, float]
    tau_c_ci: dict[int, list[float]]  # the record's 95% interval, where it gives one

    @property
    def stale(self) -> str | None:
        """Why the record did not measure the running kernel, or None when it did."""
        return _stale({"kernel_sha256": self.kernel_sha256})

    def tau(self, L: int) -> float:
        if L not in self.tau_c:
            raise IsingDataError(f"{self.path} has no tau_c for L = {L}; it must cover every L")
        return self.tau_c[L]


def read_pilot(
    data_dir: Path, run_id: str, sha256: str | None = None, allow_stale: bool = False
) -> Pilot:
    """The tau_c record of a pilot run; with `sha256`, the file must be exactly that one. A stale
    record is refused unless `allow_stale`."""
    path = Path(data_dir) / "ising" / run_id / PILOT
    if not path.is_file():
        raise IsingDataError(
            f"no pilot record at {path}: tau_c(L) comes from the #50 pilot's {PILOT}; "
            "run the pilot and name its run in data.tau_c.pilot_run"
        )
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if sha256 is not None and digest != sha256:
        raise IsingDataError(f"{path}: sha256 {digest}, but the prereg pins {sha256}")
    try:
        record = json.loads(raw)
        tau_c = {int(L): float(point["tau_c"]) for L, point in record["per_L"].items()}
        ci = {
            int(L): [float(v) for v in point["tau_c_ci"]]
            for L, point in record["per_L"].items()
            if point.get("tau_c_ci") is not None
        }
        commit, kernel = record["kernel_commit"], record.get("kernel_sha256")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise IsingDataError(f"{path}: not a pilot record ({exc!r})") from exc
    if not isinstance(commit, str) or not commit:
        raise IsingDataError(f"{path}: kernel_commit must name the kernel's commit")
    if kernel is not None and not isinstance(kernel, str):
        raise IsingDataError(f"{path}: kernel_sha256 must be the kernel's sha256")
    bad = sorted(L for L, tau in tau_c.items() if not 0 < tau < math.inf)
    if bad or not tau_c:
        raise IsingDataError(f"{path}: tau_c must be positive and finite for every L (bad: {bad})")
    pilot = Pilot(path, digest, commit, kernel, tau_c, ci)
    if pilot.stale and not allow_stale:
        raise IsingDataError(
            f"{path} is stale: {pilot.stale}. Its tau_c may belong to another kernel: re-run the "
            "pilot and tauc, or pass --allow-stale-pilot to score with it anyway"
        )
    return pilot


@dataclass(frozen=True)
class Run:
    path: Path
    manifest: dict[str, Any]
    checks: dict[str, Any]  # complete.json's end-of-run checks
    complete_sha256: str  # "" for a run that has not completed

    @property
    def run_id(self) -> str:
        return self.manifest["run_id"]

    @property
    def params(self) -> dict[str, Any]:
        return self.manifest["params"]

    @property
    def L(self) -> int:
        return int(self.params["L"])

    @property
    def Tc(self) -> float:
        return float(self.params["Tc"])

    @property
    def seed(self) -> int:
        return int(self.params["seed"])

    @property
    def schedule(self) -> dict[str, Any]:
        return self.params["schedule"]

    @property
    def kind(self) -> str:
        return self.schedule["kind"]

    def eps(self, temperature: np.ndarray | float) -> np.ndarray:
        return fss.scaled_temperature(temperature, self.L, self.Tc)


def open_run(run_dir: Path) -> Run:
    run_dir = Path(run_dir)
    done = run_dir / "complete.json"
    if not done.is_file():
        raise IsingDataError(f"{run_dir.name}: no complete.json (running or crashed); refused")
    raw = done.read_bytes()
    checks = json.loads(raw).get("checks", {})
    manifest = io.read_json(run_dir / "manifest.json")
    return Run(run_dir, manifest, checks, hashlib.sha256(raw).hexdigest())


def find_runs(data_dir: Path) -> tuple[list[Run], list[Run]]:
    """Runs under <data dir>/ising: the complete ones, and the ones still running or crashed."""
    complete, incomplete = [], []
    for manifest in sorted((Path(data_dir) / "ising").glob("*/manifest.json")):
        if (manifest.parent / "complete.json").is_file():
            complete.append(open_run(manifest.parent))
        else:
            incomplete.append(Run(manifest.parent, io.read_json(manifest), {}, ""))
    return complete, incomplete


@dataclass(frozen=True)
class Series:
    """One temperature block of a run at one cadence: samples along time, replicas along batch."""

    steps: np.ndarray  # (time,) sweep index
    temperature: np.ndarray  # (time,) bath temperature
    segment: np.ndarray  # (time,) plateau (staircase) or phase (ramp) index
    phase: np.ndarray  # (time,) 1 while measuring; saved frames are all measured
    abs_m: np.ndarray  # (batch, time) |m| per site
    energy: np.ndarray  # (batch, time) energy per site
    replicas: np.ndarray  # (batch,)
    latents: dict[str, np.ndarray] = field(default_factory=dict)  # column -> (batch, time)

    def observable(self, name: str) -> np.ndarray:
        if name in self.latents:
            return self.latents[name]
        if name not in OBSERVABLES:
            raise ValueError(f"observable must be one of {OBSERVABLES}, got {name!r}")
        return self.abs_m if name == "abs_m" else self.energy

    def take(self, index: np.ndarray) -> Series:
        return replace(
            self,
            steps=self.steps[index],
            temperature=self.temperature[index],
            segment=self.segment[index],
            phase=self.phase[index],
            abs_m=self.abs_m[:, index],
            energy=self.energy[:, index],
            latents={name: values[:, index] for name, values in self.latents.items()},
        )


def latents_name(training: str, step: int) -> str:
    """The folder a checkpoint's table has under <run>/latents/."""
    return f"{training}-step{step:07d}"


def at_plateau(temperature: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    return np.isclose(temperature, rule["tc_plateau"], rtol=1e-6, atol=1e-6)


def kept(kind: str, rows: pa.Table, rule: dict[str, Any]) -> np.ndarray:
    """Which configs rows (temperature and burst columns) of a run of `kind` a latents table's
    selection rule keeps, as jepa.extract.selection records it in the table's manifest."""
    temperature = rows["temperature"].to_numpy()
    if rule["rule"] == "all" or kind != "staircase":
        return np.ones(len(temperature), dtype=bool)
    plateau = at_plateau(temperature, rule)
    if rule["rule"] == "tc-only":
        return plateau
    return plateau | (rows["burst"].to_numpy() % rule["every"] == 0)


def _join_latents(run: Run, frames: pa.Table, name: str) -> tuple[pa.Table, dict[str, np.ndarray]]:
    """A run's frames (a configs table) that its latents table `name` holds, and the table's other
    columns, row for row. Its key columns must equal those of the frames its selection keeps, in
    their order and with configs' types; anything else, or a table not complete, is refused."""
    folder, where = run.path / LATENTS / name, f"{run.run_id}: latents/{name}"
    if not ((folder / "complete.json").is_file() and (folder / "manifest.json").is_file()):
        raise IsingDataError(f"{where} has no complete.json (running, crashed or absent); refused")
    manifest = io.read_json(folder / "manifest.json")
    rule = manifest.get("params", {}).get("selection") or {"rule": "all"}
    try:
        own = frames.filter(kept(run.kind, frames, rule))
        table = pa.concat_tables([pq.read_table(p) for p in io.chunks(folder.parent, name)])
        same = bool(table.num_rows) and table.select(KEY).equals(own.select(KEY))
    except (pa.ArrowInvalid, KeyError) as exc:
        raise IsingDataError(f"{where}: not a latents table ({exc!r}); refused") from exc
    if not same:
        raise IsingDataError(
            f"{where}: its {table.num_rows} rows are not keyed exactly as the {own.num_rows} "
            f"frames its {rule['rule']} selection keeps; refused"
        )
    columns = [c for c in table.column_names if c not in KEY]
    return own, {c: table[c].to_numpy().astype(np.float64) for c in columns}


def series(run: Run, cadence: str, latents: str | None = None) -> list[Series]:
    """Every recorded row of a cadence (dense: stats, matched: configs), one Series per
    temperature block, in step order. With `latents`, a folder under <run>/latents/, the matched
    cadence's frames that table holds, its columns joined on as observables."""
    table, names = {"dense": ("stats", _STATS), "matched": ("configs", _CONFIGS)}[cadence]
    if latents is not None and cadence != "matched":
        raise IsingDataError(
            f"{run.run_id}: latents/{latents} holds saved frames, read at the matched cadence "
            f"alone, not at {cadence}"
        )
    paths = io.chunks(run.path, table)
    if not paths:
        raise IsingDataError(f"{run.run_id}: no {table} rows for the {cadence} cadence")
    data = pa.concat_tables([pq.read_table(p, columns=list(names)) for p in paths])
    joined: dict[str, np.ndarray] = {}
    if latents is not None:
        data, joined = _join_latents(run, data, latents)
    replicas, blocks = int(run.params["replicas"]), int(run.params["blocks"])
    lattices = replicas * blocks
    rows = data.num_rows // lattices
    if rows * lattices != data.num_rows:
        raise IsingDataError(f"{run.run_id}: {table} does not hold whole steps of {lattices} rows")
    grid = {name: data[name].to_numpy().reshape(rows, lattices) for name in names}
    columns = {name: values.reshape(rows, lattices) for name, values in joined.items()}
    area = float(run.L**2)
    out = []
    for b in range(blocks):
        own, first = slice(b * replicas, (b + 1) * replicas), b * replicas
        phase = grid["phase"][:, first] if "phase" in grid else np.ones(rows, dtype=np.int8)
        out.append(
            Series(
                steps=grid["step"][:, first],
                temperature=grid["temperature"][:, first],
                segment=grid["segment"][:, first],
                phase=phase,
                abs_m=np.abs(grid["magnetisation"][:, own]).T / area,
                energy=grid["energy"][:, own].T / area,
                replicas=grid["replica"][0, own],
                latents={name: values[:, own].T for name, values in columns.items()},
            )
        )
    return out


def measured(s: Series) -> Series:
    return s.take(s.phase == 1)


def thin(s: Series, every: float) -> Series:
    """One sample per `every` sweeps: for each point of a regular grid from the first recorded
    step, the row nearest to it, never the same row twice. With `every` a multiple of the stats
    stride the grid lands on rows; otherwise each sample sits within half a stride of its point,
    so runs with different strides still share one cadence."""
    if len(s.steps) < 2:
        return s
    grid = np.arange(float(s.steps[0]), float(s.steps[-1]) + 0.5, every)
    right = np.clip(np.searchsorted(s.steps, grid), 1, len(s.steps) - 1)
    nearer = np.where(grid - s.steps[right - 1] <= s.steps[right] - grid, right - 1, right)
    return s.take(np.unique(nearer))


def plateaus(run: Run, cadence: str, latents: str | None = None) -> tuple[dict[float, Series], int]:
    """A staircase's measured rows per plateau temperature, without the plateaus that failed the
    thermalised check; returns them and how many were dropped. With `latents`, as in series(): a
    plateau the table's selection keeps nothing of is not there."""
    if run.kind != "staircase":
        raise IsingDataError(f"{run.run_id} is a {run.kind} run, not a staircase")
    checks = run.checks.get("thermalised", [])
    failed = {round(c["temperature"], 9) for c in checks if not c["passed"]}
    (s,) = series(run, cadence, latents)
    s = measured(s)
    found, dropped = {}, 0
    for segment in np.unique(s.segment):
        part = s.take(s.segment == segment)
        T = float(part.temperature[0])
        if round(T, 9) in failed:
            dropped += 1
        else:
            found[T] = part
    return found, dropped


def ramp_segment(s: Series, run: Run, top: float, bottom: float) -> tuple[int, int]:
    """Indices of the first samples at which the cooling bath reaches eps_L = top and bottom."""
    eps = run.eps(s.temperature)
    if len(eps) == 0 or eps[0] <= top:
        start = f"{eps[0]:.2f}" if len(eps) else "nothing"
        raise IsingDataError(f"{run.run_id}: starts at eps_L = {start}, not above {top}")
    below = np.flatnonzero(eps <= bottom)
    if len(below) == 0:
        raise IsingDataError(f"{run.run_id}: the bath never reaches eps_L = {bottom}")
    return int(np.flatnonzero(eps <= top)[0]), int(below[0])


def null_slices(values: np.ndarray, span: int) -> np.ndarray:
    """Non-overlapping slices of `span` samples from each replica's row, in time order."""
    count = values.shape[1] // span
    if count == 0:
        raise IsingDataError(f"a null series of {values.shape[1]} samples has no {span}-slice")
    return values[:, : count * span].reshape(len(values) * count, span)


# ---------------------------------------------------------------- which runs a prereg scores


@dataclass(frozen=True)
class Inputs:
    """The runs one L contributes, one per slot of the prereg's data section."""

    ramps: list[Run]  # one per seed
    nulls: list[tuple[Run, int]]  # (run, block) per (seed, eps_L) that passed its check
    staircases: list[Run]  # one per seed
    dropped: list[str]  # null blocks that failed complete.json's thermalised check
    latents: str | None = None  # the latents table (a folder under each run) to read them with

    def runs(self) -> Iterator[Run]:
        yield from self.ramps
        yield from (run for run, _ in self.nulls)
        yield from self.staircases


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def _thermalised(run: Run, temperature: float) -> bool | None:
    """complete.json's thermalised verdict at a temperature: passed, failed, or None when the
    run recorded none there (runs from before #50 checked only staircases)."""
    checks = [c for c in run.checks.get("thermalised", []) if _close(c["temperature"], temperature)]
    return all(c["passed"] for c in checks) if checks else None


def _slots(run: Run, data: dict[str, Any]) -> list[tuple[Any, ...]]:
    """The prereg slots a run fills: ("ramp", seed), ("staircase", seed) or
    ("null", seed, eps_L) with its block. A ramp at another x is another experiment (the fast
    ramp) and fills none; one at the prereg's x whose eps_L window is not data.ramps.eps_L, when
    the prereg gives one, is refused."""
    if run.L not in data["L"] or run.seed not in data["seeds"]:
        return []
    params = run.schedule.get("params", {})
    if run.kind == data["ramps"]["kind"]:
        x = params.get("x")
        if x is None or not _close(x, data["ramps"]["x"]):
            return []
        window = data["ramps"].get("eps_L")
        if window is not None:
            ran = [float(run.eps(params[end])) for end in ("T0", "T1")]
            if not all(map(_close, ran, map(float, window))):
                raise IsingDataError(
                    f"{run.run_id}: its ramp runs from eps_L = {ran[0]:g} to {ran[1]:g}, but the "
                    f"prereg's data.ramps.eps_L is {window}; refused (exclude it, or regenerate "
                    "it with ramp --eps-start and --eps-end)"
                )
        return [("ramp", run.seed)]
    if run.kind == data["staircase"]["kind"]:
        return [("staircase", run.seed)]
    if run.kind == data["nulls"]["kind"]:
        return [
            ("null", run.seed, float(e), block)
            for block, T in enumerate(params.get("temperatures", []))
            for e in data["nulls"]["eps_L"]
            if _close(float(run.eps(T)), float(e))
        ]
    return []


def select(
    runs: Sequence[Run],
    incomplete: Sequence[Run],
    data: dict[str, Any],
    latents: Sequence[str] = (),
) -> dict[int, Inputs]:
    """For each L, exactly one complete run per (kind, seed) and one null block per
    (seed, eps_L). A slot left empty or filled twice, or a matching run that has not completed,
    is an error; listing a run in data.exclusions takes it out. A null block is read as
    equilibrium, so it needs complete.json's thermalised verdict: without one it is refused,
    and one that failed fills its slot but is dropped from scoring and counted. Every scored run
    must hold a complete table of each name in `latents`."""
    excluded = {e["run_id"] for e in data["exclusions"]}
    pending = [r.run_id for r in incomplete if r.run_id not in excluded and _slots(r, data)]
    if pending:
        raise IsingDataError(f"runs the prereg would score have no complete.json: {pending}")
    filled: dict[tuple[Any, ...], list[tuple[Run, int]]] = {}
    for run in runs:
        if run.run_id in excluded:
            continue
        for slot in _slots(run, data):
            key = (run.L, *slot[:3]) if slot[0] == "null" else (run.L, *slot)
            filled.setdefault(key, []).append((run, slot[3] if slot[0] == "null" else 0))
    chosen = {}
    for L in sorted(data["L"]):
        wanted = [(L, "ramp", s) for s in data["seeds"]] + [
            (L, "staircase", s) for s in data["seeds"]
        ]
        wanted += [(L, "null", s, float(e)) for s in data["seeds"] for e in data["nulls"]["eps_L"]]
        missing = [k for k in wanted if k not in filled]
        doubled = {
            k: [r.run_id for r, _ in v] for k, v in filled.items() if k[0] == L and len(v) > 1
        }
        if missing:
            other_x = [
                f"{r.run_id} (x = {r.schedule.get('params', {}).get('x')})"
                for r in runs
                if r.run_id not in excluded
                and r.kind == data["ramps"]["kind"]
                and (r.L, "ramp", r.seed) in missing
            ]
            elsewhere = f"; ramps there at another x than {data['ramps']['x']}: {other_x}"
            raise IsingDataError(f"no complete run for {missing}" + (elsewhere if other_x else ""))
        if doubled:
            raise IsingDataError(f"more than one run for {doubled}; exclude the extras")
        short = [
            r.run_id
            for k in wanted
            for r, _ in filled[k]
            if r.params["replicas"] < data["replicas"]
        ]
        if short:
            raise IsingDataError(f"fewer than {data['replicas']} replicas per block: {short}")
        blocks = [filled[k][0] for k in wanted if k[1] == "null"]
        verdicts = [_thermalised(r, _block_temperature(r, b)) for r, b in blocks]
        unchecked = [
            f"{r.run_id} block {b}" for (r, b), v in zip(blocks, verdicts, strict=True) if v is None
        ]
        if unchecked:
            raise IsingDataError(
                f"null blocks with no thermalised verdict in complete.json: {unchecked}; "
                "rerun them as null runs, or exclude them"
            )
        passed = [block for block, v in zip(blocks, verdicts, strict=True) if v]
        if not passed:
            raise IsingDataError(f"L = {L}: every null block failed its thermalised check")
        chosen[L] = Inputs(
            ramps=[filled[(L, "ramp", s)][0][0] for s in data["seeds"]],
            nulls=passed,
            staircases=[filled[(L, "staircase", s)][0][0] for s in data["seeds"]],
            dropped=[f"{r.run_id} block {b}" for r, b in blocks if (r, b) not in passed],
        )
    absent = {
        f"{r.run_id}/{LATENTS}/{name}"
        for inputs in chosen.values()
        for r in inputs.runs()
        for name in latents
        if not (r.path / LATENTS / name / "complete.json").is_file()
    }
    if absent:
        raise IsingDataError(f"scored runs without a complete latents table: {sorted(absent)}")
    return chosen


def _block_temperature(run: Run, block: int) -> float:
    return float(run.schedule["params"]["temperatures"][block])


# ---------------------------------------------------------------- cadences in samples


def _stride(run: Run) -> int:
    schedule = run.schedule
    return int(schedule.get("stats_stride") or schedule["params"].get("stats_stride") or 1)


@dataclass(frozen=True)
class Prepared:
    """One L's ramps and null blocks at one cadence, with the sweeps each sample stands for."""

    cadence: str
    sweeps: float  # sweeps per sample
    ramps: list[tuple[Run, Series]]
    nulls: list[tuple[Run, Series]]  # (run, its measured block)

    def samples(self, taus: float, tau_c: float) -> int:
        return max(2, round(taus * tau_c / self.sweeps))


def _density(s: Series) -> float:
    """Saved frames per sweep over a stretch of a matched-cadence series."""
    span = float(s.steps[-1] - s.steps[0] + 1) if len(s.steps) > 1 else 0.0
    return len(s.steps) / span if span > 0 else 0.0


def prepare(
    inputs: Inputs, cadence: str, tau_c: float, thin_taus: float, band: tuple[float, float]
) -> Prepared:
    """Dense: thin every run to one sample per thin_taus tau_c (never finer than the coarsest
    stats stride), nearest rows to one shared grid spacing. Matched: keep every frame; ramps and
    nulls must save at the same rate (frames per sweep) inside the analysis band, or a window
    would span different times."""
    ramps = [(run, series(run, cadence, inputs.latents)[0]) for run in inputs.ramps]
    nulls = [
        (run, measured(series(run, cadence, inputs.latents)[block])) for run, block in inputs.nulls
    ]
    if cadence == "dense":
        coarsest = max(_stride(run) for run, _ in [*ramps, *nulls])
        every = max(float(coarsest), thin_taus * tau_c)
        return Prepared(
            cadence,
            every,
            [(r, thin(s, every)) for r, s in ramps],
            [(r, thin(s, every)) for r, s in nulls],
        )
    run, s = ramps[0]
    i0, i1 = ramp_segment(s, run, *band)
    density = _density(s.take(np.arange(i0, i1 + 1)))
    others = [_density(n) for _, n in nulls]
    if density <= 0 or any(abs(d / density - 1) > 0.1 for d in others):
        raise IsingDataError(
            f"matched cadence at L = {run.L}: the ramps save {density:.4g} frames per sweep in the "
            f"analysis band, the null runs {[round(d, 4) for d in others]}; they must match"
        )
    return Prepared(cadence, 1.0 / density, ramps, nulls)


def positives(
    prep: Prepared, observable: str, h: float, segment: float, window: int
) -> tuple[list[np.ndarray], int]:
    """Per ramp, one row per replica: the segment from eps_L = h + segment down to h, plus the
    window - 1 samples before it. Returns the rows and the segment's length in samples."""
    groups, lengths = [], set()
    for run, s in prep.ramps:
        i0, i1 = ramp_segment(s, run, h + segment, h)
        if i0 - window + 1 < 0:
            raise IsingDataError(f"{run.run_id}: no room for a window before eps_L = {h + segment}")
        lengths.add(i1 - i0 + 1)
        groups.append(s.observable(observable)[:, i0 - window + 1 : i1 + 1])
    if len(lengths) != 1:
        raise IsingDataError(
            f"ramps at L = {prep.ramps[0][0].L} disagree on the segment: {lengths}"
        )
    return groups, lengths.pop()


# ---------------------------------------------------------------- the experiment


@dataclass
class Evaluation:
    tables: dict[str, list[Row]]  # auc, surrogates, alarms, peaks, fss
    inputs: list[Run]
    pilot: Pilot
    dropped: dict[str, list[str] | int]  # failed null blocks, and how many plateaus were dropped
    latents: dict[str, dict[str, str]] = field(default_factory=dict)  # run id -> table -> sha256
    # of the table's complete.json; each run's empty when the prereg reads no table


@dataclass(frozen=True)
class Checkpoint:
    """What one pass of the scorer reads: a latents table, or, named "", the classic observables."""

    name: str  # the table's folder under each run, and the rows' `checkpoint` column
    sha256: str  # of the checkpoint file the prereg pins
    role: str  # latents.checkpoints.role, a twin's that of its trained checkpoint
    trained: bool = False  # a trained checkpoint's pass, not its twin's nor the classic one


def checkpoints(doc: dict[str, Any]) -> list[Checkpoint]:
    """The passes a prereg scores: each [[latents.checkpoints]] entry's trained checkpoint and its
    untrained twin, or the one classic pass of a prereg with no [latents]."""
    if "latents" not in doc:
        return [Checkpoint("", "", PRIMARY)]
    steps = int(doc["latents"]["steps"])
    return [
        Checkpoint(latents_name(entry["training"], step), entry[pin], entry["role"], trained)
        for entry in doc["latents"]["checkpoints"]
        for step, pin, trained in ((steps, "trained_sha256", True), (0, "twin_sha256", False))
    ]


def _scored_as(doc: dict[str, Any], role: str) -> dict[str, Any]:
    """A prereg as a checkpoint of `role` is scored under it: an ablation, descriptive, at the
    headline cell alone and with its own, smaller count of surrogates (#28, call 7)."""
    if role == PRIMARY:
        return doc
    sens, sc, tests = doc["sensitivity"], doc["scoring"], doc["scoring"]["surrogates"]
    head = sens["headline"]
    return doc | {
        "sensitivity": sens | {"window": [head["window"]], "bandwidth": [head["bandwidth"]]},
        "scoring": sc | {"surrogates": tests | {"n": tests["n_ablation"]}},
    }


class Admitted(NamedTuple):
    """What a pre-registration's scoring reads, once admit has let it through."""

    pilot: Pilot
    chosen: dict[int, Inputs]
    scored: list[Run]
    passes: list[Checkpoint]


def admit(
    doc: dict[str, Any],
    data_dir: Path,
    say: Callable[[str], None] = print,
    allow_stale: bool = False,
    commit: str | None = None,
    runs_commit: str | None = None,
) -> Admitted:
    """Everything a scoring refuses before it reads a frame; evaluate's arguments say what. A
    table extracted under another rule than latents.extraction gives its role is refused, frozen
    or not (_same_selection)."""
    data = doc["data"]
    pinned = data["tau_c"]["pilot_sha256"]
    pinned = pinned if len(pinned) == 64 else None
    pilot = read_pilot(data_dir, data["tau_c"]["pilot_run"], pinned, allow_stale)
    if pilot.stale:
        say(f"warning: {pilot.path}: {pilot.stale}; scored anyway (--allow-stale-pilot)")
    passes = checkpoints(doc)
    read = {c.name: c.sha256 for c in passes if c.name}
    chosen = select(*find_runs(data_dir), data, list(read))
    scored = list({run.run_id: run for i in chosen.values() for run in i.runs()}.values())
    _same_selection(scored, passes)
    if commit is not None:
        _same_commit(scored, runs_commit or commit)
        if pinned is not None:  # always, once frozen: the freeze refuses an unpinned record
            _same_record(chosen, pinned, data["staircase"].get("tau_sha256", {}))
        _same_latents(scored, read, commit)
    return Admitted(pilot, chosen, scored, passes)


def evaluate(
    doc: dict[str, Any],
    data_dir: Path,
    rng: np.random.Generator,
    say: Callable[[str], None] = print,
    allow_stale: bool = False,
    commit: str | None = None,
    runs_commit: str | None = None,
) -> Evaluation:
    """Score a pre-registration's Ising experiment: the AUC of every cell, the surrogate tests,
    alarms and peak positions at the headline cell, and static and dynamic finite-size scaling.
    It computes only: writing results and logging trials are the caller's. A stale pilot record
    is refused unless allow_stale. With `commit`, the commit a frozen prereg's tag points at,
    every scored run must have been made by that commit from a clean tree, every scored ramp and
    null run against the pilot record the prereg pins, and every scored staircase from the
    tau.json it pins at that L. Without it (an unfrozen prereg, exploratory use) none is asked:
    the exploratory L = 16 null runs name no record. With `runs_commit` too, the commit of a
    baseline's tag, the runs must come from that commit instead (a latent prereg reads the runs of
    the version it compares with), and each latents table from `commit` and the checkpoint the
    prereg pins (_same_latents). A prereg with a [latents] section is scored once per checkpoint."""
    pilot, chosen, scored, passes = admit(doc, data_dir, say, allow_stale, commit, runs_commit)
    read = {c.name: c.sha256 for c in passes if c.name}
    tables: dict[str, list[Row]] = {name: [] for name in (*_CELL_TABLES, "fss")}
    plateaus_dropped = 0
    for checkpoint in passes:
        own = _scored_as(doc, checkpoint.role)
        slots = {L: replace(i, latents=checkpoint.name or None) for L, i in chosen.items()}
        part: dict[str, list[Row]] = {name: [] for name in _CELL_TABLES}
        peaks: Peaks = {}
        for L, inputs in slots.items():
            for cadence in own["indicators"]["cadences"]:
                prep = _cells(own, L, inputs, cadence, pilot.tau(L), part, peaks, rng)
                say(
                    f"L = {L}, {cadence}: {prep.sweeps:g} sweeps per sample, "
                    f"tau_c = {pilot.tau(L):g}"
                )
        static, lost = _static(own, slots, rng, checkpoint.trained)
        part["fss"] = static + _dynamic(own, peaks, rng, checkpoint.trained)
        plateaus_dropped = max(plateaus_dropped, lost)  # the same plateaus, table after table
        for table, rows in part.items():
            tables[table] += [row | {"checkpoint": checkpoint.name} for row in rows]
    null_blocks = [block for i in chosen.values() for block in i.dropped]
    dropped = {"null_blocks": null_blocks, "plateaus": plateaus_dropped}
    return Evaluation(tables, scored, pilot, dropped, {r.run_id: _digests(r, read) for r in scored})


def _digests(run: Run, names: Iterable[str]) -> dict[str, str]:
    """The sha256 of each named latents table's complete.json: what a result says it read."""
    done = (run.path / LATENTS / name / "complete.json" for name in names)
    return {p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in done}


def _same_selection(runs: Sequence[Run], passes: Sequence[Checkpoint]) -> None:
    """Refuse a latents table extracted under another rule than latents.extraction gives its role
    (prereg.SELECTIONS): a staircase's table must carry that rule exactly, and a ramp's or a null
    run's, which every rule reads whole, that rule's name or `all`."""
    foreign = []
    for run in runs:
        for checkpoint in passes:
            if not checkpoint.name:
                continue
            manifest = io.read_json(run.path / LATENTS / checkpoint.name / "manifest.json")
            rule = manifest["params"].get("selection") or {"rule": "all"}
            wanted = SELECTIONS[checkpoint.role]
            same = rule["rule"] in ("all", wanted["rule"])
            if not (rule == wanted if run.kind == "staircase" else same):
                foreign.append(
                    f"{run.run_id}/{LATENTS}/{checkpoint.name} (read under {rule['rule']}, "
                    f"its {checkpoint.role} role takes {wanted['rule']})"
                )
    if foreign:
        raise IsingDataError(f"latents tables not extracted as latents.extraction says: {foreign}")


def _same_latents(runs: Sequence[Run], pinned: dict[str, str], commit: str) -> None:
    """Refuse a scored latents table that is not the one a frozen prereg pins: from another
    checkpoint than the sha256 it pins, read over another state of its run, or extracted by anything
    but the tag's commit from a clean tree (#28, call 11: the freeze comes before any extraction).
    Each table's manifest says all three."""
    foreign = []
    for run in runs:
        for name, sha in pinned.items():
            manifest = io.read_json(run.path / LATENTS / name / "manifest.json")
            params, producer = manifest["params"], manifest["producer"]
            if (
                params["checkpoint"]["sha256"] != sha
                or params["dataset"]["complete_sha256"] != run.complete_sha256
                or producer.get("git_sha") != commit
                or producer.get("dirty") is not False
            ):
                foreign.append(
                    f"{run.run_id}/{LATENTS}/{name} (checkpoint sha256 "
                    f"{params['checkpoint']['sha256']}, git_sha {producer.get('git_sha')}, "
                    f"dirty {producer.get('dirty')})"
                )
    if foreign:
        raise IsingDataError(
            f"latents tables not made by the freeze tag's commit {commit}, from a clean tree, from "
            f"the checkpoint the prereg pins and over the run as it stands: {foreign}"
        )


def _same_record(chosen: dict[int, Inputs], pinned: str, tau_pins: dict[str, str]) -> None:
    """Refuse a scored ramp or null run made against another pilot record than the one the
    prereg pins, or naming none (#26 K): its rate or length was set by another tau_c clock. And
    refuse a scored staircase whose dwell was not read from the tau.json the prereg pins at its L
    (data.staircase.tau_sha256), whether another file's or a number's, which names none: its
    plateaus were held for another time."""
    runs = {r.run_id: r for i in chosen.values() for r in [*i.ramps, *(run for run, _ in i.nulls)]}
    foreign = [
        f"{r.run_id} (pilot sha256 {sha})"
        for r in runs.values()
        if (sha := (r.params.get("pilot") or {}).get("sha256")) != pinned
    ]
    if foreign:
        raise IsingDataError(
            f"ramp and null runs not made against the pilot record the prereg pins, sha256 "
            f"{pinned}: {foreign}"
        )
    dwelled = [
        f"{r.run_id} (tau.json sha256 {sha})"
        for L, inputs in chosen.items()
        for r in inputs.staircases
        if (sha := r.schedule.get("params", {}).get("tau_sha256")) is None
        or sha != tau_pins.get(str(L))
    ]
    if dwelled:
        raise IsingDataError(
            f"staircases whose dwell was not read from the tau.json the prereg pins at their L "
            f"(data.staircase.tau_sha256 {tau_pins}): {dwelled}"
        )


def _same_commit(runs: Sequence[Run], commit: str) -> None:
    """Refuse a scored run that the frozen commit did not make from a clean tree (#26 decision
    6: a version scores only runs generated after its freeze, from the tagged commit)."""
    foreign = [
        f"{r.run_id} (git_sha {producer.get('git_sha')}, dirty {producer.get('dirty')})"
        for r in runs
        if (producer := r.manifest.get("producer", {})).get("git_sha") != commit
        or producer.get("dirty") is not False
    ]
    if foreign:
        raise IsingDataError(
            f"runs not made by the freeze tag's commit {commit} from a clean tree: {foreign}; "
            "a frozen prereg scores only runs generated from its tagged commit"
        )


def _cells(
    doc: dict[str, Any],
    L: int,
    inputs: Inputs,
    cadence: str,
    tau_c: float,
    tables: dict[str, list[Row]],
    peaks: Peaks,
    rng: np.random.Generator,
) -> Prepared:
    """Every sensitivity cell of one L and cadence: AUC rows, plus surrogate, alarm and peak rows
    at the headline cell, whose peak positions also go to `peaks` for the dynamic test. The null
    rows depend on h only through the span, which is mostly the same at every h, so they are
    scored and surrogate-tested once per pipeline and reused (#70)."""
    ind, sens, sc = doc["indicators"], doc["sensitivity"], doc["scoring"]
    cells = sc["surrogates"]["cells"]  # where the surrogate tests run
    segment = float(doc["event"]["segment"])
    horizons = [float(h) for h in doc["event"]["horizons"]]
    band = (max(horizons) + segment, min(horizons))
    prep = prepare(
        inputs, cadence, tau_c, float(ind.get("thin", 0.0)), band
    )  # dense alone reads it
    headline = (float(sens["headline"]["window"]), float(sens["headline"]["bandwidth"]))
    null_scores: dict[tuple[str, Pipeline], list[np.ndarray]] = {}
    null_pvalues: dict[tuple[str, Pipeline], dict[str, np.ndarray]] = {}
    for window_taus in map(float, sens["window"]):
        window = prep.samples(window_taus, tau_c)
        for bandwidth_taus in map(float, sens["bandwidth"]):
            bandwidth = None
            if ind["detrend"] == "gaussian":
                bandwidth = bandwidth_taus * tau_c / prep.sweeps
            top = (window_taus, bandwidth_taus) == headline
            for h in horizons:
                for observable in ind["observables"]:
                    pos_rows, length = positives(prep, observable, h, segment, window)
                    span = window + length - 1
                    neg_rows = [null_slices(s.observable(observable), span) for _, s in prep.nulls]
                    for name in _indicators(ind, observable):
                        raw = name in RAW
                        if raw and bandwidth_taus != headline[1]:
                            continue  # no bandwidth to vary: one cell per window
                        pipe = Pipeline(name, window, length, bandwidth, int(ind["lag"]))
                        key = (observable, pipe)  # the pipeline fixes the span
                        if key not in null_scores:
                            null_scores[key] = [pipe(g) for g in neg_rows]
                        cell = {
                            "indicator": name,
                            "observable": observable,
                            "cadence": cadence,
                            "L": L,
                            "h": h,
                            "window_taus": window_taus,
                            "bandwidth_taus": np.nan if raw else bandwidth_taus,
                            "headline": top,
                            "direction": _direction(ind, observable, name),
                        }
                        pos, neg = [pipe(g) for g in pos_rows], null_scores[key]
                        samples = {
                            "window": window,
                            "bandwidth": np.nan if bandwidth is None or raw else bandwidth,
                            "length": length,
                            "sweeps_per_sample": prep.sweeps,
                            "tau_c": tau_c,
                        }
                        tables["auc"].append(_auc_row(cell, pos, neg, sc, rng) | samples)
                        if (top and (h == 0 or cells != "headline-h0")) or cells == "all":
                            direction = cell["direction"]
                            test = scoring.sequential_test if "latents" in doc else None
                            if key not in null_pvalues:
                                null_pvalues[key] = _pvalues(
                                    neg_rows, pipe, sc, rng, direction, test
                                )
                            p_pos = _pvalues(pos_rows, pipe, sc, rng, direction, test)
                            rows = _surrogate_rows(cell, p_pos, null_pvalues[key], sc)
                            tables["surrogates"] += rows
            if top:
                tables["alarms"] += _alarms(doc, prep, L, tau_c, window, bandwidth)
                rows, found = _peaks(doc, prep, L, window, bandwidth)
                tables["peaks"] += rows
                for key, groups in found.items():
                    peaks.setdefault(key, {})[L] = groups
    return prep


def _indicators(ind: dict[str, Any], observable: str) -> list[str]:
    """The indicators scored on an observable: every one, or those indicators.pairs gives it."""
    if "pairs" not in ind:
        return ind["names"]
    return [name for paired, name in ind["pairs"] if paired == observable]


def _direction(ind: dict[str, Any], observable: str, name: str) -> str:
    """The trend an indicator is expected to take on an observable."""
    own = ind.get("direction_by_observable", {}).get(observable, {})
    return own.get(name, ind["direction"][name])


def _auc_row(
    cell: Row,
    pos: list[np.ndarray],
    neg: list[np.ndarray],
    sc: dict[str, Any],
    rng: np.random.Generator,
) -> Row:
    """AUC over all seeds, with a CI from resampling seeds, then replicas within each seed."""
    draws = scoring.hierarchical_bootstrap(scoring.auc, [pos, neg], int(sc["bootstrap"]), rng)
    lo, hi = scoring.percentile_interval(draws, float(sc["ci_level"]))
    p, n = np.concatenate(pos), np.concatenate(neg)
    return cell | {
        "auc": scoring.auc(p, n),
        "auc_lo": lo,
        "auc_hi": hi,
        "n_pos": int(np.isfinite(p).sum()),
        "n_neg": int(np.isfinite(n).sum()),
        "undefined": int((~np.isfinite(p)).sum() + (~np.isfinite(n)).sum()),
    }


def _pvalues(
    rows: list[np.ndarray],
    pipe: Pipeline,
    sc: dict[str, Any],
    rng: np.random.Generator,
    direction: str,
    sequential: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Per surrogate test of the prereg, the one-sided p-value of every row of every seed. A raw
    series has no ar1 test: fitted to a trend, ar1 draws near-random walks with no trend. With
    `sequential` (a latent file: _surrogate_rows reads p at alpha alone, and 999 surrogates per
    row of every checkpoint would take days) a row stops drawing once its decision is settled;
    ising-ews-v1's frozen draws stay as they were."""
    s, x = sc["surrogates"], np.concatenate(rows)
    tests = {
        kind: surrogates.KINDS[kind]
        for kind in (s["primary"], s["secondary"])
        if not (kind == "ar1" and pipe.indicator in RAW)
    }
    if sequential is not None:
        alpha = float(sc["alpha"])
        return {
            kind: sequential(x, pipe, make, int(s["n"]), rng, direction, alpha)[1]
            for kind, make in tests.items()
        }
    return {
        kind: scoring.surrogate_test(x, pipe, make, int(s["n"]), rng, direction)[1]
        for kind, make in tests.items()
    }


def _surrogate_rows(
    cell: Row, p_pos: dict[str, np.ndarray], p_neg: dict[str, np.ndarray], sc: dict[str, Any]
) -> list[Row]:
    """The share of positive and of null series each surrogate test flags at alpha."""
    s, alpha = sc["surrogates"], float(sc["alpha"])
    return [
        cell
        | {
            "surrogate": kind,
            "n_surrogates": int(s["n"]),
            "alpha": alpha,
            "detected": float(np.nanmean(p_pos[kind] <= alpha)),
            "false_positive": float(np.nanmean(p_neg[kind] <= alpha)),
            "n_pos": len(p_pos[kind]),
            "n_neg": len(p_neg[kind]),
        }
        for kind in p_pos
    ]


def _alarms(
    doc: dict[str, Any],
    prep: Prepared,
    L: int,
    tau_c: float,
    window: int,
    bandwidth: float | None,
) -> list[Row]:
    """A descriptive row per scoring.alarms indicator, or per judged pair in a prereg with
    indicators.pairs (#97): theta from the null blocks of half the seeds at the target
    false-alarm rate per tau_c, the rate on the other half, and the eps_L
    at which each ramp replica's s(t) first rises above theta inside the band (NaN: a miss, or
    censored: above theta already as the band opens, so its alarm came before the band and is not
    measured here). When no theta alarms as often as the target, the row says unattainable and
    every field that needs a theta is NaN. s(t) runs every `every` tau_c over each whole series,
    each score seeing only its past."""
    a = doc["scoring"]["alarms"]
    per = float(a["report_per"])
    pairs = doc["indicators"].get("pairs")
    if pairs is None:
        pairs = [(doc["fss"]["dynamic"]["observable"], name) for name in a["indicators"]]
    every = max(1, round(float(a["every"]) * tau_c / prep.sweeps))
    taus_per_score = every * prep.sweeps / tau_c
    lowest = min(float(h) for h in doc["event"]["horizons"])
    band_top, band_bottom = (float(v) for v in a["band"])
    rows = []
    for observable, name in pairs:
        _, length = positives(prep, observable, lowest, float(doc["event"]["segment"]), window)
        pipe = Pipeline(name, window, length, bandwidth, int(doc["indicators"]["lag"]))
        direction = _direction(doc["indicators"], observable, name)  # falling: alarms on -s(t)
        seeds = sorted({run.seed for run, _ in prep.nulls})
        halves: list[list[np.ndarray]] = [[], []]  # alternate seeds set theta and test it
        for run, s in prep.nulls:
            scores = trailing_score(s.observable(observable), pipe, every)[1]
            halves[seeds.index(run.seed) % 2].append(scoring.oriented(scores, direction))
        if not halves[1]:
            raise IsingDataError(f"L = {L}: alarms need null runs from two seeds, one per half")
        train, test = (_ragged(h) for h in halves)
        duration = tuple(sum(b.size for b in h) * taus_per_score for h in halves)
        found = scoring.threshold_at_far(train, float(a["far"]), duration[0])
        theta, nan = found.theta, float("nan")
        row = {
            "indicator": name,
            "observable": observable,
            "cadence": prep.cadence,
            "L": L,
            "direction": direction,
            "theta": theta,
            "attainable": found.attainable,
            "far_max": found.far_max * per,
            "far_set": nan,
            "far_test": nan,
            "far_per_taus": per,
            "alarm_eps_median": nan,
            "alarm_eps_q1": nan,
            "alarm_eps_q3": nan,
            "censored": nan,
            "miss_rate": nan,
            "n": sum(len(s.replicas) for _, s in prep.ramps),
        }
        rows.append(row)
        if not found.attainable:
            continue
        groups, censored = [], []
        for run, s in prep.ramps:
            times, scores = trailing_score(s.observable(observable), pipe, every)
            scores = scoring.oriented(scores, direction)
            eps = run.eps(s.temperature[times])
            inside = np.flatnonzero((eps <= band_top) & (eps >= band_bottom))
            if len(inside) == 0:
                raise IsingDataError(f"{run.run_id}: no s(t) inside the alarm band {a['band']}")
            entry = scores[:, max(0, inside[0] - 1)]  # the score as the band opens
            alarm, late = scoring.first_alarm(eps[inside], scores[:, inside], theta, entry)
            groups.append(alarm)
            censored.append(late)
        alarm, late = np.concatenate(groups), np.concatenate(censored)
        hit = alarm[np.isfinite(alarm)]
        measured = ~late
        q1, median, q3 = np.quantile(hit, [0.25, 0.5, 0.75]) if len(hit) else (nan,) * 3
        row |= {
            "far_set": scoring.false_alarm_rate(train, theta, duration[0]) * per,
            "far_test": scoring.false_alarm_rate(test, theta, duration[1]) * per,
            "alarm_eps_median": float(median),
            "alarm_eps_q1": float(q1),
            "alarm_eps_q3": float(q3),
            "censored": float(np.mean(late)),
            "miss_rate": float(np.mean(~np.isfinite(alarm[measured]))) if measured.any() else nan,
        }
    return rows


def _peaks(
    doc: dict[str, Any], prep: Prepared, L: int, window: int, bandwidth: float | None
) -> tuple[list[Row], dict[tuple[str, str], list[np.ndarray]]]:
    """Where each slow-ramp replica's indicator peaks (#98), for each one _peak_statistics names:
    the headline pipeline's detrend, window and lag along the whole ramp, and the eps_L of its
    extremum in the indicator's expected direction inside fss.dynamic.search (NaN on the band's
    edge, as near it as _edge_margin says, or with no value in the band: not located). A descriptive
    row per indicator; the per-seed positions go to the dynamic test, which counts every one not
    located. edge_fraction is their share, but a file with edge_margin (#121) keeps the replicas
    with no value apart, in `undefined`: a bug that empties a channel reads differently from a
    channel with no peak in the band. ising-ews-v1's table holds both in edge_fraction, as it froze
    them. Only a position is read, so a slope's scale, per sample here, plays no part."""
    dyn, ind = doc["fss"]["dynamic"], doc["indicators"]
    band = (float(dyn["search"][0]), float(dyn["search"][1]))
    rows, found = [], {}
    for key, observable, name, direction in _peak_statistics(doc):
        pipe = Pipeline(name, window, 2, bandwidth, int(ind["lag"]))  # rolling ignores length
        groups, empty = [], []
        for run, s in prep.ramps:
            ramp_segment(s, run, *band)  # refuses a ramp that does not cross the whole band
            values = pipe.rolling(s.observable(observable))
            eps, margin = run.eps(s.temperature), _edge_margin(doc, prep, run, window)
            groups.append(fss.peak_position(values, eps, band, direction, margin))
            empty.append(np.isnan(values[:, (eps <= band[0]) & (eps >= band[1])]).all(axis=1))
        peak = np.concatenate(groups)
        located = peak[np.isfinite(peak)]
        q1, median, q3 = np.quantile(located, [0.25, 0.5, 0.75]) if len(located) else [np.nan] * 3
        found[(key, prep.cadence)] = groups
        row = {
            "indicator": name,
            "observable": observable,
            "cadence": prep.cadence,
            "L": L,
            "direction": direction,
            "peak_eps_median": float(median),
            "peak_eps_q1": float(q1),
            "peak_eps_q3": float(q3),
            "n": len(peak),
            "edge_fraction": float(np.mean(np.isnan(peak))),
        }
        if "edge_margin" in dyn:  # ising-ews-v1's rows keep their columns and their count
            undefined = np.concatenate(empty)
            row |= {
                "edge_fraction": float(np.mean(np.isnan(peak) & ~undefined)),
                "undefined": float(np.mean(undefined)),
            }
        rows.append(row)
    return rows, found


def _edge_margin(doc: dict[str, Any], prep: Prepared, run: Run, window: int) -> float:
    """How far inside the search band, in eps_L, a peak still counts as not located: one headline
    window of this ramp, sweeps of the window over sweeps per unit of eps_L, in a file that says so
    (fss.dynamic.edge_margin, #115), and nothing in ising-ews-v1, which has no such key."""
    if "edge_margin" not in doc["fss"]["dynamic"]:
        return 0.0
    return window * prep.sweeps / (float(run.schedule["params"]["rate"]) * run.Tc / run.L)


def _peak_statistics(doc: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(its key in the dynamic test, observable, indicator, direction) of each peak read: every
    scored indicator of fss.dynamic.observable or, in a prereg with fss.dynamic.statistics, each of
    those, then each judged pair's as well, reported but never judged (AC1 peaks at Tc, as v1's
    does). A level has no peak to report: its Kendall's tau is its statistic."""
    dyn, ind = doc["fss"]["dynamic"], doc["indicators"]
    if "statistics" not in dyn:
        return [(n, dyn["observable"], n, ind["direction"][n]) for n in ind["names"]]
    directions = {"max": "greater", "min": "less"}
    judged = [
        (_statistic_key(s), s["observable"], s["indicator"], directions[s["extremum"]])
        for s in dyn["statistics"]
    ]
    keys = {key for key, *_ in judged}
    return judged + [
        (key, observable, name, _direction(ind, observable, name))
        for observable, name in ind["pairs"]
        if name not in RAW and (key := f"{observable}_{name}") not in keys
    ]


def _statistic_key(entry: dict[str, str]) -> str:
    return f"{entry['observable']}_{entry['indicator']}"


def _ragged(blocks: list[np.ndarray]) -> np.ndarray:
    """Stack s(t) rows of unequal length, padding with NaN (no score, never an alarm)."""
    width = max(b.shape[1] for b in blocks)
    return np.concatenate(
        [np.pad(b, ((0, 0), (0, width - b.shape[1])), constant_values=np.nan) for b in blocks]
    )


def _fss_row(quantity: str, cadence: str, role: str, fit: fss.Exponent, judged: Row) -> Row:
    return {
        "quantity": quantity,
        "cadence": cadence,
        "role": role,
        "estimate": fit.kappa,
        "lo": fit.ci[0],
        "hi": fit.ci[1],
        "corrected": fit.kappa_corrected,
        "draws": fit.draws,  # on every row: a Parquet table keeps only its first row's columns
        "draws_used": fit.draws_used,
        **judged,
    }


def _dynamic(
    doc: dict[str, Any], peaks: Peaks, rng: np.random.Generator, trained: bool = False
) -> list[Row]:
    """1/nu from the median peak eps_L over L, per fss.dynamic indicator and cadence. Only
    fss.dynamic.cadence carries the pre-registered verdict (role dynamic); another cadence's row
    is a robustness check (role dynamic-robustness, no verdict). A fss.dynamic.statistics entry
    is judged like an indicator, its row's role dynamic-<its role>; one of role descriptive is
    computed alike and has no verdict, and one judged on the twins alone (judged = "twin") reads
    REPORTED on a `trained` pass. A peak on the search band's edge, or inside _edge_margin, was not
    located and is left out. The verdict, in this order: inconclusive when 1/nu has no value, when
    more than half of an L's replicas were not located, or when fewer than
    fss.dynamic.min_draw_share of the bootstrap draws give 1/nu a value (the CI would rest on a
    sign-selected subset); otherwise fss.match and fss.excludes, combined."""
    dyn, sc = doc["fss"]["dynamic"], doc["scoring"]
    expected, tolerance = float(dyn["expected_inverse_nu"]), float(dyn["tolerance"])
    roles = {_statistic_key(entry): entry["role"] for entry in dyn.get("statistics", [])}
    twins = {_statistic_key(e) for e in dyn.get("statistics", []) if e.get("judged") == "twin"}
    # dispatched as _peak_statistics is: what is read is judged, whatever else the section holds
    named = roles if "statistics" in dyn else dyn["indicators"]
    rows = []
    for (name, cadence), samples in peaks.items():
        if name not in named:
            continue  # reported in the peaks table, never judged
        fit = fss.peak_convergence(samples, int(sc["bootstrap"]), rng, float(sc["ci_level"]))
        judged = cadence == dyn["cadence"]
        lost = max(np.mean(np.isnan(np.concatenate(g))) for g in samples.values())
        if not judged or roles.get(name) == "descriptive":
            verdict = ""  # a robustness check or a descriptive statistic, never a verdict
        elif trained and name in twins:
            verdict = REPORTED
        elif not np.isfinite(fit.kappa) or lost > 0.5 or fit.draws_used < dyn["min_draw_share"]:
            verdict = fss.INCONCLUSIVE  # unmeasured: neither a pass nor a fail
        else:
            match = fss.match(fit.kappa, fit.ci, expected, tolerance)
            verdict = fss.combine(match, fss.excludes(fit.ci, 0.0, tolerance))
        row = {"expected": expected, "tolerance": tolerance, "verdict": verdict}
        role = "dynamic" if judged else "dynamic-robustness"
        if name in roles:
            role = f"dynamic-{roles[name]}"
        rows.append(_fss_row(f"inverse_nu_{name}", cadence, role, fit, row))
    return rows


def _quantity(quantity: str) -> tuple[str, str]:
    """An fss.static quantity's (statistic, observable): l2_var_pc_1 is ("l2_var", "pc_1"), and
    the frozen one_minus_ac1 of ising-ews-v1 reads |m|."""
    if quantity == "one_minus_ac1":
        return quantity, "abs_m"
    for statistic in ("l2_var", "mean", "skewness", "tau_int", "one_minus_ac1"):
        if quantity.startswith(f"{statistic}_"):
            return statistic, quantity.removeprefix(f"{statistic}_")
    raise ValueError(f"not an fss.static quantity: {quantity!r}")


def _statistic(name: str) -> Callable[[int, np.ndarray, int], float]:
    """I(L) for one statistic of an fss.static quantity, from replica rows, L and the sweeps per
    row."""

    def skewness(L: int, a: np.ndarray, stride: int) -> float:
        pooled = a.ravel()
        return float(indicators.rolling_skewness(pooled, len(pooled))[-1])

    def one_minus_ac1(L: int, a: np.ndarray, stride: int) -> float:  # (replicas, bursts, frames)
        return float(1.0 - indicators.burst_autocorrelation(a.reshape(1, -1, a.shape[-1]))[0])

    statistics: dict[str, Callable[[int, np.ndarray, int], float]] = {
        "l2_var": lambda L, a, stride: float(L**2 * a.var(ddof=1)),
        "mean": lambda L, a, stride: float(a.mean()),
        "skewness": skewness,
        "tau_int": lambda L, a, stride: float(autocorr.estimate(a).tau * stride),
        "one_minus_ac1": one_minus_ac1,
    }
    return statistics[name]


def _verdict(role: str, fit: fss.Exponent, expected: float, tolerance: float) -> str:
    if role in ("positive-control", "pipeline-check"):
        return fss.match(fit.kappa, fit.ci, expected, tolerance)
    if role == "reference":
        return fss.excludes(fit.ci, 0.0, tolerance)
    return ""


def _static(
    doc: dict[str, Any], chosen: dict[int, Inputs], rng: np.random.Generator, trained: bool = False
) -> tuple[list[Row], int]:
    """Exponents at the staircase plateau nearest Tc, one row per fss.static entry: the measured
    rows feed the moments and tau_int, bursts of frames one sweep apart feed 1 - AC1. A latents
    table holds the matched cadence alone, so its frames feed both. On a `trained` pass a row judged
    on the twins alone (judged = "twin") is computed all the same and reads REPORTED."""
    T = float(doc["data"]["staircase"]["temperature"])
    sc = doc["scoring"]
    latent = any(inputs.latents for inputs in chosen.values())
    observables = {_quantity(spec["quantity"])[1] for spec in doc["fss"]["static"]}
    moments: dict[str, dict[int, list[np.ndarray]]] = {
        name: {L: [] for L in chosen} for name in observables
    }
    bursts: dict[str, dict[int, list[np.ndarray]]] = {
        name: {L: [] for L in chosen} for name in observables
    }
    strides: dict[int, int] = {}
    dropped = 0
    for L, inputs in chosen.items():
        for run in inputs.staircases:
            found, lost = plateaus(run, "matched" if latent else "dense", inputs.latents)
            dropped += lost
            here = [t for t in found if _close(t, T)]
            if not here:
                if _thermalised(run, T) is False:
                    continue  # failed its thermalised check: counted above, not scored
                raise IsingDataError(f"{run.run_id}: no plateau at T = {T} for the static fit")
            plateau = found[here[0]]
            frames = plateau
            if not latent:
                strides[L] = int(run.schedule["stats_strides"][int(plateau.segment[0])])
                frames = plateaus(run, "matched")[0].get(here[0])
            burst = run.schedule["params"]["burst"]
            for name in observables:
                moments[name][L].append(plateau.observable(name))
                if frames is not None and int(burst["stride"]) == 1:
                    per = int(burst["frames"])
                    whole = len(frames.steps) // per * per
                    values = frames.observable(name)[:, :whole]
                    bursts[name][L].append(values.reshape(len(frames.replicas), -1, per))
    rows = []
    for spec in doc["fss"]["static"]:
        quantity, role = spec["quantity"], spec["role"]
        name, observable = _quantity(quantity)
        source = bursts if name == "one_minus_ac1" else moments
        samples = {L: groups for L, groups in source[observable].items() if groups}
        if len(samples) < 2:
            usable = sorted(samples)
            raise IsingDataError(f"fss {quantity}: the T = {T} plateau is usable at {usable}")
        stat = _statistic(name)

        def statistic(L: int, a: np.ndarray, stat: Callable[..., float] = stat) -> float:
            return stat(L, a, strides.get(L, 1))

        fit = fss.fit_exponent(samples, statistic, int(sc["bootstrap"]), rng, float(sc["ci_level"]))
        expected = float(spec.get("expected", np.nan))
        tolerance = float(spec.get("tolerance", np.nan))  # validation: judged roles state one
        cadence = "matched" if latent or name == "one_minus_ac1" else "dense"
        judged = {
            "expected": expected,
            "tolerance": tolerance,
            "verdict": _verdict(role, fit, expected, tolerance),
        }
        if trained and spec.get("judged") == "twin":
            judged["verdict"] = REPORTED
        rows.append(_fss_row(quantity, cadence, role, fit, judged))
    return rows, dropped
