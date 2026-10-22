"""Ising adapter: simulator runs in, (batch, time) arrays of |m| and e per site out, and the Ising
experiment of a pre-registration scored from them.

Runs follow tipping_point.ising.io's layout. This module reads the manifest (L, Tc, seed,
replicas, blocks, schedule), `stats` (step, replica, temperature, segment, phase, magnetisation,
energy) and `configs` (the same per saved frame, with burst and frame), never the spins. A run
without complete.json is refused; a staircase plateau that failed complete.json's thermalised
check is dropped and counted. Schedules of kind ramp, fixed and staircase are read; each block of
a fixed run holds one temperature, and a null block is matched on its eps_L.

tau_c(L), the time unit, comes from the #50 pilot record <data dir>/ising/<pilot run>/pilot.json:
  {"kernel_commit": str,
   "per_L": {"16": {"tau_c", "tau_c_binning", "tau_e", "n_over_tau", "replicas", "sweeps",
                    "seed", optional "tau_c_ci": [lo, hi]}, ...},
   "fit": {"A", "z", "A_ci", "z_ci"}}
Only the measured per_L values are read (other keys are ignored); the fit, and any tau_c it would
extrapolate to an unmeasured L, never enter scoring. Nothing here holds a tau_c of its own:
without the record, or for an L it did not measure, scoring stops.

Cadences: `dense` is every stats row, thinned to one sample per `thin` tau_c; `matched` is every
saved frame (bursts of consecutive sweeps), exactly what the JEPA sees. Windows, bandwidths and
segments are converted from tau_c and eps_L into samples per cadence and L.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..ising import autocorr, io
from . import fss, indicators, scoring, surrogates
from .trend import Pipeline, trailing_score

PILOT = "pilot.json"
OBSERVABLES = ("abs_m", "energy")
_STATS = ("step", "replica", "temperature", "segment", "phase", "magnetisation", "energy")
_CONFIGS = (
    "step",
    "replica",
    "temperature",
    "segment",
    "burst",
    "frame",
    "magnetisation",
    "energy",
)
Row = dict[str, Any]


class IsingDataError(RuntimeError):
    """Input the scorer cannot use: missing, incomplete, ambiguous or inconsistent."""


# ---------------------------------------------------------------- reading


@dataclass(frozen=True)
class Pilot:
    path: Path
    sha256: str
    kernel_commit: str
    tau_c: dict[int, float]
    tau_c_ci: dict[int, list[float]]  # the record's 95% interval, where it gives one

    def tau(self, L: int) -> float:
        if L not in self.tau_c:
            raise IsingDataError(f"{self.path} has no tau_c for L = {L}; it must cover every L")
        return self.tau_c[L]


def read_pilot(data_dir: Path, run_id: str, sha256: str | None = None) -> Pilot:
    """The tau_c record of a pilot run; with `sha256`, the file must be exactly that one."""
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
        commit = record["kernel_commit"]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise IsingDataError(f"{path}: not a pilot record ({exc!r})") from exc
    if not isinstance(commit, str) or not commit:
        raise IsingDataError(f"{path}: kernel_commit must name the kernel's commit")
    bad = sorted(L for L, tau in tau_c.items() if not 0 < tau < math.inf)
    if bad or not tau_c:
        raise IsingDataError(f"{path}: tau_c must be positive and finite for every L (bad: {bad})")
    return Pilot(path, digest, commit, tau_c, ci)


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

    def observable(self, name: str) -> np.ndarray:
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
        )


def series(run: Run, cadence: str) -> list[Series]:
    """Every recorded row of a cadence (dense: stats, matched: configs), one Series per
    temperature block, in step order."""
    table, names = {"dense": ("stats", _STATS), "matched": ("configs", _CONFIGS)}[cadence]
    paths = io.chunks(run.path, table)
    if not paths:
        raise IsingDataError(f"{run.run_id}: no {table} rows for the {cadence} cadence")
    data = pa.concat_tables([pq.read_table(p, columns=list(names)) for p in paths])
    replicas, blocks = int(run.params["replicas"]), int(run.params["blocks"])
    lattices = replicas * blocks
    rows = data.num_rows // lattices
    if rows * lattices != data.num_rows:
        raise IsingDataError(f"{run.run_id}: {table} does not hold whole steps of {lattices} rows")
    grid = {name: data[name].to_numpy().reshape(rows, lattices) for name in names}
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


def plateaus(run: Run, cadence: str) -> tuple[dict[float, Series], int]:
    """A staircase's measured rows per plateau temperature, without the plateaus that failed the
    thermalised check; returns them and how many were dropped."""
    if run.kind != "staircase":
        raise IsingDataError(f"{run.run_id} is a {run.kind} run, not a staircase")
    checks = run.checks.get("thermalised", [])
    failed = {round(c["temperature"], 9) for c in checks if not c["passed"]}
    (s,) = series(run, cadence)
    s = measured(s)
    kept, dropped = {}, 0
    for segment in np.unique(s.segment):
        part = s.take(s.segment == segment)
        T = float(part.temperature[0])
        if round(T, 9) in failed:
            dropped += 1
        else:
            kept[T] = part
    return kept, dropped


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

    def runs(self) -> Iterator[Run]:
        yield from self.ramps
        yield from (run for run, _ in self.nulls)
        yield from self.staircases


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def _thermalised(run: Run, temperature: float) -> bool:
    """False when complete.json's thermalised check failed at that temperature."""
    checks = run.checks.get("thermalised", [])
    return not any(_close(c["temperature"], temperature) and not c["passed"] for c in checks)


def _slots(run: Run, data: dict[str, Any]) -> list[tuple[Any, ...]]:
    """The prereg slots a run fills: ("ramp", seed), ("staircase", seed) or
    ("null", seed, eps_L) with its block."""
    if run.L not in data["L"] or run.seed not in data["seeds"]:
        return []
    params = run.schedule.get("params", {})
    if run.kind == data["ramps"]["kind"]:
        x = params.get("x")
        return [("ramp", run.seed)] if x is not None and _close(x, data["ramps"]["x"]) else []
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
    runs: Sequence[Run], incomplete: Sequence[Run], data: dict[str, Any]
) -> dict[int, Inputs]:
    """For each L, exactly one complete run per (kind, seed) and one null block per
    (seed, eps_L). A slot left empty or filled twice, or a matching run that has not completed,
    is an error; listing a run in data.exclusions takes it out. A null block that failed its
    thermalised check fills its slot but is dropped from scoring and counted."""
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
            raise IsingDataError(f"no complete run for {missing}")
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
        passed = [(r, b) for r, b in blocks if _thermalised(r, _block_temperature(r, b))]
        if not passed:
            raise IsingDataError(f"L = {L}: every null block failed its thermalised check")
        chosen[L] = Inputs(
            ramps=[filled[(L, "ramp", s)][0][0] for s in data["seeds"]],
            nulls=passed,
            staircases=[filled[(L, "staircase", s)][0][0] for s in data["seeds"]],
            dropped=[f"{r.run_id} block {b}" for r, b in blocks if (r, b) not in passed],
        )
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
    ramps = [(run, series(run, cadence)[0]) for run in inputs.ramps]
    nulls = [(run, measured(series(run, cadence)[block])) for run, block in inputs.nulls]
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
    tables: dict[str, list[Row]]  # auc, surrogates, alarms, fss
    inputs: list[Run]
    pilot: Pilot
    dropped: dict[str, list[str] | int]  # failed null blocks, and how many plateaus were dropped


def evaluate(
    doc: dict[str, Any],
    data_dir: Path,
    rng: np.random.Generator,
    say: Callable[[str], None] = print,
) -> Evaluation:
    """Score a pre-registration's Ising experiment: the AUC of every cell, the surrogate tests and
    alarms at the headline cell, and static and dynamic finite-size scaling. It computes only:
    writing results and logging trials are the caller's."""
    data = doc["data"]
    pinned = data["tau_c"]["pilot_sha256"]
    pilot = read_pilot(data_dir, data["tau_c"]["pilot_run"], pinned if len(pinned) == 64 else None)
    chosen = select(*find_runs(data_dir), data)
    tables: dict[str, list[Row]] = {"auc": [], "surrogates": [], "alarms": []}
    alarm_eps: dict[tuple[str, str], dict[int, list[np.ndarray]]] = {}
    for L, inputs in chosen.items():
        for cadence in doc["indicators"]["cadences"]:
            prep = _cells(doc, L, inputs, cadence, pilot.tau(L), tables, alarm_eps, rng)
            say(f"L = {L}, {cadence}: {prep.sweeps:g} sweeps per sample, tau_c = {pilot.tau(L):g}")
    static, plateaus_dropped = _static(doc, chosen, rng)
    tables["fss"] = static + _dynamic(doc, alarm_eps, rng)
    inputs = list({run.run_id: run for i in chosen.values() for run in i.runs()}.values())
    null_blocks = [block for i in chosen.values() for block in i.dropped]
    dropped = {"null_blocks": null_blocks, "plateaus": plateaus_dropped}
    return Evaluation(tables, inputs, pilot, dropped)


def _cells(
    doc: dict[str, Any],
    L: int,
    inputs: Inputs,
    cadence: str,
    tau_c: float,
    tables: dict[str, list[Row]],
    alarm_eps: dict[tuple[str, str], dict[int, list[np.ndarray]]],
    rng: np.random.Generator,
) -> Prepared:
    """Every sensitivity cell of one L and cadence: AUC rows, plus surrogate and alarm rows at the
    headline cell."""
    ind, sens, sc = doc["indicators"], doc["sensitivity"], doc["scoring"]
    segment = float(doc["event"]["segment"])
    horizons = [float(h) for h in doc["event"]["horizons"]]
    band = (max(horizons) + segment, min(horizons))
    prep = prepare(inputs, cadence, tau_c, float(ind["thin"]), band)
    headline = (float(sens["headline"]["window"]), float(sens["headline"]["bandwidth"]))
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
                    for name in ind["names"]:
                        pipe = Pipeline(name, window, length, bandwidth, int(ind["lag"]))
                        cell = {
                            "indicator": name,
                            "observable": observable,
                            "cadence": cadence,
                            "L": L,
                            "h": h,
                            "window_taus": window_taus,
                            "bandwidth_taus": bandwidth_taus,
                            "headline": top,
                        }
                        pos, neg = [pipe(g) for g in pos_rows], [pipe(g) for g in neg_rows]
                        samples = {
                            "window": window,
                            "bandwidth": np.nan if bandwidth is None else bandwidth,
                            "length": length,
                            "sweeps_per_sample": prep.sweeps,
                            "tau_c": tau_c,
                        }
                        tables["auc"].append(_auc_row(cell, pos, neg, sc, rng) | samples)
                        if top or sc["surrogates"]["cells"] == "all":
                            rows = _surrogate_rows(cell, pipe, pos_rows, neg_rows, sc, rng)
                            tables["surrogates"] += rows
            if top:
                rows, eps = _alarms(doc, prep, L, tau_c, window, bandwidth)
                tables["alarms"] += rows
                for key, groups in eps.items():
                    alarm_eps.setdefault(key, {})[L] = groups
    return prep


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


def _surrogate_rows(
    cell: Row,
    pipe: Pipeline,
    pos_rows: list[np.ndarray],
    neg_rows: list[np.ndarray],
    sc: dict[str, Any],
    rng: np.random.Generator,
) -> list[Row]:
    """The share of positive and of null series each surrogate test flags at alpha."""
    s, alpha = sc["surrogates"], float(sc["alpha"])
    rows = []
    for kind in (s["primary"], s["secondary"]):
        make, n = surrogates.KINDS[kind], int(s["n"])
        _, p_pos = scoring.surrogate_test(np.concatenate(pos_rows), pipe, make, n, rng)
        _, p_neg = scoring.surrogate_test(np.concatenate(neg_rows), pipe, make, n, rng)
        rows.append(
            cell
            | {
                "surrogate": kind,
                "n_surrogates": n,
                "alpha": alpha,
                "detected": float(np.nanmean(p_pos <= alpha)),
                "false_positive": float(np.nanmean(p_neg <= alpha)),
                "n_pos": len(p_pos),
                "n_neg": len(p_neg),
            }
        )
    return rows


def _alarms(
    doc: dict[str, Any],
    prep: Prepared,
    L: int,
    tau_c: float,
    window: int,
    bandwidth: float | None,
) -> tuple[list[Row], dict[tuple[str, str], list[np.ndarray]]]:
    """theta from the null blocks of half the seeds at the target false-alarm rate per tau_c,
    the rate on the other half, and the eps_L of each ramp replica's first alarm in the band
    (NaN: a miss).
    s(t) runs every `every` tau_c over each whole series, each score seeing only its past."""
    a, dyn = doc["scoring"]["alarms"], doc["fss"]["dynamic"]
    observable, per = dyn["observable"], float(a["report_per"])
    every = max(1, round(float(a["every"]) * tau_c / prep.sweeps))
    taus_per_score = every * prep.sweeps / tau_c
    lowest = min(float(h) for h in doc["event"]["horizons"])
    _, length = positives(prep, observable, lowest, float(doc["event"]["segment"]), window)
    band_top, band_bottom = (float(v) for v in a["band"])
    rows, eps_out = [], {}
    for name in dyn["indicators"]:
        pipe = Pipeline(name, window, length, bandwidth, int(doc["indicators"]["lag"]))
        seeds = sorted({run.seed for run, _ in prep.nulls})
        halves: list[list[np.ndarray]] = [[], []]  # alternate seeds set theta and test it
        for run, s in prep.nulls:
            scores = trailing_score(s.observable(observable), pipe, every)[1]
            halves[seeds.index(run.seed) % 2].append(scores)
        if not halves[1]:
            raise IsingDataError(f"L = {L}: alarms need null runs from two seeds, one per half")
        train, test = (_ragged(h) for h in halves)
        duration = tuple(sum(b.size for b in h) * taus_per_score for h in halves)
        theta = scoring.threshold_at_far(train, float(a["far"]), duration[0])
        groups = []
        for run, s in prep.ramps:
            times, scores = trailing_score(s.observable(observable), pipe, every)
            eps = run.eps(s.temperature[times])
            inside = (eps <= band_top) & (eps >= band_bottom)
            groups.append(scoring.first_alarm(eps[inside], scores[:, inside], theta))
        eps_out[(name, prep.cadence)] = groups
        alarm = np.concatenate(groups)
        hit = alarm[np.isfinite(alarm)]
        q1, median, q3 = np.quantile(hit, [0.25, 0.5, 0.75]) if len(hit) else (np.nan,) * 3
        rows.append(
            {
                "indicator": name,
                "observable": observable,
                "cadence": prep.cadence,
                "L": L,
                "theta": theta,
                "far_set": scoring.false_alarm_rate(train, theta, duration[0]) * per,
                "far_test": scoring.false_alarm_rate(test, theta, duration[1]) * per,
                "far_per_taus": per,
                "alarm_eps_median": float(median),
                "alarm_eps_q1": float(q1),
                "alarm_eps_q3": float(q3),
                "miss_rate": float(np.mean(~np.isfinite(alarm))),
                "n": len(alarm),
            }
        )
    return rows, eps_out


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
        **judged,
    }


def _dynamic(
    doc: dict[str, Any],
    alarm_eps: dict[tuple[str, str], dict[int, list[np.ndarray]]],
    rng: np.random.Generator,
) -> list[Row]:
    """1/nu from the median alarm eps_L over L, per indicator and cadence."""
    dyn, sc = doc["fss"]["dynamic"], doc["scoring"]
    expected, tolerance = float(dyn["expected_inverse_nu"]), float(dyn["tolerance"])
    rows = []
    for (name, cadence), samples in alarm_eps.items():
        inv_nu, ci = fss.alarm_convergence(
            samples, int(sc["bootstrap"]), rng, float(sc["ci_level"])
        )
        verdict = fss.combine(
            fss.match(inv_nu, ci, expected, tolerance), fss.excludes(ci, 0.0, tolerance)
        )
        fit = fss.Exponent(inv_nu, ci, float("nan"), {})
        judged = {"expected": expected, "tolerance": tolerance, "verdict": verdict}
        rows.append(_fss_row(f"inverse_nu_{name}", cadence, "dynamic", fit, judged))
    return rows


def _statistic(quantity: str) -> Callable[[int, np.ndarray, int], float]:
    """I(L) for one fss.static quantity, from replica rows, L and the sweeps per row."""

    def skewness(L: int, a: np.ndarray, stride: int) -> float:
        pooled = a.ravel()
        return float(indicators.rolling_skewness(pooled, len(pooled))[-1])

    def one_minus_ac1(L: int, a: np.ndarray, stride: int) -> float:  # (replicas, bursts, frames)
        return float(1.0 - indicators.burst_autocorrelation(a.reshape(1, -1, a.shape[-1]))[0])

    statistics: dict[str, Callable[[int, np.ndarray, int], float]] = {
        "l2_var_abs_m": lambda L, a, stride: float(L**2 * a.var(ddof=1)),
        "mean_abs_m": lambda L, a, stride: float(a.mean()),
        "skewness_abs_m": skewness,
        "tau_int_abs_m": lambda L, a, stride: float(autocorr.estimate(a).tau * stride),
        "one_minus_ac1": one_minus_ac1,
    }
    return statistics[quantity]


def _verdict(role: str, fit: fss.Exponent, expected: float, tolerance: float) -> str:
    if role == "positive-control":
        return fss.match(fit.kappa, fit.ci, expected, tolerance)
    if role == "reference":
        return fss.excludes(fit.ci, 0.0, tolerance)
    return ""


def _static(
    doc: dict[str, Any], chosen: dict[int, Inputs], rng: np.random.Generator
) -> tuple[list[Row], int]:
    """Exponents at the staircase plateau nearest Tc, one row per fss.static entry: the measured
    rows feed the moments and tau_int, bursts of frames one sweep apart feed 1 - AC1."""
    T = float(doc["data"]["staircase"]["temperature"])
    sc, default_tolerance = doc["scoring"], float(doc["fss"]["dynamic"]["tolerance"])
    dense: dict[int, list[np.ndarray]] = {L: [] for L in chosen}
    bursts: dict[int, list[np.ndarray]] = {L: [] for L in chosen}
    strides: dict[int, int] = {}
    dropped = 0
    for L, inputs in chosen.items():
        for run in inputs.staircases:
            kept, lost = plateaus(run, "dense")
            dropped += lost
            here = [t for t in kept if _close(t, T)]
            if not here:
                continue  # failed its thermalised check: counted above, not scored
            plateau = kept[here[0]]
            dense[L].append(plateau.abs_m)
            strides[L] = int(run.schedule["stats_strides"][int(plateau.segment[0])])
            burst = run.schedule["params"]["burst"]
            frames = plateaus(run, "matched")[0].get(here[0])
            if frames is not None and int(burst["stride"]) == 1:
                per = int(burst["frames"])
                whole = len(frames.steps) // per * per
                bursts[L].append(frames.abs_m[:, :whole].reshape(len(frames.replicas), -1, per))
    rows = []
    for spec in doc["fss"]["static"]:
        quantity, role = spec["quantity"], spec["role"]
        source = bursts if quantity == "one_minus_ac1" else dense
        samples = {L: groups for L, groups in source.items() if groups}
        if len(samples) < 2:
            usable = sorted(samples)
            raise IsingDataError(f"fss {quantity}: the T = {T} plateau is usable at {usable}")
        stat = _statistic(quantity)

        def statistic(L: int, a: np.ndarray, stat: Callable[..., float] = stat) -> float:
            return stat(L, a, strides.get(L, 1))

        fit = fss.fit_exponent(samples, statistic, int(sc["bootstrap"]), rng, float(sc["ci_level"]))
        expected = float(spec.get("expected", np.nan))
        tolerance = float(spec.get("tolerance", default_tolerance))
        cadence = "matched" if quantity == "one_minus_ac1" else "dense"
        judged = {
            "expected": expected,
            "tolerance": tolerance,
            "verdict": _verdict(role, fit, expected, tolerance),
        }
        rows.append(_fss_row(quantity, cadence, role, fit, judged))
    return rows, dropped
