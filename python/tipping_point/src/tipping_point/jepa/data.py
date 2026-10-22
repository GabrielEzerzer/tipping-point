"""Simulator runs in, packed bursts in host memory out: what training samples from (Ising).

Training reads staircase plateaus only, an equilibrium world uniform in T, from dedicated seeds.
It refuses a run whose seed a pre-registration scores (PREREG_SEEDS, the data.seeds of
paper/prereg/ising-ews-v1.toml, plus any a JEPA pre-registration adds), ramps and fixed (null)
runs, and runs without complete.json: surprise is never fitted on the frames it will be scored
on. Plateaus that failed complete.json's thermalised check are dropped and counted; the
held-out-band ablation also drops the plateaus with T inside HELD_OUT (10 of the 40-point grid).

Bursts stay packed at one bit per spin, (bursts, frames, ceil(L*L/8)) uint8 in host memory
(66 MB, 262 MB and 1.05 GB for two seeds at L = 16, 32 and 64), and cross to the device as bytes;
lattice.unpack expands them there. Temperature travels along for monitors, never as an input.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor

from ..ews.ising import IsingDataError, Run, find_runs, open_run
from ..ising import io

PREREG_SEEDS = frozenset({1, 2, 3, 4, 5})  # paper/prereg/ising-ews-v1.toml data.seeds
HELD_OUT = (2.0, 2.5)  # the held-out-band ablation (#19): plateaus inside it are never trained on
_COLUMNS = ["step", "replica", "temperature", "burst", "frame", "spins"]


class DataError(IsingDataError):
    """Input that training or a readout cannot use: refused, missing, incomplete or inconsistent."""


@dataclass(frozen=True)
class Bursts:
    """The saved bursts of one size, packed."""

    L: int
    packed: Tensor  # (bursts, frames, ceil(L*L/8)) uint8, host memory
    temperature: np.ndarray  # (bursts,)

    def __len__(self) -> int:
        return int(self.packed.shape[0])

    @property
    def frames(self) -> int:
        return int(self.packed.shape[1])

    def sample(self, count: int, gen: torch.Generator) -> Tensor:
        """`count` bursts drawn uniformly, with replacement, by `gen` (a CPU generator)."""
        return self.packed[torch.randint(len(self), (count,), generator=gen)]

    @staticmethod
    def join(parts: Sequence[Bursts]) -> Bursts:
        sizes = {p.L for p in parts}
        if len(sizes) != 1:
            raise DataError(f"bursts of different sizes {sorted(sizes)} cannot be joined")
        return Bursts(
            parts[0].L,
            torch.cat([p.packed for p in parts]),
            np.concatenate([p.temperature for p in parts]),
        )


def packed_spins(column: pa.Array | pa.ChunkedArray, L: int) -> np.ndarray:
    """A spins column -> (rows, ceil(L*L/8)) uint8, still packed: a view on the Arrow buffer."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    width = io.packed_width(L)
    data = np.frombuffer(array.buffers()[1], dtype=np.uint8)
    rows = data[array.offset * width : (array.offset + len(array)) * width]
    return rows.reshape(len(array), width)


def read_bursts(run: Run, keep: Callable[[np.ndarray], np.ndarray] | None = None) -> Bursts:
    """Every saved burst of a run with its frames in order: rows sorted by (replica, burst,
    frame), each burst checked to hold its schedule's frames `stride` sweeps apart. `keep`
    selects rows by temperature (a whole plateau keeps or drops whole bursts)."""
    paths = io.chunks(run.path, "configs")
    if not paths:
        raise DataError(f"{run.run_id}: no configs table, so no saved frames")
    table = pa.concat_tables(pq.read_table(p, columns=_COLUMNS) for p in paths)
    temperature = table["temperature"].to_numpy()
    rows = np.flatnonzero(np.ones(len(temperature), bool) if keep is None else keep(temperature))
    burst = run.schedule["params"]["burst"]
    frames, stride = int(burst["frames"]), int(burst["stride"])
    keys = {name: table[name].to_numpy()[rows] for name in ("step", "replica", "burst", "frame")}
    order = np.lexsort((keys["frame"], keys["burst"], keys["replica"]))
    count = len(order) // frames
    if count == 0 or count * frames != len(order):
        raise DataError(f"{run.run_id}: {len(order)} rows kept, not whole bursts of {frames}")

    def grid(values: np.ndarray) -> np.ndarray:
        return values[order].reshape(count, frames)

    step, replica, label, frame = (grid(keys[k]) for k in ("step", "replica", "burst", "frame"))
    whole = (
        (frame == np.arange(frames)).all()
        and (np.diff(step, axis=1) == stride).all()
        and (replica == replica[:, :1]).all()
        and (label == label[:, :1]).all()
    )
    if not whole:
        raise DataError(f"{run.run_id}: bursts are not {frames} frames {stride} sweep(s) apart")
    index = rows[order]
    packed = packed_spins(table["spins"], run.L)[index].reshape(count, frames, -1)
    return Bursts(run.L, torch.from_numpy(packed), temperature[index].reshape(count, frames)[:, 0])


def plateaus(
    run: Run, held_out: tuple[float, float] | None
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    """The temperatures training may read from a staircase: its plateaus, without those that
    failed the thermalised check and those inside the held-out band. Returns the row filter and
    its record for the manifest."""
    grid = [float(T) for T in run.schedule["params"]["temperatures"]]
    checks = run.checks.get("thermalised", [])
    failed = [
        T
        for T in grid
        if any(abs(T - float(c["temperature"])) <= 1e-9 and not c["passed"] for c in checks)
    ]
    band = [
        T
        for T in grid
        if held_out is not None and held_out[0] <= T <= held_out[1] and T not in failed
    ]
    kept = np.array([T for T in grid if T not in failed and T not in band])

    def keep(temperature: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(temperature, return_inverse=True)
        if len(kept) == 0:
            return np.zeros(len(temperature), bool)
        return (np.abs(unique[:, None] - kept[None, :]).min(1) <= 1e-9)[inverse]

    record = {"plateaus": len(kept), "dropped_unthermalised": failed, "dropped_held_out": band}
    return keep, record


def check_training_run(run: Run, refused: frozenset[int]) -> None:
    if run.kind != "staircase":
        raise DataError(
            f"{run.run_id} is a {run.kind} run: training reads staircase plateaus only, "
            "never ramps or null runs"
        )
    if run.seed in refused:
        raise DataError(
            f"{run.run_id} has seed {run.seed}, which a pre-registration scores "
            f"({sorted(refused)}): it is never trained on"
        )


def find_staircases(
    data_dir: Path, sizes: Sequence[int], seeds: Sequence[int]
) -> dict[tuple[int, int], Run]:
    """Exactly one complete staircase run per (L, seed) under <data dir>/ising."""
    complete, incomplete = find_runs(data_dir)
    wanted = {(L, seed) for L in sizes for seed in seeds}

    def matches(run: Run) -> bool:
        return run.kind == "staircase" and (run.L, run.seed) in wanted

    pending = sorted(run.run_id for run in incomplete if matches(run))
    if pending:
        raise DataError(f"staircase runs without complete.json (running or crashed): {pending}")
    found: dict[tuple[int, int], list[Run]] = {}
    for run in filter(matches, complete):
        found.setdefault((run.L, run.seed), []).append(run)
    missing = sorted(wanted - found.keys())
    if missing:
        raise DataError(f"no complete staircase run for (L, seed) {missing}")
    doubled = {k: [r.run_id for r in v] for k, v in found.items() if len(v) > 1}
    if doubled:
        raise DataError(f"more than one staircase run for {doubled}")
    return {k: v[0] for k, v in found.items()}


@dataclass
class Dataset:
    train: dict[int, Bursts]
    val: dict[int, Bursts]
    sources: dict[str, Any]  # where the bursts came from: the manifest's data section


def load(
    data_dir: Path,
    train_runs: Sequence[str],
    val_runs: Sequence[str],
    held_out: tuple[float, float] | None = None,
    refused: frozenset[int] = PREREG_SEEDS,
    expect: dict[str, str] | None = None,
) -> Dataset:
    """Training and validation bursts from staircase runs named by run_id under
    <data dir>/ising. `expect` (run_id -> complete.json sha256) pins the runs a resumed training
    read before. Every size needs both; no seed may serve both."""

    def side(run_ids: Sequence[str]) -> tuple[dict[int, Bursts], list[dict[str, Any]]]:
        parts: dict[int, list[Bursts]] = {}
        records = []
        for run_id in run_ids:
            run = open_run(Path(data_dir) / "ising" / run_id)
            check_training_run(run, refused)
            if expect is not None and expect.get(run_id) != run.complete_sha256:
                raise DataError(f"{run_id}: complete.json changed since this training read it")
            keep, record = plateaus(run, held_out)
            bursts = read_bursts(run, keep)
            parts.setdefault(run.L, []).append(bursts)
            records.append(
                {
                    "run_id": run_id,
                    "L": run.L,
                    "seed": run.seed,
                    "complete_sha256": run.complete_sha256,
                    "bursts": len(bursts),
                    **record,
                }
            )
        return {L: Bursts.join(p) for L, p in sorted(parts.items())}, records

    train, train_records = side(train_runs)
    val, val_records = side(val_runs)
    if not train or set(train) != set(val):
        raise DataError(f"sizes with training runs {sorted(train)} and validation {sorted(val)}")
    shared = {r["seed"] for r in train_records} & {r["seed"] for r in val_records}
    if shared:
        raise DataError(f"seeds {sorted(shared)} serve both training and validation")
    sources = {
        "kind": "ising",
        "train": train_records,
        "val": val_records,
        "held_out": list(held_out) if held_out is not None else None,
        "refused_seeds": sorted(refused),
    }
    return Dataset(train, val, sources)


def reload(data_dir: Path, sources: dict[str, Any]) -> Dataset:
    """The dataset a training recorded in its manifest, refused if any run has changed since."""
    if sources.get("kind") != "ising":
        raise DataError(f"a {sources.get('kind')!r} dataset cannot be reloaded from its record")
    held = sources["held_out"]
    records = sources["train"] + sources["val"]
    return load(
        data_dir,
        [r["run_id"] for r in sources["train"]],
        [r["run_id"] for r in sources["val"]],
        (float(held[0]), float(held[1])) if held is not None else None,
        frozenset(sources["refused_seeds"]),
        {r["run_id"]: r["complete_sha256"] for r in records},
    )
