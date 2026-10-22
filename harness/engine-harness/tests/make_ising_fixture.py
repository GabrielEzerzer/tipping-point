"""Writes the small Ising run the engine's API tests read, `engine/tests/fixtures/ising/<RUN>/`.

The chunks come from the simulator's own writer (`tipping_point.ising.io.write_chunk`), so the
engine is tested on Parquet exactly as pyarrow writes it: every column nullable, the replica a
uint16, the spins a fixed_size_binary(2) at L = 4. Two chunks per table, so a read crosses files.
The spins are seeded random states, not a simulation: the tests need real bytes, not physics.

Run from the repo root: `uv run python harness/engine-harness/tests/make_ising_fixture.py`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from tipping_point.ising import io

ROOT = Path(__file__).resolve().parents[3]
RUN = "20261022T000000Z-fixture"
L = 4
REPLICAS = 2
STEPS_PER_CHUNK = 3


def energy(spins: np.ndarray) -> np.ndarray:
    """-sum over the 2 L^2 periodic bonds of s s', per configuration."""
    right = spins * np.roll(spins, 1, axis=2)
    down = spins * np.roll(spins, 1, axis=1)
    return -(right.sum(axis=(1, 2)) + down.sum(axis=(1, 2))).astype(np.int32)


def main() -> None:
    run_dir = ROOT / "engine" / "tests" / "fixtures" / "ising" / RUN
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    io.write_json(
        run_dir / "manifest.json",
        {
            "format": 1,
            "domain": "ising",
            "run_id": RUN,
            "producer": {"name": "make_ising_fixture", "version": "0", "git_sha": None},
            "started_ns": 0,
            "key": "step",
            "tables": ["stats", "configs"],
            "params": {"L": L, "replicas": REPLICAS},
        },
    )
    rng = np.random.default_rng(7)
    for first in (0, STEPS_PER_CHUNK):
        steps = np.arange(first, first + STEPS_PER_CHUNK, dtype=np.int64)
        rows = len(steps) * REPLICAS
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(rows, L, L))
        keys = {
            "step": np.repeat(steps, REPLICAS),
            "replica": np.tile(np.arange(REPLICAS), len(steps)),
            "temperature": np.repeat(2.0 + 0.1 * steps, REPLICAS),
            "segment": np.zeros(rows, dtype=np.int32),
        }
        totals = {
            "magnetisation": spins.sum(axis=(1, 2)).astype(np.int32),
            "energy": energy(spins),
        }
        stats = {**keys, "phase": np.zeros(rows, dtype=np.int8), **totals}
        io.write_chunk(run_dir, "stats", L, stats)
        configs = {
            **keys,
            "burst": np.repeat(steps // STEPS_PER_CHUNK, REPLICAS).astype(np.int32),
            "frame": np.repeat(steps % STEPS_PER_CHUNK, REPLICAS).astype(np.int8),
            **totals,
            "spins": io.pack(spins),
        }
        io.write_chunk(run_dir, "configs", L, configs)
    io.write_json(
        run_dir / "complete.json",
        {"ended_ns": 0, "last_key": 2 * STEPS_PER_CHUNK - 1, "rows": {}, "clean": True},
    )


if __name__ == "__main__":
    main()
