"""Linear probes on frozen frame embeddings (#19, #28): how much of |m|, the energy and the
susceptibility proxy a checkpoint's embedding holds linearly, beside its untrained twin.

A local encoder's pooled readout is a translation average of local operators, so random weights
already read |m| and e well: on the #27 toy, probes did not separate learning from random
features, prediction error did. Every number is therefore reported for the trained checkpoint and
its step-0 twin side by side; what training added is the difference.

Targets per frame, from the configs table: abs_m = |M| / L^2, energy = E / L^2, and l2var =
L^2 (|m| - <|m|>_T)^2, whose mean over a plateau is L^2 Var(|m|), the susceptibility proxy (so
probes read staircase and fixed runs, where T is held). A probe is a ridge regression on
standardised embeddings, fitted on one set of frames and scored by R^2 on another: other runs
(fit on the validation seed, test on scored staircases), or by default the odd replicas of the
fitting runs, chains independent of the even ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..ews.ising import Run, open_run
from ..ising import io
from .data import DataError, packed_spins
from .readout import embed
from .train import default_device, load_model

TARGETS = ("abs_m", "energy", "l2var")


@dataclass(frozen=True)
class Ridge:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale) @ self.weights + self.intercept


def ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> Ridge:
    """Ridge regression of y (n,) or (n, k) on x (n, d), features standardised on x, penalty
    alpha * n (so alpha is per sample)."""
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    intercept = y.mean(0)
    gram = z.T @ z + alpha * len(z) * np.eye(z.shape[1])
    return Ridge(mean, scale, np.linalg.solve(gram, z.T @ (y - intercept)), intercept)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    """Share of rows whose highest score is their label: a probe onto one-hot classes."""
    return float(np.mean(scores.argmax(1) == labels))


@dataclass(frozen=True)
class Frames:
    """Saved frames of one run, packed, with their probe targets."""

    L: int
    packed: torch.Tensor  # (m, ceil(L*L/8)) uint8
    targets: dict[str, np.ndarray]
    replica: np.ndarray

    def take(self, rows: np.ndarray) -> Frames:
        targets = {k: v[rows] for k, v in self.targets.items()}
        return Frames(self.L, self.packed[torch.from_numpy(rows)], targets, self.replica[rows])


def frames(run: Run, limit: int | None = None, seed: int = 0) -> Frames:
    """A run's saved frames with their targets; with `limit`, that many drawn without
    replacement (l2var's plateau means use every frame)."""
    if run.kind == "ramp":
        raise DataError(f"{run.run_id} is a ramp: probes need temperatures held fixed")
    paths = io.chunks(run.path, "configs")
    if not paths:
        raise DataError(f"{run.run_id}: no configs table, so no saved frames")
    names = ["replica", "temperature", "magnetisation", "energy", "spins"]
    table = pa.concat_tables(pq.read_table(p, columns=names) for p in paths)
    area = float(run.L**2)
    abs_m = np.abs(table["magnetisation"].to_numpy()) / area
    energy = table["energy"].to_numpy() / area
    _, plateau = np.unique(table["temperature"].to_numpy(), return_inverse=True)
    mean = np.bincount(plateau, abs_m) / np.bincount(plateau)
    l2var = area * (abs_m - mean[plateau]) ** 2
    rows = np.arange(len(abs_m))
    if limit is not None and limit < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(len(rows), limit, replace=False))
    packed = torch.from_numpy(packed_spins(table["spins"], run.L)[rows])
    targets = {"abs_m": abs_m[rows], "energy": energy[rows], "l2var": l2var[rows]}
    return Frames(run.L, packed, targets, table["replica"].to_numpy()[rows])


def probe(
    checkpoints: Mapping[str, Path],
    fit: Sequence[Path],
    test: Sequence[Path] = (),
    limit: int | None = 20_000,
    alpha: float = 1e-3,
    device: str | None = None,
    seed: int = 0,
    sites: int = 65_536,
) -> list[dict[str, Any]]:
    """R^2 of a ridge probe per checkpoint and target: fitted on the frames of the `fit` runs,
    scored on those of the `test` runs, or without them on the fit runs' odd replicas (fitted on
    the even ones). One row per (checkpoint, target)."""
    dev = torch.device(device or default_device())
    fit_sets = [frames(open_run(Path(p)), limit, seed) for p in fit]
    test_sets = [frames(open_run(Path(p)), limit, seed) for p in test]
    if not test_sets:
        test_sets = [f.take(np.flatnonzero(f.replica % 2 == 1)) for f in fit_sets]
        fit_sets = [f.take(np.flatnonzero(f.replica % 2 == 0)) for f in fit_sets]
    rows = []
    for name, path in checkpoints.items():
        ckpt = load_model(path, dev)

        def embeddings(sets: list[Frames], model: Any = ckpt.model) -> np.ndarray:
            return np.concatenate([embed(model, f.packed, f.L, dev, sites).numpy() for f in sets])

        x_fit, x_test = embeddings(fit_sets), embeddings(test_sets)
        for target in TARGETS:
            y_fit = np.concatenate([f.targets[target] for f in fit_sets])
            y_test = np.concatenate([f.targets[target] for f in test_sets])
            score = r2(y_test, ridge(x_fit, y_fit, alpha).predict(x_test))
            rows.append(
                {
                    "checkpoint": name,
                    "run_id": ckpt.run_id,
                    "step": ckpt.step,
                    "target": target,
                    "r2": score,
                    "n_fit": len(y_fit),
                    "n_test": len(y_test),
                }
            )
    return rows
