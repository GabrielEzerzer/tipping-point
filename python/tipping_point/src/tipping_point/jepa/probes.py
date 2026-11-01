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

Each target is probed beside two control tasks (Hewitt and Liang 2019), fitted and scored the
same way:
  iid               a Gaussian draw per frame with the target's mean and variance on the fit
                    frames: what a probe scores on noise (R^2 near 0)
  shuffle_within_T  the target permuted among the frames of each plateau (one run, one
                    temperature): it keeps all that temperature implies and scrambles the
                    frame-level fluctuation
selectivity = R^2(target) - R^2(shuffle_within_T) is what the embedding holds of a frame beyond
its temperature: near 0 for features that only know T. Both controls are drawn once from the
probe seed, so the trained checkpoint and its twin are scored on the same draws.
"""

from __future__ import annotations

from collections.abc import Sequence
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
COLUMNS = ("raw", "iid", "shuffle_within_T")


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
    temperature: np.ndarray

    def take(self, rows: np.ndarray) -> Frames:
        targets = {k: v[rows] for k, v in self.targets.items()}
        packed = self.packed[torch.from_numpy(rows)]
        return Frames(self.L, packed, targets, self.replica[rows], self.temperature[rows])


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
    temperature = table["temperature"].to_numpy()
    _, plateau = np.unique(temperature, return_inverse=True)
    mean = np.bincount(plateau, abs_m) / np.bincount(plateau)
    l2var = area * (abs_m - mean[plateau]) ** 2
    rows = np.arange(len(abs_m))
    if limit is not None and limit < len(rows):
        rows = np.sort(np.random.default_rng(seed).choice(len(rows), limit, replace=False))
    packed = torch.from_numpy(packed_spins(table["spins"], run.L)[rows])
    targets = {"abs_m": abs_m[rows], "energy": energy[rows], "l2var": l2var[rows]}
    return Frames(run.L, packed, targets, table["replica"].to_numpy()[rows], temperature[rows])


Targets = dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]  # (target, column) -> (fit, test)


def controls(fit: Sequence[Frames], test: Sequence[Frames], seed: int = 0) -> Targets:
    """Every target in every column, raw and its two controls, on the fit and the test frames."""
    rng = np.random.default_rng([seed, 1])  # a stream apart from the frames' own draw (seed)

    def shuffled(f: Frames, target: str) -> np.ndarray:
        values = f.targets[target].copy()
        for T in np.unique(f.temperature):  # a Frames is one run, so a plateau is one T
            rows = np.flatnonzero(f.temperature == T)
            values[rows] = values[rng.permutation(rows)]
        return values

    out: Targets = {}
    for target in TARGETS:
        y_fit, y_test = (np.concatenate([f.targets[target] for f in s]) for s in (fit, test))
        mean, sd = y_fit.mean(), y_fit.std()
        out[target, "raw"] = (y_fit, y_test)
        out[target, "iid"] = (rng.normal(mean, sd, len(y_fit)), rng.normal(mean, sd, len(y_test)))
        out[target, "shuffle_within_T"] = (
            np.concatenate([shuffled(f, target) for f in fit]),
            np.concatenate([shuffled(f, target) for f in test]),
        )
    return out


def fitted(
    x_fit: np.ndarray, x_test: np.ndarray, targets: Targets, alpha: float = 1e-3
) -> dict[tuple[str, str], np.ndarray]:
    """For embeddings of the fit and test frames: each (target, column) probe's prediction of every
    test frame. One solve: the columns fit independently."""
    keys = [(t, c) for t in TARGETS for c in COLUMNS]
    y_fit = np.stack([targets[k][0] for k in keys], 1)
    pred = ridge(x_fit, y_fit, alpha).predict(x_test)
    return {k: pred[:, i] for i, k in enumerate(keys)}


def replica_sums(test: Sequence[Frames], y: np.ndarray, pred: np.ndarray) -> list[np.ndarray]:
    """What r2_of needs of a probe's test frames, per run and replica, so that resampling runs and
    replicas costs nothing per frame: for each run of `test`, (replicas, 4) frames, sum of y, sum of
    y^2 and sum of squared errors."""
    out, start = [], 0
    for f in test:
        stop = start + len(f.replica)
        part, miss = y[start:stop], (y - pred)[start:stop]
        size = int(f.replica.max()) + 1
        weights = (np.ones(len(part)), part, part * part, miss * miss)
        out.append(np.stack([np.bincount(f.replica, w, size) for w in weights], 1))
        start = stop
    return out


def r2_of(sums: np.ndarray) -> float:
    """r2 from replica_sums' rows (m, 4), however many of them, each as often as it was drawn."""
    n, y, yy, sse = sums.sum(0)
    return float(1.0 - sse / (yy - y * y / n))


def score(
    x_fit: np.ndarray, x_test: np.ndarray, targets: Targets, alpha: float = 1e-3
) -> list[dict[str, Any]]:
    """For embeddings of the fit and test frames: R^2 on test of a ridge probe per target and
    column, the raw row carrying the target's selectivity."""
    return scored(fitted(x_fit, x_test, targets, alpha), targets)


def scored(pred: dict[tuple[str, str], np.ndarray], targets: Targets) -> list[dict[str, Any]]:
    """score's rows, from the predictions `fitted` made."""
    r2s = {k: r2(targets[k][1], values) for k, values in pred.items()}
    rows = []
    for t in TARGETS:
        selectivity = r2s[t, "raw"] - r2s[t, "shuffle_within_T"]
        for c in COLUMNS:
            sel = selectivity if c == "raw" else None
            rows.append({"target": t, "column": c, "r2": r2s[t, c], "selectivity": sel})
    return rows


def probe(
    checkpoints: Sequence[Path],
    fit: Sequence[Path],
    test: Sequence[Path] = (),
    limit: int | None = 20_000,
    alpha: float = 1e-3,
    device: str | None = None,
    seed: int = 0,
    sites: int = 65_536,
) -> list[dict[str, Any]]:
    """R^2 of ridge probes per checkpoint (final.pt, its twin), target and column: fitted on the
    frames of the `fit` runs, scored on those of the `test` runs, or without them on the fit
    runs' odd replicas (fitted on the even ones), all of one L. One row per (checkpoint, target,
    column), naming the checkpoint by training run id, step and sha256, its role (trained, or
    twin at step 0) and L; the raw row carries selectivity."""
    dev = torch.device(device or default_device())
    fit_sets = [frames(open_run(Path(p)), limit, seed) for p in fit]
    test_sets = [frames(open_run(Path(p)), limit, seed) for p in test]
    if not test_sets:
        test_sets = [f.take(np.flatnonzero(f.replica % 2 == 1)) for f in fit_sets]
        fit_sets = [f.take(np.flatnonzero(f.replica % 2 == 0)) for f in fit_sets]
    sizes = sorted({f.L for f in [*fit_sets, *test_sets]})
    if len(sizes) != 1:
        raise DataError(f"probes read one L at a time, but these runs have L = {sizes}")
    targets = controls(fit_sets, test_sets, seed)
    y_fit, y_test = targets[TARGETS[0], "raw"]
    counts = {"n_fit": len(y_fit), "n_test": len(y_test)}
    rows = []
    for path in checkpoints:
        ckpt = load_model(path, dev)

        def embeddings(sets: list[Frames], model: Any = ckpt.model) -> np.ndarray:
            return np.concatenate([embed(model, f.packed, f.L, dev, sites).numpy() for f in sets])

        head = {
            "run_id": ckpt.run_id,
            "step": ckpt.step,
            "sha256": ckpt.sha256,
            "role": "twin" if ckpt.untrained else "trained",
            "L": sizes[0],
        }
        scored = score(embeddings(fit_sets), embeddings(test_sets), targets, alpha)
        rows += [head | row | counts for row in scored]
    return rows
