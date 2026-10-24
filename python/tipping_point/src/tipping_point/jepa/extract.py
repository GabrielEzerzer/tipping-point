"""The latents table: a frozen checkpoint's readouts of every saved frame of a simulator run,
stored beside the frames they come from. It is #28's input: the scorer reads it as it reads
configs, never touching spins or the model again.

<data dir>/ising/<run_id>/latents/<jepa run_id>-step<step:07d>/
  manifest.json              written once at the start: the checkpoint and the dataset run, hashed
  <first_step:020>.parquet   one chunk per configs chunk, the same rows in the same order
  complete.json              rows and every file's sha256; absent = running or crashed

A dataset run holds one folder per checkpoint (model seeds, their untrained twins, ablations).
Rows follow #22, one per configs row, keyed exactly like configs, in its (step, replica) order:
  step i64 | replica u16 | temperature f64 | segment i32 | burst i32 | frame i8 |
  surprise f32 | pc_1 f32 ... pc_8 f32
surprise  delta = 0: the squared error per channel of the predicted LayerNorm'd target features
          at the hidden cells, averaged over the checkpoint's two fixed evaluation masks
pc_k      the frame embedding (target-encoder features averaged over cells) on the checkpoint's
          PCA basis, fitted on training frames and frozen with it
Frames are canonicalised one by one, so every column is even under a global spin flip. Values are
computed and stored in float32 on any device. An interrupted extraction resumes, keeping the
chunks it finished (each appears whole or not at all).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .. import __version__
from ..ews.ising import Run, open_run
from ..ising import io
from .data import DataError, packed_spins
from .lattice import canonical, unpack
from .readout import batches, readouts
from .train import (
    Checkpoint,
    default_device,
    deterministic,
    evaluation_masks,
    gpu_name,
    load_model,
    seeds,
    sha256,
)

LATENTS = "latents"
KEY = ("step", "replica", "temperature", "segment", "burst", "frame")


def schema(L: int, components: int) -> pa.Schema:
    """The latents table: configs' key columns with configs' own types, then the readouts."""
    configs = io.schema("configs", L)
    readout = [("surprise", pa.float32())] + [
        (f"pc_{k}", pa.float32()) for k in range(1, components + 1)
    ]
    return pa.schema([configs.field(name) for name in KEY] + readout)


def folder(dataset_run: Path, run_id: str, step: int) -> Path:
    return Path(dataset_run) / LATENTS / f"{run_id}-step{step:07d}"


def _refusal(checkpoint: Path, digest: str) -> str | None:
    """Why a trained checkpoint is refused, read from the complete.json of the training it sits
    in: there is none (running or crashed), it does not list this file, or it records a collapse
    (on any step, the last included) or an early stop. None when the training ran to its end
    without collapsing."""
    done = checkpoint.parent / "complete.json"
    if not done.is_file():
        return "its training has no complete.json (running or crashed)"
    record = io.read_json(done)
    if record.get("files", {}).get(checkpoint.name) != digest:
        return f"{done} does not list it with its sha256"
    collapse = record.get("collapse") or {}
    if collapse.get("collapsed"):
        fired = "; ".join((collapse.get("first_fired") or {}).get("fired", []))
        return f"{done} records a collapse at step {collapse['collapsed_at']} ({fired})"
    if record.get("stopped_early"):
        return f"{done} records stopped_early at step {record.get('last_key')}"
    return None


def _manifest(out: Path, run: Run, ckpt: Checkpoint, device: torch.device) -> dict[str, Any]:
    assert ckpt.pca is not None
    root = run.path.parents[1]
    try:
        where = ckpt.path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        where = str(ckpt.path)
    hidden, _ = evaluation_masks(ckpt.config, run.L)
    return {
        "format": 1,
        "domain": "ising",
        "run_id": out.name,
        "producer": {"name": "tipping_point.jepa", "version": __version__, **io.code_version()},
        "started_ns": time.time_ns(),
        "key": "step",
        "tables": [LATENTS],
        "params": {
            "checkpoint": {
                "path": where,
                "sha256": ckpt.sha256,
                "run_id": ckpt.run_id,
                "step": ckpt.step,
                "untrained": ckpt.untrained,
            },
            "dataset": {
                "run_id": run.run_id,
                "complete_sha256": run.complete_sha256,
                "L": run.L,
                "kind": run.kind,
                "seed": run.seed,
            },
            "pca": {
                "components": len(ckpt.pca.components),
                "variance": ckpt.pca.variance.tolist(),
                "total": ckpt.pca.total,
                "frames": ckpt.pca.frames,
            },
            "evaluation_masks": {
                "seed": seeds(ckpt.config.seed)["eval_masks"],
                "count": ckpt.config.eval_masks,
                "hidden_share": hidden.flatten(1).float().mean(1).tolist(),
            },
            "readout": {"canonical": "per frame", "precision": "float32", "delta": 0},
        },
        "env": {"device": device.type, "gpu": gpu_name(device), "torch": torch.__version__},
    }


def extract(
    checkpoint: Path, dataset_run: Path, device: str | None = None, sites: int = 65_536
) -> Path:
    """Write a checkpoint's latents table for a complete simulator run; returns its folder. A
    trained checkpoint needs its training's complete.json to vouch for it (#89); the untrained
    twin is the null whatever became of its training."""
    dev = torch.device(device or default_device())
    deterministic(dev)
    run = open_run(Path(dataset_run))
    ckpt = load_model(checkpoint, dev)
    pca = ckpt.pca
    if pca is None:
        raise DataError(f"{checkpoint} holds no PCA basis: extract from final.pt or the twin")
    why = None if ckpt.untrained else _refusal(Path(checkpoint), ckpt.sha256)
    if why is not None:
        raise DataError(
            f"{checkpoint} is step {ckpt.step} of {ckpt.config.steps}, but {why}. Latents come "
            "from a training that ran to its end without collapsing, or from its untrained twin"
        )
    chunks = io.chunks(run.path, "configs")
    if not chunks:
        raise DataError(f"{run.run_id}: no configs table, so no saved frames")
    out = folder(run.path, ckpt.run_id, ckpt.step)
    if (out / "complete.json").exists():
        raise DataError(f"{out} is complete: a latents table is written once")
    if (out / "manifest.json").exists():  # interrupted: same checkpoint, same data, or nothing
        prior = io.read_json(out / "manifest.json")["params"]
        same = (
            prior["checkpoint"]["sha256"] == ckpt.sha256
            and prior["dataset"]["complete_sha256"] == run.complete_sha256
        )
        if not same:
            raise DataError(f"{out} was begun from another checkpoint or dataset")
        for path in out.glob("*.tmp"):
            path.unlink()
    else:  # chunks only ever follow the manifest, so a folder without one holds nothing yet
        out.mkdir(parents=True, exist_ok=True)
        io.write_json(out / "manifest.json", _manifest(out, run, ckpt, dev))
    hidden, keep = (m.to(dev) for m in evaluation_masks(ckpt.config, run.L))
    table_schema = schema(run.L, len(pca.components))
    for path in chunks:
        if (out / path.name).exists():
            continue
        table = pq.read_table(path, columns=[*KEY, "spins"])
        packed = torch.from_numpy(packed_spins(table["spins"], run.L).copy())
        surprise, pcs = [], []
        for chunk in batches(packed, run.L, sites):
            spins = canonical(unpack(chunk.to(dev), run.L), 2)
            embedding, score = readouts(ckpt.model, spins, hidden, keep)
            assert score is not None
            pcs.append(pca.project(embedding.cpu()).to(torch.float32))
            surprise.append(score.cpu())
        values = np.concatenate([torch.cat(surprise).numpy()[:, None], torch.cat(pcs).numpy()], 1)
        columns = [table[name] for name in KEY] + [
            pa.array(values[:, k], type=pa.float32()) for k in range(values.shape[1])
        ]
        latents = pa.Table.from_arrays(columns, schema=table_schema)
        io.write_file(
            out / path.name, lambda f, t=latents: pq.write_table(t, f, compression="zstd")
        )
    _complete(out)
    return out


def _complete(out: Path) -> None:
    files = {p.name: sha256(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}
    rows = sum(pq.read_metadata(out / name).num_rows for name in files if name.endswith(".parquet"))
    record = {"ended_ns": time.time_ns(), "rows": {LATENTS: rows}, "clean": True, "files": files}
    io.write_json(out / "complete.json", record)


def available(dataset_run: Path) -> list[str]:
    """The finished latents tables of a simulator run, by folder name."""
    base = Path(dataset_run) / LATENTS
    return sorted(p.parent.name for p in base.glob("*/complete.json"))


def read(dataset_run: Path, name: str) -> pa.Table:
    """A finished latents table, rows in configs order; refused while it is incomplete."""
    out = Path(dataset_run) / LATENTS / name
    if not (out / "complete.json").is_file():
        raise DataError(f"{out}: no complete.json (running, crashed or absent)")
    return pa.concat_tables(pq.read_table(p) for p in sorted(out.glob("*.parquet")))
