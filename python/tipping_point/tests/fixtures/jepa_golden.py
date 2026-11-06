"""The Ising training's golden (#30 M3): a short fixed-seed CPU training through every branch of the
Ising code path (two sizes, D4 and roll augmentation, monitors, checkpoints, the twin, the PCA
basis), reduced to exact values: every loss term and monitor statistic as float.hex, and sha256 of
the final and twin weights and bases. Recorded on the trainer before the domain split; the split
must reproduce it bit for bit, because the frozen Ising pre-registration and its Kaggle extraction
depend on this code's behaviour. Write it once, from the repo root, on the commit it pins:

  uv run --extra cpu python python/tipping_point/tests/fixtures/jepa_golden.py

shortcut: CPU bits differ across machines (BLAS and oneDNN pick kernels by instruction set), so the
golden holds for the environment it names and test_golden skips elsewhere; record another
environment's golden from the base commit (0e941a6) before checking a refactor there.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "harness"))  # as a script

import jepa_toy
from tipping_point.ising import io
from tipping_point.jepa import data, train
from tipping_point.jepa.lattice import unpack

OUT = Path(__file__).with_name("jepa_ising_golden.json")
CONFIG = jepa_toy.config(
    steps=12, checkpoint_every=4, monitor_every=4, pca_frames=128, d4=True, roll=True, seed=30
)


def environment() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }


def dataset() -> data.Dataset:
    """The planted-stripe toy at L = 16 and, tiled 2 x 2 (stripes of period 4 tile), at L = 32."""
    toy = jepa_toy.dataset(256, 32)

    def tiled(bursts: data.Bursts) -> data.Bursts:
        spins = unpack(bursts.packed, bursts.L).numpy()
        big = np.tile(spins, (1, 1, 2, 2))
        flat = io.pack(big.reshape(-1, 32, 32))
        packed = torch.from_numpy(flat.reshape(len(big), big.shape[1], -1))
        return data.Bursts(32, packed, bursts.temperature)

    train_, val = toy.train[16], toy.val[16]
    return data.Dataset(
        {16: train_, 32: tiled(train_)}, {16: val, 32: tiled(val)}, {"kind": "toy-golden"}
    )


def _hex(value: Any) -> Any:
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, list):
        return [_hex(v) for v in value]
    return value


def _table(run_dir: Path, name: str) -> list[dict[str, Any]]:
    table = pa.concat_tables(pq.read_table(p) for p in io.chunks(run_dir, name))
    return [{k: _hex(v) for k, v in row.items()} for row in table.to_pylist()]


def _weights(path: Path) -> str:
    """sha256 over the model's tensors in key order and the PCA basis: the weights, not the file
    (which names its run and so differs between runs)."""
    ckpt = train.load_model(path)
    digest = hashlib.sha256()
    for key, value in ckpt.model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    assert ckpt.pca is not None
    for value in ckpt.pca.record().values():
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def compute(root: Path) -> dict[str, Any]:
    run_dir = train.start(CONFIG, dataset(), root, "golden", "cpu")
    manifest = io.read_json(run_dir / "manifest.json")
    done = io.read_json(run_dir / "complete.json")
    return {
        "train": _table(run_dir, "train"),
        "monitor": _table(run_dir, "monitor"),
        "final": _weights(run_dir / train.FINAL),
        "twin": _weights(run_dir / train.TWIN),
        "params": manifest["params"],
        "rows": done["rows"],
        "collapse": done["collapse"],
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        record = {"environment": environment(), "golden": compute(Path(tmp))}
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
