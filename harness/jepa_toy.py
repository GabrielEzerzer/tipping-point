"""The planted-stripe toy for the JEPA harness (#27): bursts whose later frames are predictable from
an earlier one, with a known class and noise level per burst, so that learning, collapse and the
untrained null can be checked in seconds on a CPU.

A burst is 8 frames of a 16 x 16 torus: one of four stripe patterns of period 4 (two sites up,
two down), horizontal, vertical, diagonal or anti-diagonal (the class), with a random phase,
moving one site per frame across its stripes; each spin of each frame is flipped independently
with probability p, drawn per burst from [0, 0.2] (the noise level). Flips are unpredictable, so
they set a floor under the prediction error: with p up to 0.3 it sits at 0.43 however long the
run, with p up to 0.2 the curve matches the research pass's sketch (0.59 near step 90, 0.32 near
step 180). The motion is fixed relative to the stripes, so a D4 reflection would make its
direction ambiguous: the toy trains with rolls only, and D4 is checked on its own (test_lattice).
"""

from dataclasses import replace

import numpy as np
import torch
from tipping_point.ising import io
from tipping_point.jepa.data import Bursts, Dataset
from tipping_point.jepa.lattice import ModelSpec
from tipping_point.jepa.train import Config

L, FRAMES, CLASSES, P_MAX = 16, 8, 4, 0.2

# d = 32 as in the research pass's sketch; 32 bursts a step at a flat twice the production
# learning rate after warm-up (a cosine to 1e-6 stalls a run this short: 0.42 at step 200 instead
# of 0.31); a budget of steps, never of wall-clock time, so the lane is deterministic
CONFIG = Config(
    steps=150,
    model=ModelSpec(dim=32, pred_width=32),
    batch_sites=32 * L * L,
    lr=2e-3,
    lr_final=2e-3,
    warmup=20,
    d4=False,
    seed=27,
    checkpoint_every=75,
    monitor_every=25,
    monitor_frames=64,
    pca_frames=512,
)


def bursts(
    count: int, seed: int, p_max: float = P_MAX
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(spins (count, FRAMES, L, L) int8, class (count,), noise level (count,))."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, CLASSES, count)
    phase = rng.integers(0, 4, count)
    p = rng.uniform(0.0, p_max, count)
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    across = np.stack([i, j, i + j, i - j])[label]  # (count, L, L): position across the stripes
    t = np.arange(FRAMES).reshape(1, FRAMES, 1, 1)
    u = (across[:, None] + phase.reshape(-1, 1, 1, 1) + t) % 4
    spins = np.where(u < 2, 1, -1).astype(np.int8)
    flips = rng.random(spins.shape) < p.reshape(-1, 1, 1, 1)
    return np.where(flips, -spins, spins).astype(np.int8), label, p


def packed(spins: np.ndarray) -> torch.Tensor:
    """(count, FRAMES, L, L) -> (count, FRAMES, width) uint8, packed as the simulator packs."""
    flat = io.pack(spins.reshape(-1, L, L))
    return torch.from_numpy(flat.reshape(len(spins), FRAMES, -1))


def dataset(train: int = 2048, val: int = 256, seed: int = 0) -> Dataset:
    """Training and validation bursts; the noise level stands in for the temperature."""
    parts = {}
    for name, count, stream in (("train", train, seed), ("val", val, seed + 1)):
        spins, _, p = bursts(count, stream)
        parts[name] = {L: Bursts(L, packed(spins), p)}
    sources = {"kind": "toy", "train": train, "val": val, "seed": seed, "p_max": P_MAX}
    return Dataset(parts["train"], parts["val"], sources)


def config(**changes: object) -> Config:
    return replace(CONFIG, **changes)  # type: ignore[arg-type]
