"""What a frozen model reads out of frames, shared by training (its PCA bases), the latents table
and the probes.

Readouts canonicalise each frame on its own, so every one of them is exactly even under a global
spin flip, and run in float32 with autocast off and no gradient, on any device.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from . import objective
from .lattice import Jepa, canonical, channel_norm, unpack

if TYPE_CHECKING:
    from .data import Bursts


def frames_per_batch(L: int, sites: int) -> int:
    return max(1, sites // (L * L))


def batches(packed: Tensor, L: int, sites: int) -> Iterator[Tensor]:
    """Packed frames (m, width) in batches of about `sites` sites."""
    step = frames_per_batch(L, sites)
    for start in range(0, len(packed), step):
        yield packed[start : start + step]


@torch.no_grad()
def readouts(
    model: Jepa, spins: Tensor, hidden: Tensor | None = None, keep: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    """For canonical frames (B, L, L): the frame embedding (B, dim) and, given evaluation masks
    hidden and keep (K, n, n), the surprise (B,): the error of the delta = 0 prediction at the
    hidden cells, averaged over the K masks. One pass of the target encoder serves both."""
    with torch.autocast(spins.device.type, enabled=False):
        features = model.target(spins)
        embedding = features.mean((-2, -1))
        if hidden is None or keep is None:
            return embedding, None
        targets = channel_norm(features)
        B, n = len(spins), hidden.shape[-1]
        delta = torch.zeros(B, dtype=torch.long, device=spins.device)
        total = torch.zeros(B, device=spins.device)
        for k in range(len(hidden)):
            sees = keep[k].expand(B, n, n)
            _, pred = model.predict(spins, sees, delta)
            total += objective.errors(pred, targets, hidden[k].expand(B, n, n))
        return embedding, total / len(hidden)


def embed(model: Jepa, packed: Tensor, L: int, device: torch.device, sites: int) -> Tensor:
    """Frame embeddings of packed frames (m, width): (m, dim) float64 on the CPU."""
    out = []
    for chunk in batches(packed, L, sites):
        spins = canonical(unpack(chunk.to(device), L), 2)
        out.append(readouts(model, spins)[0].to("cpu", torch.float64))
    return torch.cat(out)


@dataclass(frozen=True)
class Pca:
    """A frozen basis for the frame embedding: pc_k = (embedding - mean) . components[k - 1]."""

    mean: Tensor  # (dim,) float64
    components: Tensor  # (k, dim) float64, orthonormal rows, largest |loading| of each positive
    variance: Tensor  # (k,) variance along each component on the fitting frames
    total: float  # total variance of the fitting frames' embeddings
    frames: int

    def project(self, embedding: Tensor) -> Tensor:
        return (embedding.to(torch.float64) - self.mean) @ self.components.T

    def record(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "components": self.components,
            "variance": self.variance,
            "total": self.total,
            "frames": self.frames,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Pca:
        return cls(
            record["mean"],
            record["components"],
            record["variance"],
            float(record["total"]),
            int(record["frames"]),
        )


def fit_pca(
    model: Jepa,
    data: Mapping[int, Bursts],
    per_size: int,
    k: int,
    gen: torch.Generator,
    device: torch.device,
    sites: int,
) -> Pca:
    """A PCA basis fitted on `per_size` training frames of every size, drawn by `gen` (CPU), so
    the same basis serves every L and finite-size scaling compares one function across sizes."""
    rows = []
    for L in sorted(data):
        bursts = data[L]
        which = torch.randint(len(bursts), (per_size,), generator=gen)
        frame = torch.randint(bursts.frames, (per_size,), generator=gen)
        rows.append(embed(model, bursts.packed[which, frame], L, device, sites))
    x = torch.cat(rows)
    mean = x.mean(0)
    _, s, vh = torch.linalg.svd(x - mean, full_matrices=False)
    components = vh[:k]
    peak = components.abs().argmax(1, keepdim=True)
    components = components * components.gather(1, peak).sign()
    variance = s[:k].pow(2) / (len(x) - 1)
    return Pca(mean, components, variance, float(s.pow(2).sum() / (len(x) - 1)), len(x))
