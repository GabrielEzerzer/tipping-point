"""The training objective, generic over encoders (#30 reuses it with its own): what the predictor is
scored on, what keeps the representation from collapsing, and the schedules (learning rate, weight
decay, EMA momentum) as pure functions of the update index, so a resumed run sees exactly the
schedule a straight one sees.

Losses run in float32 with autocast off, whatever the forward pass ran in.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def errors(pred: Tensor, target: Tensor, hidden: Tensor) -> Tensor:
    """Per sample: the squared error per channel, averaged over the hidden cells. pred and target
    (B, dim, n, n), hidden (B, n, n) bool -> (B,). Against LayerNorm'd targets a prediction of zero
    scores exactly 1, so this is the normalised masked-target error."""
    with torch.autocast(pred.device.type, enabled=False):
        err = (pred.float() - target.float()).pow(2).mean(1)
        weight = hidden.to(err.dtype)
        return (err * weight).sum((1, 2)) / weight.sum((1, 2))


def prediction_loss(pred: Tensor, target: Tensor, hidden: Tensor) -> Tensor:
    """The squared error per channel over every hidden cell of the batch (MSE, as in the I-JEPA
    paper; the code's smooth L1 is not used, so that the training loss is the surprise)."""
    with torch.autocast(pred.device.type, enabled=False):
        err = (pred.float() - target.float()).pow(2).mean(1)
        weight = hidden.to(err.dtype)
        return (err * weight).sum() / weight.sum()


def vicreg(features: Tensor, keep: Tensor, gamma: float = 1.0) -> tuple[Tensor, Tensor]:
    """VICReg's variance hinge and covariance penalty over the tokens (cells) in `keep`, pooled
    across the batch: features (B, dim, n, n), keep (B, n, n) bool. Token level rather than per
    frame: the collapse a local encoder suffers first is spatial, and an L = 64 micro-batch holds
    too few frames for a dim x dim covariance. Dense masked sums, no boolean indexing."""
    with torch.autocast(features.device.type, enabled=False):
        x = features.float()
        weight = keep.to(x.dtype).unsqueeze(1)
        count = weight.sum()
        mean = (x * weight).sum((0, 2, 3)) / count
        centred = (x - mean.view(1, -1, 1, 1)) * weight
        cov = torch.einsum("bdij,beij->de", centred, centred) / (count - 1)
        variance = F.relu(gamma - torch.sqrt(cov.diagonal() + 1e-4)).mean()
        off_diagonal = cov - torch.diag(cov.diagonal())
        return variance, off_diagonal.pow(2).sum() / x.shape[1]


def learning_rate(update: int, total: int, base: float, final: float, warmup: int) -> float:
    """Linear warm-up to `base` over `warmup` updates, then a cosine down to `final`."""
    if update < warmup:
        return base * (update + 1) / warmup
    progress = (update - warmup) / max(1, total - warmup)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * min(1.0, progress)))


def cosine(update: int, total: int, start: float, end: float) -> float:
    """A cosine from `start` to `end` over the run (weight decay 0.04 -> 0.4, as in I-JEPA)."""
    progress = min(1.0, update / max(1, total))
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))


def momentum(update: int, total: int, start: float, end: float) -> float:
    """The EMA momentum, linear from `start` to `end` over the run (0.996 -> 1.0)."""
    return start + (end - start) * min(1.0, update / max(1, total))


@torch.no_grad()
def ema(target: nn.Module, source: nn.Module, m: float) -> None:
    """target <- m target + (1 - m) source, parameter by parameter."""
    for t, s in zip(target.parameters(), source.parameters(), strict=True):
        t.lerp_(s, 1.0 - m)
