"""Collapse monitors and the detector, generic over encoders.

Effective rank is scale-free: a representation can shrink to almost nothing while its last few
directions stay evenly spread. On the #27 toy, a mutant without stop-gradient or VICReg collapsed
spatially first (token std 0.87 -> 0.04, pooled std -> 0.006) while its effective rank stayed at
11.6, so the detector fires on absolute floors of feature spread as well as on effective rank.

Statistics are taken on whole frames of a fixed monitor batch, in float64 on the CPU:
  token_std       per channel, the spread over every cell of every frame; mean over channels
  dim_std_min     the same, for the narrowest channel (dimensional collapse)
  spatial_std     per channel and frame, the spread over the frame's cells; mean of both
  pooled_std      per channel, the spread of the frame embedding (mean over cells) over frames
  effective_rank  exp(entropy of the normalised singular values) of the centred token matrix
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor

FLOORED = ("spatial_std", "pooled_std", "effective_rank")


@dataclass(frozen=True)
class Floors:
    """Absolute floors, below which the representation counts as collapsing. Calibrated on the
    planted-stripe toy (harness/jepa_toy.py); a tuning knob, recorded with every run, to be
    checked on the validation seed before production runs."""

    spatial_std: float = 0.1
    pooled_std: float = 0.02
    effective_rank: float = 2.0

    def broken(self, stats: dict[str, Any]) -> list[str]:
        """The floored statistics of one evaluation that fell below their floor."""
        floors = asdict(self)
        return [name for name in FLOORED if stats[name] < floors[name]]


def effective_rank(tokens: Tensor) -> float:
    """Roy & Vetterli's effective rank of tokens (N, dim), centred."""
    s = torch.linalg.svdvals(tokens - tokens.mean(0))
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def spread(features: Tensor) -> dict[str, Any]:
    """The monitor statistics of features (B, dim, n, n) of whole frames."""
    x = features.detach().to("cpu", torch.float64)
    tokens = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    dim_std = tokens.std(0)
    return {
        "token_std": float(dim_std.mean()),
        "dim_std_min": float(dim_std.min()),
        "dim_std": dim_std.tolist(),
        "spatial_std": float(x.flatten(2).std(2).mean()),
        "pooled_std": float(x.mean((2, 3)).std(0).mean()),
        "effective_rank": effective_rank(tokens),
    }


@dataclass
class Detector:
    """Fires when any size's monitor statistics fall below a floor; declares the run collapsed
    after `patience` evaluations in a row fired. Its state travels in checkpoints."""

    floors: Floors
    patience: int = 3
    strikes: int = 0
    first: dict[str, Any] | None = None  # the first evaluation that fired: step and floors
    collapsed_at: int | None = None
    history: list[int] = field(default_factory=list)  # steps whose evaluation fired

    def check(self, step: int, stats: dict[int, dict[str, Any]]) -> list[str]:
        """One evaluation, statistics per L; returns the floors it broke."""
        floors = asdict(self.floors)
        fired = [
            f"L{L} {name} {s[name]:.4g} < {floors[name]:g}"
            for L, s in sorted(stats.items())
            for name in self.floors.broken(s)
        ]
        if fired:
            self.strikes += 1
            self.history.append(step)
            if self.first is None:
                self.first = {"step": step, "fired": fired}
            if self.strikes >= self.patience and self.collapsed_at is None:
                self.collapsed_at = step
        else:
            self.strikes = 0
        return fired

    @property
    def collapsed(self) -> bool:
        return self.collapsed_at is not None

    def state(self) -> dict[str, Any]:
        return {
            "strikes": self.strikes,
            "first": self.first,
            "collapsed_at": self.collapsed_at,
            "history": list(self.history),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.strikes, self.first = int(state["strikes"]), state["first"]
        self.collapsed_at, self.history = state["collapsed_at"], list(state["history"])

    def verdict(self) -> dict[str, Any]:
        return {
            "collapsed": self.collapsed,
            "collapsed_at": self.collapsed_at,
            "first_fired": self.first,
            "evaluations_fired": len(self.history),
            "floors": asdict(self.floors),
            "patience": self.patience,
        }
