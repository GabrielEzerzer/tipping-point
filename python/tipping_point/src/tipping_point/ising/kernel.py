"""Checkerboard Metropolis on a batch of periodic L x L Ising lattices (J = k_B = 1).

Spins are int8 +-1 in a (B, L, L) tensor, one lattice per replica. Sites with i + j even are
black, the rest white. A sweep updates every black site at once, then every white site: sites of
one colour share no bond, so a half-sweep is exactly a set of independent single-site Metropolis
updates. No function here changes the spins it is given; every update returns a new tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor


def checkerboard(L: int, device: torch.device | str) -> Tensor:
    """(L, L) bool mask of the black sites. L must be even for the colouring to wrap around."""
    if L % 2:
        raise ValueError(f"L must be even, got {L}")
    i = torch.arange(L, device=device)
    return (i[:, None] + i[None, :]) % 2 == 0


def accept_table(temperature: Tensor) -> Tensor:
    """(B,) float64 temperatures -> (B, 5) float32 acceptance for dE = -8, -4, 0, 4, 8:
    min(1, e^(-dE/T)), except 1/2 at dE = 0.

    Accepting every dE = 0 move traps a lattice whose sites all have two up and two down
    neighbours: each half-sweep flips every site of its colour, so it oscillates x <-> -x for
    ever (36 such states at L = 4, cut off from the rest). Both directions of a dE = 0 move get
    the same 1/2, so detailed balance and the stationary distribution are unchanged.
    """
    delta_e = torch.arange(-8.0, 9.0, 4.0, dtype=torch.float64, device=temperature.device)
    table = torch.exp(-delta_e / temperature[:, None]).clamp(max=1.0)
    table[:, 2] = 0.5  # dE = 0
    return table.float()


def neighbour_sum(s: Tensor) -> Tensor:
    """Sum of the four nearest neighbours; roll's wrap-around is the periodic boundary."""
    return s.roll(1, 1) + s.roll(-1, 1) + s.roll(1, 2) + s.roll(-1, 2)


def half_sweep(
    s: Tensor,
    colour: Tensor,
    table: Tensor,
    generator: torch.Generator,
    tallies: Tensor | None = None,
) -> Tensor:
    """Metropolis-update every site of one colour.

    `tallies` (B, 2, 5), if given, accumulates [attempted, accepted] moves per dE (Gate C).
    """
    # dE = 2*s*h = 4k - 8, so k = (s*h + 4) / 2 in 0..4 indexes the table
    k = ((s * neighbour_sum(s) + 4) // 2).long()
    p = table.gather(1, k.flatten(1)).view(k.shape)
    u = torch.rand(s.shape, generator=generator, device=s.device)  # fresh per replica and site
    flip = colour & (u < p)
    if tallies is not None:  # bin 6r + k counts replica r's sites with dE index k; 5 = not counted
        base = torch.arange(len(s), device=s.device)[:, None, None] * 6
        for row, counted in enumerate((colour, flip)):
            bins = (base + k.masked_fill(~counted, 5)).flatten()
            tallies[:, row] += torch.bincount(bins, minlength=6 * len(s)).view(-1, 6)[:, :5]
    return torch.where(flip, -s, s)


def sweep(
    s: Tensor,
    black: Tensor,
    table: Tensor,
    generator: torch.Generator,
    tallies: Tensor | None = None,
) -> Tensor:
    """Black half-sweep, then white, which sees the updated black spins. tallies: (B, 2, 2, 5)."""
    s = half_sweep(s, black, table, generator, None if tallies is None else tallies[:, 0])
    return half_sweep(s, ~black, table, generator, None if tallies is None else tallies[:, 1])


def totals(s: Tensor) -> tuple[Tensor, Tensor]:
    """Exact int32 magnetisation M = sum(s) and energy E = -sum over bonds of s*s', per replica."""
    M = s.sum(dim=(1, 2), dtype=torch.int32)
    E = -(s * (s.roll(1, 1) + s.roll(1, 2))).sum(dim=(1, 2), dtype=torch.int32)
    return M, E
