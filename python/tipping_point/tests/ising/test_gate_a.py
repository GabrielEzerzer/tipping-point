"""Gate A at L = 4: exact enumeration against a final snapshot of 4096 replicas per temperature.

Every mutant that changes what a chain samples must fail Gate A; the shared-draw mutant, which
leaves each chain's dynamics intact but couples the replicas, must fail the independence check.
An unmutated control run at the mutants' size must pass, so a failure is the mutant, not the size.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from tipping_point.ising import gates, kernel, run
from torch import Tensor

_HALF_SWEEP = kernel.half_sweep  # the real one, for mutants built on top of it


def both_colours_at_once(
    s: Tensor,
    black: Tensor,
    table: Tensor,
    generator: torch.Generator,
    tallies: Tensor | None = None,
) -> Tensor:
    """Every site updated from one neighbour field: the classic checkerboard mistake."""
    everything = torch.ones_like(black)
    return _HALF_SWEEP(s, everything, table, generator, None if tallies is None else tallies[:, 0])


def roll_on_dim_0(s: Tensor) -> Tensor:
    """Vertical neighbours taken from the adjacent replicas instead of the adjacent rows."""
    return s.roll(1, 0) + s.roll(-1, 0) + s.roll(1, 2) + s.roll(-1, 2)


def zero_padded_conv2d(s: Tensor) -> Tensor:
    """conv2d's default zero padding: free boundaries instead of periodic ones."""
    w = torch.tensor([[0.0, 1, 0], [1, 0, 1], [0, 1, 0]], device=s.device)[None, None]
    return F.conv2d(s[:, None].float(), w, padding=1)[:, 0].to(torch.int8)


def beta_for_t(temperature: Tensor) -> Tensor:
    """exp(-dE * T) where exp(-dE / T) belongs."""
    de = torch.arange(-8.0, 9.0, 4.0, dtype=torch.float64, device=temperature.device)
    return torch.exp(-de * temperature[:, None]).clamp(max=1.0).float()


def shared_draw(
    s: Tensor,
    colour: Tensor,
    table: Tensor,
    generator: torch.Generator,
    tallies: Tensor | None = None,
) -> Tensor:
    """One (L, L) uniform draw broadcast over the batch: every replica sees the same noise."""
    k = ((s * kernel.neighbour_sum(s) + 4) // 2).long()
    p = table.gather(1, k.flatten(1)).view(k.shape)
    u = torch.rand(s.shape[1:], generator=generator, device=s.device)
    flip = colour & (u < p)
    if tallies is not None:
        base = torch.arange(len(s))[:, None, None] * 6
        for row, counted in enumerate((colour, flip)):
            bins = (base + k.masked_fill(~counted, 5)).flatten()
            tallies[:, row] += torch.bincount(bins, minlength=6 * len(s)).view(-1, 6)[:, :5]
    return torch.where(flip, -s, s)


MUTANTS: dict[str, tuple[str, Callable[..., Any]]] = {
    "both colours at once": ("sweep", both_colours_at_once),
    "roll on dim 0": ("neighbour_sum", roll_on_dim_0),
    "zero-padded conv2d": ("neighbour_sum", zero_padded_conv2d),
    "beta for T": ("accept_table", beta_for_t),
}


def _small_validation(root: Path) -> list[gates.Check]:
    return run.validate(root, seed=2, replicas=1024, therm=200, sweeps=200, device="cpu")[1]


def _gate(checks: list[gates.Check], name: str) -> list[gates.Check]:
    found = [c for c in checks if c.gate == name]
    assert len(found) == 3, checks  # one per validation temperature
    return found


def test_real_kernel_passes_gate_a_at_4096_replicas(
    validation: tuple[Path, list[gates.Check]],
) -> None:
    """Catches any error in Hamiltonian, wrap, colouring or acceptance (pins all four together)."""
    for check in _gate(validation[1], "A"):
        assert check.passed, check


def test_unmutated_control_passes_at_the_mutant_size(tmp_path: Path) -> None:
    """Shows the mutant failures below come from the mutation, not from the smaller run."""
    checks = _small_validation(tmp_path)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]


@pytest.mark.parametrize("name", list(MUTANTS))
def test_mutant_fails_gate_a(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each mutant samples the wrong distribution; Gate A must say so at every temperature."""
    attribute, mutant = MUTANTS[name]
    monkeypatch.setattr(kernel, attribute, mutant)
    passed = [c for c in _gate(_small_validation(tmp_path), "A") if c.passed]
    assert not passed, f"{name} passed Gate A at {passed}"


def test_shared_draw_fails_the_independence_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Shared noise synchronises the replicas, so they stop being independent samples."""
    monkeypatch.setattr(kernel, "half_sweep", shared_draw)
    for check in _gate(_small_validation(tmp_path), "independence"):
        assert not check.passed, check
