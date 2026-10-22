"""Kernel pieces against explicit loops: colouring, neighbour sums, table, totals, tallies."""

import math

import numpy as np
import pytest
import torch
from tipping_point.ising import kernel


def _random_spins(B: int, L: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (B, L, L), generator=g, dtype=torch.int8) * 2 - 1


def test_checkerboard_colours_alternate_and_need_even_l() -> None:
    """Catches a colouring where two neighbours share a colour (then a half-sweep is not exact)."""
    black = kernel.checkerboard(6, "cpu")
    assert black[0, 0] and not black[0, 1] and not black[1, 0] and black[1, 1]
    assert int(black.sum()) == 18
    assert not (black & black.roll(1, 0)).any() and not (black & black.roll(1, 1)).any()
    with pytest.raises(ValueError):
        kernel.checkerboard(5, "cpu")


def test_neighbour_sum_matches_an_explicit_periodic_loop() -> None:
    """Catches rolling the wrong dimension or losing the periodic wrap."""
    s = _random_spins(3, 6)
    h = kernel.neighbour_sum(s)
    for b in range(3):
        for i in range(6):
            for j in range(6):
                expected = (
                    s[b, (i + 1) % 6, j] + s[b, i - 1, j] + s[b, i, (j + 1) % 6] + s[b, i, j - 1]
                )
                assert h[b, i, j] == expected


def test_accept_table_is_min_one_exp_per_replica() -> None:
    """Catches beta-for-T, a missing clamp, or one temperature applied to every replica."""
    table = kernel.accept_table(torch.tensor([2.0, 4.0], dtype=torch.float64))
    assert table.dtype == torch.float32 and table.shape == (2, 5)
    for row, T in zip(table, (2.0, 4.0), strict=True):
        expected = [min(1.0, math.exp(-dE / T)) for dE in (-8, -4, 0, 4, 8)]
        assert row.tolist() == pytest.approx(expected, rel=1e-6)


def test_totals_match_an_explicit_bond_count() -> None:
    """Catches double-counted bonds, a sign error, or summing in int8 (overflow at large L)."""
    s = _random_spins(4, 16)
    M, E = kernel.totals(s)
    assert M.dtype == torch.int32 and E.dtype == torch.int32
    a = s.numpy().astype(np.int64)
    assert M.tolist() == a.sum(axis=(1, 2)).tolist()
    bonds = a * np.roll(a, 1, 1) + a * np.roll(a, 1, 2)  # each bond once: down and right
    assert E.tolist() == (-bonds.sum(axis=(1, 2))).tolist()
    up = torch.ones((1, 128, 128), dtype=torch.int8)
    assert kernel.totals(up)[0].item() == 128 * 128  # 16384 would wrap in int8 or int16
    assert kernel.totals(up)[1].item() == -2 * 128 * 128


def test_half_sweep_touches_one_colour_and_always_takes_free_moves() -> None:
    """Catches updating both colours, or rejecting a move that does not raise the energy."""
    s = _random_spins(64, 8)
    black = kernel.checkerboard(8, "cpu")
    table = kernel.accept_table(torch.full((64,), 2.0, dtype=torch.float64))
    new = kernel.half_sweep(s, black, table, torch.Generator().manual_seed(1))
    assert torch.equal(new[:, ~black], s[:, ~black])
    free = black & (s * kernel.neighbour_sum(s) <= 0)  # dE = 2 s h <= 0
    assert torch.equal(new[free], -s[free])
    assert set(new.unique().tolist()) == {-1, 1} and new.dtype == torch.int8


def test_sweep_is_a_function_of_the_generator_state() -> None:
    """Catches hidden global randomness: same seed, same result; different seed, different."""
    s = _random_spins(8, 8)
    black = kernel.checkerboard(8, "cpu")
    table = kernel.accept_table(torch.full((8,), 2.5, dtype=torch.float64))

    def run(seed: int) -> torch.Tensor:
        g, x = torch.Generator().manual_seed(seed), s
        for _ in range(20):
            x = kernel.sweep(x, black, table, g)
        return x

    assert torch.equal(run(5), run(5))
    assert not torch.equal(run(5), run(6))


def test_tallies_count_every_attempt_and_every_flip() -> None:
    """Catches tallies that miss sites, mix colours, or disagree with the flips actually made."""
    B, L = 16, 8
    s = _random_spins(B, L)
    black = kernel.checkerboard(L, "cpu")
    table = kernel.accept_table(torch.full((B,), 2.3, dtype=torch.float64))
    tallies = torch.zeros((B, 2, 2, 5), dtype=torch.int64)
    g = torch.Generator().manual_seed(2)
    after_black = kernel.half_sweep(s, black, table, g, tallies[:, 0])
    after_white = kernel.half_sweep(after_black, ~black, table, g, tallies[:, 1])
    assert (tallies[:, :, 0].sum(-1) == L * L // 2).all()  # every site of a colour attempted once
    flipped_black = (after_black != s).sum((1, 2))
    flipped_white = (after_white != after_black).sum((1, 2))
    assert torch.equal(tallies[:, 0, 1].sum(-1), flipped_black)
    assert torch.equal(tallies[:, 1, 1].sum(-1), flipped_white)
    assert (tallies[:, :, 1] <= tallies[:, :, 0]).all()
