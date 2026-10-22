"""Kernel pieces against explicit loops: colouring, neighbour sums, table, totals, tallies, and
the reachability of every L = 4 state."""

import math

import numpy as np
import pytest
import torch
from tipping_point.ising import exact, kernel


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


def test_accept_table_is_min_one_exp_with_one_half_at_zero_per_replica() -> None:
    """Catches beta-for-T, a missing clamp, one temperature applied to every replica, or every
    dE = 0 move accepted (the #47 trap)."""
    table = kernel.accept_table(torch.tensor([2.0, 4.0], dtype=torch.float64))
    assert table.dtype == torch.float32 and table.shape == (2, 5)
    for row, T in zip(table, (2.0, 4.0), strict=True):
        expected = [0.5 if dE == 0 else min(1.0, math.exp(-dE / T)) for dE in (-8, -4, 0, 4, 8)]
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


def test_half_sweep_touches_one_colour_and_takes_downhill_moves_and_half_the_level_ones() -> None:
    """Catches updating both colours, rejecting a move that lowers the energy, or taking every
    move that leaves it unchanged (the #47 trap) or none."""
    s = _random_spins(64, 8)
    black = kernel.checkerboard(8, "cpu")
    table = kernel.accept_table(torch.full((64,), 2.0, dtype=torch.float64))
    new = kernel.half_sweep(s, black, table, torch.Generator().manual_seed(1))
    assert torch.equal(new[:, ~black], s[:, ~black])
    sh = s * kernel.neighbour_sum(s)  # dE = 2 s h
    downhill, level = black & (sh < 0), black & (sh == 0)
    assert torch.equal(new[downhill], -s[downhill])
    n = int(level.sum())  # ~770 level moves: a fraction 5 sigma from 1/2 is off by ~0.09
    flipped = (new[level] != s[level]).double().mean().item()
    assert abs(flipped - 0.5) <= 5 * 0.5 / math.sqrt(n), (flipped, n)
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


def _l4_field() -> tuple[np.ndarray, np.ndarray]:
    """Every L = 4 state (state k has spin +1 at flat site n when bit n of k is set) and s * h."""
    s, _, _ = exact.enumerate_states(4)
    h = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
    return s, s * h


def _reach(table: np.ndarray, start: np.ndarray, backward: bool = False) -> np.ndarray:
    """L = 4 states that whole sweeps (black half, then white) can take `start` to, or with
    `backward`, the states they can take to `start`. A half-sweep flips each site of its colour
    for sure (table entry 1), maybe (between 0 and 1) or never (0), independently, so x goes to
    x ^ sure[x] ^ any subset of maybe[x]."""
    s, sh = _l4_field()
    p = table[(sh.reshape(len(s), -1) + 4) // 2]  # (states, sites)
    bit = 1 << np.arange(16)
    black = kernel.checkerboard(4, "cpu").numpy().ravel()
    halves = []
    for colour in (black, ~black):
        sure, maybe = ((p >= 1) & colour) @ bit, ((p > 0) & (p < 1) & colour) @ bit
        subsets = ((np.arange(256)[:, None] >> np.arange(8)) & 1) @ bit[colour]
        halves.append((sure, maybe, subsets))

    def step(half: tuple[np.ndarray, np.ndarray, np.ndarray], front: np.ndarray) -> np.ndarray:
        sure, maybe, subsets = half
        xs = np.flatnonzero(front)
        if backward:  # the states differing from x on this colour only that can go to x
            ys = xs[:, None] ^ subsets
            hit = ys[((ys ^ sure[ys] ^ xs[:, None]) & ~maybe[ys]) == 0]
        else:
            hit = (xs ^ sure[xs])[:, None] ^ (subsets & maybe[xs][:, None])
        out = np.zeros_like(front)
        out[hit.ravel()] = True
        return out

    first, second = halves[::-1] if backward else halves
    seen, between = start.copy(), np.zeros_like(start)
    front = start
    while front.any():
        front = step(first, front) & ~between
        between |= front
        front = step(second, front) & ~seen
        seen |= front
    return seen


def test_sweeps_connect_every_l4_state_to_every_other() -> None:
    """Catches an ergodicity hole (#47). Accepting every dE = 0 move trapped the 36 states whose
    sites all have two up and two down neighbours: each half-sweep flips its whole colour, so
    they go x -> -x -> x for ever, cut off from the other 65,500. Exact, by enumeration; only
    which table entries are 0, 1 or in between matters, so one temperature stands for all."""
    table = kernel.accept_table(torch.tensor([exact.TC], dtype=torch.float64))[0].numpy()
    _, sh = _l4_field()
    trapped = (sh == 0).all(axis=(1, 2))
    all_up = np.zeros(2**16, dtype=bool)
    all_up[-1] = True

    old = table.copy()
    old[2] = 1.0  # the rule before #47: the trap is closed and the rest is one class
    assert trapped.sum() == 36
    assert np.array_equal(_reach(old, trapped), trapped)
    assert np.array_equal(_reach(old, all_up) & _reach(old, all_up, backward=True), ~trapped)

    assert _reach(table, all_up).all() and _reach(table, all_up, backward=True).all()
