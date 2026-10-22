"""Protocols are pure functions of the integer sweep index, with exact boundaries and saves."""

import itertools
import random

import pytest
from tipping_point.ising import exact, protocol
from tipping_point.ising.protocol import Burst, Fixed, Ramp, Staircase


def _answers(schedule: protocol.Schedule, steps: list[int]) -> list[tuple]:
    return [
        (
            s,
            schedule.temperatures(s),
            schedule.segment(s),
            schedule.phase(s),
            schedule.records_stats(s),
        )
        for s in steps
    ]


def _assert_pure(schedule: protocol.Schedule) -> None:
    """Asking about steps in a random order, or on a rebuilt copy, gives the same answers."""
    steps = list(range(1, schedule.total + 1))
    forward = _answers(schedule, steps)
    shuffled = steps[:]
    random.Random(0).shuffle(shuffled)
    assert sorted(_answers(schedule, shuffled)) == forward
    rebuilt = protocol.from_json(schedule.to_json())
    assert _answers(rebuilt, steps) == forward and rebuilt.frames == schedule.frames


def test_grid_runs_hot_to_cold_with_one_point_next_to_tc() -> None:
    """The 40-point grid the staircase walks, and the one point that sits 4.5e-5 above Tc."""
    assert len(protocol.GRID) == 40
    assert protocol.GRID[0] == 3.5 and protocol.GRID[-1] == 1.5
    assert all(a > b for a, b in zip(protocol.GRID, protocol.GRID[1:], strict=False))
    nearest = min(protocol.GRID, key=lambda T: abs(T - exact.TC))
    assert nearest - exact.TC == pytest.approx(4.5e-5, abs=1e-6)


def test_dwell_is_20_and_100_tau_with_floors() -> None:
    assert protocol.dwell(1.0) == (500, 2000)
    assert protocol.dwell(300.0) == (6000, 30000)
    assert protocol.dwell(30.125) == (603, 3013)  # rounded up, never short


def test_staircase_plateaus_tile_the_run() -> None:
    """Catches gaps or overlaps between plateaus and a dwell that ignores tau."""
    taus = [1.0, 40.0, 3.0]
    s = Staircase(taus, temperatures=(3.5, 2.3, 1.5), warmup=1000, saves=10)
    assert s.therm == [1500, 800, 500] and s.meas == [2000, 4000, 2000]
    assert s.starts == [1, 3501, 8301] and s.total == 10800
    assert s.temperatures(3500) == (3.5,) and s.temperatures(3501) == (2.3,)
    assert s.phase(s.meas_starts[1] - 1) == 0 and s.phase(s.meas_starts[1]) == 1
    assert s.temperatures(s.total) == (1.5,) and s.segment(s.total) == 2
    _assert_pure(s)


def test_staircase_bursts_sit_inside_measurement_windows() -> None:
    """Catches configs saved during thermalisation, overlapping bursts, or frames off-stride."""
    burst = Burst(frames=8, stride=3)
    s = Staircase([1.0, 50.0], temperatures=(3.0, 2.0), saves=20, burst=burst)
    by_burst: dict[int, list[int]] = {}
    for step, (b, f) in sorted(s.frames.items()):
        j = s.segment(step)
        assert s.phase(step) == 1 and step < s.starts[j] + s.therm[j] + s.meas[j]
        by_burst.setdefault(b, []).append(step)
        assert f == len(by_burst[b]) - 1
    assert len(by_burst) == 40  # 20 per plateau, numbered across the run
    for steps in by_burst.values():
        assert [b - a for a, b in itertools.pairwise(steps)] == [3] * 7
    starts = sorted(steps[0] for steps in by_burst.values())
    assert min(b - a for a, b in itertools.pairwise(starts)) >= burst.span


def test_short_windows_save_fewer_bursts_rather_than_overlap() -> None:
    s = Staircase([1.0], temperatures=(3.0,), min_meas=100, saves=200, burst=Burst(stride=2))
    assert len(s.frames) == (100 // 16) * 8


def test_ramp_temperature_is_computed_from_the_sweep_index() -> None:
    """Catches an accumulated temperature (drifts) and an off-by-one at either end of the ramp."""
    r = Ramp(rate=1000, L=16, pre=1000, hold=1000)
    assert (r.t0, r.t1, r.total) == (1000, 3000, 4000)
    assert r.temperatures(1) == (3.5,) and r.temperatures(1000) == (3.5,)
    assert r.temperatures(1001) == (3.5 - 1 / 1000,)
    for step in (1001, 1777, 2999):
        assert r.temperatures(step) == (3.5 - (step - 1000) / 1000,)
    assert r.temperatures(3000) == (1.5,) and r.temperatures(4000) == (1.5,)
    assert [r.phase(t) for t in (1000, 1001, 3000, 3001)] == [0, 1, 1, 2]
    _assert_pure(r)


def test_ramp_crosses_tc_exactly_at_t_c() -> None:
    """t_c is where the bath crosses Tc; lead times are measured against it."""
    r = Ramp(rate=1000, L=16)
    assert r.t_c == 1000 + 1000 * (3.5 - exact.TC)
    before, after = int(r.t_c), int(r.t_c) + 1
    assert r.temperatures(before)[0] > exact.TC > r.temperatures(after)[0]


def _starts(r: Ramp) -> list[int]:
    return sorted(step for step, (_, f) in r.frames.items() if f == 0)


def _in_band(r: Ramp, L: int, step: int) -> bool:
    return -5 <= L * (r.temperatures(step)[0] - exact.TC) / exact.TC <= 10


def _gaps(r: Ramp, L: int, in_band: bool) -> set[int]:
    """Sweeps between consecutive burst starts that both lie inside (or both outside) the band."""
    pairs = itertools.pairwise(_starts(r))
    return {b - a for a, b in pairs if _in_band(r, L, a) == _in_band(r, L, b) == in_band}


def test_ramp_saves_a_burst_every_0_005_of_cooling_outside_the_band() -> None:
    """Outside epsilon_L in [10, -5] saves stay 0.005 of cooling apart; bursts never leave the
    ramp or overlap, and a ramp too fast for a spacing saves back to back."""
    r = Ramp(rate=4000, L=128)  # 0.005 of cooling = 20 sweeps; the band's 2.5 < a burst's 8
    assert _gaps(r, 128, in_band=False) == {20} and _gaps(r, 128, in_band=True) == {8}
    assert min(b - a for a, b in itertools.pairwise(_starts(r))) == 8
    assert all(r.phase(step) == 1 for step in r.frames)
    _assert_pure(r)
    fast = Ramp(rate=96, L=16)  # 0.005 of cooling < one sweep: bursts fall back to back
    assert {b - a for a, b in itertools.pairwise(_starts(fast))} == {8}


def test_ramp_saves_evenly_in_epsilon_l_inside_the_band() -> None:
    """#26 call 5: saved every 0.005 in T, L = 128 got 3.5 bursts per unit epsilon_L to L = 16's
    28, about one per scoring window. On the slow ramp (x = 0.02) the bursts per unit epsilon_L
    inside the band must match across L, and the manifest must say where they went."""
    per_unit = {}
    for L in (16, 128):
        r = Ramp(rate=protocol.rate_for(L, 0.02, 0.13, 2.2), L=L)
        inside = [s for s in _starts(r) if _in_band(r, L, s)]
        hot_edge = min(10.0, L * (r.T0 - exact.TC) / exact.TC)  # L = 16's band starts above T0
        per_unit[L] = len(inside) / (hot_edge + 5)
        saves = r.to_json()["saves"]
        assert saves["bursts_in_band"] == len(inside) and saves["bursts"] == len(_starts(r))
        assert _gaps(r, L, in_band=True) == {saves["band_gap"]} and saves["band_gap"] > 8
    assert abs(per_unit[16] - per_unit[128]) <= 1
    assert per_unit[128] == pytest.approx(protocol.BAND_PER_EPSILON_L, abs=1)


def test_rate_for_puts_every_size_at_the_same_point_of_the_true_tau() -> None:
    """x = tau_c(L) / (sweeps per unit epsilon_L) at every L, so the window |epsilon_L| < 1 takes
    2/x tau_c: 100 at x = 0.02 and 1 at x = 2. Catches a rate that ignores A (pre-#50, the fixed
    kernel's 2.5x slower tau made every ramp 2.5x faster than its x said) or uses another z."""
    A, z = 0.13, 2.2
    for L in (16, 32, 64, 128):
        for x in (0.02, 2.0):
            per_epsilon_L = protocol.rate_for(L, x, A, z) * exact.TC / L
            assert per_epsilon_L == pytest.approx(A * L**z / x, rel=0.01)
    slower = protocol.rate_for(64, 0.02, 2.5 * A, z)
    assert slower == pytest.approx(2.5 * protocol.rate_for(64, 0.02, A, z), rel=1e-6)


def test_the_pre_50_x_values_are_x_times_a() -> None:
    """The old rate L^3.17 / (x Tc) is this rate at x A with z = 2.17: on the pre-#47 kernel
    (A = 0.052) the nominal 0.3 and 30 were 0.016 and 1.6."""
    for x_old in (0.3, 30.0):
        old = 32**3.17 / (x_old * exact.TC)
        assert protocol.rate_for(32, x_old * 0.052, 0.052, 2.17) == pytest.approx(old, rel=1e-3)


def test_fixed_blocks_and_phases() -> None:
    f = Fixed((1.5, 2.0, 3.5), therm=10, sweeps=20, stats_stride=7, saves=2)
    assert f.blocks == 3 and f.total == 30 and f.temperatures(5) == (1.5, 2.0, 3.5)
    assert f.phase(10) == 0 and f.phase(11) == 1
    assert [t for t in range(1, 31) if f.records_stats(t)] == [7, 14, 21, 28, 30]  # always the last
    assert min(f.frames) == 11 and max(f.frames) <= 30
    _assert_pure(f)
