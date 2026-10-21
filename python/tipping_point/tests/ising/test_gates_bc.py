"""Gates B and C, and the independence check, on saved runs: they pass on correct runs at L = 4,
8 and 16, and a deliberate violation of each one's premise makes it fail."""

from pathlib import Path

import pytest
from tipping_point.ising import gates, kernel, protocol, run


def _passed(checks: list[gates.Check], gate: str) -> list[bool]:
    return [c.passed for c in checks if c.gate == gate]


def test_validation_run_passes_b_c_and_independence(
    validation: tuple[Path, list[gates.Check]],
) -> None:
    """The L = 4 run from `validate`: B, C and independence pass at all three temperatures."""
    checks = validation[1]
    for gate in ("B", "C", "independence"):
        assert _passed(checks, gate) == [True] * 3, [c for c in checks if c.gate == gate]


def test_gates_b_and_c_hold_at_l16(tmp_path: Path) -> None:
    """Catches an identity that only holds on the tiny lattice: B and C are claims for any L."""
    schedule = protocol.Fixed((2.0, 3.0), therm=300, sweeps=300, stats_stride=10, saves=5)
    run_dir = run.start(run.Spec(16, 64, seed=4, tally=True), schedule, tmp_path, "bc16", "cpu")
    checks = gates.gate_b(run_dir) + gates.gate_c(run_dir)
    assert [c.passed for c in checks] == [True] * 4, checks


def test_gate_b_judges_each_staircase_plateau(tmp_path: Path) -> None:
    """Gate B reads a production-shaped output: one verdict per plateau temperature."""
    schedule = protocol.Staircase(
        [3.0, 3.0, 3.0], temperatures=(3.5, 2.6, 1.8), warmup=200, min_therm=100, min_meas=400
    )
    run_dir = run.start(run.Spec(8, 32, seed=5), schedule, tmp_path, "qs8", "cpu")
    checks = gates.gate_b(run_dir)
    assert sorted(c.temperature for c in checks) == [1.8, 2.6, 3.5]
    assert all(c.passed for c in checks), checks


def test_gate_b_fails_at_the_wrong_temperature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches sampling at the wrong temperature (here beta for T) on any lattice size."""
    real = kernel.accept_table
    monkeypatch.setattr(kernel, "accept_table", lambda T: real(1 / T))
    schedule = protocol.Fixed((2.5,), therm=200, sweeps=200, saves=5)
    run_dir = run.start(run.Spec(8, 32, seed=6), schedule, tmp_path, "beta", "cpu")
    assert not any(c.passed for c in gates.gate_b(run_dir))


def test_gate_c_fails_out_of_equilibrium(tmp_path: Path) -> None:
    """Catches tallies before equilibrium: a hot start quenched to 1.5 moves mostly downhill."""
    schedule = protocol.Fixed((1.5,), therm=0, sweeps=20)
    run_dir = run.start(run.Spec(16, 64, seed=7, tally=True), schedule, tmp_path, "quench", "cpu")
    assert not any(c.passed for c in gates.gate_c(run_dir))
