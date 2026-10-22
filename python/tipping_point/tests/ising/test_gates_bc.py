"""Gates B and C, and the independence and thermalisation checks, on saved runs: they pass on
correct runs at L = 4, 8 and 16, and a deliberate violation of each one's premise makes it fail."""

from pathlib import Path

import pytest
from tipping_point.ising import exact, gates, io, kernel, protocol, run
from tipping_point.ising.__main__ import main


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


def _thermalised(run_dir: Path) -> list[bool]:
    """The verdicts the staircase recorded in complete.json at its end."""
    return [c["passed"] for c in io.read_json(run_dir / "complete.json")["checks"]["thermalised"]]


def test_a_plateau_measured_too_soon_fails_the_thermalisation_check(tmp_path: Path) -> None:
    """Negative control: an L = 32 hot start quenched to 1.5 for 2 sweeps is still coarsening, so
    |m| keeps growing through the measurement window (~0.6 then ~0.8; t of 9 to 14 over seeds)."""
    schedule = protocol.Staircase([0.1], temperatures=(1.5,), warmup=0, min_therm=1, min_meas=100)
    assert schedule.therm == [2]
    run_dir = run.start(run.Spec(32, 64, seed=10), schedule, tmp_path, "quench", "cpu")
    assert _thermalised(run_dir) == [False]


def test_a_dwell_from_a_tau_seven_times_too_small_fails_the_thermalisation_check(
    tmp_path: Path,
) -> None:
    """Negative control: a staircase whose dwell comes from tau/7 (thermalise ~3 tau, measure
    ~14 tau) stepping onto Tc from the grid point above. The halves agree, since one grid step
    barely moves <|m|>, but the window holds ~14 of the plateau's own tau, short of 50."""
    argv = ["pilot", "--L", "16", "--T", repr(exact.TC), "--replicas", "16", "--therm", "500"]
    assert main([*argv, "--sweeps", "4000", "--data-dir", str(tmp_path), "--device", "cpu"]) == 0
    (pilot,) = (tmp_path / "ising").iterdir()
    (point,) = io.read_json(pilot / "tau.json")["points"]
    assert point["reliable"]
    tau = point["tau"]
    schedule = protocol.Staircase(
        [tau / 7, tau / 7],
        temperatures=(protocol.GRID[23], exact.TC),
        warmup=round(40 * tau),
        min_therm=1,
        min_meas=1,
    )
    run_dir = run.start(run.Spec(16, 16, seed=13), schedule, tmp_path / "qs", "short", "cpu")
    at_tc = io.read_json(run_dir / "complete.json")["checks"]["thermalised"][1]
    assert at_tc["temperature"] == exact.TC and at_tc["passed"] is False, at_tc


def test_dwelled_plateaus_pass_the_thermalisation_check(tmp_path: Path) -> None:
    """Positive control: plateaus held well past 20 tau, then measured for ~100 tau of |m|
    (L = 16: tau ~ 4 at 3.0, ~14 at 2.5), agree across halves and time reliably."""
    schedule = protocol.Staircase(
        [4.0, 14.0], temperatures=(3.0, 2.5), warmup=300, min_therm=300, min_meas=1000
    )
    assert schedule.meas == [1000, 1400]
    run_dir = run.start(run.Spec(16, 32, seed=11), schedule, tmp_path, "dwelled", "cpu")
    assert _thermalised(run_dir) == [True, True]
