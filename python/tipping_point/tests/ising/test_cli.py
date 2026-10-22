"""The command line end to end: pilot smoke, validate, ramp and staircase on tiny settings."""

import json
from pathlib import Path

import pytest
from tipping_point.ising import exact, io, pilot, protocol
from tipping_point.ising.__main__ import _taus, main

PILOT = Path(__file__).parents[1] / "fixtures" / "ising_pilot.json"  # #50's pilot record


def _only_run(root: Path) -> Path:
    (run_dir,) = (root / "ising").iterdir()
    return run_dir


def _pilot_file(path: Path, unreliable: tuple[float, ...] = ()) -> Path:
    """A tau.json covering the staircase grid, with the given temperatures marked unreliable."""
    points = [
        {"temperature": T, "tau": 1.0, "reliable": T not in unreliable} for T in protocol.GRID
    ]
    path.write_text(json.dumps({"points": points}))
    return path


def test_pilot_smoke_at_l16(tmp_path: Path) -> None:
    """Short CPU pilot at L = 16: a sane tau at T = 3, a larger one at Tc, manifest, tau.json."""
    argv = ["pilot", "--L", "16", "--T", "3.0", "--T", repr(exact.TC), "--replicas", "8"]
    argv += ["--therm", "300", "--sweeps", "2000", "--data-dir", str(tmp_path), "--device", "cpu"]
    assert main(argv) == 0
    run_dir = _only_run(tmp_path)
    manifest = io.read_json(run_dir / "manifest.json")
    assert manifest["params"]["schedule"]["kind"] == "fixed" and manifest["params"]["L"] == 16
    assert (run_dir / "complete.json").exists()
    tau = {round(p["temperature"], 3): p for p in io.read_json(run_dir / "tau.json")["points"]}
    hot, critical = tau[3.0], tau[round(exact.TC, 3)]
    assert 1.0 < hot["tau"] < 10.0 and hot["reliable"]
    assert hot["binning"] == pytest.approx(hot["tau"], rel=0.3)
    assert hot["tau"] < critical["tau"] < 2000.0  # critical slowing down, already visible at L = 16


def test_validate_cli_passes_and_writes_its_verdicts(tmp_path: Path) -> None:
    argv = ["validate", "--replicas", "256", "--therm", "100", "--sweeps", "100"]
    assert main([*argv, "--data-dir", str(tmp_path), "--device", "cpu"]) == 0
    verdicts = json.loads((_only_run(tmp_path) / "gates.json").read_text("utf-8"))
    assert {v["gate"] for v in verdicts} == {"A", "B", "C", "independence"}
    assert all(v["passed"] for v in verdicts)


def test_ramp_cli_from_a_kibble_zurek_point(tmp_path: Path) -> None:
    """--x picks r from L and the pilot record's A L^z; the tag and the manifest record x, the
    fit it was read against, and r."""
    argv = ["ramp", "--L", "4", "--x", "2", "--pilot", str(PILOT), "--replicas", "2"]
    argv += ["--pre", "10", "--hold", "10", "--data-dir", str(tmp_path), "--device", "cpu"]
    assert main(argv) == 0
    run_dir = _only_run(tmp_path)
    assert run_dir.name.endswith("-L4-ramp-x2-s1")
    params = io.read_json(run_dir / "manifest.json")["params"]["schedule"]["params"]
    fit = pilot.read(PILOT)["fit"]
    assert (params["x"], params["A"], params["z"]) == (2.0, fit["A"], fit["z"])
    assert params["rate"] == protocol.rate_for(4, 2.0, fit["A"], fit["z"])


def test_ramp_manifest_records_the_save_schedule_it_ran(tmp_path: Path) -> None:
    """The manifest's save schedule is the one the run kept: as many bursts as configs holds."""
    argv = ["ramp", "--L", "16", "--rate", "400", "--replicas", "2", "--pre", "10", "--hold", "10"]
    assert main([*argv, "--data-dir", str(tmp_path), "--device", "cpu"]) == 0
    run_dir = _only_run(tmp_path)
    schedule = io.read_json(run_dir / "manifest.json")["params"]["schedule"]
    bursts = set(io.read_table(run_dir, "configs")["burst"].to_pylist())
    assert schedule["params"]["L"] == 16 and schedule["saves"]["bursts"] == len(bursts) > 0


def test_ramp_x_needs_a_pilot_record(tmp_path: Path) -> None:
    """Catches a rate taken from a hidden constant: --x without a record is refused, no run."""
    argv = ["ramp", "--L", "4", "--x", "2", "--data-dir", str(tmp_path), "--device", "cpu"]
    with pytest.raises(SystemExit, match="--pilot"):
        main(argv)
    assert not (tmp_path / "ising").exists()


def test_staircase_cli_reads_taus_from_a_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pilot's tau.json drives the staircase dwell (shrunk here so the 40 plateaus take ms)."""
    pilot = _pilot_file(tmp_path / "tau.json")
    monkeypatch.setattr(protocol, "dwell", lambda tau, *_: (2, 6))
    argv = ["staircase", "--L", "4", "--tau", str(pilot), "--replicas", "2", "--warmup", "0"]
    assert main([*argv, "--saves", "0", "--data-dir", str(tmp_path / "d"), "--device", "cpu"]) == 0
    schedule = io.read_json(_only_run(tmp_path / "d") / "manifest.json")["params"]["schedule"]
    assert schedule["kind"] == "staircase" and schedule["total"] == 40 * 8
    assert schedule["params"]["taus"] == [1.0] * 40


def test_staircase_refuses_a_pilot_tau_marked_unreliable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a dwell set from a pilot too short to measure its own tau: such a pilot reports
    tau several times too small (AR(1) tau = 199 from 600 samples reads 41.7), so the plateau
    nearest Tc would be measured before it equilibrates. Refused before any run starts."""
    near_tc = protocol.GRID[24]
    pilot = _pilot_file(tmp_path / "tau.json", unreliable=(near_tc,))
    monkeypatch.setattr(protocol, "dwell", lambda tau, *_: (2, 6))  # keeps a wrongful run short
    argv = ["staircase", "--L", "4", "--tau", str(pilot), "--replicas", "2", "--warmup", "0"]
    with pytest.raises(SystemExit) as refused:
        main([*argv, "--saves", "0", "--data-dir", str(tmp_path / "d"), "--device", "cpu"])
    assert isinstance(refused.value.code, str) and str(near_tc) in refused.value.code  # exit 1
    assert not (tmp_path / "d").exists()


def test_pilot_too_short_for_a_window_is_unreliable_and_does_not_crash(tmp_path: Path) -> None:
    """Catches tau = 0 from the forced tail of the running sum (two samples give tau(1) = 0
    exactly): it was marked reliable, and the N/tau print divided by it after tau.json existed."""
    argv = ["pilot", "--L", "16", "--T", "3.0", "--replicas", "4", "--therm", "0", "--sweeps", "2"]
    assert main([*argv, "--data-dir", str(tmp_path), "--device", "cpu"]) == 0
    (point,) = io.read_json(_only_run(tmp_path) / "tau.json")["points"]
    assert point["tau"] > 0 and point["reliable"] is False


def test_tau_option_takes_a_number_or_a_complete_pilot(tmp_path: Path) -> None:
    assert _taus("4.5", (3.5, 2.0)) == [4.5, 4.5]
    pilot = tmp_path / "tau.json"
    pilot.write_text(json.dumps({"points": [{"temperature": 3.5, "tau": 2.0, "reliable": True}]}))
    assert _taus(str(pilot), (3.5,)) == [2.0]
    with pytest.raises(SystemExit):
        _taus(str(pilot), (3.5, 2.0))  # no tau measured at 2.0


@pytest.mark.parametrize("value", ["0", "-3", "nan", "inf"])
def test_numeric_tau_must_be_a_positive_number(value: str) -> None:
    """Catches --tau 0 or -3 silently running every plateau at the dwell floors, and nan or inf
    crashing inside dwell() with no hint of which flag was wrong."""
    with pytest.raises(SystemExit, match="positive"):
        _taus(value, (3.5, 2.0))


def test_pilot_tau_must_be_a_positive_number(tmp_path: Path) -> None:
    """The same guard covers a hand-written tau.json that marks a zero tau reliable."""
    pilot = tmp_path / "tau.json"
    pilot.write_text(json.dumps({"points": [{"temperature": 3.5, "tau": 0.0, "reliable": True}]}))
    with pytest.raises(SystemExit, match="positive"):
        _taus(str(pilot), (3.5,))


def test_data_dir_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs never default into the repository (it lives in the vault; data is GBs)."""
    monkeypatch.delenv("TP_DATA_DIR", raising=False)
    with pytest.raises(SystemExit):
        main(["pilot", "--L", "4"])
