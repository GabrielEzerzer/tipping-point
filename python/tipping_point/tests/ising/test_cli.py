"""The command line end to end: pilot smoke, validate, ramp and staircase on tiny settings."""

import json
from pathlib import Path

import pytest
from tipping_point.ising import exact, io, protocol
from tipping_point.ising.__main__ import _taus, main


def _only_run(root: Path) -> Path:
    (run_dir,) = (root / "ising").iterdir()
    return run_dir


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
    """--x picks r from L; the tag and the manifest record both."""
    argv = ["ramp", "--L", "4", "--x", "30", "--replicas", "2", "--pre", "10", "--hold", "10"]
    assert main([*argv, "--data-dir", str(tmp_path), "--device", "cpu"]) == 0
    run_dir = _only_run(tmp_path)
    assert run_dir.name.endswith("-L4-ramp-x30-s1")
    params = io.read_json(run_dir / "manifest.json")["params"]["schedule"]["params"]
    assert params["x"] == 30.0 and params["rate"] == protocol.rate_for(4, 30.0)


def test_staircase_cli_reads_taus_from_a_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pilot's tau.json drives the staircase dwell (shrunk here so the 40 plateaus take ms)."""
    pilot = tmp_path / "tau.json"
    pilot.write_text(
        json.dumps({"points": [{"temperature": T, "tau": 1.0} for T in protocol.GRID]})
    )
    monkeypatch.setattr(protocol, "dwell", lambda tau, *_: (2, 6))
    argv = ["staircase", "--L", "4", "--tau", str(pilot), "--replicas", "2", "--warmup", "0"]
    assert main([*argv, "--saves", "0", "--data-dir", str(tmp_path / "d"), "--device", "cpu"]) == 0
    schedule = io.read_json(_only_run(tmp_path / "d") / "manifest.json")["params"]["schedule"]
    assert schedule["kind"] == "staircase" and schedule["total"] == 40 * 8
    assert schedule["params"]["taus"] == [1.0] * 40


def test_tau_option_takes_a_number_or_a_complete_pilot(tmp_path: Path) -> None:
    assert _taus("4.5", (3.5, 2.0)) == [4.5, 4.5]
    pilot = tmp_path / "tau.json"
    pilot.write_text(json.dumps({"points": [{"temperature": 3.5, "tau": 2.0}]}))
    assert _taus(str(pilot), (3.5,)) == [2.0]
    with pytest.raises(SystemExit):
        _taus(str(pilot), (3.5, 2.0))  # no tau measured at 2.0


def test_data_dir_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs never default into the repository (it lives in the vault; data is GBs)."""
    monkeypatch.delenv("TP_DATA_DIR", raising=False)
    with pytest.raises(SystemExit):
        main(["pilot", "--L", "4"])
