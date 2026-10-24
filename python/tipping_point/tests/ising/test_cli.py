"""The command line end to end: pilot smoke, validate, ramp and staircase on tiny settings."""

import hashlib
import json
import re
from pathlib import Path

import pytest
from tipping_point.ising import exact, gates, io, pilot, protocol
from tipping_point.ising.__main__ import _number, _taus, main

PILOT = Path(__file__).parents[1] / "fixtures" / "ising_pilot.json"  # #50's pilot record
# The engine's run-name rule as a literal: store::run::run_id's tag charset (ASCII letters,
# digits, '-' and '_'; engine/src/store/run.rs), which api::table::safe applies to the whole run
# id, at most 128 characters, before any read (engine/src/api/table.rs).
ENGINE_RUN_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}")


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
    fit it was read against, and r, and the manifest names the record (#55)."""
    argv = ["ramp", "--L", "4", "--x", "2", "--pilot", str(PILOT), "--replicas", "2"]
    argv += ["--pre", "10", "--hold", "10", "--data-dir", str(tmp_path), "--device", "cpu"]
    assert main(argv) == 0
    run_dir = _only_run(tmp_path)
    assert run_dir.name.endswith("-L4-ramp-x2-s1")
    manifest = io.read_json(run_dir / "manifest.json")["params"]
    params = manifest["schedule"]["params"]
    record = pilot.read(PILOT)
    fit = record["fit"]
    assert (params["x"], params["A"], params["z"]) == (2.0, fit["A"], fit["z"])
    assert params["rate"] == protocol.rate_for(4, 2.0, fit["A"], fit["z"])
    assert manifest["pilot"] == {
        "run_id": PILOT.parent.name,  # a run id, never a local path
        "sha256": hashlib.sha256(PILOT.read_bytes()).hexdigest(),
        "kernel_commit": record["kernel_commit"],
        "kernel_sha256": record["kernel_sha256"],
        "stale": False,
    }
    assert manifest["algorithm"]["kernel_sha256"] == pilot.kernel_sha256()


def test_every_run_id_the_cli_makes_opens_through_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches #67: the pre-registered slow ramp, x = 0.02, was named x0.02, a run the engine
    lists and then refuses to open. Every command that starts a run, with its numbers at their
    presets, and tauc's record folder, gets an id the engine's name filter accepts."""
    monkeypatch.setattr(protocol, "dwell", lambda tau, *_: (2, 6))  # plateaus of 8 sweeps
    monkeypatch.setattr(protocol, "NULL_TAU_C", 2)  # null runs of ~100 sweeps
    small = ["--replicas", "2", "--data-dir", str(tmp_path), "--device", "cpu"]
    ramp = ["ramp", "--L", "4", "--pre", "10", "--hold", "10"]
    commands = [
        ["pilot", "--L", "4", "--T", "3.0", "--therm", "0", "--sweeps", "2"],
        ["staircase", "--L", "4", "--tau", "1", "--warmup", "0", "--saves", "0"],
        *([*ramp, "--x", x, "--pilot", str(PILOT)] for x in ("0.02", "2")),
        [*ramp, "--rate", "50"],
        *(["null", "--L", "16", "--epsilon-L", e, "--pilot", str(PILOT)] for e in ("2", "4", "6")),
    ]
    for argv in commands:
        assert main([*argv, *small]) == 0, argv
    argv = ["validate", "--replicas", "256", "--therm", "100", "--sweeps", "100"]
    main([*argv, "--data-dir", str(tmp_path), "--device", "cpu"])  # its verdicts do not matter
    pilot.write(pilot.read(PILOT), tmp_path)  # the folder tauc writes its record to
    names = [p.name for p in (tmp_path / "ising").iterdir()]
    assert len(names) == len(commands) + 2
    assert [name for name in names if not ENGINE_RUN_NAME.fullmatch(name)] == []
    assert any(name.endswith("-L4-ramp-x0p02-s1") for name in names)


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.02, "0p02"), (2.0, "2"), (1 / 3, "0p333333"), (1e-5, "1e-05"), (1e20, "1e20")],
)
def test_numbers_in_tags_keep_to_the_engines_charset(value: float, text: str) -> None:
    """Any --x the parser takes, not just the presets: no '.' and no '+'."""
    assert _number(value) == text and ENGINE_RUN_NAME.fullmatch(f"x{text}")


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


def test_null_run_is_a_complete_thermalised_fixed_run_naming_epsilon_l_and_tau_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#26 call 5's nulls end to end, shortened to 20 tau_c: a fixed run at Tc (1 + 6/16) whose
    manifest names epsilon_L and the record's tau_c, judged thermalised in complete.json like a
    staircase plateau. An epsilon_L outside the presets is refused."""
    monkeypatch.setattr(protocol, "NULL_TAU_C", 20)
    argv = ["null", "--L", "16", "--epsilon-L", "6", "--pilot", str(PILOT), "--seed", "3"]
    assert main([*argv, "--data-dir", str(tmp_path), "--device", "cpu"]) == 0
    run_dir = _only_run(tmp_path)
    assert run_dir.name.endswith("-L16-null-e6-s3")
    schedule = io.read_json(run_dir / "manifest.json")["params"]["schedule"]
    params, tau_c = schedule["params"], pilot.tau_c(pilot.read(PILOT), 16)
    assert schedule["kind"] == "fixed" and (params["epsilon_L"], params["tau_c"]) == (6.0, tau_c)
    stamp = io.read_json(run_dir / "manifest.json")["params"]["pilot"]
    assert stamp == pilot.stamp(PILOT, pilot.read(PILOT)) and stamp["stale"] is False
    assert params["temperatures"] == [exact.TC * (1 + 6 / 16)]
    (check,) = io.read_json(run_dir / "complete.json")["checks"]["thermalised"]
    assert check["passed"], check
    assert all(c.passed for c in gates.gate_b(run_dir))  # its bursts sample Boltzmann, as any run's
    with pytest.raises(SystemExit):
        main([*argv[:4], "3", *argv[5:], "--data-dir", str(tmp_path / "x"), "--device", "cpu"])


def test_ramp_and_null_refuse_a_stale_record_unless_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches #55 at the command line: a record without kernel_sha256 (the shape of every record
    written before it) is refused by ramp and null with no run started; --allow-stale-pilot runs
    with a warning, and the manifest's stamp says stale."""
    record = json.loads(PILOT.read_text(encoding="utf-8"))
    del record["kernel_sha256"]
    old = tmp_path / "old.json"
    old.write_text(json.dumps(record), encoding="utf-8")
    ramp = ["ramp", "--L", "4", "--x", "2", "--pilot", str(old), "--replicas", "2"]
    ramp += ["--pre", "10", "--hold", "10", "--data-dir", str(tmp_path / "d"), "--device", "cpu"]
    null = ["null", "--L", "16", "--epsilon-L", "4", "--pilot", str(old)]
    null += ["--data-dir", str(tmp_path / "d"), "--device", "cpu"]
    for argv in (ramp, null):
        with pytest.raises(SystemExit) as refused:
            main(argv)
        message = str(refused.value.code)
        assert "records no kernel_sha256" in message and "--allow-stale-pilot" in message
    assert not (tmp_path / "d").exists()
    assert main([*ramp, "--allow-stale-pilot"]) == 0
    assert "records no kernel_sha256" in capsys.readouterr().err
    stamp = io.read_json(_only_run(tmp_path / "d") / "manifest.json")["params"]["pilot"]
    assert stamp["stale"] is True and stamp["kernel_sha256"] is None
    assert stamp["kernel_commit"] == record["kernel_commit"]


def test_null_run_refuses_a_size_the_record_did_not_measure(tmp_path: Path) -> None:
    """Catches a null run's length resting on the fit's extrapolation, which its manifest could
    not tell from a measurement: the record has no L = 128, so the run is refused, none started."""
    argv = ["null", "--L", "128", "--epsilon-L", "4", "--pilot", str(PILOT)]
    with pytest.raises(SystemExit, match="L = 128"):
        main([*argv, "--data-dir", str(tmp_path), "--device", "cpu"])
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


def test_runs_land_where_the_engine_looks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches #81 end to end: with TP_DATA_DIR unset a run lands in the engine's default data
    dir (LOCALAPPDATA comes first in its chain) instead of being refused; set, it lands there.
    Either way never in the repository, which lives in the vault (data is GBs)."""
    argv = ["pilot", "--L", "4", "--T", "3.0", "--replicas", "2", "--therm", "0", "--sweeps", "2"]
    monkeypatch.delenv("TP_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert main([*argv, "--device", "cpu"]) == 0
    assert len(list((tmp_path / "local" / "tipping-point" / "ising").iterdir())) == 1
    monkeypatch.setenv("TP_DATA_DIR", str(tmp_path / "set"))
    assert main([*argv, "--device", "cpu"]) == 0
    assert len(list((tmp_path / "set" / "ising").iterdir())) == 1
