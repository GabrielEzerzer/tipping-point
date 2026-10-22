"""The pilot record: tau_c(L) at Tc from pilot runs, its fit A L^z, and the reader that the ramp
rate, the null runs and the EWS scorer all take tau_c through."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tipping_point.ising import exact, io, pilot
from tipping_point.ising.__main__ import main

FIXTURE = Path(__file__).parents[1] / "fixtures" / "ising_pilot.json"


def _pilot_run(root: Path, L: int, T: float = exact.TC, sweeps: int = 3000) -> Path:
    argv = ["pilot", "--L", str(L), "--T", repr(T), "--replicas", "32", "--therm", "300"]
    assert main([*argv, "--sweeps", str(sweeps), "--data-dir", str(root), "--device", "cpu"]) == 0
    (run_dir,) = (root / "ising").glob(f"*-L{L}-pilot-s1")
    return run_dir


def test_tauc_writes_the_record_from_pilot_runs_at_tc(tmp_path: Path) -> None:
    """End to end: two pilots at Tc, then `tauc`. The record names the runs' commit, takes each
    tau from the same estimator as the pilot's own tau.json, and fits a line through them."""
    # out of order on purpose; tau_c is ~12 at L = 8 and ~3 at L = 4, so both hold >= 300 tau_c
    runs = [_pilot_run(tmp_path, 8, sweeps=6000), _pilot_run(tmp_path, 4)]
    assert main(["tauc", *map(str, runs), "--data-dir", str(tmp_path), "--draws", "200"]) == 0
    (path,) = (tmp_path / "ising").glob("*-tauc-L4-8/pilot.json")
    record = pilot.read(path)
    assert set(record) == {"kernel_commit", "per_L", "fit"}
    producer = io.read_json(runs[0] / "manifest.json")["producer"]
    assert record["kernel_commit"] == producer["git_sha"] + ("-dirty" if producer["dirty"] else "")
    assert list(record["per_L"]) == ["4", "8"]
    for run_dir in runs:
        (own,) = io.read_json(run_dir / "tau.json")["points"]  # the pilot's printout, same data
        entry = record["per_L"][str(io.read_json(run_dir / "manifest.json")["params"]["L"])]
        assert set(entry) == {
            "tau_c",
            "tau_c_ci",
            "tau_c_binning",
            "tau_e",
            "n_over_tau",
            "replicas",
            "sweeps",
            "seed",
        }
        assert (entry["tau_c"], entry["tau_c_binning"], entry["tau_e"]) == (
            own["tau"],
            own["binning"],
            own["tau_energy"],
        )
        assert entry["tau_c_ci"][0] < entry["tau_c"] < entry["tau_c_ci"][1]
        assert (entry["replicas"], entry["sweeps"], entry["seed"]) == (32, own["sweeps"], 1)
        assert entry["n_over_tau"] == pytest.approx(own["sweeps"] / entry["tau_c"])
    fit = record["fit"]
    for L in (4, 8):  # two sizes: the least-squares line runs through both
        assert fit["A"] * L ** fit["z"] == pytest.approx(record["per_L"][str(L)]["tau_c"])
    assert fit["A_ci"][0] < fit["A"] < fit["A_ci"][1] and fit["z_ci"][0] < fit["z"] < fit["z_ci"][1]
    assert pilot.tau_c(record, 16) == pytest.approx(fit["A"] * 16 ** fit["z"])


def test_tauc_refuses_runs_that_cannot_pin_tau_c(tmp_path: Path) -> None:
    """A pilot that never held exactly Tc, two runs of one size, a run that did not finish, or
    one too short: each is refused before any record exists. Short means under 300 tau_c, not
    just the 50 the estimator needs to find a window: ~120 tau_c reads ~8% low and passes that."""
    hot = _pilot_run(tmp_path / "hot", 4, T=3.0, sweeps=50)
    short = _pilot_run(tmp_path / "short", 8, sweeps=1500)  # ~120 tau_c, a window and all
    at_tc = _pilot_run(tmp_path / "tc", 4)  # ~900 tau_c: a duplicate is refused as a duplicate
    unfinished = _pilot_run(tmp_path / "unfinished", 8, sweeps=50)
    (unfinished / "complete.json").unlink()
    cases = {"hot": hot, "short": short, "twice": at_tc, "unfinished": unfinished}
    refusals = {}
    for name, run_dir in cases.items():
        argv = ["tauc", str(run_dir), str(at_tc), "--data-dir", str(tmp_path / "out")]
        with pytest.raises(SystemExit) as refused:
            main([*argv, "--draws", "10"])
        refusals[name] = refused.value.code
    assert all(isinstance(message, str) for message in refusals.values())  # exit status 1
    assert "need 300" in str(refusals["short"])
    assert not (tmp_path / "out").exists()


def test_the_committed_record_reads_and_extrapolates() -> None:
    """The fixture is a copy of the record #50's pilot wrote (CPU, seed 1, the fixed kernel), the
    shape #26's scorer reads: a measured size comes from per_L, any other from A L^z."""
    record = pilot.read(FIXTURE)
    assert list(record["per_L"]) == ["16", "32", "64"]
    A, z = record["fit"]["A"], record["fit"]["z"]
    for L, entry in record["per_L"].items():
        assert pilot.tau_c(record, int(L)) == entry["tau_c"]
        assert entry["tau_c"] == pytest.approx(A * int(L) ** z, rel=0.1)  # the fit runs close by
        assert entry["n_over_tau"] == pytest.approx(entry["sweeps"] / entry["tau_c"])
        assert entry["tau_c_ci"][0] < entry["tau_c"] < entry["tau_c_ci"][1]
    assert pilot.tau_c(record, 128) == pytest.approx(A * 128**z)


@pytest.mark.parametrize(
    "damage",
    [
        lambda r: r.pop("kernel_commit"),
        lambda r: r["per_L"]["32"].pop("tau_e"),
        lambda r: r["per_L"]["16"].update(tau_c=0.0),
        lambda r: r["per_L"]["64"].update(n_over_tau=200.0),  # 200 tau_c reads ~5% short
        lambda r: r["fit"].update(z=float("nan")),
        lambda r: r["fit"].pop("A_ci"),
        lambda r: r.update(per_L={}),
    ],
)
def test_reader_refuses_a_damaged_record(
    tmp_path: Path, damage: Callable[[dict[str, Any]], object]
) -> None:
    """Catches a hand-edited or truncated record reaching a ramp rate or a scorer's time unit."""
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    damage(record)
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        pilot.read(path)
