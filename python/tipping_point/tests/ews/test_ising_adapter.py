"""The Ising adapter reads the simulator's layout and refuses what it must: runs without
complete.json, failed plateaus, a missing or altered pilot record, slots empty or doubled."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ews_planted import (
    NULL_EPS,
    PILOT_RUN,
    PLATEAU,
    REPLICAS,
    nulls,
    pilot,
    plateau_values,
    ramp,
    ramp_values,
    staircase,
    write_run,
)
from tipping_point.ews import fss, ising
from tipping_point.ising import io, protocol, run
from tipping_point.ising.exact import TC


def _data(**changes: object) -> dict:
    data = {
        "L": [16, 32],
        "seeds": [1, 2],
        "replicas": 8,
        "exclusions": [],
        "ramps": {"kind": "ramp", "x": 0.02},
        "nulls": {"kind": "fixed", "eps_L": list(NULL_EPS)},
        "staircase": {"kind": "staircase", "temperature": PLATEAU},
    }
    return data | changes


def test_series_splits_blocks_and_reads_per_site_values(planted: Any) -> None:
    """Catches replicas assigned to the wrong temperature block, or |m| not divided by L^2."""
    r = ising.open_run(planted.runs["L16-null-s1"])
    blocks = ising.series(r, "dense")
    assert [len(b.replicas) for b in blocks] == [REPLICAS, REPLICAS]
    eps = [float(fss.scaled_temperature(b.temperature[0], 16)) for b in blocks]
    assert eps == pytest.approx(list(NULL_EPS))
    stats = io.read_table(r.path, "stats")
    M = stats["magnetisation"].to_numpy().reshape(-1, 2 * REPLICAS)
    assert np.array_equal(blocks[1].abs_m, np.abs(M[:, REPLICAS:]).T / 256)
    measured = ising.measured(blocks[0])
    assert measured.steps[0] == 21 and np.all(measured.phase == 1)  # after therm = 20
    frames = ising.series(r, "matched")[0]
    assert len(frames.steps) == (500 // 8) * 8 and np.all(np.diff(frames.steps) == 1)


def test_thinning_keeps_an_even_cadence(planted: Any) -> None:
    """The grid starts at the first row. On a multiple of the stride it lands on rows; off it,
    rows within half a stride of the grid keep the mean spacing, which lets runs with different
    strides share one cadence."""
    (s,) = ising.series(ising.open_run(planted.runs["L32-ramp-s1"]), "dense")
    thinned = ising.thin(s, 4)
    assert thinned.steps[0] == s.steps[0] and np.all(np.diff(thinned.steps) == 4)
    uneven = ising.thin(s, 2.5)
    gaps = np.diff(uneven.steps)
    assert set(gaps.tolist()) <= {2, 3} and abs(gaps.mean() - 2.5) < 0.01
    grid = s.steps[0] + 2.5 * np.arange(len(uneven.steps))
    assert np.all(np.abs(uneven.steps - grid) <= 0.5)


def test_ramp_segment_finds_where_the_bath_reaches_each_eps(planted: Any) -> None:
    r = ising.open_run(planted.runs["L16-ramp-s1"])
    (s,) = ising.series(r, "dense")
    top, bottom = ising.ramp_segment(s, r, 3.0, 1.0)
    eps = r.eps(s.temperature)
    assert eps[top] <= 3.0 < eps[top - 1] and eps[bottom] <= 1.0 < eps[bottom - 1]
    with pytest.raises(ising.IsingDataError, match="not above"):
        ising.ramp_segment(s, r, 50.0, 1.0)  # the ramp starts at eps_L = 8.7


def test_a_run_without_complete_json_is_refused(tmp_path: Path) -> None:
    folder = write_run(tmp_path, "L16-ramp-s1", ramp(16), 16, 1, ramp_values(16), complete=False)
    with pytest.raises(ising.IsingDataError, match=r"no complete\.json"):
        ising.open_run(folder)
    complete, incomplete = ising.find_runs(tmp_path)
    assert not complete and [r.run_id for r in incomplete] == [folder.name]
    with pytest.raises(ising.IsingDataError, match=r"have no complete\.json"):
        ising.select(complete, incomplete, _data(L=[16], seeds=[1]))


def test_failed_plateaus_are_dropped_and_counted(tmp_path: Path) -> None:
    folder = write_run(tmp_path, "L16-qs-s1", staircase(), 16, 1, plateau_values(16), failed=(3.0,))
    kept, dropped = ising.plateaus(ising.open_run(folder), "dense")
    assert list(kept) == [pytest.approx(PLATEAU)] and dropped == 1
    frames, _ = ising.plateaus(ising.open_run(folder), "matched")
    assert list(frames) == [pytest.approx(PLATEAU)]


def test_pilot_record_is_read_and_checked(tmp_path: Path) -> None:
    """tau_c comes only from pilot.json: absent, altered, malformed or short of an L, scoring
    stops with a message that says where the record should be."""
    with pytest.raises(ising.IsingDataError, match=r"pilot\.json.*#50|#50.*pilot\.json"):
        ising.read_pilot(tmp_path, PILOT_RUN)
    path = pilot(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = ising.read_pilot(tmp_path, PILOT_RUN, digest)
    assert record.tau(16) == 4.0 and record.tau(32) == 8.0 and record.kernel_commit == "0" * 40
    with pytest.raises(ising.IsingDataError, match="no tau_c for L = 64"):
        record.tau(64)
    with pytest.raises(ising.IsingDataError, match="pins"):
        ising.read_pilot(tmp_path, PILOT_RUN, "f" * 64)
    path.write_text(json.dumps({"per_L": {"16": {"tau_c": -1.0}}, "kernel_commit": "x"}))
    with pytest.raises(ising.IsingDataError, match="positive"):
        ising.read_pilot(tmp_path, PILOT_RUN)
    path.write_text(json.dumps({"per_L": {"16": {}}}))
    with pytest.raises(ising.IsingDataError, match="not a pilot record"):
        ising.read_pilot(tmp_path, PILOT_RUN)


def test_select_fills_each_slot_exactly_once(planted: Any) -> None:
    complete, incomplete = ising.find_runs(planted.root)
    chosen = ising.select(complete, incomplete, _data())
    assert sorted(chosen) == [16, 32]
    inputs = chosen[16]
    assert [r.seed for r in inputs.ramps] == [1, 2] and [r.seed for r in inputs.staircases] == [
        1,
        2,
    ]
    blocks = [
        (r.seed, round(float(r.eps(r.schedule["params"]["temperatures"][b])), 6))
        for r, b in inputs.nulls
    ]
    assert blocks == [(1, 1.5), (1, 2.5), (2, 1.5), (2, 2.5)]
    with pytest.raises(ising.IsingDataError, match="no complete run"):
        ising.select(complete, incomplete, _data(seeds=[1, 2, 3]))
    with pytest.raises(ising.IsingDataError, match="fewer than"):
        ising.select(complete, incomplete, _data(replicas=9))
    other = _data(ramps={"kind": "ramp", "x": 0.3})  # no ramp at that Kibble-Zurek point
    with pytest.raises(ising.IsingDataError, match="no complete run"):
        ising.select(complete, incomplete, other)


def test_a_null_block_that_failed_its_check_is_dropped_and_counted(tmp_path: Path) -> None:
    """Null runs are read as equilibrium, so a block whose thermalised check failed (#50 writes
    the check for null runs too) is left out of scoring and reported, not silently used."""
    for tag, schedule, values in (
        ("L16-ramp-s1", ramp(16), ramp_values(16)),
        ("L16-qs-s1", staircase(), plateau_values(16)),
    ):
        write_run(tmp_path, tag, schedule, 16, 1, values)
    schedule = nulls(16, 500)
    hot = float(schedule.temperatures(1)[1])  # the eps_L = 2.5 block
    write_run(tmp_path, "L16-null-s1", schedule, 16, 1, ramp_values(16), failed=(hot,))
    inputs = ising.select(*ising.find_runs(tmp_path), _data(L=[16], seeds=[1]))[16]
    assert [b for _, b in inputs.nulls] == [0]
    assert inputs.dropped == ["20261023T000000Z-L16-null-s1 block 1"]


def test_the_pilot_record_from_50_reads(tmp_path: Path) -> None:
    """The committed copy of #50's measured record (fixtures/ising_pilot.json): the contract the
    scorer reads tau_c through."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "ising_pilot.json"
    folder = tmp_path / "ising" / "measured"
    folder.mkdir(parents=True)
    (folder / "pilot.json").write_bytes(fixture.read_bytes())
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    record = ising.read_pilot(tmp_path, "measured", digest)
    assert sorted(record.tau_c) == [16, 32, 64]
    assert record.tau(16) == pytest.approx(52.1, abs=0.05)
    assert record.tau(64) == pytest.approx(875.4, abs=0.05)
    assert len(record.kernel_commit.split("-")[0]) == 40


def test_a_doubled_slot_is_an_error_until_one_run_is_excluded(planted: Any, tmp_path: Path) -> None:
    extra = write_run(tmp_path, "L16-ramp-s1-again", ramp(16), 16, 1, ramp_values(16))
    complete, incomplete = ising.find_runs(planted.root)
    complete.append(ising.open_run(extra))
    with pytest.raises(ising.IsingDataError, match="more than one"):
        ising.select(complete, incomplete, _data())
    excluded = _data(exclusions=[{"run_id": extra.name, "reason": "a rerun"}])
    assert ising.select(complete, incomplete, excluded)[16].ramps[0].run_id != extra.name


def test_runs_from_the_real_simulator_read_the_same_way(tmp_path: Path) -> None:
    """The contract with the writer itself: a fixed run of two blocks with saved frames, from
    tipping_point.ising.run, comes back with the slots, shapes and values it holds. (The planted
    runs share io's writers and protocol's schedules; this pins the manifest too.)"""
    temps = (TC * (1 + 2 / 4), TC * (1 + 4 / 4))
    schedule = protocol.Fixed(temps, therm=4, sweeps=16, saves=2)
    folder = run.start(run.Spec(4, 3, seed=1), schedule, tmp_path, "null", "cpu")
    r = ising.open_run(folder)
    assert r.kind == "fixed" and r.L == 4 and r.seed == 1
    slots = ising._slots(r, _data(L=[4], seeds=[1], nulls={"kind": "fixed", "eps_L": [2.0, 4.0]}))
    assert [s[2:] for s in slots] == [(2.0, 0), (4.0, 1)]
    blocks = ising.series(r, "dense")
    M = io.read_table(folder, "stats")["magnetisation"].to_numpy().reshape(-1, 6)
    assert np.array_equal(blocks[1].abs_m, np.abs(M[:, 3:]).T / 16)
    assert np.all(ising.measured(blocks[0]).steps > 4)
    frames = ising.series(r, "matched")
    assert len(frames[0].steps) == 16 and frames[0].abs_m.shape == (3, 16)
