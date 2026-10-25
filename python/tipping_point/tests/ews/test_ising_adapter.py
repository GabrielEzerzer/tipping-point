"""The Ising adapter reads the simulator's layout and refuses what it must: runs without
complete.json, failed plateaus, a missing or altered pilot record, slots empty or doubled."""

import hashlib
import json
import tomllib
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

V1 = Path(__file__).resolve().parents[4] / "paper" / "prereg" / "ising-ews-v1.toml"


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
    with pytest.raises(ising.IsingDataError, match=r"no kernel_sha256.*--allow-stale-pilot"):
        ising.read_pilot(tmp_path, PILOT_RUN, digest)  # the planted record predates #55
    record = ising.read_pilot(tmp_path, PILOT_RUN, digest, allow_stale=True)
    assert record.tau(16) == 4.0 and record.tau(32) == 8.0 and record.kernel_commit == "0" * 40
    assert record.kernel_sha256 is None and record.stale
    with pytest.raises(ising.IsingDataError, match="no tau_c for L = 64"):
        record.tau(64)
    with pytest.raises(ising.IsingDataError, match="pins"):
        ising.read_pilot(tmp_path, PILOT_RUN, "f" * 64)
    extended = json.loads(path.read_text("utf-8"))
    extended["per_L"]["16"] |= {"tau_c_ci": [3.5, 4.6], "a_key_from_later": 1}
    path.write_text(json.dumps(extended))
    record = ising.read_pilot(tmp_path, PILOT_RUN, allow_stale=True)  # extra keys are fine
    assert record.tau(16) == 4.0 and record.tau_c_ci == {16: [3.5, 4.6]}
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


def test_a_ramp_over_another_eps_window_or_x_is_refused(planted: Any) -> None:
    """#26 decision 4: with data.ramps.eps_L, a ramp at the prereg's x must have cooled over that
    window. The planted ramps run from T = 3.5 to 1.5, eps_L 8.678 to -5.424 at L = 16: refused
    against [12, -8], naming the run and both windows, and read against their own window. A ramp
    at another x is another experiment (the fast ramp, at the same L and seed): it fills no slot,
    and the error for the slot left empty names it and its x."""
    complete, incomplete = ising.find_runs(planted.root)
    v1_window = {"kind": "ramp", "x": 0.02, "eps_L": [12.0, -8.0]}
    with pytest.raises(
        ising.IsingDataError, match=r"L16-ramp-s1: .*8\.678.*-5\.42.*\[12\.0, -8\.0"
    ):
        ising.select(complete, incomplete, _data(ramps=v1_window))
    own = [float(fss.scaled_temperature(T, 16)) for T in (3.5, 1.5)]
    chosen = ising.select(complete, incomplete, _data(L=[16], ramps=v1_window | {"eps_L": own}))
    assert [r.seed for r in chosen[16].ramps] == [1, 2]
    fast = {"kind": "ramp", "x": 0.3}
    with pytest.raises(
        ising.IsingDataError, match=r"another x than 0\.3: .*L16-ramp-s1 \(x = 0\.02\)"
    ):
        ising.select(complete, incomplete, _data(ramps=fast))


def test_scored_runs_must_come_from_the_freeze_tags_commit(tmp_path: Path) -> None:
    """#26 decision 6 as a check: under a frozen prereg a scored run must name the commit the
    freeze tag points at, from a clean tree. One that does is accepted; one from another commit,
    a dirty tree, or naming no commit (a wheel built outside a checkout, or a planted run) is
    refused, the message naming it, its git_sha and the tag's commit, and never the good run."""

    def made(name: str, git_sha: str | None, dirty: bool | None) -> ising.Run:
        manifest = {"run_id": name, "producer": {"git_sha": git_sha, "dirty": dirty}}
        return ising.Run(tmp_path / name, manifest, {}, "")

    tagged = "a" * 40
    good = made("accepted-run", tagged, False)
    ising._same_commit([good], tagged)
    for bad in (
        made("other", "b" * 40, False),
        made("dirty", tagged, True),
        made("bare", None, None),
    ):
        with pytest.raises(ising.IsingDataError) as refused:
            ising._same_commit([good, bad], tagged)
        message = str(refused.value)
        git_sha = bad.manifest["producer"]["git_sha"]
        assert f"{bad.run_id} (git_sha {git_sha}" in message and tagged in message, message
        assert "accepted-run" not in message


def test_scored_ramps_and_nulls_must_name_the_pinned_pilot_record(
    planted: Any, tmp_path: Path
) -> None:
    """#26 K: under a frozen prereg every scored ramp and null run must have been made against
    the pilot record data.tau_c pins, whose tau_c set the ramp's rate and the null's length.
    The planted ramps and nulls name a record with sha256 0...0: accepted when that is the pin,
    refused against another, the message naming each run, its sha256 and the pinned one, never
    a staircase (it took --tau, no record). A ramp naming no record is refused too, while
    selection alone, the exploratory path, still takes it."""
    named = "0" * 64  # what the planted stamps carry
    chosen = ising.select(*ising.find_runs(planted.root), _data())
    ising._same_record(chosen, named)
    with pytest.raises(ising.IsingDataError) as refused:
        ising._same_record(chosen, "f" * 64)
    message = str(refused.value)
    assert f"L16-ramp-s1 (pilot sha256 {named})" in message and "f" * 64 in message
    assert "L32-null-s2" in message and "-qs-" not in message
    stamp = {"run_id": "pilot", "sha256": named, "stale": False}
    write_run(tmp_path, "L16-ramp-s1", ramp(16), 16, 1, ramp_values(16))  # names no record
    write_run(tmp_path, "L16-null-s1", nulls(16, 500), 16, 1, ramp_values(16), pilot=stamp)
    write_run(tmp_path, "L16-qs-s1", staircase(), 16, 1, plateau_values(16))
    bare = ising.select(*ising.find_runs(tmp_path), _data(L=[16], seeds=[1]))
    with pytest.raises(ising.IsingDataError) as refused:
        ising._same_record(bare, named)
    message = str(refused.value)
    assert "L16-ramp-s1 (pilot sha256 None)" in message and "L16-null-s1" not in message


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


def test_a_null_block_without_a_verdict_is_refused(tmp_path: Path) -> None:
    """Review round 1: a fixed run from before #50 carries no thermalised verdict and was read
    as a null block anyway. Nothing then says it is equilibrium, so it is refused."""
    write_run(tmp_path, "L16-ramp-s1", ramp(16), 16, 1, ramp_values(16))
    write_run(tmp_path, "L16-qs-s1", staircase(), 16, 1, plateau_values(16))
    write_run(tmp_path, "L16-null-s1", nulls(16, 500), 16, 1, ramp_values(16), verdicts=False)
    with pytest.raises(ising.IsingDataError, match="no thermalised verdict"):
        ising.select(*ising.find_runs(tmp_path), _data(L=[16], seeds=[1]))


def test_a_staircase_without_the_fss_plateau_is_an_error(tmp_path: Path) -> None:
    """Review round 1: a staircase with no plateau at the static-fit temperature was skipped
    silently, so its L could leave the fit with nothing counted. A plateau that failed its
    check is dropped and counted; a missing one is an error."""
    doc = tomllib.loads(V1.read_text("utf-8"))
    elsewhere = protocol.Staircase(
        [2.0, 2.0], temperatures=(3.0, 2.8), warmup=0, min_therm=50, min_meas=200, saves=10
    )
    missing = ising.open_run(write_run(tmp_path, "L16-qs-s1", elsewhere, 16, 1, plateau_values(16)))
    inputs = ising.Inputs(ramps=[], nulls=[], staircases=[missing], dropped=[])
    rng = np.random.default_rng(0)
    with pytest.raises(ising.IsingDataError, match="no plateau at T"):
        ising._static(doc, {16: inputs}, rng)
    failed = write_run(
        tmp_path, "L16-qs-s2", staircase(), 16, 2, plateau_values(16), failed=(PLATEAU,)
    )
    inputs = ising.Inputs(ramps=[], nulls=[], staircases=[ising.open_run(failed)], dropped=[])
    with pytest.raises(ising.IsingDataError, match="usable at"):  # dropped, so no size is left
        ising._static(doc, {16: inputs}, rng)


def test_the_record_v1_pins_is_the_committed_fixture_and_reads(tmp_path: Path) -> None:
    """#54's record, committed byte for byte as fixtures/ising_pilot.json: v1's data.tau_c pins
    its sha256, and in a folder named as v1's pilot_run it reads through the contract the scorer
    reads tau_c by, current (it measured the running kernel), each size's tau_c and interval as
    recorded, every v1 size among them."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "ising_pilot.json"
    v1 = tomllib.loads(V1.read_text("utf-8"))["data"]
    pinned = v1["tau_c"]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == pinned["pilot_sha256"]
    folder = tmp_path / "ising" / pinned["pilot_run"]
    folder.mkdir(parents=True)
    (folder / "pilot.json").write_bytes(fixture.read_bytes())
    record = ising.read_pilot(tmp_path, pinned["pilot_run"], pinned["pilot_sha256"])
    per_L = json.loads(fixture.read_text("utf-8"))["per_L"]
    assert record.tau_c == {int(L): entry["tau_c"] for L, entry in per_L.items()}
    assert record.tau_c_ci == {int(L): entry["tau_c_ci"] for L, entry in per_L.items()}
    assert set(v1["L"]) < set(record.tau_c)  # L = 128 too, which v1 leaves out
    assert record.stale is None and len(record.kernel_commit.split("-")[0]) == 40


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
