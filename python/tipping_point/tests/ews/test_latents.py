"""The scorer on planted latents tables (#28): the adapter joins a table onto its run's frames and
refuses one it cannot trust; the scorer takes each checkpoint a prereg names through every test; the
slope statistic tells a channel whose steepest change sits at a fixed eps_L from one at a fixed T.

harness/ews_planted.py's write_latents plants the answers beside the v1 dataset's runs (fss mode, L
= 16, 32 and 64, two seeds of two replicas). The prereg is the draft of ising-latent-v1 in
tests/fixtures, set to score them: its placeholders filled, its seeds, replicas and records those of
the planted data, two trainings (a primary and an ablation) instead of eleven, and, to score in
seconds, 40 bootstrap draws, a few surrogates, one horizon, and alarm scores every 200 tau_c."""

import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import ews_v1_golden
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ews_planted import (
    LATENT_EPS,
    LATENT_FIXED_T,
    LATENT_NOISE,
    TWIN_SURPRISE,
    V1,
    V1_EPS_L,
    V1_LAG,
    V1_NULL_EPS,
    V1_PILOT_RUN,
    V1_REPLICAS,
    V1_SEARCH,
    V1_SEEDS,
    V1_SIZES,
    latent_surprise,
    restamp,
    tau_source,
    v1_copy,
    v1_dataset,
    write_latents,
)
from tipping_point.ews import fss, indicators, ising, prereg
from tipping_point.ews.__main__ import main
from tipping_point.ising import io

DRAFT = Path(__file__).resolve().parents[1] / "fixtures" / "ising_latent_draft.toml"
STEPS = 50_000
TRAINED_A, TWIN_A = "train-a-step0050000", "train-a-step0000000"
TRAINED_B, TWIN_B = "train-b-step0050000", "train-b-step0000000"
TABLES = (TRAINED_A, TWIN_A, TRAINED_B, TWIN_B)
SHA256 = {name: hashlib.sha256(name.encode()).hexdigest() for name in TABLES}
CONTROL = {"observable": "pc_2", "indicator": "slope", "extremum": "min", "role": "test"}
RISING = [  # no peak in the band: a spread that keeps growing towards its cold edge, or its hot one
    {"observable": column, "indicator": "variance", "extremum": "max", "role": "test"}
    for column in ("pc_3", "pc_4")
]
CONTROL_READ = [LATENT_FIXED_T * L / V1_SIZES[0] - V1_LAG for L in V1_SIZES]  # half a window late
CONTROL_INVERSE_NU = 1 - np.polyfit(np.log(V1_SIZES), np.log(CONTROL_READ), 1)[0]


def _write_tables(data: V1) -> None:
    """Each checkpoint's table beside every run, under the rule the plan gives its role: a primary
    and its twin keep the Tc plateau and one burst in 8 of a staircase, an ablation the plateau."""
    for run_dir in data.runs.values():
        for name in TABLES:
            rule = "tc-plus" if name in (TRAINED_A, TWIN_A) else "tc-only"
            untrained = name in (TWIN_A, TWIN_B)
            write_latents(run_dir, name, select=rule, untrained=untrained, sha256=SHA256[name])


def _planted_doc(data: V1) -> dict[str, Any]:
    """The draft, set to score `data` as the module docstring says."""
    doc = tomllib.loads(DRAFT.read_text("utf-8"))
    pilot = hashlib.sha256(data.pilot.read_bytes()).hexdigest()
    doc["data"] |= {"seeds": list(V1_SEEDS), "replicas": V1_REPLICAS}
    doc["data"]["tau_c"] |= {"pilot_run": V1_PILOT_RUN, "pilot_sha256": pilot}
    doc["data"]["staircase"]["tau_sha256"] = {str(L): tau_source(L) for L in V1_SIZES}
    doc["baseline"]["result"] = "20261101T000000Z-ising-ews-v1"
    doc["hypotheses"][2]["direction"] = "less"
    doc["latents"] |= {
        "steps": STEPS,
        "columns": ["surprise", "pc_1", "pc_2", "pc_3", "pc_4"],
        "checkpoints": [
            {
                "role": role,
                "model_seed": seed,
                "training": f"train-{letter}",
                "trained_sha256": SHA256[f"train-{letter}-step{STEPS:07d}"],
                "twin_sha256": SHA256[f"train-{letter}-step0000000"],
            }
            for role, seed, letter in (("primary", 1, "a"), ("held-out", 21, "b"))
        ],
    }
    doc["indicators"]["direction"]["level"] = "less"
    doc["indicators"]["direction_by_observable"]["surprise"] = {
        "variance": "greater",
        "level": "less",
    }
    for row in doc["fss"]["static"]:
        if row["role"] == "reference":
            row["tolerance"] = 0.1
    dynamic = doc["fss"]["dynamic"]
    dynamic["statistics"][2]["extremum"] = "min"
    dynamic["statistics"] += [CONTROL, *RISING]
    doc["event"]["horizons"] = [0.0]
    doc["scoring"]["bootstrap"] = 40
    doc["scoring"]["surrogates"] |= {"n": 2, "n_ablation": 1}
    doc["scoring"]["alarms"]["every"] = 200.0
    doc["scoring"]["margins"]["tolerance"] = 0.01
    doc["sensitivity"] |= {"window": [10.0, 20.0], "bandwidth": [5.0, 10.0]}
    doc["trials"]["budget"] = prereg.cells(doc)
    doc["probes"]["test"] = "the staircases of seeds 6-7 at each L: 20000 frames each, same rule"
    doc["disclosure"] |= {"exploratory": "none", "decisions": "none", "confirmatory": "planted"}
    return doc


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> V1:
    planted = v1_dataset(tmp_path_factory.mktemp("latents"), "fss")
    _write_tables(planted)
    return planted


@pytest.fixture(scope="module")
def doc(data: V1) -> dict[str, Any]:
    return _planted_doc(data)


@pytest.fixture(scope="module")
def scored(data: V1, doc: dict[str, Any]) -> ising.Evaluation:
    return ising.evaluate(doc, data.root, np.random.default_rng(28), say=lambda _: None)


def _rows(evaluation: ising.Evaluation, table: str, **where: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in evaluation.tables[table]
        if all(row[key] == value for key, value in where.items())
    ]


def _dynamic(evaluation: ising.Evaluation, table: str, quantity: str) -> dict[str, Any]:
    (row,) = _rows(evaluation, "fss", checkpoint=table, quantity=quantity)
    return row


# ---------------------------------------------------------------- the generator's own checks


@pytest.mark.parametrize(
    ("column", "at"),
    [
        ("surprise", lambda L: LATENT_EPS),  # at one eps_L
        ("pc_2", lambda L: LATENT_FIXED_T * L / V1_SIZES[0]),  # at one T
    ],
)
def test_the_planted_channel_changes_fastest_where_it_was_planted(
    data: V1, column: str, at: Callable[[int], float]
) -> None:
    """From the written table and nothing else: along each ramp the channel, averaged over the
    replicas in bins of 0.1 eps_L, only rises with T up to the noise, and rises fastest within a bin
    or two of where it was planted, inside the search band at every L. A generator that put the
    change elsewhere would leave the scorer tests checking the wrong answer."""
    for L in V1_SIZES:
        run = ising.open_run(data.runs["ramp", L, V1_SEEDS[0]])
        (s,) = ising.series(run, "matched", TRAINED_A)
        eps = run.eps(s.temperature)
        edges = np.arange(V1_SEARCH[1], V1_SEARCH[0] + 0.05, 0.1)
        bins = np.digitize(eps, edges) - 1
        means = np.array([s.observable(column)[:, bins == b].mean() for b in range(len(edges) - 1)])
        rises = np.diff(means)
        steepest = edges[np.argmax(rises) + 1]  # the edge between the two bins it rises across
        assert abs(steepest - at(L)) <= 0.3, (column, L, steepest)
        assert V1_SEARCH[1] < at(L) < V1_SEARCH[0]
        assert rises.min() > -5 * LATENT_NOISE, (column, L, rises.min())


def test_a_twins_surprise_is_flat_and_its_pc_1_reads_m(data: V1) -> None:
    """Catches a twin planted with the trained table's channels, which a margin could not tell from
    its checkpoint."""
    run = ising.open_run(data.runs["ramp", 32, V1_SEEDS[0]])
    (s,) = ising.series(run, "matched", TWIN_A)
    assert np.abs(s.latents["surprise"] - TWIN_SURPRISE).max() < 6 * LATENT_NOISE
    assert np.abs(s.latents["pc_2"]).max() < 6 * LATENT_NOISE
    assert np.abs(s.latents["pc_1"] - s.abs_m).max() < 6 * LATENT_NOISE


# ---------------------------------------------------------------- the join


def test_series_joins_a_tables_columns_onto_the_frames_of_each_block(data: V1) -> None:
    """Each null block gets its own lattices' readouts: the planted surprise at its own eps_L, and
    pc_1 its own replicas' |m|. Catches columns reshaped across blocks or along the wrong axis.
    Everything else is the frames' as they are without the table."""
    run = ising.open_run(data.runs["null", 16, V1_SEEDS[0]])
    plain, joined = ising.series(run, "matched"), ising.series(run, "matched", TRAINED_A)
    assert [len(s.replicas) for s in joined] == [V1_REPLICAS] * len(V1_NULL_EPS)
    for bare, s, eps in zip(plain, joined, V1_NULL_EPS, strict=True):
        assert not bare.latents and set(s.latents) == {
            "surprise",
            *(f"pc_{k}" for k in range(1, 9)),
        }
        for field in ("steps", "temperature", "segment", "abs_m", "energy", "replicas"):
            assert np.array_equal(getattr(bare, field), getattr(s, field)), field
        assert s.latents["surprise"].shape == s.abs_m.shape
        assert s.latents["surprise"].dtype == np.float64  # stored in float32, computed in float64
        assert s.observable("surprise") is s.latents["surprise"]
        assert np.abs(s.latents["surprise"] - latent_surprise(eps)).max() < 6 * LATENT_NOISE
        assert np.abs(s.observable("pc_1") - s.abs_m).max() < 6 * LATENT_NOISE
        stretch = s.take(np.arange(8))
        assert stretch.latents["surprise"].shape == (V1_REPLICAS, 8)
        assert np.array_equal(stretch.latents["pc_1"], s.latents["pc_1"][:, :8])
    with pytest.raises(ValueError, match="observable must be one of"):
        plain[0].observable("surprise")


def test_the_join_holds_every_row_where_the_table_put_it(data: V1, tmp_path: Path) -> None:
    """Row by row against the table itself, on a ramp (one block): (step, replica) in configs order
    lands at (replica, time) in the Series, so a table read in any other order misplaces values."""
    run_dir = shutil.copytree(data.runs["ramp", 16, V1_SEEDS[0]], tmp_path / "run")
    (s,) = ising.series(ising.open_run(run_dir), "matched", TRAINED_A)
    table = pa.concat_tables(
        pq.read_table(p) for p in io.chunks(run_dir / ising.LATENTS, TRAINED_A)
    )
    surprise = table["surprise"].to_numpy().reshape(-1, V1_REPLICAS).T
    assert np.array_equal(s.latents["surprise"], surprise.astype(np.float64))
    assert np.array_equal(s.steps, table["step"].to_numpy().reshape(-1, V1_REPLICAS)[:, 0])


def _tampered(run_dir: Path, name: str, edit: Callable[[pa.Table], pa.Table]) -> None:
    """Rewrite a table's biggest chunk through `edit`, leaving complete.json in place."""
    chunks = io.chunks(run_dir / ising.LATENTS, name)
    path = max(chunks, key=lambda p: pq.read_metadata(p).num_rows)
    table = edit(pq.read_table(path))
    path.unlink()
    pq.write_table(table, path, compression="zstd")


def _set(table: pa.Table, column: str, values: Any) -> pa.Table:
    index = table.column_names.index(column)
    return table.set_column(
        index, table.schema.field(index), pa.array(values, type=table[column].type)
    )


def _shifted_temperature(t: pa.Table) -> pa.Table:
    values = t["temperature"].to_numpy().copy()
    values[0] += 1e-9
    return _set(t, "temperature", values)


def _replicas_swapped(t: pa.Table) -> pa.Table:
    values = t["replica"].to_numpy().copy()
    values[:2] = values[:2][::-1]
    return _set(t, "replica", values)


def _rows_reversed(t: pa.Table) -> pa.Table:
    return t.take(np.arange(t.num_rows)[::-1])


@pytest.mark.parametrize(
    ("fault", "edit"),
    [
        ("a key off by rounding", _shifted_temperature),
        ("two rows' replicas swapped", _replicas_swapped),
        ("rows out of order", _rows_reversed),
        ("a row dropped", lambda t: t.slice(1)),
        ("a whole step dropped", lambda t: t.slice(V1_REPLICAS)),
        ("a key of another type", lambda t: t.set_column(0, "step", t["step"].cast(pa.int32()))),
        ("a key column gone", lambda t: t.drop_columns(["burst"])),
    ],
)
def test_a_key_mismatch_is_refused_naming_the_run_and_the_table(
    data: V1, tmp_path: Path, fault: str, edit: Callable[[pa.Table], pa.Table]
) -> None:
    """Whatever complete.json says, a table whose keys are not exactly its frames' is misaligned or
    short, and no row of it is read. The readings of the frames themselves stay open."""
    run_dir = shutil.copytree(data.runs["ramp", 16, V1_SEEDS[0]], tmp_path / "run")
    run = ising.open_run(run_dir)
    _tampered(run_dir, TRAINED_A, edit)
    with pytest.raises(ising.IsingDataError) as refused:
        ising.series(run, "matched", TRAINED_A)
    assert run.run_id in str(refused.value) and f"latents/{TRAINED_A}" in str(refused.value), fault
    assert ising.series(run, "matched", TWIN_A)[0].latents  # the run's other table is untouched
    assert not ising.series(run, "matched")[0].latents


def test_a_table_is_read_across_its_chunks_and_a_missing_one_is_a_mismatch(
    data: V1, tmp_path: Path
) -> None:
    """A long run's table comes in many chunks, joined in step order. One gone from a finished table
    leaves complete.json in place, so only the key check can see it; with none left there is no
    table at all."""
    run_dir = shutil.copytree(data.runs["ramp", 32, V1_SEEDS[0]], tmp_path / "run")
    run = ising.open_run(run_dir)
    (whole,) = ising.series(run, "matched", TRAINED_A)
    (first,) = io.chunks(run_dir / ising.LATENTS, TRAINED_A)
    table = pq.read_table(first)
    half = table.num_rows // 2 // V1_REPLICAS * V1_REPLICAS
    second = first.with_name(f"{table['step'][half].as_py():020d}.parquet")
    pq.write_table(table.slice(half), second)
    pq.write_table(table.slice(0, half), first)
    (split,) = ising.series(run, "matched", TRAINED_A)
    assert np.array_equal(split.latents["surprise"], whole.latents["surprise"])
    second.unlink()
    with pytest.raises(ising.IsingDataError, match=rf"{run.run_id}: latents/{TRAINED_A}: its"):
        ising.series(run, "matched", TRAINED_A)
    first.unlink()
    with pytest.raises(ising.IsingDataError, match=rf"{run.run_id}: latents/{TRAINED_A}"):
        ising.series(run, "matched", TRAINED_A)


@pytest.mark.parametrize("gone", ["complete.json", "manifest.json", None])
def test_an_incomplete_table_is_refused_naming_the_run_and_the_table(
    data: V1, tmp_path: Path, gone: str | None
) -> None:
    """A table without complete.json is running or crashed; a folder that is not there holds none.
    Both are refused however good its rows look."""
    run_dir = shutil.copytree(data.runs["ramp", 16, V1_SEEDS[0]], tmp_path / "run")
    run = ising.open_run(run_dir)
    name = TRAINED_A
    if gone is None:
        name = "no-such-table"
    else:
        (run_dir / ising.LATENTS / name / gone).unlink()
    with pytest.raises(ising.IsingDataError, match=r"no complete\.json") as refused:
        ising.series(run, "matched", name)
    assert run.run_id in str(refused.value) and f"latents/{name}" in str(refused.value)


def test_a_table_is_read_at_the_matched_cadence_only(data: V1) -> None:
    """Latent columns exist per saved frame: asking for them on the dense cadence is refused rather
    than joined onto rows they never described."""
    run = ising.open_run(data.runs["ramp", 16, V1_SEEDS[0]])
    with pytest.raises(ising.IsingDataError, match=rf"latents/{TRAINED_A} .*matched cadence"):
        ising.series(run, "dense", TRAINED_A)


@pytest.mark.parametrize(
    ("rule", "plateaus", "bursts"),
    [("tc-plus", 2, 5 + 40), ("tc-only", 1, 40), ("all", 2, 80)],
)
def test_a_subsampled_table_reads_as_the_frames_its_rule_keeps(
    data: V1, tmp_path: Path, rule: str, plateaus: int, bursts: int
) -> None:
    """jepa extract --select keeps the Tc plateau whole and one burst in 8 elsewhere (tc-plus), or
    that plateau alone (tc-only): the research pass's key equality with configs holds only against
    the rows the rule keeps. The planted staircase holds 40 bursts at 3.0 and 40 at Tc. The frames
    come back with their columns, row for row, and |m| as configs holds it on those rows."""
    run_dir = shutil.copytree(data.runs["qs", 16, V1_SEEDS[0]], tmp_path / "run")
    shutil.rmtree(run_dir / ising.LATENTS)
    write_latents(run_dir, TRAINED_A, select=rule)
    run = ising.open_run(run_dir)
    (full,), (s,) = ising.series(run, "matched"), ising.series(run, "matched", TRAINED_A)
    assert len(s.steps) == bursts * 8
    assert np.abs(s.latents["pc_1"] - s.abs_m).max() < 6 * LATENT_NOISE
    rows = np.flatnonzero(np.isin(full.steps, s.steps))
    assert np.array_equal(s.abs_m, full.abs_m[:, rows]) and np.array_equal(
        s.steps, full.steps[rows]
    )
    found, dropped = ising.plateaus(run, "matched", TRAINED_A)
    assert len(found) == plateaus and dropped == 0
    assert any(np.isclose(T, 2.269231) for T in found)
    assert sum(len(p.steps) for p in found.values()) == bursts * 8


def test_a_table_holding_other_rows_than_its_rule_keeps_is_refused(
    data: V1, tmp_path: Path
) -> None:
    """A tc-only table whose manifest says all, or a tc-plus table short of one kept burst whose
    whole steps still line up: rows keyed like configs but not the ones the rule keeps."""
    run_dir = shutil.copytree(data.runs["qs", 16, V1_SEEDS[0]], tmp_path / "run")
    shutil.rmtree(run_dir / ising.LATENTS)
    out = write_latents(run_dir, TRAINED_A, select="tc-only")
    run = ising.open_run(run_dir)
    manifest = io.read_json(out / "manifest.json")
    manifest["params"]["selection"]["rule"] = "all"
    io.write_json(out / "manifest.json", manifest, replace=True)
    with pytest.raises(ising.IsingDataError, match="its all selection keeps"):
        ising.series(run, "matched", TRAINED_A)
    out = write_latents(run_dir, TRAINED_B, select="tc-plus")
    _tampered(run_dir, TRAINED_B, lambda t: t.slice(8 * V1_REPLICAS))
    with pytest.raises(ising.IsingDataError, match="its tc-plus selection keeps"):
        ising.series(run, "matched", TRAINED_B)


def test_select_refuses_a_scored_run_missing_a_table(data: V1, tmp_path: Path) -> None:
    """Every scored run needs a complete table per checkpoint, and the error names exactly what is
    missing: an incomplete one on a ramp and an absent one on a null run, of 18 runs and 4 tables.
    The same runs without those tables asked for still select."""
    copy = v1_copy(data, tmp_path / "data")
    ramp, null = copy.runs["ramp", 32, V1_SEEDS[1]], copy.runs["null", 16, V1_SEEDS[0]]
    (ramp / ising.LATENTS / TWIN_B / "complete.json").unlink()
    shutil.rmtree(null / ising.LATENTS / TRAINED_A)
    runs = ising.find_runs(copy.root)
    plan = _planted_doc(copy)["data"]
    with pytest.raises(ising.IsingDataError) as refused:
        ising.select(*runs, plan, TABLES)
    message = str(refused.value)
    assert (
        f"{ramp.name}/latents/{TWIN_B}" in message and f"{null.name}/latents/{TRAINED_A}" in message
    )
    assert message.count("/latents/") == 2
    assert ising.select(*runs, plan, (TWIN_A,))[16].latents is None
    assert ising.select(*runs, plan)[32].ramps  # no latents asked for: v1's call


def test_the_ews_package_does_not_import_jepa() -> None:
    """jepa.extract imports ews.ising, so the other way round would be a cycle: the adapter reads a
    latents table with a plain Parquet reader, and the scorer never needs torch's model code."""
    code = (
        "import sys; import tipping_point.ews.__main__; "
        "print(sorted(m for m in sys.modules if m.startswith('tipping_point.jepa')))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]", done.stdout


# ---------------------------------------------------------------- the scorer, per checkpoint


def test_the_slope_statistic_tells_a_fixed_eps_L_from_a_fixed_T(scored: ising.Evaluation) -> None:
    """The negative control for the dynamic test (#28). Surprise's steepest change sits at one eps_L
    at every L: its rolling slope's extremum gives 1/nu = 1, a pass. pc_2's sits at one T: eps_L
    grows as L, 1/nu near 0, the planted answer, and a fail with its CI below the tolerance band,
    though it too is monotone in T. A peak test without the slope could not tell them apart: both
    have no interior peak."""
    row = _dynamic(scored, TRAINED_A, "inverse_nu_surprise_slope")
    assert row["verdict"] == fss.PASS and row["role"] == "dynamic-test", row
    assert abs(row["estimate"] - 1.0) <= 0.15 and row["lo"] > 0.0, row
    control = _dynamic(scored, TRAINED_A, "inverse_nu_pc_2_slope")
    assert control["verdict"] == fss.FAIL and control["role"] == "dynamic-test", control
    assert abs(control["estimate"] - CONTROL_INVERSE_NU) <= 0.1 and control["hi"] < 0.5, control


def test_the_slope_is_read_half_a_window_after_the_steepest_change(
    scored: ising.Evaluation,
) -> None:
    """At every L: the peaks table's slope rows sit at the planted eps_L less the headline window's
    lag, none on an edge, for the falling direction the statistic names. Reading the window's start,
    or the wrong sign, would put them elsewhere."""
    for L in V1_SIZES:
        (surprise,) = _rows(
            scored, "peaks", checkpoint=TRAINED_A, L=L, observable="surprise", indicator="slope"
        )
        assert surprise["direction"] == "less" and surprise["edge_fraction"] == 0.0, surprise
        assert abs(surprise["peak_eps_median"] - (LATENT_EPS - V1_LAG)) <= 0.1, surprise
        (control,) = _rows(
            scored, "peaks", checkpoint=TRAINED_A, L=L, observable="pc_2", indicator="slope"
        )
        expected = LATENT_FIXED_T * L / V1_SIZES[0] - V1_LAG
        assert control["edge_fraction"] == 0.0, control  # it fails by position, not the edge rule
        assert abs(control["peak_eps_median"] - expected) <= 0.35, control


def test_every_judged_pairs_peak_is_reported_and_only_the_statistics_judged(
    scored: ising.Evaluation,
) -> None:
    """As in v1, AC1's peak is reported in the peaks table and never judged (the research pass's
    table: it peaks at Tc), and a level has no peak row at all. The dynamic rows are exactly the
    file's statistics, with the role each names."""
    rows = _rows(scored, "peaks", checkpoint=TRAINED_A, L=16)
    assert sorted((r["observable"], r["indicator"]) for r in rows) == [
        ("pc_1", "ac"),
        ("pc_1", "variance"),
        ("pc_2", "slope"),
        ("pc_3", "variance"),
        ("pc_4", "variance"),
        ("surprise", "slope"),
        ("surprise", "variance"),
    ]
    dynamic = {
        r["quantity"]: r["role"]
        for r in _rows(scored, "fss", checkpoint=TRAINED_A)
        if r["role"].startswith("dynamic")
    }
    assert dynamic == {
        "inverse_nu_pc_1_variance": "dynamic-pipeline-check",
        "inverse_nu_surprise_variance": "dynamic-test",
        "inverse_nu_surprise_slope": "dynamic-test",
        "inverse_nu_pc_2_slope": "dynamic-test",
        "inverse_nu_pc_3_variance": "dynamic-test",
        "inverse_nu_pc_4_variance": "dynamic-test",
    }


def test_the_dynamic_test_judges_the_statistics_beside_a_stale_key() -> None:
    """Review, round 1: with a v1 fss.dynamic.indicators beside the statistics, _dynamic took the v1
    list, which holds none of their keys, and judged nothing while _peaks read them all. The
    validator refuses that key now, but a doc that never went through it (an exploratory call)
    still has what was read judged, and a peak reported but not listed is still not."""
    doc = tomllib.loads(DRAFT.read_text("utf-8"))
    doc["scoring"]["bootstrap"] = 20
    doc["fss"]["dynamic"]["indicators"] = ["variance", "ac"]
    rng = np.random.default_rng(29)
    at_eps = {L: [0.6 + 0.01 * rng.standard_normal(4) for _ in V1_SEEDS] for L in V1_SIZES}
    judged = ["pc_1_variance", "surprise_variance", "surprise_slope"]
    peaks = {(key, "matched"): at_eps for key in [*judged, "pc_1_ac"]}
    rows = ising._dynamic(doc, peaks, rng)
    assert [r["quantity"] for r in rows] == [f"inverse_nu_{key}" for key in judged]
    assert [r["role"] for r in rows] == ["dynamic-pipeline-check", "dynamic-test", "dynamic-test"]
    assert all(r["verdict"] == fss.PASS for r in rows), rows


def test_the_twins_surprise_is_not_scored_as_scaling(scored: ising.Evaluation) -> None:
    """Hypothesis 5(c) needs a surprise statistic to pass for the trained checkpoint and for none of
    the twins. A flat surprise is noise: its slope's extremum falls anywhere in the band, and the
    verdict is not a pass, while the same table's pc_1 passes the pipeline check, as locality says
    it will."""
    row = _dynamic(scored, TWIN_A, "inverse_nu_surprise_slope")
    assert row["verdict"] != fss.PASS, row
    assert _dynamic(scored, TWIN_A, "inverse_nu_pc_1_variance")["verdict"] == fss.PASS
    assert _dynamic(scored, TRAINED_A, "inverse_nu_pc_1_variance")["verdict"] == fss.PASS


def test_every_row_of_every_table_says_which_checkpoint(
    scored: ising.Evaluation, doc: dict[str, Any]
) -> None:
    """Every table has every checkpoint, none of them "", and the AUC rows are exactly the trials
    prereg.cells counts: a primary checkpoint and its twin on the whole grid, the level pair once
    per window, an ablation and its twin at the headline cell."""
    assert set(scored.tables) == {"auc", "surrogates", "alarms", "peaks", "fss"}
    for name, rows in scored.tables.items():
        assert {row["checkpoint"] for row in rows} == set(TABLES), name
    assert len(scored.tables["auc"]) == prereg.cells(doc) == 2 * 3 * (14 + 4)
    for table in (TRAINED_A, TWIN_A):
        rows = _rows(scored, "auc", checkpoint=table)
        assert len(rows) == 3 * 14  # (3 pairs x 2 windows x 2 bandwidths + level x 2 windows) x 3 L
        assert {r["headline"] for r in rows} == {True, False}
    for table in (TRAINED_B, TWIN_B):
        rows = _rows(scored, "auc", checkpoint=table)
        assert len(rows) == 3 * 4 and all(r["headline"] for r in rows)
        assert {(r["window_taus"]) for r in rows} == {20.0}


def test_level_has_one_cell_per_window_and_no_bandwidth(scored: ising.Evaluation) -> None:
    """The bandwidth is moot for level (Pipeline's bypass): one row per window at each L, its
    bandwidth NaN, the headline window's row the headline cell. Surprise's planted fall reads as a
    falling level, in the direction the file names for it. Variance on the same column keeps its
    two bandwidths."""
    rows = _rows(
        scored, "auc", checkpoint=TRAINED_A, observable="surprise", indicator="level", L=32
    )
    assert sorted(r["window_taus"] for r in rows) == [10.0, 20.0]
    for r in rows:
        assert np.isnan(r["bandwidth_taus"]) and np.isnan(r["bandwidth"]), r
        assert r["headline"] == (r["window_taus"] == 20.0), r
        assert r["direction"] == "less" and r["auc"] < 0.1, r
    twin = _rows(scored, "auc", checkpoint=TWIN_A, observable="surprise", indicator="level")
    assert {r["direction"] for r in twin} == {"less"} and np.mean([r["auc"] for r in twin]) > 0.2
    detrended = _rows(
        scored, "auc", checkpoint=TRAINED_A, observable="surprise", indicator="variance", L=32
    )
    assert {r["bandwidth_taus"] for r in detrended} == {5.0, 10.0}


def test_a_trend_of_pc_1_is_scored_with_the_files_direction(scored: ising.Evaluation) -> None:
    """The direction comes from indicators.direction unless direction_by_observable overrides it:
    pc_1's variance and ac take the file's, surprise's variance its own."""
    rows = _rows(scored, "auc", checkpoint=TRAINED_A, headline=True)
    pairs = {(r["observable"], r["indicator"]): r["direction"] for r in rows}
    assert pairs == {
        ("pc_1", "variance"): "greater",
        ("pc_1", "ac"): "greater",
        ("surprise", "variance"): "greater",
        ("surprise", "level"): "less",
    }


def test_level_rows_carry_the_phase_surrogate_alone(
    scored: ising.Evaluation, doc: dict[str, Any]
) -> None:
    """An AR(1) fitted to a trending raw series draws near-random walks with no trend (call 2), so
    level takes the phase surrogate only and every other pair both. N_s is the file's per role
    (call 7, amended): n for a primary checkpoint and its twin, n_ablation for an ablation's."""
    tests = doc["scoring"]["surrogates"]
    for table, n in (
        (TRAINED_A, tests["n"]),
        (TWIN_A, tests["n"]),
        (TRAINED_B, tests["n_ablation"]),
    ):
        rows = _rows(scored, "surrogates", checkpoint=table)
        assert {r["n_surrogates"] for r in rows} == {n}, table
        assert all(r["headline"] for r in rows)
        for pair in (("pc_1", "variance"), ("pc_1", "ac"), ("surprise", "variance")):
            kinds = {r["surrogate"] for r in rows if (r["observable"], r["indicator"]) == pair}
            assert kinds == {"phase", "ar1"}, pair
        kinds = {r["surrogate"] for r in rows if r["indicator"] == "level"}
        assert kinds == {"phase"}, table


def test_alarms_run_over_the_judged_pairs(scored: ising.Evaluation) -> None:
    """The reported alarm table reads each judged pair at the headline cell, each with its own
    direction, for every checkpoint and L."""
    rows = _rows(scored, "alarms", checkpoint=TRAINED_B)
    assert sorted((r["observable"], r["indicator"], r["L"]) for r in rows) == sorted(
        (o, i, L)
        for o, i in (
            ("pc_1", "variance"),
            ("pc_1", "ac"),
            ("surprise", "variance"),
            ("surprise", "level"),
        )
        for L in V1_SIZES
    )
    assert {r["direction"] for r in rows if r["indicator"] == "level"} == {"less"}


def test_the_static_fits_read_each_tables_frames_at_the_plateau(scored: ising.Evaluation) -> None:
    """Generic <statistic>_<column> rows at the matched cadence, a table's only one. pc_1 holds the
    planted L^2 Var(|m|) ~ L^7/4 and <|m|> ~ L^-1/8, so its pipeline check finds them; a surprise
    held at one level has no scaling at all. The verdicts follow each row's role."""
    rows = {r["quantity"]: r for r in _rows(scored, "fss", checkpoint=TRAINED_A)}
    static = {q: r for q, r in rows.items() if not r["role"].startswith("dynamic")}
    assert set(static) == {
        "l2_var_pc_1",
        "mean_pc_1",
        "l2_var_surprise",
        "mean_surprise",
        "one_minus_ac1_pc_1",
    }
    assert {r["cadence"] for r in static.values()} == {"matched"}
    pipeline = static["l2_var_pc_1"]
    assert pipeline["role"] == "pipeline-check" and pipeline["expected"] == 1.75
    assert abs(pipeline["estimate"] - 1.75) <= 0.15 and pipeline["verdict"] != fss.FAIL, pipeline
    assert static["mean_pc_1"]["estimate"] == pytest.approx(-0.125, abs=0.02)
    assert static["mean_pc_1"]["verdict"] == ""  # descriptive
    assert static["mean_surprise"]["verdict"] == ""
    assert static["l2_var_surprise"]["role"] == "reference"
    assert static["l2_var_surprise"]["estimate"] == pytest.approx(2.0, abs=0.15)  # white noise: L^2
    assert static["one_minus_ac1_pc_1"]["role"] == "reference"


def test_a_prereg_with_latents_refuses_runs_without_its_tables(data: V1, tmp_path: Path) -> None:
    """Nothing is scored until every run holds every table the prereg names."""
    copy = v1_copy(data, tmp_path / "data")
    shutil.rmtree(copy.runs["qs", 64, V1_SEEDS[0]] / ising.LATENTS / TWIN_B)
    with pytest.raises(ising.IsingDataError, match=rf"qs-s{V1_SEEDS[0]}/latents/{TWIN_B}"):
        ising.evaluate(_planted_doc(copy), copy.root, np.random.default_rng(1), say=lambda _: None)


# ---------------------------------------------------------------- frozen, end to end


def _toml(value: Any) -> str:
    """A TOML literal: JSON's for strings, numbers and arrays, an inline table for a table."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {_toml(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(map(_toml, value)) + "]"
    return json.dumps(value)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _commit_prereg(repo: Path, data: V1) -> Path:
    """The planted prereg in a scratch repository, committed, with the headline cell alone: the
    end-to-end tests read the wiring, which every cell shares, and score in half the time."""
    doc = _planted_doc(data)
    doc["sensitivity"] |= {"window": [20.0], "bandwidth": [10.0]}
    doc["trials"]["budget"] = prereg.cells(doc)
    text = "\n".join(f"{key} = {_toml(value)}" for key, value in doc.items()) + "\n"
    assert tomllib.loads(text) == doc
    path = repo / "paper" / "prereg" / "ising-latent-v1.toml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "latent prereg")
    return path


@pytest.fixture(scope="module")
def frozen_latent(data: V1, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, V1, str, str]:
    """A scratch repository where ising-ews-v1 was frozen first and ising-latent-v1 after it, and
    the data as the plan makes it: the runs from v1's tag, the latents tables from the latent one's.
    Returns the prereg's path, the data, and both commits."""
    base = tmp_path_factory.mktemp("frozen-latent")
    repo = base / "repo"
    repo.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Test"),
        ("config", "user.email", "t@example.com"),
        ("config", "core.autocrlf", "false"),
        ("commit", "-q", "--allow-empty", "-m", "v1 frozen here"),
        ("tag", "-a", "prereg/ising-ews-v1", "-m", "v1"),
    ):
        _git(repo, *args)
    v1_commit = _git(repo, "rev-parse", "HEAD")
    path = _commit_prereg(repo, data)
    assert main(["prereg", "freeze", "ising-latent", "--root", str(repo)]) == 0
    commit = _git(repo, "rev-parse", "prereg/ising-latent-v1^{commit}")
    copy = v1_copy(data, base / "data")
    for run_dir in copy.runs.values():
        restamp(run_dir, v1_commit, False)
        for name in TABLES:
            restamp(run_dir / ising.LATENTS / name, commit, False)
    return path, copy, v1_commit, commit


def test_a_frozen_latent_prereg_scores_stamps_and_logs_its_checkpoints(
    frozen_latent: tuple[Path, V1, str, str],
) -> None:
    """The chain on data made as the plan says: score writes one result, every row of every table
    names its checkpoint, the manifest names each table a run was read through by the sha256 of its
    complete.json, and log_trial's line for each AUC cell names its checkpoint, cells() of them."""
    path, data_, _, _ = frozen_latent
    root = data_.root
    assert main(["score", "--prereg", str(path), "--data-dir", str(root), "--cells-only"]) == 0
    (run_dir,) = (root / "ews").glob("*-ising-latent-v1")
    pre = prereg.load(path)
    for name in ("auc", "surrogates", "alarms", "peaks", "fss"):
        rows = pq.read_table(run_dir / name / f"{0:020d}.parquet")
        assert set(rows["checkpoint"].to_pylist()) == set(TABLES), name
    manifest = io.read_json(run_dir / "manifest.json")
    inputs = manifest["params"]["inputs"]
    assert len(inputs) == 18
    for record in inputs:
        (run_dir_,) = [r for r in data_.runs.values() if r.name == record["run_id"]]
        assert record["latents"] == {
            name: hashlib.sha256(
                (run_dir_ / ising.LATENTS / name / "complete.json").read_bytes()
            ).hexdigest()
            for name in TABLES
        }
    lines = prereg.trials_path(root).read_text("utf-8").splitlines()
    logged = [json.loads(line) for line in lines]
    assert len(logged) == prereg.cells(pre.doc) == prereg.trials(root, pre)
    assert {t["cell"]["checkpoint"] for t in logged} == set(TABLES)
    assert all(t["prereg"]["sha256"] == pre.sha256 for t in logged)


@pytest.mark.parametrize(
    ("fault", "named"),
    [
        ("a table from another commit", "latents"),
        ("a table from a dirty tree", "latents"),
        ("a table from another checkpoint", "latents"),
        ("a table read over another state of its run", "latents"),
        ("a run from the latent tag, not v1's", "run"),
    ],
)
def test_a_frozen_latent_prereg_refuses_what_it_does_not_pin(
    frozen_latent: tuple[Path, V1, str, str],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    fault: str,
    named: str,
) -> None:
    """The runs come from v1's tag, each table from the latent tag (the freeze comes before any
    extraction, call 11) and the checkpoint it pins, over the run as it stands. One table or one run
    that is not, and nothing is scored or logged: the error names that one and no other."""
    path, data_, _, commit = frozen_latent
    copy = v1_copy(data_, tmp_path / "data")
    run_dir = copy.runs["ramp", 32, V1_SEEDS[0]]
    out = run_dir / ising.LATENTS / TWIN_A
    manifest = io.read_json(out / "manifest.json")
    if fault == "a table from another commit":
        manifest["producer"]["git_sha"] = "f" * 40
    elif fault == "a table from a dirty tree":
        manifest["producer"]["dirty"] = True
    elif fault == "a table from another checkpoint":
        manifest["params"]["checkpoint"]["sha256"] = "e" * 64
    elif fault == "a table read over another state of its run":
        manifest["params"]["dataset"]["complete_sha256"] = "e" * 64
    else:
        restamp(run_dir, commit, False)
    if named == "latents":
        io.write_json(out / "manifest.json", manifest, replace=True)
    argv = ["score", "--prereg", str(path), "--data-dir", str(copy.root), "--cells-only"]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert run_dir.name in err, err
    assert {r.name for r in copy.runs.values() if r.name in err} == {run_dir.name}, err
    if named == "latents":
        assert f"latents/{TWIN_A}" in err and TRAINED_A not in err, err
    assert not (copy.root / "ews").exists()


def test_a_latent_prereg_is_not_frozen_or_scored_before_its_baseline(
    data: V1, tmp_path: Path
) -> None:
    """With no tag for the version whose runs it reads there is no commit to hold them to: the
    freeze says so, and leaves no tag of its own."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Test"),
        ("config", "user.email", "t@example.com"),
        ("config", "core.autocrlf", "false"),
    ):
        _git(repo, *args)
    _commit_prereg(repo, data)
    with pytest.raises(prereg.PreregError, match="prereg/ising-ews-v1"):
        prereg.freeze(repo, "ising-latent")
    assert not _git(repo, "tag", "--list")


def test_what_the_planted_answers_assume_of_the_draft(doc: dict[str, Any]) -> None:
    """The freeze's ramp window and search band, its tolerance and headline cell: move one and the
    slope tests above read something else."""
    assert doc["data"]["ramps"]["eps_L"] == list(V1_EPS_L)
    assert doc["fss"]["dynamic"]["search"] == list(V1_SEARCH)
    assert doc["fss"]["dynamic"]["expected_inverse_nu"] == 1.0
    assert doc["fss"]["dynamic"]["tolerance"] == 0.5
    assert doc["fss"]["dynamic"]["edge_margin"] == "window"
    assert doc["sensitivity"]["headline"] == {"window": 20.0, "bandwidth": 10.0}


# ---------------------------------------------------------------- a channel with no peak (#115)


Plant = Callable[[int, np.ndarray, np.ndarray], np.ndarray]  # L, a ramp's eps_L, its rows -> rows


def _planted(L: int, run: ising.Run, s: ising.Series, plant: dict[str, Plant]) -> ising.Series:
    eps = run.eps(s.temperature)
    new = {column: rows(L, eps, s.latents[column]) for column, rows in plant.items()}
    return replace(s, latents=s.latents | new)


def _peaks_alone(
    data: V1,
    doc: dict[str, Any],
    table: str,
    seed: int = 115,
    plant: dict[str, Plant] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, float]]:
    """What the edge rule touches of a pass, and nothing else of it: the peaks rows of one table at
    the headline cell, its dynamic rows, and per L what _edge_margin took as one window of ramp.
    `plant` replaces a column's rows on every ramp, once the table is read."""
    chosen = ising.select(*ising.find_runs(data.root), doc["data"], TABLES)
    pilot = ising.read_pilot(data.root, V1_PILOT_RUN)
    head = doc["sensitivity"]["headline"]
    rows, peaks, reach = [], {}, {}
    for L, inputs in chosen.items():
        tau_c = pilot.tau(L)
        band = (
            doc["event"]["segment"] + max(doc["event"]["horizons"]),
            min(doc["event"]["horizons"]),
        )
        prep = ising.prepare(replace(inputs, latents=table), "matched", tau_c, 0.0, band)
        window = prep.samples(head["window"], tau_c)
        bandwidth = head["bandwidth"] * tau_c / prep.sweeps
        if plant:
            prep = replace(prep, ramps=[(r, _planted(L, r, s, plant)) for r, s in prep.ramps])
        found_rows, found = ising._peaks(doc, prep, L, window, bandwidth)
        rows += found_rows
        for key, groups in found.items():
            peaks.setdefault(key, {})[L] = groups
        reach[L] = ising._edge_margin(doc, prep, prep.ramps[0][0], window)
    return rows, ising._dynamic(doc, peaks, np.random.default_rng(seed)), reach


def _without_the_margin(doc: dict[str, Any]) -> dict[str, Any]:
    dynamic = {k: v for k, v in doc["fss"]["dynamic"].items() if k != "edge_margin"}
    return doc | {"fss": doc["fss"] | {"dynamic": dynamic}}


def _edge_share(rows: list[dict[str, Any]], column: str) -> dict[int, float]:
    return {r["L"]: r["edge_fraction"] for r in rows if r["observable"] == column}


@pytest.mark.parametrize(("column", "towards"), [("pc_3", -1.0), ("pc_4", 1.0)])
def test_the_planted_rising_channels_keep_rising_past_the_band(
    data: V1, column: str, towards: float
) -> None:
    """From the written table: at every L the spread, pooled over the replicas in units of eps_L,
    only grows towards the end it is planted at, through the whole search band, and the rolling
    variance has no interior peak to find. A channel that turned inside the band would test
    nothing here."""
    for L in V1_SIZES:
        run = ising.open_run(data.runs["ramp", L, V1_SEEDS[0]])
        (s,) = ising.series(run, "matched", TRAINED_A)
        eps = run.eps(s.temperature)
        edges = np.arange(V1_SEARCH[1], V1_SEARCH[0] + 0.5, 1.0)
        spread = np.array(
            [
                s.observable(column)[:, (eps >= lo) & (eps < hi)].std()
                for lo, hi in itertools.pairwise(edges)
            ]
        )
        steps = np.diff(np.log(spread)) * towards
        assert (steps > 0.1).all() and (steps < 0.4).all(), (column, L, steps)


@pytest.mark.parametrize(("column", "edge"), [("pc_3", V1_SEARCH[1]), ("pc_4", V1_SEARCH[0])])
def test_a_channel_rising_through_the_band_comes_out_inconclusive(
    data: V1, doc: dict[str, Any], scored: ising.Evaluation, column: str, edge: float
) -> None:
    """#115: a spread that keeps growing towards an edge has its maximum a few noisy samples inside
    it, not on it, at one eps_L at every L. Read as v1 reads it, that is located, and it passes with
    1/nu near 1, the fixed-eps_L answer, on a channel with no peak at all: that is the bug this
    guards, shown here on the twin's table, where no replica's noise happens to trip the edge rule.
    Within one headline window of the edge it is not located, more than half of an L's replicas are
    not, and the same statistic on the same table comes out inconclusive, in the full scoring too,
    and for the trained table. The planted channel is the same in both."""
    quantity = f"inverse_nu_{column}_variance"
    rows, dynamic, _ = _peaks_alone(data, doc, TWIN_A)
    (fixed,) = [r for r in dynamic if r["quantity"] == quantity]
    assert fixed["verdict"] == fss.INCONCLUSIVE, fixed
    assert max(_edge_share(rows, column).values()) > 0.5, rows
    assert _dynamic(scored, TWIN_A, quantity)["verdict"] == fss.INCONCLUSIVE
    assert _dynamic(scored, TRAINED_A, quantity)["verdict"] == fss.INCONCLUSIVE
    old_rows, old_dynamic, reach = _peaks_alone(data, _without_the_margin(doc), TWIN_A)
    assert set(reach.values()) == {0.0}
    (bug,) = [r for r in old_dynamic if r["quantity"] == quantity]
    assert bug["verdict"] == fss.PASS and abs(bug["estimate"] - 1.0) < 0.1, bug
    assert max(_edge_share(old_rows, column).values()) <= 0.5, old_rows
    for row in old_rows:
        if row["observable"] == column:  # every L's read just inside that edge
            assert abs(row["peak_eps_median"] - edge) < 0.4, row


def test_a_genuine_peak_near_the_middle_of_the_band_still_passes(
    data: V1, doc: dict[str, Any]
) -> None:
    """The margin takes nothing from a real peak: pc_1's variance peak and surprise's steepest
    change sit at 0.6 eps_L as read, every replica located, the same positions with the margin as
    without it, and both pass with 1/nu near 1. A pc_2 anchored to one T keeps failing: it walks up
    the band with L, still clear of the hot edge, so the margin cannot have made it inconclusive."""
    rows, dynamic, _ = _peaks_alone(data, doc, TRAINED_A)
    old_rows, _, _ = _peaks_alone(data, _without_the_margin(doc), TRAINED_A)
    judged = {r["quantity"]: r for r in dynamic}
    for quantity in ("inverse_nu_pc_1_variance", "inverse_nu_surprise_slope"):
        assert judged[quantity]["verdict"] == fss.PASS, judged[quantity]
        assert abs(judged[quantity]["estimate"] - 1.0) < 0.2, judged[quantity]
    assert judged["inverse_nu_pc_2_slope"]["verdict"] == fss.FAIL
    for row, old in zip(rows, old_rows, strict=True):
        pair = (row["observable"], row["indicator"])
        if pair in (("pc_1", "variance"), ("surprise", "slope"), ("pc_2", "slope")):
            assert row["edge_fraction"] == old["edge_fraction"] == 0.0, row
            assert row["peak_eps_median"] == old["peak_eps_median"], row


# ---------------------------------------------------------------- which way a peak was lost (#121)


def _emptied(L: int, eps: np.ndarray, own: np.ndarray) -> np.ndarray:
    """A bug that emptied the channel: NaN for every replica, along the whole ramp."""
    return np.full_like(own, np.nan)


def _rising_to_the_cold_edge(L: int, eps: np.ndarray, own: np.ndarray) -> np.ndarray:
    """A spread that keeps growing past the cold edge of the band, its variance by e^4 per eps_L
    with nothing to blur it: every replica's maximum sits within a window of that edge."""
    return np.exp(-2.0 * eps) * np.random.default_rng(L).standard_normal(own.shape)


def _steepest_at_the_hot_edge(L: int, eps: np.ndarray, own: np.ndarray) -> np.ndarray:
    """A fall that is steepest at the hot edge of the band and past it: where a slope's minimum goes
    when a channel has no steepest change inside the band."""
    return np.tile(np.exp(eps), (len(own), 1))


def _row(rows: list[dict[str, Any]], column: str, indicator: str, L: int) -> dict[str, Any]:
    (row,) = [
        r for r in rows if (r["observable"], r["indicator"], r["L"]) == (column, indicator, L)
    ]
    return row


def _verdict(dynamic: list[dict[str, Any]], quantity: str) -> dict[str, Any]:
    (row,) = [r for r in dynamic if r["quantity"] == quantity]
    return row


@pytest.mark.parametrize(
    ("plant", "undefined", "edge"),
    [(_emptied, 1.0, 0.0), (_rising_to_the_cold_edge, 0.0, 1.0)],
    ids=["an all-NaN channel", "a channel rising to an edge"],
)
def test_the_peaks_table_tells_an_emptied_channel_from_an_edge_bound_one(
    data: V1, doc: dict[str, Any], plant: Plant, undefined: float, edge: float
) -> None:
    """#121: a bug that empties a channel and a channel with no peak in the band were one number,
    edge_fraction 1.0. A file with edge_margin says which, at every L, every replica counted. Both
    stay inconclusive, as neither can pass; a channel left as planted, beside them, has neither."""
    rows, dynamic, _ = _peaks_alone(data, doc, TWIN_A, plant={"pc_3": plant})
    for L in V1_SIZES:
        row = _row(rows, "pc_3", "variance", L)
        assert (row["undefined"], row["edge_fraction"]) == (undefined, edge), row
        assert row["n"] == len(V1_SEEDS) * V1_REPLICAS, row
        assert np.isnan(row["peak_eps_median"]), row
        other = _row(rows, "pc_1", "variance", L)
        assert (other["undefined"], other["edge_fraction"]) == (0.0, 0.0), other
    assert _verdict(dynamic, "inverse_nu_pc_3_variance")["verdict"] == fss.INCONCLUSIVE


def test_a_channel_with_no_value_in_the_band_is_undefined_whatever_is_outside(
    data: V1, doc: dict[str, Any]
) -> None:
    """A slope is read window by window, so a stretch of NaN frames leaves its row finite on either
    side, unlike a variance, where one NaN empties the row: a row that is NaN only across the whole
    band still has no peak to lose to an edge. peak_position gives it NaN for having no value there,
    and that is the share `undefined` counts."""

    def holed(L: int, eps: np.ndarray, own: np.ndarray) -> np.ndarray:
        return np.where((eps <= V1_SEARCH[0]) & (eps >= V1_SEARCH[1]), np.nan, own)

    rows, dynamic, _ = _peaks_alone(data, doc, TRAINED_A, plant={"surprise": holed})
    for L in V1_SIZES:
        slope = _row(rows, "surprise", "slope", L)
        assert (slope["undefined"], slope["edge_fraction"]) == (1.0, 0.0), slope
    assert _verdict(dynamic, "inverse_nu_surprise_slope")["verdict"] == fss.INCONCLUSIVE
    # it is the band that is empty, not the row, which a test of the whole row would miss
    run = ising.open_run(data.runs["ramp", V1_SIZES[0], V1_SEEDS[0]])
    (s,) = ising.series(run, "matched", TRAINED_A)
    values = indicators.rolling_slope(holed(16, run.eps(s.temperature), s.latents["surprise"]), 10)
    assert not np.isnan(values).all(axis=1).any()


def test_an_undefined_share_counts_against_locating(data: V1, doc: dict[str, Any]) -> None:
    """The verdict reads the peak positions, not the table, so `undefined` taking replicas out of
    edge_fraction takes none out of the edge rule. Six replicas a seed at L = 32: a third emptied, a
    third on an edge, a third with the planted peak, every seed keeping two located so that
    draws_used does not decide it. Neither share is over half, together they are, and it is
    inconclusive. The same rows with a peak in place of the emptied ones pass, so it is that third
    alone that decides."""
    mid = V1_SIZES[1]

    def mixed(gone: Callable[[np.ndarray], np.ndarray]) -> dict[str, Plant]:
        def rows(L: int, eps: np.ndarray, own: np.ndarray) -> np.ndarray:
            if mid != L:
                return own
            return np.vstack([gone(own), _steepest_at_the_hot_edge(L, eps, own), own])

        return {"surprise": rows}

    table, dynamic, _ = _peaks_alone(
        data, doc, TRAINED_A, plant=mixed(lambda own: np.full_like(own, np.nan))
    )
    row = _row(table, "surprise", "slope", mid)
    assert row["n"] == 12, row
    assert row["undefined"] == pytest.approx(1 / 3), row
    assert row["edge_fraction"] == pytest.approx(1 / 3), row
    judged = _verdict(dynamic, "inverse_nu_surprise_slope")
    assert judged["verdict"] == fss.INCONCLUSIVE, judged
    assert abs(judged["estimate"] - 1.0) < 0.2 and judged["draws_used"] >= 0.95, judged
    table, dynamic, _ = _peaks_alone(data, doc, TRAINED_A, plant=mixed(lambda own: own))
    row = _row(table, "surprise", "slope", mid)
    assert row["undefined"] == 0.0 and row["edge_fraction"] == pytest.approx(1 / 3), row
    assert _verdict(dynamic, "inverse_nu_surprise_slope")["verdict"] == fss.PASS


def test_v1_keeps_its_one_edge_fraction(data: V1, doc: dict[str, Any]) -> None:
    """A file with no edge_margin, as ising-ews-v1 is, keeps the table it froze: no `undefined`
    column, and an emptied channel in edge_fraction as before. The golden holds v1's columns but no
    such channel, so it cannot tell a count that moved. The verdict is the same either way."""
    old = _without_the_margin(doc)
    rows, dynamic, _ = _peaks_alone(data, old, TWIN_A, plant={"pc_3": _emptied})
    assert all("undefined" not in row for row in rows), rows[0]
    for L in V1_SIZES:
        assert _row(rows, "pc_3", "variance", L)["edge_fraction"] == 1.0
    assert _verdict(dynamic, "inverse_nu_pc_3_variance")["verdict"] == fss.INCONCLUSIVE


def test_one_headline_window_of_ramp_is_0p4_eps_L_at_every_L(data: V1, doc: dict[str, Any]) -> None:
    """20 tau_c at 50 tau_c per unit of eps_L, whatever L: the number harness/explore/peaks.py read.
    Catches sweeps per unit of T taken for sweeps per unit of eps_L, off by L / Tc. ising-ews-v1's
    file has no edge_margin, so the same function gives it exactly 0, and fss.peak_position its old
    path."""
    _, _, reach = _peaks_alone(data, doc, TRAINED_A)
    assert set(reach) == set(V1_SIZES)
    for L, value in reach.items():
        assert value == pytest.approx(0.4, rel=0.05), L
    v1 = tomllib.loads(ews_v1_golden.V1_TOML.read_text("utf-8"))
    assert "edge_margin" not in v1["fss"]["dynamic"]
    assert ising._edge_margin(v1, None, None, 90) == 0.0  # type: ignore[arg-type]
