"""The experiment runner (harness/ising_experiment.py, #28 slice 5) end to end on planted data: a
frozen file in a scratch repository to the result folder, twice for its bytes; the draft on a
planted null, whose margins must not pass; what it refuses; and the figures, drawn in the isolated
environment."""

import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ews_planted as planted
import ews_v1_golden
import ising_experiment as runner
import numpy as np
import pyarrow.parquet as pq
import pytest
from ews_planted import V1, V1_SEEDS, V1_SIZES
from tipping_point.ews import fss, ising, margins, prereg
from tipping_point.ews.__main__ import main as ews_main
from tipping_point.ews.__main__ import score
from tipping_point.ising import io

TABLES = ("auc", "surrogates", "alarms", "peaks", "fss", "margins", "probes", "curves", "baseline")
FIGURES = ("scaling.png", "peaks.png", "curves.png", "collapse.png", "probes.png")
TRAININGS = planted.experiment_trainings(ablations=2)
STEPS = planted.EXPERIMENT_STEPS
TRAINED = [ising.latents_name(t, STEPS) for _, _, t in TRAININGS]
TWINS = [ising.latents_name(t, 0) for _, _, t in TRAININGS]
CHECKPOINTS = {*TRAINED, *TWINS}
PRIMARY = [ising.latents_name(t, STEPS) for role, _, t in TRAININGS if role == "primary"]
ABLATIONS = [ising.latents_name(t, STEPS) for role, _, t in TRAININGS if role != "primary"]
ABLATION_TWINS = [ising.latents_name(t, 0) for role, _, t in TRAININGS if role != "primary"]


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> V1:
    """The planted runs and the probes' fit runs, with no latents table yet."""
    data = planted.v1_dataset(tmp_path_factory.mktemp("experiment-base"), "fss")
    planted.add_fit_runs(data)
    return data


@dataclass(frozen=True)
class Frozen:
    path: Path  # the prereg, frozen in its scratch repository
    data: V1  # the data, as the plan makes it: runs from v1's tag, tables from the latent one's
    spare: V1  # the same data again, for what must not touch the first
    result: str  # the baseline result both hold


@pytest.fixture(scope="module")
def frozen(base: V1, tmp_path_factory: pytest.TempPathFactory) -> Frozen:
    root = tmp_path_factory.mktemp("experiment-frozen")
    repo = root / "repo"
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
    data = planted.v1_copy(base, root / "data")
    planted.write_experiment_tables(data, TRAININGS)
    result = planted.write_baseline(data, v1_commit)
    path = repo / "paper" / "prereg" / "ising-latent-v1.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        planted.latent_toml(planted.latent_doc(data, TRAININGS, result)),
        encoding="utf-8",
        newline="\n",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "latent prereg")
    prereg.freeze(repo, "ising-latent")
    commit = _git(repo, "rev-parse", "prereg/ising-latent-v1^{commit}")
    for key, run_dir in data.runs.items():
        planted.restamp(run_dir, v1_commit, False)
        if key[0] != "fit":
            for name in CHECKPOINTS:
                planted.restamp(run_dir / ising.LATENTS / name, commit, False)
    spare = _copy(data, root / "spare", result)
    return Frozen(path, data, spare, result)


def _copy(data: V1, root: Path, result: str) -> V1:
    """A data dir with its own runs and tables, and the baseline, before any scoring ran there."""
    copy = planted.v1_copy(data, root)
    shutil.copytree(data.root / "ews" / result, root / "ews" / result)
    return copy


def _run(path: Path, data: V1, *extra: str, embed: Any = None) -> int:
    argv = ["--prereg", str(path), "--data-dir", str(data.root), *extra]
    return runner.main(argv, embed=embed or planted.planted_embedding())


def _results(data: V1) -> list[Path]:
    return sorted((data.root / "ews").glob("*-ising-latent-v1"))


def _rows(run_dir: Path, name: str) -> list[dict[str, Any]]:
    return pq.read_table(run_dir / name / f"{0:020d}.parquet").to_pylist()


@pytest.fixture(scope="module")
def scored(frozen: Frozen) -> Path:
    assert _run(frozen.path, frozen.data, "--no-figures") == 0
    (run_dir,) = _results(frozen.data)
    return run_dir


@pytest.fixture(scope="module")
def again(frozen: Frozen, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same frozen file and seed on its own copy of the data."""
    copy = _copy(frozen.spare, tmp_path_factory.mktemp("experiment-again"), frozen.result)
    assert _run(frozen.path, copy, "--no-figures") == 0
    (run_dir,) = _results(copy)
    return run_dir


@pytest.fixture(scope="module")
def null(base: V1, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, V1, int]:
    """The draft on planted data with nothing learned, through --planted: every trained table and
    embedding is a twin's. The figure step fails, on purpose: see its test."""
    root = tmp_path_factory.mktemp("experiment-null")
    data = planted.v1_copy(base, root / "data")
    trainings = planted.experiment_trainings(ablations=0)
    planted.write_experiment_tables(data, trainings, null=True)
    result = planted.write_baseline(data)
    path = root / "planted.toml"
    path.write_text(
        planted.latent_toml(planted.latent_doc(data, trainings, result)),
        encoding="utf-8",
        newline="\n",
    )
    environment = runner.FIGURE_ENVIRONMENT
    runner.FIGURE_ENVIRONMENT = ("no-such-program-for-figures",)
    try:
        code = _run(path, data, "--planted", embed=planted.planted_embedding(null=True))
    finally:
        runner.FIGURE_ENVIRONMENT = environment
    (run_dir,) = _results(data)
    return run_dir, data, code


# ---------------------------------------------------------------- 1: the tables


def test_every_table_is_written_each_row_with_its_checkpoint(scored: Path) -> None:
    """Every table in Parquet and CSV, listed by the manifest, and `checkpoint` on every row: a
    scorer's and a probe's the table it read, a margin's what it reads, a classic row's ""."""
    manifest = io.read_json(scored / "manifest.json")
    assert set(manifest["tables"]) == set(TABLES)
    assert manifest["params"]["prereg"]["tag"] == "prereg/ising-latent-v1"
    for name in ("auc", "surrogates", "alarms", "peaks", "fss", "probes"):
        assert {r["checkpoint"] for r in _rows(scored, name)} == CHECKPOINTS, name
    assert {r["checkpoint"] for r in _rows(scored, "curves")} == {"", *CHECKPOINTS}
    assert {r["checkpoint"] for r in _rows(scored, "baseline")} == {""}
    for row in _rows(scored, "margins"):
        names = PRIMARY if row["role"] == "primary" else ABLATIONS
        assert row["checkpoint"].split(";") == names, row["quantity"]
        assert len(row["margins"]) == len(row["model_seeds"]) == len(names)
    for name in TABLES:
        lines = (scored / "csv" / f"{name}.csv").read_text("utf-8").splitlines()
        assert len(lines) == 1 + len(_rows(scored, name)), name
        assert "checkpoint" in lines[0].split(",")


def test_the_probe_table_holds_the_files_targets_columns_and_arms(scored: Path) -> None:
    """One row per checkpoint, L, target and column, with its role, arm and pin, on the draws the
    file states. A planted energy is read by the trained arm alone and |m| by both, on the same
    frames."""
    rows = _rows(scored, "probes")
    assert len(rows) == len(CHECKPOINTS) * len(V1_SIZES) * 3 * 3
    keys = {(r["checkpoint"], r["L"], r["target"], r["column"]) for r in rows}
    assert len(keys) == len(rows)
    by = {(r["checkpoint"], r["L"], r["target"], r["column"]): r for r in rows}
    for L in V1_SIZES:
        for name, twin in zip(TRAINED, TWINS, strict=True):
            assert by[name, L, "energy", "raw"]["r2"] > 0.5, (name, L)
            assert by[twin, L, "energy", "raw"]["r2"] < 0.05, (twin, L)
            assert by[name, L, "abs_m", "raw"]["r2"] > 0.9
            assert by[twin, L, "abs_m", "raw"]["r2"] > 0.9
            assert abs(by[name, L, "energy", "iid"]["r2"]) < 0.05
            assert by[name, L, "energy", "raw"]["selectivity"] == (
                by[name, L, "energy", "raw"]["r2"] - by[name, L, "energy", "shuffle_within_T"]["r2"]
            )
    assert {r["n_fit"] for r in rows if r["L"] == 16} == {1280}  # all there are, under 20,000
    assert {r["n_test"] for r in rows if r["L"] == 16} == {2 * 1280}  # the two scored seeds
    roles = {(r["checkpoint"], r["role"], r["arm"], r["sha256"]) for r in rows}
    assert roles == {
        (
            ising.latents_name(t, step),
            role,
            arm,
            planted.checkpoint_sha256(ising.latents_name(t, step)),
        )
        for role, _, t in TRAININGS
        for step, arm in ((STEPS, "trained"), (0, "twin"))
    }


def test_curves_hold_each_column_per_plateau_and_the_classic_chi(scored: Path) -> None:
    """L^2 Var and the mean per plateau: the classic row on every saved frame, a primary's on what
    tc-plus keeps, an ablation's on the Tc plateau alone. pc_1 reads |m| here, so its curve is the
    classic one wherever both have the plateau whole."""
    rows = _rows(scored, "curves")
    classic = {(r["L"], r["temperature"]): r for r in rows if not r["checkpoint"]}
    assert {r["observable"] for r in classic.values()} == {"abs_m"}
    assert len(classic) == len(V1_SIZES) * 2 and {r["frames"] for r in classic.values()} == {
        2 * 2 * 40 * 8
    }
    primary = [r for r in rows if r["checkpoint"] == TRAINED[0]]
    assert {r["observable"] for r in primary} == {"surprise", *(f"pc_{k}" for k in range(1, 9))}
    at_16 = {r["temperature"]: r for r in primary if r["L"] == 16 and r["observable"] == "pc_1"}
    assert set(at_16) == {3.0, 2.269231}
    assert at_16[2.269231]["frames"] == classic[16, 2.269231]["frames"]
    assert at_16[3.0]["frames"] == 2 * 2 * 5 * 8  # one burst in 8
    assert at_16[2.269231]["l2_var"] == pytest.approx(classic[16, 2.269231]["l2_var"], rel=0.05)
    ablation = next(name for name in TRAINED if name not in PRIMARY)
    temperatures = {r["temperature"] for r in rows if r["checkpoint"] == ablation}
    assert temperatures == {2.269231}


def test_the_baseline_rows_are_the_named_results_and_read_nothing_else(scored: Path) -> None:
    """|m|'s kappa and its matched-cadence AUC, never a dense row, and the margins' parity column
    reads them: an indicator v1 has (variance), and one it has not (level)."""
    rows = _rows(scored, "baseline")
    (kappa,) = [r for r in rows if r["quantity"] == "l2_var_abs_m"]
    assert (kappa["estimate"], kappa["lo"], kappa["hi"]) == planted.PLANTED_KAPPA
    assert {r["quantity"] for r in rows} == {"l2_var_abs_m", "auc_variance", "auc_ac"}
    for r in rows:
        if r["quantity"].startswith("auc_"):
            assert r["estimate"] == planted.PLANTED_AUC[r["quantity"].removeprefix("auc_")]
    margins = {r["quantity"]: r for r in _rows(scored, "margins")}
    assert margins["auc_pc_1_variance"]["reference"] == pytest.approx(
        planted.PLANTED_AUC["variance"]
    )
    assert margins["auc_pc_1_ac"]["reference"] == pytest.approx(planted.PLANTED_AUC["ac"])
    assert math.isnan(margins["auc_surprise_level"]["reference"])


def test_a_latent_results_peaks_table_says_undefined_on_every_row(scored: Path) -> None:
    """#121 through the whole run: a column of its own beside edge_fraction, in the Parquet and the
    CSV the figures read, at 0 where nothing is emptied."""
    rows = _rows(scored, "peaks")
    assert rows and all(r["undefined"] == 0.0 for r in rows), rows[0]
    header = (scored / "csv" / "peaks.csv").read_text("utf-8").splitlines()[0].split(",")
    assert header[header.index("edge_fraction") + 1] == "undefined"


# ---------------------------------------------------------------- 2: the bytes


@pytest.mark.parametrize("name", TABLES)
def test_twice_on_the_planted_data_gives_byte_identical_tables(
    scored: Path, again: Path, name: str
) -> None:
    """The same frozen file and seed on another copy of the data: every Parquet and CSV file, to
    the byte. An RNG drawn in another order, a set's order or a clock in a row would show."""
    assert scored != again
    for relative in (Path(name) / f"{0:020d}.parquet", Path("csv") / f"{name}.csv"):
        first, second = (scored / relative).read_bytes(), (again / relative).read_bytes()
        assert first and first == second, relative


LATENT_GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "ews_latent_golden.json"


def _fingerprint(rows: list[dict[str, Any]]) -> str:
    """A table's rows in order, every float to 7 significant digits, hashed: no machine's rounding
    reaches the 7th digit, and a draw moved anywhere moves the CIs after it at the 2nd or 3rd."""

    def rounded(value: Any) -> Any:
        if isinstance(value, list):
            return [rounded(v) for v in value]
        return f"{value + 0.0:.6e}" if isinstance(value, float) else value  # + 0.0: no -0.0

    plain = [{k: rounded(v) for k, v in row.items()} for row in ews_v1_golden.plain(rows)]
    return hashlib.sha256(json.dumps(plain).encode()).hexdigest()


def test_the_planted_tables_are_the_ones_taken_at_5f94ac1(scored: Path) -> None:
    """The golden is this fixture's tables at 5f94ac1, before the scorer saved any part of itself
    (their rows dumped from that tree, then fingerprinted as here). Every table, row for row: saving
    each pass and probe as it ends moved no draw of the one generator nor of any keyed stream."""
    golden = json.loads(LATENT_GOLDEN.read_text("utf-8"))
    assert golden["made_by"] == "5f94ac15cc52065ffa28a09eed186a98ce39e9d2"
    found = {}
    for name in TABLES:
        rows = _rows(scored, name)
        found[name] = {"rows": len(rows), "sha256": _fingerprint(rows)}
    assert found == golden["tables"]


def test_the_fingerprint_moves_with_a_value_and_not_with_rounding(scored: Path) -> None:
    """The control for the golden: a CI bound off by a part in 10^12, rounding between machines, is
    the same fingerprint, and off by a part in 10^5, less than any moved draw, is another."""
    rows = _rows(scored, "auc")
    first = _fingerprint(rows)
    i = next(i for i, row in enumerate(rows) if math.isfinite(row["auc_lo"]) and row["auc_lo"])
    for nudge, same in ((1e-12, True), (1e-5, False)):
        nudged = [dict(row) for row in rows]
        nudged[i]["auc_lo"] *= 1 + nudge
        assert (_fingerprint(nudged) == first) is same, nudge


# ---------------------------------------------------------------- 3: the counter


def test_a_second_scoring_past_the_budget_is_refused_before_it_reads_anything(
    frozen: Frozen, scored: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The budget is the file's cells, all logged by the first scoring: one more cell would pass it,
    so nothing is written and no trial is logged."""
    pre = prereg.load(frozen.path)
    budget = pre.doc["trials"]["budget"]
    assert budget == prereg.cells(pre.doc) == prereg.trials(frozen.data.root, pre)
    assert _run(frozen.path, frozen.data, "--no-figures") == 1
    assert "exceed its budget" in capsys.readouterr().err
    assert _results(frozen.data) == [scored]
    assert prereg.trials(frozen.data.root, pre) == budget
    lines = prereg.trials_path(frozen.data.root).read_text("utf-8").splitlines()
    assert {json.loads(line)["cell"]["checkpoint"] for line in lines} == CHECKPOINTS


def test_the_counter_refuses_the_3133rd_cell(tmp_path: Path) -> None:
    """On the draft itself, whose budget is its cells: all 3,132 of them may be spent, and at 3,132
    logged one more is refused, whichever way the 3,133rd comes."""
    draft = prereg.load(planted.DRAFT)
    assert prereg.cells(draft.doc) == 3132 == draft.doc["trials"]["budget"]
    assert prereg.charge(tmp_path, draft, 3132) == 0
    stamp = {"id": draft.id, "version": draft.version}
    line = json.dumps({"logged_ns": 0, "prereg": stamp, "cell": {"checkpoint": "t"}}) + "\n"
    path = prereg.trials_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(line * 3131, encoding="utf-8")
    assert prereg.charge(tmp_path, draft, 1) == 3131
    with pytest.raises(prereg.PreregError, match="exceed its budget of 3132"):
        prereg.charge(tmp_path, draft, 2)
    prereg.log_trial(tmp_path, {"prereg": stamp, "cell": {"checkpoint": "t"}})
    assert prereg.trials(tmp_path, draft) == 3132
    with pytest.raises(prereg.PreregError, match="3132 trials logged and 1 more"):
        prereg.charge(tmp_path, draft, 1)
    other = prereg.Prereg(draft.path, draft.doc | {"version": 2}, draft.sha256)
    assert prereg.charge(tmp_path, other, 1) == 0  # an amendment has its own count


def test_a_scoring_that_evaluates_other_cells_than_counted_is_refused(
    frozen: Frozen, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget is only as good as cells(): a scorer that evaluates more than it counted would log
    past it unseen. Nothing is written or logged then."""
    pre = prereg.load(frozen.path)
    cells = prereg.cells(pre.doc)
    row = {"auc": 0.5}
    empty: list[Any] = []
    evaluation = ising.Evaluation({"auc": [row] * (cells + 1)}, empty, None, {})  # type: ignore[arg-type]
    monkeypatch.setattr(ising, "evaluate", lambda *args, **kwargs: evaluation)
    with pytest.raises(
        prereg.PreregError, match=f"{cells + 1} AUC cells evaluated, {cells} counted"
    ):
        score(pre, prereg.Frozen("", ""), tmp_path, 1)
    assert not (tmp_path / "ews").exists()


# ---------------------------------------------------------------- 3b: a crash and the counter


KILLED = """
import json, os, sys
from pathlib import Path
from tipping_point.ews import ising, prereg
from tipping_point.ews.__main__ import CELL, score
from tipping_point.ising import io

draft, root, where = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
pre = prereg.load(draft)
row = dict.fromkeys(CELL, "x") | {"auc": 0.5, "auc_lo": 0.0, "auc_hi": 1.0, "n_pos": 1, "n_neg": 1}
pilot = ising.Pilot(root / "pilot" / "pilot.json", "0" * 64, "k", None, {16: 1.0}, {})
evaluation = ising.Evaluation({"auc": [row] * prereg.cells(pre.doc)}, [], pilot, {})
ising.evaluate = lambda *args, **kwargs: evaluation
write_json = io.write_json


def before_complete(path, record, replace=False):
    if path.name == "complete.json":
        os._exit(3)
    write_json(path, record, replace)


def a_torn_write(data_dir, records):
    text = "".join(json.dumps(record, sort_keys=True) + "\\n" for record in records)
    # lines of one length: half of them alone would end on a whole one
    keep = 40 if where == "inside the first record" else len(text) // 2 + 7
    with open(prereg.trials_path(data_dir), "ab") as f:
        f.write(text[:keep].encode())
        f.flush()
        os._exit(3)


if where == "before the trials":
    prereg.log_trials = lambda *args: os._exit(3)
elif where.startswith("inside"):
    prereg.log_trials = a_torn_write
elif where == "after the trials":
    io.write_json = before_complete
score(pre, prereg.Frozen("", ""), root, 1)
"""
SCORED_FILES = (planted.DRAFT, ews_v1_golden.V1_TOML)  # v1 goes through this code too


def _scoring(path: Path, root: Path, where: str) -> subprocess.CompletedProcess[str]:
    """A scoring of `path` in its own process, killed at `where`, or left to finish or refuse."""
    command = [sys.executable, "-c", KILLED, str(path), str(root), where]
    return subprocess.run(command, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("path", SCORED_FILES, ids=lambda path: path.stem)
@pytest.mark.parametrize("where", ["before the trials", "after the trials", "nowhere"])
def test_a_killed_scoring_leaves_no_complete_result_short_of_its_trials(
    tmp_path: Path, path: Path, where: str
) -> None:
    """The whole last step of a real scoring of a file's cells, the draft's 3,132 and v1's 972, in a
    process that dies by os._exit, with nothing flushed for it, at either side of the trial log.
    Killed, it leaves its tables and no complete.json, its trials all logged or none: never a
    complete-looking result with 1,200 of them charged. Unkilled, complete.json comes last, over all
    of them. #120: the second attempt, on either side of the log, is refused as a torn result, not
    by the budget, and adds nothing. With the folder deleted, none logged lets scoring go ahead, to
    a result of its own, while all logged leaves the budget spent, the tables gone with the folder:
    which is why the refusal says deleting gives back no trial."""
    pre = prereg.load(path)
    cells = prereg.cells(pre.doc)
    logged = 0 if where == "before the trials" else cells
    done = _scoring(path, tmp_path, where)
    assert done.returncode == (0 if where == "nowhere" else 3), done.stderr
    results = (tmp_path / "ews").glob
    (run_dir,) = results(f"*-{pre.name}")
    assert (run_dir / "auc" / f"{0:020d}.parquet").is_file()
    assert (run_dir / "complete.json").is_file() == (where == "nowhere")
    assert prereg.trials(tmp_path, pre) == logged
    if logged:
        lines = prereg.trials_path(tmp_path).read_text("utf-8").splitlines()
        assert {json.loads(line)["result"] for line in lines} == {run_dir.name}
    second = _scoring(path, tmp_path, "nowhere")
    assert second.returncode == 1, second.stderr
    assert list(results(f"*-{pre.name}")) == [run_dir] and prereg.trials(tmp_path, pre) == logged
    if where == "nowhere":  # complete, and spent: no torn result to speak of
        assert "exceed its budget" in second.stderr and "torn result" not in second.stderr
        return
    assert "exceed its budget" not in second.stderr
    assert f"{pre.name}: {run_dir.name} in" in second.stderr
    assert "has no complete.json: a torn result from a crash" in second.stderr
    assert "delete it or amend" in second.stderr
    shutil.rmtree(run_dir)
    third = _scoring(path, tmp_path, "nowhere")
    if logged:
        assert third.returncode == 1 and "exceed its budget" in third.stderr, third.stderr
        assert not list(results(f"*-{pre.name}"))
        return
    assert third.returncode == 0, third.stderr
    (again_dir,) = results(f"*-{pre.name}")
    assert (again_dir / "complete.json").is_file()
    assert prereg.trials(tmp_path, pre) == cells


def test_a_kill_inside_the_log_is_the_torn_log_not_the_torn_folder(tmp_path: Path) -> None:
    """Half of the log's one write, as a power cut leaves it, and a folder with no complete.json:
    both refusals could speak, and only #119's repair is right. Some of the folder's trials are
    charged, so deleting it is no way out. It stays refused by the log with the folder gone too."""
    draft = prereg.load(planted.DRAFT)
    done = _scoring(planted.DRAFT, tmp_path, "inside the trials")
    assert done.returncode == 3, done.stderr
    (run_dir,) = (tmp_path / "ews").glob("*-ising-latent-v1")
    assert not (run_dir / "complete.json").exists()
    whole = prereg.trials_path(tmp_path).read_bytes().count(b"\n")
    assert 0 < whole < prereg.cells(draft.doc)
    second = _scoring(planted.DRAFT, tmp_path, "nowhere")
    assert second.returncode == 1, second.stderr
    assert f"{prereg.trials_path(tmp_path)}, line {whole + 1}:" in second.stderr
    assert "count it as spent and amend the pre-registration" in second.stderr
    assert "torn result" not in second.stderr
    shutil.rmtree(run_dir)
    with pytest.raises(prereg.PreregError, match=f"line {whole + 1}:"):
        prereg.charge(tmp_path, draft, 1)


@pytest.mark.parametrize("path", SCORED_FILES, ids=lambda path: path.stem)
def test_the_torn_log_repair_to_the_letter_leaves_no_second_scoring(
    tmp_path: Path, path: Path
) -> None:
    """A kill inside the first record of the one write, so no whole line of this version is in the
    log. Two literal repairs must not add up to a rescoring: the torn log's, then, once that speaks,
    the torn folder's. The record here is the message's own template, filled from the run's
    manifest. Then the folder is deleted as offered, and the budget refuses by itself. The end shows
    what the old repair, cutting the line, let through: 0 trials and nothing torn, so twice the
    evaluations, once logged, with the first result's tables gone."""
    pre = prereg.load(path)
    cells = prereg.cells(pre.doc)
    assert pre.doc["trials"]["budget"] == cells  # one record leaves no room for a whole scoring
    done = _scoring(path, tmp_path, "inside the first record")
    assert done.returncode == 3, done.stderr
    (run_dir,) = (tmp_path / "ews").glob(f"*-{pre.name}")
    log = prereg.trials_path(tmp_path)
    assert b"\n" not in log.read_bytes()
    second = _scoring(path, tmp_path, "nowhere")
    assert second.returncode == 1, second.stderr
    assert f"{log}, line 1:" in second.stderr and "Do not cut the line" in second.stderr
    source = "<params.prereg from the manifest.json of that folder>"
    template = '{"prereg": ' + source + "}"
    assert f"replace this line with one whole record ending in a newline: {template}" in (
        second.stderr
    )
    stamp = io.read_json(run_dir / "manifest.json")["params"]["prereg"]
    log.write_text(template.replace(source, json.dumps(stamp)) + "\n", encoding="utf-8")
    assert prereg.trials(tmp_path, pre) == 1
    third = _scoring(path, tmp_path, "nowhere")
    assert third.returncode == 1 and "delete it or amend" in third.stderr, third.stderr
    shutil.rmtree(run_dir)
    fourth = _scoring(path, tmp_path, "nowhere")
    assert fourth.returncode == 1 and "exceed its budget" in fourth.stderr, fourth.stderr
    assert not list((tmp_path / "ews").glob(f"*-{pre.name}"))
    assert prereg.trials(tmp_path, pre) == 1
    log.write_bytes(b"")
    assert prereg.charge(tmp_path, pre, cells) == 0


def test_a_torn_result_refuses_its_own_file_and_version_alone(tmp_path: Path) -> None:
    """How write_result stamps a folder, its manifest's params.prereg, decides. Another file's,
    though its name ends the same, another version's, v10 among them, and a complete result of this
    one are all no torn result of this file, and neither is a folder killed before its manifest,
    which holds no table. A torn folder of its own is refused by name, and the others not named."""
    draft = prereg.load(planted.DRAFT)

    def result(prereg_id: str, version: int) -> Path:
        stamped = prereg.Prereg(draft.path, {"id": prereg_id, "version": version}, "0" * 64)
        ref = prereg.Frozen("", "")
        return prereg.write_result(tmp_path, stamped, ref, {"auc": [{"L": 16}]}, [], {})

    others = [result(*key) for key in (("x-ising-latent", 1), ("ising-latent", 2),
                                       ("ising-latent", 10))]  # fmt: skip
    for other in others:
        (other / "complete.json").unlink()
    complete = result("ising-latent", 1)
    no_manifest = tmp_path / "ews" / f"no-manifest-{complete.name}"
    no_manifest.mkdir()
    (no_manifest / "manifest.json.tmp").write_bytes(b'{"format"')
    assert prereg.charge(tmp_path, draft, 3132) == 0
    torn = tmp_path / "ews" / f"torn-{complete.name}"
    shutil.copytree(complete, torn)
    (torn / "complete.json").unlink()
    with pytest.raises(prereg.PreregError, match="a torn result from a crash") as refused:
        prereg.charge(tmp_path, draft, 3132)
    assert torn.name in str(refused.value)
    named = [p.name for p in [*others, no_manifest] if p.name in str(refused.value)]
    assert named == []
    assert complete.name not in str(refused.value).replace(torn.name, "")


def test_the_runner_refuses_a_torn_result_before_any_work(
    frozen: Frozen, scored: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A whole result but for its complete.json, its trials unlogged: exit 1 and the refusal, with
    no embedding asked for and nothing logged. The tables were readable, so the next scoring must
    not stand beside them as if they had never been."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    torn = copy.root / "ews" / scored.name
    shutil.copytree(scored, torn)
    (torn / "complete.json").unlink()

    def embed(checkpoint: ising.Checkpoint, sets: Any) -> np.ndarray:
        raise AssertionError("the refusal comes before any work")

    assert _run(frozen.path, copy, "--no-figures", embed=embed) == 1
    err = capsys.readouterr().err
    assert f"{torn.name} in" in err and "a torn result from a crash" in err
    assert not prereg.trials_path(copy.root).exists() and _results(copy) == [torn]


@pytest.mark.parametrize("entry", ["the runner", "ews score"])
def test_a_torn_log_stops_a_scoring_as_a_refusal_not_a_traceback(
    frozen: Frozen, tmp_path: Path, capsys: pytest.CaptureFixture[str], entry: str
) -> None:
    """#119 from the command line: both entries catch PreregError alone, so anything else was a
    traceback at the operator. Exit 1, the file, the line and the repair, and nothing scored."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    prereg.log_trial(copy.root, {"prereg": {"id": "ising-latent", "version": 1}})
    log = prereg.trials_path(copy.root)
    log.write_bytes(log.read_bytes() + b'{"auc": 0.5, "auc_ci": [0.')
    if entry == "the runner":
        code = _run(frozen.path, copy, "--no-figures")
    else:
        argv = ["score", "--prereg", str(frozen.path), "--data-dir", str(copy.root), "--cells-only"]
        code = ews_main(argv)
    assert code == 1
    err = capsys.readouterr().err
    assert f"{log}, line 2:" in err and "count it as spent and amend the pre-registration" in err
    assert not _results(copy)


# ---------------------------------------------------------------- 3c: a killed run, resumed

CLEAN = {"git_sha": "c" * 40, "dirty": False}  # what a resume asks of the code, whatever this is
RUNNER_KILLED = """
import os, sys
sys.path.insert(0, sys.argv[1])
import ews_planted as planted
import ising_experiment as runner
from tipping_point.ews import ising, prereg
from tipping_point.ews.__main__ import main as ews_main
from tipping_point.ising import io

harness, path, data, kill, entry, *extra = sys.argv[1:]
io.code_version = lambda: {"git_sha": "c" * 40, "dirty": False}
save_part, write_json = prereg.save_part, io.write_json


def save_then_die(run_dir, unit, part):
    save_part(run_dir, unit, part)
    if unit == kill:
        os._exit(3)


def before_complete(path, record, replace=False):
    if path.name == "complete.json" and kill == "complete.json":
        os._exit(3)
    write_json(path, record, replace)


prereg.save_part = ising.save_part = save_then_die
io.write_json = before_complete
argv = ["--prereg", path, "--data-dir", data, *extra]
if entry == "the runner":
    sys.exit(runner.main([*argv, "--no-figures"], embed=planted.planted_embedding()))
sys.exit(ews_main(["score", *argv, "--cells-only"]))
"""
LAST_PASS = f"pass-{len(CHECKPOINTS) - 1:02d}"
UNITS = len(CHECKPOINTS) * (len(V1_SIZES) + 1)  # a probe per L and checkpoint, and its pass


def _killed_run(
    path: Path, data: V1, kill: str, *extra: str, entry: str = "the runner"
) -> subprocess.CompletedProcess[str]:
    """A scoring by the runner, or by `ews score --cells-only`, in its own process and from a clean
    checkout, killed by os._exit once it has saved the unit `kill` (pass-<i>, probes-<L>-<j>), or
    as it is about to write complete.json, or let end: "nowhere"."""
    harness = Path(runner.__file__).parent
    command = [sys.executable, "-c", RUNNER_KILLED, str(harness), str(path), str(data.root), kill,
               entry, *extra]  # fmt: skip
    return subprocess.run(command, capture_output=True, text=True, timeout=900)


@pytest.fixture
def clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every result here is begun and carried on from one clean checkout, the only kind a resume
    trusts, whatever state the worktree running the tests is in."""
    monkeypatch.setattr(io, "code_version", lambda: dict(CLEAN))


class _Killed(Exception):
    """What stops a run in this process where a crash would."""


def _crash(path: Path, data: V1, unit: str, *extra: str) -> Path:
    """The runner in this process, stopped as soon as it saves `unit`: the torn folder it leaves."""
    save = prereg.save_part

    def save_then_stop(run_dir: Path | None, name: str, part: Any) -> None:
        save(run_dir, name, part)
        if name == unit:
            raise _Killed

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(prereg, "save_part", save_then_stop)
        patch.setattr(ising, "save_part", save_then_stop)
        with pytest.raises(_Killed):
            _run(path, data, "--no-figures", *extra)
    (run_dir,) = _results(data)
    assert not (run_dir / "complete.json").exists()
    return run_dir


def _ews_files(data: V1) -> dict[str, bytes]:
    """Every file under <data dir>/ews, the trial log among them: what a refusal leaves alone."""
    root = data.root / "ews"
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


RESUMES = {
    "during the probes": [f"probes-{V1_SIZES[1]}-06"],
    "after pass 1": ["pass-00"],
    "after the last pass": [LAST_PASS],
    "between the log and complete.json": ["complete.json"],
    "killed twice": [f"probes-{V1_SIZES[0]}-03", "pass-08"],
}


@pytest.mark.parametrize("kills", RESUMES.values(), ids=RESUMES.keys())
def test_a_killed_run_resumed_writes_the_bytes_of_one_never_killed(
    frozen: Frozen, scored: Path, tmp_path: Path, kills: list[str]
) -> None:
    """The runner killed by os._exit, nothing flushed for it: during the probes, after the first
    pass, after the last, between the trial log and complete.json, and twice. Each time the same
    command runs again with --resume (the twice-killed run passes it from its first start, where it
    finds nothing to carry on, says so and begins a result of its own), with a part torn as saved
    left beside the rest. It ends in the result the unbroken run wrote, every Parquet and CSV file
    to the byte. What was saved before a kill is read back, never computed or written again, and
    every unit is saved by the end; the log holds the budget, each line naming this folder, in one
    write, logged once even where the kill came after the log; the manifest records each start."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    pre = prereg.load(frozen.path)
    budget = pre.doc["trials"]["budget"]
    first = ["--resume"] if len(kills) > 1 else []
    for i, kill in enumerate(kills):
        done = _killed_run(frozen.path, copy, kill, *(["--resume"] if i else first))
        assert done.returncode == 3, done.stderr
        fresh = "--resume: no torn result folder found, starting a fresh run" in done.stderr
        assert fresh == (i == 0 and bool(first)), done.stderr
        assert [(r / "complete.json").exists() for r in _results(copy)] == [False]
    (run_dir,) = _results(copy)
    parts = run_dir / "parts"
    saved = {p.name: p.read_bytes() for p in parts.iterdir()}
    assert kills[-1] == "complete.json" or f"{kills[-1]}.pkl" in saved
    (parts / "pass-01.pkl.tmp").write_bytes(b"half a part")
    assert prereg.trials(copy.root, pre) == (budget if kills[-1] == "complete.json" else 0)
    done = _killed_run(frozen.path, copy, "nowhere", "--resume")
    assert done.returncode == 0, done.stderr
    assert _results(copy) == [run_dir] and (run_dir / "complete.json").is_file()
    for name in TABLES:
        for relative in (Path(name) / f"{0:020d}.parquet", Path("csv") / f"{name}.csv"):
            assert (run_dir / relative).read_bytes() == (scored / relative).read_bytes(), relative
    assert {name: (parts / name).read_bytes() for name in saved} == saved
    assert len(list(parts.glob("*.pkl"))) == UNITS
    log = prereg.trials_path(copy.root).read_text("utf-8").splitlines()
    lines = [json.loads(line) for line in log]
    assert len(lines) == budget == prereg.trials(copy.root, pre)
    assert {line["result"] for line in lines} == {run_dir.name}
    assert len({line["logged_ns"] for line in lines}) == 1
    manifest = io.read_json(run_dir / "manifest.json")
    starts = manifest["resumptions"]
    assert [{k: s[k] for k in CLEAN} for s in starts] == [CLEAN] * (len(kills) + 1)
    assert [s["started_ns"] for s in starts] == sorted(s["started_ns"] for s in starts)
    assert manifest["tables"] == io.read_json(scored / "manifest.json")["tables"]


def test_ews_score_resumes_a_killed_scoring_to_the_same_cell_tables(
    frozen: Frozen, scored: Path, tmp_path: Path
) -> None:
    """The other entry, `ews score --cells-only`, killed after its sixth pass and run again with
    --resume: its five tables are the runner's to the byte, its trials logged once. Its first
    start passes --resume too, finds nothing to carry on and says so."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    done = _killed_run(frozen.path, copy, "pass-05", "--resume", entry="ews score")
    assert done.returncode == 3, done.stderr
    assert "--resume: no torn result folder found, starting a fresh run" in done.stderr
    (run_dir,) = _results(copy)
    done = _killed_run(frozen.path, copy, "nowhere", "--resume", entry="ews score")
    assert done.returncode == 0, done.stderr
    assert _results(copy) == [run_dir] and (run_dir / "complete.json").is_file()
    for name in ("auc", "surrogates", "alarms", "peaks", "fss"):
        relative = Path(name) / f"{0:020d}.parquet"
        assert (run_dir / relative).read_bytes() == (scored / relative).read_bytes(), name
    pre = prereg.load(frozen.path)
    budget = pre.doc["trials"]["budget"]
    assert prereg.trials(copy.root, pre) == prereg.trials(copy.root, pre, run_dir.name) == budget
    assert len(io.read_json(run_dir / "manifest.json")["resumptions"]) == 2


REFUSED = {  # what changed since the crash: what the refusal says
    "seed": "seed: 26 when it began, 27 now",
    "a latents table": "inputs: the runs ",
    "a run": "latents tables not made by",
    "a fit run": "fit_runs: {",
    "a dirty checkout": "code: this checkout has changes not committed",
    "another commit": f"code: '{'c' * 40}' when it began, '{'d' * 40}' now",
    "the prereg stamp": "prereg: {",
    "the device": "'cuda:7' now",
    "the other entry": "device: ",
}


@pytest.mark.parametrize(("change", "said"), REFUSED.items(), ids=REFUSED.keys())
def test_a_resume_under_another_key_is_refused_and_touches_nothing(
    frozen: Frozen,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    clean: None,
    change: str,
    said: str,
) -> None:
    """--resume carries a torn result on only for a scoring that would end in the bytes it began to
    write: the same seed, inputs and fit runs, file and stamp, code and embedding, and the same
    entry. Anything else is refused before any work, saying what differs, and nothing is written or
    logged. A scored run whose complete.json changed is refused before its key is read, by the
    latents tables made over the run as it was (and the baseline read it as it was): the key holds
    it too, for v1. A fit run has no latents table, so the key alone refuses it."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    planted_ = ["--planted"] if change == "the prereg stamp" else []
    run_dir = _crash(frozen.path, copy, f"probes-{V1_SIZES[0]}-00", *planted_)
    resume = ["--resume"]
    ramp = copy.runs["ramp", V1_SIZES[0], V1_SEEDS[0]]
    if change == "seed":
        resume += ["--seed", "27"]
    elif change in ("a latents table", "a run"):
        folder = ramp / ising.LATENTS / TRAINED[0] if change == "a latents table" else ramp
        done = io.read_json(folder / "complete.json") | {"ended_ns": 1}
        io.write_json(folder / "complete.json", done, replace=True)  # a new file: links stay apart
    elif change == "a fit run":  # the staircase the probes fit on, read by no scored cell
        fit = copy.runs["fit", V1_SIZES[0], planted.FIT_SEED]
        done = io.read_json(fit / "complete.json") | {"ended_ns": 1}
        io.write_json(fit / "complete.json", done, replace=True)
    elif change == "a dirty checkout":
        monkeypatch.setattr(io, "code_version", lambda: CLEAN | {"dirty": True})
    elif change == "another commit":
        monkeypatch.setattr(io, "code_version", lambda: CLEAN | {"git_sha": "d" * 40})
    elif change == "the device":
        resume += ["--device", "cuda:7"]
    before = _ews_files(copy)
    if change == "the other entry":
        argv = ["score", "--prereg", str(frozen.path), "--data-dir", str(copy.root), "--cells-only"]
        code = ews_main([*argv, "--resume"])
    else:
        code = _run(frozen.path, copy, "--no-figures", *resume)
    assert code == 1
    err = capsys.readouterr().err
    assert said in err, err
    if change == "a run":
        assert ramp.name in err, err
    else:
        assert f"{run_dir.name} in" in err and "is not carried on, since it began otherwise" in err
    if change == "a latents table":  # the run named, the one part that differs
        assert f"scoring writes: inputs: the runs {ramp.name}. Score" in err, err
    assert _ews_files(copy) == before and not prereg.trials_path(copy.root).exists()


@pytest.mark.parametrize("entry", ["the runner", "ews score"])
def test_without_resume_a_killed_runs_folder_is_still_refused_as_torn(
    frozen: Frozen, tmp_path: Path, capsys: pytest.CaptureFixture[str], clean: None, entry: str
) -> None:
    """The torn rule as it was: a folder a crash left, its parts saved, refuses a scoring that does
    not ask to carry it on, whichever entry, and nothing is written or logged. The refusal offers
    --resume before deletion, which would throw away every part saved."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    run_dir = _crash(frozen.path, copy, f"probes-{V1_SIZES[0]}-00")
    before = _ews_files(copy)
    if entry == "the runner":
        code = _run(frozen.path, copy, "--no-figures")
    else:
        argv = ["score", "--prereg", str(frozen.path), "--data-dir", str(copy.root), "--cells-only"]
        code = ews_main(argv)
    assert code == 1
    err = capsys.readouterr().err
    assert f"ising-latent-v1: {run_dir.name} in" in err, err
    assert "has no complete.json: a torn result from a crash" in err
    assert "carry it on with --resume" in err and "or delete it or amend" in err
    assert _ews_files(copy) == before


def test_a_resume_finding_some_of_its_trials_logged_is_refused(
    frozen: Frozen, tmp_path: Path, capsys: pytest.CaptureFixture[str], clean: None
) -> None:
    """A folder's own lines in the log are none, and it logs them, or all, from a kill between the
    log and complete.json, and it logs nothing. Some alone is a torn log's repair, which spends the
    folder: refused, nothing written. A torn line not yet repaired is refused by trials itself."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    run_dir = _crash(frozen.path, copy, f"probes-{V1_SIZES[0]}-00")
    stamp = io.read_json(run_dir / "manifest.json")["params"]["prereg"]
    own = {"prereg": stamp, "result": run_dir.name, "cell": {"checkpoint": TRAINED[0]}}
    log = prereg.trials_path(copy.root)
    records = [own, own, own, {"prereg": stamp}]  # the last as the repair writes it: no result
    log.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    before = _ews_files(copy)
    assert _run(frozen.path, copy, "--no-figures", "--resume") == 1
    cells = prereg.cells(prereg.load(frozen.path).doc)
    assert f"{run_dir.name} has 3 of its {cells} trials logged" in capsys.readouterr().err
    assert _ews_files(copy) == before
    log.write_bytes(log.read_bytes() + b'{"prereg": {"id": "ising-lat')
    before = _ews_files(copy)
    assert _run(frozen.path, copy, "--no-figures", "--resume") == 1
    assert f"{log}, line 5:" in capsys.readouterr().err
    assert _ews_files(copy) == before


def test_charge_takes_off_its_own_folder_and_its_own_trials_alone(tmp_path: Path) -> None:
    """The ledger's rule in charge, on the draft (budget = cells): the folder a scoring carries on
    is no torn result to it, and every line it logged before a kill is its own and off the count,
    where without `own` it is refused as torn. Another result's line still counts, some of its own
    lines alone are refused, and with two torn folders there is none to carry on."""
    draft = prereg.load(planted.DRAFT)
    cells, ref = prereg.cells(draft.doc), prereg.Frozen("", "")
    torn = prereg.begin_result(tmp_path, draft, ref, {})
    with pytest.raises(prereg.PreregError, match="a torn result from a crash"):
        prereg.charge(tmp_path, draft, cells)
    assert prereg.resumable(tmp_path, draft) == torn.name
    assert prereg.charge(tmp_path, draft, cells, torn.name) == 0
    stamp = {"id": draft.id, "version": draft.version}
    prereg.log_trials(tmp_path, [{"prereg": stamp, "result": torn.name}] * cells)
    assert prereg.trials(tmp_path, draft) == prereg.trials(tmp_path, draft, torn.name) == cells
    assert prereg.charge(tmp_path, draft, cells, torn.name) == 0
    with pytest.raises(prereg.PreregError, match="a torn result from a crash"):
        prereg.charge(tmp_path, draft, cells)
    prereg.log_trial(tmp_path, {"prereg": stamp, "result": "another"})
    with pytest.raises(prereg.PreregError, match=f"1 trials logged and {cells} more would exceed"):
        prereg.charge(tmp_path, draft, cells, torn.name)
    prereg.trials_path(tmp_path).write_text(
        json.dumps({"prereg": stamp, "result": torn.name}) + "\n", encoding="utf-8"
    )
    with pytest.raises(prereg.PreregError, match=f"has 1 of its {cells} trials logged"):
        prereg.charge(tmp_path, draft, cells, torn.name)
    shutil.copytree(torn, tmp_path / "ews" / f"another-{torn.name}")
    assert prereg.resumable(tmp_path, draft) is None
    with pytest.raises(prereg.PreregError, match=f"another-{torn.name} in"):
        prereg.charge(tmp_path, draft, cells, torn.name)


def test_the_torn_refusal_offers_resume_only_when_one_torn_folder_is_there(tmp_path: Path) -> None:
    """With one torn folder the refusal offers --resume; with two, resumable finds none, so --resume
    would only repeat the refusal: it is not offered, and the refusal names every torn folder as
    the ones of which only one can be carried on, whether or not a scoring carries one on."""
    draft = prereg.load(planted.DRAFT)
    cells, ref = prereg.cells(draft.doc), prereg.Frozen("", "")
    torn = prereg.begin_result(tmp_path, draft, ref, {})
    with pytest.raises(prereg.PreregError) as one:
        prereg.charge(tmp_path, draft, cells)
    assert "carry it on with --resume" in str(one.value)
    assert "Only one torn folder" not in str(one.value)
    other = tmp_path / "ews" / f"another-{torn.name}"
    shutil.copytree(torn, other)
    names = ", ".join(sorted([torn.name, other.name]))
    for own in (None, torn.name):
        with pytest.raises(prereg.PreregError) as two:
            prereg.charge(tmp_path, draft, cells, own)
        assert "--resume, which" not in str(two.value), two.value
        assert f"Only one torn folder can be carried on: {names}" in str(two.value), two.value


def test_a_new_result_from_a_dirty_checkout_warns_it_cannot_be_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run begun from a checkout no commit names is refused --resume later (the code part of its
    key is None), so begin_result says so on stderr as it starts, and only then: not from a clean
    checkout, not for a result with no key (written whole, nothing to resume), not for planted
    data (an unfrozen ref), and not on a resume, which refuses instead."""
    warning = "warning: this checkout has uncommitted changes, so a crash in this run cannot be"
    draft, real = prereg.load(planted.DRAFT), prereg.Frozen("prereg/x-v1", "a" * 40)
    key = {"seed": 26}
    for dirty in (True, None):  # None: nothing names the commit, which a resume refuses alike
        monkeypatch.setattr(io, "code_version", lambda d=dirty: CLEAN | {"dirty": d})
        prereg.begin_result(tmp_path / f"dirty-{dirty}", draft, real, key)
        assert warning in capsys.readouterr().err
    for ref, k in ((real, {}), (prereg.Frozen("", ""), key)):
        prereg.begin_result(tmp_path / f"quiet-{bool(k)}", draft, ref, k)
        assert capsys.readouterr().err == ""
    monkeypatch.setattr(io, "code_version", lambda: dict(CLEAN))
    run_dir = prereg.begin_result(tmp_path / "clean", draft, real, key)
    assert capsys.readouterr().err == ""
    monkeypatch.setattr(io, "code_version", lambda: CLEAN | {"dirty": True})
    with pytest.raises(prereg.PreregError, match="changes not committed"):
        prereg.begin_result(tmp_path / "clean", draft, real, key, run_dir.name)
    assert capsys.readouterr().err == ""


def test_a_result_is_carried_on_under_its_own_key_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """begin_result's manifest holds the stamp, the key with the commit of a clean checkout, and one
    record per start. Carried on under another key it is refused, by what differs (the runs named,
    not their digests), and so is a dirty checkout, however the rest matches; nothing is written by
    a refusal. Under its own key it gets one more record."""
    monkeypatch.setattr(io, "code_version", lambda: dict(CLEAN))
    draft, ref = prereg.load(planted.DRAFT), prereg.Frozen("", "")
    inputs = {r: {"complete_sha256": r * 32, "latents": {"t": "e" * 64}} for r in ("a1", "b2")}
    key = {"seed": 26, "inputs": inputs}
    run_dir = prereg.begin_result(tmp_path, draft, ref, key)
    manifest = io.read_json(run_dir / "manifest.json")
    assert manifest["resume"] == key | {"code": CLEAN["git_sha"]}
    assert manifest["params"] == {"prereg": prereg.stamp(draft, ref)} and manifest["tables"] == []
    (start,) = manifest["resumptions"]
    assert list(start) == ["started_ns", "host", "git_sha", "dirty", "pid"]
    changed = inputs | {"b2": inputs["b2"] | {"complete_sha256": "f" * 64}}
    for other, said in (
        (key | {"inputs": changed}, "inputs: the runs b2. Score it as it began"),
        (key | {"seed": 27}, "seed: 26 when it began, 27 now"),
        (key | {"numpy": "0"}, "numpy: None when it began, '0' now"),
    ):
        with pytest.raises(prereg.PreregError, match=re.escape(said)):
            prereg.begin_result(tmp_path, draft, ref, other, run_dir.name)
    monkeypatch.setattr(io, "code_version", lambda: CLEAN | {"dirty": True})
    with pytest.raises(prereg.PreregError, match="code: this checkout has changes not committed"):
        prereg.begin_result(tmp_path, draft, ref, key, run_dir.name)
    assert io.read_json(run_dir / "manifest.json") == manifest
    monkeypatch.setattr(io, "code_version", lambda: dict(CLEAN))
    assert prereg.begin_result(tmp_path, draft, ref, key, run_dir.name) == run_dir
    assert len(io.read_json(run_dir / "manifest.json")["resumptions"]) == 2


def test_the_generator_state_a_part_keeps_continues_its_stream_exactly(tmp_path: Path) -> None:
    """A pass's part keeps rng.bit_generator.state and a resumed scoring sets it back: the stream
    goes on exactly where it stopped, with the half of a 64-bit word a 32-bit draw leaves buffered.
    Through a JSON round trip, and through the part file the scorer writes. The control: without
    the buffered half, the next 32-bit draws differ."""
    rng = np.random.default_rng(26)
    rng.random(3)
    rng.integers(0, 2**32, size=3, dtype=np.uint32)  # an odd count leaves half a word
    state = rng.bit_generator.state
    assert state["has_uint32"] == 1
    prereg.save_part(tmp_path, "state", state)

    def drawn(generator: np.random.Generator) -> list[np.ndarray]:
        return [
            generator.integers(0, 2**32, size=5, dtype=np.uint32),
            generator.random(5),
            generator.standard_normal(5),
            generator.integers(0, 10, size=5),
        ]

    want = drawn(rng)
    for kept in (json.loads(json.dumps(state)), prereg.load_part(tmp_path, "state")):
        other = np.random.default_rng(0)
        other.bit_generator.state = kept
        assert all(np.array_equal(a, b) for a, b in zip(drawn(other), want, strict=True))
    lost = np.random.default_rng(0)
    lost.bit_generator.state = {**state, "has_uint32": 0}
    assert not np.array_equal(lost.integers(0, 2**32, size=5, dtype=np.uint32), want[0])


# ---------------------------------------------------------------- 4: the margins


def test_two_checkpoints_on_one_embedding_draw_the_same_resamples(frozen: Frozen) -> None:
    """#122, the runner's side of what margins.paired asks of its draws: column b is one resample of
    the data for every checkpoint of a quantity, so that within a column a trained-twin difference
    is the model's alone. Two checkpoints given one embedding in one probe_tables call have equal
    draws, to the bit, for every target, quantity and L. A third, with its own embedding, has not,
    and every set of draws spreads, so equal is not degenerate. A stream per checkpoint would move
    every margin's CI, and nothing else here would go red."""
    doc = prereg.load(frozen.path).doc
    admitted = ising.admit(doc, frozen.spare.root)
    first, second, third = admitted.passes[:3]
    embedding = planted.planted_embedding()

    def embed(checkpoint: ising.Checkpoint, sets: Any) -> np.ndarray:
        return embedding(first if checkpoint == second else checkpoint, sets)

    _, draws = runner.probe_tables(
        doc, frozen.spare.root, admitted.scored, (first, second, third), embed, 26
    )
    targets = doc["probes"]["targets"]
    assert len(draws) == len(targets) * 2 * len(V1_SIZES) * 3
    for target in targets:
        for quantity in ("r2", "selectivity"):
            for L in V1_SIZES:
                key = (target, quantity, L)
                one, two, other = (draws[(*key, c.name)] for c in (first, second, third))
                assert one.shape == (doc["scoring"]["bootstrap"],) and one.std() > 0, key
                assert np.array_equal(one, two), key
                assert not np.array_equal(one, other), key


def test_under_every_size_the_sign_test_reads_the_per_L_rows_the_runner_emits(
    frozen: Frozen,
) -> None:
    """scoring.margins.sizes = "every" (#28) through the runner's own path: probe_tables' rows and
    draws, then probe_quantities, then margins.table. Every planted energy margin is positive, so
    both judged rows pass. Give one primary seed's energy R^2 at one L back to its twin, values and
    draws swapped there, and that row's sign test fails under "every" but not under "mean", with
    the same margins and CI: the per-L quantities probe_quantities emits are the ones the sign test
    reads. Named otherwise, "every" would find no per-L margin and pass on the mean alone."""
    doc = copy.deepcopy(prereg.load(frozen.path).doc)
    for statistic in doc["fss"]["dynamic"]["statistics"]:
        statistic["role"] = "descriptive"  # as ising-latent-v1 has it: no surprise rule to feed
    admitted = ising.admit(doc, frozen.spare.root)
    rows, draws = runner.probe_tables(
        doc, frozen.spare.root, admitted.scored, admitted.passes, planted.planted_embedding(), 26
    )
    emitted = runner.probe_quantities(doc, rows, draws)
    assert {q.scope for q in emitted if q.name == "energy_r2"} == {
        *(f"L={L}" for L in V1_SIZES),
        margins.ALL,
    }
    L, seed = V1_SIZES[1], len(PRIMARY) - 1
    trained, twin = TRAINED[seed], TWINS[seed]
    turned = [dict(r) for r in rows]
    pair = [
        r
        for r in turned
        if (r["target"], r["column"], r["L"]) == ("energy", "raw", L)
        and r["checkpoint"] in (trained, twin)
    ]
    assert len(pair) == 2
    pair[0]["r2"], pair[1]["r2"] = pair[1]["r2"], pair[0]["r2"]
    swapped = dict(draws)
    mine, its = ("energy", "r2", L, trained), ("energy", "r2", L, twin)
    swapped[mine], swapped[its] = draws[its], draws[mine]

    def table(sizes: str, given: list[dict[str, Any]], drawn: dict[Any, Any]) -> list[Any]:
        own = copy.deepcopy(doc)
        own["scoring"]["margins"]["sizes"] = sizes
        found = runner.probe_quantities(own, given, drawn)
        return margins.table(own, found, [], np.random.default_rng([26, 28]))

    def judged(found: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {r["quantity"]: r for r in found if r["judged"]}

    whole = judged(table("every", rows, draws))
    assert set(whole) == {"energy_r2", "energy_selectivity"}
    for row in whole.values():
        assert (row["sign_test"], row["verdict"]) == (fss.PASS, fss.PASS), row
    every_rows = table("every", turned, swapped)
    every, mean = judged(every_rows), judged(table("mean", turned, swapped))
    (at_L,) = [
        r
        for r in every_rows
        if (r["quantity"], r["scope"], r["role"]) == ("energy_r2", f"L={L}", "primary")
    ]
    assert at_L["margins"][seed] < 0 < min(mean["energy_r2"]["margins"]), at_L  # turned at L alone
    assert (every["energy_r2"]["sign_test"], every["energy_r2"]["verdict"]) == (fss.FAIL, fss.FAIL)
    assert (mean["energy_r2"]["sign_test"], mean["energy_r2"]["verdict"]) == (fss.PASS, fss.PASS)
    same = ("margins", "margin", "lo", "hi", "n_draws")
    assert [every["energy_r2"][k] for k in same] == [mean["energy_r2"][k] for k in same]
    assert every["energy_selectivity"]["verdict"] == fss.PASS  # its margins were left alone


def test_a_planted_positive_margin_passes(scored: Path) -> None:
    """The three judged rows and no others: energy's R^2 and selectivity by the CI and the sign
    test, every seed's margin positive, and the surprise FSS rule, at least 4 trained checkpoints of
    5 (two seeds of two replicas leave a planted slope a miss now and then) and no twin."""
    rows = _rows(scored, "margins")
    judged = {r["quantity"]: r for r in rows if r["judged"]}
    assert set(judged) == {"energy_r2", "energy_selectivity", "surprise_fss"}
    for name in ("energy_r2", "energy_selectivity"):
        row = judged[name]
        assert row["verdict"] == fss.PASS and row["sign_test"] == fss.PASS, row
        assert row["lo"] > 0.3 and all(m > 0.3 for m in row["margins"]), row
        assert row["n_seeds"] == 5 and row["n_draws"] == row["n_bootstrap"] == 40, row
    assert judged["surprise_fss"]["verdict"] == fss.PASS
    assert judged["surprise_fss"]["trained_passes"] >= 4
    assert judged["surprise_fss"]["twin_passes"] == 0


def test_every_other_margin_is_descriptive(scored: Path) -> None:
    """AUC is parity and the chi alignment descriptive, each L's a reading of the judged value, and
    each statistic's rule a reading of the judged row: none has a verdict, however large its
    margin."""
    rows = [r for r in _rows(scored, "margins") if r["role"] == "primary"]
    others = [r for r in rows if not r["judged"]]
    assert all(r["verdict"] == "" for r in others)
    names = {r["quantity"] for r in others}
    assert {
        "auc_pc_1_variance",
        "auc_pc_1_ac",
        "auc_surprise_variance",
        "auc_surprise_level",
    } <= names
    assert {"chi_alignment", "abs_m_r2", "l2var_r2", "energy_r2", "energy_selectivity"} <= names
    assert {"surprise_fss_surprise_slope", "surprise_fss_surprise_variance"} <= names
    energy = {r["scope"]: r for r in others if r["quantity"] == "energy_r2"}
    assert set(energy) == {f"L={L}" for L in V1_SIZES}
    judged = next(r for r in rows if r["judged"] and r["quantity"] == "energy_r2")
    mean = np.mean([energy[f"L={L}"]["margin"] for L in V1_SIZES])
    assert judged["margin"] == pytest.approx(mean)
    (level,) = [r for r in others if r["quantity"] == "auc_surprise_level"]
    assert (
        level["margin"] < -0.1
    )  # surprise falls as planted, a twin's stays flat: no verdict either


def test_an_ablations_margins_report_their_ci_with_no_verdict_end_to_end(scored: Path) -> None:
    """Two held-out seeds through the whole run: every quantity the primaries have and each surprise
    statistic, reading its own checkpoints in order, a planted margin large and its CI nested in the
    data draws, and nothing judged, for its role has no verdict. Its tc-only tables hold one
    plateau, so the chi alignment has no value, said as NaN."""
    rows = [r for r in _rows(scored, "margins") if r["role"] == "held-out"]
    assert rows and not any(r["judged"] or r["verdict"] for r in rows)
    primary = {(r["quantity"], r["scope"]) for r in _rows(scored, "margins") if not r["judged"]}
    assert {(r["quantity"], r["scope"]) for r in rows} == {
        key for key in primary if key[0] != "surprise_fss"
    }
    for row in rows:
        assert row["checkpoint"].split(";") == ABLATIONS, row["quantity"]
        assert row["twin"].split(";") == ABLATION_TWINS, row["quantity"]
        assert row["model_seeds"] == [21, 22] and row["n_seeds"] == 2, row["quantity"]
    energy = next(r for r in rows if (r["quantity"], r["scope"]) == ("energy_r2", "all"))
    assert energy["margin"] > 0.5 and 0.0 < energy["lo"] <= energy["hi"], energy
    assert energy["n_draws"] == 40 and len(energy["margins"]) == 2, energy
    chi = next(r for r in rows if (r["quantity"], r["scope"]) == ("chi_alignment", "all"))
    assert math.isnan(chi["margin"]) and math.isnan(chi["lo"]), chi
    slope = next(r for r in rows if r["quantity"] == "surprise_fss_surprise_slope")
    assert slope["trained_passes"] is not None and slope["twin_passes"] == 0, slope


def test_a_planted_null_does_not_pass(null: tuple[Path, V1, int]) -> None:
    """Every trained table and embedding is a twin's: none of the three judged rows passes. The
    rows are there, and say why: margins around 0, no surprise statistic passing."""
    run_dir, _, _ = null
    judged = {r["quantity"]: r for r in _rows(run_dir, "margins") if r["judged"]}
    assert set(judged) == {"energy_r2", "energy_selectivity", "surprise_fss"}
    for row in judged.values():
        assert row["verdict"] != fss.PASS, row
    assert abs(judged["energy_r2"]["margin"]) < 0.02
    assert judged["surprise_fss"]["trained_passes"] == 0


def test_a_planted_null_runs_unfrozen_with_no_tag(null: tuple[Path, V1, int]) -> None:
    """--planted stamps the result as what it is, and its trials are counted like any others."""
    run_dir, data, _ = null
    assert io.read_json(run_dir / "manifest.json")["params"]["prereg"]["tag"] == ""
    assert len(prereg.trials_path(data.root).read_text("utf-8").splitlines()) == len(
        _rows(run_dir, "auc")
    )


# ---------------------------------------------------------------- 5: the extraction rule


def _tamper(run_dir: Path, name: str, rule: dict[str, Any]) -> None:
    out = run_dir / ising.LATENTS / name
    manifest = io.read_json(out / "manifest.json")
    manifest["params"]["selection"] = rule
    io.write_json(out / "manifest.json", manifest, replace=True)


TC_PLUS = prereg.SELECTIONS["primary"]
TC_ONLY = prereg.SELECTIONS["held-out"]
ALL = TC_PLUS | {"rule": "all"}
ABLATION = ABLATIONS[0]


@pytest.mark.parametrize(
    ("kind", "table", "rule", "refused"),
    [
        ("qs", TRAINED[0], TC_ONLY, True),  # a primary's staircase read as an ablation's
        ("qs", TWINS[0], ALL, True),  # a primary's twin read whole
        ("qs", TRAINED[0], TC_PLUS | {"every": 4}, True),  # the rule's name, another stride
        ("qs", TRAINED[0], TC_PLUS | {"tc_plateau": 2.3}, True),  # another plateau
        ("qs", ABLATION, TC_PLUS, True),  # an ablation's staircase read as a primary's
        ("ramp", ABLATION, TC_PLUS, True),  # another role's rule, whatever a ramp keeps
        ("null", TRAINED[0], TC_ONLY, True),
        ("ramp", TRAINED[0], ALL, False),  # every rule reads a ramp whole: all says just that
        ("null", ABLATION, TC_ONLY, False),
        ("qs", TRAINED[0], TC_PLUS, False),
    ],
)
def test_a_table_extracted_under_another_rule_is_refused(
    frozen: Frozen, tmp_path: Path, kind: str, table: str, rule: dict[str, Any], refused: bool
) -> None:
    """Held to the rule latents.extraction gives the checkpoint's role, frozen or not, naming the
    run, the table and both rules; the same table under its own rule is admitted."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    run_dir = copy.runs[kind, V1_SIZES[0], V1_SEEDS[0]]
    _tamper(run_dir, table, rule)
    doc = prereg.load(frozen.path).doc
    if not refused:
        assert ising.admit(doc, copy.root).passes
        return
    with pytest.raises(ising.IsingDataError, match=r"as latents\.extraction says") as caught:
        ising.admit(doc, copy.root)
    message = str(caught.value)
    assert f"{run_dir.name}/{ising.LATENTS}/{table}" in message and message.count("/latents/") == 1


def test_the_runner_refuses_a_table_under_another_rule_before_any_work(
    frozen: Frozen, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    _tamper(copy.runs["qs", V1_SIZES[-1], V1_SEEDS[-1]], TRAINED[1], TC_ONLY)
    assert _run(frozen.path, copy, "--no-figures") == 1
    assert "as latents.extraction says" in capsys.readouterr().err
    assert not _results(copy) and not prereg.trials_path(copy.root).exists()


def test_the_selections_are_the_extractors_own() -> None:
    """ews cannot import jepa, so prereg.py holds the rule each role takes: it must stay the one
    jepa.extract.selection records in a table's manifest."""
    pytest.importorskip("torch")
    from tipping_point.jepa import extract

    assert prereg.SELECTIONS["primary"] == extract.selection("tc-plus")
    assert prereg.SELECTIONS["ema-only"] == extract.selection("tc-only")
    assert prereg.SELECTIONS["held-out"] == extract.selection("tc-only")
    assert set(prereg.SELECTIONS) == prereg.CHECKPOINT_ROLES


# ---------------------------------------------------------------- what else it refuses


def test_planted_mode_refuses_any_run_the_generator_did_not_write(
    frozen: Frozen, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--planted has no freeze behind it, so a data dir holding one real run must never be read: the
    confirmatory seeds stay unread until the tag."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    run_dir = copy.runs["qs", V1_SIZES[0], V1_SEEDS[0]]
    manifest = io.read_json(run_dir / "manifest.json")
    manifest["producer"]["name"] = "tipping_point.ising"
    io.write_json(run_dir / "manifest.json", manifest, replace=True)
    assert _run(frozen.path, copy, "--no-figures", "--planted") == 1
    assert run_dir.name in capsys.readouterr().err
    assert not _results(copy)


def test_an_unfrozen_file_is_refused_without_planted(
    frozen: Frozen, base: V1, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = frozen.path.read_text("utf-8") + "\n"
    repo_copy = frozen.path.parent / "not-the-frozen-one"
    repo_copy.mkdir()
    edited = repo_copy / "ising-latent-v1.toml"
    edited.write_text(text, encoding="utf-8", newline="\n")
    try:
        assert _run(edited, frozen.spare, "--no-figures") == 1
    finally:
        shutil.rmtree(repo_copy)
    assert "ising-latent-v1" in capsys.readouterr().err
    assert not _results(frozen.spare)


def test_a_draft_with_placeholders_is_refused_by_name(
    frozen: Frozen, capsys: pytest.CaptureFixture[str]
) -> None:
    """The draft as it stands has no direction to score by: said at once, not as a stack trace from
    the first surrogate test."""
    assert _run(planted.DRAFT, frozen.spare, "--planted", "--no-figures") == 1
    err = capsys.readouterr().err
    assert "hypotheses[2].direction: placeholder" in err
    assert not _results(frozen.spare)


@pytest.mark.parametrize("fault", ["absent", "another version", "another commit", "another run"])
def test_a_baseline_result_that_is_not_the_files_is_refused(
    frozen: Frozen,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fault: str,
) -> None:
    """[baseline] names a result of the version whose runs this file reads, on these runs: anything
    else is no parity reference, and the run stops before reading a frame."""
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    folder = copy.root / "ews" / frozen.result
    manifest = io.read_json(folder / "manifest.json")
    if fault == "absent":
        shutil.rmtree(folder)
    else:
        if fault == "another version":
            manifest["params"]["prereg"]["version"] = 2
        elif fault == "another commit":
            manifest["params"]["prereg"]["commit"] = "f" * 40
        else:
            manifest["params"]["inputs"] = manifest["params"]["inputs"][1:]
        io.write_json(folder / "manifest.json", manifest, replace=True)
    assert _run(frozen.path, copy, "--no-figures") == 1
    assert "baseline.result" in capsys.readouterr().err
    assert not _results(copy)


def test_a_probe_fit_run_must_exist_and_be_one(
    frozen: Frozen, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = _copy(frozen.spare, tmp_path / "data", frozen.result)
    shutil.rmtree(copy.runs["fit", V1_SIZES[1], planted.FIT_SEED])
    assert _run(frozen.path, copy, "--no-figures") == 1
    assert f"0 complete staircases of seed {planted.FIT_SEED} at L = {V1_SIZES[1]}" in (
        capsys.readouterr().err
    )
    assert not _results(copy)


def test_ews_score_will_not_spend_a_latent_files_budget_on_a_part_of_it(
    frozen: Frozen, capsys: pytest.CaptureFixture[str]
) -> None:
    """A latent file's trials can only be spent once. `ews score` writes its cell tables and logs
    them with no probes or margins, and the runner could then never run: refused unless asked."""
    argv = ["score", "--prereg", str(frozen.path), "--data-dir", str(frozen.spare.root)]
    assert ews_main(argv) == 1
    err = capsys.readouterr().err
    assert "harness/ising_experiment.py" in err and "--cells-only" in err
    assert not prereg.trials_path(frozen.spare.root).exists()


def test_alignment_is_the_correlation_over_plateaus() -> None:
    """On designed curves, since planted staircases hold two plateaus: a latent curve that is the
    classic one scaled reads 1, its mirror -1, and a table with fewer than three plateaus (tc-only)
    has none. Catches a correlation taken on the wrong column or across L."""
    temperatures = (3.0, 2.5, 2.3, 2.0)
    classic = (1.0, 5.0, 40.0, 3.0)
    curves = [
        {"checkpoint": "", "observable": "abs_m", "L": 16, "temperature": T, "l2_var": v}
        for T, v in zip(temperatures, classic, strict=True)
    ]
    for name, values in (("same", classic), ("mirror", [-v for v in classic])):
        curves += [
            {"checkpoint": name, "observable": "pc_1", "L": 16, "temperature": T, "l2_var": 7 * v}
            for T, v in zip(temperatures, values, strict=True)
        ]
        curves += [
            {"checkpoint": name, "observable": "surprise", "L": 16, "temperature": T, "l2_var": 1.0}
            for T in temperatures
        ]
    curves += [
        {"checkpoint": "short", "observable": "pc_1", "L": 16, "temperature": 2.3, "l2_var": 1.0}
    ]
    doc = {"data": {"L": [16]}}
    per_L, mean = runner.alignment_quantities(doc, curves)
    assert (per_L.scope, mean.scope) == ("L=16", "all")
    assert per_L.values["same"] == pytest.approx(1.0) and mean.values["same"] == pytest.approx(1.0)
    assert per_L.values["mirror"] == pytest.approx(-1.0)
    assert math.isnan(per_L.values["short"])


# ---------------------------------------------------------------- the figures


def _environment_works() -> str | None:
    """Why the isolated environment cannot be made here (no uv, or offline with an empty cache)."""
    command = [*runner.FIGURE_ENVIRONMENT, "python", "-c", "import matplotlib"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return repr(exc)
    return None if done.returncode == 0 else done.stderr[-300:]


def test_the_figures_are_drawn_from_the_tables_and_deterministic(scored: Path, again: Path) -> None:
    """In the isolated environment, never the project's: five files from the CSV tables of a result,
    the same bytes when drawn again from the second run's tables."""
    why = _environment_works()
    if why is not None:
        pytest.skip(f"no isolated matplotlib here: {why}")
    assert runner.draw_figures(scored) is None
    first = {name: (scored / "figures" / name).read_bytes() for name in FIGURES}
    assert all(body.startswith(b"\x89PNG") and len(body) > 5000 for body in first.values())
    assert sorted(p.name for p in (scored / "figures").iterdir()) == sorted(FIGURES)
    assert runner.draw_figures(again) is None
    assert {name: (again / "figures" / name).read_bytes() for name in FIGURES} == first


def test_the_pipeline_warning_reads_the_static_and_the_dynamic_check(scored: Path) -> None:
    """_dynamic writes its role as dynamic-pipeline-check and _static as pipeline-check: a miss in
    either is a pipeline fault, and hypothesis 1 says nothing else is read until it is fixed. A
    judged test that fails is no such thing, and neither is a check that passes. The roles are the
    ones the scorer writes."""
    assert {"pipeline-check", "dynamic-pipeline-check"} <= {r["role"] for r in _rows(scored, "fss")}
    rows = [
        {"checkpoint": "a", "quantity": "l2_var_pc_1", "role": "pipeline-check", "verdict": "fail"},
        {
            "checkpoint": "b",
            "quantity": "inverse_nu_pc_1_variance",
            "role": "dynamic-pipeline-check",
            "verdict": "inconclusive",
        },
        {
            "checkpoint": "c",
            "quantity": "inverse_nu_surprise_slope",
            "role": "dynamic-test",
            "verdict": "fail",
        },
        {"checkpoint": "d", "quantity": "l2_var_pc_1", "role": "pipeline-check", "verdict": "pass"},
        {
            "checkpoint": "e",
            "quantity": "inverse_nu_pc_1_variance",
            "role": "dynamic-pipeline-check",
            "verdict": "pass",
        },
    ]
    assert runner.pipeline_faults(rows) == [
        "a: l2_var_pc_1 fail",
        "b: inverse_nu_pc_1_variance inconclusive",
    ]


def test_a_figure_program_that_exits_nonzero_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing program is the fixture's fault below; a script that dies is this one. Either way
    the caller gets the reason, and the tables behind it are never at stake."""
    monkeypatch.setattr(runner, "FIGURE_ENVIRONMENT", (sys.executable,))
    why = runner.draw_figures(tmp_path)
    assert why is not None and "python" in why.lower()


def _spent_copy(frozen: Frozen, scored: Path, root: Path) -> tuple[Path, Path]:
    """The scored data's ews/ folder in `root`, its budget spent and its tables safe: the result
    and the trial log, as a run that died after its last trial leaves them."""
    shutil.copytree(frozen.data.root / "ews", root / "ews")
    return root / "ews" / scored.name, prereg.trials_path(root)


def _csv(run_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in (run_dir / "csv").iterdir()}


@pytest.mark.parametrize("left", ["no csv/", "a torn CSV", "a stray temp file"])
def test_figures_only_writes_the_csv_again_and_charges_nothing(
    frozen: Frozen,
    scored: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    left: str,
) -> None:
    """#118: the budget is spent, and whatever a crash left of csv/, the entry gives the same bytes
    back, then draws, with the whole csv/ there to draw from. No line joins the log, its count does
    not move, no result folder appears, no Parquet file changes."""
    run_dir, log = _spent_copy(frozen, scored, tmp_path)
    pre = prereg.load(frozen.path)
    budget = pre.doc["trials"]["budget"]
    assert prereg.trials(tmp_path, pre) == budget
    want, charged = _csv(run_dir), log.read_bytes()
    tables = {p: p.read_bytes() for p in run_dir.rglob("*.parquet")}
    folders = sorted((tmp_path / "ews").iterdir())
    if left == "no csv/":
        shutil.rmtree(run_dir / "csv")
    elif left == "a torn CSV":
        (run_dir / "csv" / "auc.csv").write_bytes(want["auc.csv"][:50])
        (run_dir / "csv" / "margins.csv").unlink()
    else:
        (run_dir / "csv" / "auc.csv.tmp").write_bytes(b"half a table")
        (run_dir / "csv" / "auc.csv").unlink()
    drawn: list[tuple[Path, dict[str, bytes]]] = []
    monkeypatch.setattr(runner, "draw_figures", lambda at: drawn.append((at, _csv(at))))
    assert runner.main(["--figures-only", str(run_dir)]) == 0
    assert drawn == [(run_dir, want)]
    assert _csv(run_dir) == want
    assert log.read_bytes() == charged and prereg.trials(tmp_path, pre) == budget
    assert sorted((tmp_path / "ews").iterdir()) == folders
    assert {p: p.read_bytes() for p in run_dir.rglob("*.parquet")} == tables


def test_figures_only_reports_a_figure_failure_as_the_run_does(
    frozen: Frozen,
    scored: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 2 and why, with the CSV written: a failed figure step costs no more here than in a run.
    --no-figures stops at the CSV."""
    run_dir, _ = _spent_copy(frozen, scored, tmp_path)
    want = _csv(run_dir)
    shutil.rmtree(run_dir / "csv")
    monkeypatch.setattr(runner, "draw_figures", lambda at: "a reason planted here")
    assert runner.main(["--figures-only", str(run_dir)]) == 2
    assert "a reason planted here" in capsys.readouterr().err
    assert _csv(run_dir) == want
    shutil.rmtree(run_dir / "csv")
    monkeypatch.setattr(runner, "draw_figures", lambda at: pytest.fail("--no-figures drew"))
    assert runner.main(["--figures-only", str(run_dir), "--no-figures"]) == 0
    assert _csv(run_dir) == want


def test_figures_only_refuses_a_folder_with_no_complete_json(
    frozen: Frozen,
    scored: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A torn result's cells may be unlogged, and #120 stops anyone scoring past them: this entry
    must not be the way to read them. Exit 1 by name, no CSV, no figure. A path that holds no result
    at all is refused the same way."""
    run_dir, _ = _spent_copy(frozen, scored, tmp_path)
    shutil.rmtree(run_dir / "csv")
    (run_dir / "complete.json").unlink()
    monkeypatch.setattr(runner, "draw_figures", lambda at: pytest.fail("a torn result was drawn"))
    assert runner.main(["--figures-only", str(run_dir)]) == 1
    assert f"{run_dir}: no complete.json" in capsys.readouterr().err
    assert not (run_dir / "csv").exists()
    assert runner.main(["--figures-only", str(tmp_path / "nothing-here")]) == 1
    assert "no complete.json" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["--prereg", "a.toml", "--figures-only", "b"]])
def test_figures_only_is_one_entry_or_the_other(argv: list[str]) -> None:
    """It takes the place of --prereg: neither, or both, is argparse's usage error."""
    with pytest.raises(SystemExit) as stopped:
        runner.main(argv)
    assert stopped.value.code == 2


def test_figures_only_draws_what_the_tables_draw(
    frozen: Frozen, scored: Path, tmp_path: Path
) -> None:
    """The whole way, with figure_command in its isolated environment: csv/ and figures/ gone, as
    they are after a crash before csv_tables finished, the entry draws the five figures, to the
    byte, the ones a whole run's tables draw."""
    why = _environment_works()
    if why is not None:
        pytest.skip(f"no isolated matplotlib here: {why}")
    assert runner.draw_figures(scored) is None
    first = {name: (scored / "figures" / name).read_bytes() for name in FIGURES}
    run_dir, _ = _spent_copy(frozen, scored, tmp_path)
    shutil.rmtree(run_dir / "csv")
    shutil.rmtree(run_dir / "figures")
    assert runner.main(["--figures-only", str(run_dir)]) == 0
    assert {name: (run_dir / "figures" / name).read_bytes() for name in FIGURES} == first
    assert runner.main(["--figures-only", str(run_dir)]) == 0  # and again, over what is there
    assert {name: (run_dir / "figures" / name).read_bytes() for name in FIGURES} == first


def test_the_figure_script_imports_nothing_of_the_project() -> None:
    """It runs in an environment holding matplotlib alone."""
    text = runner.FIGURES.read_text("utf-8")
    assert "tipping_point" not in text.split('"""', 2)[2] and "numpy" not in text


def test_matplotlib_is_no_dependency_of_the_project() -> None:
    root = Path(__file__).resolve().parents[4]
    assert "matplotlib" not in (root / "python" / "tipping_point" / "pyproject.toml").read_text(
        "utf-8"
    )
    assert "matplotlib" not in (root / "pyproject.toml").read_text("utf-8")


def test_a_figure_step_that_fails_loses_no_table(null: tuple[Path, V1, int]) -> None:
    """The fixture's figure program does not exist: the run exits 2, its result is complete, every
    table is what complete.json hashed, and the CSV the figures read is there for the next
    attempt."""
    run_dir, _, code = null
    assert code == 2
    assert (run_dir / "complete.json").is_file()
    record = io.read_json(run_dir / "complete.json")
    for name in TABLES:
        path = run_dir / name / f"{0:020d}.parquet"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["files"][f"{name}/{0:020d}.parquet"] == digest, name
        assert (run_dir / "csv" / f"{name}.csv").is_file()
