"""The experiment runner (harness/ising_experiment.py, #28 slice 5) end to end on planted data: a
frozen file in a scratch repository to the result folder, twice for its bytes; the draft on a
planted null, whose margins must not pass; what it refuses; and the figures, drawn in the isolated
environment."""

import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ews_planted as planted
import ising_experiment as runner
import numpy as np
import pyarrow.parquet as pq
import pytest
from ews_planted import V1, V1_SEEDS, V1_SIZES
from tipping_point.ews import fss, ising, prereg
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
import os, sys
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


if where == "before the trials":
    prereg.log_trials = lambda *args: os._exit(3)
elif where == "after the trials":
    io.write_json = before_complete
score(pre, prereg.Frozen("", ""), root, 1)
"""


@pytest.mark.parametrize(
    ("where", "logged"),
    [("before the trials", 0), ("after the trials", 3132), ("nowhere", 3132)],
)
def test_a_killed_scoring_leaves_no_complete_result_short_of_its_trials(
    tmp_path: Path, where: str, logged: int
) -> None:
    """The whole last step of a real scoring of the draft's 3,132 cells, in a process that dies by
    os._exit, with nothing flushed for it, at either side of the trial log. Killed, it leaves its
    tables and no complete.json, its trials all logged or none: never a complete-looking result with
    1,200 of them charged. The log it leaves decides what comes next: none, and the file may be
    scored again; all, and its budget is spent, the tables on disk, until it is amended. Unkilled,
    complete.json comes last, over all of them."""
    done = subprocess.run(
        [sys.executable, "-c", KILLED, str(planted.DRAFT), str(tmp_path), where],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == (0 if where == "nowhere" else 3), done.stderr
    (run_dir,) = (tmp_path / "ews").glob("*-ising-latent-v1")
    assert (run_dir / "auc" / f"{0:020d}.parquet").is_file()
    assert (run_dir / "complete.json").is_file() == (where == "nowhere")
    draft = prereg.load(planted.DRAFT)
    assert prereg.trials(tmp_path, draft) == logged
    if logged:
        lines = prereg.trials_path(tmp_path).read_text("utf-8").splitlines()
        assert {json.loads(line)["result"] for line in lines} == {run_dir.name}
        with pytest.raises(prereg.PreregError, match="exceed its budget"):
            prereg.charge(tmp_path, draft, 1)
    else:
        assert prereg.charge(tmp_path, draft, 3132) == 0


# ---------------------------------------------------------------- 4: the margins


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
