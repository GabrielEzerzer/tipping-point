"""The exploratory pipeline on planted data (harness/ews_planted.py): every script runs, on a
latents table and on |m|, and the planted answers come back; seeds 6-10 are refused at every door,
nothing can log a trial, and the data dir is joined without touching what it joins.
"""

import ast
import hashlib
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import ews_planted
import numpy as np
import pytest
from ews_planted import LATENT_EPS, V1_LAG, V1_PILOT_RUN, V1_REPLICAS, V1_TAU_C, write_latents
from explore import common, fast_ramps, join_pilot, latent_pass, latent_report, static_tc
from explore.__main__ import SCRIPTS
from tipping_point.ews import ising, prereg
from tipping_point.ising import protocol
from tipping_point.ising.exact import TC

ROOT = Path(__file__).resolve().parents[2]
TRAINING = "20261101T000000Z-L16-32-m7"
STEP = 1000
TRAINED, TWIN = ising.latents_name(TRAINING, STEP), ising.latents_name(TRAINING, 0)
SIZES = (16, 32)
SEEDS = (1, 2)


def _import(name: str) -> Any:
    return importlib.import_module(f"explore.{name}")


def _fast_ramp(root: Path, L: int, seed: int) -> Path:
    A, z = ews_planted._fit()
    schedule = protocol.Ramp(
        protocol.rate_for(L, 2.0, V1_TAU_C[L]),
        L,
        T0=TC * (1 + ews_planted.V1_EPS_L[0] / L),
        T1=TC * (1 + ews_planted.V1_EPS_L[1] / L),
        pre=ews_planted.V1_PRE,
        hold=ews_planted.V1_HOLD,
        x=2.0,
        A=A,
        z=z,
        rate_source="measured",
        eps_L=ews_planted.V1_EPS_L,
    )
    return ews_planted.write_run(
        root,
        f"L{L}-ramp-x2-s{seed}",
        schedule,
        L,
        seed,
        ews_planted.peak_values("fss", L),
        pilot=ews_planted._stamp(root),
        replicas=V1_REPLICAS,
        rng=np.random.default_rng([seed, L, 9]),
    )


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The v1 planted dataset at L = 16 and 32 on seeds 1 and 2, the only seeds the pipeline reads,
    with a fast ramp per (L, seed) and a trained table and its twin beside every run."""
    root = tmp_path_factory.mktemp("explore-data")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ews_planted, "V1_SEEDS", SEEDS)
        patch.setattr(ews_planted, "V1_SIZES", SIZES)
        planted = ews_planted.v1_dataset(root, "fss")
        runs = [*planted.runs.values()]
        runs += [_fast_ramp(root, L, seed) for L in SIZES for seed in SEEDS]
        for run in runs:
            write_latents(run, TRAINED, select="tc-plus")
            write_latents(run, TWIN, select="tc-plus", untrained=True)
    return root


def _shared(root: Path, out: Path) -> list[str]:
    digest = hashlib.sha256((root / "ising" / V1_PILOT_RUN / "pilot.json").read_bytes()).hexdigest()
    return [
        "--data-dir",
        str(root),
        "--out",
        str(out),
        "--seeds",
        *(str(s) for s in SEEDS),
        "--replicas",
        str(V1_REPLICAS),
        "--pilot-run",
        V1_PILOT_RUN,
        "--pilot-sha256",
        digest,
    ]


@pytest.fixture(scope="module")
def passed(data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The latent pass's core steps, with log_trial wired to fail."""
    out = tmp_path_factory.mktemp("explore-out")

    def refused(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the exploratory pipeline logged a trial")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(prereg, "log_trial", refused)
        latent_pass.main(
            [
                "--training",
                TRAINING,
                "--step",
                str(STEP),
                *_shared(data, out),
                "--headline-only",
                "--extra=--bootstrap=20",
            ]
        )
    return out


def _json(out: Path, table: str, L: int | None, name: str) -> dict[str, Any]:
    folder = out / table if L is None else out / table / f"L{L}"
    return json.loads((folder / name).read_text("utf-8"))


# ---------------------------------------------------------------- the planted answers come back


def test_the_pass_writes_its_report_and_logs_no_trial(data: Path, passed: Path) -> None:
    text = (passed / f"latent-report-{TRAINING}.md").read_text("utf-8")
    for heading in ("Directions", "monotone or peaked", "What pc_1 follows", "floors", "1/nu"):
        assert heading in text
    assert "Never logged as trials" in text
    assert not prereg.trials_path(data).exists()
    assert _json(passed, TRAINED, 16, "static.json")["provenance"]["exploratory"] is True


def test_surprise_reads_as_planted(passed: Path) -> None:
    """Planted: surprise falls as the bath cools, steepest at a fixed eps_L, so the level's
    direction is "less" and the slope's extremum "min"; its median read position is half a window
    late, and 1/nu near 1. The twin's is flat: no trend to speak of."""
    answer = latent_report.directions(passed, TRAINED, list(SIZES))
    assert (answer["level"], answer["slope_extremum"]) == ("less", "min")
    combined = _json(passed, TRAINED, None, "peaks-combined.json")["indicators"]
    slope = combined["surprise_slope_min"]
    for L in SIZES:
        assert slope["medians"][str(L)] == pytest.approx(LATENT_EPS - V1_LAG, abs=0.25), L
    assert slope["inverse_nu"] == pytest.approx(1.0, abs=0.5)
    for L in SIZES:
        (level,) = [
            r
            for r in _json(passed, TRAINED, L, "auc_alarms.json")["auc"]
            if r["headline"] and r["h"] == 0.0 and r["indicator"] == "level"
        ]
        assert level["auc"] < 0.1 and level["pos_median_s"] < -0.5
        (flat,) = [
            r
            for r in _json(passed, TWIN, L, "auc_alarms.json")["auc"]
            if r["headline"] and r["h"] == 0.0 and r["indicator"] == "level"
        ]
        assert abs(flat["pos_median_s"]) < 0.5


def test_the_curve_says_where_a_plateau_mean_turns() -> None:
    """The curve the real surprise draws: flat, a shallow dip and a hump near Tc, then a steep fall
    far larger than either. The dip counts as a turn though the curve's minimum is elsewhere; a
    monotone curve with the same noise turns nowhere."""
    eps = np.linspace(8.0, -5.0, 40)
    rng = np.random.default_rng(1)
    fall = 0.3 * np.clip(-eps - 1.0, 0.0, None)
    curve = (
        0.8 - 0.01 * np.exp(-((eps - 0.7) ** 2) / 1.5) + 0.01 * np.exp(-((eps + 1.0) ** 2)) - fall
    )
    draws = curve + 0.0005 * rng.standard_normal((200, len(eps)))
    shape = static_tc.shape(eps, curve, draws)
    assert shape["valley"]["turns"] and shape["peak"]["turns"]
    assert shape["valley"]["eps_L"] == pytest.approx(0.7, abs=0.4)
    assert shape["peak"]["eps_L"] == pytest.approx(-1.0, abs=0.5)
    falling = 0.8 + 0.05 * eps
    draws = falling + 0.0005 * rng.standard_normal((200, len(eps)))
    shape = static_tc.shape(eps, falling, draws)
    assert not shape["valley"]["turns"] and not shape["peak"]["turns"]
    assert shape["kendall_tau_cooling"] == pytest.approx(-1.0)


def test_pc_1_reads_m_on_trained_and_twin_alike(passed: Path) -> None:
    """Planted: pc_1 is |m| plus noise on both; pc_3 is noise. Each r has a percentile CI and says
    so."""
    for table in (TRAINED, TWIN):
        for L in SIZES:
            found = _json(passed, table, L, "correlations.json")
            assert "hierarchical bootstrap" in found["ci_method"]
            at = found["columns"]["pc_1"]["abs_m"]["at_fss_plateau"]
            assert at["value"] > 0.8 and at["ci"][0] <= at["value"] <= at["ci"][1]
            assert abs(found["columns"]["pc_3"]["abs_m"]["at_fss_plateau"]["value"]) < 0.5


def test_probes_score_the_planted_targets_beside_their_floors(passed: Path) -> None:
    """Planted: |m| is pc_1, so its R^2 is high, and e is plain noise on every frame, so its R^2
    sits on the floor. The controls behave: selectivity is the difference, as stated."""
    for table in (TRAINED, TWIN):
        found = _json(passed, table, 16, "probes.json")
        targets = found["targets"]
        assert targets["abs_m"]["r2"]["value"] > 0.9
        assert targets["energy"]["r2"]["value"] < 0.5
        for t, v in targets.items():
            assert v["iid"]["value"] < 0.5, t
            assert v["selectivity"]["value"] == pytest.approx(
                v["r2"]["value"] - v["shuffle_within_T"]["value"]
            ), t
            assert v["selectivity_noise_q95"] >= 0.0, t
        assert "128-dimensional" in found["note"]


def test_the_classic_path_still_runs(data: Path, tmp_path: Path) -> None:
    """Without --latents: |m| at the dense cadence, through every script but the latent ones."""
    args = _shared(data, tmp_path)
    for L in SIZES:
        where = ["--L", str(L), *args]
        static_tc.main([*where, "--bootstrap", "20"])
        _import("peaks").main([*where, "--bootstrap", "20"])
        _import("nulls_explore").main([*where, "--bootstrap", "20"])
        fast_ramps.main([*where, "--bootstrap", "20"])
    where = ["--L", "16", *args]
    _import("auc_alarms").main([*where, "--headline-only", "--bootstrap", "20"])
    _import("surrogates_cal").main([*where, "--n-surrogates", "2", "--kinds", "phase"])
    _import("alarms_explore").main([*where, "--every", "200"])
    _import("report").main(where)
    _import("scaling").main(["--sizes", *map(str, SIZES), *args])
    _import("peaks").main(["--combine", *map(str, SIZES), "--bootstrap", "20", *args])
    _import("fixed_eps").main(["--sizes", *map(str, SIZES), *args])
    report = (tmp_path / "classic" / "L16" / "report.md").read_text("utf-8")
    assert "Headline cell" in report and "Peak positions" in report
    found = _json(tmp_path, "classic", None, "scaling.json")
    assert {r["quantity"] for r in found["rows"]} >= {"l2_var_abs_m", "one_minus_ac1"}


def test_surrogates_and_alarms_run_on_a_table_once_directions_are_read(
    data: Path, tmp_path: Path
) -> None:
    """The long steps: without directions they refuse, and with them a level takes phase alone."""
    where = ["--L", "16", "--latents", TRAINED, *_shared(data, tmp_path)]
    with pytest.raises(SystemExit, match="--surprise-variance"):
        _import("surrogates_cal").main(where)
    with pytest.raises(SystemExit, match="--surprise-variance"):
        _import("alarms_explore").main(where)
    oriented = ["--surprise-variance", "greater", "--surprise-level", "less"]
    _import("surrogates_cal").main([*where, *oriented, "--n-surrogates", "2"])
    _import("alarms_explore").main([*where, *oriented, "--every", "200"])
    folder = tmp_path / TRAINED / "L16"
    assert (folder / "surrogates-surprise-level-phase.json").is_file()
    assert not (folder / "surrogates-surprise-level-ar1.json").exists()
    assert (folder / "surrogates-surprise-variance-ar1.json").is_file()
    assert (folder / "alarms_explore-surprise-level.json").is_file()


# ---------------------------------------------------------------- plainly exploratory


@pytest.mark.parametrize("seed", [6, 7, 8, 9, 10, 11])
def test_seeds_outside_1_to_5_are_refused_by_every_script(seed: int, tmp_path: Path) -> None:
    for name in (
        "static_tc",
        "auc_alarms",
        "peaks",
        "correlations",
        "probes_explore",
        "nulls_explore",
        "surrogates_cal",
        "alarms_explore",
        "fast_ramps",
    ):
        argv = ["--L", "16", "--latents", "x", "--seeds", str(seed), "--data-dir", str(tmp_path)]
        with pytest.raises(common.Refused, match="exploratory"):
            _import(name).main(argv)
    with pytest.raises(common.Refused, match="exploratory"):
        latent_pass.main(["--training", "t", "--step", "1", "--seeds", "1", str(seed)])


def test_a_run_at_a_confirmatory_seed_cannot_be_opened_through_the_pipeline(tmp_path: Path) -> None:
    run = ews_planted.write_run(
        tmp_path,
        "L16-ramp-x2-s6",
        protocol.Ramp(100, 16, x=2.0),
        16,
        6,
        ews_planted.peak_values("fss", 16),
        replicas=V1_REPLICAS,
    )
    with pytest.raises(common.Refused, match="seed 6"):
        common.allowed(ising.open_run(run))
    with pytest.raises(common.Refused, match="seed 6"):
        fast_ramps.ramps_of(tmp_path, 16, 2.0, [6])


def test_no_script_can_log_a_trial_or_write_a_result() -> None:
    """By the code, not by the run: no script names what would make a number count as a trial, and
    none imports the module that holds them."""
    banned = {"log_trial", "write_result", "freeze", "trials_path", "stamp", "frozen"}
    for path in (ROOT / "harness" / "explore").glob("*.py"):
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned, (path.name, node.attr)
            elif isinstance(node, ast.Name):
                assert node.id not in banned, (path.name, node.id)
            elif isinstance(node, ast.ImportFrom):
                assert "prereg" not in {a.name for a in node.names}, path.name
                assert not (node.module or "").endswith("prereg"), path.name


@pytest.mark.parametrize(
    "name", [s for s in SCRIPTS if s not in ("join_pilot", "latent_pass", "latent_report")]
)
def test_every_script_takes_the_size_and_the_table(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        _import(name).main(["--help"])
    usage = capsys.readouterr().out
    assert "--L" in usage and "--latents" in usage


# ---------------------------------------------------------------- the join


def _joinable(data: Path, tmp_path: Path) -> tuple[Path, Path, str]:
    local, output = tmp_path / "local", tmp_path / "out"
    (local / "ising").mkdir(parents=True)
    run_id = next(p.name for p in (data / "ising").iterdir() if "-qs-s1" in p.name)
    shutil.copytree(
        data / "ising" / run_id, local / "ising" / run_id, ignore=shutil.ignore_patterns("latents")
    )
    shutil.copytree(data / "ising" / run_id / "latents", output / "ising" / run_id / "latents")
    return local, output, run_id


def test_the_join_links_runs_and_tables_and_skips_what_it_may_not_read(
    data: Path, tmp_path: Path
) -> None:
    local, output, run_id = _joinable(data, tmp_path)
    shutil.copytree(data / "ising" / V1_PILOT_RUN, local / "ising" / V1_PILOT_RUN)
    confirmatory = ews_planted.write_run(
        local,
        "L16-ramp-x2-s6",
        protocol.Ramp(100, 16, x=2.0),
        16,
        6,
        ews_planted.peak_values("fss", 16),
        replicas=V1_REPLICAS,
    )
    into = tmp_path / "join"
    counted = join_pilot.join(local, output, into)
    assert counted == {"runs": 1, "tables": 2, "records": 1}
    assert not (into / "ising" / confirmatory.name).exists()
    (s,) = ising.series(ising.open_run(into / "ising" / run_id), "matched", TRAINED)
    assert s.latents["surprise"].shape == s.abs_m.shape
    shutil.rmtree(into)  # hard links: removing the join leaves what it joined
    assert (local / "ising" / run_id / "complete.json").is_file()


def test_the_join_refuses_a_table_read_from_other_bytes(data: Path, tmp_path: Path) -> None:
    local, output, run_id = _joinable(data, tmp_path)
    done = local / "ising" / run_id / "complete.json"
    done.write_text(done.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="other bytes"):
        join_pilot.join(local, output, tmp_path / "join")
