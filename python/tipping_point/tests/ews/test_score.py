"""`score` end to end on the planted dataset: a frozen pre-registration in a scratch repository,
the result folder it writes, the values it holds where the planted answer is known, the trials it
logs, and the refusals that keep it honest."""

import hashlib
import json
import math
import subprocess
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from ews_planted import NULL_EPS, PILOT_RUN, PLATEAU
from tipping_point.ews import ising, prereg, scoring
from tipping_point.ews.__main__ import main
from tipping_point.ews.trend import Pipeline

PREREG = """
id = "{id}"
version = 1
supersedes = ""
reason = ""
question = "Do the planted warnings come through the pipeline?"

[[hypotheses]]
name = "variance-rises"
statement = "The planted spread grows towards Tc."
prediction = "AUC of variance above 0.5."
direction = "greater"

[data]
domain = "ising"
L = [16, 32]
seeds = [1, 2]
replicas = 8
exclusions = []

[data.tau_c]
source = "pilot"
pilot_run = "{pilot_run}"
pilot_sha256 = "{pilot_sha256}"

[data.ramps]
kind = "ramp"
x = 0.02

[data.nulls]
kind = "fixed"
eps_L = {eps}

[data.staircase]
kind = "staircase"
temperature = {plateau}

[indicators]
names = ["variance", "ac", "skewness"]
observables = ["abs_m"]
cadences = ["dense", "matched"]
thin = 0.5
lag = 1
detrend = "gaussian"
direction = {{ variance = "greater", ac = "greater", skewness = "less" }}

[event]
definition = "The bath crosses Tc at the manifest's t_c."
horizons = [0.0, 1.0]
segment = 2.0

[scoring]
positives = "ramp segments ending at eps_L = h"
negatives = "equal slices of the fixed-T nulls"
auc = "Mann-Whitney, ties 1/2"
alpha = 0.05
ci_level = 0.95
bootstrap = 50

[scoring.surrogates]
primary = "phase"
secondary = "ar1"
n = 19
cells = "headline"

[scoring.alarms]
far = 0.02
report_per = 100.0
band = [2.0, -1.0]
every = 8.0

[[fss.static]]
quantity = "l2_var_abs_m"
role = "positive-control"
expected = 1.75
tolerance = 0.1

[[fss.static]]
quantity = "mean_abs_m"
role = "positive-control"
expected = -0.125
tolerance = 0.02

[[fss.static]]
quantity = "skewness_abs_m"
role = "positive-control"
expected = 0.0
tolerance = 0.1

[[fss.static]]
quantity = "tau_int_abs_m"
role = "descriptive"
expected = 2.0

[[fss.static]]
quantity = "one_minus_ac1"
role = "reference"

[fss.dynamic]
indicators = ["variance", "ac"]
observable = "abs_m"
expected_inverse_nu = 1.0
tolerance = 0.1

[sensitivity]
window = [8.0]
bandwidth = [4.0, 6.0]
headline = {{ window = 8.0, bandwidth = 4.0 }}

[trials]
budget = 48
counted = "one per AUC cell"

[deviations]
policy = "a new version"
log = []
"""


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _text(data_dir: Path, prereg_id: str, pilot_run: str) -> str:
    """The planted prereg, pinned to the planted pilot record."""
    pilot = data_dir / "ising" / PILOT_RUN / "pilot.json"
    digest = hashlib.sha256(pilot.read_bytes()).hexdigest()
    return PREREG.format(
        id=prereg_id, pilot_run=pilot_run, pilot_sha256=digest, eps=list(NULL_EPS), plateau=PLATEAU
    )


def _repo(root: Path, data_dir: Path, **files: str) -> dict[str, Path]:
    """A scratch repository with one committed paper/prereg/<id>-v1.toml per keyword."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    for key, value in (
        ("user.name", "Test"),
        ("user.email", "t@example.com"),
        ("core.autocrlf", "false"),
    ):
        _git(root, "config", key, value)
    folder = root / "paper" / "prereg"
    folder.mkdir(parents=True)
    paths = {}
    for prereg_id, pilot_run in files.items():
        paths[prereg_id] = folder / f"{prereg_id}-v1.toml"
        paths[prereg_id].write_bytes(_text(data_dir, prereg_id, pilot_run).encode())
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "preregs")
    return paths


def _score(path: Path, data_dir: Path, *extra: str) -> int:
    argv = ["score", "--prereg", str(path), "--data-dir", str(data_dir), "--seed", "3", *extra]
    return main(argv)


def test_score_writes_stamped_results_and_logs_every_cell(
    planted: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole chain on data whose answers are planted: AUC cells for both cadences, surrogate
    and alarm rows at the headline cell, static and dynamic FSS, all stamped with the frozen
    prereg and the inputs' completion hashes, one trial logged per cell. The planted pilot record
    predates kernel_sha256, so like any record that cannot be matched to the running kernel it is
    refused (#55: an old tau_c clock scored without a trace), and with --allow-stale-pilot it
    scores and the result says stale. #64: the alarm, surrogate and dynamic rows are held to the
    planted answers, not only counted: a slip in their wiring (the false-alarm budget per sample
    instead of per tau_c, the threshold set on the wrong half, censoring off) puts wrong numbers
    in the pre-registered results while every row still exists. Skewness, expected to fall, runs
    the falling path end to end; no skewness trend is planted, so it is detected only by chance."""
    data_dir = planted.root
    paths = _repo(tmp_path / "repo", data_dir, demo=PILOT_RUN)
    assert main(["prereg", "freeze", "demo", "--root", str(tmp_path / "repo")]) == 0
    assert _score(paths["demo"], data_dir) == 1
    assert "records no kernel_sha256" in capsys.readouterr().err
    assert prereg.trials(data_dir, prereg.load(paths["demo"])) == 0  # nothing scored or logged
    assert _score(paths["demo"], data_dir, "--allow-stale-pilot") == 0
    (run_dir,) = (data_dir / "ews").glob("*-demo-v1")
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    demo = prereg.load(paths["demo"])
    assert manifest["params"]["prereg"]["sha256"] == demo.sha256
    assert manifest["params"]["prereg"]["tag"] == "prereg/demo-v1"
    assert manifest["params"]["pilot"]["tau_c"] == {"16": 4.0, "32": 8.0}
    assert manifest["params"]["pilot"]["stale"] is True
    assert manifest["params"]["pilot"]["kernel_sha256"] is None
    inputs = {i["run_id"]: i for i in manifest["params"]["inputs"]}
    assert len(inputs) == 12  # 2 sizes x 2 seeds x (ramp, null, staircase)
    for tag, run_folder in planted.runs.items():
        digest = hashlib.sha256((run_folder / "complete.json").read_bytes()).hexdigest()
        assert inputs[run_folder.name]["complete_sha256"] == digest
        # each ramp and null names the pilot record it was made with (#55); staircases none
        assert inputs[run_folder.name]["pilot"] == planted.stamps.get(tag)
    auc = pq.read_table(run_dir / "auc" / f"{0:020d}.parquet").to_pylist()
    assert len(auc) == prereg.cells(demo.doc) == 48
    rise = {(r["cadence"], r["L"], r["h"]): r["auc"] for r in auc if r["indicator"] == "variance"}
    assert all(v > (0.85 if h == 0 else 0.6) for (_, _, h), v in rise.items()), rise  # planted
    surrogate = pq.read_table(run_dir / "surrogates" / f"{0:020d}.parquet").to_pylist()
    assert {(r["surrogate"], r["headline"]) for r in surrogate} == {("phase", True), ("ar1", True)}
    # with 19 surrogates p reaches alpha = 1/20 only for a series that beats every one of them
    assert len(surrogate) == 48 and {r["n_surrogates"] for r in surrogate} == {19}
    rising = [r["detected"] for r in surrogate if r["indicator"] == "variance" and r["h"] == 0]
    assert min(rising) >= 0.25 and np.mean(rising) >= 0.4, rising  # planted rise: 0.44-0.69
    null = [r["false_positive"] for r in surrogate]
    assert max(null) <= 0.25 and np.mean(null) <= 0.1, null  # about alpha: 0-0.11 planted
    falling = [r for r in surrogate if r["indicator"] == "skewness"]
    assert falling and all(r["direction"] == "less" and r["detected"] <= 0.25 for r in falling)
    alarms = pq.read_table(run_dir / "alarms" / f"{0:020d}.parquet").to_pylist()
    assert {(r["indicator"], r["cadence"], r["L"]) for r in alarms} == {
        (i, c, L) for i in ("variance", "ac") for c in ("dense", "matched") for L in (16, 32)
    }
    for r in alarms:
        assert math.isfinite(r["theta"]), r  # a budget 8 times too loose puts it at -inf
        # the rate theta was set at is at most the target of 2 per 100 tau_c, and close to it:
        # 1.90-2.00 planted, 1.39-1.79 when the budget is counted per score instead of per tau_c
        assert 1.8 <= r["far_set"] <= 2.0 + 1e-9 and 0.5 < r["far_test"] < 6, r
        # inside the band [2, -1]; NaN, every replica censored or missed, fails. Where in the
        # band depends on the draw: the band-entry rule is pinned on designed s(t) below
        assert -1.0 <= r["alarm_eps_median"] <= 2.0, r
        assert 0.0 < r["censored"] < 1.0 and r["miss_rate"] <= 0.25, r
    rows = pq.read_table(run_dir / "fss" / f"{0:020d}.parquet").to_pylist()
    fss = {r["quantity"]: r for r in rows}
    for quantity in ("l2_var_abs_m", "mean_abs_m"):  # planted at the exact exponents
        assert fss[quantity]["verdict"] == "pass", fss[quantity]
    assert fss["skewness_abs_m"]["estimate"] == pytest.approx(0.0, abs=0.2)
    assert fss["one_minus_ac1"]["verdict"] == "pass"  # a planted memory growing with L
    assert fss["tau_int_abs_m"]["verdict"] == ""  # descriptive: reported, never judged
    dynamic = [r for r in rows if r["role"] == "dynamic"]
    assert {(r["quantity"], r["cadence"]) for r in dynamic} == {
        (f"inverse_nu_{i}", c) for i in ("variance", "ac") for c in ("dense", "matched")
    }
    # the planted rise sits at one eps_L at both sizes (1/nu = 1); two seeds of 8 replicas are
    # too few to pass, so every CI holds 1 and none fails
    assert all(r["lo"] < 1.0 < r["hi"] and r["verdict"] != "fail" for r in dynamic), dynamic
    lines = prereg.trials_path(data_dir).read_text("utf-8").splitlines()
    logged = [json.loads(line) for line in lines]
    assert len(logged) == 48 and {t["result"] for t in logged} == {run_dir.name}
    assert all(t["prereg"]["sha256"] == demo.sha256 for t in logged)


def test_null_rows_are_scored_and_surrogate_tested_once_per_pipeline(
    planted: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#70: the null rows depend on h only through the span, which is mostly the same at every
    h (here everywhere but L = 16 matched, 230 and 231 samples), yet they were rebuilt, scored
    and tested against their surrogates once per h, more than half of the day the first real run
    would take. No input may go through the same pipeline, or the same pipeline and surrogate
    test, twice; and every pipeline still tests the nulls at its own span, so a cache that
    ignored the span would leave a pipeline with its positives tested and no null."""
    called: Counter[tuple[Pipeline, str]] = Counter()
    tested: Counter[tuple[Pipeline, Callable[..., np.ndarray], str]] = Counter()
    real_call, real_test = Pipeline.__call__, scoring.surrogate_test

    def digest(x: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64)).hexdigest()

    def call(pipe: Pipeline, x: np.ndarray) -> np.ndarray:
        called[(pipe, digest(x))] += 1
        return real_call(pipe, x)

    def surrogate_test(
        x: np.ndarray,
        pipe: Pipeline,
        make: Callable[..., np.ndarray],
        n: int,
        rng: np.random.Generator,
        direction: str = "greater",
    ) -> tuple[np.ndarray, np.ndarray]:
        tested[(pipe, make, digest(x))] += 1
        return real_test(x, pipe, make, n, rng, direction)

    monkeypatch.setattr(Pipeline, "__call__", call)
    monkeypatch.setattr(scoring, "surrogate_test", surrogate_test)
    doc = tomllib.loads(_text(planted.root, "demo", PILOT_RUN))
    doc["indicators"]["names"] = ["variance", "ac"]  # the call pattern needs neither skewness
    doc["scoring"]["surrogates"]["n"] = 4  # nor the full 19 surrogates
    evaluation = ising.evaluate(doc, planted.root, np.random.default_rng(3), allow_stale=True)
    assert len(evaluation.tables["surrogates"]) == 32  # 2 tests x 16 headline cells, as before
    assert max(called.values()) == 1, [k[0] for k, v in called.items() if v > 1]
    assert max(tested.values()) == 1, [k[0] for k, v in tested.items() if v > 1]
    per_pipeline = Counter((pipe, make) for pipe, make, _ in tested)
    assert min(per_pipeline.values()) >= 2  # its positives, and the nulls at its span


def test_a_falling_direction_reaches_the_surrogate_test_and_the_alarms(planted: Any) -> None:
    """The falling path on an answer known in advance: the planted variance rises, so read as an
    indicator expected to fall it never beats its surrogates (every p near 1, none at 1/20), and
    its alarms, raised on -s(t), mostly miss. A direction dropped on the way to the surrogate
    test (review round 1's bug, the one-sided test looking only for rises) detects about half
    the replicas here, as the rising reading does; dropped from the alarms, none miss."""
    doc = tomllib.loads(_text(planted.root, "demo", PILOT_RUN))
    doc["indicators"] |= {"names": ["variance"], "direction": {"variance": "less"}}
    doc["fss"]["dynamic"]["indicators"] = ["variance"]
    doc["event"]["horizons"] = [0.0]
    inputs = ising.select(*ising.find_runs(planted.root), doc["data"])[32]
    tau_c = ising.read_pilot(planted.root, PILOT_RUN, allow_stale=True).tau(32)
    tables: dict[str, list[dict[str, Any]]] = {"auc": [], "surrogates": [], "alarms": []}
    ising._cells(doc, 32, inputs, "dense", tau_c, tables, {}, np.random.default_rng(3))
    assert [(r["surrogate"], r["detected"]) for r in tables["surrogates"]] == [
        ("phase", 0.0),
        ("ar1", 0.0),
    ]
    (alarm,) = tables["alarms"]
    assert alarm["direction"] == "less" and alarm["miss_rate"] > 0.3, alarm  # rising: 0


def test_the_first_seed_sets_theta_and_the_second_tests_it(planted: Any) -> None:
    """#64: alternate seeds set theta and test its false-alarm rate. Swapped, every theta, rate
    and alarm position in the results moves while the planted ranges do not notice (one median
    alarm eps_L goes from 1.30 to 0.97). With the second seed's null blocks blanked, theta still
    comes from the first seed's and is finite, and the blank half raises no false alarm; set on
    the blank half, theta would be -inf."""
    doc = tomllib.loads(_text(planted.root, "demo", PILOT_RUN))
    ind, top, event = doc["indicators"], doc["sensitivity"]["headline"], doc["event"]
    band = (max(event["horizons"]) + event["segment"], min(event["horizons"]))
    inputs = ising.select(*ising.find_runs(planted.root), doc["data"])[16]
    tau_c = ising.read_pilot(planted.root, PILOT_RUN, allow_stale=True).tau(16)
    prep = ising.prepare(inputs, "dense", tau_c, ind["thin"], band)
    blank = [
        (r, replace(s, abs_m=np.full_like(s.abs_m, np.nan)) if r.seed == 2 else s)
        for r, s in prep.nulls
    ]
    window, bandwidth = prep.samples(top["window"], tau_c), top["bandwidth"] * tau_c / prep.sweeps
    rows, _ = ising._alarms(doc, replace(prep, nulls=blank), 16, tau_c, window, bandwidth)
    assert [r["indicator"] for r in rows] == ["variance", "ac"]
    assert all(math.isfinite(r["theta"]) and r["far_test"] == 0.0 for r in rows), rows


def test_an_alarm_is_a_rise_inside_the_band_and_one_before_it_is_censored(
    planted: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#64, the band-entry rule on designed s(t). The nulls score 0.5 throughout, so theta =
    0.5; the ramp scores at times j - 2 .. j + 2, j the first inside the band [2, -1]. A replica
    that rises above theta between j - 2 and j - 1 was above it as the band opened: censored,
    never an alarm at eps_L ~ 2. One that rises at j, the first score inside, alarms there.
    Reading the entry score one step too early (j - 2) turns the first into an alarm at the
    band's edge; one step too late (j) censors the second."""
    doc = tomllib.loads(_text(planted.root, "demo", PILOT_RUN))
    doc["fss"]["dynamic"]["indicators"] = ["variance"]
    ind, top, event = doc["indicators"], doc["sensitivity"]["headline"], doc["event"]
    band = (max(event["horizons"]) + event["segment"], min(event["horizons"]))
    inputs = ising.select(*ising.find_runs(planted.root), doc["data"])[16]
    tau_c = ising.read_pilot(planted.root, PILOT_RUN, allow_stale=True).tau(16)
    prep = ising.prepare(inputs, "dense", tau_c, ind["thin"], band)
    run, ramp = prep.ramps[0]
    eps = run.eps(ramp.temperature)
    j = int(np.flatnonzero(eps <= doc["scoring"]["alarms"]["band"][0])[0])
    rising = np.array([[0.0, 0.9, 0.9, 0.9, 0.9], [0.0, 0.0, 0.9, 0.9, 0.9]])
    designed = {
        id(s.abs_m): (np.arange(2), np.full((len(s.replicas), 2), 0.5)) for _, s in prep.nulls
    }
    designed[id(ramp.abs_m)] = (np.arange(j - 2, j + 3), rising)
    monkeypatch.setattr(ising, "trailing_score", lambda x, pipe, every=1: designed[id(x)])
    window, bandwidth = prep.samples(top["window"], tau_c), top["bandwidth"] * tau_c / prep.sweeps
    one_ramp = replace(prep, ramps=[(run, ramp)])
    (row,), found = ising._alarms(doc, one_ramp, 16, tau_c, window, bandwidth)
    (alarm,), censored = found["variance", "dense"]
    assert row["theta"] == 0.5 and censored == 0.5
    assert np.isnan(alarm[0]) and alarm[1] == pytest.approx(eps[j]) and eps[j - 1] > 2.0
    assert row["alarm_eps_median"] == pytest.approx(eps[j]) and row["miss_rate"] == 0.0


def test_score_refuses_unfrozen_over_budget_and_pilotless(
    planted: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing is scored, written or logged from a prereg that is not frozen, whose pilot run
    holds no pilot.json, or whose trial budget is spent."""
    data_dir = tmp_path / "data"
    pilot_dir = data_dir / "ising" / PILOT_RUN
    pilot_dir.mkdir(parents=True)
    source = planted.root / "ising" / PILOT_RUN / "pilot.json"
    (pilot_dir / "pilot.json").write_bytes(source.read_bytes())
    root = tmp_path / "repo"
    paths = _repo(root, data_dir, nopilot="absent-run", spent=PILOT_RUN)
    assert _score(paths["spent"], data_dir) == 1
    assert "not frozen" in capsys.readouterr().err
    for prereg_id in ("nopilot", "spent"):
        assert main(["prereg", "freeze", prereg_id, "--root", str(root)]) == 0
    assert _score(paths["nopilot"], data_dir) == 1
    assert "pilot.json" in capsys.readouterr().err
    spent = prereg.load(paths["spent"])
    budget = spent.doc["trials"]["budget"]
    for _ in range(budget):
        prereg.log_trial(data_dir, {"prereg": {"id": "spent", "version": 1}})
    assert _score(paths["spent"], data_dir) == 1
    assert "budget" in capsys.readouterr().err
    assert not list((data_dir / "ews").glob("*-v1"))  # no result folder for any of them
    assert prereg.trials(data_dir, spent) == budget
