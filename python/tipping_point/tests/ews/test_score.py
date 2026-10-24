"""`score` end to end on the planted dataset: a frozen pre-registration in a scratch repository,
the result folder it writes, the trials it logs, and the refusals that keep it honest."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from ews_planted import NULL_EPS, PILOT_RUN, PLATEAU
from tipping_point.ews import prereg
from tipping_point.ews.__main__ import main

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
names = ["variance", "ac"]
observables = ["abs_m"]
cadences = ["dense", "matched"]
thin = 0.5
lag = 1
detrend = "gaussian"
direction = {{ variance = "greater", ac = "greater" }}

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
n = 4
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
budget = 32
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
    pilot = data_dir / "ising" / PILOT_RUN / "pilot.json"
    digest = hashlib.sha256(pilot.read_bytes()).hexdigest()
    paths = {}
    for prereg_id, pilot_run in files.items():
        text = PREREG.format(
            id=prereg_id,
            pilot_run=pilot_run,
            pilot_sha256=digest,
            eps=list(NULL_EPS),
            plateau=PLATEAU,
        )
        paths[prereg_id] = folder / f"{prereg_id}-v1.toml"
        paths[prereg_id].write_bytes(text.encode())
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
    scores and the result says stale."""
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
    assert len(auc) == prereg.cells(demo.doc) == 32
    rise = {(r["cadence"], r["L"], r["h"]): r["auc"] for r in auc if r["indicator"] == "variance"}
    assert all(v > (0.85 if h == 0 else 0.6) for (_, _, h), v in rise.items()), rise  # planted
    surrogate = pq.read_table(run_dir / "surrogates" / f"{0:020d}.parquet").to_pylist()
    assert {(r["surrogate"], r["headline"]) for r in surrogate} == {("phase", True), ("ar1", True)}
    alarms = pq.read_table(run_dir / "alarms" / f"{0:020d}.parquet").to_pylist()
    assert {(r["indicator"], r["cadence"], r["L"]) for r in alarms} == {
        (i, c, L) for i in ("variance", "ac") for c in ("dense", "matched") for L in (16, 32)
    }
    fss = {
        r["quantity"]: r for r in pq.read_table(run_dir / "fss" / f"{0:020d}.parquet").to_pylist()
    }
    for quantity in ("l2_var_abs_m", "mean_abs_m"):  # planted at the exact exponents
        assert fss[quantity]["verdict"] == "pass", fss[quantity]
    assert fss["skewness_abs_m"]["estimate"] == pytest.approx(0.0, abs=0.2)
    assert fss["one_minus_ac1"]["verdict"] == "pass"  # a planted memory growing with L
    assert fss["tau_int_abs_m"]["verdict"] == ""  # descriptive: reported, never judged
    assert {"inverse_nu_variance", "inverse_nu_ac"} <= set(fss)
    lines = prereg.trials_path(data_dir).read_text("utf-8").splitlines()
    logged = [json.loads(line) for line in lines]
    assert len(logged) == 32 and {t["result"] for t in logged} == {run_dir.name}
    assert all(t["prereg"]["sha256"] == demo.sha256 for t in logged)


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
    for _ in range(32):
        prereg.log_trial(data_dir, {"prereg": {"id": "spent", "version": 1}})
    assert _score(paths["spent"], data_dir) == 1
    assert "budget" in capsys.readouterr().err
    assert not list((data_dir / "ews").glob("*-v1"))  # no result folder for any of them
    assert prereg.trials(data_dir, prereg.load(paths["spent"])) == 32
