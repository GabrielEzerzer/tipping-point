"""Pre-registration files: the committed ones validate, a freeze binds a file to a git tag, and
results and trials are stamped with what was frozen."""

import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.ews import prereg
from tipping_point.ising import io, protocol, run

REPO = Path(__file__).resolve().parents[4]
V1 = REPO / "paper" / "prereg" / "ising-ews-v1.toml"
TEMPLATE = REPO / "paper" / "prereg" / "TEMPLATE.toml"
FIXTURE = REPO / "python" / "tipping_point" / "tests" / "fixtures" / "ising_pilot.json"


def _ready(text: str, prereg_id: str = "demo") -> str:
    """v1 under a new id: a file that may be frozen."""
    return text.replace('id = "ising-ews"', f'id = "{prereg_id}"', 1)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A scratch repository holding paper/prereg/demo-v1.toml, committed."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "core.autocrlf", "false")
    folder = tmp_path / "paper" / "prereg"
    folder.mkdir(parents=True)
    (folder / "demo-v1.toml").write_bytes(_ready(V1.read_text("utf-8")).encode())
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "demo prereg")
    return tmp_path


def test_committed_files_validate_and_v1_may_be_frozen() -> None:
    """The template and v1 parse with every key the scorer reads. v1 passes the freeze's strict
    check, nothing left as a placeholder now that its pilot record is pinned; the template does
    not. A fourth size would need a budget to match."""
    for path in (TEMPLATE, V1):
        prereg.load(path)
    assert prereg.problems(prereg.load(V1).doc, strict=True) == []
    assert any("placeholder" in p for p in prereg.problems(prereg.load(TEMPLATE).doc, strict=True))
    grown = tomllib.loads(V1.read_text("utf-8"))
    grown["data"]["L"].append(128)  # a fourth size without a budget to match
    assert any("trials.budget" in p for p in prereg.problems(grown, strict=True))


def test_v1_holds_the_research_pass_protocol() -> None:
    """The numbers the brief fixes (#26 sections 2-4, #50's relabelled ramps)."""
    doc = prereg.load(V1).doc
    assert (doc["id"], doc["version"]) == ("ising-ews", 1)
    assert {k: doc["data"]["ramps"][k] for k in ("kind", "x")} == {"kind": "ramp", "x": 0.02}
    assert doc["data"]["nulls"]["kind"] == "fixed" and doc["data"]["nulls"]["eps_L"] == [2, 4, 6]
    assert doc["data"]["tau_c"]["source"] == "pilot"  # never numbers in the file
    assert doc["event"]["horizons"] == [0, 1, 2] and doc["event"]["segment"] == 6
    assert "t_c" in doc["event"]["definition"]
    assert doc["indicators"]["thin"] == 0.1 and doc["indicators"]["observables"][0] == "abs_m"
    sens = doc["sensitivity"]
    assert (sens["window"], sens["bandwidth"]) == ([10, 20, 40], [5, 10, 20])
    assert sens["headline"] == {"window": 20, "bandwidth": 10}
    static = {row["quantity"]: row for row in doc["fss"]["static"]}
    assert static["l2_var_abs_m"]["expected"] == 1.75 and static["l2_var_abs_m"]["tolerance"] == 0.1
    assert static["mean_abs_m"]["tolerance"] == 0.02
    assert static["one_minus_ac1"]["role"] == "reference"  # AC1 against its measured kappa, not z
    assert doc["scoring"]["surrogates"]["n"] == 999 and doc["scoring"]["bootstrap"] == 2000
    assert doc["trials"]["budget"] == prereg.cells(doc) == 972


def test_v1_holds_the_freeze_decisions() -> None:
    """#26 comment 6402030372: new seeds and replicas, ramps over one eps_L window, the
    dynamic test on Var(|m|) peak positions at 0.5, alarms as a reported table at 1 per 1000
    tau_c, every judged static entry with its own tolerance (1 - AC1 keeps the 0.1 it took from
    the dynamic test), h = 2 at every L, the disclosure of what was seen before, and #54's
    tau_c record pinned: the sha256 of the committed fixture, so anyone can check it."""
    doc = prereg.load(V1).doc
    data, dyn = doc["data"], doc["fss"]["dynamic"]
    assert data["tau_c"]["pilot_run"] == "20261025T194846Z-tauc-L16-32-64-128"
    assert data["tau_c"]["pilot_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert (data["L"], data["seeds"], data["replicas"]) == ([16, 32, 64], [6, 7, 8, 9, 10], 64)
    assert data["ramps"]["eps_L"] == [12.0, -8.0]
    assert (dyn["statistic"], dyn["indicators"], dyn["search"]) == ("peak", ["variance"], [6, -4])
    assert (dyn["expected_inverse_nu"], dyn["tolerance"]) == (1.0, 0.5)
    assert (dyn["cadence"], dyn["min_draw_share"]) == ("dense", 0.95)  # fix round 1: R1, R2
    dynamic_fss = next(h for h in doc["hypotheses"] if h["name"] == "dynamic-fss")["prediction"]
    for clause in (
        "robustness check",
        "half-width is at most 0.5",
        "draws_used <",
        "in this order",
    ):
        assert clause in dynamic_fss, clause  # R3: the frozen text states the rule the code runs
    alarms = doc["scoring"]["alarms"]
    assert (alarms["indicators"], alarms["far"]) == (["variance", "ac", "skewness"], 0.001)
    assert (alarms["report_per"], alarms["band"], alarms["every"]) == (1000.0, [2, -2], 1.0)
    tolerance = {row["quantity"]: row.get("tolerance") for row in doc["fss"]["static"]}
    assert tolerance == {
        "l2_var_abs_m": 0.1,
        "mean_abs_m": 0.02,
        "skewness_abs_m": 0.1,
        "tau_int_abs_m": None,  # descriptive: never judged
        "one_minus_ac1": 0.1,
    }
    assert doc["event"]["horizons"] == [0, 1, 2]  # h = 2 is legal at L = 16 now
    shown = doc["disclosure"]
    assert "seeds 1-5" in shown["exploratory"] and "L = 16 and 32" in shown["exploratory"]
    assert "in view" in shown["decisions"] and "from the tagged commit" in shown["confirmatory"]
    comments = {link.rsplit("-", 1)[1] for link in shown["links"]}
    assert comments == {"6402030372", "6387307847", "6390788147", "6393467391"}


def test_v1_says_what_the_file_itself_binds() -> None:
    """Freeze review, round 1: the text and the numbers beside it disagreed, and only a reader
    could tell. The file binds grid pilots on a scored seed, so "no earlier run" was false: it must
    name them, and the only freeze-time claim it may make is the one the scorer can check, the
    tagged commit. A null run holds two slices the length of a positive segment at every window of
    the grid, not one: protocol.NULL_TAU_C against segment / x plus the window."""
    raw = V1.read_text("utf-8")
    doc = tomllib.loads(raw)
    earlier = {int(seed) for seed in re.findall(r"-pilot-s(\d+)", raw)}
    assert earlier & set(doc["data"]["seeds"])  # the pilots sit on a seed v1 scores: seed 6
    for false_claim in ("no earlier run used", "used by no earlier run", "after the freeze tag"):
        assert false_claim not in raw, false_claim
    confirmatory = doc["disclosure"]["confirmatory"]
    assert "three grid pilots under data.staircase" in confirmatory
    assert "set the dwell and are never scored" in confirmatory
    assert "derived from its seed and its identity" in confirmatory  # run.stream_key
    span = doc["event"]["segment"] / doc["data"]["ramps"]["x"]  # tau_c: 6 eps_L x 50 tau_c
    assert {protocol.NULL_TAU_C // (span + w) for w in doc["sensitivity"]["window"]} == {2}
    (nulls,) = [line for line in raw.splitlines() if line.startswith("eps_L = [2.0, 4.0, 6.0]")]
    assert "two slices the length of a positive segment" in nulls


@pytest.mark.parametrize(
    ("edit", "complaint"),
    [
        (lambda d: d.pop("question"), "question: missing"),
        (lambda d: d["data"].update(L="16"), "data.L: expected an array"),
        (lambda d: d["hypotheses"][0].update(direction="up"), "direction"),
        (lambda d: d["sensitivity"]["headline"].update(window=30.0), "sensitivity.headline"),
        (lambda d: d["fss"]["static"][0].pop("tolerance"), "positive control needs"),
        (lambda d: d.update(supersedes="ising-ews-v1"), "reason: an amendment"),
        (lambda d: d["indicators"].update(names=["kurtosis"]), "indicators.names"),
        (lambda d: d["scoring"].update(bootstrap=True), "scoring.bootstrap: expected int"),
        (lambda d: d["data"]["tau_c"].update(source="fit"), "never fitted"),
        (lambda d: d["data"].update(L=[16]), "data.L: at least two sizes"),
        (lambda d: d["indicators"].update(detrend="none"), "ar1 needs detrending"),
        (lambda d: d["indicators"]["direction"].pop("skewness"), "indicators.direction.skewness"),
        (lambda d: d["scoring"]["surrogates"].update(primary="fourier"), "surrogates.primary"),
        # the freeze's keys (#26 comment 6402030372)
        (lambda d: d["fss"]["static"][4].pop("tolerance"), "a reference needs its own tolerance"),
        (lambda d: d["fss"]["dynamic"].pop("statistic"), "fss.dynamic.statistic: missing"),
        (lambda d: d["fss"]["dynamic"].update(statistic="alarm"), "fss.dynamic.statistic: one of"),
        (lambda d: d["fss"]["dynamic"].pop("search"), "fss.dynamic.search: missing"),
        (lambda d: d["fss"]["dynamic"].update(search=[-4.0, 6.0]), "the hot edge first"),
        (lambda d: d["fss"]["dynamic"].update(search=[6.0]), "fss.dynamic.search: [from"),
        (lambda d: d["fss"]["dynamic"].update(search=["6", "-4"]), "search[0]: expected int"),
        (lambda d: d["scoring"]["alarms"].pop("indicators"), "scoring.alarms.indicators: missing"),
        (lambda d: d["scoring"]["alarms"].update(indicators=["kurtosis"]), "must be scored"),
        (lambda d: d["data"]["ramps"].update(eps_L=[-8.0, 12.0]), "start above end"),
        (lambda d: d["data"]["ramps"].update(eps_L=[12.0]), "data.ramps.eps_L: [start, end]"),
        (lambda d: d["data"]["ramps"].update(eps_L=[True, False]), "data.ramps.eps_L"),
        (lambda d: d["data"]["ramps"].update(eps_L="12, -8"), "data.ramps.eps_L"),
        (lambda d: d.pop("disclosure"), "disclosure: missing"),
        (lambda d: d["disclosure"].pop("confirmatory"), "disclosure.confirmatory: missing"),
        (lambda d: d["disclosure"].update(links="#26"), "disclosure.links: expected an array"),
        # fix round 1: one verdict's cadence (R1), the draws it needs (R2), the band (R4)
        (lambda d: d["fss"]["dynamic"].pop("cadence"), "fss.dynamic.cadence: missing"),
        (lambda d: d["fss"]["dynamic"].update(cadence="hourly"), "fss.dynamic.cadence: one of"),
        (lambda d: d["indicators"].update(cadences=["matched"]), "fss.dynamic.cadence: one of"),
        (lambda d: d["fss"]["dynamic"].pop("min_draw_share"), "min_draw_share: missing"),
        (lambda d: d["fss"]["dynamic"].update(min_draw_share=0.0), "min_draw_share: a share"),
        (lambda d: d["fss"]["dynamic"].update(min_draw_share=1.5), "min_draw_share: a share"),
        (lambda d: d["fss"]["dynamic"].update(min_draw_share="95%"), "expected int or float"),
        (lambda d: d["data"]["ramps"].update(eps_L=[5.0, -8.0]), "must lie inside data.ramps"),
        (lambda d: d["data"]["ramps"].update(eps_L=[12.0, -3.0]), "must lie inside data.ramps"),
    ],
)
def test_malformed_files_are_named(edit: Any, complaint: str) -> None:
    doc = tomllib.loads(V1.read_text("utf-8"))
    edit(doc)
    assert any(complaint in problem for problem in prereg.problems(doc)), prereg.problems(doc)


def test_the_search_band_may_reach_the_ramp_windows_ends() -> None:
    """R4's boundary: a search band that touches the ramp window's ends lies inside it, and
    without data.ramps.eps_L any band is accepted."""
    doc = tomllib.loads(V1.read_text("utf-8"))
    doc["data"]["ramps"]["eps_L"] = list(doc["fss"]["dynamic"]["search"])
    assert prereg.problems(doc) == []
    del doc["data"]["ramps"]["eps_L"]
    doc["fss"]["dynamic"]["search"] = [20.0, -20.0]
    assert prereg.problems(doc) == []


def test_freeze_binds_the_file_and_results_carry_the_stamp(tmp_path: Path, repo: Path) -> None:
    """Unfrozen, a file cannot be scored; frozen, the tag is annotated and on the committed file,
    results are stamped with it in the shared layout (manifest, one Parquet file per non-empty
    table keyed by cell, complete.json hashes); edited after the freeze, it is refused."""
    path = repo / "paper" / "prereg" / "demo-v1.toml"
    with pytest.raises(prereg.PreregError, match="not frozen"):
        prereg.frozen(prereg.load(path))
    tag = prereg.freeze(repo, "demo")
    assert tag == "prereg/demo-v1"
    assert _git(repo, "cat-file", "-t", tag) == "tag"  # annotated: it carries its own date
    demo = prereg.load(path)
    ref = prereg.frozen(demo)
    assert ref.tag == tag and ref.commit == _git(repo, "rev-parse", "HEAD")
    inputs = [{"domain": "ising", "run_id": "r1", "complete_sha256": "c" * 64}]
    tables = {"auc": [{"L": 16, "auc": 0.9}, {"L": 32, "auc": 0.8}], "alarms": []}
    data_dir = tmp_path / "data"
    run_dir = prereg.write_result(data_dir, demo, ref, tables, inputs, {"pilot": {"run_id": "p"}})
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    assert manifest["domain"] == "ews" and manifest["key"] == "cell"
    assert manifest["tables"] == ["auc"] and manifest["params"]["inputs"] == inputs
    stamp = {"id": "demo", "version": 1, "sha256": demo.sha256, "tag": tag, "commit": ref.commit}
    assert manifest["params"]["prereg"] == stamp
    assert manifest["params"]["pilot"] == {"run_id": "p"}
    table = pq.read_table(run_dir / "auc" / f"{0:020d}.parquet").to_pydict()
    assert table["cell"] == [0, 1] and table["auc"] == [0.9, 0.8]
    done = json.loads((run_dir / "complete.json").read_text("utf-8"))
    assert done["rows"] == {"auc": 2} and done["last_key"] == 1 and done["clean"]
    for relative, digest in done["files"].items():
        assert hashlib.sha256((run_dir / relative).read_bytes()).hexdigest() == digest
    path.write_bytes(path.read_bytes().replace(b"x = 0.02", b"x = 0.03"))
    with pytest.raises(prereg.PreregError, match="differs from the version frozen"):
        prereg.frozen(prereg.load(path))


def test_runs_and_results_take_their_commit_from_one_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches #88: an Ising run's producer and the EWS result scored from it each ran their own
    copy of git rev-parse and status, so a fix to one copy (say, untracked files no longer counted
    as dirty) would leave one result with two definitions of a dirty checkout. Both read
    io.code_version: a value no checkout gives reaches both manifests."""
    planted = {"git_sha": "f" * 40, "dirty": True}
    monkeypatch.setattr(io, "code_version", lambda: dict(planted))
    spec, schedule = run.Spec(L=4, replicas=1, seed=1), protocol.Fixed((2.0,), 1, 1)
    ising = run._manifest(spec, schedule, "r", torch.device("cpu"), None)["producer"]
    tables = {"auc": [{"L": 16, "auc": 0.5}]}
    result = prereg.write_result(tmp_path, prereg.load(V1), prereg.Frozen("t", "c"), tables, [], {})
    ews = json.loads((result / "manifest.json").read_text("utf-8"))["producer"]
    for producer in (ising, ews):
        assert {key: producer[key] for key in planted} == planted, producer


def test_freeze_refuses_what_it_should(repo: Path) -> None:
    path = repo / "paper" / "prereg" / "demo-v1.toml"
    original = path.read_bytes()
    path.write_bytes(original + b"\n# an edit not yet committed\n")
    with pytest.raises(prereg.PreregError, match="differs from HEAD"):
        prereg.freeze(repo, "demo")
    path.write_bytes(original)
    prereg.freeze(repo, "demo")
    with pytest.raises(prereg.PreregError, match="never re-tagged"):
        prereg.freeze(repo, "demo")
    (repo / "paper" / "prereg" / "demo-v2.toml").write_bytes(
        original.replace(b"version = 1", b"version = 2")
    )
    with pytest.raises(prereg.PreregError, match="not committed"):
        prereg.freeze(repo, "demo")  # the latest version is the one frozen
    unready = _ready(V1.read_text("utf-8"), "raw").replace('pilot_run = "', 'pilot_run = "TBD', 1)
    (repo / "paper" / "prereg" / "raw-v1.toml").write_bytes(unready.encode())
    with pytest.raises(prereg.PreregError, match="placeholder"):
        prereg.freeze(repo, "raw")
    with pytest.raises(prereg.PreregError, match="no paper/prereg"):
        prereg.freeze(repo, "absent")


def test_trials_append_and_count_per_version(tmp_path: Path) -> None:
    v1 = prereg.load(V1)
    for i in range(3):
        prereg.log_trial(tmp_path, {"prereg": {"id": "ising-ews", "version": 1}, "cell": i})
    prereg.log_trial(tmp_path, {"prereg": {"id": "ising-ews", "version": 2}, "cell": 0})
    lines = prereg.trials_path(tmp_path).read_bytes().split(b"\n")
    assert len(lines) == 5 and lines[-1] == b"" and b"\r" not in lines[0]
    assert all("logged_ns" in json.loads(line) for line in lines[:-1])
    assert prereg.trials(tmp_path, v1) == 3
