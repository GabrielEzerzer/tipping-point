"""Pre-registration files: the committed ones validate, a freeze binds a file to a git tag, and
results and trials are stamped with what was frozen."""

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from tipping_point.ews import prereg

REPO = Path(__file__).resolve().parents[4]
V1 = REPO / "paper" / "prereg" / "ising-ews-v1.toml"
TEMPLATE = REPO / "paper" / "prereg" / "TEMPLATE.toml"
PILOT_SHA = "ab" * 32


def _ready(text: str, prereg_id: str = "demo") -> str:
    """v1 with its pilot pinned and a new id: a file that may be frozen."""
    text = text.replace('id = "ising-ews"', f'id = "{prereg_id}"', 1)
    text = text.replace('pilot_run = "TBD"', 'pilot_run = "20261023T000000Z-L16-pilot-s1"', 1)
    return text.replace('pilot_sha256 = "TBD"', f'pilot_sha256 = "{PILOT_SHA}"', 1)


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


def test_committed_files_validate_and_v1_waits_only_for_its_pilot() -> None:
    """The template and v1 parse with every key the scorer reads. v1 cannot be frozen yet, and
    the only things missing are the #50 pilot record's run and hash."""
    for path in (TEMPLATE, V1):
        prereg.load(path)
    v1 = prereg.load(V1)
    missing = prereg.problems(v1.doc, strict=True)
    assert missing and all(m.startswith("data.tau_c.pilot_") for m in missing), missing
    assert not prereg.problems(tomllib.loads(_ready(V1.read_text("utf-8"))), strict=True)


def test_v1_holds_the_research_pass_protocol() -> None:
    """The numbers the brief fixes (#26 sections 2-4, #50's relabelled ramps)."""
    doc = prereg.load(V1).doc
    assert (doc["id"], doc["version"]) == ("ising-ews", 1)
    assert doc["data"]["ramps"] == {"kind": "ramp", "x": 0.02}
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
    ],
)
def test_malformed_files_are_named(edit: Any, complaint: str) -> None:
    doc = tomllib.loads(V1.read_text("utf-8"))
    edit(doc)
    assert any(complaint in problem for problem in prereg.problems(doc)), prereg.problems(doc)


def test_freeze_tags_the_commit_and_scoring_checks_the_blob(repo: Path) -> None:
    """Catches a freeze that tags the wrong thing, or a check that would accept an edit made
    after the freeze."""
    tag = prereg.freeze(repo, "demo")
    assert tag == "prereg/demo-v1"
    assert _git(repo, "cat-file", "-t", tag) == "tag"  # annotated: it carries its own date
    path = repo / "paper" / "prereg" / "demo-v1.toml"
    ref = prereg.frozen(prereg.load(path))
    assert ref.tag == tag and ref.commit == _git(repo, "rev-parse", "HEAD")
    path.write_bytes(path.read_bytes().replace(b"x = 0.02", b"x = 0.03"))
    with pytest.raises(prereg.PreregError, match="differs from the version frozen"):
        prereg.frozen(prereg.load(path))


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
    unready = V1.read_text("utf-8").replace('id = "ising-ews"', 'id = "raw"').encode()
    (repo / "paper" / "prereg" / "raw-v1.toml").write_bytes(unready)
    with pytest.raises(prereg.PreregError, match="placeholder"):
        prereg.freeze(repo, "raw")
    with pytest.raises(prereg.PreregError, match="no paper/prereg"):
        prereg.freeze(repo, "absent")


def test_an_unfrozen_file_cannot_be_scored(repo: Path) -> None:
    with pytest.raises(prereg.PreregError, match="not frozen"):
        prereg.frozen(prereg.load(repo / "paper" / "prereg" / "demo-v1.toml"))


def test_trials_append_and_count_per_version(tmp_path: Path, repo: Path) -> None:
    demo = prereg.load(repo / "paper" / "prereg" / "demo-v1.toml")
    stamp = {"id": "demo", "version": 1}
    for i in range(3):
        prereg.log_trial(tmp_path, {"prereg": stamp, "cell": i})
    prereg.log_trial(tmp_path, {"prereg": {"id": "demo", "version": 2}, "cell": 0})
    lines = prereg.trials_path(tmp_path).read_bytes().split(b"\n")
    assert len(lines) == 5 and lines[-1] == b"" and b"\r" not in lines[0]
    assert all("logged_ns" in json.loads(line) for line in lines[:-1])
    assert prereg.trials(tmp_path, demo) == 3


def test_results_follow_the_shared_layout_and_carry_the_stamp(tmp_path: Path, repo: Path) -> None:
    """manifest.json stamped with the prereg and inputs, one Parquet file per non-empty table
    keyed by cell, and complete.json whose hashes match the files."""
    demo = prereg.load(repo / "paper" / "prereg" / "demo-v1.toml")
    prereg.freeze(repo, "demo")
    ref = prereg.frozen(demo)
    inputs = [{"domain": "ising", "run_id": "r1", "complete_sha256": "c" * 64}]
    tables = {"auc": [{"L": 16, "auc": 0.9}, {"L": 32, "auc": 0.8}], "alarms": []}
    run_dir = prereg.write_result(tmp_path, demo, ref, tables, inputs, {"pilot": {"run_id": "p"}})
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    assert manifest["domain"] == "ews" and manifest["key"] == "cell"
    assert manifest["tables"] == ["auc"] and manifest["params"]["inputs"] == inputs
    assert manifest["params"]["prereg"] == {
        "id": "demo", "version": 1, "sha256": demo.sha256, "tag": "prereg/demo-v1",
        "commit": ref.commit,
    }  # fmt: skip
    assert manifest["params"]["pilot"] == {"run_id": "p"}
    table = pq.read_table(run_dir / "auc" / f"{0:020d}.parquet")
    assert table.column("cell").to_pylist() == [0, 1] and table.column("auc").to_pylist() == [
        0.9,
        0.8,
    ]
    done = json.loads((run_dir / "complete.json").read_text("utf-8"))
    assert done["rows"] == {"auc": 2} and done["last_key"] == 1 and done["clean"]
    for relative, digest in done["files"].items():
        assert hashlib.sha256((run_dir / relative).read_bytes()).hexdigest() == digest
