"""The schema of a market-labels pre-registration (#29 S0): the template loads with its
placeholders and holds the labeller's and the cascade rule's own parameters, every problem with a
file is named, the freeze refuses a file until no placeholder is left, and a filled file goes
through freeze, frozen, stamp and the trial log like an Ising one."""

import copy
import dataclasses
import re
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from tipping_point.ews import __main__ as ews_cli
from tipping_point.ews import prereg
from tipping_point.market import __main__ as market_cli
from tipping_point.market import archive, cascade, msvol

REPO = Path(__file__).resolve().parents[4]
TEMPLATE = REPO / "paper" / "prereg" / "TEMPLATE-market-labels.toml"
ISING = REPO / "paper" / "prereg" / "ising-ews-v1.toml"
DELETE = object()


def _ready(text: str, prereg_id: str = "market-labels") -> str:
    """The template under a real id, its other placeholders kept."""
    return text.replace('id = "<lowercase-slug>"', f'id = "{prereg_id}"', 1)


def _filled(text: str) -> str:
    """The template with every placeholder set: a file that may be frozen."""
    text = re.sub(r'rule = "<[^"]*>"', 'rule = "refractory-from-episodes"', _ready(text))
    text = re.sub(r'smallest = "<[^"]*>"', "smallest = 5", text)
    return re.sub(r'"<[^"]*>"', '"set"', text)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _repo(root: Path, text: str) -> Path:
    """A scratch repository holding paper/prereg/market-labels-v1.toml, committed."""
    root.mkdir(exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "core.autocrlf", "false")
    path = root / "paper" / "prereg" / "market-labels-v1.toml"
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode())
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "market prereg")
    return path


@pytest.fixture
def doc() -> dict[str, Any]:
    return copy.deepcopy(prereg.load(TEMPLATE).doc)


def test_the_template_loads_with_its_placeholders(doc: dict[str, Any]) -> None:
    """A draft: load takes it, and the freeze's strict check names each placeholder, nothing else.
    nulls.rule is the choice the S5 pilot read left to the owner."""
    assert prereg.problems(doc) == []
    strict = prereg.problems(doc, strict=True)
    assert all(": placeholder " in p for p in strict), strict
    assert {p.split(":")[0] for p in strict} == {
        "id",
        "question",
        "nulls.rule",
        "horizon.smallest",
        "disclosure.exploratory",
        "disclosure.decisions",
        "disclosure.confirmatory",
        "deviations.policy",
    }


def test_the_template_holds_the_code_parameters(doc: dict[str, Any]) -> None:
    """[labeller] and [cascade] are the labeller's and the rule's own fields at their start values,
    so a frozen file builds the Params the code runs, and the span's boundary is the one the market
    command refuses to label past."""
    assert doc["labeller"] == dataclasses.asdict(msvol.Params())
    assert doc["cascade"] == dataclasses.asdict(cascade.Params())
    assert msvol.Params(**doc["labeller"]) == msvol.Params()
    assert cascade.Params(**doc["cascade"]) == cascade.Params()
    assert prereg.MARKET_SYMBOLS == archive.SYMBOLS
    assert doc["data"]["symbols"] == list(archive.SYMBOLS)
    span = doc["data"]["span"]
    assert span["confirmatory"] == market_cli.CONFIRMATORY
    assert (span["confirmatory"] - span["start"]).days > doc["labeller"]["warmup_days"]
    assert doc["bars"] == {"estimator": "garman-klass", "missing": prereg.MISSING}


def test_a_filled_template_goes_through_freeze_frozen_stamp_and_the_trial_log(
    tmp_path: Path,
) -> None:
    """The round trip: the filled template passes the strict check, freezes into its tag, is
    frozen at it, and is stamped and counted in the trial log by its id and version."""
    path = _repo(tmp_path / "repo", _filled(TEMPLATE.read_text("utf-8")))
    pre = prereg.load(path)
    assert prereg.problems(pre.doc, strict=True) == []
    assert prereg.freeze(path.parents[2], "market-labels") == "prereg/market-labels-v1"
    ref = prereg.frozen(pre)
    assert ref.commit == _git(path.parents[2], "rev-parse", "HEAD")
    stamp = prereg.stamp(pre, ref)
    assert stamp == {
        "id": "market-labels",
        "version": 1,
        "sha256": pre.sha256,
        "tag": "prereg/market-labels-v1",
        "commit": ref.commit,
    }
    prereg.log_trial(tmp_path / "data", {"prereg": stamp})
    assert prereg.trials(tmp_path / "data", pre) == 1


@pytest.mark.parametrize("left", ["nulls.rule", "horizon.smallest", "question"])
def test_the_freeze_refuses_a_placeholder(tmp_path: Path, left: str) -> None:
    """Every placeholder set but one: the freeze names it and tags nothing."""
    filled = _filled(TEMPLATE.read_text("utf-8"))
    kept = {
        "nulls.rule": ('rule = "refractory-from-episodes"', 'rule = "<settled by the owner>"'),
        "horizon.smallest": ("smallest = 5", 'smallest = "TBD"'),
        "question": ('question = "set"', 'question = "<one sentence>"'),
    }[left]
    path = _repo(tmp_path, filled.replace(*kept, 1))
    prereg.load(path)  # a draft loads
    with pytest.raises(prereg.PreregError, match="not ready to freeze") as refused:
        prereg.freeze(tmp_path, "market-labels")
    assert f"{left}: placeholder" in str(refused.value)
    assert _git(tmp_path, "tag", "--list") == ""


def _set(doc: dict[str, Any], where: str, value: Any) -> dict[str, Any]:
    *parents, key = where.split(".")
    table = doc
    for part in parents:
        table = table[part]
    if value is DELETE:
        del table[key]
    else:
        table[key] = value
    return doc


@pytest.mark.parametrize(
    ("where", "value", "named"),
    [
        # missing keys
        ("labeller.d_min", DELETE, "labeller.d_min: missing"),
        ("data.span", DELETE, "data.span: missing"),
        ("nulls", DELETE, "nulls: missing"),
        # wrong types
        ("labeller.d_min", "12", "labeller.d_min: expected int, got '12'"),
        ("labeller.enter", True, "labeller.enter: expected int or float, got True"),
        ("data.symbols", "BTCUSDT", "data.symbols: expected an array"),
        ("data.span.start", "2022-07-01", "data.span.start: expected date"),
        ("cascade", 3, "cascade: expected a table"),
        ("horizon.smallest", 2.5, "horizon.smallest: expected int or str"),
        # out of range
        ("labeller.d_min", 24, "labeller.d_min: one of [6, 12, 36]"),
        ("labeller.enter", 0.75, "labeller.enter, exit: one of"),
        ("labeller.baseline_days", 30, "labeller.baseline_days: one of [14, 28, 56]"),
        ("labeller.start_p", 1.0, "labeller.start_p: a probability"),
        ("labeller.start_low", 80.0, "labeller.start_low, start_high"),
        ("labeller.min_share", 0.5, "labeller.min_share"),
        ("labeller.refractory", 0, "labeller.refractory: at least 1"),
        ("labeller.tol", 0.0, "labeller.tol: positive"),
        ("labeller.warmup_days", 600, "data.span: start, then labeller.warmup_days"),
        ("data.span.end", date(2023, 12, 31), "data.span.end: on or after confirmatory"),
        ("data.span.start", datetime(2022, 7, 1), "data.span.start: a UTC day"),
        ("data.symbols", ["BTCUSDT", "SOLUSDT"], "data.symbols"),
        ("data.symbols", [], "data.symbols"),
        ("data.symbols", ["BTCUSDT", "BTCUSDT"], "data.symbols"),
        ("bars.estimator", "parkinson", "bars.estimator: one of ['garman-klass']"),
        ("bars.missing", "a missing bar is skipped", "bars.missing"),
        ("nulls.rule", "events-only", "nulls.rule: one of ['refractory-from-episodes']"),
        ("cascade.a", -0.03, "cascade.a: positive"),
        ("cascade.c", 1.0, "cascade.c: below 1"),
        ("cascade.onset_percentile", 100.0, "cascade.onset_percentile"),
        ("cascade.merge_hours", 0, "cascade.merge_hours: at least 1"),
        ("horizon.smallest", 0, "horizon.smallest: a number of bars"),
        ("horizon.smallest", "five", "horizon.smallest: a number of bars"),
        ("version", 0, "version: must be >= 1"),
        ("id", "Market Labels", "id: 'Market Labels' is not a lowercase slug"),
        ("supersedes", "market-labels-v1", "reason: an amendment must say why"),
        # a key nothing reads, and a domain with no schema
        ("labeller.lag", 1, "labeller.lag: nothing reads this key in a market-labels file"),
        ("data.domain", "market", "data.domain: 'market' is not one of ['ising', 'market-labels']"),
    ],
)
def test_each_problem_is_named(doc: dict[str, Any], where: str, value: Any, named: str) -> None:
    found = prereg.problems(_set(doc, where, value))
    assert any(named in p for p in found), found


def test_load_reports_a_broken_file(tmp_path: Path) -> None:
    """load refuses a file with its problems, as for an Ising one."""
    text = TEMPLATE.read_text("utf-8").replace("d_min = 12", 'd_min = "12"', 1)
    path = tmp_path / "market-labels-v1.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(prereg.PreregError, match=r"labeller\.d_min: expected int, got '12'"):
        prereg.load(path)
    path.write_text(_ready(TEMPLATE.read_text("utf-8")) + "\n[extra]\nx = 1\n", encoding="utf-8")
    with pytest.raises(prereg.PreregError, match=r"extra: nothing reads this key"):
        prereg.load(path)


def test_the_domain_picks_the_schema() -> None:
    """An Ising file is checked as before; a file without a domain, or with one that is not a
    string, still gets the Ising schema's own message; an Ising file renamed market-labels is held
    to the market schema."""
    ising = prereg.load(ISING).doc
    assert prereg.problems(copy.deepcopy(ising)) == []
    assert prereg.problems(_set(copy.deepcopy(ising), "data.domain", DELETE)) == [
        "data.domain: missing"
    ]
    assert prereg.problems(_set(copy.deepcopy(ising), "data.domain", 3)) == [
        "data.domain: expected str, got 3"
    ]
    market = prereg.problems(_set(copy.deepcopy(ising), "data.domain", "market-labels"))
    assert "labeller: missing" in market and "data.symbols: missing" in market


def test_score_refuses_a_market_labels_file(capsys: pytest.CaptureFixture[str]) -> None:
    """ews score has no market scorer: it says so, before reading any Ising key."""
    assert ews_cli.main(["score", "--prereg", str(TEMPLATE)]) == 1
    assert "no scorer for domain 'market-labels' yet" in capsys.readouterr().err
