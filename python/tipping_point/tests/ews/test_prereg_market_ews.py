"""The schema of a market-ews pre-registration (#30 M5): the template loads with its placeholders
and holds the values the code runs, every problem with a file is named, the freeze refuses a file
until no placeholder is left, a filled file goes through freeze, frozen and stamp, and the Ising
and market-labels files are checked as before."""

import copy
import dataclasses
import hashlib
import re
import subprocess
import tomllib
from datetime import date, datetime
from pathlib import Path
from typing import Any

import market_floor
import pytest
from tipping_point.ews import prereg, trend
from tipping_point.market import msvol, nulls, reports

REPO = Path(__file__).resolve().parents[4]
TEMPLATE = REPO / "paper" / "prereg" / "TEMPLATE-market-ews.toml"
LABELS_V1 = REPO / "paper" / "prereg" / "market-labels-v1.toml"
LABELS_TEMPLATE = REPO / "paper" / "prereg" / "TEMPLATE-market-labels.toml"
ISING = REPO / "paper" / "prereg" / "ising-ews-v1.toml"
DELETE = object()
# each open choice of the template, set to an option that is built
CHOICES = {
    "detrend": "none",
    "method": "scale",
    "fit": "pilot",
    "noise": "bootstrap",
    "empty": "mark",
}


def filled(text: str, prereg_id: str = "market-ews") -> str:
    """The template with every placeholder set: a file that may be frozen."""
    text = text.replace('id = "<lowercase-slug>"', f'id = "{prereg_id}"', 1)
    for key, value in CHOICES.items():
        text = re.sub(rf'^{key} = "<[^"]*>"', f'{key} = "{value}"', text, count=1, flags=re.M)
    text = re.sub(r'far_per_day = "<[^"]*>"', "far_per_day = 1.0", text)
    return re.sub(r'"<[^"]*>"', '"set"', text)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _repo(root: Path, text: str) -> Path:
    """A scratch repository holding paper/prereg/market-ews-v1.toml, committed."""
    root.mkdir(exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "core.autocrlf", "false")
    path = root / "paper" / "prereg" / "market-ews-v1.toml"
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode())
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "market-ews prereg")
    return path


@pytest.fixture
def doc() -> dict[str, Any]:
    return copy.deepcopy(prereg.load(TEMPLATE).doc)


def test_the_template_loads_with_its_placeholders(doc: dict[str, Any]) -> None:
    """A draft: load takes it, and the strict check names each placeholder, nothing else: the
    owner's open choices and the texts written at the freeze."""
    assert prereg.problems(doc) == []
    strict = prereg.problems(doc, strict=True)
    assert all(": placeholder " in p for p in strict), strict
    assert {p.split(":")[0] for p in strict} == {
        "id",
        "question",
        "hypotheses[0].prediction",
        "hypotheses[1].prediction",
        "indicators.detrend",
        "seasonality.method",
        "scoring.floor.fit",
        "scoring.floor.noise",
        "scoring.floor.empty",
        "alarms.far_per_day",
        "disclosure.exploratory",
        "disclosure.decisions",
        "disclosure.confirmatory",
        "deviations.policy",
    }


def test_the_template_holds_what_the_code_runs(doc: dict[str, Any]) -> None:
    """The labels it names are the frozen file's bytes and tag; its pipelines are the floor's; its
    floor's GARCH is the pilot fit; the shift is the null's own defaults; the smallest horizon is
    the labels'; the constants prereg holds are the code's."""
    labels = doc["data"]["labels"]
    assert labels["sha256"] == hashlib.sha256(LABELS_V1.read_bytes()).hexdigest()
    assert labels["tag"] == prereg.load(LABELS_V1).tag
    assert _git(REPO, "tag", "--list", labels["tag"]) == labels["tag"]
    ind = doc["indicators"]
    for name in ind["names"]:
        assert market_floor.SCORERS[name] == trend.Pipeline(name, ind["window"], ind["length"])
    assert set(prereg.EWS_INDICATORS) <= set(trend.INDICATORS)
    assert {
        s: tuple(getattr(g, k) for k in prereg.GARCH_FIELDS)
        for s, g in market_floor.PILOT_GARCH.items()
    } == prereg.PILOT_GARCH
    garch = doc["scoring"]["floor"]["garch"]
    assert {s: tuple(g[k] for k in prereg.GARCH_FIELDS) for s, g in garch.items()} == (
        prereg.PILOT_GARCH
    )
    assert tuple(f.name for f in dataclasses.fields(nulls.Garch)) == prereg.GARCH_FIELDS
    assert prereg.FLOOR_FIRST == nulls.FIRST
    shift = doc["scoring"]["shift"]
    assert shift["draws"] == nulls.SHIFT_DRAWS
    assert shift["min_offset_bars"] == nulls.SHIFT_MIN_DAYS * msvol.DAY_BARS
    assert prereg.DAY_BARS == msvol.DAY_BARS
    smallest = prereg.load(LABELS_V1).doc["horizon"]["smallest"]
    assert prereg.HORIZON_FLOOR == smallest == min(doc["segments"]["horizons"])
    assert doc["trials"]["budget"] == prereg.market_ews_cells(doc) == 32
    # the pilot floor scores the template's horizons with the template's pipelines
    assert prereg.FLOOR_HORIZONS == market_floor.HORIZONS
    assert set(doc["segments"]["horizons"]) <= set(market_floor.HORIZONS)
    floor = prereg.FLOOR_PIPELINE
    for name, scorer in market_floor.SCORERS.items():
        assert scorer == trend.Pipeline(name, floor["window"], floor["length"])
        assert scorer.bandwidth is None and floor["detrend"] == "none"
    assert {k: ind[k] for k in ("window", "length")} == {k: floor[k] for k in ("window", "length")}
    # eth-only's window is the coincidence window the labels reports use
    assert prereg.ETH_ONLY_BARS * msvol.BAR_NS == reports.COINCIDENCE_NS
    assert doc["reporting"]["eth_only_bars"] == prereg.ETH_ONLY_BARS
    assert set(prereg.DESEASON_METHODS) == {"scale", "mean"}


def test_a_filled_template_goes_through_freeze_frozen_and_stamp(tmp_path: Path) -> None:
    path = _repo(tmp_path / "repo", filled(TEMPLATE.read_text("utf-8")))
    pre = prereg.load(path)
    assert prereg.problems(pre.doc, strict=True) == []
    assert prereg.freeze(path.parents[2], "market-ews") == "prereg/market-ews-v1"
    ref = prereg.frozen(pre)
    assert prereg.stamp(pre, ref)["tag"] == "prereg/market-ews-v1"


@pytest.mark.parametrize(
    ("kept", "named"),
    [
        (('detrend = "none"', 'detrend = "<none|gaussian, open>"'), "indicators.detrend"),
        (('fit = "pilot"', 'fit = "TBD"'), "scoring.floor.fit"),
        (("far_per_day = 1.0", 'far_per_day = "<a rate>"'), "alarms.far_per_day"),
    ],
)
def test_the_freeze_refuses_a_placeholder(
    tmp_path: Path, kept: tuple[str, str], named: str
) -> None:
    path = _repo(tmp_path, filled(TEMPLATE.read_text("utf-8")).replace(*kept, 1))
    prereg.load(path)  # a draft loads
    with pytest.raises(prereg.PreregError, match="not ready to freeze") as refused:
        prereg.freeze(tmp_path, "market-ews")
    assert f"{named}: placeholder" in str(refused.value)
    assert _git(tmp_path, "tag", "--list") == ""


def test_the_strict_check_counts_the_trials() -> None:
    doc = tomllib.loads(filled(TEMPLATE.read_text("utf-8")))
    doc["trials"]["budget"] = 31
    assert prereg.problems(doc) == []
    assert prereg.problems(doc, strict=True) == [
        "trials.budget: one scoring run evaluates 32 cells"
    ]


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
        # missing keys and wrong types
        ("segments.horizons", DELETE, "segments.horizons: missing"),
        ("scoring.floor", DELETE, "scoring.floor: missing"),
        ("indicators.window", "60", "indicators.window: expected int, got '60'"),
        ("scoring.ci_level", True, "scoring.ci_level: expected int or float, got True"),
        ("seasonality.fit.start", "2023-01-18", "seasonality.fit.start: expected date"),
        ("data.symbols", "BTCUSDT", "data.symbols: expected an array"),
        # the frozen labels
        ("data.labels.tag", "prereg/market-labels", "data.labels.tag"),
        ("data.labels.sha256", "d5988eab", "data.labels.sha256"),
        ("data.events.primary", "cascades", "data.events"),
        # symbols, series, rules held word for word
        ("data.symbols", ["BTCUSDT", "SOLUSDT"], "data.symbols"),
        ("data.symbols", [], "data.symbols"),
        ("data.series", "Bybit mid returns", "data.series"),
        ("data.missing", "gaps are filled", "data.missing"),
        ("segments.rule", "shorter at long horizons", "segments.rule"),
        ("segments.controls", "hour-matched", "segments.controls"),
        ("seasonality.method", "median", "seasonality.method: one of ['scale', 'mean']"),
        ("alarms.rule", "any", "alarms.rule"),
        # indicators
        ("indicators.names", ["variance", "skewness"], "indicators.names"),
        ("indicators.names", ["ac", "ac"], "indicators.names"),
        ("indicators.window", 1, "indicators.window: at least 2"),
        ("indicators.lag", 2, "indicators.lag: 1"),
        ("indicators.detrend", "linear", "indicators.detrend: one of ['none', 'gaussian']"),
        ("indicators.detrend", "gaussian", "indicators.bandwidth: a positive number"),
        ("indicators.bandwidth", 120, "indicators.bandwidth: only with detrend 'gaussian'"),
        ("indicators.thin", 1, "indicators.thin: nothing reads this key in a market-ews file"),
        # hypotheses
        ("hypotheses", [], "hypotheses: one per scored indicator"),
        # seasonality
        ("seasonality.fit.end", date(2024, 3, 1), "seasonality.fit: start on or before end"),
        ("seasonality.fit.start", date(2023, 10, 1), "seasonality.fit: start on or before end"),
        ("seasonality.fit.end", datetime(2023, 9, 30), "seasonality.fit: UTC days"),
        ("seasonality.strata", ["weekday"], "seasonality.strata"),
        # horizons and controls
        ("segments.horizons", [3, 6, 12], "segments.horizons: distinct numbers of bars"),
        ("segments.horizons", [6, 6], "segments.horizons: distinct numbers of bars"),
        ("segments.horizons", [], "segments.horizons: distinct numbers of bars"),
        ("segments.subsample", -1, "segments.subsample: at least 0"),
        # scoring
        ("scoring.ci_level", 1.0, "scoring.ci_level: in (0, 1)"),
        ("scoring.bootstrap", 0, "scoring.bootstrap: at least 1 draw"),
        ("scoring.floor.fit", "deseasonalised", "scoring.floor.fit: one of ['pilot']"),
        ("scoring.floor.noise", "matched", "scoring.floor.noise: one of ['bootstrap']"),
        ("scoring.floor.empty", "refuse", "scoring.floor.empty: one of ['mark']"),
        ("scoring.floor.days", 547, "scoring.floor.days: 1 to 546"),
        ("scoring.floor.paths", 0, "scoring.floor: at least 1 path"),
        ("scoring.floor.garch", {"BTCUSDT": {}}, "scoring.floor.garch: one table per symbol"),
        ("scoring.floor.garch.BTCUSDT.beta", 0.95, "alpha + beta < 1"),
        ("scoring.floor.garch.BTCUSDT.nu", "5", "scoring.floor.garch.BTCUSDT: the numbers"),
        ("scoring.shift.min_offset_bars", 287, "scoring.shift.min_offset_bars: at least 288"),
        ("scoring.surrogates.secondary", "ar1", "scoring.surrogates.secondary: 'phase'"),
        # reporting, alarms, trials
        ("reporting.groups", ["pooled"], "reporting.groups: pooled and per-asset"),
        ("reporting.groups", ["pooled", "per-asset", "btc-only"], "reporting.groups"),
        ("reporting.eth_only_bars", 6, "reporting.eth_only_bars: 12"),
        # the pilot floor binds horizons and pipelines
        ("segments.horizons", [5, 7], "segments.horizons: with the pilot floor, among"),
        ("indicators.length", 120, "indicators.length: 240 with the pilot floor"),
        ("indicators.window", 30, "indicators.window: 60 with the pilot floor"),
        ("alarms.far_per_day", 0, "alarms.far_per_day: a positive number"),
        ("alarms.far_per_day", "often", "alarms.far_per_day: a positive number"),
        ("trials.budget", 0, "trials.budget: at least 1"),
        ("version", 0, "version: must be >= 1"),
        ("supersedes", "market-ews-v1", "reason: an amendment must say why"),
    ],
)
def test_each_problem_is_named(doc: dict[str, Any], where: str, value: Any, named: str) -> None:
    doc["indicators"]["detrend"] = "none"
    doc["scoring"]["floor"]["fit"] = "pilot"
    found = prereg.problems(_set(doc, where, value))
    assert any(named in p for p in found), found


def test_the_pilot_floor_refuses_a_detrend(doc: dict[str, Any]) -> None:
    """The floor's pipelines detrend nothing; a placeholder fit leaves the detrend open."""
    doc["indicators"]["detrend"], doc["indicators"]["bandwidth"] = "gaussian", 120
    assert prereg.problems(doc) == []
    doc["scoring"]["floor"]["fit"] = "pilot"
    found = prereg.problems(doc)
    assert len(found) == 1 and found[0].startswith("indicators.detrend: 'none' with the pilot"), (
        found
    )


def test_hypotheses_are_distinct_and_rising(doc: dict[str, Any]) -> None:
    """No scorer reads a falling indicator yet, and two hypotheses may not share a name."""
    doc["hypotheses"][0]["direction"] = "less"
    assert any("hypotheses[0].direction: 'greater'" in p for p in prereg.problems(doc))
    doc["hypotheses"][0]["direction"] = "greater"
    doc["hypotheses"][1]["name"] = doc["hypotheses"][0]["name"]
    assert any("hypotheses" in p and "distinct" in p for p in prereg.problems(doc))


def test_the_pilot_fit_is_held_to_its_numbers(doc: dict[str, Any]) -> None:
    """With fit 'pilot' the GARCH must be the disclosed pilot fit; a placeholder fit lets any
    stationary GARCH stand until the owner chooses."""
    doc["scoring"]["floor"]["garch"]["ETHUSDT"]["alpha"] = 0.08
    assert prereg.problems(doc) == []
    doc["scoring"]["floor"]["fit"] = "pilot"
    assert any("scoring.floor.garch.ETHUSDT: the pilot fit" in p for p in prereg.problems(doc))


def test_eth_only_needs_both_symbols(doc: dict[str, Any]) -> None:
    doc["data"]["symbols"] = ["ETHUSDT"]
    del doc["scoring"]["floor"]["garch"]["BTCUSDT"]
    found = prereg.problems(doc)
    assert found == ["reporting.groups: eth-only reads ETH onsets against BTC's: both symbols"]


def test_a_number_that_is_not_finite_is_refused(doc: dict[str, Any]) -> None:
    """#147: TOML allows nan, and every comparison with nan is false."""
    doc["scoring"]["ci_level"] = float("nan")
    assert prereg.problems(doc) == ["scoring.ci_level: a finite number, not nan"]


def test_the_domain_picks_the_schema(doc: dict[str, Any]) -> None:
    """The Ising and market-labels files are checked as before, and a market-labels file renamed
    market-ews is held to this schema."""
    assert prereg.problems(prereg.load(ISING).doc) == []
    assert prereg.problems(prereg.load(LABELS_V1).doc, strict=True) == []
    assert prereg.problems(prereg.load(LABELS_TEMPLATE).doc) == []
    labels = prereg.load(LABELS_V1).doc
    labels["data"]["domain"] = "market-ews"
    found = prereg.problems(labels)
    assert "hypotheses: missing" in found and "segments: missing" in found
