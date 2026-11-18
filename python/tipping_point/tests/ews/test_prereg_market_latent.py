# The market-latent pre-registration schema (#30 M9): the template, the strict validator, the
# binding of its cells to the market-ews file it names as its baseline, and the score refusal.

import copy
import hashlib
import itertools
import re
import subprocess
import tomllib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import m4_pilot
import m8_primaries
import numpy as np
import pytest
from explore import common, market_pass
from tipping_point.book import features
from tipping_point.ews import __main__ as ews_cli
from tipping_point.ews import prereg
from tipping_point.jepa import minutes

REPO = Path(__file__).resolve().parents[4]
TEMPLATE = REPO / "paper" / "prereg" / "TEMPLATE-market-latent.toml"
EWS_TEMPLATE = REPO / "paper" / "prereg" / "TEMPLATE-market-ews.toml"
LABELS_V1 = REPO / "paper" / "prereg" / "market-labels-v1.toml"
ISING_LATENT = REPO / "paper" / "prereg" / "ising-latent-v1.toml"
DELETE = object()
EWS_CHOICES = {"detrend": "none", "method": "mean", "fit": "pilot", "noise": "bootstrap"}
SIGN_IN_CELLS_ONLY = (
    "trained minus twin of |AUC - 0.5|, same model seed, same read symbol, same segments and the "
    "same bootstrap draws; a judged cell passes only when every primary's trained AUC - 0.5 has "
    "the same sign, so that seeds warning in opposite directions never pass on their margins"
)
DECIDED_BY_THE_CI = (
    "the latent pair's |AUC - 0.5| minus the classic indicator's AUC - 0.5 taken in the direction "
    "the baseline's hypothesis on that indicator pre-registers (AUC - 0.5 for 'greater'), the "
    "classic AUCs recomputed on the baseline's price series over the same segments, with the same "
    "bootstrap draws (a paired bootstrap); 'matching' compares each variance and level pair with "
    "the classic variance and the ac pair with the classic ac"
)


def filled_ews(text: str) -> str:
    text = text.replace('id = "<lowercase-slug>"', 'id = "market-ews"', 1)
    for key, value in (*EWS_CHOICES.items(), ("empty", "mark")):
        text = re.sub(rf'^{key} = "<[^"]*>"', f'{key} = "{value}"', text, count=1, flags=re.M)
    text = re.sub(r'far_per_day = "<[^"]*>"', "far_per_day = 1.0", text)
    return re.sub(r'"<[^"]*>"', '"set"', text)


def filled(text: str, baseline_sha256: str = "0" * 64) -> str:
    count, scaling = itertools.count(), itertools.count()
    per_symbol = len(prereg.LATENT_PRIMARIES + prereg.LATENT_SPARES)

    def digest(match: re.Match[str]) -> str:
        if match[1] == "scaling_sha256":
            seed = f"scaling{next(scaling) // per_symbol}"
        else:
            seed = f"{match[1]}{next(count)}"
        return f'{match[1]} = "{hashlib.sha256(seed.encode()).hexdigest()}"'

    text = text.replace('id = "<lowercase-slug>"', 'id = "market-latent"', 1)
    text = re.sub(r'^(\w+_sha256) = "<[^"]*>"', digest, text, flags=re.M)
    text = re.sub(
        r'^run_id = "<[^"]*>"', lambda _: f'run_id = "run-{next(count)}"', text, flags=re.M
    )
    text = re.sub(r'^data_commit = "<[^"]*>"', 'data_commit = "abc1234"', text, flags=re.M)
    text = re.sub(r'^(sha256 = )"<[^"]*>"', rf'\1"{baseline_sha256}"', text, flags=re.M)
    text = re.sub(r'^judged = "<[^"]*>"', 'judged = "same-asset"', text, flags=re.M)
    text = re.sub(r'^against = "<[^"]*>"', 'against = "each"', text, flags=re.M)
    text = re.sub(r'^tolerance = "<[^"]*>"', "tolerance = 0.02", text, flags=re.M)
    return re.sub(r'"<[^"]*>"', '"set"', text)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _commit(root: Path, name: str, text: str) -> Path:
    if not (root / ".git").exists():
        root.mkdir(exist_ok=True)
        _git(root, "init", "-q")
        _git(root, "config", "user.name", "Test")
        _git(root, "config", "user.email", "test@example.com")
        _git(root, "config", "core.autocrlf", "false")
    path = root / "paper" / "prereg" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", name)
    return path


@pytest.fixture
def doc() -> dict[str, Any]:
    return tomllib.loads(filled(TEMPLATE.read_text("utf-8")))


@pytest.fixture
def ews() -> dict[str, Any]:
    return tomllib.loads(filled_ews(EWS_TEMPLATE.read_text("utf-8")))


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


def test_the_template_loads_with_its_placeholders() -> None:
    draft = prereg.load(TEMPLATE).doc
    assert prereg.problems(draft) == []
    strict = prereg.problems(draft, strict=True)
    assert all(": placeholder " in p for p in strict), strict
    assert {re.sub(r"\[\d+\]", "[]", p.split(":")[0]) for p in strict} == {
        "id",
        "question",
        "hypotheses[].prediction",
        "baseline.sha256",
        "training.replacement",
        *(
            f"latents.checkpoints[].{key}"
            for key in (
                "run_id",
                "data_commit",
                "trained_sha256",
                "twin_sha256",
                "pca_sha256",
                "scaling_sha256",
            )
        ),
        "cells.judged",
        "scoring.twin.seeds",
        "scoring.twin.tolerance",
        "scoring.classic.against",
        "disclosure.exploratory",
        "disclosure.decisions",
        "disclosure.confirmatory",
        "deviations.policy",
    }
    assert sum("hypotheses[" in p for p in strict) == 4
    assert sum("latents.checkpoints[" in p for p in strict) == 14 * 6


def test_the_template_holds_what_the_primaries_were_trained_with() -> None:
    doc = prereg.load(TEMPLATE).doc
    assert prereg.BOOK_FEATURES == features.FEATURES
    assert prereg.BOOK_FORMAT == features.FORMAT
    assert (prereg.LATENT_SIZE, prereg.LATENT_STEPS) == (m8_primaries.SIZE, m8_primaries.STEPS)
    assert prereg.LATENT_PRIMARIES == m8_primaries.PRIMARIES
    assert prereg.LATENT_SPARES == m8_primaries.SPARES
    assert minutes.PRESETS[prereg.LATENT_SIZE].dim == prereg.LATENT_DIM
    assert minutes.PRESETS[prereg.LATENT_SIZE].context == prereg.LATENT_CONTEXT
    config = m4_pilot.config("btc", prereg.LATENT_SIZE, prereg.LATENT_STEPS, 101)
    assert (config.pca_frames, config.pca_components) == (4096, 8)
    assert "8 components fitted on 4,096 valid minutes" in prereg.LATENT_PCA
    train_days, val_days = m4_pilot.split()
    assert {"start": train_days[0], "end": train_days[-1]} == prereg.LATENT_TRAINING
    assert {"start": val_days[0], "end": val_days[-1]} == prereg.LATENT_VALIDATION
    assert val_days[-1] + timedelta(days=1) == m4_pilot.END
    assert set(prereg.MARKET_LATENT_PAIRS) == set(common.LATENT_PAIRS)
    assert {c for c, _ in prereg.MARKET_LATENT_PAIRS} <= prereg.LATENT_COLUMNS
    assert {i for _, i in prereg.MARKET_LATENT_PAIRS} <= prereg.LATENT_INDICATORS
    assert tuple(doc["segments"]["horizons"]) == market_pass.HORIZONS
    ind = doc["indicators"]
    assert (ind["window"], ind["length"]) == (market_pass.WINDOW, market_pass.LENGTH)
    assert doc["trials"]["budget"] == prereg.market_latent_cells(doc) == 1312
    labels = doc["data"]["labels"]
    assert labels["sha256"] == hashlib.sha256(LABELS_V1.read_bytes()).hexdigest()


def test_the_template_restates_the_market_ews_template(ews: dict[str, Any]) -> None:
    assert prereg.problems(ews, strict=True) == []
    assert prereg.baseline_problems(prereg.load(TEMPLATE).doc, ews) == []


@pytest.mark.parametrize(
    ("where", "value"),
    [
        ("data.labels.sha256", "1" * 64),
        ("data.events.stress", "none"),
        ("indicators.window", 30),
        ("indicators.detrend", "gaussian"),
        ("seasonality.method", "scale"),
        ("seasonality.fit.start", date(2023, 2, 1)),
        ("segments.horizons", [5, 6]),
        ("segments.subsample", 40),
        ("scoring.bootstrap", 1000),
        ("scoring.seed", 31),
        ("scoring.shift.draws", 50),
        ("scoring.surrogates.n", 199),
        ("reporting.groups", ["pooled", "per-asset"]),
    ],
)
def test_a_cell_other_than_the_baseline_s_is_named(
    doc: dict[str, Any], ews: dict[str, Any], where: str, value: Any
) -> None:
    found = prereg.baseline_problems(_set(doc, where, value), ews)
    cell = next(c for c in prereg.BASELINE_CELLS if where == c or where.startswith(f"{c}."))
    assert len(found) == 1 and found[0].startswith(f"{cell}: "), found


def test_the_baseline_binds_the_bandwidth_and_the_classic_indicator(
    doc: dict[str, Any], ews: dict[str, Any]
) -> None:
    ews["indicators"]["detrend"], ews["indicators"]["bandwidth"] = "gaussian", 120
    found = prereg.baseline_problems(doc, ews)
    assert any(p.startswith("indicators.detrend:") for p in found)
    assert any(p.startswith("indicators.bandwidth: 120") for p in found)
    ews["indicators"]["detrend"] = "none"
    del ews["indicators"]["bandwidth"]
    ews["indicators"]["names"] = ["variance"]
    doc["scoring"]["classic"]["against"] = "ac"
    assert prereg.baseline_problems(doc, ews) == [
        "scoring.classic.against: 'each', 'matching', or one of the baseline's indicators.names"
    ]
    doc["scoring"]["classic"]["against"] = "matching"
    assert prereg.baseline_problems(doc, ews) == [
        "scoring.classic.against: 'matching' compares with ['ac', 'variance']"
    ]
    doc["scoring"]["classic"]["against"] = "variance"
    assert prereg.baseline_problems(doc, ews) == []
    ews["data"]["domain"] = "market-labels"
    assert prereg.baseline_problems(doc, ews) == ["baseline: not a market-ews file"]


def test_the_freeze_holds_the_file_to_its_frozen_baseline(tmp_path: Path) -> None:
    ews_text = filled_ews(EWS_TEMPLATE.read_text("utf-8"))
    sha = hashlib.sha256(ews_text.encode()).hexdigest()
    text = filled(TEMPLATE.read_text("utf-8"), sha)
    _commit(tmp_path, "market-ews-v1.toml", ews_text)
    path = _commit(tmp_path, "market-latent-v1.toml", text)
    with pytest.raises(prereg.PreregError, match="scored on the segments of prereg/market-ews-v1"):
        prereg.freeze(tmp_path, "market-latent")
    assert prereg.freeze(tmp_path, "market-ews") == "prereg/market-ews-v1"

    _commit(tmp_path, "market-latent-v1.toml", text.replace(sha, "f" * 64))
    with pytest.raises(prereg.PreregError, match=r"baseline\.sha256: not the sha256 of"):
        prereg.freeze(tmp_path, "market-latent")
    _commit(tmp_path, "market-latent-v1.toml", text.replace("seed = 30", "seed = 31", 1))
    with pytest.raises(prereg.PreregError, match="differs from prereg/market-ews-v1") as refused:
        prereg.freeze(tmp_path, "market-latent")
    assert "segments: " in str(refused.value)
    assert _git(tmp_path, "tag", "--list") == "prereg/market-ews-v1"

    _commit(tmp_path, "market-ews-v1.toml", ews_text.replace("seed = 30", "seed = 31", 1))
    _commit(tmp_path, "market-latent-v1.toml", text)
    assert prereg.freeze(tmp_path, "market-latent") == "prereg/market-latent-v1"
    pre = prereg.load(path)
    assert prereg.stamp(pre, prereg.frozen(pre))["tag"] == "prereg/market-latent-v1"
    assert prereg.baseline_commit(pre) == _git(tmp_path, "rev-parse", "prereg/market-ews-v1^{}")


def test_the_freeze_refuses_a_placeholder(tmp_path: Path) -> None:
    text = filled(TEMPLATE.read_text("utf-8")).replace(
        'judged = "same-asset"', 'judged = "<same-asset or all>"', 1
    )
    _commit(tmp_path, "market-latent-v1.toml", text)
    with pytest.raises(prereg.PreregError, match="not ready to freeze") as refused:
        prereg.freeze(tmp_path, "market-latent")
    assert "cells.judged: placeholder" in str(refused.value)


def test_a_filled_file_passes_strict_and_counts_its_trials(
    doc: dict[str, Any], ews: dict[str, Any]
) -> None:
    assert prereg.problems(doc, strict=True) == []
    assert prereg.baseline_problems(doc, ews) == []
    doc["trials"]["budget"] = 1151
    assert prereg.problems(doc) == []
    assert prereg.problems(doc, strict=True) == []
    assert prereg.baseline_problems(doc, ews) == [
        "trials.budget: one scoring run evaluates 1152 cells"
    ]
    ews["indicators"]["names"] = ["variance"]
    doc["trials"]["budget"] = 1136
    assert prereg.baseline_problems(doc, ews) == []
    doc["scoring"]["classic"]["against"] = "variance"
    doc["trials"]["budget"] = 1135
    assert prereg.problems(doc, strict=True) == [
        "trials.budget: one scoring run evaluates 1136 cells"
    ]


def _auc(events: np.ndarray, controls: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(controls), events).mean(axis=-1) / len(controls)


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_the_shift_null_holds_the_latent_minus_classic_rule_near_its_level(rho: float) -> None:
    rng = np.random.default_rng(30)
    cov = [[1.0, rho], [rho, 1.0]]
    observed, passed, reps = [], 0, 1000
    for _ in range(reps):
        controls = rng.multivariate_normal([0.0, 0.0], cov, 300)
        events = rng.multivariate_normal([0.0, 0.0], cov, (101, 60))
        latent, classic = _auc(events[..., 0], controls[:, 0]), _auc(events[..., 1], controls[:, 1])
        difference = np.abs(latent - 0.5) - (classic - 0.5)
        observed.append(difference[0])
        passed += difference[0] > np.quantile(difference[1:], 0.975)
    assert np.mean(observed) > 0.02
    nominal = 3 / 101
    assert nominal / 2 <= passed / reps <= 2 * nominal


@pytest.mark.parametrize(
    ("judged", "against", "groups", "cells"),
    [
        ("all", "each", ["pooled", "per-asset", "eth-only"], 2 * 4 * 4 * 40 + 2 * 4 * 4),
        ("same-asset", "each", ["pooled", "per-asset", "eth-only"], 2 * 4 * 4 * 35 + 2 * 4 * 4),
        ("same-asset", "matching", ["pooled", "per-asset", "eth-only"], 1120 + 2 * 4 * 4),
        ("same-asset", "variance", ["pooled", "per-asset", "eth-only"], 1120 + 1 * 4 * 4),
        ("all", "ac", ["pooled", "per-asset"], 2 * 4 * 4 * 30 + 1 * 4 * 3),
        ("same-asset", "ac", ["pooled", "per-asset"], 2 * 4 * 4 * 25 + 1 * 4 * 3),
    ],
)
def test_the_trial_count_follows_the_judged_cells_and_the_classic_side(
    doc: dict[str, Any], judged: str, against: str, groups: list[str], cells: int
) -> None:
    doc["cells"]["judged"], doc["scoring"]["classic"]["against"] = judged, against
    doc["reporting"]["groups"] = groups
    assert prereg.market_latent_cells(doc) == cells


@pytest.mark.parametrize(
    ("where", "value", "named"),
    [
        ("training", DELETE, "training: missing"),
        ("disclosure.confirmatory_contact", DELETE, "disclosure.confirmatory_contact: missing"),
        ("indicators.thin", 1, "indicators.thin: nothing reads this key in a market-latent file"),
        ("latents.table", "latents", "latents.table: nothing reads this key"),
        ("training.steps", "10000", "training.steps: expected int, got '10000'"),
        ("descriptive.rv_minutes", 60.0, "descriptive.rv_minutes: expected int"),
        ("scoring.twin.sign_test", "true", "scoring.twin.sign_test: expected bool"),
        ("scoring.ci_level", True, "scoring.ci_level: expected int or float, got True"),
        ("training.span.start", "2023-01-18", "training.span.start: expected date"),
        ("latents.checkpoints", "pins.json", "latents.checkpoints: expected an array"),
        ("indicators.pairs", [["pc_1", 1]], "indicators.pairs[0][1]: expected str"),
        ("data.symbols", ["BTCUSDT"], "data.symbols: ['BTCUSDT', 'ETHUSDT']"),
        ("data.symbols", ["BTCUSDT", "BTCUSDT"], "data.symbols: ['BTCUSDT', 'ETHUSDT']"),
        ("data.missing", "gaps are filled", "data.missing"),
        ("data.labels.tag", "prereg/market-labels", "data.labels.tag"),
        ("data.events.primary", "cascades", "data.events"),
        ("baseline.tag", "prereg/ising-ews-v1", "baseline.tag: prereg/market-ews-v<k>"),
        ("baseline.tag", "market-ews-v1", "baseline.tag: prereg/market-ews-v<k>"),
        ("baseline.sha256", "abc", "baseline.sha256"),
        ("training.span.end", date(2023, 9, 30), "training.span: 2023-01-18 to 2023-08-31"),
        ("training.validation.start", datetime(2023, 9, 1), "training.validation: UTC days"),
        ("training.features", ["ret", "rv100"], "training.features"),
        ("training.feature_format", 3, "training.feature_format: 4"),
        ("training.size", "M", "training.size: 'S'"),
        ("training.dim", 96, "training.dim: 64"),
        ("training.steps", 20000, "training.steps: 10000"),
        ("training.primaries", [101, 102, 103], "training.primaries"),
        ("training.spares", [106], "training.spares"),
        ("training.scaling", "refitted", "training.scaling"),
        ("training.refusals", "none", "training.refusals"),
        ("training.replacement", " ", "training.replacement"),
        ("indicators.pairs", [["pc_1", "variance"]], "indicators.pairs"),
        ("indicators.pairs", [["pc_2", "variance"], ["pc_1", "ac"]], "indicators.pairs"),
        ("latents.columns", ["pc_1", "surprise", "pc_2"], "latents.columns"),
        ("latents.columns", ["pc_1"], "latents.columns"),
        ("latents.context", 95, "latents.context: 96"),
        ("latents.series", "the latents", "latents.series"),
        ("latents.pca", "a basis", "latents.pca"),
        ("indicators.lag", 2, "indicators.lag: 1"),
        ("indicators.window", 1, "indicators.window: at least 2"),
        ("indicators.detrend", "linear", "indicators.detrend: one of ['none', 'gaussian']"),
        ("indicators.detrend", "gaussian", "indicators.bandwidth: a positive number"),
        ("indicators.bandwidth", 120, "indicators.bandwidth: only with detrend 'gaussian'"),
        ("hypotheses", [], "hypotheses: one per pair of indicators.pairs"),
        ("seasonality.method", "median", "seasonality.method: one of ['scale', 'mean']"),
        ("seasonality.fit.end", date(2024, 3, 1), "seasonality.fit: start on or before end"),
        ("segments.horizons", [3, 6], "segments.horizons: distinct numbers of bars"),
        ("segments.rule", "shorter at long horizons", "segments.rule"),
        ("segments.seed", -1, "segments.seed: at least 0"),
        ("cells.read", [["BTCUSDT", "BTCUSDT"], ["ETHUSDT", "ETHUSDT"]], "cells.read"),
        ("cells.judged", "cross-asset", "cells.judged: one of ['same-asset', 'all']"),
        ("scoring.statistic", "AUC", "scoring.statistic: '|AUC - 0.5|'"),
        ("scoring.ci_level", 1.0, "scoring.ci_level: in (0, 1)"),
        ("scoring.bootstrap", 0, "scoring.bootstrap: at least 1 draw"),
        ("scoring.shift.min_offset_bars", 287, "scoring.shift.min_offset_bars: at least 288"),
        ("scoring.surrogates.secondary", "ar1", "scoring.surrogates.secondary: 'phase'"),
        ("cells.groups", "per model", "cells.groups"),
        ("scoring.twin.pairing", "trained minus twin", "scoring.twin.pairing"),
        ("scoring.twin.pairing", prereg.TWIN_PAIRING.split(";")[0], "scoring.twin.pairing"),
        ("scoring.twin.seeds", " ", "scoring.twin.seeds"),
        ("scoring.twin.tolerance", 0, "scoring.twin.tolerance: a number in (0, 0.5)"),
        ("scoring.twin.tolerance", 0.5, "scoring.twin.tolerance: a number in (0, 0.5)"),
        ("scoring.twin.tolerance", "small", "scoring.twin.tolerance: a number in (0, 0.5)"),
        ("scoring.twin.pairing", SIGN_IN_CELLS_ONLY, "scoring.twin.pairing"),
        ("scoring.classic.pairing", "latent minus classic", "scoring.classic.pairing"),
        ("scoring.classic.pairing", DECIDED_BY_THE_CI, "scoring.classic.pairing"),
        (
            "scoring.classic.pairing",
            prereg.CLASSIC_PAIRING.replace("AUC - 0.5 taken in the direction", "|AUC - 0.5|", 1),
            "scoring.classic.pairing",
        ),
        ("scoring.classic.against", "skewness", "scoring.classic.against: one of"),
        ("descriptive.same_input", "the tier", "descriptive.same_input"),
        ("descriptive.incremental_rv", "a logistic model", "descriptive.incremental_rv"),
        ("descriptive.rv_minutes", 0, "descriptive.rv_minutes: at least 1"),
        ("descriptive.ablations", ["frames-100ms"], "descriptive.ablations"),
        ("descriptive.ablations", ["ema-only", "ema-only"], "descriptive.ablations"),
        ("reporting.groups", ["pooled"], "reporting.groups: pooled and per-asset"),
        ("reporting.eth_only_bars", 6, "reporting.eth_only_bars: 12"),
        ("trials.budget", 0, "trials.budget: at least 1"),
        ("disclosure.confirmatory_contact", " ", "disclosure.confirmatory_contact"),
        ("version", 0, "version: must be >= 1"),
        ("supersedes", "market-latent-v1", "reason: an amendment must say why"),
        ("id", "Market Latent", "id: 'Market Latent' is not a lowercase slug"),
    ],
)
def test_each_problem_is_named(doc: dict[str, Any], where: str, value: Any, named: str) -> None:
    found = prereg.problems(_set(doc, where, value))
    assert any(named in p for p in found), found


def test_every_hypothesis_is_two_sided_and_distinct(doc: dict[str, Any]) -> None:
    doc["hypotheses"][0]["direction"] = "greater"
    assert prereg.problems(doc) == ["hypotheses[0].direction: 'two-sided', read as |AUC - 0.5|"]
    doc["hypotheses"][0]["direction"] = "two-sided"
    doc["hypotheses"][1]["name"] = doc["hypotheses"][0]["name"]
    assert prereg.problems(doc) == ["hypotheses: distinct names"]
    doc["hypotheses"][1]["name"] = "pc1-ac"
    doc["hypotheses"][1]["column"] = "surprise"
    assert any(p.startswith("hypotheses: one per pair") for p in prereg.problems(doc))


def _entry(doc: dict[str, Any], symbol: str, seed: int) -> dict[str, Any]:
    return next(
        e for e in doc["latents"]["checkpoints"] if (e["symbol"], e["seed"]) == (symbol, seed)
    )


def test_each_checkpoint_problem_is_named(doc: dict[str, Any]) -> None:
    base = copy.deepcopy(doc)

    def found(change: Any) -> list[str]:
        mine = copy.deepcopy(base)
        change(mine)
        return prereg.problems(mine)

    first = "latents.checkpoints[0]"
    assert found(lambda d: _entry(d, "BTCUSDT", 101).update(trained_sha256="abc")) == [
        f"{first}.trained_sha256: a sha256 (64 hex digits)"
    ]
    assert found(lambda d: _entry(d, "BTCUSDT", 101).update(pca_sha256="A" * 64)) == [
        f"{first}.pca_sha256: a sha256 (64 hex digits)"
    ]
    assert found(lambda d: _entry(d, "BTCUSDT", 101).update(data_commit="main")) == [
        f"{first}.data_commit: a commit (7 to 40 hex digits)"
    ]
    assert found(lambda d: _entry(d, "BTCUSDT", 101).update(run_id=" ")) == [
        f"{first}.run_id: the training's run id"
    ]

    files = (
        "latents.checkpoints: every trained_sha256, twin_sha256 and pca_sha256 names a "
        "different file"
    )

    def copied(to: tuple[str, int, str], source: tuple[str, int, str]) -> Any:
        return lambda d: _entry(d, *to[:2]).update({to[2]: _entry(d, *source[:2])[source[2]]})

    for to, source in (
        (("BTCUSDT", 101, "twin_sha256"), ("BTCUSDT", 101, "trained_sha256")),
        (("ETHUSDT", 101, "trained_sha256"), ("BTCUSDT", 101, "trained_sha256")),
        (("BTCUSDT", 102, "twin_sha256"), ("BTCUSDT", 101, "trained_sha256")),
        (("BTCUSDT", 101, "pca_sha256"), ("BTCUSDT", 101, "trained_sha256")),
        (("ETHUSDT", 105, "pca_sha256"), ("BTCUSDT", 101, "pca_sha256")),
    ):
        assert found(copied(to, source)) == [files], (to, source)

    def one_pca(d: dict[str, Any]) -> None:
        for entry in d["latents"]["checkpoints"]:
            entry["pca_sha256"] = d["latents"]["checkpoints"][0]["pca_sha256"]

    assert found(one_pca) == [files]
    assert found(copied(("ETHUSDT", 101, "run_id"), ("BTCUSDT", 101, "run_id"))) == [
        "latents.checkpoints: no two entries share a run_id"
    ]
    assert found(
        copied(("BTCUSDT", 103, "scaling_sha256"), ("ETHUSDT", 101, "scaling_sha256"))
    ) == [
        "latents.checkpoints: BTCUSDT's entries share one scaling_sha256, the record of its "
        "training days",
        "latents.checkpoints: each symbol's scaling_sha256 differs from the other's",
    ]

    def eth_takes_btc_scaling(d: dict[str, Any]) -> None:
        btc = _entry(d, "BTCUSDT", 101)["scaling_sha256"]
        for entry in d["latents"]["checkpoints"]:
            entry["scaling_sha256"] = btc

    assert found(eth_takes_btc_scaling) == [
        "latents.checkpoints: each symbol's scaling_sha256 differs from the other's"
    ]
    moved = found(lambda d: _entry(d, "BTCUSDT", 101).update(symbol="SOLUSDT"))
    assert f"{first}.symbol: one of data.symbols" in moved
    held = found(lambda d: _entry(d, "BTCUSDT", 101).update(role="held-out"))
    assert f"{first}.role: one of ['primary', 'spare', 'ema-only']" in held
    assert found(lambda d: d["latents"]["checkpoints"].pop(4)) == [
        "latents.checkpoints: BTCUSDT's primary seeds are training's, once"
    ]
    assert found(lambda d: _entry(d, "ETHUSDT", 106).update(role="primary")) == [
        "latents.checkpoints: ETHUSDT's primary seeds are training's, once",
        "latents.checkpoints: ETHUSDT's spare seeds are training's, once",
    ]

    def twice(d: dict[str, Any]) -> None:
        d["latents"]["checkpoints"].append(_entry(d, "BTCUSDT", 107) | {"run_id": "again"})

    assert "latents.checkpoints: one entry per seed of BTCUSDT" in found(twice)


def test_ema_only_checkpoints_come_with_the_ablation(doc: dict[str, Any]) -> None:
    extra = [
        _entry(doc, symbol, 101)
        | {
            "seed": 111,
            "role": "ema-only",
            "run_id": f"ema-{symbol}",
            "trained_sha256": hashlib.sha256(f"t{symbol}".encode()).hexdigest(),
            "twin_sha256": hashlib.sha256(f"w{symbol}".encode()).hexdigest(),
            "pca_sha256": hashlib.sha256(f"p{symbol}".encode()).hexdigest(),
        }
        for symbol in prereg.MARKET_SYMBOLS
    ]
    doc["latents"]["checkpoints"].append(extra[0])
    assert prereg.problems(doc) == [
        "latents.checkpoints[14].role: 'ema-only' only as one of descriptive.ablations"
    ]
    doc["descriptive"]["ablations"] = ["ema-only"]
    assert prereg.problems(doc) == ["latents.checkpoints: ETHUSDT has no ema-only checkpoint"]
    doc["latents"]["checkpoints"].append(extra[1])
    assert prereg.problems(doc, strict=True) == []
    assert prereg.market_latent_cells(doc) == 1152


def test_a_number_that_is_not_finite_is_refused(doc: dict[str, Any]) -> None:
    doc["scoring"]["twin"]["tolerance"] = float("inf")
    assert prereg.problems(doc) == ["scoring.twin.tolerance: a finite number, not inf"]
    doc["scoring"]["twin"]["tolerance"] = 0.02
    doc["scoring"]["ci_level"] = float("nan")
    assert prereg.problems(doc) == ["scoring.ci_level: a finite number, not nan"]


def test_a_draft_may_leave_a_pin_open_and_strict_refuses_it(doc: dict[str, Any]) -> None:
    _entry(doc, "ETHUSDT", 107)["twin_sha256"] = "<sha256 of the step-0 twin, from pins.json>"
    assert prereg.problems(doc) == []
    assert prereg.problems(doc, strict=True) == [
        "latents.checkpoints[13].twin_sha256: placeholder "
        "'<sha256 of the step-0 twin, from pins.json>'"
    ]


def test_score_refuses_a_market_latent_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert ews_cli.main(["score", "--prereg", str(TEMPLATE)]) == 1
    assert "no scorer for domain 'market-latent' yet" in capsys.readouterr().err


def test_the_domain_picks_the_schema(ews: dict[str, Any]) -> None:
    assert prereg.problems(prereg.load(ISING_LATENT).doc) == []
    assert prereg.problems(prereg.load(LABELS_V1).doc, strict=True) == []
    ews["data"]["domain"] = "market-latent"
    found = prereg.problems(ews)
    assert "training: missing" in found and "baseline: missing" in found
    assert "market-latent" in prereg.DOMAINS
