# The market-latent scorer on synthetic latents: a planted signal and a null, the twin, classic and
# finite-score rules, the stated random streams, the 1312-trial budget through a crash, the gate
# record checked before any read, and a dry run that reads no day from 2024-01-01 and logs no trial.

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tipping_point.ews import market_latent_score as mls
from tipping_point.ews import market_score as msc
from tipping_point.ews import market_segments as ms
from tipping_point.ews import prereg, scoring
from tipping_point.ews.__main__ import main
from tipping_point.ews.surrogates import phase_randomised
from tipping_point.market import archive, msvol, nulls

PREREG = Path(__file__).resolve().parents[4] / "paper" / "prereg"
BTC, ETH = "BTCUSDT", "ETHUSDT"
DAYS = 14
EVENTS = [288 + 96 * k for k in range(18)]
ETH_LAG = 48
NULL_DAYS = range(8, DAYS)
RAMP = 420
PILOT = date(2023, 3, 1)
CONFIRMATORY = prereg.CONFIRMATORY
BUDGET = 1312
Frozen = tuple[prereg.Prereg, prereg.Prereg, prereg.Prereg]


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)
    return done.stdout.decode().strip()


@pytest.fixture(scope="module")
def real(tmp_path_factory: pytest.TempPathFactory) -> Frozen:
    root = tmp_path_factory.mktemp("repo")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "core.autocrlf", "false")
    folder = root / "paper" / "prereg"
    folder.mkdir(parents=True)
    names = ("market-labels-v1.toml", "market-ews-v1.toml", "market-latent-v1.toml")
    for name in names:
        (folder / name).write_bytes((PREREG / name).read_bytes())
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "preregs")
    for prereg_id in ("market-labels", "market-ews", "market-latent"):
        prereg.freeze(root, prereg_id)
    labels, baseline, latent = (prereg.load(folder / name) for name in names)
    return labels, baseline, latent


def _small(pre: prereg.Prereg, surrogates: int = 1) -> prereg.Prereg:
    sc = pre.doc["scoring"]
    small = {"bootstrap": 200, "shift": sc["shift"] | {"draws": 10}}
    doc = pre.doc | {"scoring": sc | small | {"surrogates": sc["surrogates"] | {"n": surrogates}}}
    return replace(pre, doc=doc)


def _onsets(symbol: str) -> list[int]:
    return [b + (ETH_LAG if symbol == ETH else 0) for b in EVENTS]


def _labels_run(data: Path, labels: prereg.Prereg, first: date, stamp: bool = False) -> Path:
    t0 = archive.day_ms(first) * 1_000_000
    ts = t0 + msvol.BAR_NS * np.arange(DAYS * msvol.DAY_BARS, dtype=np.int64)
    rows, episodes, stretches = [], [], []
    for s in (BTC, ETH):
        rows += [{"ts_ns": int(t), "symbol": s, "observed": True, "h": 0} for t in ts]
        episodes += [
            {"ts_ns": int(ts[b]), "symbol": s, "bars": 40, "kind": "event", "missing": 0}
            for b in _onsets(s)
        ]
        for d in NULL_DAYS:
            b = d * msvol.DAY_BARS
            stretches.append(
                {"ts_ns": int(ts[b]), "symbol": s, "end_ts_ns": int(ts[b + 280])}
                | {"bars": 280, "missing": 0}
            )
    tables = {
        "labels": pa.Table.from_pylist(rows, schema=msvol.SCHEMAS["labels"]),
        "episodes": pa.Table.from_pylist(episodes, schema=msvol.SCHEMAS["episodes"]),
        "nulls": pa.Table.from_pylist(stretches, schema=msvol.SCHEMAS["nulls"]),
    }
    warmup = timedelta(days=labels.doc["labeller"]["warmup_days"])
    params: dict[str, Any] = {
        "start": (first - warmup).isoformat(),
        "end": (first + timedelta(days=DAYS - 1)).isoformat(),
        "symbols": [BTC, ETH],
        "labeller": labels.doc["labeller"],
    }
    if stamp:
        params["prereg"] = {"sha256": labels.sha256, "tag": labels.tag}
    return msvol.write_run(data, "synthetic", tables, params, [], [], {})


def _prices(start: date, first: date, seed: int) -> dict[str, ms.Minutes]:
    rng = np.random.default_rng(seed)
    n = (first - start).days * 1440 + DAYS * 1440
    return {s: ms.Minutes(archive.day_ms(start), 1e-4 * rng.standard_normal(n)) for s in (BTC, ETH)}


def _book(prices: dict[str, ms.Minutes], seed: int) -> dict[str, dict[str, ms.Minutes]]:
    rng = np.random.default_rng(seed)
    return {
        s: {"ofi": ms.Minutes(m.start_ms, rng.standard_normal(len(m.values)))}
        for s, m in prices.items()
    }


def _cascades(first: date) -> dict[str, np.ndarray]:
    t0 = archive.day_ms(first) * 1_000_000
    return {s: t0 + msvol.BAR_NS * np.array(_onsets(s)[:6], np.int64) for s in (BTC, ETH)}


class Synthetic:
    def __init__(
        self,
        doc: dict[str, Any],
        start: date,
        first: date,
        planted: bool,
        seed: int,
        fit: tuple[date, int] | None = None,
        flat: str | None = None,
    ) -> None:
        self.cps = {cp.name: k for k, cp in enumerate(mls.checkpoints(doc))}
        self.columns = list(doc["latents"]["columns"])
        self.start, self.first, self.planted, self.seed = start, first, planted, seed
        self.fit, self.flat = fit, flat
        self.reads: list[tuple[str, str, str]] = []

    def read(self, checkpoint: mls.Checkpoint, symbol: str, span: str) -> dict[str, ms.Minutes]:
        self.reads.append((checkpoint.name, symbol, span))
        if span == "fit" and self.fit is not None:
            day, n = self.fit
            return {c: ms.Minutes(archive.day_ms(day), np.zeros(n)) for c in self.columns}
        offset = (self.first - self.start).days * 1440
        n = offset + DAYS * 1440
        out = {}
        for k, c in enumerate(self.columns):
            code = [self.seed, self.cps[checkpoint.name], k, (BTC, ETH).index(symbol)]
            sigma = np.ones(n)
            if self.planted and checkpoint.kind == "trained" and c == "pc_1":
                for b in _onsets(symbol):
                    onset = offset + 5 * b
                    sigma[onset - RAMP : onset] = 1 + 3 * np.arange(RAMP) / RAMP
            values = sigma * np.random.default_rng(code).standard_normal(n)
            if (checkpoint.name, symbol, c) == (self.flat, BTC, "pc_1"):
                onset = offset + 5 * _onsets(BTC)[0]
                values[onset - 480 : onset] = 0.0
            out[c] = ms.Minutes(archive.day_ms(self.start), values)
        return out


def _dry(tmp: Path, frozen: Frozen, planted: bool, surrogates: int = 19) -> mls.Evaluation:
    labels, baseline, pre = frozen
    pre = _small(pre, surrogates)
    run = _labels_run(tmp, labels, PILOT)
    start = PILOT - timedelta(days=2)
    prices = _prices(start, PILOT, seed=1)
    tables = Synthetic(pre.doc, start, PILOT, planted, seed=5)
    return mls.evaluate(
        pre,
        baseline,
        labels,
        run,
        prices,
        _cascades(PILOT),
        _book(prices, 2),
        tables,
        confirmatory=False,
    )


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory, real: Frozen) -> mls.Evaluation:
    return _dry(tmp_path_factory.mktemp("planted"), real, planted=True, surrogates=39)


def test_a_planted_latent_signal_passes_and_the_other_pairs_fail(planted: mls.Evaluation) -> None:
    assert planted.verdicts == {
        "pc1-variance": "pass",
        "pc1-ac": "fail",
        "surprise-variance": "fail",
        "surprise-level": "fail",
    }
    (row,) = [r for r in planted.tables["hypotheses"] if r["hypothesis"] == "pc1-variance"]
    assert (row["warns_pass"], row["learned_pass"], row["beyond_classic_pass"]) == (12, 12, 12)
    assert len(planted.tables["cells"]) == 2 * 4 * 4 * 40
    assert len(planted.tables["classic_cells"]) == 2 * 4 * 4
    failed = [r for r in planted.tables["hypotheses"] if r["verdict"] == "fail"]
    assert all(r["learned"] == r["beyond_classic"] == "" for r in failed)


def test_a_null_fails_and_its_dry_run_logs_no_trial(tmp_path: Path, real: Frozen) -> None:
    labels, baseline, pre = real
    pre = _small(pre)
    run = _labels_run(tmp_path / "labels", labels, PILOT)
    start = PILOT - timedelta(days=2)
    prices = _prices(start, PILOT, seed=1)
    tables = Synthetic(pre.doc, start, PILOT, planted=False, seed=6)
    given = (baseline, labels, run, prices, _cascades(PILOT), _book(prices, 2), tables)
    data = tmp_path / "data"
    with pytest.raises(prereg.PreregError):
        mls.dry_run(pre, *given, data, data / "ews" / "x", [], {})
    out, got = mls.dry_run(pre, *given, data, tmp_path / "dry", [], {})
    assert set(got.verdicts.values()) == {"fail"}
    assert not prereg.trials_path(data).exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["exploratory"] == mls.EXPLORATORY
    assert (out / "complete.json").exists()
    assert "EXPLORATORY DRY RUN" in (out / "report.md").read_text()
    assert {span for _, _, span in tables.reads} == {"fit"}


@pytest.mark.parametrize(
    ("lo", "hi", "margins", "aucs", "want"),
    [
        (0.021, 0.3, [0.1] * 5, [0.8] * 5, "pass"),
        (0.02, 0.3, [0.1] * 5, [0.8] * 5, "inconclusive"),
        (-0.1, 0.02, [0.1] * 5, [0.8] * 5, "inconclusive"),
        (-0.1, 0.019, [0.1] * 5, [0.8] * 5, "fail"),
        (0.021, 0.3, [0.1] * 4 + [-0.01], [0.8] * 5, "fail"),
        (0.021, 0.3, [0.1] * 4 + [0.0], [0.8] * 5, "fail"),
        (0.021, 0.3, [0.1] * 5, [0.8] * 4 + [0.3], "fail"),
        (-0.1, 0.3, [0.1] * 4 + [-0.01], [0.8] * 5, "fail"),
        (float("nan"), 0.3, [0.1] * 5, [0.8] * 5, "inconclusive"),
        (0.021, 0.3, [0.1] * 5, [0.8] * 4 + [float("nan")], "inconclusive"),
    ],
)
def test_the_twin_rule_reads_the_ci_against_the_tolerance_and_every_sign(
    lo: float, hi: float, margins: list[float], aucs: list[float], want: str
) -> None:
    margin = float(np.mean(margins))
    assert mls.learned_verdict(margin, lo, hi, margins, aucs, 0.02)[0] == want


def test_a_zero_combined_margin_fails_the_sign_test() -> None:
    assert mls.learned_verdict(0.0, 0.03, 0.3, [0.1, -0.1, 0.0, 0.0, 0.0], [0.8] * 5, 0.02)[0] == (
        "fail"
    )


@pytest.mark.parametrize(
    ("differences", "thresholds", "want"),
    [
        ([0.1] * 5, [0.05] * 5, "pass"),
        ([0.1] * 4 + [0.05], [0.05] * 5, "fail"),
        ([0.1] * 4 + [-0.01], [0.05] * 4 + [-0.05], "fail"),
        ([0.1] * 4 + [0.0], [0.05] * 4 + [-0.05], "fail"),
        ([0.1] * 5, [0.05] * 4 + [float("nan")], "inconclusive"),
    ],
)
def test_beyond_classic_needs_every_difference_above_its_shift_cutoff_and_zero(
    differences: list[float], thresholds: list[float], want: str
) -> None:
    assert mls.beyond_verdict(differences, thresholds) == want


def test_a_shift_cutoff_over_a_draw_without_an_auc_has_no_value() -> None:
    draws = np.linspace(-0.2, 0.2, 100)
    assert mls.difference_threshold(draws) == float(np.quantile(draws, 0.975))
    assert np.isnan(mls.difference_threshold(np.append(draws, np.nan)))
    aucs = np.linspace(0.3, 0.7, 100)
    assert mls.folded_threshold(aucs) == float(np.quantile(np.abs(aucs - 0.5), 0.95))
    assert np.isnan(mls.folded_threshold(np.append(aucs, np.nan)))


@pytest.mark.parametrize(
    ("aucs", "los", "his", "want"),
    [
        ([0.8] * 5, [0.6] * 5, [0.9] * 5, "pass"),
        ([0.2] * 5, [0.1] * 5, [0.4] * 5, "pass"),
        ([0.8] * 4 + [0.2], [0.6] * 4 + [0.1], [0.9] * 4 + [0.4], "fail"),
        ([0.8] * 5, [0.6] * 4 + [0.4], [0.9] * 5, "fail"),
        ([0.8] * 4 + [0.65], [0.6] * 5, [0.9] * 5, "fail"),
        ([0.8] * 5, [0.6] * 5, [float("nan")] * 5, "inconclusive"),
    ],
)
def test_warns_needs_every_primary_on_one_side_and_beyond_its_shift_cutoff(
    aucs: list[float], los: list[float], his: list[float], want: str
) -> None:
    assert mls.warn_verdict(aucs, los, his, [0.2] * 5) == want


def test_the_parts_combine_into_one_verdict_and_are_read_only_for_a_pair_that_warns() -> None:
    assert mls.pair_verdict("pass", "pass", "pass") == "pass"
    assert mls.pair_verdict("pass", "inconclusive", "fail") == "fail"
    assert mls.pair_verdict("pass", "pass", "inconclusive") == "inconclusive"
    assert mls.pair_verdict("fail", "", "") == "fail"
    assert mls.pair_verdict("inconclusive", "", "") == "inconclusive"


def _scores(
    tables: Synthetic, doc: dict[str, Any], cp: mls.Checkpoint, symbol: str, p: int, first: Any
) -> np.ndarray:
    column = doc["indicators"]["pairs"][p][0]
    x = mls.latent_series(tables, cp, symbol, doc, "fit")[column]
    return mls.latent_pipes(doc)[p](mls.rows_at(x, first, 299))


def _hand_draws(pos: list[np.ndarray], neg: list[np.ndarray], key: list[int]) -> np.ndarray:
    rng = np.random.default_rng(key)
    draws = []
    for _ in range(200):
        p = [v[rng.integers(0, len(v), len(v))] for v in pos]
        n = [v[rng.integers(0, len(v), len(v))] for v in neg]
        draws.append(scoring.auc(np.concatenate(p), np.concatenate(n)))
    return np.array(draws)


def test_the_random_streams_are_the_stated_keys(planted: mls.Evaluation, real: Frozen) -> None:
    _, baseline, pre = real
    pre = _small(pre, 39)
    doc = pre.doc
    tables = Synthetic(doc, PILOT - timedelta(days=2), PILOT, True, seed=5)
    segs, kept = planted.segments, planted.kept
    cps = mls.checkpoints(doc)
    ev = segs.events[BTC, 6].first_ms[kept["events"][BTC, 6]]
    ct = segs.controls[BTC].first_ms[kept["controls"][BTC]]
    eth_models = [cp for cp in cps if cp.model == ETH]
    boot = {}
    for cp in eth_models:
        pos, neg = _scores(tables, doc, cp, BTC, 0, ev), _scores(tables, doc, cp, BTC, 0, ct)
        boot[cp.seed, cp.kind] = _hand_draws([pos], [neg], [30, 4, 0, 6, 1, 1])
        cell = next(
            c
            for c in planted.tables["cells"]
            if (c["pair"], c["horizon_bars"], c["model"], c["group"], c["seed"], c["checkpoint"])
            == ("pc1-variance", 6, ETH, BTC, cp.seed, cp.kind)
        )
        lo, hi = np.quantile(boot[cp.seed, cp.kind], [0.025, 0.975])
        assert (cell["auc_lo"], cell["auc_hi"]) == pytest.approx((lo, hi), abs=0, rel=1e-12)

    seeds = sorted({cp.seed for cp in eth_models})
    d = np.stack([np.abs(boot[s, "trained"] - 0.5) - np.abs(boot[s, "twin"] - 0.5) for s in seeds])
    rng = np.random.default_rng([30, 7, 0, 6, 1, 1])
    picks = rng.integers(0, 5, (2000, 5))
    means = d[picks, rng.integers(0, 200, (2000, 1))].mean(1)
    row = next(
        r
        for r in planted.tables["margins"]
        if (r["pair"], r["horizon_bars"], r["model"], r["group"]) == ("pc1-variance", 6, ETH, BTC)
    )
    want = np.quantile(means, [0.025, 0.975])
    assert (row["margin_lo"], row["margin_hi"]) == pytest.approx(tuple(want), abs=0, rel=1e-9)

    cp = next(c for c in eth_models if (c.seed, c.kind) == (101, "trained"))
    keep5 = kept["events"][BTC, 5]
    anchors = segs.events[BTC, 5].anchor_ns[keep5]
    moved, sources = nulls.shift_onsets(
        anchors, segs.eligible[BTC], 288 * msvol.BAR_NS, 10, np.random.default_rng([30, 5, 5, 1])
    )
    first = moved[0] // 1_000_000 - (5 * 5 + 299) * ms.MINUTE_MS
    shifted = _scores(tables, doc, cp, BTC, 0, first)
    draw0 = next(
        r
        for r in planted.tables["shift_draws"]
        if (r["pair"], r["horizon_bars"], r["model"], r["group"], r["seed"], r["checkpoint"])
        == ("pc1-variance", 5, ETH, BTC, 101, "trained")
        and r["draw"] == 0
    )
    assert sources.all() and np.isfinite(shifted).all()
    assert draw0["auc"] == scoring.auc(shifted, _scores(tables, doc, cp, BTC, 0, ct))

    eth = segs.events[ETH, 5].first_ms[kept["events"][ETH, 5]]
    column = mls.latent_series(tables, cp, ETH, doc, "fit")["pc_1"]
    pipe = mls.latent_pipes(doc)[0]
    p = []
    for j, x in enumerate(mls.rows_at(column, eth, 299)):
        key = [30, 6, 0, 5, 1, 101, 0, 2, 0, j]
        one = [
            scoring.surrogate_test(x[None], pipe, phase_randomised, 39, rng, way)[1][0]
            for way, rng in (
                ("greater", np.random.default_rng(key)),
                ("less", np.random.default_rng(key)),
            )
        ]
        p.append(min(1.0, 2 * min(one)))
    row = next(
        r
        for r in planted.tables["surrogates"]
        if (r["pair"], r["horizon_bars"], r["model"], r["group"], r["seed"], r["checkpoint"])
        == ("pc1-variance", 5, ETH, ETH, 101, "trained")
    )
    assert row["detected"] == float(np.mean(np.array(p) <= 0.05)) > 0

    classic = msc.pipelines(baseline.doc)["variance"]
    pos = classic(segs.events[BTC, 6].x[kept["events"][BTC, 6]])
    neg = classic(segs.controls[BTC].x[kept["controls"][BTC]])
    cell = next(
        c
        for c in planted.tables["classic_cells"]
        if (c["indicator"], c["horizon_bars"], c["group"]) == ("variance", 6, BTC)
    )
    lo, hi = np.quantile(_hand_draws([pos], [neg], [30, 1, 0, 6, 1]), [0.025, 0.975])
    assert (cell["auc_lo"], cell["auc_hi"]) == pytest.approx((lo, hi), abs=0, rel=1e-12)


Given = tuple[
    prereg.Prereg,
    prereg.Prereg,
    Path,
    dict[str, ms.Minutes],
    dict[str, np.ndarray],
    dict[str, dict[str, ms.Minutes]],
    Synthetic,
]


def _confirmatory(
    tmp: Path, frozen: Frozen
) -> tuple[prereg.Prereg, prereg.Frozen, Given, Synthetic]:
    labels, baseline, pre = frozen
    pre = _small(pre)
    run = _labels_run(tmp / "labels", labels, CONFIRMATORY, stamp=True)
    start = date(2023, 9, 24)
    prices = _prices(start, CONFIRMATORY, seed=3)
    flat = mls.checkpoints(pre.doc)[0].name
    tables = Synthetic(pre.doc, start, CONFIRMATORY, True, seed=7, fit=(start, 7 * 1440), flat=flat)
    given = (baseline, labels, run, prices, _cascades(CONFIRMATORY), _book(prices, 4), tables)
    return pre, prereg.frozen(pre), given, tables


@pytest.fixture(scope="module")
def confirmed(
    tmp_path_factory: pytest.TempPathFactory, real: Frozen
) -> tuple[Path, Synthetic, mls.Evaluation]:
    tmp = tmp_path_factory.mktemp("confirmed")
    pre, ref, given, tables = _confirmatory(tmp, real)
    run_dir, evaluation = mls.score(pre, ref, *given, tmp / "data", [], {})
    return run_dir, tables, evaluation


def test_a_scoring_logs_its_1312_cells_and_a_second_is_refused(
    real: Frozen, confirmed: tuple[Path, Synthetic, mls.Evaluation]
) -> None:
    run_dir, tables, evaluation = confirmed
    data = run_dir.parent.parent
    pre, ref, given, _ = _confirmatory(run_dir.parent.parent.parent / "again", real)
    assert prereg.trials(data, pre) == BUDGET == prereg.market_latent_cells(pre.doc)
    records = [json.loads(line) for line in prereg.trials_path(data).read_text().splitlines()]
    assert len(records) == BUDGET
    assert all(r["result"] == run_dir.name for r in records)
    assert len({json.dumps(r["cell"], sort_keys=True) for r in records}) == BUDGET
    assert (run_dir / "complete.json").exists() and (run_dir / "report.md").exists()
    for table in ("cells", "classic_cells", "units", "margins", "beyond_classic", "hypotheses"):
        assert (run_dir / table).is_dir()
    assert {span for _, _, span in tables.reads} == {"fit", "confirmatory"}
    with pytest.raises(prereg.PreregError):
        mls.score(pre, ref, *given, data, [], {})
    assert prereg.trials(data, pre) == BUDGET

    counts = {
        (r["symbol"], r["horizon_bars"]): r
        for r in evaluation.tables["segments"]
        if r["series"] == "paired" and r["kind"] == "event"
    }
    assert all(counts[BTC, h]["excluded_finite"] >= 1 for h in (5, 6, 12, 24))
    for h in (5, 6, 12, 24):
        sizes = {
            c["n_pos"]
            for c in evaluation.tables["cells"]
            if (c["horizon_bars"], c["group"]) == (h, BTC)
        }
        classic = {
            c["n_pos"]
            for c in evaluation.tables["classic_cells"]
            if (c["horizon_bars"], c["group"]) == (h, BTC)
        }
        assert sizes == classic == {counts[BTC, h]["kept"]}


def test_a_crash_after_scoring_leaves_the_trials_spent(
    tmp_path: Path,
    real: Frozen,
    confirmed: tuple[Path, Synthetic, mls.Evaluation],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pre, ref, given, _ = _confirmatory(tmp_path, real)
    data = tmp_path / "data"
    monkeypatch.setattr(mls, "evaluate", lambda *args, **kwargs: confirmed[2])

    def killed(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(prereg, "finish_result", killed)
    with pytest.raises(KeyboardInterrupt):
        mls.score(pre, ref, *given, data, [], {})
    (torn,) = [p for p in (data / "ews").iterdir() if p.is_dir()]
    assert not (torn / "report.md").exists()
    assert prereg.trials(data, pre) == BUDGET
    with pytest.raises(prereg.PreregError):
        mls.score(pre, ref, *given, data, [], {})
    shutil.rmtree(torn)
    with pytest.raises(prereg.PreregError, match="budget"):
        mls.score(pre, ref, *given, data, [], {})
    assert prereg.trials(data, pre) == BUDGET


def _pins(doc: dict[str, Any]) -> bytes:
    rows = [dict(r) | {"asset": r["symbol"][:3].lower()} for r in doc["latents"]["checkpoints"]]
    return json.dumps({"steps": doc["training"]["steps"], "checkpoints": rows}).encode()


def _record(pre: prereg.Prereg, pins: bytes) -> dict[str, Any]:
    ref = prereg.frozen(pre)
    root = prereg.repo_root(pre.path)
    return {
        "format": 2,
        "prereg": {
            **prereg.stamp(pre, ref),
            "tag_object": _git(root, "rev-parse", ref.tag),
            "path": "paper/prereg/market-latent-v1.toml",
            "text": pre.path.read_bytes().decode("utf-8"),
        },
        "code": {"changed": [], "uncommitted": [], "head": ref.commit},
        "pins_sha256": hashlib.sha256(pins).hexdigest(),
        "checkpoints": mls._listed(pre.doc),
    }


def _swap_pin(pins: bytes) -> bytes:
    found = json.loads(pins)
    found["checkpoints"][0]["trained_sha256"] = "0" * 64
    return json.dumps(found).encode()


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        (lambda r, p: (r | {"format": 1}, p), "format 2"),
        (lambda r, p: (r | {"prereg": r["prereg"] | {"sha256": "0" * 64}}, p), "sha256"),
        (
            lambda r, p: (r | {"prereg": r["prereg"] | {"text": r["prereg"]["text"] + " "}}, p),
            "text",
        ),
        (lambda r, p: (r | {"prereg": r["prereg"] | {"tag_object": "0" * 40}}, p), "tag_object"),
        (lambda r, p: (r | {"code": r["code"] | {"changed": ["ews/x.py"]}}, p), "changed"),
        (lambda r, p: (r, p + b" "), "pins.json is not"),
        (lambda r, p: (r | {"checkpoints": r["checkpoints"][1:]}, p), "checkpoints are not"),
        (
            lambda r, p: (
                r | {"pins_sha256": hashlib.sha256(_swap_pin(p)).hexdigest()},
                _swap_pin(p),
            ),
            "differ",
        ),
        (lambda r, p: ({k: v for k, v in r.items() if k != "code"}, p), "not a whole"),
    ],
)
def test_the_gate_record_must_be_the_frozen_files_and_pins(
    real: Frozen, change: Any, refusal: str
) -> None:
    pre = real[2]
    pins = _pins(pre.doc)
    record = _record(pre, pins)
    mls.check_gate(pre, prereg.frozen(pre), record, pins)
    bad, bad_pins = change(json.loads(json.dumps(record)), pins)
    with pytest.raises(prereg.PreregError, match=refusal):
        mls.check_gate(pre, prereg.frozen(pre), bad, bad_pins)


def _short(pre: prereg.Prereg, end: date) -> prereg.Prereg:
    season = pre.doc["seasonality"]
    fit = season["fit"] | {"start": date(2023, 3, 1), "end": end}
    return replace(pre, doc=pre.doc | {"seasonality": season | {"fit": fit}})


def _merged(root: Path, pre: prereg.Prereg, labels: prereg.Prereg, record: dict[str, Any]) -> None:
    start, minutes, _ = mls.extent(pre.doc, labels, "fit")
    ts = start * 1_000_000 + 60_000_000_000 * np.arange(minutes, dtype=np.int64)
    entries = {e["run_id"]: e for e in pre.doc["latents"]["checkpoints"]}
    scaling = {e["symbol"]: e["scaling_sha256"] for e in pre.doc["latents"]["checkpoints"]}
    for k, cp in enumerate(mls.checkpoints(pre.doc)):
        entry = entries[cp.run_id]
        for symbol in (BTC, ETH):
            folder = root / "fit" / cp.name
            folder.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng([k, len(symbol)])
            valid = np.ones(minutes, bool)
            valid[:10] = False
            table = pa.table(
                {"ts_ns": ts, "valid": valid}
                | {c: rng.standard_normal(minutes) for c in ("surprise", "surprise5", "pc_1")}
            )
            path = folder / f"{symbol}-fit.parquet"
            pq.write_table(table, path)
            manifest = {
                "format": 1,
                "pre_registered": True,
                "gate": mls.stamp_of(record),
                "code": {"git_sha": record["code"]["head"], "dirty": False},
                "checkpoint": {
                    "run_id": cp.run_id,
                    "kind": cp.kind,
                    "step": cp.step,
                    "sha256": entry[f"{cp.kind}_sha256"],
                    "pca_sha256": entry["pca_sha256"],
                    "symbol": cp.model,
                    "seed": cp.seed,
                    "role": "primary",
                },
                "scaling_sha256": scaling[symbol],
                "symbol": symbol,
                "span": "fit",
                "pre_minutes": 0,
                "minutes": minutes,
                "parquet": path.name,
                "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            path.with_suffix(".json").write_text(json.dumps(manifest))


def test_the_merged_latents_are_read_only_as_their_manifests_and_the_gate_name_them(
    tmp_path: Path, real: Frozen
) -> None:
    labels, _, pre = real
    pins = _pins(pre.doc)
    record = _record(pre, pins)
    short = _short(pre, date(2023, 3, 2))
    _merged(tmp_path, short, labels, record)
    merged = mls.Merged(tmp_path, record, short, labels, ("fit",))
    cp = mls.checkpoints(pre.doc)[3]
    got = merged.read(cp, ETH, "fit")
    assert set(got) == {"pc_1", "surprise"}
    assert np.isnan(got["pc_1"].values[:10]).all() and np.isfinite(got["pc_1"].values[10:]).all()
    with pytest.raises(ms.SegmentError):
        merged.read(cp, ETH, "confirmatory")
    with pytest.raises(ms.SegmentError):
        mls.Merged(tmp_path, record, _short(pre, CONFIRMATORY), labels, ("fit",))

    path = tmp_path / "fit" / cp.name / "ETHUSDT-fit.parquet"
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(prereg.PreregError, match="not the file"):
        merged.read(cp, ETH, "fit")
    other = record | {"code": record["code"] | {"head": "0" * 40}}
    with pytest.raises(prereg.PreregError, match="another gate record"):
        mls.Merged(tmp_path, other, short, labels, ("fit",))


def _unread(tmp: Path, labels: prereg.Prereg, monkeypatch: pytest.MonkeyPatch) -> Path:
    span = labels.doc["data"]["span"]
    run = tmp / "run"
    run.mkdir()
    params = {"start": span["start"].isoformat(), "end": span["end"].isoformat()}
    (run / "manifest.json").write_text(json.dumps({"run_id": run.name, "params": params}))

    def read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("read past the gate")

    for module, name in (
        (ms, "check_labels"),
        (msc, "cascade_onsets"),
        (msc, "check_sources"),
        (archive, "collect"),
        (mls.Merged, "read"),
    ):
        monkeypatch.setattr(module, name, read)
    return run


CLEAN_CODE = {"changed": [], "uncommitted": []}


@pytest.mark.parametrize(
    ("code", "dirty", "spent", "gate", "refusal"),
    [
        ({"changed": ["ews/market_score.py"], "uncommitted": []}, False, False, True, "not at"),
        ({"changed": [], "uncommitted": ["ews/prereg.py"]}, False, False, True, "not committed"),
        (CLEAN_CODE, True, False, True, "uncommitted changes"),
        (CLEAN_CODE, False, True, True, "budget"),
        (CLEAN_CODE, False, False, False, "another file"),
    ],
)
def test_a_confirmatory_run_refuses_before_reading(
    tmp_path: Path,
    real: Frozen,
    monkeypatch: pytest.MonkeyPatch,
    code: dict[str, list[str]],
    dirty: bool,
    spent: bool,
    gate: bool,
    refusal: str,
) -> None:
    labels, _, pre = real
    run = _unread(tmp_path, labels, monkeypatch)
    monkeypatch.setattr(prereg, "code_check", lambda *_: code)
    monkeypatch.setattr(mls.io, "code_version", lambda: {"git_sha": "0" * 40, "dirty": dirty})
    data = tmp_path / "data"
    if spent:
        stamp = {"id": pre.id, "version": pre.version}
        prereg.log_trials(data, [{"prereg": stamp}] * BUDGET)
    pins = _pins(pre.doc)
    record = _record(pre, pins)
    if not gate:
        record["prereg"]["sha256"] = "0" * 64
    (tmp_path / "gate.json").write_text(json.dumps(record))
    (tmp_path / "pins.json").write_bytes(pins)
    none = tmp_path / "none"
    with pytest.raises(prereg.PreregError, match=refusal):
        mls.run(
            pre,
            run,
            none,
            none,
            none,
            none,
            tmp_path / "gate.json",
            tmp_path / "pins.json",
            data,
        )
    assert prereg.trials(data, pre) == (BUDGET if spent else 0)


def test_a_dry_run_refuses_any_day_from_2024(tmp_path: Path, real: Frozen) -> None:
    labels, baseline, pre = real
    small = _small(pre)
    late = _labels_run(tmp_path / "late", labels, date(2023, 12, 25))
    none = tmp_path / "none"
    with pytest.raises(ms.SegmentError):
        mls.run(pre, late, none, none, none, none, none, none, tmp_path, exploratory=True)
    early = _labels_run(tmp_path / "early", labels, PILOT)
    start = PILOT - timedelta(days=2)
    prices = _prices(start, PILOT, seed=1)
    reaching = _prices(start, CONFIRMATORY - timedelta(days=DAYS - 1), seed=1)
    tables = Synthetic(small.doc, start, PILOT, False, seed=5)
    late_tables = Synthetic(small.doc, start, CONFIRMATORY - timedelta(days=DAYS - 1), False, 5)
    late_cascades = {s: np.array([archive.day_ms(CONFIRMATORY) * 1_000_000]) for s in (BTC, ETH)}
    for given in (
        (reaching, _cascades(PILOT), _book(prices, 2), tables),
        (prices, late_cascades, _book(prices, 2), tables),
        (prices, _cascades(PILOT), _book(reaching, 2), tables),
        (prices, _cascades(PILOT), _book(prices, 2), late_tables),
    ):
        with pytest.raises(ms.SegmentError):
            mls.evaluate(small, baseline, labels, early, *given, confirmatory=False)
    args = ["score", "--prereg", str(pre.path), "--exploratory", "--run", str(late)]
    args += ["--cascades", str(none), "--book", str(none), "--latents", str(none)]
    args += ["--gate", str(none), "--pins", str(none), "--data-dir", str(tmp_path / "data")]
    assert main(args) == 1
    assert main([*args[:-8], "--data-dir", str(tmp_path / "data")]) == 1
    assert not prereg.trials_path(tmp_path / "data").exists()


def test_the_scorer_refuses_a_file_whose_rules_it_does_not_carry_out(real: Frozen) -> None:
    pre = real[2]
    mls.check_file(pre)
    for section, key, value in (
        ("cells", "judged", "same-asset"),
        ("descriptive", "ablations", ["ema"]),
    ):
        doc = pre.doc | {section: pre.doc[section] | {key: value}}
        with pytest.raises(prereg.PreregError):
            mls.check_file(replace(pre, doc=doc))
    hypotheses = [h | {"prediction": h["prediction"] + " "} for h in pre.doc["hypotheses"]]
    with pytest.raises(prereg.PreregError, match="PREDICTION"):
        mls.check_file(replace(pre, doc=pre.doc | {"hypotheses": hypotheses}))


def test_the_incremental_auc_of_a_score_that_adds_nothing_is_zero() -> None:
    rng = np.random.default_rng(0)
    rv = rng.standard_normal(60)
    y = (rv + rng.standard_normal(60) > 0).astype(float)
    ones = np.ones((1, 60))
    alone = mls.logistic_auc(rv[:, None], y, ones)[0]
    assert alone == pytest.approx(scoring.auc(rv[y == 1], rv[y == 0]), abs=1e-12)
    both = mls.logistic_auc(np.column_stack([rv, 2 * rv]), y, ones)[0]
    assert both == pytest.approx(alone, abs=1e-12)
    w = np.vstack([ones[0], rng.integers(0, 3, 60).astype(float)])
    got = mls.weighted_auc(np.vstack([rv, rv]), y, w)
    keep = np.repeat(np.arange(60), w[1].astype(int))
    assert got[1] == pytest.approx(scoring.auc(rv[keep][y[keep] == 1], rv[keep][y[keep] == 0]))
