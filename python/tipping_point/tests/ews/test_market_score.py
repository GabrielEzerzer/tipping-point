# The market-ews scorer on synthetic runs: planted signal and null, verdict edges, the stated
# random streams, the trial budget through a crash and a race, the confirmatory gates before any
# read, and a dry run that reads no day from 2024-01-01 and logs no trial.

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from tipping_point.ews import market_score as msc
from tipping_point.ews import market_segments as ms
from tipping_point.ews import prereg, scoring
from tipping_point.ews.__main__ import main
from tipping_point.ews.surrogates import phase_randomised
from tipping_point.market import archive, msvol, nulls

BTC, ETH = "BTCUSDT", "ETHUSDT"
DAYS = 14
EVENTS = [288 + 96 * k for k in range(18)]
ETH_LAG = 48
NULL_DAYS = range(8, DAYS)
RAMP = 420
FLOOR = 0.55


def _small(pre: prereg.Prereg) -> prereg.Prereg:
    sc = pre.doc["scoring"]
    small = {"bootstrap": 200, "shift": sc["shift"] | {"draws": 20}}
    doc = pre.doc | {"scoring": sc | small | {"surrogates": sc["surrogates"] | {"n": 19}}}
    return replace(pre, doc=doc)


def _onsets(symbol: str) -> list[int]:
    return [b + (ETH_LAG if symbol == ETH else 0) for b in EVENTS]


def _labels_run(
    data: Path, labels: prereg.Prereg, first: date, end: date | None = None, stamp: bool = False
) -> Path:
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
        "end": (end or first + timedelta(days=DAYS - 1)).isoformat(),
        "symbols": [BTC, ETH],
        "labeller": labels.doc["labeller"],
    }
    if stamp:
        params["prereg"] = {"sha256": labels.sha256, "tag": labels.tag}
    return msvol.write_run(data, "synthetic", tables, params, [], [], {})


def _series(start: date, first: date, planted: bool, seed: int) -> dict[str, ms.Minutes]:
    rng = np.random.default_rng(seed)
    offset = (first - start).days * 1440
    n = offset + DAYS * 1440
    out = {}
    for s in (BTC, ETH):
        sigma = np.ones(n)
        if planted:
            for b in _onsets(s):
                onset = offset + 5 * b
                sigma[onset - RAMP : onset] = 1 + 3 * np.arange(RAMP) / RAMP
        out[s] = ms.Minutes(archive.day_ms(start), 1e-4 * sigma * rng.standard_normal(n))
    return out


def _book(prices: dict[str, ms.Minutes], seed: int) -> dict[str, dict[str, ms.Minutes]]:
    rng = np.random.default_rng(seed)
    return {
        s: {"ofi": ms.Minutes(m.start_ms, rng.standard_normal(len(m.values)))}
        for s, m in prices.items()
    }


def _floor(pre: prereg.Prereg) -> dict[str, nulls.Floor]:
    cells = [
        {"scorer": name, "horizon_bars": h, "auc": FLOOR, "n_pos": 1, "n_neg": 1}
        for name in pre.doc["indicators"]["names"]
        for h in pre.doc["segments"]["horizons"]
    ]
    return {s: nulls.Floor(cells, [], []) for s in (BTC, ETH)}


def _cascades(first: date) -> dict[str, np.ndarray]:
    t0 = archive.day_ms(first) * 1_000_000
    return {s: t0 + msvol.BAR_NS * np.array(_onsets(s)[:6], np.int64) for s in (BTC, ETH)}


PILOT = date(2023, 3, 1)


def _dry(tmp: Path, frozen: tuple[prereg.Prereg, prereg.Prereg], planted: bool) -> msc.Evaluation:
    labels, pre = frozen
    pre = _small(pre)
    run = _labels_run(tmp, labels, PILOT)
    prices = _series(PILOT - timedelta(days=2), PILOT, planted, seed=1)
    return msc.evaluate(
        pre,
        labels,
        run,
        prices,
        _cascades(PILOT),
        _book(prices, 2),
        _floor(pre),
        confirmatory=False,
    )


@pytest.fixture(scope="module")
def planted(
    tmp_path_factory: pytest.TempPathFactory, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> msc.Evaluation:
    return _dry(tmp_path_factory.mktemp("planted"), frozen, planted=True)


def test_a_planted_variance_ramp_passes_and_ac_on_the_same_data_fails(
    planted: msc.Evaluation,
) -> None:
    assert planted.verdicts == {"variance-rises": "pass", "ac1-rises": "fail"}
    cells = planted.tables["cells"]
    assert len(cells) == 32
    assert sum(c["decisive"] for c in cells) == 12
    decisive = [c for c in cells if c["decisive"] and c["indicator"] == "variance"]
    assert {(c["horizon_bars"], c["group"]) for c in decisive} == {
        (h, g) for h in (5, 6) for g in ("pooled", BTC, ETH)
    }
    for c in decisive:
        assert c["floor_auc"] == FLOOR
        assert c["auc_lo"] > max(FLOOR, c["shift_q975"])
        assert c["n_pos"] == (36 if c["group"] == "pooled" else 18)
    eth_only = [c for c in cells if c["group"] == "eth-only"]
    assert not any(c["decisive"] for c in eth_only)
    assert all(c["n_pos"] == 18 for c in eth_only)


def test_a_lower_end_at_a_threshold_is_not_above_it() -> None:
    assert msc.cell_verdict(0.5, 0.4, 0.4) == "fail"
    assert msc.cell_verdict(0.6, 0.6, 0.4) == "fail"
    assert msc.cell_verdict(0.6, 0.4, 0.6) == "fail"
    assert msc.cell_verdict(np.nextafter(0.6, 1), 0.6, 0.6) == "pass"
    for missing in ((np.nan, 0.4, 0.4), (0.9, np.nan, 0.4), (0.9, 0.4, np.nan)):
        assert msc.cell_verdict(*missing) == "inconclusive"


def test_an_empty_or_inconclusive_cell_does_not_pass() -> None:
    assert msc.hypothesis_verdict(["pass"] * 6) == "pass"
    assert msc.hypothesis_verdict(["pass"] * 5 + ["inconclusive"]) == "inconclusive"
    assert msc.hypothesis_verdict(["pass"] * 5 + ["fail"]) == "fail"
    assert msc.hypothesis_verdict(["inconclusive", "fail"]) == "fail"
    assert msc.hypothesis_verdict([]) == "inconclusive"
    doc = {"scoring": {"seed": 30, "bootstrap": 50, "ci_level": 0.95}}
    empty = msc.auc_cell([np.empty(0)], [np.ones(5)], doc, 0, 5, 1)
    assert empty["n_pos"] == 0
    assert msc.cell_verdict(empty["auc_lo"], 0.4, 0.4) == "inconclusive"


def _hand_ci(pos: list[np.ndarray], neg: list[np.ndarray], key: list[int]) -> tuple[float, float]:
    rng = np.random.default_rng(key)
    draws = []
    for _ in range(200):
        p = [v[rng.integers(0, len(v), len(v))] for v in pos]
        n = [v[rng.integers(0, len(v), len(v))] for v in neg]
        draws.append(scoring.auc(np.concatenate(p), np.concatenate(n)))
    lo, hi = np.quantile(draws, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def test_the_random_streams_are_the_stated_keys(
    planted: msc.Evaluation, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    pre = _small(frozen[1])
    pipes = msc.pipelines(pre.doc)
    segs = planted.segments
    ev = {s: pipes["ac"](segs.events[s, 6].x) for s in (BTC, ETH)}
    ct = {s: pipes["ac"](segs.controls[s].x) for s in (BTC, ETH)}
    cells = {(c["indicator"], c["horizon_bars"], c["group"]): c for c in planted.tables["cells"]}
    for group, code, chosen in ((ETH, 2, [ETH]), ("pooled", 0, [BTC, ETH])):
        cell = cells["ac", 6, group]
        want = _hand_ci([ev[s] for s in chosen], [ct[s] for s in chosen], [30, 1, 1, 6, code])
        assert (cell["auc_lo"], cell["auc_hi"]) == pytest.approx(want, abs=0, rel=1e-12)

    block = segs.events[BTC, 5]
    rng = np.random.default_rng([30, 2, 5, 1])
    moved, kept = nulls.shift_onsets(
        block.anchor_ns, segs.eligible[BTC], 288 * msvol.BAR_NS, 20, rng
    )
    first = moved[0] // 1_000_000 - (5 * 5 + segs.span) * ms.MINUTE_MS
    rows = ms._cut(planted.series[BTC], first, segs.span, np.zeros(0, bool), 0)[0]
    draw0 = next(
        d
        for d in planted.tables["shift_draws"]
        if (d["indicator"], d["horizon_bars"], d["group"], d["draw"]) == ("variance", 5, BTC, 0)
    )
    assert kept.all()
    assert draw0["auc"] == scoring.auc(
        pipes["variance"](rows), pipes["variance"](segs.controls[BTC].x)
    )

    x = segs.controls[ETH].x
    p = []
    for j, row in enumerate(x):
        rng = np.random.default_rng([30, 3, 0, 0, 2, 1, j])
        p.append(
            scoring.surrogate_test(row[None], pipes["variance"], phase_randomised, 19, rng)[1][0]
        )
    row = next(
        r
        for r in planted.tables["surrogates"]
        if (r["indicator"], r["horizon_bars"], r["group"]) == ("variance", 5, ETH)
    )
    assert row["false_positive"] == float(np.mean(np.array(p) <= 0.05))


CONFIRMATORY = prereg.CONFIRMATORY


def test_a_scoring_logs_its_32_cells_and_a_second_is_refused(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg], monkeypatch: pytest.MonkeyPatch
) -> None:
    pre, ref, given = _confirmatory(tmp_path, frozen, monkeypatch)
    data = tmp_path / "data"
    run_dir, evaluation = msc.score(pre, ref, *given, data, [], {})
    assert prereg.trials(data, pre) == 32
    assert evaluation.verdicts["variance-rises"] == "pass"
    records = [json.loads(line) for line in prereg.trials_path(data).read_text().splitlines()]
    assert {
        (r["cell"]["indicator"], r["cell"]["horizon_bars"], r["cell"]["group"]) for r in records
    } == {(c["indicator"], c["horizon_bars"], c["group"]) for c in evaluation.tables["cells"]}
    assert (run_dir / "complete.json").exists() and (run_dir / "report.md").exists()
    for table in ("cells", "hypotheses", "shift_draws", "seasonality", "strata", "cascades"):
        assert (run_dir / table).is_dir()
    with pytest.raises(prereg.PreregError):
        msc.score(pre, ref, *given, data, [], {})
    assert prereg.trials(data, pre) == 32


Given = tuple[
    prereg.Prereg,
    Path,
    dict[str, ms.Minutes],
    dict[str, np.ndarray],
    dict[str, dict[str, ms.Minutes]],
    dict[str, nulls.Floor],
]


def _confirmatory(
    tmp: Path, frozen: tuple[prereg.Prereg, prereg.Prereg], monkeypatch: pytest.MonkeyPatch
) -> tuple[prereg.Prereg, prereg.Frozen, Given]:
    labels, frozen_pre = frozen
    gate = ms.check_scoring
    monkeypatch.setattr(ms, "check_scoring", lambda _: gate(frozen_pre))
    pre = _small(frozen_pre)
    run = _labels_run(tmp / "labels", labels, CONFIRMATORY, stamp=True)
    prices = _series(date(2023, 9, 24), CONFIRMATORY, planted=True, seed=3)
    given = (labels, run, prices, _cascades(CONFIRMATORY), _book(prices, 4), _floor(pre))
    return pre, prereg.frozen(pre), given


def test_a_crash_before_the_tables_leaves_the_trials_spent(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg], monkeypatch: pytest.MonkeyPatch
) -> None:
    pre, ref, given = _confirmatory(tmp_path, frozen, monkeypatch)
    data = tmp_path / "data"
    real = prereg.finish_result

    def killed(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(prereg, "finish_result", killed)
    with pytest.raises(KeyboardInterrupt):
        msc.score(pre, ref, *given, data, [], {})
    (torn,) = [p for p in (data / "ews").iterdir() if p.is_dir()]
    assert not (torn / "report.md").exists()
    assert prereg.trials(data, pre) == 32
    with pytest.raises(prereg.PreregError):
        msc.score(pre, ref, *given, data, [], {})
    shutil.rmtree(torn)
    monkeypatch.setattr(prereg, "finish_result", real)
    with pytest.raises(prereg.PreregError, match="budget"):
        msc.score(pre, ref, *given, data, [], {})
    assert prereg.trials(data, pre) == 32


def test_a_concurrent_scoring_refuses_before_evaluating(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg], monkeypatch: pytest.MonkeyPatch
) -> None:
    pre, ref, given = _confirmatory(tmp_path, frozen, monkeypatch)
    data = tmp_path / "data"
    begin = prereg.begin_result

    def raced(*args: Any, **kwargs: Any) -> Path:
        mine = begin(*args, **kwargs)
        shutil.copytree(mine, mine.with_name(f"{mine.name}-other"))
        return mine

    def evaluated(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("evaluated")

    monkeypatch.setattr(prereg, "begin_result", raced)
    monkeypatch.setattr(msc, "evaluate", evaluated)
    with pytest.raises(prereg.PreregError, match=r"no complete\.json"):
        msc.score(pre, ref, *given, data, [], {})
    assert prereg.trials(data, pre) == 0


def _unread(tmp: Path, labels: prereg.Prereg, monkeypatch: pytest.MonkeyPatch) -> Path:
    span = labels.doc["data"]["span"]
    run = tmp / "run"
    run.mkdir()
    params = {"start": span["start"].isoformat(), "end": span["end"].isoformat()}
    (run / "manifest.json").write_text(json.dumps({"run_id": run.name, "params": params}))

    def read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("read past the gate")

    for module, name in ((ms, "check_labels"), (msc, "cascade_onsets"), (archive, "collect")):
        monkeypatch.setattr(module, name, read)
    return run


CLEAN_CODE = {"changed": [], "uncommitted": []}


@pytest.mark.parametrize(
    ("code", "dirty", "spent", "refusal"),
    [
        ({"changed": ["ews/market_segments.py"], "uncommitted": []}, False, False, "not at"),
        ({"changed": [], "uncommitted": ["ews/market_score.py"]}, False, False, "not committed"),
        (CLEAN_CODE, True, False, "uncommitted changes"),
        (CLEAN_CODE, False, True, "budget"),
    ],
)
def test_a_confirmatory_run_refuses_before_reading(
    tmp_path: Path,
    frozen: tuple[prereg.Prereg, prereg.Prereg],
    monkeypatch: pytest.MonkeyPatch,
    code: dict[str, list[str]],
    dirty: bool,
    spent: bool,
    refusal: str,
) -> None:
    labels, pre = frozen
    run = _unread(tmp_path, labels, monkeypatch)
    monkeypatch.setattr(prereg, "code_check", lambda *_: code)
    monkeypatch.setattr(msc.io, "code_version", lambda: {"git_sha": "0" * 40, "dirty": dirty})
    data = tmp_path / "data"
    if spent:
        stamp = {"id": pre.id, "version": pre.version}
        prereg.log_trials(data, [{"prereg": stamp}] * pre.doc["trials"]["budget"])
    none = tmp_path / "none"
    with pytest.raises(prereg.PreregError, match=refusal):
        msc.run(pre, run, none, none, none, data)


def test_a_confirmatory_run_refuses_while_the_span_is_closed(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg], monkeypatch: pytest.MonkeyPatch
) -> None:
    labels, pre = frozen
    run = _unread(tmp_path, labels, monkeypatch)

    def closed(_: prereg.Prereg) -> None:
        raise ms.SegmentError("closed")

    def checked(*args: Any) -> None:
        raise AssertionError("code checked")

    monkeypatch.setattr(ms, "check_scoring", closed)
    monkeypatch.setattr(prereg, "code_check", checked)
    none = tmp_path / "none"
    with pytest.raises(ms.SegmentError, match="closed"):
        msc.run(pre, run, none, none, none, tmp_path / "data")


def test_a_null_fails_and_its_dry_run_logs_no_trial(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    labels, pre = frozen
    pre = _small(pre)
    run = _labels_run(tmp_path / "labels", labels, PILOT)
    prices = _series(PILOT - timedelta(days=2), PILOT, planted=False, seed=1)
    given = (labels, run, prices, _cascades(PILOT), _book(prices, 2), _floor(pre))
    data = tmp_path / "data"
    with pytest.raises(prereg.PreregError):
        msc.dry_run(pre, *given, data, data / "ews" / "x", [], {})
    out, got = msc.dry_run(pre, *given, data, tmp_path / "dry", [], {})
    assert got.verdicts == {"variance-rises": "fail", "ac1-rises": "fail"}
    assert not prereg.trials_path(data).exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["exploratory"] == msc.EXPLORATORY
    assert (out / "complete.json").exists()
    assert "EXPLORATORY DRY RUN" in (out / "report.md").read_text()


def test_a_dry_run_refuses_any_day_from_2024(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    labels, pre = frozen
    small = _small(pre)
    late = _labels_run(tmp_path / "late", labels, date(2023, 12, 25))
    with pytest.raises(ms.SegmentError):
        msc.run(
            pre,
            late,
            tmp_path / "none",
            tmp_path / "none",
            tmp_path / "none",
            tmp_path,
            exploratory=True,
        )
    with pytest.raises(ms.SegmentError):
        msc.evaluate(small, labels, late, {}, {}, {}, {}, confirmatory=False)
    early = _labels_run(tmp_path / "early", labels, PILOT)
    reaching = _series(PILOT - timedelta(days=2), CONFIRMATORY - timedelta(days=DAYS - 1), False, 7)
    with pytest.raises(ms.SegmentError):
        msc.evaluate(
            small,
            labels,
            early,
            reaching,
            _cascades(PILOT),
            {},
            _floor(small),
            confirmatory=False,
        )
    cascade = tmp_path / "cascade"
    cascade.mkdir()
    params = {"cascade": labels.doc["cascade"], "symbols": [BTC, ETH], "end": "2024-01-01"}
    (cascade / "manifest.json").write_text(json.dumps({"params": params}))
    with pytest.raises(ms.SegmentError):
        msc.cascade_onsets(cascade, labels, [BTC, ETH], confirmatory=False)
    args = ["score", "--prereg", str(pre.path), "--exploratory", "--run", str(late)]
    args += ["--cascades", str(cascade), "--book", str(tmp_path / "none")]
    code = main([*args, "--data-dir", str(tmp_path / "data")])
    assert code == 1
    assert not prereg.trials_path(tmp_path / "data").exists()


def test_each_event_segment_reports_its_own_high_share(planted: msc.Evaluation) -> None:
    rows = planted.tables["high_share"]
    segs = planted.segments
    for s in (BTC, ETH):
        for h in (5, 6, 12, 24):
            mine = [r for r in rows if (r["symbol"], r["horizon_bars"]) == (s, h)]
            assert [r["onset_ns"] for r in mine] == segs.events[s, h].anchor_ns.tolist()
            assert [r["high_share"] for r in mine] == segs.events[s, h].high_share.tolist()
    assert len(rows) == 2 * 4 * len(EVENTS)
    assert len(planted.tables["high_share_mean"]) == 2 * 4


def test_the_eth_only_shift_counts_moved_onsets_near_btc_as_dropped(
    planted: msc.Evaluation,
) -> None:
    by_cell: dict[tuple[str, int], int] = {}
    for d in planted.tables["shift_draws"]:
        if d["group"] == "eth-only":
            key = (d["indicator"], d["horizon_bars"])
            by_cell[key] = by_cell.get(key, 0) + len(EVENTS) - d["n_pos"]
    cells = [c for c in planted.tables["cells"] if c["group"] == "eth-only"]
    dropped = {(c["indicator"], c["horizon_bars"]): c["shift_dropped_near_btc"] for c in cells}
    assert dropped == by_cell
    assert sum(by_cell.values()) > 0
    others = [c for c in planted.tables["cells"] if c["group"] != "eth-only"]
    assert all(c["shift_dropped_near_btc"] == 0 for c in others)


def _shift_segments(btc_bar: int) -> ms.Segments:
    bar = msvol.BAR_NS
    eth = ms.Block(np.zeros((1, 3)), np.zeros(1, np.int64), np.array([0]), np.zeros(1))
    return ms.Segments(
        span=3,
        events={(ETH, 5): eth},
        controls={},
        onsets={BTC: np.array([btc_bar * bar]), ETH: np.array([0])},
        counts=[],
        eligible={ETH: bar * np.arange(1000, 1011)},
    )


@pytest.mark.parametrize(("btc_bar", "alone"), [(1005, False), (5000, True)])
def test_a_moved_eth_onset_is_eth_only_only_away_from_btc(btc_bar: int, alone: bool) -> None:
    doc = {
        "scoring": {"seed": 30, "shift": {"draws": 7, "min_offset_bars": 288}},
        "reporting": {"eth_only_bars": 12},
    }
    got = msc.moved_eth_alone(_shift_segments(btc_bar), [BTC, ETH], 5, np.array([True]), doc)
    assert got.shape == (7, 1)
    assert (got == alone).all()
    kept = {(ETH, 5): np.array([True])}
    shift = msc.Shift({("variance", ETH, 5): np.ones((7, 1))}, kept, {5: got})
    scored = msc.Scored({}, {}, {5: np.array([True])})
    positives = msc.shift_positives(shift, scored, [BTC, ETH], "variance", 5, "eth-only")
    assert np.isfinite(positives).all() == alone
    assert np.isnan(ms.shift_threshold(positives, np.zeros(4))) != alone


def test_the_book_tier_keeps_the_same_minutes_for_price_and_every_feature() -> None:
    first, fit = date(2023, 2, 6), (date(2023, 2, 6), date(2023, 2, 8))
    n = 4 * 1440
    rng = np.random.default_rng(8)
    price = ms.Minutes(archive.day_ms(first), rng.standard_normal(n))
    feature = rng.standard_normal(n)
    for day in range(3):
        feature[day * 1440 : day * 1440 + 5] = np.nan
    book = {BTC: {"ofi": ms.Minutes(price.start_ms, feature)}}
    got = msc.tier_series({BTC: price}, book, fit, "mean")
    kept = np.isfinite(got["price"][BTC].values)
    assert np.array_equal(kept, np.isfinite(got["ofi"][BTC].values))
    assert not kept[3 * 1440 : 3 * 1440 + 5].any()
    assert kept[3 * 1440 + 5 :].all()


SOURCE_SPAN = (date(2022, 12, 31), date(2023, 1, 1))


def _archive(root: Path) -> list[dict[str, Any]]:
    recorded = []
    for source in archive.sources([BTC, ETH], *SOURCE_SPAN):
        path = root / source.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.url.encode())
        sha256 = hashlib.sha256(source.url.encode()).hexdigest()
        path.with_name(path.name + ".CHECKSUM").write_text(f"{sha256}  {path.name}\n")
        recorded.append({"url": source.url, "bytes": path.stat().st_size, "sha256": sha256})
    return recorded


def _changed(root: Path, source: archive.Source) -> None:
    path = root / source.path
    path.write_bytes(b"changed")
    sha256 = hashlib.sha256(b"changed").hexdigest()
    path.with_name(path.name + ".CHECKSUM").write_text(f"{sha256}  {path.name}\n")


def test_the_archive_must_be_the_one_the_labels_were_made_from(tmp_path: Path) -> None:
    recorded = _archive(tmp_path)
    klines = archive.sources([BTC, ETH], *SOURCE_SPAN)[0]
    manifest = {"run_id": "r", "sources": recorded, "missing_sources": []}
    msc.check_sources(manifest, [BTC, ETH], *SOURCE_SPAN, tmp_path)
    unrecorded = manifest | {"sources": [r for r in recorded if r["url"] != klines.url]}
    gone = manifest | {"sources": unrecorded["sources"], "missing_sources": [klines.url]}
    for refused in (unrecorded, gone):
        with pytest.raises(ms.SegmentError):
            msc.check_sources(refused, [BTC, ETH], *SOURCE_SPAN, tmp_path)
    _changed(tmp_path, klines)
    with pytest.raises(ms.SegmentError, match="sha256"):
        msc.check_sources(manifest, [BTC, ETH], *SOURCE_SPAN, tmp_path)
    (tmp_path / klines.path).unlink()
    with pytest.raises(ms.SegmentError):
        msc.check_sources(manifest, [BTC, ETH], *SOURCE_SPAN, tmp_path)
    msc.check_sources(gone, [BTC, ETH], *SOURCE_SPAN, tmp_path)


def test_a_dry_run_refuses_a_changed_archive_before_reading(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg], monkeypatch: pytest.MonkeyPatch
) -> None:
    labels, pre = frozen
    root = tmp_path / "archive"
    recorded = _archive(root)
    _changed(root, archive.sources([BTC, ETH], *SOURCE_SPAN)[3])
    run = _unread(tmp_path, labels, monkeypatch)
    params = {"start": "2022-07-01", "end": SOURCE_SPAN[1].isoformat()}
    manifest = {"run_id": run.name, "params": params, "sources": recorded, "missing_sources": []}
    (run / "manifest.json").write_text(json.dumps(manifest))
    none = tmp_path / "none"
    with pytest.raises(ms.SegmentError, match="sha256"):
        msc.run(pre, run, none, none, root, tmp_path / "data", exploratory=True)


@pytest.mark.parametrize("swap", ["bytes", "missing"])
def test_a_confirmatory_run_refuses_an_archive_changed_while_it_was_read(
    tmp_path: Path,
    frozen: tuple[prereg.Prereg, prereg.Prereg],
    monkeypatch: pytest.MonkeyPatch,
    swap: str,
) -> None:
    labels, pre = frozen
    root = tmp_path / "archive"
    recorded = _archive(root)
    sources = archive.sources([BTC, ETH], *SOURCE_SPAN)
    gone = sources[-1]
    (root / gone.path).unlink()
    run = _unread(tmp_path, labels, monkeypatch)
    manifest = json.loads((run / "manifest.json").read_text())
    kept = [r for r in recorded if r["url"] != gone.url]
    manifest |= {"sources": kept, "missing_sources": [gone.url]}
    (run / "manifest.json").write_text(json.dumps(manifest))
    real = archive.sources
    monkeypatch.setattr(archive, "sources", lambda symbols, *_: real(symbols, *SOURCE_SPAN))
    monkeypatch.setattr(prereg, "code_check", lambda *_: CLEAN_CODE)
    monkeypatch.setattr(msc.io, "code_version", lambda: {"git_sha": "0" * 40, "dirty": False})
    monkeypatch.setattr(ms, "check_labels", lambda *_: None)
    monkeypatch.setattr(msc, "cascade_onsets", lambda *_: ({}, None))

    def collect(*args: Any) -> None:
        if swap == "bytes":
            _changed(root, sources[0])
        else:
            (root / gone.path).write_bytes(b"late")

    def scored(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("scored a changed archive")

    monkeypatch.setattr(archive, "collect", collect)
    monkeypatch.setattr(msc, "score", scored)
    monkeypatch.setattr(msc, "evaluate", scored)
    data = tmp_path / "data"
    none = tmp_path / "none"
    with pytest.raises(prereg.PreregError, match="changed while it was read"):
        msc.run(pre, run, none, none, root, data)
    assert prereg.trials(data, pre) == 0
    assert not [p for p in (data / "ews").glob("*") if p.is_dir()]
