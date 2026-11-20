# The market JEPA's latent warning test's scorer (#30 M9): a frozen market-latent file applied as
# written to the confirmatory span beside the classic scorer's segments, or the same code run on the
# latents' fit span as an exploratory dry run that logs no trial. docs/market.md holds the rules.

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import __version__, engine
from ..ising import io
from ..market import archive, msvol, nulls
from . import margins, prereg, same_input, scoring
from . import market_score as msc
from . import market_segments as ms
from .surrogates import phase_randomised
from .trend import Pipeline

Row = dict[str, Any]
BTC, ETH = msc.BTC, msc.ETH
PREDICTION = (
    "Decisive at h = 5 and 6; h = 12 and 24 are scored, counted and reported, not decisive. A "
    "CI lies above a value when its lower end does. The CI is the percentile bootstrap interval"
    " of the scoring.bootstrap draws, and every percentile here, the shift's 95th, the combined"
    " margin's interval and the latent-minus-classic 97.5th included, uses numpy's linear "
    "interpolation (scoring.percentile_interval, np.nanquantile). The units read are the four "
    "judged cells, each holding the five primaries of its model's symbol, and, per model "
    "symbol, its five primaries' pooled groups, formed as cells.groups says. A primary's pooled"
    " AUC is that of both read symbols' events against both read symbols' controls, read by "
    "that checkpoint. After data.missing's exclusions, per horizon and read symbol, a segment "
    "whose trend score is not finite (a constant indicator series gives Kendall's tau no value)"
    " for any of the four pairs on any primary of either model, trained or twin, or for either "
    "classic indicator the pairs are compared with, is excluded for every pair, checkpoint and "
    "classic indicator and counted, so that every AUC, its bootstrap draws (which resample only"
    " the segments kept) and the classic side read the same segments; a unit left with no event"
    " or no control segment has no AUC and is inconclusive. The event-time shift at each "
    "horizon follows market-ews-v1's rule (ews.market_segments.shifted_rows, "
    "market.nulls.shift_onsets) on the paired segments: its source onsets are the event onsets "
    "kept in that horizon's event block after the baseline's exclusions, data.missing's and the"
    " finite-score exclusion above, in onset order, not every onset; in each of "
    "scoring.shift.draws draws on its own, each is moved to a bar drawn uniformly from the read"
    " symbol's eligible bars (Segments.eligible: the bars where the event rule could have "
    "placed an onset, inside the scored span) at least scoring.shift.min_offset_bars from it, "
    "from the stream [scoring] states; an onset with no such bar is left out of every draw and "
    "counted. A moved segment holding a minute with no value under data.missing or a minute "
    "with no return in the baseline's data.series, or reaching outside either series, or whose "
    "trend score is not finite for any of the checkpoints and classic indicators above, is "
    "dropped from that draw for every checkpoint and for the classic side, never redrawn. A "
    "draw's AUC is that of its moved segments' scores against the unit's control scores "
    "(ews.scoring.auc, which leaves NaN scores out); a draw with no scorable moved segment has "
    "no AUC, and a threshold taken over draws of which any has no AUC is NaN, so the unit is "
    "inconclusive, as ews.market_segments.shift_threshold rules. A pooled group's draw k joins "
    "both read symbols' draw k, each read symbol's onsets moved within its own eligible bars, "
    "and its threshold is the 95th percentile of the pooled |AUC - 0.5| over the draws. (1) "
    "Warns: in every unit, every primary's trained AUC 95 % CI excludes 0.5, all five on the "
    "same side of 0.5, and its |AUC - 0.5| lies above the 95th percentile of the same statistic"
    " over the event-time shift's draws, the two-sided counterpart, on a statistic folded about"
    " 0.5, of market-ews-v1's one-sided 97.5th percentile. (2) Learned: in every unit, the five"
    " primaries' trained-minus-twin margins, combined as scoring.twin.seeds says, pass under "
    "the tolerance rule, and every primary's margin carries the sign of the combined margin. "
    "(3) Beyond classic: in every unit, every primary's latent-minus-classic difference passes "
    "as scoring.classic.pairing says, against the classic indicator scoring.classic.against "
    "matches to the pair, and is itself above 0, so that a latent reading worse than the "
    "classic indicator never passes on a shift cutoff below 0. Its shift draws are the warns "
    "null's, the latent and classic sides moved together, and a draw on which either side has "
    "no AUC leaves its threshold NaN and the unit inconclusive. Tolerance rule, with "
    "scoring.twin.tolerance 0.02 in |AUC - 0.5|: a combined margin passes when its 95 % CI's "
    "lower end is above 0.02; it fails when its CI's upper end is below 0.02, which covers both"
    " no margin of that size and a twin that reads better; otherwise it is inconclusive. The "
    "pair warns only when (1) holds at both h = 5 and h = 6. (2) and (3) are read, and claimed,"
    " only for a pair that warns, and each holds only when it holds at both h = 5 and h = 6; "
    "for a pair that does not warn they are reported with no verdict. A unit that is "
    "inconclusive does not pass, and every unit's outcome is reported."
)
TWIN_SEEDS = (
    "a percentile bootstrap over the 5 primary model seeds, 2000 draws, nested with each "
    "checkpoint's data draws"
)
REPLICATES = 2000
BOOTSTRAP_STREAM, SHIFT_STREAM, SURROGATE_STREAM, TWIN_STREAM = 4, 5, 6, 7
WARN_Q, CLASSIC_Q = 0.95, 0.975
CHANCE = msc.CHANCE
DECISIVE_HORIZONS = msc.DECISIVE_HORIZONS
KINDS = ("trained", "twin")
GATE_FORMAT = 2
MINUTE_MS = ms.MINUTE_MS
NEWTON_STEPS, NEWTON_TOL, RIDGE = 50, 1e-10, 1e-9
SURROGATE_ALPHA = msc.SURROGATE_ALPHA
SRC = msc.SRC
HELD = (*msc.HELD, "ews/margins.py", "ews/fss.py", "ews/ising.py")
CLEAN = ("ews/market_latent_score.py", "ews/market_score.py", "ews/__main__.py", "ews/prereg.py")
EXPLORATORY = (
    "EXPLORATORY dry run on the latents' fit span, 2023-01-18..2023-09-30, the days the models "
    "were trained and validated on: its numbers check the code and are no evidence for or against "
    "any hypothesis; nothing is logged to ews/trials.jsonl"
)


@dataclass(frozen=True)
class Checkpoint:
    model: str
    seed: int
    kind: str
    run_id: str
    step: int

    @property
    def name(self) -> str:
        return f"{self.run_id}-step{self.step}"


def checkpoints(doc: Mapping[str, Any]) -> list[Checkpoint]:
    steps = doc["training"]["steps"]
    return [
        Checkpoint(e["symbol"], e["seed"], kind, e["run_id"], steps if kind == "trained" else 0)
        for e in doc["latents"]["checkpoints"]
        if e["role"] == "primary"
        for kind in KINDS
    ]


def pair_names(doc: Mapping[str, Any]) -> list[str]:
    by_pair = {(h["column"], h["indicator"]): h["name"] for h in doc["hypotheses"]}
    return [by_pair[column, indicator] for column, indicator in doc["indicators"]["pairs"]]


def latent_pipes(doc: Mapping[str, Any]) -> list[Pipeline]:
    ind = doc["indicators"]
    return [
        Pipeline(name, ind["window"], ind["length"], None, ind["lag"]) for _, name in ind["pairs"]
    ]


def classic_index(doc: Mapping[str, Any], baseline: Mapping[str, Any], p: int) -> int:
    indicator = doc["indicators"]["pairs"][p][1]
    return list(baseline["indicators"]["names"]).index(prereg.CLASSIC_MATCHING[indicator])


def classic_sign(baseline: Mapping[str, Any], name: str) -> float:
    direction = next(h["direction"] for h in baseline["hypotheses"] if h["indicator"] == name)
    return 1.0 if direction == "greater" else -1.0


def check_file(pre: prereg.Prereg) -> prereg.Prereg:
    doc = pre.doc
    if doc["data"]["domain"] != prereg.MARKET_LATENT:
        raise prereg.PreregError(f"{pre.path} is not a {prereg.MARKET_LATENT} file")
    found = prereg.problems(doc, strict=True)
    if found:
        raise prereg.PreregError(f"{pre.path}:\n  " + "\n  ".join(found))
    for h in doc["hypotheses"]:
        if h["prediction"] != PREDICTION:
            raise prereg.PreregError(
                f"hypotheses {h['name']}: its prediction is not the decision rule this scorer "
                "carries out (market_latent_score.PREDICTION)"
            )
    sc = doc["scoring"]
    fixed = (
        (sc["twin"]["seeds"] == TWIN_SEEDS, f"scoring.twin.seeds: {TWIN_SEEDS!r}"),
        (doc["cells"]["judged"] == "all", "cells.judged: 'all'"),
        (sc["classic"]["against"] == "matching", "scoring.classic.against: 'matching'"),
        (not doc["descriptive"]["ablations"], "descriptive.ablations: none"),
        (list(doc["data"]["symbols"]) == [BTC, ETH], f"data.symbols: {[BTC, ETH]}"),
        (
            set(DECISIVE_HORIZONS) <= set(doc["segments"]["horizons"]),
            f"segments.horizons: the decisive {list(DECISIVE_HORIZONS)}",
        ),
        (
            sorted((h["column"], h["indicator"]) for h in doc["hypotheses"])
            == sorted(tuple(p) for p in doc["indicators"]["pairs"]),
            "one hypothesis per pair of indicators.pairs",
        ),
    )
    for ok, need in fixed:
        if not ok:
            raise prereg.PreregError(f"{pre.name}: this scorer carries out {need} only")
    named = doc["baseline"]
    path = prereg.repo_root(pre.path) / prereg.PREREG_DIR / f"{named['tag'][7:]}.toml"
    baseline = prereg.load(path)
    ref = prereg.frozen(baseline)
    if (ref.tag, baseline.sha256) != (named["tag"], named["sha256"]):
        raise prereg.PreregError(
            f"{pre.name} is scored on {named['tag']} (sha256 {named['sha256']}); {path} is "
            f"{ref.tag} (sha256 {baseline.sha256})"
        )
    msc.check_file(baseline)
    return baseline


def stamp_of(record: Mapping[str, Any]) -> Row:
    keys = ("id", "version", "sha256", "tag", "tag_object", "commit")
    return {
        **{k: record["prereg"][k] for k in keys},
        "head": record["code"]["head"],
        "pins_sha256": record["pins_sha256"],
    }


def _listed(doc: Mapping[str, Any]) -> list[Row]:
    return sorted((dict(row) for row in doc["latents"]["checkpoints"]), key=lambda r: r["run_id"])


def check_gate(
    pre: prereg.Prereg, ref: prereg.Frozen, record: Any, pins_raw: bytes
) -> Mapping[str, Any]:
    if not isinstance(record, dict) or record.get("format") != GATE_FORMAT:
        raise prereg.PreregError(f"not an extraction gate record of format {GATE_FORMAT}")
    try:
        stamp, code = record["prereg"], record["code"]
        text = stamp["text"].encode("utf-8")
        root = prereg.repo_root(pre.path)
        want = {
            "id": pre.id,
            "version": pre.version,
            "sha256": pre.sha256,
            "tag": ref.tag,
            "commit": ref.commit,
            "tag_object": prereg._git(root, "rev-parse", ref.tag),
        }
        differ = sorted(k for k, v in want.items() if stamp[k] != v)
        if text != pre.path.read_bytes() or hashlib.sha256(text).hexdigest() != pre.sha256:
            differ.append("text")
        if differ:
            raise prereg.PreregError(
                f"the gate record was made from another file than {pre.name} at {ref.tag}: "
                + ", ".join(differ)
            )
        if code["changed"] or code["uncommitted"]:
            raise prereg.PreregError("the gate record names changed or uncommitted code")
        if record["pins_sha256"] != hashlib.sha256(pins_raw).hexdigest():
            raise prereg.PreregError("pins.json is not the one the gate record was made with")
        listed = _listed(pre.doc)
        if record["checkpoints"] != listed:
            raise prereg.PreregError(f"the gate record's checkpoints are not {pre.name}'s")
        pins = json.loads(pins_raw)
        keys = sorted({k for row in listed for k in row})
        pinned = sorted(
            ({k: p.get(k) for k in keys} for p in pins["checkpoints"]),
            key=lambda r: str(r["run_id"]),
        )
        if pinned != listed:
            raise prereg.PreregError(f"pins.json and {pre.name}'s checkpoints differ")
        if pins["steps"] != pre.doc["training"]["steps"]:
            raise prereg.PreregError(f"pins.json pins step {pins['steps']}, not training.steps")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, prereg.PreregError):
            raise
        raise prereg.PreregError(f"not a whole gate record or pins.json: {exc!r}") from exc
    return pins


def extent(doc: Mapping[str, Any], labels: prereg.Prereg, span: str) -> tuple[int, int, int]:
    if span == "fit":
        fit = doc["seasonality"]["fit"]
        return archive.day_ms(fit["start"]), ((fit["end"] - fit["start"]).days + 1) * 1440, 0
    if span != "confirmatory":
        raise prereg.PreregError(f"no latent span {span!r}")
    ind = doc["indicators"]
    before = 5 * max(doc["segments"]["horizons"]) + ind["window"] + ind["length"] - 1
    before += doc["latents"]["context"] - 1
    end = labels.doc["data"]["span"]["end"] + timedelta(days=1)
    start = archive.day_ms(prereg.CONFIRMATORY) - before * MINUTE_MS
    return start, (end - prereg.CONFIRMATORY).days * 1440 + before, before


class Tables(Protocol):
    def read(self, checkpoint: Checkpoint, symbol: str, span: str) -> dict[str, ms.Minutes]: ...


class Merged:
    def __init__(
        self,
        root: Path,
        record: Mapping[str, Any],
        pre: prereg.Prereg,
        labels: prereg.Prereg,
        spans: Sequence[str],
    ) -> None:
        self.root, self.spans, self.doc, self.labels = Path(root), tuple(spans), pre.doc, labels
        self.columns = list(pre.doc["latents"]["columns"])
        for span in self.spans:
            start, minutes, _ = extent(self.doc, labels, span)
            if span != "confirmatory" and start + minutes * MINUTE_MS > ms.CONFIRMATORY_MS:
                raise ms.SegmentError(f"the {span} span reaches {prereg.CONFIRMATORY}")
        entries = {e["run_id"]: e for e in pre.doc["latents"]["checkpoints"]}
        scaling = {e["symbol"]: e["scaling_sha256"] for e in pre.doc["latents"]["checkpoints"]}
        stamp = stamp_of(record)
        made_by = {"git_sha": record["code"]["head"], "dirty": False}
        self.manifests: dict[tuple[str, str, str], Row] = {}
        for cp in checkpoints(pre.doc):
            entry = entries[cp.run_id]
            for symbol in pre.doc["data"]["symbols"]:
                for span in self.spans:
                    path = self._path(cp, symbol, span).with_suffix(".json")
                    try:
                        m = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        raise prereg.PreregError(f"{path}: {exc}") from exc
                    _, minutes, before = extent(self.doc, labels, span)
                    c = m.get("checkpoint") or {}
                    checks = (
                        (m.get("pre_registered") is True, "not made under a gate record"),
                        (m.get("gate") == stamp, "made under another gate record"),
                        (m.get("code") == made_by, "not made by the gate's commit, clean"),
                        (
                            (c.get("run_id"), c.get("kind"), c.get("step"))
                            == (cp.run_id, cp.kind, cp.step),
                            "another checkpoint's table",
                        ),
                        (c.get("sha256") == entry[f"{cp.kind}_sha256"], "checkpoint not pinned"),
                        (
                            (c.get("symbol"), c.get("seed"), c.get("role"))
                            == (cp.model, cp.seed, "primary"),
                            "another model's table",
                        ),
                        (
                            cp.kind == "twin" or c.get("pca_sha256") == entry["pca_sha256"],
                            "PCA basis not pinned",
                        ),
                        (m.get("scaling_sha256") == scaling[symbol], "another symbol's scaling"),
                        ((m.get("symbol"), m.get("span")) == (symbol, span), "another read"),
                        (
                            (m.get("pre_minutes"), m.get("minutes")) == (before, minutes),
                            "not the span's minutes",
                        ),
                        (m.get("parquet") == path.with_suffix(".parquet").name, "another file"),
                    )
                    why = next((why for ok, why in checks if not ok), None)
                    if why:
                        raise prereg.PreregError(f"{path}: {why}")
                    self.manifests[cp.name, symbol, span] = m

    def _path(self, cp: Checkpoint, symbol: str, span: str) -> Path:
        return self.root / span / cp.name / f"{symbol}-{span}.parquet"

    def read(self, checkpoint: Checkpoint, symbol: str, span: str) -> dict[str, ms.Minutes]:
        if span not in self.spans:
            raise ms.SegmentError(f"the {span} latents are not open to this run")
        manifest = self.manifests[checkpoint.name, symbol, span]
        path = self._path(checkpoint, symbol, span)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["parquet_sha256"]:
            raise prereg.PreregError(f"{path}: not the file its manifest names")
        table = pq.read_table(pa.BufferReader(raw), columns=["ts_ns", "valid", *self.columns])
        start, minutes, _ = extent(self.doc, self.labels, span)
        grid = start * 1_000_000 + 60_000_000_000 * np.arange(minutes, dtype=np.int64)
        if not np.array_equal(table["ts_ns"].to_numpy(), grid):
            raise prereg.PreregError(f"{path}: its minutes are not the {span} span's")
        valid = table["valid"].to_numpy(zero_copy_only=False)
        return {
            c: ms.Minutes(start, np.where(valid, table[c].to_numpy(zero_copy_only=False), np.nan))
            for c in self.columns
        }


def latent_series(
    tables: Tables, cp: Checkpoint, symbol: str, doc: Mapping[str, Any], span: str
) -> dict[str, ms.Minutes]:
    season = doc["seasonality"]
    fit, method = (season["fit"]["start"], season["fit"]["end"]), season["method"]
    raw_fit = tables.read(cp, symbol, "fit")
    if any(m.end_ms > ms.CONFIRMATORY_MS for m in raw_fit.values()):
        raise ms.SegmentError(f"{cp.name} {symbol}: its fit latents reach {prereg.CONFIRMATORY}")
    scored = raw_fit if span == "fit" else tables.read(cp, symbol, span)
    return {
        c: ms.deseasonalise(scored[c], ms.slot_profile(raw_fit[c], fit, method), method)
        for c in doc["latents"]["columns"]
    }


def on_grid(series: ms.Minutes, start_ms: int, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    offset = (series.start_ms - start_ms) // MINUTE_MS
    lo, hi = max(offset, 0), min(offset + len(series.values), n)
    if hi > lo:
        out[lo:hi] = series.values[lo - offset : hi - offset]
    return out


def rows_at(series: ms.Minutes, first_ms: np.ndarray, span: int) -> np.ndarray:
    first_ms = np.asarray(first_ms, np.int64)
    start = (first_ms - series.start_ms) // MINUTE_MS
    inside = (start >= 0) & (start + span <= len(series.values))
    out = np.full((*first_ms.shape, span), np.nan)
    out[inside] = series.values[start[inside, None] + np.arange(span)]
    return out


def score_rows(pipe: Pipeline, rows: np.ndarray) -> np.ndarray:
    lead, span = rows.shape[:-1], rows.shape[-1]
    flat = rows.reshape(-1, span)
    ok = np.isfinite(flat).all(axis=1)
    out = np.full(len(flat), np.nan)
    if ok.any():
        out[ok] = pipe(flat[ok])
    return out.reshape(lead)


def counts(sizes_ev: Sequence[int], sizes_ct: Sequence[int], rng: np.random.Generator, draws: int):
    wp = np.zeros((draws, sum(sizes_ev)))
    wn = np.zeros((draws, sum(sizes_ct)))
    off_ev = np.cumsum([0, *sizes_ev])
    off_ct = np.cumsum([0, *sizes_ct])
    for d in range(draws):
        for k, n in enumerate(sizes_ev):
            wp[d, off_ev[k] : off_ev[k + 1]] = np.bincount(rng.integers(0, n, n), minlength=n)
        for k, n in enumerate(sizes_ct):
            wn[d, off_ct[k] : off_ct[k + 1]] = np.bincount(rng.integers(0, n, n), minlength=n)
    return wp, wn


def counted_aucs(pos: np.ndarray, neg: np.ndarray, wp: np.ndarray, wn: np.ndarray) -> np.ndarray:
    if not len(pos) or not len(neg):
        return np.full(len(wp), np.nan)
    c = 2.0 * (pos[:, None] > neg[None, :]) + (pos[:, None] == neg[None, :])
    total = ((wp @ c) * wn).sum(axis=1)
    return total / (2.0 * len(pos) * len(neg))


def folded_threshold(aucs: np.ndarray) -> float:
    a = np.asarray(aucs, np.float64)
    if not a.size or np.isnan(a).any():
        return math.nan
    return float(np.nanquantile(np.abs(a - CHANCE), WARN_Q))


def difference_threshold(differences: np.ndarray) -> float:
    d = np.asarray(differences, np.float64)
    if not d.size or np.isnan(d).any():
        return math.nan
    return float(np.nanquantile(d, CLASSIC_Q))


def _nan(*values: Any) -> bool:
    return any(np.isnan(np.asarray(v, np.float64)).any() for v in values)


def warn_verdict(
    aucs: Sequence[float], los: Sequence[float], his: Sequence[float], thresholds: Sequence[float]
) -> str:
    if _nan(aucs, los, his, thresholds):
        return "inconclusive"
    sides = {
        1 if lo > CHANCE else -1 if hi < CHANCE else 0 for lo, hi in zip(los, his, strict=True)
    }
    above = all(abs(a - CHANCE) > t for a, t in zip(aucs, thresholds, strict=True))
    return "pass" if sides in ({1}, {-1}) and above else "fail"


def learned_verdict(
    margin: float, lo: float, hi: float, seeds: Sequence[float], aucs: Sequence[float], tol: float
) -> tuple[str, bool | None, bool | None]:
    if _nan(margin, lo, hi, seeds, aucs):
        return "inconclusive", None, None
    signs = margin != 0 and all(np.sign(m) == np.sign(margin) for m in seeds)
    same = {float(np.sign(a - CHANCE)) for a in aucs} in ({1.0}, {-1.0})
    ci = "pass" if lo > tol else "fail" if hi < tol else "inconclusive"
    if ci == "fail" or not signs or not same:
        return "fail", bool(signs), bool(same)
    return ci, True, True


def beyond_verdict(differences: Sequence[float], thresholds: Sequence[float]) -> str:
    if _nan(differences, thresholds):
        return "inconclusive"
    ok = all(d > t and d > 0 for d, t in zip(differences, thresholds, strict=True))
    return "pass" if ok else "fail"


def pair_verdict(warns: str, learned: str, beyond: str) -> str:
    if warns != "pass":
        return warns
    if learned == beyond == "pass":
        return "pass"
    return "fail" if "fail" in (learned, beyond) else "inconclusive"


def two_sided_pvalues(
    x: np.ndarray, pipe: Pipeline, n: int, keys: Sequence[Sequence[int]]
) -> np.ndarray:
    out = np.empty(len(x))
    if not len(x):
        return out
    residuals = pipe.residuals(x)
    step = max(1, scoring.SURROGATE_BYTES // (8 * (n + 1) * x.shape[-1]))
    for start in range(0, len(x), step):
        own = residuals[start : start + step]
        drawn = [
            phase_randomised(own[k : k + 1], n, np.random.default_rng(keys[start + k]))
            for k in range(len(own))
        ]
        batch = np.concatenate([own[:, None, :], np.concatenate(drawn)], axis=1)
        scores = pipe.trend(batch)
        greater = scoring.surrogate_pvalue(scores[:, 0], scores[:, 1:])
        less = scoring.surrogate_pvalue(-scores[:, 0], -scores[:, 1:])
        out[start : start + step] = np.minimum(1.0, 2 * np.minimum(greater, less))
    return out


def weighted_auc(eta: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    b, n = eta.shape
    order = np.argsort(eta, axis=1, kind="stable")
    s = np.take_along_axis(eta, order, 1)
    wy = np.take_along_axis(w * y, order, 1)
    wn = np.take_along_axis(w * (1 - y), order, 1)
    new = np.ones((b, n), bool)
    new[:, 1:] = s[:, 1:] != s[:, :-1]
    group = np.cumsum(new, axis=1) - 1 + (n * np.arange(b))[:, None]
    neg = np.bincount(group.ravel(), wn.ravel(), minlength=b * n).reshape(b, n)
    pos = np.bincount(group.ravel(), wy.ravel(), minlength=b * n).reshape(b, n)
    below = np.cumsum(neg, axis=1) - neg
    with np.errstate(invalid="ignore", divide="ignore"):
        return (pos * (below + 0.5 * neg)).sum(axis=1) / (wy.sum(axis=1) * wn.sum(axis=1))


def logistic_auc(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0)
    z = np.where(sd > 0, (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)
    design = np.column_stack([np.ones(len(z)), z])
    k = design.shape[1]
    beta = np.zeros((len(w), k))
    for _ in range(NEWTON_STEPS):
        mu = 1.0 / (1.0 + np.exp(-np.clip(beta @ design.T, -30, 30)))
        grad = (w * (y - mu)) @ design
        hess = ((w * mu * (1 - mu))[:, :, None] * design).transpose(0, 2, 1) @ design
        step = np.linalg.solve(hess + RIDGE * np.eye(k), grad[:, :, None])[:, :, 0]
        beta += step
        if np.abs(step).max() < NEWTON_TOL:
            break
    return weighted_auc(beta @ design.T, y, w)


@dataclass(frozen=True)
class Evaluation:
    tables: dict[str, list[Row]]
    verdicts: dict[str, str]
    segments: ms.Segments
    series: dict[str, ms.Minutes]
    kept: dict[str, dict[Any, np.ndarray]]


def _alone_sides(group: str, symbols: Sequence[str]) -> list[str]:
    if group == "eth-only":
        return [ETH]
    return list(symbols) if group == "pooled" else [group]


def _group_events(
    values: Mapping[str, np.ndarray], group: str, symbols: Sequence[str], alone: np.ndarray
) -> list[np.ndarray]:
    if group == "eth-only":
        return [values[ETH][alone]]
    return [values[s] for s in _alone_sides(group, symbols)]


def evaluate(
    pre: prereg.Prereg,
    baseline: prereg.Prereg,
    labels: prereg.Prereg,
    run: Path,
    prices: Mapping[str, ms.Minutes],
    cascades: Mapping[str, np.ndarray],
    book: Mapping[str, Mapping[str, ms.Minutes]],
    tables: Tables,
    *,
    confirmatory: bool,
) -> Evaluation:
    doc, base = pre.doc, baseline.doc
    symbols = list(doc["data"]["symbols"])
    horizons = list(doc["segments"]["horizons"])
    sc, seed = doc["scoring"], doc["scoring"]["seed"]
    season = doc["seasonality"]
    fit = (season["fit"]["start"], season["fit"]["end"])
    method = season["method"]
    names, pipes = pair_names(doc), latent_pipes(doc)
    columns = [column for column, _ in doc["indicators"]["pairs"]]
    classic = msc.pipelines(base)
    classic_names = list(classic)
    span = doc["indicators"]["window"] + doc["indicators"]["length"] - 1
    cps = checkpoints(doc)
    bars = doc["reporting"]["eth_only_bars"]
    latent_span = "confirmatory" if confirmatory else "fit"
    groups = ["pooled", *symbols, "eth-only"]
    seg = doc["segments"]
    cut: dict[str, Any] = {"subsample": seg["subsample"], "seed": seg["seed"]}
    if confirmatory:
        cut["scoring"] = baseline
        first_ms, end_ms = ms.CONFIRMATORY_MS, None
    else:
        cut["score_from"] = fit[0]
        first_ms, end_ms = archive.day_ms(fit[0]), archive.day_ms(fit[1] + timedelta(days=1))

    if not confirmatory:
        late = [s for s, m in prices.items() if m.end_ms > ms.CONFIRMATORY_MS]
        late += [
            f"{s} cascades"
            for s, t in cascades.items()
            if len(t) and int(t.max()) // 1_000_000 >= ms.CONFIRMATORY_MS
        ]
        late += [
            f"{s} {name}"
            for s, features in book.items()
            for name, m in features.items()
            if m.end_ms > ms.CONFIRMATORY_MS
        ]
        if late:
            raise ms.SegmentError(
                f"a dry run reads no day from {prereg.CONFIRMATORY}: {', '.join(late)}"
            )

    def truncated(m: ms.Minutes) -> ms.Minutes:
        if end_ms is None:
            return m
        return ms.Minutes(m.start_ms, m.values[: max(0, (end_ms - m.start_ms) // MINUTE_MS)])

    price = {
        s: truncated(ms.deseasonalise(r, ms.slot_profile(r, fit, method), method))
        for s, r in prices.items()
    }
    raw_returns = {s: truncated(r) for s, r in prices.items()}
    keep = {s: np.ones(len(price[s].values), bool) for s in symbols}
    for cp in cps:
        for s in symbols:
            for x in latent_series(tables, cp, s, doc, latent_span).values():
                keep[s] &= np.isfinite(on_grid(x, price[s].start_ms, len(keep[s])))
    masked = {s: msc.masked(price[s], keep[s]) for s in symbols}

    def build(series: Mapping[str, ms.Minutes]) -> ms.Segments:
        segs = ms.build(run, labels, series, horizons, span, **cut)
        if end_ms is None:
            return segs
        eligible = {s: e[e // 1_000_000 < end_ms] for s, e in segs.eligible.items()}
        return replace(segs, eligible=eligible)

    base_segs = build(price)
    segs = build(masked)
    cascade_first: dict[tuple[str, int], np.ndarray] = {}
    cascade_onsets: dict[tuple[str, int], np.ndarray] = {}
    cascade_counts: list[Row] = []
    for s in symbols:
        mine = cascades[s][cascades[s] // 1_000_000 >= first_ms]
        if end_ms is not None:
            mine = mine[mine // 1_000_000 < end_ms]
        for h in horizons:
            first = mine // 1_000_000 - (ms.BAR_MINUTES * h + span) * MINUTE_MS
            rows, kept, _, outside, invalid = ms._cut(masked[s], first, span, np.zeros(0, bool), 0)
            cascade_first[s, h], cascade_onsets[s, h] = first[kept], mine[kept]
            cascade_counts.append(
                {"series": "cascade", "symbol": s, "kind": "cascade", "horizon_bars": h}
                | {"cut": len(first), "kept": len(rows), "excluded_invalid": invalid}
                | {"excluded_outside": outside}
            )

    ev_first = {(s, h): segs.events[s, h].first_ms for s in symbols for h in horizons}
    ct_first = {s: segs.controls[s].first_ms for s in symbols}
    lat_ev: dict[tuple[int, int, str, int], np.ndarray] = {}
    lat_ct: dict[tuple[int, int, str], np.ndarray] = {}
    lat_cas: dict[tuple[int, int, str, int], np.ndarray] = {}
    for ci, cp in enumerate(cps):
        for s in symbols:
            series = latent_series(tables, cp, s, doc, latent_span)
            parts = [ct_first[s], *(ev_first[s, h] for h in horizons)]
            parts += [cascade_first[s, h] for h in horizons]
            bounds = np.cumsum([len(v) for v in parts])[:-1]
            firsts = np.concatenate(parts)
            for p, pipe in enumerate(pipes):
                scored = score_rows(pipe, rows_at(series[columns[p]], firsts, span))
                split = np.split(scored, bounds)
                lat_ct[ci, p, s] = split[0]
                for k, h in enumerate(horizons):
                    lat_ev[ci, p, s, h] = split[1 + k]
                    lat_cas[ci, p, s, h] = split[1 + len(horizons) + k]
    cls_ev = {
        (i, s, h): score_rows(pipe, segs.events[s, h].x)
        for i, pipe in enumerate(classic.values())
        for s in symbols
        for h in horizons
    }
    cls_ct = {
        (i, s): score_rows(pipe, segs.controls[s].x)
        for i, pipe in enumerate(classic.values())
        for s in symbols
    }
    cls_cas = {
        (i, s, h): score_rows(pipe, rows_at(masked[s], cascade_first[s, h], span))
        for i, pipe in enumerate(classic.values())
        for s in symbols
        for h in horizons
    }

    def finite(parts: Sequence[np.ndarray], n: int) -> np.ndarray:
        keep = np.ones(n, bool)
        for v in parts:
            keep &= np.isfinite(v)
        return keep

    ev_keep = {
        (s, h): finite(
            [lat_ev[ci, p, s, h] for ci in range(len(cps)) for p in range(len(pipes))]
            + [cls_ev[i, s, h] for i in range(len(classic))],
            len(ev_first[s, h]),
        )
        for s in symbols
        for h in horizons
    }
    ct_keep = {
        s: finite(
            [lat_ct[ci, p, s] for ci in range(len(cps)) for p in range(len(pipes))]
            + [cls_ct[i, s] for i in range(len(classic))],
            len(ct_first[s]),
        )
        for s in symbols
    }
    cas_keep = {
        (s, h): finite(
            [lat_cas[ci, p, s, h] for ci in range(len(cps)) for p in range(len(pipes))]
            + [cls_cas[i, s, h] for i in range(len(classic))],
            len(cascade_first[s, h]),
        )
        for s in symbols
        for h in horizons
    }
    lat_ev = {k: v[ev_keep[k[2], k[3]]] for k, v in lat_ev.items()}
    lat_ct = {k: v[ct_keep[k[2]]] for k, v in lat_ct.items()}
    lat_cas = {k: v[cas_keep[k[2], k[3]]] for k, v in lat_cas.items()}
    cls_ev = {k: v[ev_keep[k[1], k[2]]] for k, v in cls_ev.items()}
    cls_ct = {k: v[ct_keep[k[1]]] for k, v in cls_ct.items()}
    cls_cas = {k: v[cas_keep[k[1], k[2]]] for k, v in cls_cas.items()}
    anchors = {
        (s, h): segs.events[s, h].anchor_ns[ev_keep[s, h]] for s in symbols for h in horizons
    }
    ev_first = {k: v[ev_keep[k]] for k, v in ev_first.items()}
    ct_first = {s: v[ct_keep[s]] for s, v in ct_first.items()}
    cascade_first = {k: v[cas_keep[k]] for k, v in cascade_first.items()}
    cascade_onsets = {k: v[cas_keep[k]] for k, v in cascade_onsets.items()}
    alone = {h: ms.eth_only(anchors[ETH, h], segs.onsets[BTC], bars) for h in horizons}
    cascade_alone = {h: ms.eth_only(cascade_onsets[ETH, h], cascades[BTC], bars) for h in horizons}

    shift = sc["shift"]
    moved_first: dict[tuple[str, int], np.ndarray] = {}
    moved_ok: dict[tuple[str, int], np.ndarray] = {}
    sources: dict[tuple[str, int], np.ndarray] = {}
    moved_alone: dict[int, np.ndarray] = {}
    for s in symbols:
        for h in horizons:
            rng = np.random.default_rng([seed, SHIFT_STREAM, h, ms.group_code(symbols, s)])
            moved, sources[s, h] = nulls.shift_onsets(
                anchors[s, h],
                segs.eligible[s],
                shift["min_offset_bars"] * msvol.BAR_NS,
                shift["draws"],
                rng,
            )
            first = moved // 1_000_000 - (ms.BAR_MINUTES * h + span) * MINUTE_MS
            moved_first[s, h] = first
            moved_ok[s, h] = np.isfinite(rows_at(masked[s], first, span)).all(axis=-1)
            if s == ETH:
                moved_alone[h] = ms.eth_only(moved.ravel(), segs.onsets[BTC], bars).reshape(
                    moved.shape
                )
    lat_sh: dict[tuple[int, int, str, int], np.ndarray] = {}
    for ci, cp in enumerate(cps):
        for s in symbols:
            series = latent_series(tables, cp, s, doc, latent_span)
            for h in horizons:
                ok = moved_ok[s, h]
                for p, pipe in enumerate(pipes):
                    out = np.full(ok.shape, np.nan)
                    rows = rows_at(series[columns[p]], moved_first[s, h][ok], span)
                    out[ok] = score_rows(pipe, rows)
                    lat_sh[ci, p, s, h] = out
    cls_sh: dict[tuple[int, str, int], np.ndarray] = {}
    for i, pipe in enumerate(classic.values()):
        for s in symbols:
            for h in horizons:
                ok = moved_ok[s, h]
                out = np.full(ok.shape, np.nan)
                out[ok] = score_rows(pipe, rows_at(masked[s], moved_first[s, h][ok], span))
                cls_sh[i, s, h] = out
    for s in symbols:
        for h in horizons:
            parts = [lat_sh[ci, p, s, h] for ci in range(len(cps)) for p in range(len(pipes))]
            parts += [cls_sh[i, s, h] for i in range(len(classic))]
            moved_ok[s, h] = moved_ok[s, h] & np.logical_and.reduce([np.isfinite(v) for v in parts])
    lat_sh = {k: np.where(moved_ok[k[2], k[3]], v, np.nan) for k, v in lat_sh.items()}
    cls_sh = {k: np.where(moved_ok[k[1], k[2]], v, np.nan) for k, v in cls_sh.items()}

    def shift_positives(values: Mapping[str, np.ndarray], group: str, h: int) -> np.ndarray:
        if group == "eth-only":
            cols = alone[h][sources[ETH, h]]
            return np.where(moved_alone[h][:, cols], values[ETH][:, cols], np.nan)
        return np.concatenate([values[s] for s in _alone_sides(group, symbols)], axis=1)

    def shift_counts(group: str, h: int) -> Row:
        if group == "eth-only":
            cols = alone[h][sources[ETH, h]]
            ok = moved_ok[ETH, h][:, cols] & moved_alone[h][:, cols]
            near = int((~moved_alone[h][:, cols]).sum())
            unshifted = int((~sources[ETH, h])[alone[h]].sum())
        else:
            chosen = _alone_sides(group, symbols)
            ok = np.concatenate([moved_ok[s, h] for s in chosen], axis=1)
            near = 0
            unshifted = sum(int((~sources[s, h]).sum()) for s in chosen)
        return {
            "shift_draws": len(ok),
            "shifted_onsets": ok.shape[1],
            "unshifted_onsets": unshifted,
            "shift_dropped": int((~ok).sum()) - near,
            "shift_dropped_near_btc": near,
        }

    tol = float(sc["twin"]["tolerance"])
    level = float(sc["ci_level"])
    draws = int(sc["bootstrap"])
    cells: list[Row] = []
    units: list[Row] = []
    margin_rows: list[Row] = []
    beyond_rows: list[Row] = []
    shift_rows: list[Row] = []
    incremental: list[Row] = []
    cascade_rows: list[Row] = []
    for p, name in enumerate(names):
        i = classic_index(doc, base, p)
        orient = classic_sign(base, classic_names[i])
        column, indicator = doc["indicators"]["pairs"][p]
        for h in horizons:
            for mi, model in enumerate(symbols):
                mine = [ci for ci, cp in enumerate(cps) if cp.model == model]
                seeds = sorted({cps[ci].seed for ci in mine})
                by = {(cps[ci].seed, cps[ci].kind): ci for ci in mine}
                for group in groups:
                    g = ms.group_code(symbols, group)
                    decisive = h in DECISIVE_HORIZONS and group != "eth-only"
                    head = {"pair": name, "horizon_bars": h, "model": model, "group": group}

                    read = _alone_sides(group, symbols)
                    events_of = {
                        ci: _group_events(
                            {s: lat_ev[ci, p, s, h] for s in symbols}, group, symbols, alone[h]
                        )
                        for ci in mine
                    }
                    controls_of = {ci: [lat_ct[ci, p, s] for s in read] for ci in mine}
                    cascades_of = {
                        ci: _group_events(
                            {s: lat_cas[ci, p, s, h] for s in symbols},
                            group,
                            symbols,
                            cascade_alone[h],
                        )
                        for ci in mine
                    }
                    cls_pos = _group_events(
                        {s: cls_ev[i, s, h] for s in symbols}, group, symbols, alone[h]
                    )
                    cls_neg = [cls_ct[i, s] for s in _alone_sides(group, symbols)]
                    sizes_ev = [len(v) for v in cls_pos]
                    sizes_ct = [len(v) for v in cls_neg]
                    rng4 = np.random.default_rng([seed, BOOTSTRAP_STREAM, p, h, mi, g])
                    wp, wn = counts(sizes_ev, sizes_ct, rng4, draws)
                    c_pos, c_neg = np.concatenate(cls_pos), np.concatenate(cls_neg)
                    c_auc = scoring.auc(c_pos, c_neg)
                    c_draws = counted_aucs(c_pos, c_neg, wp, wn)
                    c_shift = np.array(
                        [
                            scoring.auc(row, c_neg)
                            for row in shift_positives(
                                {s: cls_sh[i, s, h] for s in symbols}, group, h
                            )
                        ]
                    )
                    moved = shift_counts(group, h)
                    found: dict[tuple[int, str], Row] = {}
                    boot: dict[tuple[int, str], np.ndarray] = {}
                    for s_ in seeds:
                        for kind in KINDS:
                            ci = by[s_, kind]
                            pos, neg = (
                                np.concatenate(events_of[ci]),
                                np.concatenate(controls_of[ci]),
                            )
                            auc = scoring.auc(pos, neg)
                            boot[s_, kind] = counted_aucs(pos, neg, wp, wn)
                            lo, hi = scoring.percentile_interval(boot[s_, kind], level)
                            sh = np.array(
                                [
                                    scoring.auc(row, neg)
                                    for row in shift_positives(
                                        {s: lat_sh[ci, p, s, h] for s in symbols}, group, h
                                    )
                                ]
                            )
                            q95 = folded_threshold(sh)
                            for k, a in enumerate(sh):
                                shift_rows.append(
                                    head | {"seed": s_, "checkpoint": kind, "draw": k, "auc": a}
                                )
                            found[s_, kind] = {
                                "auc": auc,
                                "lo": lo,
                                "hi": hi,
                                "q95": q95,
                                "shift": sh,
                                "pos": pos,
                                "neg": neg,
                            }
                            cells.append(
                                head
                                | {"column": column, "indicator": indicator, "seed": s_}
                                | {"checkpoint": kind, "decisive": decisive}
                                | {"auc": auc, "auc_lo": lo, "auc_hi": hi}
                                | {"n_pos": len(pos), "n_neg": len(neg)}
                                | {"shift_q95": q95}
                                | moved
                                | {
                                    "ci_excludes_chance": bool(lo > CHANCE or hi < CHANCE),
                                    "above_shift": bool(abs(auc - CHANCE) > q95),
                                }
                            )
                    trained = [found[s_, "trained"] for s_ in seeds]
                    warns = warn_verdict(
                        [f["auc"] for f in trained],
                        [f["lo"] for f in trained],
                        [f["hi"] for f in trained],
                        [f["q95"] for f in trained],
                    )
                    rng7 = np.random.default_rng([seed, TWIN_STREAM, p, h, mi, g])
                    twin = [found[s_, "twin"] for s_ in seeds]
                    pairing = margins.paired(
                        [abs(f["auc"] - CHANCE) for f in trained],
                        [abs(f["auc"] - CHANCE) for f in twin],
                        "greater",
                        tol,
                        rng7,
                        np.abs(np.stack([boot[s_, "trained"] for s_ in seeds]) - CHANCE),
                        np.abs(np.stack([boot[s_, "twin"] for s_ in seeds]) - CHANCE),
                        n=REPLICATES,
                        level=level,
                    )
                    learned, signs, same = learned_verdict(
                        pairing["margin"],
                        pairing["lo"],
                        pairing["hi"],
                        pairing["margins"],
                        [f["auc"] for f in trained],
                        tol,
                    )
                    margin_rows.append(
                        head
                        | {"seeds": seeds, "margins": pairing["margins"]}
                        | {"margin": pairing["margin"], "margin_lo": pairing["lo"]}
                        | {"margin_hi": pairing["hi"], "tolerance": tol, "sign_test": signs}
                        | {"same_sign": same, "replicates": REPLICATES, "outcome": learned}
                    )
                    diffs, thresholds = [], []
                    for s_, f in zip(seeds, trained, strict=True):
                        diff = abs(f["auc"] - CHANCE) - orient * (c_auc - CHANCE)
                        null = np.abs(f["shift"] - CHANCE) - orient * (c_shift - CHANCE)
                        q975 = difference_threshold(null)
                        d_lo, d_hi = scoring.percentile_interval(
                            np.abs(boot[s_, "trained"] - CHANCE) - orient * (c_draws - CHANCE),
                            level,
                        )
                        diffs.append(diff)
                        thresholds.append(q975)
                        beyond_rows.append(
                            head
                            | {"seed": s_, "classic_indicator": classic_names[i]}
                            | {"latent_auc": f["auc"], "classic_auc": c_auc, "difference": diff}
                            | {"difference_lo": d_lo, "difference_hi": d_hi, "shift_q975": q975}
                            | {"above_shift": bool(diff > q975), "above_zero": bool(diff > 0)}
                        )
                    beyond = beyond_verdict(diffs, thresholds)
                    units.append(
                        head
                        | {"decisive": decisive, "classic_indicator": classic_names[i]}
                        | {"warns": warns, "learned": learned, "beyond_classic": beyond}
                    )
                    if group in symbols:
                        incremental += incremental_rows(
                            head,
                            seeds,
                            found,
                            [
                                rv_of(raw_returns[group], ev_first[group, h], span, doc),
                                rv_of(raw_returns[group], ct_first[group], span, doc),
                            ],
                            wp,
                            wn,
                            level,
                        )
                    cascade_rows += cascade_unit(
                        head,
                        seeds,
                        by,
                        cascades_of,
                        controls_of,
                        np.random.default_rng([seed, BOOTSTRAP_STREAM, p, h, mi, g]),
                        draws,
                        level,
                    )

    classic_cells, classic_draws, classic_cascades = [], [], []
    for i, name in enumerate(classic_names):
        for h in horizons:
            for group in groups:
                code = ms.group_code(symbols, group)
                pos = _group_events({s: cls_ev[i, s, h] for s in symbols}, group, symbols, alone[h])
                neg = [cls_ct[i, s] for s in _alone_sides(group, symbols)]
                head = {"indicator": name, "horizon_bars": h, "group": group}
                classic_cells.append(head | msc.auc_cell(pos, neg, doc, i, h, code))
                negatives = np.concatenate(neg)
                moved_rows = shift_positives({s: cls_sh[i, s, h] for s in symbols}, group, h)
                for k, row in enumerate(moved_rows):
                    classic_draws.append(head | {"draw": k, "auc": scoring.auc(row, negatives)})
                cas = _group_events(
                    {s: cls_cas[i, s, h] for s in symbols}, group, symbols, cascade_alone[h]
                )
                classic_cascades.append(head | msc.auc_cell(cas, neg, doc, i, h, code))

    hyp, verdicts = hypothesis_rows(units, doc)
    paired_counts = [
        {"series": "paired", "symbol": s, "kind": "event", "horizon_bars": h}
        | {"cut": len(segs.events[s, h].first_ms), "kept": int(ev_keep[s, h].sum())}
        | {"excluded_finite": int((~ev_keep[s, h]).sum())}
        for s in symbols
        for h in horizons
    ] + [
        {"series": "paired", "symbol": s, "kind": "control", "horizon_bars": None}
        | {"cut": len(segs.controls[s].first_ms), "kept": int(ct_keep[s].sum())}
        | {"excluded_finite": int((~ct_keep[s]).sum())}
        for s in symbols
    ]
    for row in cascade_counts:
        row["excluded_finite"] = int((~cas_keep[row["symbol"], row["horizon_bars"]]).sum())
        row["kept"] -= row["excluded_finite"]
    segment_rows = (
        [{"series": "returns"} | c for c in base_segs.counts]
        + [{"series": "latent-masked"} | c for c in segs.counts]
        + paired_counts
        + cascade_counts
    )
    kept_share = {
        (s, h): segs.events[s, h].high_share[ev_keep[s, h]] for s in symbols for h in horizons
    }
    tier, tier_counts = tier_rows(prices, book, fit, method, ev_first, ct_first, alone, doc, base)
    tables_out = {
        "cells": cells,
        "classic_cells": classic_cells,
        "units": units,
        "margins": margin_rows,
        "beyond_classic": beyond_rows,
        "hypotheses": hyp,
        "shift_draws": shift_rows,
        "classic_shift_draws": classic_draws,
        "surrogates": surrogate_rows(
            pipes,
            names,
            cps,
            symbols,
            horizons,
            columns,
            tables,
            doc,
            latent_span,
            ev_first,
            ct_first,
            alone,
            span,
        ),
        "strata": strata_rows(
            names,
            cps,
            symbols,
            horizons,
            groups,
            lat_ev,
            lat_ct,
            cls_ev,
            cls_ct,
            classic_names,
            anchors,
            alone,
        ),
        "cascades": cascade_rows,
        "classic_cascades": classic_cascades,
        "same_input": tier,
        "same_input_segments": tier_counts,
        "incremental_rv": incremental,
        "segments": segment_rows,
        "high_share": [
            {"symbol": s, "horizon_bars": h, "onset_ns": int(t), "high_share": float(v)}
            for s in symbols
            for h in horizons
            for t, v in zip(anchors[s, h], kept_share[s, h], strict=True)
        ],
        "high_share_mean": [
            {"symbol": s, "horizon_bars": h, "events": len(kept_share[s, h])}
            | {
                "mean_high_share": float(np.mean(kept_share[s, h]))
                if len(kept_share[s, h])
                else math.nan
            }
            for s in symbols
            for h in horizons
        ],
    }
    kept = {"events": ev_keep, "controls": ct_keep}
    return Evaluation(tables_out, verdicts, segs, masked, kept)


def rv_of(
    returns: ms.Minutes, first_ms: np.ndarray, span: int, doc: Mapping[str, Any]
) -> np.ndarray:
    minutes = doc["descriptive"]["rv_minutes"]
    rows = rows_at(returns, first_ms, span)[:, span - minutes :]
    return (rows**2).sum(axis=1)


def incremental_rows(
    head: Row,
    seeds: Sequence[int],
    found: Mapping[tuple[int, str], Row],
    rv: Sequence[np.ndarray],
    wp: np.ndarray,
    wn: np.ndarray,
    level: float,
) -> list[Row]:
    variance = np.concatenate(rv)
    first = found[seeds[0], KINDS[0]]
    y = np.concatenate([np.ones(len(first["pos"])), np.zeros(len(first["neg"]))])
    w = np.hstack([wp, wn])
    ones = np.ones((1, len(y)))
    rows = []
    if not len(first["pos"]) or not len(first["neg"]):
        rv_auc, rv_draws = math.nan, np.full(len(w), np.nan)
    else:
        rv_auc = float(logistic_auc(variance[:, None], y, ones)[0])
        rv_draws = logistic_auc(variance[:, None], y, w)
    for s_ in seeds:
        for kind in KINDS:
            f = found[s_, kind]
            if not len(f["pos"]) or not len(f["neg"]):
                full, full_draws = math.nan, np.full(len(w), np.nan)
            else:
                x = np.column_stack([np.concatenate([f["pos"], f["neg"]]), variance])
                full = float(logistic_auc(x, y, ones)[0])
                full_draws = logistic_auc(x, y, w)
            lo, hi = scoring.percentile_interval(full_draws - rv_draws, level)
            rows.append(
                {k: v for k, v in head.items() if k != "group"}
                | {"read": head["group"], "seed": s_, "checkpoint": kind}
                | {"auc_latent_rv": full, "auc_rv": rv_auc, "incremental": full - rv_auc}
                | {"incremental_lo": lo, "incremental_hi": hi}
                | {"n_pos": len(f["pos"]), "n_neg": len(f["neg"])}
            )
    return rows


def cascade_unit(
    head: Row,
    seeds: Sequence[int],
    by: Mapping[tuple[int, str], int],
    events_of: Mapping[int, list[np.ndarray]],
    controls_of: Mapping[int, list[np.ndarray]],
    rng: np.random.Generator,
    draws: int,
    level: float,
) -> list[Row]:
    some = by[seeds[0], KINDS[0]]
    wp, wn = counts(
        [len(v) for v in events_of[some]], [len(v) for v in controls_of[some]], rng, draws
    )
    rows = []
    for s_ in seeds:
        for kind in KINDS:
            ci = by[s_, kind]
            pos, neg = np.concatenate(events_of[ci]), np.concatenate(controls_of[ci])
            lo, hi = scoring.percentile_interval(counted_aucs(pos, neg, wp, wn), level)
            rows.append(
                head
                | {"seed": s_, "checkpoint": kind, "auc": scoring.auc(pos, neg)}
                | {"auc_lo": lo, "auc_hi": hi, "n_pos": len(pos), "n_neg": len(neg)}
            )
    return rows


def hypothesis_rows(
    units: Sequence[Row], doc: Mapping[str, Any]
) -> tuple[list[Row], dict[str, str]]:
    rows, verdicts = [], {}
    for name in pair_names(doc):
        mine = [u for u in units if u["pair"] == name and u["decisive"]]
        warns = msc.hypothesis_verdict([u["warns"] for u in mine])
        learned = beyond = ""
        if warns == "pass":
            learned = msc.hypothesis_verdict([u["learned"] for u in mine])
            beyond = msc.hypothesis_verdict([u["beyond_classic"] for u in mine])
        verdicts[name] = pair_verdict(warns, learned, beyond)
        rows.append(
            {"hypothesis": name, "verdict": verdicts[name], "warns": warns, "learned": learned}
            | {"beyond_classic": beyond, "decisive_units": len(mine)}
            | {
                f"{part}_{outcome}": sum(u[part] == outcome for u in mine)
                for part in ("warns", "learned", "beyond_classic")
                for outcome in ("pass", "fail", "inconclusive")
            }
        )
    return rows, verdicts


def surrogate_rows(
    pipes: Sequence[Pipeline],
    names: Sequence[str],
    cps: Sequence[Checkpoint],
    symbols: Sequence[str],
    horizons: Sequence[int],
    columns: Sequence[str],
    tables: Tables,
    doc: Mapping[str, Any],
    latent_span: str,
    ev_first: Mapping[tuple[str, int], np.ndarray],
    ct_first: Mapping[str, np.ndarray],
    alone: Mapping[int, np.ndarray],
    span: int,
) -> list[Row]:
    sc = doc["scoring"]
    seed, n = sc["seed"], sc["surrogates"]["n"]
    rows = []
    for cp in cps:
        m = list(symbols).index(cp.model)
        t = KINDS.index(cp.kind)
        found: dict[tuple[int, str, int], np.ndarray] = {}
        for s in symbols:
            r = ms.group_code(symbols, s)
            series = latent_series(tables, cp, s, doc, latent_span)
            blocks = [(0, 1, ct_first[s]), *((h, 0, ev_first[s, h]) for h in horizons)]
            bounds = np.cumsum([len(f) for _, _, f in blocks])[:-1]
            firsts = np.concatenate([f for _, _, f in blocks])
            for p, pipe in enumerate(pipes):
                keys = [
                    [seed, SURROGATE_STREAM, p, h, m, cp.seed, t, r, c, j]
                    for h, c, f in blocks
                    for j in range(len(f))
                ]
                x = rows_at(series[columns[p]], firsts, span)
                split = np.split(two_sided_pvalues(x, pipe, n, keys), bounds)
                found[p, s, 0] = split[0]
                for k, h in enumerate(horizons):
                    found[p, s, h] = split[1 + k]
        for p, name in enumerate(names):
            for h in horizons:
                for group in ["pooled", *symbols, "eth-only"]:
                    ev = np.concatenate(
                        _group_events(
                            {s: found[p, s, h] for s in symbols}, group, symbols, alone[h]
                        )
                    )
                    ct = np.concatenate([found[p, s, 0] for s in _alone_sides(group, symbols)])
                    rows.append(
                        {"pair": name, "horizon_bars": h, "model": cp.model, "seed": cp.seed}
                        | {"checkpoint": cp.kind, "group": group}
                        | {"surrogate": sc["surrogates"]["secondary"], "n_surrogates": n}
                        | {"alpha": SURROGATE_ALPHA}
                        | {
                            "detected": float(np.mean(ev <= SURROGATE_ALPHA))
                            if len(ev)
                            else math.nan,
                            "false_positive": float(np.mean(ct <= SURROGATE_ALPHA))
                            if len(ct)
                            else math.nan,
                            "n_pos": len(ev),
                            "n_neg": len(ct),
                        }
                    )
    return rows


def strata_rows(
    names: Sequence[str],
    cps: Sequence[Checkpoint],
    symbols: Sequence[str],
    horizons: Sequence[int],
    groups: Sequence[str],
    lat_ev: Mapping[tuple[int, int, str, int], np.ndarray],
    lat_ct: Mapping[tuple[int, int, str], np.ndarray],
    cls_ev: Mapping[tuple[int, str, int], np.ndarray],
    cls_ct: Mapping[tuple[int, str], np.ndarray],
    classic_names: Sequence[str],
    anchors: Mapping[tuple[str, int], np.ndarray],
    alone: Mapping[int, np.ndarray],
) -> list[Row]:
    rows = []
    keys = [(s, h) for s in symbols for h in horizons]
    sides: list[tuple[Row, dict[tuple[str, int], np.ndarray], dict[str, np.ndarray]]] = []
    for p, name in enumerate(names):
        for ci, cp in enumerate(cps):
            sides.append(
                (
                    {"kind": "latent", "pair": name, "model": cp.model, "seed": cp.seed}
                    | {"checkpoint": cp.kind},
                    {(s, h): lat_ev[ci, p, s, h] for s, h in keys},
                    {s: lat_ct[ci, p, s] for s in symbols},
                )
            )
    for i, name in enumerate(classic_names):
        sides.append(
            (
                {"kind": "classic", "pair": name, "model": "", "seed": 0, "checkpoint": ""},
                {(s, h): cls_ev[i, s, h] for s, h in keys},
                {s: cls_ct[i, s] for s in symbols},
            )
        )
    for head, ev, ct in sides:
        for h in horizons:
            hours = {s: (anchors[s, h] // 3_600_000_000_000) % 24 for s in symbols}
            for hour in range(24):
                for group in groups:
                    pos = np.concatenate(
                        _group_events(
                            {s: ev[s, h][hours[s] == hour] for s in symbols},
                            group,
                            symbols,
                            alone[h][hours[ETH] == hour],
                        )
                    )
                    neg = np.concatenate([ct[s] for s in _alone_sides(group, symbols)])
                    rows.append(
                        head
                        | {"horizon_bars": h, "group": group, "hour": hour}
                        | {"auc": scoring.auc(pos, neg), "n_pos": len(pos)}
                        | {"n_neg": len(neg)}
                    )
    return rows


def tier_rows(
    prices: Mapping[str, ms.Minutes],
    book: Mapping[str, Mapping[str, ms.Minutes]],
    fit: tuple[date, date],
    method: str,
    ev_first: Mapping[tuple[str, int], np.ndarray],
    ct_first: Mapping[str, np.ndarray],
    alone: Mapping[int, np.ndarray],
    doc: Mapping[str, Any],
    base: Mapping[str, Any],
) -> tuple[list[Row], list[Row]]:
    symbols, horizons = list(doc["data"]["symbols"]), list(doc["segments"]["horizons"])
    tier = msc.tier_series(prices, book, fit, method)
    si = base["same_input"]
    pipes = {
        name: Pipeline(name, si["window"], si["length"], None, base["indicators"]["lag"])
        for name in si["indicators"]
    }
    span = next(iter(pipes.values())).span
    first = next(iter(tier.values()))
    ev_ok = {
        (s, h): np.isfinite(rows_at(first[s], ev_first[s, h], span)).all(axis=-1)
        for s in symbols
        for h in horizons
    }
    ct_ok = {s: np.isfinite(rows_at(first[s], ct_first[s], span)).all(axis=-1) for s in symbols}
    counts_out = [
        {"symbol": s, "kind": "event", "horizon_bars": h}
        | {"paired": len(ev_ok[s, h]), "kept": int(ev_ok[s, h].sum())}
        for s in symbols
        for h in horizons
    ] + [
        {"symbol": s, "kind": "control", "horizon_bars": None}
        | {"paired": len(ct_ok[s]), "kept": int(ct_ok[s].sum())}
        for s in symbols
    ]
    rows = []
    for series_name, by_symbol in tier.items():
        for i, (name, pipe) in enumerate(pipes.items()):
            neg = {
                s: score_rows(pipe, rows_at(by_symbol[s], ct_first[s][ct_ok[s]], span))
                for s in symbols
            }
            for h in horizons:
                pos = {
                    s: score_rows(pipe, rows_at(by_symbol[s], ev_first[s, h][ev_ok[s, h]], span))
                    for s in symbols
                }
                lonely = alone[h][ev_ok[ETH, h]]
                for group in ["pooled", *symbols, "eth-only"]:
                    ev = _group_events(pos, group, symbols, lonely)
                    ct = [neg[s] for s in _alone_sides(group, symbols)]
                    code = ms.group_code(symbols, group)
                    rows.append(
                        {"series": series_name, "indicator": name, "horizon_bars": h}
                        | {"group": group}
                        | msc.auc_cell(ev, ct, doc, i, h, code)
                    )
    return rows, counts_out


def trial_records(stamp: Mapping[str, Any], evaluation: Evaluation) -> list[Row]:
    latent = [
        {
            "prereg": dict(stamp),
            "cell": {"kind": "latent"}
            | {k: row[k] for k in ("pair", "horizon_bars", "model", "seed", "checkpoint", "group")},
            "auc": row["auc"],
            "auc_ci": [row["auc_lo"], row["auc_hi"]],
            "n_pos": row["n_pos"],
            "n_neg": row["n_neg"],
        }
        for row in evaluation.tables["cells"]
    ]
    classic = [
        {
            "prereg": dict(stamp),
            "cell": {"kind": "classic"}
            | {k: row[k] for k in ("indicator", "horizon_bars", "group")},
            "auc": row["auc"],
            "auc_ci": [row["auc_lo"], row["auc_hi"]],
            "n_pos": row["n_pos"],
            "n_neg": row["n_neg"],
        }
        for row in evaluation.tables["classic_cells"]
    ]
    return latent + classic


def score(
    pre: prereg.Prereg,
    ref: prereg.Frozen,
    baseline: prereg.Prereg,
    labels: prereg.Prereg,
    run: Path,
    prices: Mapping[str, ms.Minutes],
    cascades: Mapping[str, np.ndarray],
    book: Mapping[str, Mapping[str, ms.Minutes]],
    tables: Tables,
    data_dir: Path,
    inputs: list[Row],
    params: Row,
) -> tuple[Path, Evaluation]:
    planned = prereg.market_latent_cells(pre.doc)
    prereg.charge(data_dir, pre, planned)
    run_dir = prereg.begin_result(data_dir, pre, ref, {})
    prereg.charge(data_dir, pre, planned, own=run_dir.name)
    evaluation = evaluate(
        pre, baseline, labels, run, prices, cascades, book, tables, confirmatory=True
    )
    trials = trial_records(prereg.stamp(pre, ref), evaluation)
    if len(trials) != planned:
        raise prereg.PreregError(
            f"{pre.name}: {len(trials)} AUC cells evaluated, {planned} counted"
        )
    prereg.log_trials(data_dir, [t | {"result": run_dir.name} for t in trials])
    prereg.finish_result(data_dir, pre, run_dir, evaluation.tables, inputs, params, trials)
    (run_dir / "report.md").write_text(markdown(evaluation, run_dir.name, None), encoding="utf-8")
    return run_dir, evaluation


def dry_run(
    pre: prereg.Prereg,
    baseline: prereg.Prereg,
    labels: prereg.Prereg,
    run: Path,
    prices: Mapping[str, ms.Minutes],
    cascades: Mapping[str, np.ndarray],
    book: Mapping[str, Mapping[str, ms.Minutes]],
    tables: Tables,
    data_dir: Path,
    out: Path,
    inputs: list[Row],
    params: Row,
) -> tuple[Path, Evaluation]:
    counted = (Path(data_dir) / "ews").resolve()
    if counted == out.resolve() or counted in out.resolve().parents:
        raise prereg.PreregError(f"{out} is inside {counted}, where scored results live")
    evaluation = evaluate(
        pre, baseline, labels, run, prices, cascades, book, tables, confirmatory=False
    )
    out.mkdir(parents=True)
    stamp = {"id": pre.id, "version": pre.version, "sha256": pre.sha256, "tag": "", "commit": ""}
    manifest = {
        "format": 1,
        "domain": "ews",
        "exploratory": EXPLORATORY,
        "run_id": out.name,
        "producer": {"name": "tipping_point.ews", "version": __version__, **io.code_version()},
        "started_ns": time.time_ns(),
        "key": "cell",
        "tables": [],
        "params": {"prereg": stamp},
    }
    io.write_json(out / "manifest.json", manifest)
    (out / "report.md").write_text(markdown(evaluation, out.name, EXPLORATORY), encoding="utf-8")
    prereg.finish_result(
        data_dir, pre, out, evaluation.tables, inputs, {"exploratory": EXPLORATORY, **params}
    )
    return out, evaluation


def run(
    pre: prereg.Prereg,
    labels_run: Path,
    cascade_run: Path,
    book_root: Path,
    archive_root: Path,
    latents_root: Path,
    gate_path: Path,
    pins_path: Path,
    data_dir: Path,
    *,
    exploratory: bool = False,
    out: Path | None = None,
) -> tuple[Path, Evaluation, bool]:
    confirmatory = not exploratory
    baseline = check_file(pre)
    labels = msc.labels_file(baseline)
    manifest = json.loads((Path(labels_run) / "manifest.json").read_text(encoding="utf-8"))
    msc.check_span(manifest, labels, confirmatory)
    doc, symbols = pre.doc, list(pre.doc["data"]["symbols"])
    ref = prereg.frozen(pre)
    code = None
    if confirmatory:
        ms.check_scoring(baseline)
        code = prereg.code_check(pre, ref, SRC, HELD, CLEAN)
        if code["changed"] or code["uncommitted"]:
            raise prereg.PreregError(
                f"files not at {ref.tag}'s commit: {', '.join(code['changed']) or 'none'}; "
                f"not committed: {', '.join(code['uncommitted']) or 'none'}"
            )
        if io.code_version()["dirty"] is not False:
            raise prereg.PreregError(
                "the checkout has uncommitted changes or untracked files: the result's commit "
                "must name what ran"
            )
        prereg.charge(data_dir, pre, prereg.market_latent_cells(doc))
    try:
        record = json.loads(Path(gate_path).read_text(encoding="utf-8"))
        pins_raw = Path(pins_path).read_bytes()
    except (OSError, ValueError) as exc:
        raise prereg.PreregError(f"the gate record or pins.json: {exc}") from exc
    check_gate(pre, ref, record, pins_raw)
    spans = ("fit", "confirmatory") if confirmatory else ("fit",)
    tables = Merged(latents_root, record, pre, labels, spans)
    first = msc.ALARM_SPAN[0] - timedelta(days=1)
    last = date.fromisoformat(manifest["params"]["end"])
    msc.check_sources(manifest, symbols, first, last, archive_root)
    ms.check_labels(labels_run, labels, baseline if confirmatory else None)
    onsets, _ = msc.cascade_onsets(cascade_run, labels, symbols, confirmatory)
    span = archive.collect(symbols, first, last, archive_root)
    try:
        msc.check_sources(manifest, symbols, first, last, archive_root)
    except ms.SegmentError as exc:
        raise prereg.PreregError(f"the archive changed while it was read: {exc}") from exc
    start, minutes = archive.day_ms(first), ((last - first).days + 1) * 1440
    prices = {s: ms.close_returns(span.columns[(s, "klines")], start, minutes) for s in symbols}
    aggregation = baseline.doc["same_input"]["aggregation"]
    opened = baseline if confirmatory else None
    reads = {
        s: same_input.read(book_root, s, first, last, aggregation, scoring=opened) for s in symbols
    }
    sources = hashlib.sha256(json.dumps(span.sources, sort_keys=True).encode()).hexdigest()
    inputs = [
        msc._run_input("labels", labels_run),
        msc._run_input("cascades", cascade_run),
        {"domain": "archive", "files": len(span.sources), "sources_sha256": sources},
        {
            "domain": "bybit-book",
            "read": {
                s: {"format": r.format, "missing_days": r.missing_days} for s, r in reads.items()
            },
        },
        {
            "domain": "latents",
            "gate_sha256": hashlib.sha256(Path(gate_path).read_bytes()).hexdigest(),
            "pins_sha256": hashlib.sha256(pins_raw).hexdigest(),
            "tables": {
                f"{name}/{symbol}-{s}": m["parquet_sha256"]
                for (name, symbol, s), m in sorted(tables.manifests.items())
            },
        },
    ]
    params = {"read": [first.isoformat(), last.isoformat()], "code": code}
    book = {s: r.series for s, r in reads.items()}
    given = (baseline, labels, labels_run, prices, onsets, book, tables, data_dir)
    if confirmatory:
        run_dir, evaluation = score(pre, ref, *given, inputs, params)
    else:
        out = out or Path(data_dir) / "ews-exploratory" / engine.run_id(f"{pre.name}-dry-run")
        run_dir, evaluation = dry_run(pre, *given, out, inputs, params)
    return run_dir, evaluation, confirmatory


def _fmt(x: float) -> str:
    return "nan" if x is None or math.isnan(x) else f"{x:.3f}"


def markdown(evaluation: Evaluation, name: str, exploratory: str | None) -> str:
    t = evaluation.tables
    title = "Market latent warning test" + (" (EXPLORATORY DRY RUN)" if exploratory else "")
    lines = [f"# {title}", "", f"Result `{name}`.", ""]
    if exploratory:
        lines += [f"**{exploratory}.**", ""]
    lines += [
        "| hypothesis | verdict | warns | learned | beyond classic |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['hypothesis']} | {r['verdict']} | {r['warns']} | {r['learned'] or '-'} | "
        f"{r['beyond_classic'] or '-'} |"
        for r in t["hypotheses"]
    ]
    lines += [
        "",
        "Decisive units (h = 5 and 6): the trained AUCs of the five primaries, and each part's "
        "outcome.",
        "",
        "| pair | h | model | group | trained AUCs | warns | learned | beyond classic |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for u in t["units"]:
        if not u["decisive"]:
            continue
        aucs = [
            c["auc"]
            for c in t["cells"]
            if c["checkpoint"] == "trained"
            and (c["pair"], c["horizon_bars"], c["model"], c["group"])
            == (u["pair"], u["horizon_bars"], u["model"], u["group"])
        ]
        lines.append(
            f"| {u['pair']} | {u['horizon_bars']} | {u['model']} | {u['group']} | "
            f"{' '.join(_fmt(a) for a in aucs)} | {u['warns']} | {u['learned']} | "
            f"{u['beyond_classic']} |"
        )
    lines += [
        "",
        "| classic indicator | h | group | n events | n controls | AUC [95 % CI] |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['indicator']} | {r['horizon_bars']} | {r['group']} | {r['n_pos']} | "
        f"{r['n_neg']} | {_fmt(r['auc'])} [{_fmt(r['auc_lo'])}, {_fmt(r['auc_hi'])}] |"
        for r in t["classic_cells"]
    ]
    return "\n".join(lines) + "\n"
