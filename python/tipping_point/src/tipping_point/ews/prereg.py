"""Pre-registration files: paper/prereg/<id>-v<k>.toml, read with the standard library's tomllib.

load      parse and check the structure (placeholders such as "TBD" or "<...>" are allowed)
freeze    refuse a file with placeholders, or one that is untracked or differs from HEAD, then tag
          the commit prereg/<id>-v<k> (an annotated tag: it carries its own timestamp)
frozen    the file's sha256 must equal that of the blob at its tag; scoring refuses otherwise
stamp     what a result's manifest records: the prereg's id, version, sha256, tag and commit
code_check  what a run reading a frozen file records of its code and refuses on: the files that
          make its output held to the frozen commit, the others it runs held to HEAD, and the
          sha256 of uv.lock
begin_result  <data dir>/ews/<run_id>/ and its manifest.json (stamped, with the resume key a
          scoring must share to carry the folder on after a crash), before anything is scored
save_part a finished unit of a scoring (a checkpoint's pass, a probe) in the folder's parts/, which
          a resumed scoring reads back instead of computing again
finish_result  the rest of #22's layout: one Parquet file per table, its trials logged,
          complete.json last: a crash leaves a folder with no complete.json, or one whose every
          trial is logged. write_result is both at once
log_trials append evaluations to <data dir>/ews/trials.jsonl in one write; every evaluation counts
          as a trial, sensitivity cells included, and the count goes in the paper
charge    refuse a scoring past the file's budget, or past a torn result of its own version other
          than the one it carries on (resumable)

An amendment is a new file, <id>-v<k+1>.toml, whose `supersedes` names the version it replaces
and whose `reason` says why; results scored under an earlier version stay bound to it.

A file with a [latents] section scores a JEPA's latent columns (#28). It is checked against its own
schema (_LATENT_SCHEMA): the checkpoints it scores, each with its untrained twin, the (observable,
indicator) pairs judged by AUC, the statistics judged by the dynamic test, and the [baseline] whose
runs it reads. As a draft it may leave as a placeholder a direction, an extremum or a reference's
tolerance, which the file's experiment sets before the freeze; the freeze's strict check refuses
every placeholder, whichever key holds it. Frozen text must not claim what the scorer ignores: a key
outside the schema is refused, and so is a value the scorer would not act on (_latent_problems).

A file's data.domain picks its schema. "ising" takes the checks above, unchanged, and so does a file
whose domain is missing or not a string, which those checks then report. "market-labels" takes
_MARKET_SCHEMA (#29 S0): the parameters of the market labeller and the cascade proxy, its [labeller]
and [cascade] tables field for field those of market.msvol.Params and market.cascade.Params, the
span and its pilot/confirmatory boundary, and the null-stretch rule and smallest horizon, which a
draft may leave as placeholders (_market_problems). The research pass's "source files' manifest
hash" is left out: the fetch manifest grows with every fetch, and a labels run records each
file's sha256 itself. "market-ews" takes _MARKET_EWS_SCHEMA (#30 M5), the classic warning test
on the frozen labels: the labels file's tag and sha256, the series, indicators and segment rule
that ews.market_segments carries out (held word for word, as bars.missing is), the
deseasonalisation, horizons from the labels' smallest, the AUC bootstrap, the GARCH floor, the
event-time shift and the phase surrogates, the hypotheses, reporting groups, alarm rule and trial
budget (_market_ews_problems). A draft may leave the detrending, the floor's three open choices
and the alarm rate as placeholders; a floor option is accepted only once its code is built. Any
other domain is refused. load, freeze, frozen and stamp take a file of any domain; cells, charge
and the result functions are the Ising scorer's.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import socket
import subprocess
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .. import __version__, engine
from ..ising import io

# ews.trend.RAW, held here so that the market labeller can import this module without reaching the
# feature pipelines (tests/market/test_circularity.py); test_prereg_market holds the two equal
RAW = frozenset({"level", "slope"})

PREREG_DIR = Path("paper") / "prereg"
INDICATORS = {"variance", "ac", "skewness"}
OBSERVABLES = {"abs_m", "energy"}
CADENCES = {"dense", "matched"}
DIRECTIONS = {"greater", "less", "two-sided", "match"}
ROLES = {"positive-control", "descriptive", "reference"}
QUANTITIES = {"l2_var_abs_m", "mean_abs_m", "skewness_abs_m", "tau_int_abs_m", "one_minus_ac1"}
SURROGATES = {"phase", "ar1"}
STATISTICS = {"peak"}  # fss.dynamic.statistic: what the dynamic test reads off each ramp
# in a file with a [latents] section, which has its own observables, roles and statistics
LATENT_COLUMNS = {"surprise", *(f"pc_{k}" for k in range(1, 9))}
LATENT_INDICATORS = INDICATORS | RAW
LATENT_ROLES = ROLES | {"pipeline-check"}
LATENT_STATISTICS = ("l2_var", "mean", "one_minus_ac1")  # a quantity is <statistic>_<column>
CHECKPOINT_ROLES = {"primary", "ema-only", "held-out"}  # primary: the whole grid; else the headline
DYNAMIC_ROLES = {"pipeline-check", "test", "descriptive"}
# a judged row's optional `judged`: on trained checkpoints and twins (the default), or twins alone
JUDGED_ON = ("both", "twin")  # a tuple: membership must not hash whatever a file holds there
EXTREMA = {"max", "min"}
# jepa.probes.TARGETS and COLUMNS but raw, which this package cannot import
PROBE_TARGETS, PROBE_CONTROLS = {"abs_m", "energy", "l2var"}, {"iid", "shuffle_within_T"}
PROBE_SELECTIVITY = "R^2(target) - R^2(shuffle_within_T)"  # what jepa.probes.score computes
_PROBE_FIT = re.compile(
    r"the staircase of simulator seed (\d+) at each L: "
    r"(\d+) frames drawn without replacement, probe seed (\d+)"
)
# latents.extraction, word for word, and the jepa.extract.selection a staircase's table must have
# been read under, per checkpoint role, which the scorer holds each table to (ising.admit)
EXTRACTION = (
    "ramps and nulls in full; staircases: the T = 2.269231 plateau in full and one burst in eight "
    "elsewhere (primary and twins); ablation checkpoints: the Tc plateau, ramps and nulls only"
)
_TC_PLUS = {"rule": "tc-plus", "tc_plateau": 2.269231, "every": 8}
SELECTIONS = {
    "primary": _TC_PLUS,
    "ema-only": _TC_PLUS | {"rule": "tc-only"},
    "held-out": _TC_PLUS | {"rule": "tc-only"},
}
# fss.dynamic.edge_rule, word for word: what ising._dynamic does with what edge_margin leaves
EDGE_RULE = (
    "inconclusive when more than half of an L's replicas take their extremum on an edge of the "
    "band or within one headline window of it"
)
LEARNED = "learned-beyond-random"  # the hypothesis whose direction is the margins' predicted sign
TREND = "surprise-trend"  # the hypothesis that states the level's direction in words
# "TBD", "<words>" as in TEMPLATE, or a choice still open, "<min|max, set before the freeze>"
_PLACEHOLDER = re.compile(r"TBD|<[a-z][^<>|]*>|<[a-z]+(\|[a-z]+)+,[^<>]*>")
_SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# a market-labels file (#29): tuples, so that membership never hashes whatever a file holds there
MARKET_LABELS = "market-labels"
MARKET_EWS = "market-ews"
DOMAINS = ("ising", MARKET_LABELS, MARKET_EWS)
MARKET_SYMBOLS = ("BTCUSDT", "ETHUSDT")  # market.archive.SYMBOLS
RV_ESTIMATORS = ("garman-klass",)  # what market.bars computes; Parkinson is a reported variant
# what market.bars and market.msvol do with a bar they cannot use, word for word
MISSING = (
    "a bar short of any of its five 1-minute klines, with no minute traded, or with a Garman-Klass "
    "variance of 0 is a missing observation: the filter carries its prediction without an update, "
    "and nothing is floored"
)
# what market.msvol.nulls does: calm bars with no event onset within Q bars either side, the
# owner's choice after the S5 pilot (5 November 2026)
NULL_RULES = ("matched-calm-windows",)
# the ranges the 2023 pilot chooses inside (#29 research pass, section 2, item 13)
D_MIN = (6, 12, 36)
THRESHOLDS = ((0.7, 0.3), (0.8, 0.2), (0.9, 0.1))  # (enter, exit)
BASELINE_DAYS = (14, 28, 56)
# C and Q at their start values: #29 declares no range for them, and the S5 pilot read sends a
# shorter Q to the owner with an amendment, so a frozen file may not change them on its own
CLEAN = (72,)
REFRACTORY = (288,)
CONFIRMATORY = date(2024, 1, 1)  # market.__main__.CONFIRMATORY: the first day the pilot never saw
# a market-ews file (#30 M5): the classic warning test on the frozen labels. The texts are the
# rules ews.market_segments carries out, word for word
EWS_INDICATORS = ("variance", "ac")  # ews.trend.INDICATORS' classic pair: variance and AC1
DETRENDS = ("none", "gaussian")  # ews.trend.Pipeline: bandwidth None, or a Gaussian bandwidth
HORIZON_FLOOR = 5  # market-labels-v1's horizon.smallest: the p90 detection delay, in bars
DAY_BARS = 288
EVENT_SETS = {"primary": "regime-shifts", "stress": "cascades"}
SERIES = (
    "Binance USD-M 1-minute klines, the archives market-labels-v1 reads: close-to-close log "
    "returns, the return of minute m being ln(close_m / close_(m-1)), known when minute m closes"
)
MISSING_MINUTE = (
    "a minute with no kline, or whose previous minute has none, has no return; a segment holding "
    "such a minute is excluded for every indicator and counted"
)
SEGMENT = (
    "an event segment at horizon h bars ends with the minute that closes at the onset minus 5h "
    "minutes and holds the window + length - 1 minutes up to it; the length is the same at every "
    "horizon, a segment may start before the labels' clean window, and the share of its minutes "
    "in bars with H = 1 is reported"
)
CONTROLS = (
    "control segments are the non-overlapping runs of window + length - 1 minutes cut in time "
    "order from the start of each null stretch of the labels run, one set for every horizon; with "
    "subsample n > 0, n of each symbol's are drawn without replacement under the seed, after the "
    "exclusions"
)
DESEASON = (
    "each 1-minute return is divided by the root mean square of the returns of its slot, one of "
    "2 x 288 (weekday or weekend, by 5-minute time of day, the labeller's slots), over the minutes "
    "of the fit span; the raw series is scored beside it, descriptively"
)
STRATA = ("onset-hour",)  # AUCs also reported per UTC hour of the onset, descriptively
REPORT_GROUPS = ("pooled", "per-asset", "eth-only")  # eth-only: ETH onsets no BTC onset is near
ALARMS = (
    "per indicator, horizon and symbol, the alarm threshold is the lowest whose false alarms on "
    "the 2023 control segments stay within far_per_day; lead times and false alarms per day under "
    "it are reported, descriptively"
)
# the floor options built so far: #154 adds a deseasonalised refit, #153 count-matched noise and
# #138 may add a refusal; each joins its tuple when its code lands
FLOOR_FITS = ("pilot",)  # the GARCH(1,1)-t fits on raw 2023 returns: PILOT_GARCH
FLOOR_NOISE = ("bootstrap",)  # the real AUC's own bootstrap CI is the noise band
FLOOR_EMPTY = ("mark",)  # an empty side gives no AUC (nulls._auc), and the cell is inconclusive
FLOOR_FIRST = date(2022, 7, 4)  # market.nulls.FIRST: simulated days stand from here
# harness/market_floor.PILOT_GARCH, per minute (omega, alpha, beta, nu): the fit market-labels-v1
# discloses; test_prereg_market_ews holds the two equal
PILOT_GARCH = {
    "BTCUSDT": (8.651445052105347e-11, 0.07110265726317705, 0.9288274749829264, 5.085328354482213),
    "ETHUSDT": (6.419879719145039e-10, 0.0801166904491908, 0.9197468505253973, 4.97264822567892),
}
GARCH_FIELDS = ("omega", "alpha", "beta", "nu")
_LABELS_TAG = re.compile(r"prereg/market-labels-v[1-9][0-9]*")


class PreregError(ValueError):
    """A pre-registration that is malformed, unfrozen or modified since its freeze."""


@dataclass(frozen=True)
class Prereg:
    path: Path
    doc: dict[str, Any]
    sha256: str

    @property
    def id(self) -> str:
        return self.doc["id"]

    @property
    def version(self) -> int:
        return self.doc["version"]

    @property
    def name(self) -> str:
        return f"{self.id}-v{self.version}"

    @property
    def tag(self) -> str:
        return f"prereg/{self.name}"


# key -> (type, required); a nested dict is a table checked recursively
_NUMBER = (int, float)
_SCHEMA: dict[str, Any] = {
    "id": str,
    "version": int,
    "supersedes": str,
    "reason": str,
    "question": str,
    "hypotheses": [{"name": str, "statement": str, "prediction": str, "direction": str}],
    "data": {
        "domain": str,
        "L": [int],
        "seeds": [int],
        "replicas": int,
        "exclusions": [{"run_id": str, "reason": str}],
        "tau_c": {"source": str, "pilot_run": str, "pilot_sha256": str},
        "ramps": {"kind": str, "x": _NUMBER},
        "nulls": {"kind": str, "eps_L": [_NUMBER]},
        "staircase": {"kind": str, "temperature": _NUMBER},
    },
    "indicators": {
        "names": [str],
        "observables": [str],
        "cadences": [str],
        "thin": _NUMBER,
        "lag": int,
        "detrend": str,
        "direction": {},
    },
    "event": {"definition": str, "horizons": [_NUMBER], "segment": _NUMBER},
    "scoring": {
        "positives": str,
        "negatives": str,
        "auc": str,
        "alpha": _NUMBER,
        "ci_level": _NUMBER,
        "bootstrap": int,
        "surrogates": {"primary": str, "secondary": str, "n": int, "cells": str},
        "alarms": {
            "indicators": [str],
            "far": _NUMBER,
            "report_per": _NUMBER,
            "band": [_NUMBER],
            "every": _NUMBER,
        },
    },
    "fss": {
        "static": [{"quantity": str, "role": str}],
        "dynamic": {
            "statistic": str,
            "indicators": [str],
            "observable": str,
            "cadence": str,
            "search": [_NUMBER],
            "expected_inverse_nu": _NUMBER,
            "tolerance": _NUMBER,
            "min_draw_share": _NUMBER,
        },
    },
    "disclosure": {"exploratory": str, "decisions": str, "confirmatory": str, "links": [str]},
    "sensitivity": {
        "window": [_NUMBER],
        "bandwidth": [_NUMBER],
        "headline": {"window": _NUMBER, "bandwidth": _NUMBER},
    },
    "trials": {"budget": int, "counted": str},
    "deviations": {"policy": str, "log": [{"date": str, "change": str, "reason": str}]},
}
_LATENT_SCHEMA: dict[str, Any] = _SCHEMA | {
    "baseline": {"prereg": str, "result": str},
    "latents": {
        "table": str,
        "columns": [str],
        "training_seeds": [int],
        "validation_seed": int,
        "steps": int,
        "extraction": str,
        "spares": [int],
        "checkpoints": [
            {
                "role": str,
                "model_seed": int,
                "training": str,
                "trained_sha256": str,
                "twin_sha256": str,
            }
        ],
    },
    "indicators": {  # thin is the dense cadence's, which a latents table does not have
        **{key: kind for key, kind in _SCHEMA["indicators"].items() if key != "thin"},
        "pairs": [[str]],
    },
    "scoring": _SCHEMA["scoring"]
    | {
        "surrogates": _SCHEMA["scoring"]["surrogates"] | {"n_ablation": int},
        "alarms": {  # every judged pair's: indicators.pairs
            "far": _NUMBER,
            "report_per": _NUMBER,
            "band": [_NUMBER],
            "every": _NUMBER,
        },
        "margins": {
            "pairing": str,
            "seeds": str,
            "sizes": str,
            "sign_test": bool,
            "tolerance": (*_NUMBER, str),  # a string only as a placeholder: _margin_problems
            "ablations": str,
        },
    },
    "fss": {
        "static": _SCHEMA["fss"]["static"],
        "dynamic": {
            "statistics": [{"observable": str, "indicator": str, "extremum": str, "role": str}],
            "cadence": str,
            "search": [_NUMBER],
            "expected_inverse_nu": _NUMBER,
            "tolerance": _NUMBER,
            "edge_rule": str,
            "edge_margin": str,
            "min_draw_share": _NUMBER,
        },
    },
    "probes": {
        "targets": [str],
        "controls": [str],
        "fit": str,
        "test": str,
        "ridge_alpha": _NUMBER,
        "selectivity": str,
        "per": str,
    },
}
# every key a file with a [latents] section may hold: the schema's, and those it may leave out
_LATENT_KEYS: dict[str, Any] = _LATENT_SCHEMA | {
    "data": _SCHEMA["data"]
    | {
        "ramps": _SCHEMA["data"]["ramps"] | {"eps_L": [_NUMBER]},
        "staircase": _SCHEMA["data"]["staircase"] | {"tau_sha256": {}},
    },
    "indicators": _LATENT_SCHEMA["indicators"] | {"direction_by_observable": {}},
    "fss": {
        "static": [
            _SCHEMA["fss"]["static"][0] | {"expected": _NUMBER, "tolerance": _NUMBER, "judged": str}
        ],
        "dynamic": _LATENT_SCHEMA["fss"]["dynamic"]
        | {"statistics": [_LATENT_SCHEMA["fss"]["dynamic"]["statistics"][0] | {"judged": str}]},
    },
}
# a file whose data.domain is "market-labels": every key it holds, none optional
_MARKET_SCHEMA: dict[str, Any] = {
    "id": str,
    "version": int,
    "supersedes": str,
    "reason": str,
    "question": str,
    "data": {
        "domain": str,
        "symbols": [str],
        "span": {"start": date, "confirmatory": date, "end": date},
    },
    "bars": {"estimator": str, "missing": str},
    "labeller": {  # market.msvol.Params
        "warmup_days": int,
        "refit_days": int,
        "baseline_days": int,
        "start_low": _NUMBER,
        "start_high": _NUMBER,
        "start_p": _NUMBER,
        "tol": _NUMBER,
        "max_iter": int,
        "max_iter_warm": int,
        "min_sigma": _NUMBER,
        "min_share": _NUMBER,
        "min_gap": _NUMBER,
        "enter": _NUMBER,
        "exit": _NUMBER,
        "d_min": int,
        "clean": int,
        "refractory": int,
    },
    "nulls": {"rule": str},
    "cascade": {  # market.cascade.Params
        "a": _NUMBER,
        "b": _NUMBER,
        "c": _NUMBER,
        "v": _NUMBER,
        "window": int,
        "trailing_days": int,
        "onset_percentile": _NUMBER,
        "merge_hours": int,
        "both_hours": int,
        "bounds_hours": int,
    },
    "horizon": {"smallest": (int, str)},  # a string only as a placeholder: _market_problems
    "disclosure": _SCHEMA["disclosure"],
    "deviations": _SCHEMA["deviations"],
}
# a file whose data.domain is "market-ews" (#30 M5): every key it holds; indicators.bandwidth is
# the one it may leave out (_MARKET_EWS_KEYS), and must, unless detrend is "gaussian"
_MARKET_EWS_SCHEMA: dict[str, Any] = {
    "id": str,
    "version": int,
    "supersedes": str,
    "reason": str,
    "question": str,
    "hypotheses": [
        {"name": str, "indicator": str, "statement": str, "prediction": str, "direction": str}
    ],
    "data": {
        "domain": str,
        "symbols": [str],
        "series": str,
        "missing": str,
        "labels": {"tag": str, "sha256": str},
        "events": {"primary": str, "stress": str},
    },
    "indicators": {"names": [str], "window": int, "length": int, "lag": int, "detrend": str},
    "seasonality": {"rule": str, "fit": {"start": date, "end": date}, "strata": [str]},
    "segments": {"horizons": [int], "rule": str, "controls": str, "subsample": int, "seed": int},
    "scoring": {
        "ci_level": _NUMBER,
        "bootstrap": int,
        "seed": int,
        "floor": {
            "fit": str,
            "noise": str,
            "empty": str,
            "paths": int,
            "days": int,
            "seed": int,
            "garch": {},  # per symbol: GARCH_FIELDS, checked in _market_ews_problems
        },
        "shift": {"draws": int, "min_offset_bars": int},
        "surrogates": {"secondary": str, "n": int},
    },
    "reporting": {"groups": [str]},
    "alarms": {"rule": str, "far_per_day": (*_NUMBER, str)},  # a string only as a placeholder
    "trials": {"budget": int, "counted": str},
    "disclosure": _SCHEMA["disclosure"],
    "deviations": _SCHEMA["deviations"],
}
_MARKET_EWS_KEYS: dict[str, Any] = _MARKET_EWS_SCHEMA | {
    "indicators": _MARKET_EWS_SCHEMA["indicators"] | {"bandwidth": _NUMBER}
}


def _check(value: Any, schema: Any, where: str, problems: list[str]) -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            problems.append(f"{where}: expected a table")
            return
        for key, sub in schema.items():
            if key not in value:
                problems.append(f"{where}.{key}: missing".lstrip("."))
            else:
                _check(value[key], sub, f"{where}.{key}".lstrip("."), problems)
    elif isinstance(schema, list):
        if not isinstance(value, list):
            problems.append(f"{where}: expected an array")
            return
        for i, item in enumerate(value):
            _check(item, schema[0], f"{where}[{i}]", problems)
    elif (isinstance(value, bool) and schema is not bool) or not isinstance(value, schema):
        # TOML booleans are not numbers
        names = " or ".join(
            t.__name__ for t in (schema if isinstance(schema, tuple) else (schema,))
        )
        problems.append(f"{where}: expected {names}, got {value!r}")


def _unread(value: Any, schema: Any, where: str = "") -> list[str]:
    """Where a document holds a key its schema does not name: nothing reads it, so a frozen file
    would be claiming what the scorer ignores. An empty table in the schema takes any key."""
    if isinstance(schema, list) and isinstance(value, list):
        return [k for i, v in enumerate(value) for k in _unread(v, schema[0], f"{where}[{i}]")]
    if not (schema and isinstance(schema, dict) and isinstance(value, dict)):
        return []
    found = []
    for key, item in value.items():
        here = f"{where}.{key}".lstrip(".")
        found += _unread(item, schema[key], here) if key in schema else [here]
    return found


def _strings(value: Any, where: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _strings(v, f"{where}.{k}".lstrip("."))]
    if isinstance(value, list):
        return [s for i, v in enumerate(value) for s in _strings(v, f"{where}[{i}]")]
    return [(where, value)] if isinstance(value, str) else []


def _placeholder(value: Any) -> bool:
    """A value still to be set: a latent draft may leave one where it has to choose, and strict
    refuses it anywhere."""
    return isinstance(value, str) and _PLACEHOLDER.search(value) is not None


def probe_protocol(doc: dict[str, Any]) -> tuple[int, int]:
    """(frames drawn per run, probe seed) as a latent file's probes.fit states them: the scorer
    reads the numbers it uses from the sentence the file freezes (_latent_problems checks its
    form)."""
    fit = _PROBE_FIT.fullmatch(doc["probes"]["fit"])
    if fit is None:
        raise PreregError(f"probes.fit: not a sentence the scorer reads: {doc['probes']['fit']!r}")
    return int(fit[2]), int(fit[3])


def _probe_problems(doc: dict[str, Any]) -> list[str]:
    """What a latent file's probe sentences say must be what the scorer will do with its seeds."""
    probes, seeds = doc["probes"], sorted(doc["data"]["seeds"])
    found = []
    if not probes["targets"] or not set(probes["targets"]) <= PROBE_TARGETS:
        found.append(f"probes.targets: a non-empty subset of {sorted(PROBE_TARGETS)}")
    if "energy" not in probes["targets"]:
        found.append("probes.targets: must hold energy, whose margins are judged")
    if set(probes["controls"]) != PROBE_CONTROLS:
        found.append(f"probes.controls: {sorted(PROBE_CONTROLS)}")
    if probes["ridge_alpha"] <= 0:
        found.append("probes.ridge_alpha: positive, per sample")
    if probes["selectivity"] != PROBE_SELECTIVITY:
        found.append(f"probes.selectivity: {PROBE_SELECTIVITY!r}, what the scorer computes")
    fit = _PROBE_FIT.fullmatch(probes["fit"])
    if fit is None or int(fit[1]) != doc["latents"]["validation_seed"]:
        found.append(
            "probes.fit: 'the staircase of simulator seed <latents.validation_seed> at each L: "
            "<n> frames drawn without replacement, probe seed <k>'"
        )
    elif seeds and seeds == list(range(seeds[0], seeds[-1] + 1)):
        said = f"the staircases of seeds {seeds[0]}-{seeds[-1]} at each L: {fit[2]} frames each"
        if probes["test"] != f"{said}, same rule":
            found.append(f"probes.test: {said + ', same rule'!r}, as data.seeds and probes.fit say")
    else:
        found.append("data.seeds: consecutive, so that probes.test can name them")
    return found


def _margin_problems(doc: dict[str, Any]) -> list[str]:
    """The margin rule's switches have one setting each, the one ews.margins.table carries out."""
    margins, found = doc["scoring"]["margins"], []
    if margins["sign_test"] is not True:
        found.append("scoring.margins.sign_test: true, the margin rule always takes it")
    if margins["sizes"] not in ("mean", "every"):
        found.append(
            "scoring.margins.sizes: 'mean', each checkpoint's judged value over L, or 'every', "
            "that value with its sign test at every L"
        )
    if margins["ablations"] != "descriptive":
        found.append("scoring.margins.ablations: 'descriptive', a margin with no verdict")
    tolerance = margins["tolerance"]
    if not _placeholder(tolerance) and (isinstance(tolerance, str) or tolerance <= 0):
        found.append("scoring.margins.tolerance: positive, a number in the margin's units")
    named = {h["name"]: h["direction"] for h in doc["hypotheses"]}
    if named.get(LEARNED) not in ("greater", "less"):
        found.append(f"hypotheses: {LEARNED} with 'greater' or 'less', the margins' predicted sign")
    return found


def _indicator_problems(doc: dict[str, Any]) -> list[str]:
    """indicators.pairs is what the scorer reads: a name, an observable or a direction that no pair
    uses would stand in the frozen file unread."""
    ind, found = doc["indicators"], []
    pairs = [tuple(pair) for pair in ind["pairs"]]
    if (
        not pairs
        or len(set(pairs)) != len(pairs)
        or any(
            len(pair) != 2 or pair[0] not in ind["observables"] or pair[1] not in ind["names"]
            for pair in pairs
        )
    ):
        return ["indicators.pairs: distinct [observable, indicator] pairs of the scored ones"]
    if set(ind["observables"]) != {observable for observable, _ in pairs}:
        found.append("indicators.observables: exactly those indicators.pairs scores")
    if set(ind["names"]) != {name for _, name in pairs}:
        found.append("indicators.names: exactly those indicators.pairs scores")
    by_observable: dict[str, dict[str, Any]] = {}
    for observable, own in ind.get("direction_by_observable", {}).items():
        if observable not in ind["observables"] or not isinstance(own, dict):
            found.append(f"indicators.direction_by_observable.{observable}: a scored observable")
            continue
        by_observable[observable] = own
        for name, value in own.items():
            where = f"indicators.direction_by_observable.{observable}.{name}"
            if value not in ("greater", "less") and not _placeholder(value):
                found.append(f"{where}: 'greater' or 'less', the expected trend")
            if (observable, name) not in pairs:
                found.append(f"{where}: no pair of indicators.pairs reads it")
    for name, value in ind["direction"].items():
        read = [by_observable.get(o, {}).get(name, value) for o, paired in pairs if paired == name]
        if value not in read and not any(map(_placeholder, (value, *read))):
            found.append(f"indicators.direction.{name}: no pair of indicators.pairs reads it")
    trend = {h["name"]: h["direction"] for h in doc["hypotheses"]}.get(TREND)
    level = by_observable.get("surprise", {}).get("level", ind["direction"].get("level"))
    if (
        trend is not None
        and level is not None
        and trend != level
        and not (_placeholder(trend) or _placeholder(level))
    ):
        found.append(f"hypotheses: {TREND} says {trend}, but the level is scored as {level}")
    return found


def _latent_problems(doc: dict[str, Any]) -> list[str]:
    """What a file with a [latents] section can have wrong beyond the checks every file takes."""
    found = [
        f"{key}: nothing reads this key in a file with [latents]; take it out"
        for key in _unread(doc, _LATENT_KEYS)
    ]
    ind, latents, dyn = doc["indicators"], doc["latents"], doc["fss"]["dynamic"]
    columns = set(latents["columns"])
    if latents["table"] != "latents":
        found.append("latents.table: 'latents', the folder extract writes under each run")
    if not columns or not columns <= LATENT_COLUMNS:
        found.append(f"latents.columns: a non-empty subset of {sorted(LATENT_COLUMNS)}")
    if not set(ind["observables"]) <= columns:
        found.append("indicators.observables: must be latents.columns")
    if latents["extraction"] != EXTRACTION:
        found.append(f"latents.extraction: {EXTRACTION!r}, the rule each table is held to")
    if doc["data"]["staircase"]["temperature"] != _TC_PLUS["tc_plateau"]:
        found.append(
            f"data.staircase.temperature: {_TC_PLUS['tc_plateau']}, as extraction keeps it"
        )
    found += _indicator_problems(doc) + _probe_problems(doc) + _margin_problems(doc)
    entries = latents["checkpoints"]
    for i, entry in enumerate(entries):
        if entry["role"] not in CHECKPOINT_ROLES:
            found.append(f"latents.checkpoints[{i}].role: one of {sorted(CHECKPOINT_ROLES)}")
    if not any(entry["role"] == "primary" for entry in entries):
        found.append("latents.checkpoints: at least one primary")
    seeds = [entry["model_seed"] for entry in entries]
    if len(set(seeds)) != len(seeds):
        found.append("latents.checkpoints: one entry per model_seed")
    named = [entry["training"] for entry in entries if not _placeholder(entry["training"])]
    if len(set(named)) != len(named):
        found.append("latents.checkpoints: one entry per training")
    trained = {*latents["training_seeds"], latents["validation_seed"]}
    if trained & set(doc["data"]["seeds"]):
        found.append(
            "latents.training_seeds: a simulator seed this file scores is never trained on"
        )
    if doc["scoring"]["surrogates"]["n_ablation"] < 1:
        found.append("scoring.surrogates.n_ablation: the surrogates per ablation checkpoint")
    if not _SLUG.fullmatch(doc["baseline"]["prereg"]):
        found.append("baseline.prereg: the <id>-v<k> whose runs this file reads")
    if not dyn["statistics"]:
        found.append("fss.dynamic.statistics: at least one")
    if dyn["edge_margin"] != "window":
        found.append("fss.dynamic.edge_margin: 'window', one headline window of ramp (#115)")
    if dyn["edge_rule"] != EDGE_RULE:
        found.append(f"fss.dynamic.edge_rule: {EDGE_RULE!r}, the rule the scorer carries out")
    for i, entry in enumerate(dyn["statistics"]):
        where = f"fss.dynamic.statistics[{i}]"
        if entry["observable"] not in columns:
            found.append(f"{where}.observable: one of latents.columns")
        if entry["indicator"] not in LATENT_INDICATORS:
            found.append(f"{where}.indicator: one of {sorted(LATENT_INDICATORS)}")
        if entry["extremum"] not in EXTREMA and not _placeholder(entry["extremum"]):
            found.append(f"{where}.extremum: one of {sorted(EXTREMA)}")
        if entry["role"] not in DYNAMIC_ROLES:
            found.append(f"{where}.role: one of {sorted(DYNAMIC_ROLES)}")
        if entry.get("judged", "both") not in JUDGED_ON:
            found.append(f"{where}.judged: one of {sorted(JUDGED_ON)}")
        if entry["role"] == "descriptive" and "judged" in entry:
            found.append(f"{where}: a descriptive statistic is judged by nothing it states")
        if entry["role"] == "test" and entry.get("judged") == "twin":
            found.append(f"{where}.judged: 'both' for a test, whose trained verdicts margins read")
    used = {*ind["observables"], *(entry["observable"] for entry in dyn["statistics"])}
    for i, row in enumerate(doc["fss"]["static"]):
        tolerance = row.get("tolerance", 0.0)
        number = isinstance(tolerance, _NUMBER) and not isinstance(tolerance, bool)
        if not number and not _placeholder(tolerance):
            found.append(f"fss.static[{i}].tolerance: a number")
        used |= {c for c in columns if row["quantity"].endswith(f"_{c}")}
        if row["role"] == "descriptive" and {"expected", "tolerance", "judged"} & row.keys():
            found.append(f"fss.static[{i}]: a descriptive row is judged by nothing it states")
        if row.get("judged", "both") not in JUDGED_ON:
            found.append(f"fss.static[{i}].judged: one of {sorted(JUDGED_ON)}")
        if row["role"] == "reference" and "expected" in row:
            found.append(f"fss.static[{i}].expected: a reference is judged against 0 alone")
    if columns - used:
        found.append(f"latents.columns: {sorted(columns - used)} scored by nothing in this file")
    return found


def _ising_problems(doc: dict[str, Any], strict: bool = False) -> list[str]:
    """Everything wrong with a pre-registration document; strict also refuses placeholders."""
    found: list[str] = []
    latent = "latents" in doc
    _check(doc, _LATENT_SCHEMA if latent else _SCHEMA, "", found)
    if found:
        return found
    if not _SLUG.fullmatch(doc["id"]) and not _PLACEHOLDER.search(doc["id"]):
        found.append(f"id: {doc['id']!r} is not a lowercase slug")
    if doc["version"] < 1:
        found.append("version: must be >= 1")
    if doc["supersedes"] and not doc["reason"].strip():
        found.append("reason: an amendment must say why it supersedes " + doc["supersedes"])
    if not doc["hypotheses"]:
        found.append("hypotheses: at least one")
    for i, h in enumerate(doc["hypotheses"]):
        if h["direction"] not in DIRECTIONS and not (latent and _placeholder(h["direction"])):
            found.append(f"hypotheses[{i}].direction: one of {sorted(DIRECTIONS)}")
    ind, data, fss = doc["indicators"], doc["data"], doc["fss"]
    for key, allowed in (
        ("names", LATENT_INDICATORS if latent else INDICATORS),
        ("observables", LATENT_COLUMNS if latent else OBSERVABLES),
        ("cadences", {"matched"} if latent else CADENCES),  # a latents table holds saved frames
    ):
        if not ind[key] or not set(ind[key]) <= allowed:
            found.append(f"indicators.{key}: a non-empty subset of {sorted(allowed)}")
    if ind["detrend"] not in {"gaussian", "none"}:
        found.append("indicators.detrend: 'gaussian' or 'none'")
    if len(data["L"]) < 2:
        found.append("data.L: at least two sizes; a finite-size fit needs them")
    if not data["seeds"]:
        found.append("data.seeds: at least one")
    directions = ind["direction"]
    for name in ind["names"]:
        value = directions.get(name)
        if value not in ("greater", "less") and not (latent and _placeholder(value)):
            found.append(f"indicators.direction.{name}: 'greater' or 'less', the expected trend")
    if (data["ramps"]["kind"], data["nulls"]["kind"]) != ("ramp", "fixed"):
        found.append("data: ramps are kind 'ramp' and nulls kind 'fixed'")
    window = data["ramps"].get("eps_L")  # optional: without it, any window is scored
    window_ok = (
        isinstance(window, list)
        and len(window) == 2
        and all(isinstance(e, _NUMBER) and not isinstance(e, bool) for e in window)
        and window[0] > window[1]
    )
    if window is not None and not window_ok:
        found.append("data.ramps.eps_L: [start, end] in eps_L, start above end (a cooling ramp)")
    if data["tau_c"]["source"] != "pilot":
        found.append("data.tau_c.source: 'pilot', tau_c as measured per L (never fitted)")
    quantities, roles = QUANTITIES, ROLES
    if latent:
        columns = doc["latents"]["columns"]
        quantities = {f"{stat}_{column}" for stat in LATENT_STATISTICS for column in columns}
        roles = LATENT_ROLES
    for i, row in enumerate(fss["static"]):
        if row["quantity"] not in quantities or row["role"] not in roles:
            found.append(
                f"fss.static[{i}]: quantity in {sorted(quantities)}, role in {sorted(roles)}"
            )
        if (
            row["role"] in ("positive-control", "pipeline-check")
            and not {"expected", "tolerance"} <= row.keys()
        ):
            judged = row["role"].replace("-", " ")
            found.append(f"fss.static[{i}]: a {judged} needs expected and tolerance")
        if row["role"] == "reference" and "tolerance" not in row:
            found.append(f"fss.static[{i}]: a reference needs its own tolerance")
    dyn = fss["dynamic"]
    if latent:
        found += _latent_problems(doc)
    else:
        if dyn["statistic"] not in STATISTICS:
            found.append(f"fss.dynamic.statistic: one of {sorted(STATISTICS)}")
        if not set(dyn["indicators"]) <= set(ind["names"]):
            found.append("fss.dynamic.indicators: must be scored indicators")
        if dyn["observable"] not in ind["observables"]:
            found.append("fss.dynamic.observable: must be a scored observable")
        if not set(doc["scoring"]["alarms"]["indicators"]) <= set(ind["names"]):
            found.append("scoring.alarms.indicators: must be scored indicators")
        if "edge_margin" in dyn:
            found.append(
                "fss.dynamic.edge_margin: a file with no [latents] reads the band's first and last "
                "samples alone, as ising-ews-v1 froze it"
            )
    if dyn["cadence"] not in ind["cadences"]:
        found.append("fss.dynamic.cadence: one of indicators.cadences, the verdict's cadence")
    if not 0 < dyn["min_draw_share"] <= 1:
        found.append("fss.dynamic.min_draw_share: a share of the bootstrap draws, in (0, 1]")
    search = dyn["search"]
    if len(search) != 2 or search[0] <= search[1]:
        found.append("fss.dynamic.search: [from eps_L, to eps_L], the hot edge first")
    elif (
        window_ok
        and isinstance(window, list)  # window_ok says so; pyright needs it said here
        and not (window[0] >= search[0] and window[1] <= search[1])
    ):
        found.append("fss.dynamic.search: must lie inside data.ramps.eps_L, the window ramps run")
    sens = doc["sensitivity"]
    if (
        sens["headline"]["window"] not in sens["window"]
        or sens["headline"]["bandwidth"] not in sens["bandwidth"]
    ):
        found.append("sensitivity.headline: must be a cell of the grid")
    if len(doc["scoring"]["alarms"]["band"]) != 2:
        found.append("scoring.alarms.band: [from eps_L, to eps_L]")
    tests = doc["scoring"]["surrogates"]
    if tests["cells"] not in {"headline", "headline-h0", "all"}:
        found.append("scoring.surrogates.cells: 'headline', 'headline-h0' or 'all'")
    elif tests["cells"] == "headline-h0" and 0 not in doc["event"]["horizons"]:
        found.append(
            "scoring.surrogates.cells: 'headline-h0' tests h = 0 alone, which event.horizons lacks"
        )
    for key in ("primary", "secondary"):
        if tests[key] not in SURROGATES:
            found.append(f"scoring.surrogates.{key}: one of {sorted(SURROGATES)}")
    if ind["detrend"] == "none" and "ar1" in (tests["primary"], tests["secondary"]):
        found.append(
            "scoring.surrogates: ar1 needs detrending; fitted to a trending series it draws "
            "near-random walks with no trend"
        )
    if strict:
        found += [
            f"{where}: placeholder {text!r}"
            for where, text in _strings(doc)
            if _PLACEHOLDER.search(text)
        ]
        if not _SHA256.fullmatch(data["tau_c"]["pilot_sha256"]):
            found.append("data.tau_c.pilot_sha256: the pilot record's sha256 (64 hex digits)")
        pins = data["staircase"].get("tau_sha256")
        unpinned = [
            L
            for L in data["L"]
            if not (isinstance(pins, dict) and _SHA256.fullmatch(str(pins.get(str(L)))))
        ]
        if unpinned:
            found.append(
                "data.staircase.tau_sha256: per L, the sha256 of the tau.json the staircases' "
                f"dwell is read from (64 hex digits); none for L = {unpinned}"
            )
        if doc["trials"]["budget"] < cells(doc):
            found.append(f"trials.budget: one scoring run evaluates {cells(doc)} cells")
        for i, entry in enumerate(doc["latents"]["checkpoints"] if latent else ()):
            for key in ("trained_sha256", "twin_sha256"):
                if not _SHA256.fullmatch(entry[key]):
                    found.append(f"latents.checkpoints[{i}].{key}: a sha256 (64 hex digits)")
    return found


def _market_problems(doc: dict[str, Any], strict: bool = False) -> list[str]:
    """Everything wrong with a market-labels document. A draft may leave a placeholder in any text,
    in nulls.rule and in horizon.smallest; strict refuses every placeholder."""
    found: list[str] = []
    _check(doc, _MARKET_SCHEMA, "", found)
    if found:
        return found
    found += [
        f"{key}: nothing reads this key in a market-labels file; take it out"
        for key in _unread(doc, _MARKET_SCHEMA)
    ]
    for table in ("labeller", "cascade"):  # TOML allows nan and inf, and nan passes every check
        found += [
            f"{table}.{key}: a finite number, not {value!r}"
            for key, value in doc[table].items()
            if isinstance(value, float) and not math.isfinite(value)
        ]
    if not _SLUG.fullmatch(doc["id"]) and not _PLACEHOLDER.search(doc["id"]):
        found.append(f"id: {doc['id']!r} is not a lowercase slug")
    if doc["version"] < 1:
        found.append("version: must be >= 1")
    if doc["supersedes"] and not doc["reason"].strip():
        found.append("reason: an amendment must say why it supersedes " + doc["supersedes"])
    symbols, span, lab = doc["data"]["symbols"], doc["data"]["span"], doc["labeller"]
    if not symbols or len(set(symbols)) != len(symbols) or not set(symbols) <= set(MARKET_SYMBOLS):
        found.append(
            f"data.symbols: distinct symbols, a non-empty subset of {list(MARKET_SYMBOLS)}"
        )
    days = [key for key, day in span.items() if isinstance(day, datetime)]
    if days:
        found += [f"data.span.{key}: a UTC day, with no time of day" for key in days]
    elif span["confirmatory"] != CONFIRMATORY:
        found.append(
            f"data.span.confirmatory: {CONFIRMATORY.isoformat()}, the first day the pilot never "
            "labelled and the market command refuses to label before the freeze"
        )
    elif (span["confirmatory"] - span["start"]).days <= lab["warmup_days"]:
        found.append(
            "data.span: start, then labeller.warmup_days of warm-up, must end before confirmatory; "
            "the pilot is the labelled days between"
        )
    elif span["confirmatory"] > span["end"]:
        found.append("data.span.end: on or after confirmatory, the first confirmatory day")
    if doc["bars"]["estimator"] not in RV_ESTIMATORS:
        found.append(f"bars.estimator: one of {list(RV_ESTIMATORS)}, what the labels compute")
    if doc["bars"]["missing"] != MISSING:
        found.append(f"bars.missing: {MISSING!r}, the rule the labeller carries out")
    for key in ("warmup_days", "refit_days", "max_iter", "max_iter_warm", "clean", "refractory"):
        if lab[key] < 1:
            found.append(f"labeller.{key}: at least 1")
    for key in ("tol", "min_sigma", "min_gap"):
        if lab[key] <= 0:
            found.append(f"labeller.{key}: positive")
    if not 0 < lab["start_low"] < lab["start_high"] < 100:
        found.append("labeller.start_low, start_high: percentiles of z, 0 < low < high < 100")
    if not 0 < lab["start_p"] < 1:
        found.append("labeller.start_p: a probability, in (0, 1)")
    if not 0 < lab["min_share"] < 0.5:
        found.append("labeller.min_share: in (0, 0.5), since two states cannot both pass more")
    if lab["d_min"] not in D_MIN:
        found.append(f"labeller.d_min: one of {list(D_MIN)}, the declared range")
    if (lab["enter"], lab["exit"]) not in THRESHOLDS:
        found.append(f"labeller.enter, exit: one of {list(THRESHOLDS)}, the declared range")
    if lab["baseline_days"] not in BASELINE_DAYS:
        found.append(f"labeller.baseline_days: one of {list(BASELINE_DAYS)}, the declared range")
    for key, allowed in (("clean", CLEAN), ("refractory", REFRACTORY)):
        if lab[key] not in allowed:
            found.append(
                f"labeller.{key}: one of {list(allowed)}; another value needs an amendment"
            )
    rule = doc["nulls"]["rule"]
    if rule not in NULL_RULES and not _placeholder(rule):
        found.append(f"nulls.rule: one of {list(NULL_RULES)}, the rules the labeller carries out")
    cas = doc["cascade"]
    for key in ("a", "b", "c", "v", "window", "trailing_days"):
        if cas[key] <= 0:
            found.append(f"cascade.{key}: positive")
    for key in ("merge_hours", "both_hours", "bounds_hours"):
        if cas[key] < 1:
            found.append(f"cascade.{key}: at least 1")
    if cas["c"] >= 1:
        found.append("cascade.c: below 1, since the flow imbalance lies in [-1, 1]")
    if not 0 < cas["onset_percentile"] < 100:
        found.append("cascade.onset_percentile: a percentile, in (0, 100)")
    smallest = doc["horizon"]["smallest"]
    if not _placeholder(smallest) and (isinstance(smallest, str) or smallest < 1):
        found.append("horizon.smallest: a number of bars, at least 1")
    if strict:
        found += [
            f"{where}: placeholder {text!r}"
            for where, text in _strings(doc)
            if _PLACEHOLDER.search(text)
        ]
    return found


def _numbers(value: Any, where: str = "") -> list[tuple[str, float]]:
    if isinstance(value, dict):
        return [n for k, v in value.items() for n in _numbers(v, f"{where}.{k}".lstrip("."))]
    if isinstance(value, list):
        return [n for i, v in enumerate(value) for n in _numbers(v, f"{where}[{i}]")]
    return [(where, value)] if isinstance(value, float) else []


def market_ews_cells(doc: dict[str, Any]) -> int:
    """AUC evaluations one scoring of a market-ews file makes, one trial each: per indicator and
    horizon, one per reporting group, per-asset counting once per symbol."""
    groups = doc["reporting"]["groups"]
    per = sum(len(doc["data"]["symbols"]) if g == "per-asset" else 1 for g in groups)
    return len(doc["indicators"]["names"]) * len(doc["segments"]["horizons"]) * per


def _market_ews_problems(doc: dict[str, Any], strict: bool = False) -> list[str]:
    """Everything wrong with a market-ews document (#30 M5). A draft may leave a placeholder in
    any text, in indicators.detrend, in scoring.floor's fit, noise and empty, and in
    alarms.far_per_day; strict refuses every placeholder. Each rule the segment builder carries
    out is held word for word, and each choice to the options built."""
    found: list[str] = []
    _check(doc, _MARKET_EWS_SCHEMA, "", found)
    if found:
        return found
    found += [
        f"{key}: nothing reads this key in a market-ews file; take it out"
        for key in _unread(doc, _MARKET_EWS_KEYS)
    ]
    found += [  # TOML allows nan and inf, and nan passes every check
        f"{where}: a finite number, not {value!r}"
        for where, value in _numbers(doc)
        if not math.isfinite(value)
    ]
    if found:
        return found
    if not _SLUG.fullmatch(doc["id"]) and not _PLACEHOLDER.search(doc["id"]):
        found.append(f"id: {doc['id']!r} is not a lowercase slug")
    if doc["version"] < 1:
        found.append("version: must be >= 1")
    if doc["supersedes"] and not doc["reason"].strip():
        found.append("reason: an amendment must say why it supersedes " + doc["supersedes"])
    data, ind, seg = doc["data"], doc["indicators"], doc["segments"]
    symbols = data["symbols"]
    if not symbols or len(set(symbols)) != len(symbols) or not set(symbols) <= set(MARKET_SYMBOLS):
        found.append(
            f"data.symbols: distinct symbols, a non-empty subset of {list(MARKET_SYMBOLS)}"
        )
    for key, rule in (("series", SERIES), ("missing", MISSING_MINUTE)):
        if data[key] != rule:
            found.append(f"data.{key}: {rule!r}, the rule the segment builder carries out")
    if not _LABELS_TAG.fullmatch(data["labels"]["tag"]):
        found.append("data.labels.tag: prereg/market-labels-v<k>, the frozen labels' tag")
    if not _SHA256.fullmatch(data["labels"]["sha256"]):
        found.append("data.labels.sha256: the frozen labels file's sha256 (64 hex digits)")
    if data["events"] != EVENT_SETS:
        found.append(f"data.events: {EVENT_SETS}, regime shifts primary, cascades a stress test")
    names = ind["names"]
    if not names or len(set(names)) != len(names) or not set(names) <= set(EWS_INDICATORS):
        found.append(f"indicators.names: distinct, a non-empty subset of {list(EWS_INDICATORS)}")
    for key in ("window", "length"):
        if ind[key] < 2:
            found.append(f"indicators.{key}: at least 2")
    if ind["lag"] != 1:
        found.append("indicators.lag: 1, AC1's lag in minutes")
    detrend, bandwidth = ind["detrend"], ind.get("bandwidth")
    if detrend not in DETRENDS and not _placeholder(detrend):
        found.append(f"indicators.detrend: one of {list(DETRENDS)}")
    if detrend == "gaussian" and not (
        isinstance(bandwidth, _NUMBER) and not isinstance(bandwidth, bool) and bandwidth > 0
    ):
        found.append("indicators.bandwidth: a positive number of minutes, with detrend 'gaussian'")
    if detrend != "gaussian" and bandwidth is not None:
        found.append("indicators.bandwidth: only with detrend 'gaussian'; nothing else reads it")
    hyp = doc["hypotheses"]
    if sorted(h["indicator"] for h in hyp) != sorted(names):
        found.append("hypotheses: one per scored indicator, naming it in `indicator`")
    for i, h in enumerate(hyp):
        if h["direction"] not in ("greater", "less"):
            found.append(f"hypotheses[{i}].direction: 'greater' or 'less', the expected trend")
    season = doc["seasonality"]
    if season["rule"] != DESEASON:
        found.append(f"seasonality.rule: {DESEASON!r}, the rule the scorer carries out")
    fit = season["fit"]
    if any(isinstance(day, datetime) for day in fit.values()):
        found.append("seasonality.fit: UTC days, with no time of day")
    elif not fit["start"] <= fit["end"] < CONFIRMATORY:
        found.append(
            f"seasonality.fit: start on or before end, and end before {CONFIRMATORY.isoformat()}: "
            "the slot profile is fitted on pilot days only"
        )
    if not set(season["strata"]) <= set(STRATA):
        found.append(f"seasonality.strata: a subset of {list(STRATA)}")
    horizons = seg["horizons"]
    if not horizons or len(set(horizons)) != len(horizons) or min(horizons) < HORIZON_FLOOR:
        found.append(
            f"segments.horizons: distinct numbers of bars, each at least {HORIZON_FLOOR}, the "
            "labels' p90 detection delay"
        )
    for key, rule in (("rule", SEGMENT), ("controls", CONTROLS)):
        if seg[key] != rule:
            found.append(f"segments.{key}: {rule!r}, the rule the segment builder carries out")
    for key in ("subsample", "seed"):
        if seg[key] < 0:
            found.append(f"segments.{key}: at least 0")
    sc = doc["scoring"]
    if not 0 < sc["ci_level"] < 1:
        found.append("scoring.ci_level: in (0, 1)")
    if sc["bootstrap"] < 1:
        found.append("scoring.bootstrap: at least 1 draw")
    if sc["seed"] < 0:
        found.append("scoring.seed: at least 0")
    floor = sc["floor"]
    for key, built in (("fit", FLOOR_FITS), ("noise", FLOOR_NOISE), ("empty", FLOOR_EMPTY)):
        if floor[key] not in built and not _placeholder(floor[key]):
            found.append(f"scoring.floor.{key}: one of {list(built)}, the options built")
    if floor["paths"] < 1 or floor["seed"] < 0:
        found.append("scoring.floor: at least 1 path, and a seed of at least 0")
    last = (CONFIRMATORY - FLOOR_FIRST).days
    if not 1 <= floor["days"] <= last:
        found.append(
            f"scoring.floor.days: 1 to {last}, so that the simulated days from "
            f"{FLOOR_FIRST.isoformat()} end before {CONFIRMATORY.isoformat()}"
        )
    garch = floor["garch"]
    if sorted(garch) != sorted(symbols):
        found.append("scoring.floor.garch: one table per symbol of data.symbols")
    for symbol, params in garch.items():
        where = f"scoring.floor.garch.{symbol}"
        numbers = isinstance(params, dict) and sorted(params) == sorted(GARCH_FIELDS)
        if not numbers or not all(
            isinstance(v, _NUMBER) and not isinstance(v, bool) for v in params.values()
        ):
            found.append(f"{where}: the numbers {list(GARCH_FIELDS)}, per minute")
            continue
        omega, alpha, beta, nu = (params[k] for k in GARCH_FIELDS)
        if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1 and nu > 2):
            found.append(f"{where}: omega > 0, alpha and beta >= 0, alpha + beta < 1, nu > 2")
        elif floor["fit"] == "pilot" and (omega, alpha, beta, nu) != PILOT_GARCH.get(symbol):
            found.append(f"{where}: the pilot fit, {PILOT_GARCH.get(symbol)}, with fit 'pilot'")
    shift = sc["shift"]
    if shift["draws"] < 1:
        found.append("scoring.shift.draws: at least 1")
    if shift["min_offset_bars"] < DAY_BARS:
        found.append(f"scoring.shift.min_offset_bars: at least {DAY_BARS}, a day")
    if sc["surrogates"]["secondary"] != "phase":
        found.append("scoring.surrogates.secondary: 'phase', the secondary null (#29 call 7)")
    if sc["surrogates"]["n"] < 1:
        found.append("scoring.surrogates.n: at least 1")
    groups = doc["reporting"]["groups"]
    known = set(groups) <= set(REPORT_GROUPS) and len(set(groups)) == len(groups)
    if not known or not {"pooled", "per-asset"} <= set(groups):
        found.append("reporting.groups: pooled and per-asset, and may add eth-only")
    if "eth-only" in groups and set(symbols) != set(MARKET_SYMBOLS):
        found.append("reporting.groups: eth-only reads ETH onsets against BTC's: both symbols")
    alarms = doc["alarms"]
    if alarms["rule"] != ALARMS:
        found.append(f"alarms.rule: {ALARMS!r}, the rule the alarm table carries out")
    far = alarms["far_per_day"]
    if not _placeholder(far) and (isinstance(far, str) or not far > 0):
        found.append("alarms.far_per_day: a positive number of false alarms per day")
    if doc["trials"]["budget"] < 1:
        found.append("trials.budget: at least 1")
    if strict:
        found += [
            f"{where}: placeholder {text!r}"
            for where, text in _strings(doc)
            if _PLACEHOLDER.search(text)
        ]
        if not found and doc["trials"]["budget"] < market_ews_cells(doc):
            found.append(f"trials.budget: one scoring run evaluates {market_ews_cells(doc)} cells")
    return found


def problems(doc: dict[str, Any], strict: bool = False) -> list[str]:
    """Everything wrong with a pre-registration document, checked against the schema of its
    data.domain; strict also refuses placeholders. A file with no domain, or with one that is not
    a string, is checked as an Ising file, whose schema reports it."""
    data = doc.get("data")
    domain = data.get("domain") if isinstance(data, dict) else None
    if not isinstance(domain, str) or domain == "ising":
        return _ising_problems(doc, strict)
    if domain == MARKET_LABELS:
        return _market_problems(doc, strict)
    if domain == MARKET_EWS:
        return _market_ews_problems(doc, strict)
    return [f"data.domain: {domain!r} is not one of {list(DOMAINS)}"]


def load(path: Path) -> Prereg:
    path = Path(path)
    raw = path.read_bytes()
    try:
        doc = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PreregError(f"{path}: {exc}") from exc
    found = problems(doc)
    if found:
        raise PreregError(f"{path}:\n  " + "\n  ".join(found))
    return Prereg(path, doc, hashlib.sha256(raw).hexdigest())


def _git(root: Path, *args: str, binary: bool = False) -> Any:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if done.returncode != 0:
        raise PreregError(f"git {' '.join(args)}: {done.stderr.decode(errors='replace').strip()}")
    return done.stdout if binary else done.stdout.decode().strip()


def repo_root(path: Path) -> Path:
    start = path if path.is_dir() else path.parent
    return Path(_git(start, "rev-parse", "--show-toplevel"))


def latest(root: Path, prereg_id: str) -> Path:
    """The highest version of paper/prereg/<id>-v<k>.toml."""
    pattern = re.compile(re.escape(prereg_id) + r"-v(\d+)\.toml")
    versions = {
        int(m.group(1)): p
        for p in (root / PREREG_DIR).glob(f"{prereg_id}-v*.toml")
        if (m := pattern.fullmatch(p.name))
    }
    if not versions:
        raise PreregError(f"no {PREREG_DIR.as_posix()}/{prereg_id}-v<k>.toml under {root}")
    return versions[max(versions)]


def freeze(root: Path, prereg_id: str) -> str:
    """Tag the commit holding the latest version of a pre-registration; returns the tag."""
    path = latest(root, prereg_id)
    prereg = load(path)
    relative = path.relative_to(root).as_posix()
    if prereg.name != path.stem:
        raise PreregError(
            f"{relative}: holds {prereg.name}, so it must be named {prereg.name}.toml"
        )
    found = problems(prereg.doc, strict=True)
    if found:
        raise PreregError(f"{relative} is not ready to freeze:\n  " + "\n  ".join(found))
    if not _git(root, "ls-files", "--", relative):
        raise PreregError(f"{relative} is not committed")
    if _git(root, "status", "--porcelain", "--", relative):
        raise PreregError(f"{relative} differs from HEAD: commit it before the freeze")
    if _git(root, "tag", "--list", prereg.tag):
        raise PreregError(f"{prereg.tag} exists: a frozen version is never re-tagged; amend it")
    baseline_commit(prereg)  # the runs it reads must already be fixed
    message = f"Pre-registration {prereg.name}, sha256 {prereg.sha256}"
    _git(root, "tag", "-a", prereg.tag, "-m", message, "HEAD")
    return prereg.tag


@dataclass(frozen=True)
class Frozen:
    tag: str
    commit: str


def frozen(prereg: Prereg) -> Frozen:
    """The prereg's tag and commit, if the file is byte for byte the blob frozen at its tag."""
    root = repo_root(prereg.path)
    relative = prereg.path.resolve().relative_to(root.resolve()).as_posix()
    if not _git(root, "tag", "--list", prereg.tag):
        command = f"python -m tipping_point.ews prereg freeze {prereg.id}"
        raise PreregError(f"{prereg.name} is not frozen: run `{command}`")
    blob = _git(root, "cat-file", "blob", f"{prereg.tag}:{relative}", binary=True)
    if hashlib.sha256(blob).hexdigest() != prereg.sha256:
        raise PreregError(f"{relative} differs from the version frozen at {prereg.tag}")
    return Frozen(prereg.tag, _git(root, "rev-parse", f"{prereg.tag}^{{commit}}"))


def baseline_commit(prereg: Prereg) -> str | None:
    """The commit of the tag of the version a file's [baseline] names, whose runs it reads and which
    therefore made them; None for a file with no [baseline]."""
    baseline = prereg.doc.get("baseline")
    if baseline is None:
        return None
    root, tag = repo_root(prereg.path), f"prereg/{baseline['prereg']}"
    if not _git(root, "tag", "--list", tag):
        raise PreregError(f"{prereg.name} reads the runs of {baseline['prereg']}: no tag {tag}")
    return _git(root, "rev-parse", f"{tag}^{{commit}}")


def stamp(prereg: Prereg, freeze_ref: Frozen) -> dict[str, Any]:
    return {
        "id": prereg.id,
        "version": prereg.version,
        "sha256": prereg.sha256,
        "tag": freeze_ref.tag,
        "commit": freeze_ref.commit,
    }


def code_check(
    prereg: Prereg, freeze_ref: Frozen, src: Path, held: Sequence[str], clean: Sequence[str]
) -> dict[str, Any]:
    """What a run that reads a frozen file records of the code it runs, as paths in the repository:
    the files `held` (relative to the folder `src`), which make its output, and `clean`, the
    others it runs, such as its command line. It refuses unless `changed` and `uncommitted` are
    empty: every held file must be its blob at the frozen commit and every clean one its blob at
    HEAD, which `head` then pins. That is how it runs "from the tagged commit" without checking
    the tag out. Files beside these, new ones included, are not checked, since the run reads none
    of them. The record also holds the sha256 of the repository's uv.lock, which pins the
    libraries."""
    root = repo_root(prereg.path)
    try:
        base = Path(src).resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PreregError(f"{src} is not inside {root}, where {prereg.name} is frozen") from exc
    lock = root / "uv.lock"
    if not lock.is_file():
        raise PreregError(f"no uv.lock in {root}: the run records its sha256 to pin the libraries")
    held_paths = [f"{base}/{name}" for name in held]
    clean_paths = [f"{base}/{name}" for name in clean]
    return {
        "head": _git(root, "rev-parse", "HEAD"),
        "held": held_paths,
        "clean": clean_paths,
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "changed": _not_at(root, freeze_ref.commit, held_paths),
        "uncommitted": _not_at(root, "HEAD", clean_paths),
    }


def _not_at(root: Path, commit: str, paths: Sequence[str]) -> list[str]:
    """Those of `paths` (relative to `root`) whose file in the working tree is not their blob at
    `commit`. Each file's bytes are read and hashed as `git add` would store them, so no stat the
    index caches can hide an edit, and git's line-end rule applies: a file checked out with CRLF
    is still its LF blob. A path missing from either side is not at the commit."""
    listed = _git(root, "ls-tree", "-z", commit, "--", *paths).split("\0")
    blobs = {path: meta.split()[2] for meta, _, path in (e.partition("\t") for e in listed if e)}
    present = [p for p in paths if (root / p).is_file()]
    local = dict(zip(present, _git(root, "hash-object", "--", *present).split(), strict=True))
    return [p for p in paths if p not in local or local[p] != blobs.get(p)]


def trials_path(data_dir: Path) -> Path:
    return Path(data_dir) / "ews" / "trials.jsonl"


def log_trials(data_dir: Path, records: Sequence[dict[str, Any]]) -> None:
    """Append evaluations, one line each, in one write, flushed and synced before returning; the log
    only ever grows."""
    path = trials_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time_ns()
    lines = [json.dumps({"logged_ns": now, **record}, sort_keys=True) + "\n" for record in records]
    with open(path, "ab") as f:
        f.write("".join(lines).encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def log_trial(data_dir: Path, record: dict[str, Any]) -> None:
    log_trials(data_dir, [record])


def trials(data_dir: Path, prereg: Prereg, result: str | None = None) -> int:
    """Evaluations logged so far under this prereg's id and version; with `result`, those naming
    that result folder alone. A line that is no whole record, as a power cut inside log_trials' one
    write leaves, is refused: no count can rest on it. Its repair replaces it, never cuts it: a cut
    would leave a version whose first record tore with no line and its budget back, and with its
    folder deleted, as charge offers, it could be scored again. A blank line is skipped, since that
    repair is a hand edit; a zeroed tail is not blank."""
    path = trials_path(data_dir)
    if not path.exists():
        return 0
    own, count = (prereg.id, prereg.version), 0
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            stamp = record["prereg"]
            mine = (stamp["id"], stamp["version"]) == own
        except (ValueError, KeyError, TypeError) as exc:  # ValueError: bad JSON, or bad UTF-8
            raise PreregError(
                f"{path}, line {number}: not a whole trial record, a write torn by a crash. The "
                f"run that was writing it is the newest folder in {path.parent} with no "
                "complete.json, its tables on disk: count it as spent and amend the "
                "pre-registration. Every count reads this whole log, the amendment's too, so "
                "first, keeping a copy of the log, replace this line with one whole record ending "
                'in a newline: {"prereg": <params.prereg from the manifest.json of that folder>}. '
                "Do not cut the line: a version with no line left would get its budget back. If "
                "no such folder is left, nothing here says whose line it was: write "
                '{"prereg": {"id": "<id>", "version": <k>}} for the file that was being scored, '
                "or, if that cannot be told, one such record for each file it could have been"
            ) from exc
        count += mine and (result is None or record.get("result") == result)
    return count


def _torn_results(data_dir: Path, prereg: Prereg) -> list[str]:
    """The folders of <data dir>/ews stamped with this prereg's id and version and with no
    complete.json: scorings that died before finish_result wrote it. The manifest's stamp decides,
    since a folder's name ends alike under another id (x-demo-v1, demo-v1). A folder with no
    manifest yet holds no table either: begin_result writes that first."""
    own, found = (prereg.id, prereg.version), []
    for manifest in sorted((Path(data_dir) / "ews").glob("*/manifest.json")):
        stamp = io.read_json(manifest)["params"]["prereg"]
        done = (manifest.parent / "complete.json").is_file()
        if (stamp["id"], stamp["version"]) == own and not done:
            found.append(manifest.parent.name)
    return found


def resumable(data_dir: Path, prereg: Prereg) -> str | None:
    """The torn result of this version that --resume carries on: the one there is, or None when
    there is none, or several, which charge then refuses."""
    torn = _torn_results(data_dir, prereg)
    return torn[0] if len(torn) == 1 else None


def charge(data_dir: Path, prereg: Prereg, planned: int, own: str | None = None) -> int:
    """Refuse a scoring whose `planned` cells would take this version past its budget, so that
    cell budget + 1 is never evaluated, or that would score past a torn result of this version,
    whose tables can be read with their cells unlogged; returns the trials logged so far by other
    results. `own` names the folder the scoring writes or carries on (begin_result): no torn result
    to it, and the trials it logged are its own cells, taken off the count. They are all of them or
    none, as a crash after the log or before it leaves them; some alone is a torn log's repair,
    which spends the folder, so it is refused. A torn log speaks first: its folder's trials are part
    charged, and deleting the folder is not its repair."""
    logged, budget = trials(data_dir, prereg), prereg.doc["trials"]["budget"]
    every = _torn_results(data_dir, prereg)
    torn = [name for name in every if name != own]
    if torn:
        one = len(every) == 1  # resumable carries on a torn result only when it is the one
        offer = "carry it on with --resume, which refuses one it cannot end as it began, or "
        raise PreregError(
            f"{prereg.name}: {', '.join(torn)} in {Path(data_dir) / 'ews'} has no complete.json: "
            "a torn result from a crash, or one being written right now. Its tables can be read "
            "whether or not its trials were logged, so nothing scores past it: once nothing writes "
            f"it, {offer if one else ''}delete it or amend the pre-registration. Deleting it gives "
            "back no trial already logged"
            + ("" if one else f". Only one torn folder can be carried on: {', '.join(every)}")
        )
    mine = trials(data_dir, prereg, own) if own else 0
    if mine not in (0, planned):
        raise PreregError(
            f"{prereg.name}: {own} has {mine} of its {planned} trials logged, which only a torn "
            "log's repair leaves: its trials are part spent, so it is not carried on. Count it as "
            "spent and amend the pre-registration"
        )
    logged -= mine
    if logged + planned > budget:
        raise PreregError(
            f"{prereg.name}: {logged} trials logged and {planned} more would exceed its budget of "
            f"{budget}; amend the pre-registration to score again"
        )
    return logged


def cells(doc: dict[str, Any]) -> int:
    """AUC evaluations one scoring run makes: one trial each. A file with a [latents] section makes
    them per checkpoint and per pair: each primary checkpoint and its twin on the whole grid, where
    an indicator in RAW has no bandwidth and so one cell per window, each ablation checkpoint and
    its twin at the headline cell alone."""
    ind, sens = doc["indicators"], doc["sensitivity"]
    sizes = len(doc["data"]["L"]) * len(doc["event"]["horizons"]) * len(ind["cadences"])
    grid = len(sens["window"]) * len(sens["bandwidth"])
    if "latents" not in doc:
        return len(ind["names"]) * len(ind["observables"]) * sizes * grid
    raw = sum(name in RAW for _, name in ind["pairs"])
    primary = (len(ind["pairs"]) - raw) * grid + raw * len(sens["window"])
    roles = [entry["role"] for entry in doc["latents"]["checkpoints"]]
    return 2 * sizes * sum(primary if role == "primary" else len(ind["pairs"]) for role in roles)


def _differences(began: dict[str, Any], key: dict[str, Any]) -> dict[str, str]:
    """Each part of a resume key that is not the one a result began under: both values, or of the
    inputs the runs whose differ."""
    found = {}
    for part in sorted(began.keys() | key.keys()):
        then, now = began.get(part), key.get(part)
        if then == now:
            continue
        if part == "inputs":
            then, now = then or {}, now or {}
            runs = sorted(r for r in then.keys() | now.keys() if then.get(r) != now.get(r))
            found[part] = f"the runs {', '.join(runs)}"
        else:
            found[part] = f"{then!r} when it began, {now!r} now"
    return found


def begin_result(
    data_dir: Path, prereg: Prereg, freeze_ref: Frozen, key: dict[str, Any], own: str | None = None
) -> Path:
    """A result's folder and manifest.json, before anything is scored into it: the prereg's stamp,
    the resume key, which is `key` and the commit of the running code when its checkout is clean,
    and a record of this start. With `own`, the torn result --resume carries on (resumable), that
    folder instead, given one more record: refused, naming what differs, unless it began under this
    very key, since only then can it end in the bytes an uninterrupted scoring would write. A new
    result with a key, of a frozen file, from a checkout no commit names is warned of on stderr:
    a resume of it would be refused."""
    code = io.code_version()
    if own is None and key and freeze_ref.commit and code["dirty"] is not False:
        print(
            "warning: this checkout has uncommitted changes, so a crash in this run cannot be "
            "resumed",
            file=sys.stderr,
        )
    key = key | {"code": code["git_sha"] if code["dirty"] is False else None}
    started = {
        "started_ns": time.time_ns(),
        "host": socket.gethostname(),
        **code,
        "pid": os.getpid(),
    }
    if own is not None:
        run_dir = Path(data_dir) / "ews" / own
        manifest = io.read_json(run_dir / "manifest.json")
        found = _differences(manifest.get("resume") or {}, key)
        if key["code"] is None:
            found["code"] = "this checkout has changes not committed, which no key can tell apart"
        if found:
            raise PreregError(
                f"{prereg.name}: {own} in {run_dir.parent} is not carried on, since it began "
                "otherwise and could not end in the bytes an uninterrupted scoring writes: "
                + "; ".join(f"{part}: {why}" for part, why in found.items())
                + ". Score it as it began, or delete it or amend the pre-registration"
            )
        manifest["resumptions"].append(started)
        io.write_json(run_dir / "manifest.json", manifest, replace=True)
        return run_dir
    run_id = engine.run_id(prereg.name)
    run_dir = Path(data_dir) / "ews" / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "format": 1,
        "domain": "ews",
        "run_id": run_id,
        "producer": {"name": "tipping_point.ews", "version": __version__, **code},
        "started_ns": started["started_ns"],
        "key": "cell",
        "tables": [],
        "params": {"prereg": stamp(prereg, freeze_ref)},
        "resume": key,
        "resumptions": [started],
    }
    io.write_json(run_dir / "manifest.json", manifest)
    return run_dir


def save_part(run_dir: Path | None, unit: str, part: Any) -> None:
    """A finished unit of a scoring (a checkpoint's pass, a probe) as <result>/parts/<unit>.pkl,
    whole or not at all, so that --resume skips it; nothing without a folder. Pickle, since a
    Parquet table takes its columns from its first row and a round trip can lose a value: the
    folder is the scorer's own, so what is read back is trusted."""
    if run_dir is not None:
        (run_dir / "parts").mkdir(exist_ok=True)
        io.write_file(run_dir / "parts" / f"{unit}.pkl", lambda f: pickle.dump(part, f))


def load_part(run_dir: Path | None, unit: str) -> Any:
    """What save_part kept of a unit, or None."""
    path = None if run_dir is None else run_dir / "parts" / f"{unit}.pkl"
    return pickle.loads(path.read_bytes()) if path is not None and path.is_file() else None


def finish_result(
    data_dir: Path,
    prereg: Prereg,
    run_dir: Path,
    tables: dict[str, list[dict[str, Any]]],
    inputs: list[dict[str, Any]],
    params: dict[str, Any],
    records: Sequence[dict[str, Any]] = (),
) -> Path:
    """The rest of a result begin_result made, in the shared layout: the manifest given its tables,
    inputs and params, each non-empty table as <table>/00000000000000000000.parquet keyed by `cell`,
    the trial `records` it evaluated logged in one write, each naming this result, and complete.json
    once all are synced. A crash so leaves a folder with no complete.json, its trials logged or not,
    or a complete one with all of them. Carried on, it writes the same bytes over what an earlier
    attempt left, and logs nothing if that attempt logged it all (charge allows all or none)."""
    filled = {name: rows for name, rows in tables.items() if rows}
    manifest = io.read_json(run_dir / "manifest.json")
    manifest["tables"] = list(filled)
    manifest["params"] = {"prereg": manifest["params"]["prereg"], "inputs": inputs, **params}
    io.write_json(run_dir / "manifest.json", manifest, replace=True)
    files: dict[str, str] = {}
    for name, rows in filled.items():
        table = pa.Table.from_pylist([{"cell": i, **row} for i, row in enumerate(rows)])
        path = run_dir / name / f"{0:020d}.parquet"
        path.parent.mkdir(exist_ok=True)
        io.write_file(
            path, lambda f, table=table: pq.write_table(table, f, compression="zstd"), replace=True
        )
        files[path.relative_to(run_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if records and not trials(data_dir, prereg, run_dir.name):
        log_trials(data_dir, [trial | {"result": run_dir.name} for trial in records])
    record = {
        "ended_ns": time.time_ns(),
        "last_key": max((len(rows) - 1 for rows in filled.values()), default=-1),
        "rows": {name: len(rows) for name, rows in filled.items()},
        "clean": True,
        "files": files,
    }
    io.write_json(run_dir / "complete.json", record)
    return run_dir


def write_result(
    data_dir: Path,
    prereg: Prereg,
    freeze_ref: Frozen,
    tables: dict[str, list[dict[str, Any]]],
    inputs: list[dict[str, Any]],
    params: dict[str, Any],
    records: Sequence[dict[str, Any]] = (),
) -> Path:
    """begin_result and finish_result at once: a result written whole, by a scoring that saves no
    part of itself on the way."""
    run_dir = begin_result(data_dir, prereg, freeze_ref, {})
    return finish_result(data_dir, prereg, run_dir, tables, inputs, params, records)
