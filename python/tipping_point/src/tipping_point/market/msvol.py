"""The regime labeller (#29 S2; research pass section 2, items 3-11) and the labels run.

Per symbol, from the 5-minute bars' rv (bars.py), y = ln rv, missing where rv is NaN or 0:

  x = y - m_c   m_c the mean of y over the earlier bars of slot c, one of 2 x 288 (weekday or
                weekend, by time of day)
  z = x - b     b the median of x over the `baseline_days` days ending at the bar's UTC midnight
  theta         a two-state Gaussian hidden Markov model on z: N(mu_s, sigma_s^2), p00, p11;
                state 1 has the larger mu (label matching after every M-step)
  P(t)          P(s_t = 1 | z_1..z_t) under the theta in force at t: filtered, never smoothed
  H(t)          1 once P >= enter, 0 once P <= exit, otherwise unchanged; 0 at the first label

Refits fall at the end of `warmup_days`, then at every `refit_days`-th UTC midnight counted from a
Monday (Mondays 00:00 by default). A refit renews the slot means from the bars before it and x, z
and the baselines from them, fits theta by EM on the z before it (from the deterministic start the
first time, warm from the theta in force after), and labels the bars that start from it until the
next refit (bars closing in (r_k, r_k+1], item 8) with a forward filter run from the series start;
a missing bar carries the prediction without an update. A degenerate fit (a sigma below
`min_sigma`, a state's smoothed share below `min_share`, mu1 - mu0 below `min_gap`, or a numerical
failure) is flagged, and the theta before it stays in force, read on the renewed z; before the
first good fit there are no labels. So each value at bar t comes from bars at or before t and is
never rewritten (tests/market/test_lookahead.py checks it bit for bit).

An episode is a maximal run of H = 1. One of at least `d_min` bars is an event when H = 0 through
the `clean` bars before its onset and no event began in the `refractory` bars before it; it is
confirmed `d_min` bars after its onset. A closed run shorter than `d_min` is a flicker, an open one
at the end of the data pending; the rest are plain episodes. Null stretches are the bars at least
`refractory` bars from every episode but flickers (pending ones included) and `clean` bars from
every flicker. History the labels do not cover counts as unclean: no event begins and no null
stretch lies within `refractory` bars of the first label, nor within `refractory` of the end.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .. import __version__, engine
from ..ising import io

BAR_NS = 5 * 60 * 10**9
DAY_NS = 86_400 * 10**9
DAY_BARS = 288
MONDAY = 4  # 1970-01-05, a Monday, as a day number
LOG_2PI = math.log(2 * math.pi)


@dataclass(frozen=True)
class Params:
    """Every number the labels pre-registration freezes, at the research pass's start values
    (#29, section 2; S5 chooses the final ones on the 2023 pilot). Bars are counted in bars."""

    warmup_days: int = 184  # item 7: the first fit comes after it (2022-07-01 -> 2023-01-01)
    refit_days: int = 7  # item 7: refit at every 7th UTC midnight from a Monday
    baseline_days: int = 28  # item 4
    start_low: float = 25.0  # item 6: the first start puts mu at these percentiles of z
    start_high: float = 75.0
    start_p: float = 0.95  # item 6: and p00 = p11 = start_p
    tol: float = 1e-6  # item 6: stop at a relative log-likelihood gain below tol
    max_iter: int = 200  # item 6: M-steps from the deterministic start
    max_iter_warm: int = 50  # item 6: M-steps from a warm start
    min_sigma: float = 0.05  # item 6, degeneracy guard
    min_share: float = 0.02  # item 6, degeneracy guard
    min_gap: float = 0.25  # item 6, degeneracy guard: mu1 - mu0
    enter: float = 0.8  # item 9
    exit: float = 0.2  # item 9
    d_min: int = 12  # item 10
    clean: int = 72  # item 10: C
    refractory: int = 288  # item 10: Q


class Theta(NamedTuple):
    mu0: float
    mu1: float
    sigma0: float
    sigma1: float
    p00: float
    p11: float


class Fit(NamedTuple):
    theta: Theta
    loglik: float
    iterations: int  # M-steps
    converged: bool
    shares: tuple[float, float]  # each state's smoothed share of the observed bars
    failure: str | None


@dataclass(frozen=True)
class Labels:
    """One symbol's labels, the arrays aligned with its bars."""

    x: np.ndarray
    z: np.ndarray
    p: np.ndarray  # P(high); NaN before the first theta
    h: np.ndarray  # int8 hysteresis state; -1 before the first label
    refit: np.ndarray  # int32: index in `refits` of the theta in force; -1 before the first
    refits: list[dict[str, Any]]  # bar, warm, observed, the fit, flagged, reason
    episodes: list[dict[str, Any]]  # onset, end (None while open), bars, kind, confirm, missing
    nulls: list[dict[str, Any]]  # start, end, bars, missing


def label(ts_ns: np.ndarray, rv: np.ndarray, params: Params | None = None) -> Labels:
    """One symbol's labels from its bars' rv (NaN = missing) on a gapless 5-minute grid that
    starts at a UTC midnight."""
    params = params or Params()
    n = len(ts_ns)
    if n and (ts_ns[0] % DAY_NS or np.any(np.diff(ts_ns) != BAR_NS)):
        raise ValueError("bars must lie on a gapless 5-minute grid starting at a UTC midnight")
    if not 0 < params.exit < params.enter < 1:
        raise ValueError(f"need 0 < exit < enter < 1, not exit {params.exit}, enter {params.enter}")
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(rv > 0, np.log(rv), np.nan)
    slot = _slots(ts_ns)
    x, z, p = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    refit_of = np.full(n, -1, np.int32)
    theta: Theta | None = None
    in_force = -1
    refits: list[dict[str, Any]] = []
    schedule = refit_bars(ts_ns, params)
    for j, r in enumerate(schedule):
        end = schedule[j + 1] if j + 1 < len(schedule) else n
        xj, zj = _features(y, slot, r, end, params.baseline_days)
        train = zj[:r]
        if theta is None:
            result = fit(train, start(train, params), params.tol, params.max_iter)
        else:
            result = fit(train, theta, params.tol, params.max_iter_warm)
        reason = guard(result, params)
        refits.append(
            {
                "bar": r,
                "warm": theta is not None,
                "observed": int(np.sum(~np.isnan(train))),
                **result.theta._asdict(),
                "loglik": result.loglik,
                "iterations": result.iterations,
                "converged": result.converged,
                "share0": result.shares[0],
                "share1": result.shares[1],
                "flagged": reason is not None,
                "reason": reason,
            }
        )
        if reason is None:
            theta, in_force = result.theta, j
        x[r:end], z[r:end] = xj[r:end], zj[r:end]
        if theta is not None:
            e0, e1, _ = _emissions(zj, theta)
            p[r:end] = _forward(e0, e1, theta)[r:end]
            refit_of[r:end] = in_force
    h = hysteresis(p, params.enter, params.exit)
    missing = np.isnan(y)
    found = episodes(h, missing, params)
    return Labels(x, z, p, h, refit_of, refits, found, nulls(h, missing, found, params))


def refit_bars(ts_ns: np.ndarray, params: Params) -> list[int]:
    """Bar indices of the refits: the end of the warm-up, then every refit_days-th UTC midnight
    counted from a Monday."""
    if not len(ts_ns):
        return []
    day0, last = int(ts_ns[0] // DAY_NS), int(ts_ns[-1] // DAY_NS)
    first = day0 + params.warmup_days
    later = [d for d in range(first + 1, last + 1) if (d - MONDAY) % params.refit_days == 0]
    return [(d - day0) * DAY_BARS for d in [first, *later] if d <= last]


def _slots(ts_ns: np.ndarray) -> np.ndarray:
    weekend = (ts_ns // DAY_NS + 3) % 7 >= 5  # day 0, 1970-01-01, was a Thursday
    return weekend * DAY_BARS + (ts_ns % DAY_NS) // BAR_NS


def _features(
    y: np.ndarray, slot: np.ndarray, refit: int, end: int, baseline_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """x and z of bars [0, end) under the slot means of the bars before `refit`."""
    seen = ~np.isnan(y[:refit])
    sums = np.bincount(slot[:refit][seen], weights=y[:refit][seen], minlength=2 * DAY_BARS)
    counts = np.bincount(slot[:refit][seen], minlength=2 * DAY_BARS)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = y[:end] - (sums / counts)[slot[:end]]  # a slot never seen leaves x missing
    days = -(-end // DAY_BARS)
    daily = np.full(days * DAY_BARS, np.nan)
    daily[:end] = x
    daily = daily.reshape(days, DAY_BARS)
    base = np.full(days, np.nan)
    for d in range(baseline_days, days):
        window = daily[d - baseline_days : d].ravel()
        window = window[~np.isnan(window)]
        if window.size:
            base[d] = np.median(window)
    return x, x - np.repeat(base, DAY_BARS)[:end]


def start(z: np.ndarray, params: Params) -> Theta:
    """The deterministic first start: mu at two percentiles of the observed z, both sigmas their
    standard deviation, p00 = p11 = start_p. NaN when fewer than two bars are observed."""
    seen = z[~np.isnan(z)]
    if seen.size < 2:
        return Theta(*([math.nan] * 6))
    low, high = np.percentile(seen, [params.start_low, params.start_high])
    sd = float(np.std(seen))
    return Theta(float(low), float(high), sd, sd, params.start_p, params.start_p)


def fit(z: np.ndarray, theta: Theta, tol: float, max_iter: int) -> Fit:
    """EM on z (NaN = missing) from `theta`: closed-form M-steps, each followed by label matching,
    until a relative log-likelihood gain below `tol` or `max_iter` M-steps. No random start, so a
    fit repeats bit for bit."""
    nan2 = (math.nan, math.nan)
    if not _valid(theta):
        return Fit(theta, math.nan, 0, False, nan2, "no valid start: too few observed bars")
    observed = ~np.isnan(z)
    with np.errstate(all="ignore"):
        step = _estep(z, observed, theta)
        iterations, converged = 0, False
        while iterations < max_iter and not converged:
            candidate = _mstep(z[observed], step, observed)
            iterations += 1
            new = _estep(z, observed, candidate) if _valid(candidate) else None
            if new is None or not math.isfinite(new.loglik):
                return Fit(candidate, math.nan, iterations, False, nan2, "EM failed numerically")
            converged = new.loglik - step.loglik < tol * abs(step.loglik)
            theta, step = candidate, new
        g0, g1 = step.g0[observed], step.g1[observed]
        return Fit(theta, step.loglik, iterations, converged, (g0.mean(), g1.mean()), None)


def guard(result: Fit, params: Params) -> str | None:
    """Why a fit is degenerate, or None to put it in force."""
    if result.failure:
        return result.failure
    t, found = result.theta, []
    if min(t.sigma0, t.sigma1) < params.min_sigma:
        found.append(f"sigma {min(t.sigma0, t.sigma1):.4g} < {params.min_sigma:g}")
    if min(result.shares) < params.min_share:
        found.append(f"smoothed share {min(result.shares):.4g} < {params.min_share:g}")
    if t.mu1 - t.mu0 < params.min_gap:
        found.append(f"mu1 - mu0 {t.mu1 - t.mu0:.4g} < {params.min_gap:g}")
    return "; ".join(found) or None


class _Step(NamedTuple):
    """An E-step: the log-likelihood, smoothed state probabilities and expected transitions."""

    loglik: float
    g0: np.ndarray
    g1: np.ndarray
    s00: float
    s01: float
    s10: float
    s11: float


def _emissions(z: np.ndarray, theta: Theta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both states' Gaussian densities scaled per bar so the larger is 1, and the log of the
    scale (without -log(2 pi)/2); (1, 1, 0) where z is missing, which makes no update."""
    lp0 = -0.5 * ((z - theta.mu0) / theta.sigma0) ** 2 - math.log(theta.sigma0)
    lp1 = -0.5 * ((z - theta.mu1) / theta.sigma1) ** 2 - math.log(theta.sigma1)
    missing = np.isnan(z)
    shift = np.where(missing, 0.0, np.maximum(lp0, lp1))
    e0 = np.where(missing, 1.0, np.exp(lp0 - shift))
    e1 = np.where(missing, 1.0, np.exp(lp1 - shift))
    return e0, e1, shift


def _stationary(theta: Theta) -> float:
    p01, p10 = 1.0 - theta.p00, 1.0 - theta.p11
    return p01 / (p01 + p10)


def _forward(e0: np.ndarray, e1: np.ndarray, theta: Theta) -> np.ndarray:
    """P(s_t = 1 | z_1..z_t) from the scaled densities, the chain starting stationary. A plain
    loop, so each value is computed from the bars before it alone, however long the series."""
    p01, lam = 1.0 - theta.p00, theta.p00 + theta.p11 - 1.0
    a1 = _stationary(theta)
    out = [0.0] * len(e0)
    for i, (x0, x1) in enumerate(zip(e0.tolist(), e1.tolist(), strict=True)):
        b1 = p01 + lam * a1
        u1 = b1 * x1
        a1 = u1 / (u1 + (1.0 - b1) * x0)
        out[i] = a1
    return np.array(out)


def _estep(z: np.ndarray, observed: np.ndarray, theta: Theta) -> _Step:
    e0, e1, shift = _emissions(z, theta)
    f1 = _forward(e0, e1, theta)
    p00, p11 = theta.p00, theta.p11
    p01, p10 = 1.0 - p00, 1.0 - p11
    prior = np.concatenate([[_stationary(theta)], f1[:-1]])
    b1 = p01 + (p00 + p11 - 1.0) * prior  # predicted P(s_t = 1 | z_1..z_t-1)
    b0 = 1.0 - b1
    u0 = b0 * e0
    c = b1 * e1 + u0
    f0 = u0 / c
    loglik = float(np.log(c).sum() + shift.sum() - 0.5 * LOG_2PI * observed.sum())
    # Smoothing is linear backwards: gamma_t(i) = sum_j f_i(t) p_ij gamma_t+1(j) / b_j(t+1).
    k0, k1 = 1.0 / b0[1:], 1.0 / b1[1:]
    c00, c01 = f0[:-1] * p00 * k0, f0[:-1] * p01 * k1
    c10, c11 = f1[:-1] * p10 * k0, f1[:-1] * p11 * k1
    g0, g1 = _backward(c00, c01, c10, c11, float(f0[-1]), float(f1[-1]))
    return _Step(
        loglik,
        g0,
        g1,
        float((c00 * g0[1:]).sum()),
        float((c01 * g1[1:]).sum()),
        float((c10 * g0[1:]).sum()),
        float((c11 * g1[1:]).sum()),
    )


def _backward(
    c00: np.ndarray, c01: np.ndarray, c10: np.ndarray, c11: np.ndarray, s0: float, s1: float
) -> tuple[np.ndarray, np.ndarray]:
    n = len(c00) + 1
    g0, g1 = [0.0] * n, [0.0] * n
    g0[-1], g1[-1] = s0, s1
    i = n - 1
    for a, b, c, d in zip(
        c00[::-1].tolist(), c01[::-1].tolist(), c10[::-1].tolist(), c11[::-1].tolist(), strict=True
    ):
        s0, s1 = a * s0 + b * s1, c * s0 + d * s1
        i -= 1
        g0[i], g1[i] = s0, s1
    return np.array(g0), np.array(g1)


def _mstep(z: np.ndarray, step: _Step, observed: np.ndarray) -> Theta:
    g0, g1 = step.g0[observed], step.g1[observed]
    n0, n1 = g0.sum(), g1.sum()
    mu0, mu1 = (g0 * z).sum() / n0, (g1 * z).sum() / n1
    sigma0 = np.sqrt((g0 * (z - mu0) ** 2).sum() / n0)
    sigma1 = np.sqrt((g1 * (z - mu1) ** 2).sum() / n1)
    p00 = step.s00 / (step.s00 + step.s01)
    p11 = step.s11 / (step.s11 + step.s10)
    theta = Theta(float(mu0), float(mu1), float(sigma0), float(sigma1), p00, p11)
    if theta.mu0 > theta.mu1:  # label matching: state 1 is the high state
        theta = Theta(theta.mu1, theta.mu0, theta.sigma1, theta.sigma0, theta.p11, theta.p00)
    return theta


def _valid(theta: Theta) -> bool:
    return (
        all(math.isfinite(v) for v in theta)
        and theta.sigma0 > 0
        and theta.sigma1 > 0
        and 0 < theta.p00 < 1
        and 0 < theta.p11 < 1
    )


def hysteresis(p: np.ndarray, enter: float, exit: float) -> np.ndarray:
    """H per bar (int8): 1 once P >= enter, 0 once P <= exit, otherwise the H before, from 0;
    -1 where P is NaN (before the first theta)."""
    decided = np.full(len(p), -1, np.int8)
    decided[p >= enter] = 1
    decided[p <= exit] = 0
    last = np.maximum.accumulate(np.where(decided >= 0, np.arange(len(p)), -1))
    h = np.where(last >= 0, decided[np.maximum(last, 0)], 0).astype(np.int8)
    h[np.isnan(p)] = -1
    return h


def episodes(h: np.ndarray, missing: np.ndarray, params: Params) -> list[dict[str, Any]]:
    """Each maximal run of H = 1, classified in time order (the refractory rule reads the events
    before it): bar indices, its missing bars, and for an event its confirmation bar."""
    labelled = np.flatnonzero(h >= 0)
    if not labelled.size:
        return []
    first, n = int(labelled[0]), len(h)
    gaps = np.concatenate([[0], np.cumsum(missing)])
    history = max(params.clean, params.refractory)
    found: list[dict[str, Any]] = []
    last_event: int | None = None
    for a, b in zip(*_runs(h == 1), strict=True):
        onset, end = int(a), int(b)
        if end - onset < params.d_min:
            kind = "flicker" if end < n else "pending"
        elif (
            onset - history >= first
            and not np.any(h[onset - params.clean : onset] == 1)
            and (last_event is None or onset - last_event > params.refractory)
        ):
            kind, last_event = "event", onset
        else:
            kind = "episode"
        found.append(
            {
                "onset": onset,
                "end": end if end < n else None,
                "bars": end - onset,
                "kind": kind,
                "confirm": onset + params.d_min if kind == "event" else None,
                "missing": int(gaps[end] - gaps[onset]),
            }
        )
    return found


def nulls(
    h: np.ndarray, missing: np.ndarray, found: list[dict[str, Any]], params: Params
) -> list[dict[str, Any]]:
    """The null stretches, as [start, end) bar indices with their missing bars."""
    labelled = np.flatnonzero(h >= 0)
    if not labelled.size:
        return []
    first, n = int(labelled[0]), len(h)
    cover = np.zeros(n + 1, np.int64)

    def exclude(a: int, b: int) -> None:
        a, b = min(max(a, 0), n), min(max(b, 0), n)
        if a < b:
            cover[a] += 1
            cover[b] -= 1

    exclude(0, first + params.refractory)
    exclude(n - params.refractory, n)
    for e in found:
        margin = params.clean if e["kind"] == "flicker" else params.refractory
        exclude(e["onset"] - margin, (n if e["end"] is None else e["end"]) + margin)
    gaps = np.concatenate([[0], np.cumsum(missing)])
    return [
        {"start": int(a), "end": int(b), "bars": int(b - a), "missing": int(gaps[b] - gaps[a])}
        for a, b in zip(*_runs(np.cumsum(cover[:n]) == 0), strict=True)
    ]


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The [start, end) bounds of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8)))
    return edges[0::2], edges[1::2]


# ---------------------------------------------------------------- the labels run

_TIME = [("ts_ns", pa.int64()), ("symbol", pa.string())]
SCHEMAS = {
    "bars": pa.schema(
        [
            *_TIME,
            ("minutes", pa.int8()),
            ("trades", pa.int64()),
            *[
                (name, pa.float64())
                for name in ("rv", "close", "volume", "taker_buy_share", "oi", "funding_rate")
            ],
        ]
    ),
    "labels": pa.schema(
        [
            *_TIME,
            ("observed", pa.bool_()),
            ("x", pa.float64()),
            ("z", pa.float64()),
            ("p_high", pa.float64()),
            ("h", pa.int8()),
            ("refit_ts_ns", pa.int64()),
        ]
    ),
    "refits": pa.schema(
        [
            *_TIME,
            ("warm", pa.bool_()),
            ("observed", pa.int64()),
            ("iterations", pa.int32()),
            ("converged", pa.bool_()),
            ("loglik", pa.float64()),
            *[(name, pa.float64()) for name in (*Theta._fields, "share0", "share1")],
            ("flagged", pa.bool_()),
            ("reason", pa.string()),
        ]
    ),
    "episodes": pa.schema(
        [
            *_TIME,
            ("end_ts_ns", pa.int64()),
            ("bars", pa.int32()),
            ("kind", pa.string()),
            ("confirm_ts_ns", pa.int64()),
            ("missing", pa.int32()),
        ]
    ),
    "nulls": pa.schema(
        [*_TIME, ("end_ts_ns", pa.int64()), ("bars", pa.int32()), ("missing", pa.int32())]
    ),
}


def tables(series: dict[str, tuple[dict[str, np.ndarray], Labels]]) -> dict[str, pa.Table]:
    """The run's tables from each symbol's bars and labels, in (ts_ns, symbol) order. Every
    table's ts_ns is a bar's start: the bar's own, a refit's, an episode's onset, a stretch's
    start; NaN is written as null."""
    parts: dict[str, list[pa.Table]] = {name: [] for name in SCHEMAS}
    for symbol, (bar, out) in series.items():
        ts = bar["ts_ns"]
        t0 = int(ts[0]) if len(ts) else 0

        def at(i: int | None, t0: int = t0) -> int | None:
            return None if i is None else t0 + i * BAR_NS

        names = [symbol] * len(ts)
        refit_ts = np.array([at(r["bar"]) for r in out.refits] + [0], np.int64)[out.refit]
        parts["bars"].append(_table("bars", {**bar, "symbol": names}))
        parts["labels"].append(
            _table(
                "labels",
                {
                    "ts_ns": ts,
                    "symbol": names,
                    "observed": bar["rv"] > 0,
                    "x": out.x,
                    "z": out.z,
                    "p_high": out.p,
                    "h": pa.array(out.h, pa.int8(), mask=out.h < 0),
                    "refit_ts_ns": pa.array(refit_ts, pa.int64(), mask=out.refit < 0),
                },
            )
        )
        rows = {
            "refits": [{**r, "ts_ns": at(r["bar"]), "symbol": symbol} for r in out.refits],
            "episodes": [
                {
                    **e,
                    "ts_ns": at(e["onset"]),
                    "symbol": symbol,
                    "end_ts_ns": at(e["end"]),
                    "confirm_ts_ns": at(e["confirm"]),
                }
                for e in out.episodes
            ],
            "nulls": [
                {**s, "ts_ns": at(s["start"]), "symbol": symbol, "end_ts_ns": at(s["end"])}
                for s in out.nulls
            ],
        }
        for name, found in rows.items():
            parts[name].append(pa.Table.from_pylist(found, schema=SCHEMAS[name]))
    return {
        name: pa.concat_tables(found).sort_by([("ts_ns", "ascending"), ("symbol", "ascending")])
        if found
        else SCHEMAS[name].empty_table()
        for name, found in parts.items()
    }


def _table(name: str, columns: dict[str, Any]) -> pa.Table:
    arrays = []
    for field in SCHEMAS[name]:
        values = columns[field.name]
        if isinstance(values, np.ndarray) and values.dtype.kind == "f":
            values = pa.array(values, field.type, mask=np.isnan(values))
        arrays.append(values if isinstance(values, pa.Array) else pa.array(values, field.type))
    return pa.Table.from_arrays(arrays, schema=SCHEMAS[name])


def write_run(
    data_dir: Path,
    tag: str,
    found: dict[str, pa.Table],
    params: dict[str, Any],
    sources: list[dict[str, Any]],
    missing: list[str],
    checks: dict[str, Any],
) -> Path:
    """A labels run in the shared layout: <data dir>/labels/<run_id>/ with manifest.json first,
    one Parquet file per table, named by its first key, and complete.json once all are synced."""
    run_id = engine.run_id(tag)
    run_dir = Path(data_dir) / "labels" / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "format": 1,
        "domain": "labels",
        "run_id": run_id,
        "producer": {"name": "tipping_point.market", "version": __version__, **io.code_version()},
        "started_ns": time.time_ns(),
        "key": "ts_ns",
        "tables": list(found),
        "params": params,
        "sources": sources,
        "missing_sources": missing,
        "licence": "CC BY-NC-SA 4.0",
        "attribution": "Binance Vision",
    }
    io.write_json(run_dir / "manifest.json", manifest)
    files: dict[str, str] = {}
    last_key = -1
    for name, table in found.items():
        keys = table["ts_ns"].to_numpy()
        path = run_dir / name / f"{int(keys[0]) if len(keys) else 0:020d}.parquet"
        path.parent.mkdir()
        io.write_file(path, lambda f, table=table: pq.write_table(table, f, compression="zstd"))
        files[path.relative_to(run_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        last_key = max(last_key, int(keys.max()) if len(keys) else -1)
    record = {
        "ended_ns": time.time_ns(),
        "last_key": last_key,
        "rows": {name: table.num_rows for name, table in found.items()},
        "clean": True,
        "files": files,
        "checks": checks,
    }
    io.write_json(run_dir / "complete.json", record)
    return run_dir
