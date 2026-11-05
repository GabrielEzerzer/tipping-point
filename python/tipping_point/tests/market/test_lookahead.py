"""The labeller only looks back, bit for bit (#29 S2; research pass section 4, "Lookahead (exact)").

For cut times T at every refit boundary (the bar before a refit and the refit's own bar) and at 20
seeded bars of the planted 3-month series, the labeller run (i) on the bars truncated at T and
(ii) on the bars perturbed after T must give, bit for bit, the full run's x, z, P, H and theta in
force at every bar t <= T, every refit at r <= T, every episode whose class is settled by T (each
event confirmed by T among them), and every null stretch that ends by T + 1 - d_min - Q. That last
bound is d_min - 1 bars short of section 4's T - Q. Whether bar t is in a stretch depends on H[t],
on the first label and on the event onsets in [t - Q, t + Q]. A run of H = 1 that began at bar o is
known to be an event, or not, at bar o + d_min - 1: its length reaches d_min then, and its clean and
refractory checks read only earlier bars. So bar t is settled by T once every onset up to t + Q is,
that is when t + Q + d_min - 1 <= T. A stretch [start, end) is settled when its bars and the bar
`end` that closes it are, so when end <= T + 1 - d_min - Q; the run truncated at T, whose last Q
bars are cut, keeps every such bar. An event onset at T + 2 - d_min is still pending at T, and in
the full run it ends a stretch Q bars before it, one bar past this bound
(test_the_null_stretch_bound_holds_at_its_edge).

The sweep keeps to seconds two ways. It memoises msvol.fit on its exact inputs (the training z's
bytes, the start, tol and max_iter): EM is a pure function of those, so a hit returns what
recomputing would, and a lookahead changes the training z and misses. And its perturbed runs end
8 days after T, so each holds at least one refit on perturbed bars; truncation at every cut
already catches a dependence on how far the data reach. One cut runs without the memo, perturbed
to the end of the series.
"""

import hashlib
import json
from typing import Any

import numpy as np
import pytest
from tipping_point.market import msvol

SEEDED = 20
REACH = 8 * 288  # bars a perturbed run carries past T: more than the 7 days between refits


def _cuts(planted: Any) -> list[int]:
    bars = [r["bar"] for r in planted.full.refits]
    seeded = np.random.default_rng(4).integers(bars[0], len(planted.ts) - 1, SEEDED)
    return sorted({*bars, *(b - 1 for b in bars), *seeded.tolist()})


def _perturbed(rv: np.ndarray, T: int, reach: int | None = None) -> np.ndarray:
    """rv through T + reach (to the end when None), every bar after T rescaled by noise and 5%
    of them dropped."""
    rng = np.random.default_rng([7, T])
    out = rv[: None if reach is None else T + 1 + reach].copy()
    after = out[T + 1 :]
    after *= np.exp(rng.normal(0.0, 0.5, after.size))
    after[rng.uniform(size=after.size) < 0.05] = np.nan
    return out


def _settled(labels: msvol.Labels, T: int, d_min: int) -> dict[int, tuple[Any, ...]]:
    """By onset, the episodes whose class is known at T: (kind, confirm), plus (end, bars,
    missing) for those closed by T."""
    out = {}
    for e in labels.episodes:
        closed = e["end"] is not None and e["end"] <= T
        if closed or e["onset"] + d_min - 1 <= T:
            tail = (e["end"], e["bars"], e["missing"]) if closed else ()
            out[e["onset"]] = (e["kind"], e["confirm"], *tail)
    return out


def _assert_same_through(full: msvol.Labels, part: msvol.Labels, T: int, p: msvol.Params) -> None:
    for name in ("x", "z", "p", "h", "refit"):
        a, b = getattr(full, name)[: T + 1], getattr(part, name)[: T + 1]
        assert a.tobytes() == b.tobytes(), f"{name} differs at or before T = {T}"
    done = [json.dumps(r) for r in full.refits if r["bar"] <= T]
    assert [json.dumps(r) for r in part.refits[: len(done)]] == done, f"refits by T = {T}"
    assert len([r for r in part.refits if r["bar"] <= T]) == len(done)
    assert _settled(full, T, p.d_min) == _settled(part, T, p.d_min), f"episodes by T = {T}"
    assert _final(full.nulls, T, p) == _final(part.nulls, T, p), f"null stretches by T = {T}"


def _final(nulls: list[dict[str, Any]], T: int, p: msvol.Params) -> list[dict[str, Any]]:
    """The null stretches no bar after T can change: those ending by T + 1 - d_min - Q."""
    return [s for s in nulls if s["end"] <= T + 1 - p.d_min - p.refractory]


def _moved(planted: Any, T: int, reach: int | None) -> msvol.Labels:
    rv = _perturbed(planted.rv, T, reach)
    return msvol.label(planted.ts[: len(rv)], rv, planted.params)


@pytest.fixture
def memo(monkeypatch: pytest.MonkeyPatch) -> dict[Any, msvol.Fit]:
    cache: dict[Any, msvol.Fit] = {}
    real = msvol.fit

    def fit(z: np.ndarray, theta: msvol.Theta, tol: float, max_iter: int) -> msvol.Fit:
        key = (hashlib.sha256(z.tobytes()).digest(), len(z), tuple(theta), tol, max_iter)
        if key not in cache:
            cache[key] = real(z, theta, tol, max_iter)
        return cache[key]

    monkeypatch.setattr(msvol, "fit", fit)
    return cache


def test_no_label_at_or_before_t_depends_on_a_later_bar(planted: Any, memo: Any) -> None:
    full, params, n = planted.full, planted.params, len(planted.ts)
    kinds = [e["kind"] for e in full.episodes]
    assert len(full.refits) == 9 and not any(r["flagged"] for r in full.refits)
    assert kinds.count("event") >= 20 and kinds.count("flicker") >= 5 and len(full.nulls) >= 10
    cuts = _cuts(planted)
    assert len(cuts) >= 2 * len(full.refits) + SEEDED - 2
    refitted_after = 0
    for cut in cuts:
        truncated = msvol.label(planted.ts[: cut + 1], planted.rv[: cut + 1], params)
        _assert_same_through(full, truncated, cut, params)
        moved = _moved(planted, cut, REACH)
        _assert_same_through(full, moved, cut, params)
        later = slice(cut + 1, len(moved.p))
        assert cut + 1 == n or not np.array_equal(moved.p[later], full.p[later], equal_nan=True)
        refitted_after += any(r["bar"] > cut for r in moved.refits)
    last = full.refits[-1]["bar"]
    assert refitted_after == sum(cut < last for cut in cuts)  # every refit once on moved bars
    assert len(memo) > len(full.refits)


def test_refits_repeat_bit_for_bit_without_the_memo(planted: Any) -> None:
    full, params = planted.full, planted.params
    T = full.refits[4]["bar"]
    truncated = msvol.label(planted.ts[: T + 1], planted.rv[: T + 1], params)
    _assert_same_through(full, truncated, T, params)
    moved = _moved(planted, T, None)
    assert len(moved.p) == len(full.p) and len(moved.refits) == len(full.refits)
    _assert_same_through(full, moved, T, params)


def test_the_null_stretch_bound_holds_at_its_edge() -> None:
    """d_min 3, C 4, Q 10: one event, a run of H = 1 at [60, 64), and nothing else in 100 bars. Cut
    at T = 61, two bars into it, the truncated run takes it for pending, rules out no bar for it and
    holds the stretch [10, 52), cut Q bars from its end; the full run knows it is an event and ends
    that stretch at 50, Q bars before the onset. T + 1 - d_min - Q = 49 leaves the stretch out; the
    looser bound, 50, compares it and fails a correct labeller. At T = 62 the event is confirmed and
    the bound, now 50, compares the same stretch in both."""
    p = msvol.Params(d_min=3, clean=4, refractory=10)
    n = 100
    h = np.zeros(n, np.int8)
    h[60:64] = 1
    missing = np.zeros(n, bool)

    def stretches(path: np.ndarray) -> list[dict[str, Any]]:
        found = msvol.episodes(path, missing[: len(path)], p)
        return msvol.nulls(path, missing[: len(path)], found, p)

    full = stretches(h)
    assert [(s["start"], s["end"]) for s in full] == [(10, 50), (71, 90)]
    T = 61
    part = stretches(h[: T + 1])
    assert [e["kind"] for e in msvol.episodes(h[: T + 1], missing[: T + 1], p)] == ["pending"]
    assert [(s["start"], s["end"]) for s in part] == [(10, 52)]
    assert _final(full, T, p) == _final(part, T, p) == []
    loose = T + 2 - p.d_min - p.refractory
    assert [s for s in full if s["end"] <= loose] != [s for s in part if s["end"] <= loose]
    T = 62
    part = stretches(h[: T + 1])
    assert [(s["start"], s["end"]) for s in part] == [(10, 50)]
    assert _final(full, T, p) == _final(part, T, p) == full[:1]
