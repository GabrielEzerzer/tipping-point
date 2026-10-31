"""Known answers for the ising-ews v1 freeze (#97, #98; decisions: #26, comment 6402030372).

harness/ews_planted.py's v1 dataset plants, at L = 16, 32 and 64 (two seeds of two replicas, six
at the edge mode's L = 32), slow ramps whose mean |m| rises through Tc and whose spread of |m|
peaks
  fss      at eps_L 0.8 at every L: the dynamic test passes, 1/nu = 1
  fixed_T  at one temperature, eps_L 0.8, 1.6 and 3.2: it fails, 1/nu near 0
  nu065    drifting so that 1/nu = 0.65: a pass 0.15 inside the tolerance of 0.5, which any
           tolerance under ~0.3 fails
  nu035    drifting so that 1/nu = 0.35: a fail 0.15 outside it, which any tolerance over ~0.7
           passes
  edge     as fss, but at L = 32 four of each seed's six replicas peak past the search band's
           cold edge: inconclusive by the edge rule alone, the located peaks passing every other
           clause
and the scorer, ising.evaluate, must find those answers under a copy of the v1 prereg.

The lag. The rolling indicator at sample t describes the headline window of 20 tau_c ending at t.
The ramp spends 1/x = 50 tau_c per unit eps_L, so the window spans 20 x = 0.4 eps_L and follows
the spread at its middle, 0.2 eps_L back: the indicator peaks 0.2 eps_L below the spread it
measures, later on the cooling ramp. Each L's median peak must sit within 0.08 eps_L (4 tau_c of
ramp) of the planted eps_L less that lag.

The cadences. tau_c is 12, 16 and 24 sweeps, so the dense cadence keeps a sample every 1.2, 1.6
and 2.4 sweeps and the matched one bursts of 8 frames every 21, 29 and 43 sweeps: a window
counted in sweeps or frames instead of samples, or a matched density turned over, moves the
peaks. The mean's rise, steepest at Tc, is what the detrend removes; without it the rolling
variance peaks at the rise, eps_L -0.2 as read.

The prereg copy keeps every setting the dynamic test reads as frozen (the headline window and
bandwidth, thin, lag, detrend and direction, fss.dynamic, data.ramps, data.nulls) and changes what
the planted data require (seeds, replicas, the tau_c record, the tau.json pins), plus, to score
in seconds, 40 bootstrap draws and settings the dynamic test does not read: the sensitivity grid
cut to its headline cell, |m| the only observable, horizon 0 alone, one surrogate per test, and
alarm scores every 200 tau_c instead of every one. The alarm tests read that last one: one score
per null path puts far_max at 5 per 1000 tau_c whatever the data, so the attainable control checks
the flag, not the rate. As frozen, a mode takes minutes to score: the alarm s(t) alone is ~100,000
Kendall scores of 300 tau_c each.

The generator's own checks read its output only; the rest need the freeze's scorer.
"""

import hashlib
import itertools
import json
import re
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ews_planted import (
    DRIFT,
    EDGE_L,
    EDGE_REPLICAS,
    EDGE_SEED_REPLICAS,
    MODES,
    SPREAD,
    V1,
    V1_EPS_L,
    V1_LAG,
    V1_NULL_EPS,
    V1_PILOT_RUN,
    V1_REPLICAS,
    V1_SEARCH,
    V1_SEEDS,
    V1_SIZES,
    V1_TAU_C,
    V1_X,
    restamp,
    restamp_pilot,
    restamp_tau,
    tau_source,
    v1_copy,
    v1_dataset,
    v1_mean,
    v1_peak,
    v1_ramp_run,
    v1_replicas,
)
from tipping_point.ews import indicators, ising
from tipping_point.ews.__main__ import main
from tipping_point.ising import io, protocol
from tipping_point.ising import pilot as tauc
from tipping_point.ising.exact import TC

V1_TOML = Path(__file__).resolve().parents[4] / "paper" / "prereg" / "ising-ews-v1.toml"
CADENCES = ("dense", "matched")
CONTRACT: dict[str, Any] = {  # what the planted answers assume of the frozen prereg
    "data.L": list(V1_SIZES),
    "data.ramps.x": V1_X,
    "data.ramps.eps_L": list(V1_EPS_L),
    "data.nulls.eps_L": list(V1_NULL_EPS),
    "indicators.cadences": list(CADENCES),
    "indicators.direction.variance": "greater",
    "sensitivity.headline": {"window": 20.0, "bandwidth": 10.0},
    "fss.dynamic.statistic": "peak",
    "fss.dynamic.indicators": ["variance"],
    "fss.dynamic.observable": "abs_m",
    "fss.dynamic.search": list(V1_SEARCH),
    "fss.dynamic.expected_inverse_nu": 1.0,
    "fss.dynamic.tolerance": 0.5,
    "fss.dynamic.cadence": "dense",
    "fss.dynamic.min_draw_share": 0.95,
    "scoring.alarms.indicators": ["variance", "ac", "skewness"],
}
HEADLINE = CONTRACT["sensitivity.headline"]
LAG = HEADLINE["window"] * V1_X / 2  # eps_L the indicator's peak trails the spread's: 0.2
TOLERANCE = 0.08  # eps_L between an L's median peak and the planted eps_L less LAG
UNATTAINABLE = 2.0  # false alarms per tau_c: s(t) is scored at most once per tau_c
STAMP = "deadbeef" * 5  # the commit every mode's runs but fss's say made them


# ---------------------------------------------------------------- the planted data and the prereg


@pytest.fixture(scope="module")
def v1(tmp_path_factory: pytest.TempPathFactory) -> dict[str, V1]:
    """One data dir per mode. The others share fss's record, null runs and staircases, and say
    STAMP made them from a clean tree, which only the stamping test reads: an unfrozen prereg
    scores any commit."""
    root = tmp_path_factory.mktemp("ews-v1")
    fss = v1_dataset(root / "fss", "fss")
    return {"fss": fss} | {
        mode: v1_dataset(root / mode, mode, git_sha=STAMP, dirty=False, shared=fss)
        for mode in MODES
        if mode != "fss"
    }


def _get(doc: dict[str, Any], name: str) -> Any:
    for key in name.split("."):
        doc = doc[key]
    return doc


def _toml(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(map(_toml, value)) + "]"
    raise TypeError(f"no TOML literal here for {value!r}")


def _edit(text: str, settings: dict[str, Any]) -> str:
    """TOML text with the value of each "table.key" of settings replaced, the rest byte for byte;
    each value it replaces must sit on its key's line."""
    lines, table, done = [], "", set()
    for line in text.splitlines(keepends=True):
        header = re.fullmatch(r"\s*\[\[?\s*([\w.-]+)\s*\]\]?\s*(#.*)?", line.rstrip("\r\n"))
        key = re.match(r"\s*([\w-]+)\s*=", line)
        if header:
            table = header.group(1)
        elif key and f"{table}.{key.group(1)}" in settings:
            tomllib.loads(line)  # a value spanning lines raises here
            done.add(name := f"{table}.{key.group(1)}")
            line = f"{key.group(1)} = {_toml(settings[name])}\n"
        lines.append(line)
    assert done == set(settings), f"not in {V1_TOML.name}: {sorted(set(settings) - done)}"
    return "".join(lines)


def _prereg(data: V1, changes: dict[str, Any] | None = None) -> str:
    """The v1 prereg's text, set to score `data` as the module docstring says, then `changes`."""
    text = V1_TOML.read_text("utf-8")
    expected = tomllib.loads(text)
    head = expected["sensitivity"]["headline"]
    settings = {
        "data.seeds": list(V1_SEEDS),
        "data.replicas": V1_REPLICAS,
        "data.tau_c.pilot_run": V1_PILOT_RUN,
        "data.tau_c.pilot_sha256": hashlib.sha256(data.pilot.read_bytes()).hexdigest(),
        **{f"data.staircase.tau_sha256.{L}": tau_source(L) for L in V1_SIZES},
        "scoring.bootstrap": 40,
        "sensitivity.window": [head["window"]],
        "sensitivity.bandwidth": [head["bandwidth"]],
        "indicators.observables": ["abs_m"],
        "event.horizons": [0.0],
        "scoring.surrogates.n": 1,
        "scoring.alarms.every": 200.0,
    } | (changes or {})
    edited = _edit(text, settings)
    for name, value in settings.items():
        *tables, key = name.split(".")
        _get(expected, ".".join(tables))[key] = value
    assert tomllib.loads(edited) == expected  # nothing else changed
    return edited


@pytest.fixture(scope="module")
def frozen_v1() -> dict[str, Any]:
    """The committed v1 prereg, which must hold what the planted answers assume (CONTRACT)."""
    doc = tomllib.loads(V1_TOML.read_text("utf-8"))

    def value(name: str) -> Any:
        try:
            return _get(doc, name)
        except KeyError:
            return "<missing>"

    wrong = {name: value(name) for name, want in CONTRACT.items() if value(name) != want}
    assert not wrong, f"{V1_TOML.name} is not the v1 the planted answers assume: {wrong}"
    return doc


def _evaluate(root: Path, text: str) -> ising.Evaluation:
    doc = tomllib.loads(text)
    return ising.evaluate(doc, root, np.random.default_rng(26), say=lambda _: None)


@pytest.fixture(scope="module")
def fss_scored(v1: dict[str, V1], frozen_v1: dict[str, Any]) -> ising.Evaluation:
    return _evaluate(v1["fss"].root, _prereg(v1["fss"]))


@pytest.fixture(scope="module")
def fixed_T_scored(v1: dict[str, V1], frozen_v1: dict[str, Any]) -> ising.Evaluation:
    return _evaluate(v1["fixed_T"].root, _prereg(v1["fixed_T"]))


@pytest.fixture(scope="module")
def edge_scored(v1: dict[str, V1], frozen_v1: dict[str, Any]) -> ising.Evaluation:
    """The edge mode, its alarms set at a target no theta reaches: the dynamic verdict reads no
    alarm, so one scoring serves both."""
    return _evaluate(v1["edge"].root, _prereg(v1["edge"], {"scoring.alarms.far": UNATTAINABLE}))


@pytest.fixture(scope="module")
def nu065_scored(v1: dict[str, V1], frozen_v1: dict[str, Any]) -> ising.Evaluation:
    return _evaluate(v1["nu065"].root, _prereg(v1["nu065"]))


@pytest.fixture(scope="module")
def nu035_scored(v1: dict[str, V1], frozen_v1: dict[str, Any]) -> ising.Evaluation:
    return _evaluate(v1["nu035"].root, _prereg(v1["nu035"]))


def _dynamic(evaluation: ising.Evaluation) -> dict[str, dict[str, Any]]:
    """The dynamic test's rows by cadence: inverse_nu_variance at each, the verdict at
    fss.dynamic.cadence (dense) alone; the matched row is a robustness check, with none."""
    rows = [r for r in evaluation.tables["fss"] if r["role"].startswith("dynamic")]
    assert sorted((r["quantity"], r["cadence"], r["role"]) for r in rows) == [
        ("inverse_nu_variance", "dense", "dynamic"),
        ("inverse_nu_variance", "matched", "dynamic-robustness"),
    ], rows
    by_cadence = {r["cadence"]: r for r in rows}
    assert by_cadence["matched"]["verdict"] == "", rows
    return by_cadence


def _peaks(evaluation: ising.Evaluation) -> dict[tuple[str, int], dict[str, Any]]:
    """The peaks table's variance rows by (cadence, L)."""
    rows = [r for r in evaluation.tables["peaks"] if r["indicator"] == "variance"]
    by_cell = {(r["cadence"], r["L"]): r for r in rows}
    assert len(rows) == len(by_cell) and set(by_cell) == {
        (c, L) for c in CADENCES for L in V1_SIZES
    }, rows
    return by_cell


def _ramp(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """A ramp run's bath eps_L at every step, and |m| (replicas, steps), read with the simulator's
    io from its stats rows, one per sweep (the dense cadence keeps one per 0.1 tau_c of them)."""
    params = io.read_json(run_dir / "manifest.json")["params"]
    L, replicas = params["L"], params["replicas"]
    stats = io.read_table(run_dir, "stats")
    temperature = stats["temperature"].to_numpy().reshape(-1, replicas)[:, 0]
    M = stats["magnetisation"].to_numpy().reshape(-1, replicas)
    return L * (temperature - TC) / TC, np.abs(M).T / L**2


# ---------------------------------------------------------------- the generator's own checks


@pytest.mark.parametrize("mode", MODES)
def test_the_planted_spread_peaks_where_it_was_planted(v1: dict[str, V1], mode: str) -> None:
    """The planted answer, from the generated |m| and nothing else: about the planted mean, each
    group of replicas with one planted peak, pooled over seeds, has its largest spread, in bins
    of 0.1 eps_L across the search band, within a bin of that peak (of the band's cold edge for
    a peak past it), and there the planted sd. A generator that put the peak elsewhere, or let
    the rounding to whole spins flatten it, would leave the scorer tests checking the wrong
    answer."""
    top, bottom = V1_SEARCH
    edges = np.linspace(top, bottom, 101)
    centres = (edges[:-1] + edges[1:]) / 2
    for L in V1_SIZES:
        groups: dict[float, list[np.ndarray]] = {}
        eps = np.empty(0)
        for seed in V1_SEEDS:
            eps, abs_m = _ramp(v1[mode].runs["ramp", L, seed])
            for replica, row in enumerate(abs_m):
                groups.setdefault(v1_peak(mode, L, replica), []).append(row - v1_mean(eps))
        for planted, rows in groups.items():
            deviation = np.stack(rows)
            spread = np.array(
                [
                    np.sqrt(np.mean(deviation[:, (eps <= hi) & (eps > lo)] ** 2))
                    for hi, lo in itertools.pairwise(edges)
                ]
            )
            k = int(np.argmax(spread))
            where = (mode, L, planted, centres[k])
            assert abs(centres[k] - min(max(planted, bottom), top)) <= 0.1, where
            if bottom < planted < top:
                assert spread[k] == pytest.approx(SPREAD[1], rel=0.1), where
            else:  # from the background at the band's far end, the spread rises to its edge
                far = spread[0] if planted < bottom else spread[-1]
                assert spread[k] > 1.4 * SPREAD[0] and far < 1.2 * SPREAD[0], where


@pytest.mark.parametrize("mode", MODES)
def test_the_headline_rolling_variance_peaks_half_a_window_after_the_spread(
    v1: dict[str, V1], mode: str
) -> None:
    """What the scorer tests expect of the headline pipeline, checked on the generated |m| with
    its own pieces (Gaussian detrend over 10 tau_c, rolling variance over 20 tau_c, on every
    sweep) and the freeze's rule for a peak: the extremum inside the search band, not located when
    it is the band's first or last sample. At every L the median located peak sits within
    TOLERANCE of the planted eps_L less LAG, the lag the generator plants its drifts against,
    and none is unlocated but EDGE_REPLICAS of each seed's EDGE_SEED_REPLICAS at the edge mode's
    EDGE_L."""
    assert pytest.approx(V1_LAG) == LAG
    for L in V1_SIZES:
        peaks, planted = _peaks_by_hand(v1[mode], mode, L, detrend=True)
        share = EDGE_REPLICAS / EDGE_SEED_REPLICAS if (mode, L) == ("edge", EDGE_L) else 0.0
        assert np.mean(np.isnan(peaks)) == share, (mode, L, peaks)
        median = float(np.nanmedian(peaks))
        assert abs(median - (planted - LAG)) <= TOLERANCE, (mode, L, median, planted)


@pytest.mark.parametrize("mode", MODES)
def test_without_the_detrend_the_rolling_variance_peaks_at_the_mean_rise(
    v1: dict[str, V1], mode: str
) -> None:
    """The planted mean |m| rises through Tc steeply enough that the rolling variance of |m|
    without the detrend peaks at the rise (eps_L 0, read at -LAG) at every L and for every
    replica, edge replicas included: at least 0.8 eps_L from every planted answer, so a scorer
    that skips the detrend misses them all."""
    for L in V1_SIZES:
        peaks, planted = _peaks_by_hand(v1[mode], mode, L, detrend=False)
        assert np.all(np.abs(peaks + LAG) <= TOLERANCE), (mode, L, peaks)
        assert planted >= 0.8, (mode, L, planted)  # read at planted - LAG, the rise at -LAG


def _plain_rolling_variance(x: np.ndarray, window: int) -> np.ndarray:
    """Each trailing window's sample variance from running sums of a row-centred x, NaN for the
    first window - 1: what indicators.rolling_variance gives, less its two-pass recompute of every
    window whose level dwarfs its spread, as in an undetrended |m|, where it takes seconds. Only
    the argmax is read, which rounding at 1e-14 cannot move."""
    y = x - x.mean(axis=-1, keepdims=True)
    s1, s2 = (np.pad(np.cumsum(v, axis=-1), [(0, 0), (1, 0)]) for v in (y, y * y))
    sums, squares = s1[:, window:] - s1[:, :-window], s2[:, window:] - s2[:, :-window]
    out = np.full(x.shape, np.nan)
    out[:, window - 1 :] = (squares - sums * sums / window) / (window - 1)
    return out


def _peaks_by_hand(data: V1, mode: str, L: int, detrend: bool) -> tuple[np.ndarray, float]:
    """Each replica's peak at L under the freeze's rule (NaN on the search band's edge) from the
    headline rolling variance of every sweep's |m|, with or without the Gaussian detrend; and
    the planted peak of the replicas whose peak lies inside the band."""
    top, bottom = V1_SEARCH
    tau, peaks = V1_TAU_C[L], []
    window = round(HEADLINE["window"] * tau)
    for seed in V1_SEEDS:
        eps, abs_m = _ramp(data.runs["ramp", L, seed])
        if detrend:
            residuals = indicators.gaussian_detrend(abs_m, HEADLINE["bandwidth"] * tau)
            values = indicators.rolling_variance(residuals, window)
        else:
            values = _plain_rolling_variance(abs_m, window)
        inside = np.flatnonzero((eps <= top) & (eps >= bottom))
        k = np.argmax(values[:, inside], axis=1)
        located = (k > 0) & (k < len(inside) - 1)
        peaks += list(np.where(located, eps[inside][k], np.nan))
    (planted,) = {
        p
        for replica in range(v1_replicas(mode, L))
        if bottom < (p := v1_peak(mode, L, replica)) < top
    }
    return np.array(peaks), planted


def test_v1_ramps_are_laid_out_as_the_ramp_command_lays_them_out(v1: dict[str, V1]) -> None:
    """Each ramp's manifest as `ramp --x 0.02 --eps-start 12 --eps-end -8` writes it under the
    freeze: the eps_L window with T0 and T1 set from it, a rate of 50 tau_c per unit eps_L from
    the tau_c the record measured at that L, rate_source "measured", and the record's stamp, in a
    schedule protocol rebuilds as it was (what a resumed run reads); its bath cools from
    eps_L = 12 to -8."""
    data = v1["fss"]
    stamp = tauc.stamp(data.pilot, io.read_json(data.pilot))
    for L in V1_SIZES:
        for seed in V1_SEEDS:
            run_dir = data.runs["ramp", L, seed]
            params = io.read_json(run_dir / "manifest.json")["params"]
            ramp = params["schedule"]["params"]
            assert params["schedule"]["kind"] == "ramp" and params["pilot"] == stamp
            assert (ramp["x"], ramp["eps_L"], ramp["rate_source"]) == (
                V1_X,
                list(V1_EPS_L),
                "measured",
            )
            assert ramp["T0"] == pytest.approx(TC * (1 + V1_EPS_L[0] / L))
            assert ramp["T1"] == pytest.approx(TC * (1 + V1_EPS_L[1] / L))
            per_eps_L = ramp["rate"] * TC / L / V1_TAU_C[L]  # tau_c of cooling per unit eps_L
            assert per_eps_L == pytest.approx(1 / V1_X, rel=0.005), (L, ramp["rate"])
            assert protocol.from_json(params["schedule"]).to_json() == params["schedule"]
            eps, _ = _ramp(run_dir)
            assert (eps[0], eps[-1]) == pytest.approx(V1_EPS_L)


def test_the_planted_record_is_a_fresh_record_of_every_v1_size(v1: dict[str, V1]) -> None:
    """The simulator's own reader takes the planted tau_c record (every field, each size over
    300 tau_c) and finds it measured on the running kernel, as the #54 record will be."""
    record = tauc.read(v1["fss"].pilot)  # refuses a damaged or a stale record
    assert {int(L): entry["tau_c"] for L, entry in record["per_L"].items()} == V1_TAU_C
    assert tauc.stale(record) is None


def test_runs_name_the_commit_they_were_stamped_with(v1: dict[str, V1], tmp_path: Path) -> None:
    """Every run of a dataset, copied ones included, says the commit and dirty flag it was given
    where the simulator writes them (producer.git_sha and producer.dirty); restamp rewrites those
    two fields and nothing else."""
    for mode in MODES:
        made = (None, None) if mode == "fss" else (STAMP, False)
        for run_dir in v1[mode].runs.values():
            producer = io.read_json(run_dir / "manifest.json")["producer"]
            assert (producer["git_sha"], producer["dirty"]) == made, run_dir
    run_dir = shutil.copytree(v1["fss"].runs["ramp", 16, V1_SEEDS[0]], tmp_path / "run")
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    restamp(run_dir, "f" * 40, True)
    manifest = run_dir / "manifest.json"
    changed = io.read_json(manifest)
    assert (changed["producer"]["git_sha"], changed["producer"]["dirty"]) == ("f" * 40, True)
    changed["producer"] |= {"git_sha": None, "dirty": None}
    assert changed == json.loads(before.pop(manifest))
    assert all(path.read_bytes() == data for path, data in before.items())


# ---------------------------------------------------------------- the freeze's scorer on them


def test_fss_passes_with_a_ci_that_excludes_0(fss_scored: ising.Evaluation) -> None:
    """A peak at one eps_L at every size: 1/nu = 1 (planted exactly; the lag is the same eps_L
    at every L), within 0.15, the CI above 0 and every bootstrap draw giving 1/nu a value, so the
    dense verdict is a pass; the matched robustness row meets the same clauses, without a verdict
    of its own. (Whether the CI holds 1 exactly is left to chance: it spans ~0.02.)"""
    rows = _dynamic(fss_scored)
    assert rows["dense"]["verdict"] == "pass", rows
    for row in rows.values():
        assert abs(row["estimate"] - 1.0) <= 0.15 and row["lo"] > 0.0, row
        assert row["draws_used"] == 1.0, row


@pytest.mark.usefixtures("frozen_v1")
@pytest.mark.parametrize("mode", ["fss", "fixed_T", "nu065", "nu035"])
def test_each_L_peaks_at_the_planted_eps_L_less_the_window_lag(
    mode: str, request: pytest.FixtureRequest
) -> None:
    """The peaks table's variance rows: at every L and cadence the median peak sits within
    TOLERANCE of the planted eps_L less LAG, no replica is on an edge, all count, and the
    quartiles bracket the median. A peak read at the window's start or middle instead of its end
    is 0.4 or 0.2 eps_L off; a window counted in sweeps instead of samples, 0.12-0.28 eps_L
    (dense, L = 32 and 64) and more at the matched cadence; no detrend, 0.8 eps_L or more."""
    evaluation: ising.Evaluation = request.getfixturevalue(f"{mode}_scored")
    for (cadence, L), row in _peaks(evaluation).items():
        expected = v1_peak(mode, L) - LAG
        assert abs(row["peak_eps_median"] - expected) <= TOLERANCE, (cadence, L, row)
        assert row["edge_fraction"] == 0.0, row
        assert row["n"] == len(V1_SEEDS) * v1_replicas(mode, L), row
        assert row["peak_eps_q1"] <= row["peak_eps_median"] <= row["peak_eps_q3"], row


@pytest.mark.usefixtures("frozen_v1")
@pytest.mark.parametrize(("mode", "verdict"), [("nu065", "pass"), ("nu035", "fail")])
def test_the_verdict_turns_on_the_tolerance_of_0_5(
    mode: str, verdict: str, request: pytest.FixtureRequest
) -> None:
    """Peaks read at 0.6 (L/16)^drift: 1/nu = 1 - drift, 0.65 for nu065 and 0.35 for nu035, 0.15
    inside and outside the tolerance of 0.5. Each CI is narrow (half-width under 0.1) and above
    0, with every bootstrap draw giving 1/nu a value, so nothing but the tolerance decides: nu065
    passes, though any tolerance under ~0.3 would fail it, and nu035 fails, though any over ~0.7
    would pass it. The matched robustness row finds the same 1/nu."""
    expected = 1.0 - DRIFT[mode]
    rows = _dynamic(request.getfixturevalue(f"{mode}_scored"))
    assert rows["dense"]["verdict"] == verdict, rows
    for row in rows.values():
        assert abs(row["estimate"] - expected) <= 0.1, row
        assert (row["hi"] - row["lo"]) / 2 < 0.1 and row["lo"] > 0.0, row
        assert row["draws_used"] == 1.0, row


def test_fixed_T_fails_with_inverse_nu_near_0(fixed_T_scored: ising.Evaluation) -> None:
    """A peak at one temperature: eps_L_peak grows as L, 1/nu near 0. With the same lag in eps_L
    at every L the planted answer is 1 - d log(0.8 L/16 - LAG) / d log L = -0.16; the CI lies
    below 1 - 0.5, under the whole tolerance band, so the dense verdict is a fail, and the
    matched robustness row finds the same, without a verdict."""
    sizes = np.log(V1_SIZES)
    peaks = np.log([v1_peak("fixed_T", L) - LAG for L in V1_SIZES])
    expected = 1.0 - np.polyfit(sizes, peaks, 1)[0]
    rows = _dynamic(fixed_T_scored)
    assert rows["dense"]["verdict"] == "fail", rows
    for row in rows.values():
        assert abs(row["estimate"] - expected) <= 0.15 and row["hi"] < 0.5, (expected, row)


def test_edge_is_inconclusive_by_the_edge_rule_at_one_L(edge_scored: ising.Evaluation) -> None:
    """Four of each seed's six replicas at L = 32 peak on an edge of the search band, two on each,
    and are not located (edge_fraction 2/3 there, 7/12 had one strayed, 0 at the other sizes), so
    the dense verdict is inconclusive; the matched robustness row carries none. Every other clause
    of the rule passes, at both cadences: the located peaks give 1/nu near 1 with a narrow CI
    above 0, and at least 95% of the bootstrap draws give 1/nu a value (each seed keeps two
    located peaks at L = 32, so under 1% of draws take none). The edge rule alone decides:
    without it, with the edges pooled over the sizes (8 of 24), or with one edge read (4 of 12),
    this is a pass, or a 1/nu thrown off by peaks found at the band's other edge."""
    located = v1_peak("edge", EDGE_L, EDGE_REPLICAS) - LAG  # the fifth replica's, at every L
    for (cadence, L), row in _peaks(edge_scored).items():
        unlocated = EDGE_REPLICAS / EDGE_SEED_REPLICAS if L == EDGE_L else 0.0
        assert row["edge_fraction"] == pytest.approx(unlocated), (cadence, L, row)
        assert row["n"] == len(V1_SEEDS) * v1_replicas("edge", L), (cadence, L, row)
        assert abs(row["peak_eps_median"] - located) <= TOLERANCE, (cadence, L, row)
    rows = _dynamic(edge_scored)
    assert rows["dense"]["verdict"] == "inconclusive", rows
    for row in rows.values():
        assert abs(row["estimate"] - 1.0) <= 0.15 and row["lo"] > 0.0, row
        assert (row["hi"] - row["lo"]) / 2 < 0.1 and row["draws_used"] >= 0.95, row


def test_every_indicator_is_tabled_and_only_variance_is_judged(
    fss_scored: ising.Evaluation, frozen_v1: dict[str, Any]
) -> None:
    """AC1 and skewness peaks are reported, one row per indicator, cadence and L, never judged:
    the only dynamic rows are variance's, one verdict at the dense cadence and a matched
    robustness row."""
    names = frozen_v1["indicators"]["names"]
    cells = sorted((r["indicator"], r["cadence"], r["L"]) for r in fss_scored.tables["peaks"])
    assert cells == sorted((i, c, L) for i in names for c in CADENCES for L in V1_SIZES)
    assert len(_dynamic(fss_scored)) == len(CADENCES)


def test_an_unattainable_alarm_target_says_so_in_every_row(
    edge_scored: ising.Evaluation,
) -> None:
    """At 2 false alarms per tau_c, more than s(t) scored once per tau_c can give, no theta reaches
    the target: every alarm row (each indicator, cadence and L) says unattainable with theta NaN
    and the highest rate any theta gives, finite and below the target in the report's units, and
    none says censored = 1.0, the trace the -inf threshold left (#97)."""
    rows = edge_scored.tables["alarms"]
    alarm_indicators = CONTRACT["scoring.alarms.indicators"]
    assert sorted((r["indicator"], r["cadence"], r["L"]) for r in rows) == sorted(
        (i, c, L) for i in alarm_indicators for c in CADENCES for L in V1_SIZES
    )
    for row in rows:
        assert row["attainable"] is False and np.isnan(row["theta"]), row
        assert 0.0 < row["far_max"] < UNATTAINABLE * row["far_per_taus"], row
        assert row["censored"] != 1.0, row


def test_the_frozen_alarm_target_is_attainable_on_the_planted_nulls(
    fss_scored: ising.Evaluation, frozen_v1: dict[str, Any]
) -> None:
    """The control for the unattainable target: at the frozen 1 per 1000 tau_c some theta alarms
    more often than the target on the same nulls, so every row has a finite theta and a far_max
    above the target (one that only meets it is unattainable). With alarm scores every 200 tau_c
    each null path has one score, so far_max is 5 per 1000 tau_c whatever the data: this checks
    the attainable flag and theta, not the rate."""
    alarms = frozen_v1["scoring"]["alarms"]
    target = alarms["far"] * alarms["report_per"]
    for row in fss_scored.tables["alarms"]:
        assert row["attainable"] is True and np.isfinite(row["theta"]), row
        assert row["far_max"] > target, row


def test_a_ramp_over_another_eps_L_window_is_refused(
    v1: dict[str, V1], frozen_v1: dict[str, Any], tmp_path: Path
) -> None:
    """One ramp of an otherwise scorable dataset cooled from eps_L = 10 instead of 12: the scorer
    refuses the dataset and names that ramp, rather than score a ramp the frozen window does not
    describe."""
    copy = v1_copy(v1["fss"], tmp_path / "data")
    shutil.rmtree(copy.runs["ramp", 16, V1_SEEDS[1]])
    stray = v1_ramp_run(copy.root, "fss", 16, V1_SEEDS[1], eps_L=(10.0, V1_EPS_L[1]))
    with pytest.raises(ising.IsingDataError, match=re.escape(stray.name)):
        _evaluate(copy.root, _prereg(copy))


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.fixture(scope="module")
def frozen_copy(
    v1: dict[str, V1], frozen_v1: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, V1, str]:
    """A scratch repository whose prereg copy for fss's data is frozen, a copy of that data made
    by the freeze tag's commit from a clean tree, and that commit."""
    base = tmp_path_factory.mktemp("frozen")
    repo = base / "repo"
    path = repo / "paper" / "prereg" / V1_TOML.name
    path.parent.mkdir(parents=True)
    path.write_bytes(_prereg(v1["fss"]).encode("utf-8"))
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Test"),
        ("config", "user.email", "t@example.com"),
        ("config", "core.autocrlf", "false"),
        ("add", "."),
        ("commit", "-q", "-m", "prereg"),
    ):
        _git(repo, *args)
    assert main(["prereg", "freeze", "ising-ews", "--root", str(repo)]) == 0
    commit = _git(repo, "rev-parse", "prereg/ising-ews-v1^{commit}")
    copy = v1_copy(v1["fss"], base / "data")
    for run_dir in copy.runs.values():
        restamp(run_dir, commit, False)
    return path, copy, commit


@pytest.mark.parametrize(
    ("kind", "fault"),
    [
        ("ramp", "another commit"),
        ("null", "no commit"),
        ("qs", "a dirty tree"),
        ("null", "no pilot record"),
        ("ramp", "another pilot record"),
        ("qs", "a number for tau"),
        ("qs", "another tau.json"),
        ("qs", "another size's tau.json"),
    ],
)
def test_a_frozen_prereg_refuses_a_run_its_commit_and_record_did_not_make(
    frozen_copy: tuple[Path, V1, str],
    capsys: pytest.CaptureFixture[str],
    kind: str,
    fault: str,
) -> None:
    """Under a frozen prereg ews score takes only runs its tag's commit made from a clean tree
    (#26 decision 6), only ramps and null runs made against the pilot record it pins (#26 K), and
    only staircases whose dwell was read from the tau.json it pins at their L (freeze review,
    round 1). One run from another commit or from none (as runs made before the freeze), from a
    dirty tree, naming no record (as the exploratory null runs do) or another one, or a staircase
    given --tau 50, or another file, or the file pinned at another size, and nothing is scored or
    logged: the error names that run and no other."""
    path, copy, commit = frozen_copy
    run_dir = copy.runs[kind, 32, V1_SEEDS[0]]
    params = io.read_json(run_dir / "manifest.json")["params"]
    pilot, tau = params.get("pilot"), params["schedule"]["params"].get("tau_sha256")
    made = {
        "another commit": ("f" * 40, False),
        "no commit": (None, None),
        "a dirty tree": (commit, True),
    }
    dwells = {
        "a number for tau": None,
        "another tau.json": "e" * 64,
        "another size's tau.json": tau_source(16),
    }
    if fault in made:
        restamp(run_dir, *made[fault])
    elif fault in dwells:
        restamp_tau(run_dir, dwells[fault])
    else:
        restamp_pilot(
            run_dir, None if fault == "no pilot record" else {**pilot, "sha256": "e" * 64}
        )
    try:
        assert main(["score", "--prereg", str(path), "--data-dir", str(copy.root)]) == 1
    finally:
        restamp(run_dir, commit, False)
        if fault in dwells:
            restamp_tau(run_dir, tau)
        elif fault not in made:
            restamp_pilot(run_dir, pilot)
    err = capsys.readouterr().err
    assert {r.name for r in copy.runs.values() if r.name in err} == {run_dir.name}, err
    assert not (copy.root / "ews").exists()  # no result folder, no trials logged
