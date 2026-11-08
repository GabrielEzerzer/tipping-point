"""The market segment builder (#30 M5) on a synthetic labels run of the real frozen labels file:
each event segment lands on its minutes at every horizon, exactly where the floor's rule puts it,
controls do not overlap, a missing minute excludes and counts a segment, the H = 1 share is the
planted one, the labels must be the frozen file's, and the confirmatory span stays closed without
a frozen market-ews file."""

import re
import subprocess
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from tipping_point.ews import market_segments as ms
from tipping_point.ews import prereg
from tipping_point.market import archive, msvol, nulls

REPO = Path(__file__).resolve().parents[4]
LABELS_V1 = REPO / "paper" / "prereg" / "market-labels-v1.toml"
TEMPLATE = REPO / "paper" / "prereg" / "TEMPLATE-market-ews.toml"
SPAN = 299  # Pipeline("variance", 60, 240).span
HORIZONS = (5, 6, 12, 24)
DAYS = 4
ONSET, EARLY = 600, 50  # event onsets, in bars from the run's first; EARLY has no room before it
HIGH = (520, 530)  # bars with H = 1 before ONSET, which only the longer horizons reach
STRETCHES = ((800, 200), (1100, 40))  # null stretches (start bar, bars): 3 and 0 controls
GAP = 4400  # a minute with no return, inside the second control segment


def filled(text: str) -> str:
    """The market-ews template with every placeholder set, as test_prereg_market_ews fills it."""
    text = text.replace('id = "<lowercase-slug>"', 'id = "market-ews"', 1)
    for key, value in {
        "detrend": "none",
        "method": "scale",
        "fit": "pilot",
        "noise": "bootstrap",
        "empty": "mark",
    }.items():
        text = re.sub(rf'^{key} = "<[^"]*>"', f'{key} = "{value}"', text, count=1, flags=re.M)
    text = re.sub(r'far_per_day = "<[^"]*>"', "far_per_day = 1.0", text)
    return re.sub(r'"<[^"]*>"', '"set"', text)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> tuple[prereg.Prereg, prereg.Prereg]:
    """A scratch repository with market-labels-v1 byte for byte and a filled market-ews file that
    names it, both frozen at their tags."""
    root = tmp_path_factory.mktemp("repo")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "core.autocrlf", "false")
    folder = root / "paper" / "prereg"
    folder.mkdir(parents=True)
    (folder / "market-labels-v1.toml").write_bytes(LABELS_V1.read_bytes())
    (folder / "market-ews-v1.toml").write_bytes(filled(TEMPLATE.read_text("utf-8")).encode())
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "preregs")
    prereg.freeze(root, "market-labels")
    prereg.freeze(root, "market-ews")
    return prereg.load(folder / "market-labels-v1.toml"), prereg.load(folder / "market-ews-v1.toml")


def _run(
    data: Path,
    labels: prereg.Prereg,
    first: date = date(2023, 3, 1),
    stamp: bool = False,
    **labeller: Any,
) -> Path:
    """A labels run of DAYS days from `first` for BTCUSDT: H = 1 on HIGH and from ONSET for 100
    bars, events at EARLY and ONSET, the null STRETCHES."""
    t0 = archive.day_ms(first) * 1_000_000
    n = DAYS * msvol.DAY_BARS
    h = np.zeros(n, np.int8)
    h[slice(*HIGH)] = 1
    h[ONSET : ONSET + 100] = 1
    ts = t0 + msvol.BAR_NS * np.arange(n, dtype=np.int64)
    rows = [
        {"ts_ns": int(t), "symbol": "BTCUSDT", "observed": True, "h": int(v)}
        for t, v in zip(ts, h, strict=True)
    ]
    episodes = [
        {"ts_ns": int(ts[b]), "symbol": "BTCUSDT", "bars": 40, "kind": "event", "missing": 0}
        for b in (EARLY, ONSET)
    ]
    stretches = [
        {
            "ts_ns": int(ts[b]),
            "symbol": "BTCUSDT",
            "end_ts_ns": int(ts[b + k]),
            "bars": k,
            "missing": 0,
        }
        for b, k in STRETCHES
    ]
    tables = {
        "labels": pa.Table.from_pylist(rows, schema=msvol.SCHEMAS["labels"]),
        "episodes": pa.Table.from_pylist(episodes, schema=msvol.SCHEMAS["episodes"]),
        "nulls": pa.Table.from_pylist(stretches, schema=msvol.SCHEMAS["nulls"]),
    }
    params: dict[str, Any] = {
        "start": first.isoformat(),
        "end": (first + timedelta(days=DAYS - 1)).isoformat(),
        "symbols": ["BTCUSDT"],
        "labeller": labels.doc["labeller"] | labeller,
    }
    if stamp:
        params["prereg"] = {"sha256": labels.sha256, "tag": labels.tag}
    return msvol.write_run(data, "synthetic", tables, params, [], [], {})


def _minutes(first: date = date(2023, 3, 1)) -> ms.Minutes:
    """Each minute's value is its own index, so a row says which minutes it holds; GAP has none."""
    values = np.arange(DAYS * 1440, dtype=np.float64)
    values[GAP] = np.nan
    return ms.Minutes(archive.day_ms(first), values)


def test_each_segment_lands_on_its_minutes(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    labels, _ = frozen
    run = _run(tmp_path, labels)
    got = ms.build(run, labels, {"BTCUSDT": _minutes()}, HORIZONS, SPAN)
    onset_minute = 5 * ONSET
    for h in HORIZONS:
        block = got.events["BTCUSDT", h]
        (row,) = block.x  # EARLY reaches before the series and is left out
        # the last minute closes at onset - 5h minutes: it starts one minute before
        assert row[-1] == onset_minute - 5 * h - 1
        assert row[0] == onset_minute - 5 * h - SPAN
        assert np.array_equal(row, np.arange(row[0], row[-1] + 1))
        # the floor's own rule gives the same rows
        floor = nulls.pre_onset_rows(_minutes().values, np.array([EARLY, ONSET]), h, SPAN)
        assert np.array_equal(floor, block.x)
    counts = {(c["kind"], c["horizon_bars"]): c for c in got.counts}
    assert counts["event", 5] == {
        "symbol": "BTCUSDT",
        "kind": "event",
        "horizon_bars": 5,
        "cut": 2,
        "kept": 1,
        "scored": 1,
        "excluded_invalid": 0,
        "excluded_outside": 1,
    }


def test_the_share_of_high_minutes_is_reported(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    """HIGH (bars 520-529, minutes 2600-2649) lies inside the clean window's reach only for the long
    horizons: none at 5 and 6 bars, minutes 2641-2649 at 12, all fifty at 24 (call 11)."""
    labels, _ = frozen
    got = ms.build(_run(tmp_path, labels), labels, {"BTCUSDT": _minutes()}, HORIZONS, SPAN)
    shares = {h: float(got.events["BTCUSDT", h].high_share[0]) for h in HORIZONS}
    assert shares == {5: 0.0, 6: 0.0, 12: 9 / SPAN, 24: 50 / SPAN}


def test_controls_do_not_overlap_and_a_gap_is_counted(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    labels, _ = frozen
    got = ms.build(_run(tmp_path, labels), labels, {"BTCUSDT": _minutes()}, HORIZONS, SPAN)
    block = got.controls["BTCUSDT"]
    start = 5 * STRETCHES[0][0]
    # three spans fit in the first stretch; the second holds GAP and is out; the last stretch none
    assert [row[0] for row in block.x] == [start, start + 2 * SPAN]
    assert np.all(np.diff(block.first_ms) >= SPAN * ms.MINUTE_MS)
    (control,) = [c for c in got.counts if c["kind"] == "control"]
    assert (control["cut"], control["kept"], control["excluded_invalid"]) == (3, 2, 1)
    assert control["excluded_outside"] == 0
    # the floor cuts the same spans
    stretch = [{"start": b, "bars": k} for b, k in STRETCHES]
    floor = nulls.null_rows(_minutes().values, stretch, SPAN)
    assert np.array_equal(floor[np.isfinite(floor).all(axis=1)], block.x)


def test_a_subsample_is_seeded_and_in_time_order(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    labels, _ = frozen
    run = _run(tmp_path, labels)
    series = {"BTCUSDT": _minutes()}
    one = [ms.build(run, labels, series, (5,), SPAN, subsample=1, seed=s) for s in (1, 1, 2, 3, 4)]
    picks = [b.controls["BTCUSDT"].x[0, 0] for b in one]
    assert picks[0] == picks[1] and set(picks) <= {5 * 800, 5 * 800 + 2 * SPAN}
    assert len(set(picks)) == 2  # some seed draws each
    (count,) = [c for c in one[0].counts if c["kind"] == "control"]
    assert (count["kept"], count["scored"]) == (2, 1)


def test_days_before_score_from_are_history_only(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    """An onset before score_from is not cut, a later onset whose segment reaches back before it
    still is, and a control segment starting before it is not cut."""
    labels, _ = frozen
    run = _run(tmp_path, labels)
    series = {"BTCUSDT": _minutes()}
    onset_day = date(2023, 3, 1) + timedelta(days=ONSET // msvol.DAY_BARS)  # EARLY is on day 0
    got = ms.build(run, labels, series, HORIZONS, SPAN, score_from=onset_day)
    whole = ms.build(run, labels, series, HORIZONS, SPAN)
    for h in HORIZONS:
        assert np.array_equal(got.events["BTCUSDT", h].x, whole.events["BTCUSDT", h].x)
        assert got.events["BTCUSDT", h].first_ms[0] < archive.day_ms(onset_day)  # reaches back
    events = {c["horizon_bars"]: c for c in got.counts if c["kind"] == "event"}
    assert all((c["cut"], c["kept"], c["excluded_outside"]) == (1, 1, 0) for c in events.values())
    assert np.array_equal(got.controls["BTCUSDT"].x, whole.controls["BTCUSDT"].x)
    # a day later the onset goes, and of the stretch's three controls the two starting on
    # onset_day; the third starts after midnight and stays
    after = onset_day + timedelta(1)
    later = ms.build(run, labels, series, HORIZONS, SPAN, score_from=after)
    assert all(len(later.events["BTCUSDT", h].x) == 0 for h in HORIZONS)
    (control,) = [c for c in later.counts if c["kind"] == "control"]
    assert control["cut"] == 1 and len(later.controls["BTCUSDT"].x) == 1
    assert np.array_equal(later.controls["BTCUSDT"].x, whole.controls["BTCUSDT"].x[-1:])
    assert later.controls["BTCUSDT"].first_ms[0] >= archive.day_ms(after)
    assert len(later.onsets["BTCUSDT"]) == 2  # every onset is still known, for eth-only


def test_the_labels_must_be_the_frozen_files(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    labels, _ = frozen
    series = {"BTCUSDT": _minutes()}
    other = _run(tmp_path / "a", labels, d_min=12)
    with pytest.raises(ms.SegmentError, match="labelled with other parameters"):
        ms.build(other, labels, series, (5,), SPAN)
    unfrozen = prereg.load(LABELS_V1)
    unfrozen = replace(unfrozen, sha256="0" * 64)
    with pytest.raises(prereg.PreregError, match="differs from the version frozen"):
        ms.build(_run(tmp_path / "b", labels), unfrozen, series, (5,), SPAN)
    with pytest.raises(ms.SegmentError, match="no labels for"):
        ms.build(_run(tmp_path / "c", labels), labels, {"ETHUSDT": _minutes()}, (5,), SPAN)


def test_the_confirmatory_span_needs_a_frozen_market_ews_file(
    tmp_path: Path, frozen: tuple[prereg.Prereg, prereg.Prereg]
) -> None:
    labels, scoring = frozen
    # a series reaching 2024, from a pilot run
    reaching = _minutes(date(2023, 12, 29))
    run = _run(tmp_path / "a", labels)
    with pytest.raises(ms.SegmentError, match="confirmatory span"):
        ms.build(run, labels, {"BTCUSDT": reaching}, (5,), SPAN)
    # a run labelling 2024, without and with the stamp, without and with the frozen file
    late = date(2024, 1, 2)
    bare = _run(tmp_path / "b", labels, first=late)
    with pytest.raises(ms.SegmentError, match="confirmatory span"):
        ms.build(bare, labels, {"BTCUSDT": _minutes(late)}, (5,), SPAN)
    with pytest.raises(ms.SegmentError, match="without market-labels-v1's stamp"):
        ms.build(
            bare, labels, {"BTCUSDT": _minutes(late)}, HORIZONS, SPAN, seed=30, scoring=scoring
        )
    stamped = _run(tmp_path / "c", labels, first=late, stamp=True)
    with pytest.raises(ms.SegmentError, match="confirmatory span"):
        ms.build(stamped, labels, {"BTCUSDT": _minutes(late)}, (5,), SPAN)
    got = ms.build(
        stamped, labels, {"BTCUSDT": _minutes(late)}, HORIZONS, SPAN, seed=30, scoring=scoring
    )
    assert len(got.events["BTCUSDT", 5].x) == 1
    # with the frozen file, its horizons and span are the ones cut, and no others
    with pytest.raises(ms.SegmentError, match="fixes horizons"):
        ms.build(stamped, labels, {"BTCUSDT": _minutes(late)}, (5,), SPAN, seed=30, scoring=scoring)
    with pytest.raises(ms.SegmentError, match="fixes span"):
        ms.build(
            stamped, labels, {"BTCUSDT": _minutes(late)}, HORIZONS, 239, seed=30, scoring=scoring
        )
    with pytest.raises(ms.SegmentError, match="fixes subsample: "):
        ms.build(
            stamped,
            labels,
            {"BTCUSDT": _minutes(late)},
            HORIZONS,
            SPAN,
            subsample=3,
            seed=30,
            scoring=scoring,
        )
    # an onset in 2024 in a run that claims to end before it, read with a series that does not
    claims = _run(tmp_path / "d", labels, first=date(2023, 12, 31))
    manifest = claims / "manifest.json"
    text = manifest.read_text("utf-8")
    assert '"end": "2024-01-03"' in text
    manifest.write_text(text.replace('"end": "2024-01-03"', '"end": "2023-12-31"'), "utf-8")
    early = ms.Minutes(archive.day_ms(date(2023, 12, 27)), np.zeros(5 * 1440))
    with pytest.raises(ms.SegmentError, match="onsets reach"):
        ms.build(claims, labels, {"BTCUSDT": early}, (5,), SPAN)


def test_returns_and_the_slot_profile() -> None:
    """Close-to-close returns leave a minute without a kline, and the one after it, empty; the slot
    profile divides each return by its slot's root mean square over the fit days."""
    start = archive.day_ms(date(2023, 3, 1))
    times = start + ms.MINUTE_MS * np.array([0, 1, 2, 4, 5, 5], np.int64)  # 3 missing, 5 twice
    closes = np.exp(np.array([0.0, 0.1, 0.3, 0.6, 1.0, 9.0]))
    got = ms.close_returns({"open_time": times, "close": closes}, start, 6)
    assert got.start_ms == start
    assert np.allclose(got.values, [np.nan, 0.1, 0.2, np.nan, np.nan, 0.4], equal_nan=True)
    rng = np.random.default_rng(3)
    n = 14 * 1440
    values = rng.standard_normal(n) * np.tile(np.repeat([1.0, 4.0], 720), 14)  # a loud afternoon
    series = ms.Minutes(start, values)
    scale = ms.slot_profile(series, (date(2023, 3, 1), date(2023, 3, 7)), "scale")
    assert scale.shape == (576,)
    flat = ms.deseasonalise(series, scale, "scale").values
    weekday = (np.arange(n) // 1440 + 3) % 7 < 5  # 2023-03-01 was a Wednesday
    morning = np.tile(np.arange(1440) < 720, 14)
    for part in (morning, ~morning):
        assert np.std(flat[part & weekday]) == pytest.approx(1.0, abs=0.08)
    # the fit reads only its days: a profile from day 8 on is another number
    later = ms.slot_profile(series, (date(2023, 3, 8), date(2023, 3, 14)), "scale")
    assert not np.allclose(scale, later)
    # "mean" subtracts each slot's mean return over the fit days
    drift = ms.Minutes(start, values + np.tile(np.repeat([0.5, -2.0], 720), 14))
    mean = ms.slot_profile(drift, (date(2023, 3, 1), date(2023, 3, 7)), "mean")
    centred = ms.deseasonalise(drift, mean, "mean").values
    for part in (morning, ~morning):
        assert np.mean(centred[part & weekday]) == pytest.approx(0.0, abs=0.08)
    assert np.allclose(centred - drift.values, -mean[ms._slots(drift)], equal_nan=True)
    with pytest.raises(ValueError, match="method"):
        ms.slot_profile(series, (date(2023, 3, 1), date(2023, 3, 7)), "median")


def test_the_slot_profile_is_fitted_before_the_confirmatory_span() -> None:
    late = ms.Minutes(archive.day_ms(date(2023, 12, 1)), np.ones(40 * 1440))
    with pytest.raises(ms.SegmentError, match="confirmatory"):
        ms.slot_profile(late, (date(2023, 12, 1), date(2024, 1, 5)), "scale")


def test_eth_only_keeps_the_eth_onsets_with_no_btc_onset_within_the_window() -> None:
    bar = msvol.BAR_NS
    eth = np.array([0, 100, 200, 300]) * bar
    btc = np.array([112, 188, 400]) * bar  # 12 bars after 100 and before 200: both inclusive
    assert ms.eth_only(eth, btc, 12).tolist() == [True, False, False, True]
    assert ms.eth_only(eth, btc, 11).tolist() == [True, True, True, True]
    assert ms.eth_only(eth, np.array([], np.int64), 12).all()
