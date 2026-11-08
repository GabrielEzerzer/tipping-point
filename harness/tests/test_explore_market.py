"""The market exploratory pass (#30 M6) on planted series: a surprise that rises before planted
onsets reads as "greater" with an AUC above 0.5 and a flat one near 0.5; a segment touching an
invalid minute is left out and counted; a day from 2024 is refused; reading in blocks gives what
one read of the whole series gives.
"""

from datetime import date

import numpy as np
import pytest
import torch
from explore import market_latents, market_pass
from tipping_point.jepa import days, minutes

DAY_MINUTES = 1440
START = np.datetime64("2023-10-01", "ns").astype(np.int64)
MINUTE = market_latents.MINUTE_NS
BAR = market_pass.BAR_NS


def _planted(rise: float, n_days: int = 30, seed: int = 0) -> tuple[dict, np.ndarray, list]:
    """A noise series over n_days with an onset each day at 12:00, surprise ramping up by `rise`
    over the 6 hours before each onset (gone from the onset on), and one null stretch per day over
    the calm 15:00-23:55."""
    rng = np.random.default_rng(seed)
    M = n_days * DAY_MINUTES
    ts = START + MINUTE * np.arange(M, dtype=np.int64)
    surprise = 0.8 + 0.05 * rng.standard_normal(M)
    onsets, stretches = [], []
    for d in range(n_days):
        onset = d * DAY_MINUTES + 12 * 60
        surprise[onset - 360 : onset] += rise * np.linspace(0.0, 1.0, 360)
        onsets.append(START + MINUTE * onset)
        stretches.append({"ts_ns": int(START + MINUTE * (d * DAY_MINUTES + 15 * 60)), "bars": 107})
    pc = rng.standard_normal(M)
    columns = {"surprise": surprise, "pc_1": pc, "ts_ns": ts}
    return columns, np.array(onsets, np.int64), stretches


def _score(columns: dict, onsets: np.ndarray, stretches: list, valid: np.ndarray | None = None):
    ts = columns["ts_ns"]
    valid = np.ones(len(ts), bool) if valid is None else valid
    tables = {"trained": columns, "twin": columns}
    pairs = (("surprise", "level"), ("pc_1", "variance"))
    return market_pass.score_symbol(
        tables, ts, valid, onsets, stretches, pairs=pairs, horizons=(5, 12), draws=200
    )


def test_a_planted_rise_reads_greater_and_a_flat_series_near_half() -> None:
    columns, onsets, stretches = _planted(rise=0.3)
    rows, counts = _score(columns, onsets, stretches)
    assert counts["events"] == 30 and counts["controls"]["kept"] == 30  # 535 minutes: one each
    level = [r for r in rows if r["indicator"] == "level"]
    for r in level:
        assert r["trained"]["median_s_events"] > 0.5, r["h"]
        assert r["trained"]["auc"] > 0.9 and r["trained"]["auc_ci"][0] > 0.5
        assert r["margin"]["auc"] == 0.0  # the same table twice
    noise = [r for r in rows if r["indicator"] == "variance"]
    for r in noise:
        assert abs(r["trained"]["auc"] - 0.5) < 0.2
    found = market_pass.directions(
        [{"cell": "BTC->BTC", "kind": "same", **r} for r in rows],
        pairs=(("surprise", "level"),),
    )
    assert found["surprise level"]["same_asset"] == "greater"
    flat, onsets, stretches = _planted(rise=0.0, n_days=100)  # 30 a side reads 0.5 +- 0.15
    rows, _ = _score(flat, onsets, stretches)
    for r in rows:
        assert abs(r["trained"]["auc"] - 0.5) < 0.1, (r["column"], r["h"])
        lo, hi = r["trained"]["auc_ci"]
        assert lo < 0.5 < hi, (r["column"], r["h"])


def test_directions_say_disagree_and_never_choose() -> None:
    def row(cell: str, kind: str, h: int, s: float) -> dict:
        return {
            "cell": cell,
            "kind": kind,
            "column": "surprise",
            "indicator": "level",
            "h": h,
            "trained": {"median_s_events": s},
        }

    rows = [row("BTC->BTC", "same", 5, 0.3), row("ETH->ETH", "same", 5, -0.1)]
    rows.append(row("BTC->ETH", "cross", 5, 0.2))
    found = market_pass.directions(rows, pairs=(("surprise", "level"),))["surprise level"]
    assert found["same_asset"] == "disagree" and found["cross_asset"] == "greater"


def test_a_segment_touching_an_invalid_minute_is_excluded_and_counted() -> None:
    columns, onsets, stretches = _planted(rise=0.3, n_days=10)
    valid = np.ones(len(columns["ts_ns"]), bool)
    first = 12 * 60 - 5 * 5 - 10  # inside day 0's event segment at h = 5, before h = 12's starts
    valid[first] = False
    valid[15 * 60 + 100] = False  # inside day 0's control segment
    rows, counts = _score(columns, onsets, stretches, valid)
    assert counts["events_by_h"][5] == {"kept": 9, "too_early": 0, "invalid": 1}
    assert counts["events_by_h"][12] == {"kept": 10, "too_early": 0, "invalid": 0}
    assert counts["controls"] == {"kept": 9, "invalid": 1}
    for r in rows:
        assert r["trained"]["n_events"] == (9 if r["h"] == 5 else 10)
        assert r["trained"]["n_controls"] == 9


def test_an_onset_too_early_for_its_segment_is_counted_and_stretches_are_clipped() -> None:
    ts = START + MINUTE * np.arange(2 * DAY_MINUTES, dtype=np.int64)
    valid = np.ones(len(ts), bool)
    onsets = np.array([START + BAR * 40, START + BAR * 200], np.int64)  # 200 and 1000 minutes in
    stretch = [{"ts_ns": int(START - BAR * 10), "bars": 70}]  # 10 bars before the series
    events, controls, counts = market_pass.segments(ts, valid, onsets, stretch, horizons=(5,))
    assert counts["events_by_h"][5] == {"kept": 1, "too_early": 1, "invalid": 0}
    assert events[5][0, -1] == 5 * (200 - 5) - 1
    assert len(controls) == 1 and controls[0, 0] == 0  # 60 bars inside: one 299-minute span


def test_a_day_from_2024_is_refused() -> None:
    with pytest.raises(days.DataError, match="confirmatory"):
        market_latents.check([date(2024, 1, 1)])
    with pytest.raises(days.DataError, match="confirmatory"):
        market_latents.Folders(market_latents.Path("nowhere")).read("BTCUSDT", date(2024, 2, 3))
    assert market_latents.span_days("exploratory")[-1] == date(2023, 12, 31)
    assert market_latents.span_days("training")[0] == date(2023, 1, 18)


def test_blocks_lead_with_the_seven_days_before_them() -> None:
    wanted = market_latents.span_days("exploratory")
    (lead, block), (lead2, block2) = list(market_latents.blocks(wanted, 16))[:2]
    assert lead == [date(2023, 9, 24 + k) for k in range(7)] and block[0] == date(2023, 10, 1)
    assert lead2 == block[-7:] and block2[0] == date(2023, 10, 17)
    first = next(market_latents.blocks(market_latents.span_days("training")))
    assert first[0] == []


def test_reading_in_blocks_gives_what_one_read_gives() -> None:
    spec = minutes.MinuteSpec(dim=16, depth=1, pred_depth=1, heads=2, context=12)
    model = minutes.MarketJepa(spec)
    model.initialise(torch.Generator().manual_seed(0))
    model.eval()
    gen = torch.Generator().manual_seed(1)
    frames = torch.randn(100, days.MINUTE, spec.features, generator=gen)
    valid = torch.rand(100, generator=gen) > 0.1
    whole = minutes.readouts(model, frames, valid)
    parts = [(frames[a:b], valid[a:b]) for a, b in ((0, 30), (30, 37), (37, 100))]
    z, s = market_latents.carried(model, parts)
    torch.testing.assert_close(z, whole[0], equal_nan=True, rtol=0, atol=1e-6)
    torch.testing.assert_close(s, whole[1], equal_nan=True, rtol=0, atol=1e-6)
