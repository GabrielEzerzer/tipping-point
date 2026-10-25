"""5-minute bars (#29 S2; research pass section 2, items 1-2): the Garman-Klass variance summed
over each bar's 1-minute klines, missing bars as missing observations, never floored, open
interest at the bar's start stamp and funding as of it."""

import math

import numpy as np
from tipping_point.market import bars

START = 1_672_617_600_000  # 2023-01-02T00:00:00Z, in ms
NO_METRICS = {"create_time": np.empty(0, np.int64), "sum_open_interest": np.empty(0)}
NO_FUNDING = {"calc_time": np.empty(0, np.int64), "last_funding_rate": np.empty(0)}


def _flat_day(price: float = 100.0) -> dict[str, np.ndarray]:
    """1440 minutes that all trade, at a flat price."""
    ones = np.ones(1440)
    return {
        "open_time": START + 60_000 * np.arange(1440, dtype=np.int64),
        "open": ones * price,
        "high": ones * price,
        "low": ones * price,
        "close": ones * price,
        "volume": ones * 2.0,
        "count": np.full(1440, 10, np.int64),
        "taker_buy_volume": ones * 0.5,
    }


def _gk(o: float, h: float, low: float, c: float) -> float:
    return 0.5 * math.log(h / low) ** 2 - (2 * math.log(2) - 1) * math.log(c / o) ** 2


def test_rv_is_the_garman_klass_sum_over_the_minutes_that_traded() -> None:
    k = _flat_day()
    ohlc = [
        (100.0, 101.0, 99.5, 100.5),
        (100.5, 100.8, 100.1, 100.2),
        (100.2, 100.2, 99.0, 99.1),
        (99.1, 99.9, 99.1, 99.8),  # no trades: adds nothing, whatever its prices say
        (99.8, 100.3, 99.6, 100.0),
    ]
    for i, row in enumerate(ohlc):
        for name, value in zip(("open", "high", "low", "close"), row, strict=True):
            k[name][i] = value
    k["count"][3] = 0
    columns, checks = bars.bars(START, 1, k, NO_METRICS, NO_FUNDING)
    expected = sum(_gk(*row) for i, row in enumerate(ohlc) if i != 3)
    assert math.isclose(columns["rv"][0], expected, rel_tol=1e-12)
    assert columns["rv"][1] == 0.0  # a flat bar that traded is observed, at zero
    assert (columns["minutes"][0], columns["trades"][0], columns["close"][0]) == (5, 40, 100.0)
    assert (columns["volume"][0], columns["taker_buy_share"][0]) == (10.0, 0.25)
    assert len(columns["ts_ns"]) == 288 and columns["ts_ns"][0] == START * 1_000_000
    assert np.all(np.diff(columns["ts_ns"]) == 300 * 10**9)
    assert checks["bars"] == 288 and checks["bars_without_rv"] == 0


def test_a_bar_short_of_a_minute_or_without_trades_is_missing() -> None:
    k = _flat_day()
    k["count"][10:15] = 0  # bar 2: no minute trades
    k["volume"][10:15] = 0.0
    keep = np.arange(1440) != 7  # bar 1: minute 7 is absent
    columns, checks = bars.bars(
        START, 2, {n: v[keep] for n, v in k.items()}, NO_METRICS, NO_FUNDING
    )
    assert columns["minutes"][1] == 4 and columns["trades"][1] == 40
    assert np.all(np.isnan([columns[c][1] for c in ("rv", "close", "volume", "taker_buy_share")]))
    assert columns["minutes"][2] == 5 and math.isnan(columns["rv"][2])
    assert columns["volume"][2] == 0.0 and math.isnan(columns["taker_buy_share"][2])
    assert columns["close"][2] == 100.0
    assert np.all(columns["minutes"][288:] == 0) and np.all(np.isnan(columns["rv"][288:]))
    assert checks["bars_incomplete"] == 1 + 288 and checks["bars_without_rv"] == 2 + 288


def test_a_duplicate_minute_keeps_its_first_row_and_off_grid_rows_drop() -> None:
    k = _flat_day()
    extra = {n: v[[24, 24]].copy() for n, v in k.items()}
    extra["close"][:] = 250.0  # a second, different row for minute 24
    extra["open_time"][1] += 30_000  # and a row half a minute off the grid
    joined = {n: np.concatenate([k[n], extra[n]]) for n in k}
    columns, checks = bars.bars(START, 1, joined, NO_METRICS, NO_FUNDING)
    assert columns["close"][4] == 100.0
    assert (checks["kline_duplicate_rows"], checks["kline_rows_off_grid"]) == (1, 1)


def test_open_interest_joins_at_the_bar_start_and_funding_as_of_it() -> None:
    steps = START + 300_000 * np.arange(288, dtype=np.int64)
    values = 1000.0 + np.arange(288)
    rows = np.r_[150:288, 0:3, 4:150, 10]  # out of order, step 3 absent, step 10 twice
    stamps, oi = steps[rows], values[rows]
    oi[-1] = -1.0  # the second row for step 10 is never used
    hours = 3_600_000
    funding = {
        "calc_time": np.array([START + 8 * hours + 4, START - 8 * hours + 3], np.int64),
        "last_funding_rate": np.array([0.0002, 0.0001]),
    }
    metrics = {"create_time": stamps, "sum_open_interest": oi}
    columns, checks = bars.bars(START, 1, _flat_day(), metrics, funding)
    assert columns["oi"][0] == 1000.0 and columns["oi"][10] == 1010.0
    assert math.isnan(columns["oi"][3]) and checks["bars_without_oi"] == 1
    assert checks["metric_duplicate_rows"] == 1
    rate = columns["funding_rate"]
    assert rate[0] == 0.0001 and rate[96] == 0.0001  # 08:00's bar starts 4 ms before settlement
    assert rate[97] == 0.0002 and rate[-1] == 0.0002
    later = {name: column[:1] for name, column in funding.items()}  # only the 08:00 settlement
    rate = bars.bars(START, 1, _flat_day(), metrics, later)[0]["funding_rate"]
    assert np.all(np.isnan(rate[:97])) and np.all(rate[97:] == 0.0002)
