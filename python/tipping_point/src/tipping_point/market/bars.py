"""5-minute bars on the UTC grid from Binance's 1-minute klines, open interest and funding (#29
S2; research pass section 2, items 1-2, and section 3's inputs).

Bar k of a symbol covers [t_k, t_k + 5 min):

  minutes          kline minutes present, 0-5
  trades           trades in those minutes
  rv               the Garman-Klass variance summed over the bar's minutes:
                   0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2 for each, 0 for a minute without trades
  close            its last minute's close
  volume           base-asset volume
  taker_buy_share  taker-buy volume over volume
  oi               open interest in contracts (sum_open_interest) stamped t_k
  funding_rate     the last funding rate settled at or before t_k

A bar short of any of its five minutes is a missing observation: rv, close, volume and the share
are NaN, and so is rv when no minute traded. Nothing is ever floored. The open-interest and funding
joins read stamps at or before t_k only, so a bar never sees a later value, whichever end of its
step Binance's stamp marks. A stamp given twice keeps its first row and rows off the grid are
dropped; both are counted in the checks.
"""

from __future__ import annotations

import math

import numpy as np

MINUTE_MS = 60_000
BAR_MS = 5 * MINUTE_MS
DAY_BARS = 288
GK = 2 * math.log(2) - 1


def bars(
    start_ms: int,
    days: int,
    klines: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    funding: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """One symbol's bars over `days` UTC days from `start_ms` (a midnight), from the columns of
    archive.collect, with their checks."""
    n = days * DAY_BARS
    rows, minute, (kline_off, kline_dups) = _grid(klines["open_time"], start_ms, MINUTE_MS, 5 * n)
    o, high, low, c = (klines[name][rows] for name in ("open", "high", "low", "close"))
    count = klines["count"][rows]
    traded = count > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        gk = np.where(traded, 0.5 * np.log(high / low) ** 2 - GK * np.log(c / o) ** 2, 0.0)
    bar = minute // 5

    def total(weights: np.ndarray) -> np.ndarray:
        return np.bincount(bar, weights=weights, minlength=n)

    complete = np.bincount(bar, minlength=n) == 5
    volume = total(klines["volume"][rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        share = total(klines["taker_buy_volume"][rows]) / volume
    close = np.full(n, np.nan)
    last = minute % 5 == 4
    close[bar[last]] = c[last]
    rv = np.where(complete & (total(traded) > 0), total(gk), np.nan)

    oi = np.full(n, np.nan)
    at, step, (metric_off, metric_dups) = _grid(metrics["create_time"], start_ms, BAR_MS, n)
    oi[step] = metrics["sum_open_interest"][at]

    starts = start_ms + BAR_MS * np.arange(n, dtype=np.int64)
    order = np.argsort(funding["calc_time"], kind="stable")
    settled = np.searchsorted(funding["calc_time"][order], starts, side="right") - 1
    funding_rate = np.full(n, np.nan)
    known = settled >= 0
    funding_rate[known] = funding["last_funding_rate"][order][settled[known]]

    columns = {
        "ts_ns": starts * 1_000_000,
        "minutes": np.bincount(bar, minlength=n).astype(np.int8),
        "trades": total(count).astype(np.int64),
        "rv": rv,
        "close": np.where(complete, close, np.nan),
        "volume": np.where(complete, volume, np.nan),
        "taker_buy_share": np.where(complete & (volume > 0), share, np.nan),
        "oi": oi,
        "funding_rate": funding_rate,
    }
    checks = {
        "kline_rows_off_grid": kline_off,
        "kline_duplicate_rows": kline_dups,
        "metric_rows_off_grid": metric_off,
        "metric_duplicate_rows": metric_dups,
        "bars": n,
        "bars_incomplete": int(np.sum(~complete)),
        "bars_without_rv": int(np.sum(np.isnan(rv))),
        "bars_without_oi": int(np.sum(np.isnan(oi))),
    }
    return columns, checks


def _grid(
    stamps: np.ndarray, start_ms: int, step_ms: int, count: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """The rows to use, as positions in `stamps` in grid order, and their grid indices: the first
    row of each stamp on the grid of `count` steps from start_ms. Also the number of rows dropped
    off the grid and as duplicates."""
    offset = stamps - start_ms
    index = offset // step_ms
    on = np.flatnonzero((offset % step_ms == 0) & (index >= 0) & (index < count))
    _, first = np.unique(index[on], return_index=True)
    rows = on[first]
    return rows, index[rows], (len(stamps) - len(on), len(on) - len(rows))
