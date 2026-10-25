"""Planted market data for the label tests (#29 S1-S2). Binance Vision's files in their own
formats, and 5-minute series with planted volatility regimes, so the fetcher, the bars and the
labeller can be checked against known answers in seconds. None of it is market data.

  Vision       a fake archive: {url: bytes} of zips and .CHECKSUMs, served by `get`, which logs
               every request
  vision       a whole span for some symbols: klines from a planted regime path, metrics, funding
  kline_csv    one day of 1-minute klines from per-minute log volatility, header optional
  metrics_csv  one day of 5-minute metrics; funding_csv one month of settlements
  regimes      a two-state path: calm stretches, planted episodes, short bursts (flickers)
  hmm          draws of the labeller's own model at a planted theta
  rv_series    5-minute realised variance whose log is a daily and weekend pattern, plus the
               regime's offset, plus noise
"""

from __future__ import annotations

import calendar
import hashlib
import io
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import numpy as np
from tipping_point.market import archive

STAMP = (2026, 10, 25, 0, 0, 0)  # every zip entry's time, fixed so the bytes repeat
# The header lines as Binance writes them (checked on 2023-01-02's files), kept apart from the
# parser's own column lists so a slip in those cannot hide behind the generator.
KLINES_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
    "taker_buy_quote_volume,ignore"
)
METRICS_HEADER = (
    "create_time,symbol,sum_open_interest,sum_open_interest_value,"
    "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,"
    "sum_taker_long_short_vol_ratio"
)
FUNDING_HEADER = "calc_time,funding_interval_hours,last_funding_rate"
MINUTE_MS = 60_000
DAY_NS = 86_400 * 10**9
BAR_NS = 300 * 10**9
DAY_BARS = 288


def zipped(member: str, data: bytes) -> bytes:
    info = zipfile.ZipInfo(member, date_time=STAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as bundle:
        bundle.writestr(info, data)
    return out.getvalue()


def checksum(name: str, data: bytes) -> bytes:
    """Binance's .CHECKSUM: `<sha256>  <name>`."""
    return f"{hashlib.sha256(data).hexdigest()}  {name}\n".encode()


@dataclass
class Vision:
    files: dict[str, bytes] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)

    def add(self, source: archive.Source, csv: bytes) -> None:
        """Publish `csv` as `source`, zipped, with its .CHECKSUM."""
        data = zipped(source.name.removesuffix(".zip") + ".csv", csv)
        self.files[source.url] = data
        self.files[source.url + ".CHECKSUM"] = checksum(source.name, data)

    def get(self, url: str) -> bytes | None:
        self.requests.append(url)
        return self.files.get(url)


def kline_csv(
    day: date,
    log_vol: np.ndarray,
    rng: np.random.Generator,
    *,
    price: float = 20_000.0,
    header: bool = True,
    idle: Iterable[int] = (),
    drop: Iterable[int] = (),
    duplicate: Iterable[int] = (),
    swap: Iterable[int] = (),
) -> tuple[bytes, float]:
    """A day of 1-minute klines (and its last close) whose log price moves in four steps a
    minute with sd exp(log_vol[minute]) / 2 each. `idle` minutes have no trades (flat at the last
    close, volume 0); `drop` minutes are left out, `duplicate` ones written twice, and each minute
    in `swap` trades places with the next row."""
    start = archive.day_ms(day)
    sd = np.exp(log_vol)[:, None] / 2
    steps = rng.normal(size=(1440, 4)) * sd
    quiet = np.zeros(1440, bool)
    quiet[list(idle)] = True
    steps[quiet] = 0.0
    path = price * np.exp(np.cumsum(steps.ravel())).reshape(1440, 4)
    opens = np.concatenate([[price], path[:-1, -1]])
    highs = np.maximum(opens, path.max(axis=1))
    lows = np.minimum(opens, path.min(axis=1))
    closes = path[:, -1]
    count = np.where(quiet, 0, rng.integers(50, 1000, 1440))
    volume = np.where(quiet, 0.0, rng.uniform(5.0, 50.0, 1440)).round(3)
    taker = (volume * rng.uniform(0.3, 0.7, 1440)).round(3)
    rows = [
        f"{start + i * MINUTE_MS},{opens[i]:.2f},{highs[i]:.2f},{lows[i]:.2f},{closes[i]:.2f},"
        f"{volume[i]:.3f},{start + (i + 1) * MINUTE_MS - 1},{volume[i] * closes[i]:.5f},"
        f"{count[i]},{taker[i]:.3f},{taker[i] * closes[i]:.5f},0"
        for i in range(1440)
    ]
    rows = _quirks(rows, drop, duplicate, swap)
    head = [KLINES_HEADER] if header else []
    return ("\n".join(head + rows) + "\n").encode(), float(closes[-1])


def metrics_csv(
    day: date,
    symbol: str,
    rng: np.random.Generator,
    *,
    oi: float = 100_000.0,
    drop: Iterable[int] = (),
    duplicate: Iterable[int] = (),
    swap: Iterable[int] = (),
) -> bytes:
    """A day of 5-minute metrics, open interest a random walk from `oi`; quirks as in kline_csv,
    by step."""
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    level = oi * np.exp(np.cumsum(rng.normal(0.0, 0.002, DAY_BARS)))
    rows = [
        f"{start + timedelta(minutes=5 * i):%Y-%m-%d %H:%M:%S},{symbol},{level[i]:.8f},"
        f"{level[i] * 20_000:.8f},1.9,1.05,2.1,0.95"
        for i in range(DAY_BARS)
    ]
    rows = _quirks(rows, drop, duplicate, swap)
    return ("\n".join([METRICS_HEADER, *rows]) + "\n").encode()


def funding_csv(month: date, rate: float = 0.0001) -> bytes:
    """A month of 8-hourly settlements, stamped a few ms late as Binance's often are."""
    start = archive.day_ms(month.replace(day=1))
    days = calendar.monthrange(month.year, month.month)[1]
    rows = [f"{start + i * 8 * 3_600_000 + (i % 3) * 4},8,{rate:.8f}" for i in range(3 * days)]
    return ("\n".join([FUNDING_HEADER, *rows]) + "\n").encode()


def _quirks(
    rows: list[str], drop: Iterable[int], duplicate: Iterable[int], swap: Iterable[int]
) -> list[str]:
    rows = list(rows)
    for i in swap:
        rows[i], rows[i + 1] = rows[i + 1], rows[i]
    twice = set(duplicate)
    gone = set(drop)
    return [
        r for i, row in enumerate(rows) if i not in gone for r in [row] * (2 if i in twice else 1)
    ]


def regimes(
    n: int,
    rng: np.random.Generator,
    calm: tuple[int, int] = (400, 900),
    episode: tuple[int, int] = (24, 96),
    burst: tuple[int, int] = (3, 8),
    burst_every: int = 3,
) -> np.ndarray:
    """A path of n bars (int8, 1 = high): calm stretches of `calm` bars, each followed by an
    episode of `episode` bars; every `burst_every`-th calm stretch carries a short burst of
    `burst` bars at its middle."""
    states = np.zeros(n, np.int8)
    t = k = 0
    while t < n:
        quiet = int(rng.integers(*calm))
        if k % burst_every == 1:
            middle = t + quiet // 2
            states[middle : middle + int(rng.integers(*burst))] = 1
        t += quiet
        length = int(rng.integers(*episode))
        states[t : t + length] = 1
        t += length
        k += 1
    return states


def hmm(theta: Sequence[float], n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n draws (z, states) of the two-state Gaussian HMM theta = (mu0, mu1, sigma0, sigma1, p00,
    p11), its chain starting stationary."""
    mu0, mu1, sigma0, sigma1, p00, p11 = theta
    u = rng.uniform(size=n)
    states = np.empty(n, np.int8)
    s = int(u[0] < (1 - p00) / (2 - p00 - p11))
    for t in range(n):
        if t:
            s = s if u[t] < (p11 if s else p00) else 1 - s
        states[t] = s
    noise = rng.normal(size=n)
    return np.where(states == 1, mu1 + sigma1 * noise, mu0 + sigma0 * noise), states


def rv_series(
    first: date,
    days: int,
    rng: np.random.Generator,
    states: np.ndarray | None = None,
    *,
    level: float = -13.0,
    mu: Sequence[float] = (0.0, 1.2),
    sigma: Sequence[float] = (0.35, 0.45),
    daily: float = 0.5,
    weekend: float = -0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ts_ns, rv, states) for `days` days of 5-minute bars from `first`: ln rv = level + a daily
    sine of amplitude `daily` + `weekend` on Saturdays and Sundays + mu[s] + sigma[s] noise."""
    n = days * DAY_BARS
    ts = archive.day_ms(first) * 1_000_000 + BAR_NS * np.arange(n, dtype=np.int64)
    states = regimes(n, rng) if states is None else states
    season = daily * np.sin(2 * np.pi * (ts % DAY_NS) / DAY_NS)
    season += weekend * ((ts // DAY_NS + 3) % 7 >= 5)
    high = states == 1
    noise = rng.normal(size=n) * np.where(high, sigma[1], sigma[0])
    return ts, np.exp(level + season + np.where(high, mu[1], mu[0]) + noise), states


def vision(symbols: Sequence[str], first: date, days: int, seed: int) -> Vision:
    """A fake archive holding `days` days from `first` of every kind for each symbol. Klines move
    with a planted regime path (per-minute log volatility -8, +0.6 in the high state) and a
    daily pattern; one minute an hour has no trades."""
    out = Vision()
    for k, symbol in enumerate(symbols):
        rng = np.random.default_rng([seed, k])
        states = regimes(days * DAY_BARS, rng)
        minute = np.arange(1440)
        price = 20_000.0 / (k + 1)
        for d in range(days):
            day = first + timedelta(days=d)
            bar_states = states[d * DAY_BARS : (d + 1) * DAY_BARS].repeat(5)
            log_vol = -8.0 + 0.6 * bar_states + 0.25 * np.sin(2 * np.pi * minute / 1440)
            csv, price = kline_csv(day, log_vol, rng, price=price, idle=range(17, 1440, 60))
            out.add(archive.Source("klines", symbol, f"{day:%Y-%m-%d}"), csv)
            metrics = metrics_csv(day, symbol, rng)
            out.add(archive.Source("metrics", symbol, f"{day:%Y-%m-%d}"), metrics)
        for month in sorted({(first + timedelta(days=d)).replace(day=1) for d in range(days)}):
            out.add(archive.Source("fundingRate", symbol, f"{month:%Y-%m}"), funding_csv(month))
    return out
