# The panel's coins: USD-M symbols matching [A-Z0-9]+USDT, less EXCLUDED, with a 1-day kline file
# for every month of YEAR and a metrics file on both METRIC_DAYS, ranked by YEAR's quote volume;
# ALWAYS and the top OTHERS others. coins.txt holds the committed panel.

from __future__ import annotations

import csv
import hashlib
import io
import re
import urllib.parse
import zipfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from ..market import archive
from . import fetch, span

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINES = "data/futures/um/monthly/klines/"
METRICS = "data/futures/um/daily/metrics/"
PATTERN = re.compile(r"[A-Z0-9]+USDT")
EXCLUDED = frozenset({"BTCDOMUSDT", "DEFIUSDT", "USDCUSDT"})
YEAR = 2023
MONTHS = tuple(f"{YEAR}-{m:02d}" for m in range(1, 13))
METRIC_DAYS = (date(2021, 12, 1), date(2023, 12, 31))
ALWAYS = ("BTCUSDT", "ETHUSDT")
OTHERS = 28
COMMITTED = Path(__file__).with_name("coins.txt")


def committed() -> list[str]:
    return COMMITTED.read_text(encoding="utf-8").split()


def rank(volumes: Mapping[str, float]) -> list[str]:
    absent = [s for s in ALWAYS if s not in volumes]
    if absent:
        raise archive.ArchiveError(f"not eligible: {', '.join(absent)}")
    others = sorted((s for s in volumes if s not in ALWAYS), key=lambda s: (-volumes[s], s))
    return [*ALWAYS, *others[:OTHERS]]


def named(symbols: Iterable[str]) -> list[str]:
    return sorted(s for s in symbols if PATTERN.fullmatch(s) and s not in EXCLUDED)


def listing(get: archive.Get, prefix: str) -> tuple[list[str], list[str]]:
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        query = f"delimiter=/&prefix={prefix}"
        if marker:
            query += "&marker=" + urllib.parse.quote(marker, safe="")
        body = get(f"{BUCKET}?{query}")
        if body is None:
            raise archive.ArchiveError(f"{BUCKET}: no listing for {prefix}")
        root = ElementTree.fromstring(body)
        ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
        page_prefixes = [e.text or "" for e in root.findall(f"{ns}CommonPrefixes/{ns}Prefix")]
        page_keys = [e.text or "" for e in root.findall(f"{ns}Contents/{ns}Key")]
        prefixes += page_prefixes
        keys += page_keys
        if root.findtext(f"{ns}IsTruncated") != "true":
            return prefixes, keys
        marker = root.findtext(f"{ns}NextMarker") or max(page_prefixes + page_keys)


def symbols(get: archive.Get) -> list[str]:
    return [p.rstrip("/").rsplit("/", 1)[-1] for p in listing(get, KLINES)[0]]


def kline_url(symbol: str, month: str) -> str:
    return f"{archive.BASE_URL}/monthly/klines/{symbol}/1d/{symbol}-1d-{month}.zip"


def eligible(get: archive.Get, symbol: str) -> bool:
    keys = listing(get, f"{KLINES}{symbol}/1d/{symbol}-1d-{YEAR}-")[1]
    names = {key.rsplit("/", 1)[-1] for key in keys}
    if any(f"{symbol}-1d-{month}.zip" not in names for month in MONTHS):
        return False
    for day in METRIC_DAYS:
        key = f"{METRICS}{symbol}/{symbol}-metrics-{day:%Y-%m-%d}.zip"
        if key not in listing(get, key)[1]:
            return False
    return True


def verified(get: archive.Get, url: str) -> bytes:
    name = url.rsplit("/", 1)[-1]
    checksum = get(url + ".CHECKSUM")
    data = None if checksum is None else get(url)
    if checksum is None or data is None:
        raise archive.ArchiveError(f"{url}: not published")
    expected = archive._expected(checksum.decode("utf-8", "replace"), name, url)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise archive.ArchiveError(f"{url}: its SHA-256 is {actual}, its .CHECKSUM says {expected}")
    return data


def month_volume(data: bytes, month: str) -> float:
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        (member,) = bundle.namelist()
        text = bundle.read(member).decode("utf-8")
    first = archive.day_ms(date.fromisoformat(month + "-01"))
    year, number = map(int, month.split("-"))
    after = archive.day_ms(date(year + number // 12, number % 12 + 1, 1))
    total = 0.0
    for row in csv.reader(io.StringIO(text)):
        if not row or row[0] == "open_time":
            continue
        if not first <= int(row[0]) < after:
            raise archive.ArchiveError(f"{member}: a row at {row[0]} lies outside {month}")
        total += float(row[7])
    return total


def quote_volume(get: archive.Get, symbol: str) -> float:
    return sum(month_volume(verified(get, kline_url(symbol, m)), m) for m in MONTHS)


def derive(get: archive.Get, workers: int = fetch.WORKERS) -> dict[str, float]:
    span.refuse(date(YEAR, 1, 1), *(date.fromisoformat(m + "-01") for m in MONTHS), *METRIC_DAYS)
    safe = fetch.guarded(get)
    names = named(symbols(safe))
    with ThreadPoolExecutor(workers) as pool:
        keep = pool.map(lambda s: eligible(safe, s), names)
        ok = [s for s, kept in zip(names, keep, strict=True) if kept]
        volumes = pool.map(lambda s: quote_volume(safe, s), ok)
        return dict(zip(ok, volumes, strict=True))
