"""Binance Vision's free USD-M archive: the files the regime labels read (#29 S1).

Per symbol, three kinds of file, each published with a `.CHECKSUM` (`<sha256>  <name>`) beside it:

  klines       daily/klines/<SYM>/1m/<SYM>-1m-<day>.zip: 1-minute klines, with a header line in
               later files and none in early ones
  metrics      daily/metrics/<SYM>/<SYM>-metrics-<day>.zip: 5-minute open interest and ratios,
               rows not always in order
  fundingRate  monthly/fundingRate/<SYM>/<SYM>-fundingRate-<month>.zip: one row per settlement

`fetch` keeps each in <dest>/<kind>/<SYM>/ beside its .CHECKSUM. A file already there that matches
its .CHECKSUM is skipped; a download whose SHA-256 differs is refused and never written; a file the
archive does not publish (404) is recorded as missing; a file that matches its .CHECKSUM but cannot
be parsed is kept and recorded as unreadable. <dest>/manifest.json lists every file of every span
fetched: URL, bytes, sha256, status and row checks (rows, header, rows out of order and, for the
daily kinds, stamps missing from the day's grid as [from, to) ranges, stamps given twice, rows off
the grid). A re-download whose sha256 differs from the one recorded keeps the old one as
`previous_sha256`. `collect` hands the labeller a span's columns, from verified, readable files
only.

The archive is CC BY-NC-SA 4.0: anything derived from it and published carries that licence and
the "Binance Vision" attribution.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from .. import __version__
from ..ising import io
from .msvol import runs

BASE_URL = "https://data.binance.vision/data/futures/um"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
KINDS = ("klines", "metrics", "fundingRate")
USER_AGENT = f"tipping-point/{__version__}"  # names the software, nothing about who runs it
LICENCE = "CC BY-NC-SA 4.0, Binance Vision"
MINUTE_MS = 60_000
STEP_MS = 5 * MINUTE_MS
DAY_MS = 1440 * MINUTE_MS
_SHA256 = re.compile(r"[0-9a-f]{64}")

# Each kind's columns in file order (the first is its stamp), and those read, with their types.
_COLUMNS = {
    "klines": (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ),
    "metrics": (
        "create_time",
        "symbol",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ),
    "fundingRate": ("calc_time", "funding_interval_hours", "last_funding_rate"),
}
_TYPES: dict[str, dict[str, pa.DataType]] = {
    "klines": {
        "open_time": pa.int64(),
        "open": pa.float64(),
        "high": pa.float64(),
        "low": pa.float64(),
        "close": pa.float64(),
        "volume": pa.float64(),
        "count": pa.int64(),
        "taker_buy_volume": pa.float64(),
    },
    "metrics": {"create_time": pa.timestamp("ms"), "sum_open_interest": pa.float64()},
    "fundingRate": {"calc_time": pa.int64(), "last_funding_rate": pa.float64()},
}

Get = Callable[[str], bytes | None]
Columns = dict[str, np.ndarray]


class ArchiveError(ValueError):
    """An archive file that is not fetched, fails its checksum, or cannot be read."""


def day_ms(day: date) -> int:
    """A UTC day's (or month's first day's) start in ms since the epoch."""
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * 1000


@dataclass(frozen=True)
class Source:
    """One archive file: kind, symbol, and day (yyyy-mm-dd) or, for fundingRate, month (yyyy-mm)."""

    kind: str
    symbol: str
    period: str

    @property
    def name(self) -> str:
        infix = "1m" if self.kind == "klines" else self.kind
        return f"{self.symbol}-{infix}-{self.period}.zip"

    @property
    def url(self) -> str:
        if self.kind == "fundingRate":
            return f"{BASE_URL}/monthly/fundingRate/{self.symbol}/{self.name}"
        interval = "/1m" if self.kind == "klines" else ""
        return f"{BASE_URL}/daily/{self.kind}/{self.symbol}{interval}/{self.name}"

    @property
    def path(self) -> str:
        """Where the file is kept, relative to the archive root."""
        return f"{self.kind}/{self.symbol}/{self.name}"

    @property
    def start_ms(self) -> int:
        return day_ms(
            date.fromisoformat(self.period if len(self.period) == 10 else self.period + "-01")
        )


def sources(symbols: Iterable[str], start: date, end: date) -> list[Source]:
    """Every file the days start..end (inclusive) need: each day's klines and metrics, and the
    funding file of each month they touch."""
    if end < start:
        raise ValueError(f"the span ends ({end}) before it starts ({start})")
    days = [start + timedelta(days=d) for d in range((end - start).days + 1)]
    months = sorted({f"{d:%Y-%m}" for d in days})
    out: list[Source] = []
    for symbol in symbols:
        out += [
            Source(kind, symbol, f"{d:%Y-%m-%d}") for kind in ("klines", "metrics") for d in days
        ]
        out += [Source("fundingRate", symbol, month) for month in months]
    return out


def read(kind: str, path: Path) -> tuple[Columns, bool]:
    """A file's columns as numpy arrays, its stamp in UTC ms, and whether it had a header line."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise ArchiveError(f"{path}: holds {len(names)} files, not one CSV")
            data = archive.read(names[0])
        columns, types = _COLUMNS[kind], _TYPES[kind]
        header = data.startswith(columns[0].encode())
        table = pacsv.read_csv(
            pa.py_buffer(data),
            read_options=pacsv.ReadOptions(column_names=None if header else list(columns)),
            convert_options=pacsv.ConvertOptions(
                column_types=types,
                include_columns=list(types),
                timestamp_parsers=["%Y-%m-%d %H:%M:%S"],
            ),
        )
    except (zipfile.BadZipFile, zlib.error, EOFError, pa.ArrowInvalid, KeyError) as error:
        raise ArchiveError(f"{path}: {error}") from error
    stamp = columns[0]
    if table[stamp].null_count:
        raise ArchiveError(f"{path}: {table[stamp].null_count} rows have no {stamp}")
    out = {name: table[name].to_numpy() for name in types}
    if out[stamp].dtype.kind == "M":  # metrics write their stamp as text
        out[stamp] = out[stamp].astype("datetime64[ms]").astype(np.int64)
    return out, header


def check(source: Source, columns: Columns, header: bool) -> dict[str, Any]:
    """A file's row checks (see the module notes)."""
    stamps = columns[_COLUMNS[source.kind][0]]
    found: dict[str, Any] = {
        "rows": len(stamps),
        "header": header,
        "unsorted": int(np.sum(np.diff(stamps) < 0)),
    }
    if source.kind == "fundingRate":
        return found
    step = MINUTE_MS if source.kind == "klines" else STEP_MS
    start, slots = source.start_ms, DAY_MS // step
    offset = stamps - start
    slot = offset // step
    on = (offset % step == 0) & (slot >= 0) & (slot < slots)
    counts = np.bincount(slot[on], minlength=slots)
    found["off_grid"] = int(np.sum(~on))
    found["duplicates"] = [_iso(start + int(s) * step) for s in np.flatnonzero(counts > 1)]
    found["gaps"] = [
        [_iso(start + int(a) * step), _iso(start + int(b) * step)]
        for a, b in zip(*runs(counts == 0), strict=True)
    ]
    return found


def verify(path: Path) -> str:
    """An archive file's sha256, checked against the .CHECKSUM beside it."""
    checksum = _checksum_path(path)
    if not (path.is_file() and checksum.is_file()):
        raise ArchiveError(f"{path}: not fetched (the file or its .CHECKSUM is absent)")
    expected = _expected(checksum.read_text(encoding="utf-8"), path.name, checksum)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ArchiveError(f"{path}: its SHA-256 is {actual}, but its .CHECKSUM says {expected}")
    return actual


def http_get(url: str) -> bytes | None:
    """The body at `url`, or None when the archive has no such file (404). The request carries a
    plain user agent and nothing else."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def fetch(symbols: Iterable[str], start: date, end: date, dest: Path, get: Get) -> dict[str, Any]:
    """Fetches and verifies the files of a span into `dest`, then writes dest/manifest.json, which
    keeps the entries of earlier spans. Returns this span's entries by path. A file that fails
    (network) is recorded so and fetched again by the next run."""
    manifest_path = dest / "manifest.json"
    files = io.read_json(manifest_path)["files"] if manifest_path.exists() else {}
    done = {s.path: _fetch(s, dest, get, files.get(s.path)) for s in sources(symbols, start, end)}
    files.update(done)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "source": BASE_URL,
        "licence": LICENCE,
        "files": dict(sorted(files.items())),
    }
    io.write_json(manifest_path, manifest, replace=True)
    return done


def _fetch(source: Source, dest: Path, get: Get, previous: dict[str, Any] | None) -> dict[str, Any]:
    path = dest / source.path
    entry: dict[str, Any] = {"url": source.url, "fetched": False}
    try:
        verify(path)
    except ArchiveError:
        try:
            checksum = get(source.url + ".CHECKSUM")
            data = None if checksum is None else get(source.url)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            return {**entry, "status": "failed", "error": str(error)}
        if checksum is None or data is None:
            return {**entry, "status": "missing"}
        try:
            expected = _expected(checksum.decode("utf-8", "replace"), source.name, source.url)
        except ArchiveError as error:
            return {**entry, "status": "refused", "error": str(error)}
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            return {**entry, "status": "refused", "sha256": actual, "expected": expected}
        path.parent.mkdir(parents=True, exist_ok=True)
        _replace(path, data)
        _replace(_checksum_path(path), checksum)
        entry["fetched"] = True
    sha256 = verify(path)
    entry |= {"bytes": path.stat().st_size, "sha256": sha256}
    before = previous or {}
    changed = before.get("sha256") if before.get("sha256") not in (None, sha256) else None
    if changed or before.get("previous_sha256"):
        entry["previous_sha256"] = changed or before["previous_sha256"]
    try:
        columns, header = read(source.kind, path)
    except ArchiveError as error:  # kept and recorded so the run goes on; collect refuses it
        return {**entry, "status": "unreadable", "error": str(error)}
    return {**entry, "status": "verified", "checks": check(source, columns, header)}


@dataclass(frozen=True)
class Span:
    """A span's columns by (symbol, kind), every file read ({url, bytes, sha256}), and the URLs of
    files the archive does not publish."""

    columns: dict[tuple[str, str], Columns]
    sources: list[dict[str, Any]]
    missing: list[str]


def collect(symbols: Iterable[str], start: date, end: date, root: Path) -> Span:
    """A span's files read from `root`. Each must match its .CHECKSUM, or be absent and recorded
    as not published in root/manifest.json; anything else is refused."""
    manifest_path = root / "manifest.json"
    files = io.read_json(manifest_path)["files"] if manifest_path.exists() else {}
    symbols = list(symbols)
    parts: dict[tuple[str, str], list[Columns]] = {}
    read_from: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in sources(symbols, start, end):
        path = root / source.path
        if not path.exists() and files.get(source.path, {}).get("status") == "missing":
            missing.append(source.url)
            continue
        sha256 = verify(path)
        parts.setdefault((source.symbol, source.kind), []).append(read(source.kind, path)[0])
        read_from.append({"url": source.url, "bytes": path.stat().st_size, "sha256": sha256})
    columns = {}
    for symbol in symbols:
        for kind in KINDS:
            chunks = parts.get((symbol, kind), [])
            columns[(symbol, kind)] = {
                name: np.concatenate([c[name] for c in chunks])
                if chunks
                else np.empty(0, _dtype(t))
                for name, t in _TYPES[kind].items()
            }
    return Span(columns, read_from, missing)


def _dtype(kind: pa.DataType) -> type:
    return np.float64 if pa.types.is_floating(kind) else np.int64


def _checksum_path(path: Path) -> Path:
    return path.with_name(path.name + ".CHECKSUM")


def _replace(path: Path, data: bytes) -> None:
    """Writes beside `path`, then renames. No fsync: every read checks the file against its
    .CHECKSUM, so one torn by a power cut is fetched again."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _expected(text: str, name: str, where: object) -> str:
    """The sha256 a .CHECKSUM gives for the file `name`."""
    parts = text.split()
    if len(parts) != 2 or parts[1].lstrip("*") != name or not _SHA256.fullmatch(parts[0]):
        raise ArchiveError(f"{where}: not a .CHECKSUM of {name}: {text[:200]!r}")
    return parts[0]


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
