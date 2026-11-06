"""Bybit's free order-book archive, fetched and imported a day at a time, keeping no raw data
(#30 M1).

One file per linear perpetual per UTC day, `<day>_<SYMBOL>_ob<N>.data.zip` under BASE_URL: a
snapshot then 100 ms deltas, 500 levels a side until 2025-08-20 and 200 from 2025-08-21.

The days are split by purpose into RANGES, each with its own folder `<out>/<range>/` and its own
manifest.json, so each range can be fetched at its own time, on its own machine. `<out>` is a
parameter (by default <data dir>/bybit-book); nothing else is written anywhere.

Per day, `fetch` downloads the zip into `<range>/staging/`, checks it, has the engine's
`store import` (engine/src/store/import.rs) write the day's `book_samples` Parquet (20 levels a
side on the 100 ms grid), deletes the zip, runs the `reduce` command if one is given (its argv
with the Parquet's path appended; a non-zero exit fails the day), records the day and deletes the
samples. When the reducer's last line of output is a JSON object, the manifest keeps it as the day's
`reduced` (tipping_point.book prints its feature file's path, relative to the range folder, its
SHA-256, rows and gap seconds). With `keep_samples`, the samples are imported into
`<range>/samples/` and kept instead.
At most one day's zip and samples exist at a time; the manifest records each day's peak
(`peak_bytes`: the zip and the run on disk together, after the import).

Integrity (#23). Bybit publishes no .CHECKSUM, so a download must have the size its
Content-Length gave and, when the ETag is a plain MD5 (an S3 object uploaded in one part, as every
file probed so far is), that MD5. The engine then refuses a zip that does not hold exactly its one
member, fails its CRC or holds a malformed record, and counts Bybit's `u` breaks between snapshots
(`id_gaps`), resyncs, crossed books and sample rows written while unsynced. The manifest keeps the
zip's URL, bytes and SHA-256, the sample count, the samples' bytes and SHA-256, and the engine's
whole integrity record, whether or not the samples are kept.

A day already imported is skipped when it was imported with the same `reduce` command, the file
its `reduced` record names (if any) is present and, if this run keeps samples, its Parquet is
present. A day that fails (network, a size or MD5 mismatch, the engine or the reducer refusing
it) is recorded with why and the range goes on; the next run tries it again, as it does a day the
archive does not have (404, `missing`). The manifest is rewritten after every day, so a stopped
run loses at most the day in hand.

The confirmatory range is refused unless `confirmatory` is set (#30 M7).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from ..ising import io
from .archive import USER_AGENT

BASE_URL = "https://quote-saver.bycsi.com/orderbook/linear"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
OB200 = date(2025, 8, 21)  # 200 levels from this day, 500 before
RANGES = {  # the archive starts on 2023-01-18; splits as #30 decision 5 took them
    "training": (date(2023, 1, 18), date(2023, 9, 30)),
    "exploratory": (date(2023, 10, 1), date(2023, 12, 31)),
    "confirmatory": (date(2024, 1, 1), date(2026, 9, 30)),
}
CHUNK = 1 << 20
TABLE = Path("book_samples") / f"{0:020d}.parquet"  # an import's one file of book samples
_MD5 = re.compile(r'"?([0-9a-f]{32})"?')
# The integrity counts that flag a day. `outside_day` is kept in the manifest but flags nothing:
# every real day probed has a few records stamped just past its end (2 to 12).
_FLAGS = ("id_gaps", "errors", "resyncs", "crossed", "unsynced_samples")


class BookError(ValueError):
    """A day that could not be fetched, checked or imported, or a span that is refused."""


class Response(Protocol):
    headers: Any

    def read(self, size: int, /) -> bytes: ...

    def close(self) -> None: ...


Open = Callable[[str], Response | None]  # the body at a URL, or None when it is not published
Importer = Callable[[Path, str, Path], Path]  # (zip, its URL, data dir) -> the run's folder


def file_name(symbol: str, day: date) -> str:
    return f"{day:%Y-%m-%d}_{symbol}_ob{200 if day >= OB200 else 500}.data.zip"


def url(symbol: str, day: date) -> str:
    return f"{BASE_URL}/{symbol}/{file_name(symbol, day)}"


def http_open(address: str) -> Response | None:
    """The response at `address`, or None on a 404. The request carries a plain user agent and
    nothing else."""
    request = urllib.request.Request(address, headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def engine_importer(store: Path) -> Importer:
    """Runs the engine's `store import` (the binary at `store`) and returns the run's folder."""

    def run(archive: Path, address: str, data_dir: Path) -> Path:
        argv = [str(store), "import", str(archive), "--data-dir", str(data_dir), "--url", address]
        done = subprocess.run(argv, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise BookError(f"store import exited {done.returncode}: {done.stderr.strip()[-2000:]}")
        return run_dir(done.stdout)

    return run


def run_dir(stdout: str) -> Path:
    """The run's folder from `store import`'s output: the line `<run dir>: <table>=<rows> ...`."""
    for line in stdout.splitlines():
        head, sep, tail = line.rpartition(": ")
        if sep and re.fullmatch(r"(\w+=\d+ ?)+", tail):
            return Path(head)
    raise BookError(f"store import named no run: {stdout.strip()[-2000:]!r}")


def span(name: str, start: date | None, end: date | None, confirmatory: bool) -> list[date]:
    """The days of range `name` from `start` to `end` (inclusive; the range's ends by default)."""
    if name not in RANGES:
        raise BookError(f"{name!r} is not a range (one of {', '.join(RANGES)})")
    if name == "confirmatory" and not confirmatory:
        raise BookError("the confirmatory range (#30 M7) is fetched only with --confirmatory")
    first, last = RANGES[name]
    start, end = start or first, end or last
    if not first <= start <= end <= last:
        raise BookError(f"{start}..{end} is not inside the {name} range, {first}..{last}")
    return [start + timedelta(days=d) for d in range((end - start).days + 1)]


def fetch(
    name: str,
    symbols: Iterable[str],
    out: Path,
    open_url: Open,
    importer: Importer,
    start: date | None = None,
    end: date | None = None,
    confirmatory: bool = False,
    keep_samples: bool = False,
    reduce: Sequence[str] | None = None,
    log: Callable[[str], object] = lambda _: None,
) -> dict[str, dict[str, Any]]:
    """Fetches and imports the days of range `name` for each symbol (see the module notes).
    Returns this run's manifest entries, keyed `<SYMBOL>/<day>`."""
    days = span(name, start, end, confirmatory)
    where = out / name
    staging = where / "staging"
    shutil.rmtree(staging, ignore_errors=True)  # whatever a stopped run left
    staging.mkdir(parents=True)
    manifest_path = where / "manifest.json"
    manifest = (
        io.read_json(manifest_path)
        if manifest_path.exists()
        else {"format": 1, "source": BASE_URL, "range": name, "days": {}}
    )
    reducer = list(reduce) if reduce else None
    done: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        for day in days:
            key = f"{symbol}/{day:%Y-%m-%d}"
            entry = manifest["days"].get(key)
            if entry and _done(entry, where, keep_samples, reducer):
                done[key] = entry
                log(f"{key}: imported before, skipped")
                continue
            data_dir = where / "samples" if keep_samples else staging
            entry = _day(symbol, day, where, data_dir, open_url, importer, keep_samples, reducer)
            done[key] = manifest["days"][key] = entry
            manifest["days"] = dict(sorted(manifest["days"].items()))
            io.write_json(manifest_path, manifest, replace=True)
            log(f"{key}: {summary(entry)}")
    shutil.rmtree(staging, ignore_errors=True)
    return done


def _done(entry: dict[str, Any], where: Path, keep: bool, reduce: list[str] | None) -> bool:
    """Whether a day recorded before needs nothing: imported with the same reducer, with the file
    its reducer reported still there and, when this run keeps samples, its Parquet still there."""
    kept = entry.get("parquet") and (where / entry["parquet"]).is_file()
    reduced = entry.get("reduced")
    named = reduced.get("path") if isinstance(reduced, dict) else None
    output = named is None or (where / named).is_file()
    same = entry["status"] == "imported" and entry.get("reduce") == reduce
    return same and output and (kept or not keep)


def _day(
    symbol: str,
    day: date,
    where: Path,
    data_dir: Path,
    open_url: Open,
    importer: Importer,
    keep: bool,
    reduce: list[str] | None,
) -> dict[str, Any]:
    address = url(symbol, day)
    archive = where / "staging" / file_name(symbol, day)
    entry: dict[str, Any] = {
        "url": address,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reduce": reduce,
    }
    run: Path | None = None
    try:
        began = time.monotonic()
        try:
            got = _download(address, archive, open_url)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            return {**entry, "status": "failed", "error": f"download: {error}"}
        if got is None:
            return {**entry, "status": "missing", "error": "the archive has no such file (404)"}
        entry |= got | {"download_s": round(time.monotonic() - began, 1)}
        began = time.monotonic()
        try:
            run = importer(archive, address, data_dir)
            complete = io.read_json(run / "complete.json")
        except (BookError, OSError, ValueError) as error:
            return {**entry, "status": "failed", "error": str(error)}
        entry["import_s"] = round(time.monotonic() - began, 1)
        entry["peak_bytes"] = archive.stat().st_size + _bytes(run)
        archive.unlink()
        parquet = run / TABLE
        entry |= {
            "samples": complete["rows"]["book_samples"],
            "integrity": complete["integrity"],
            "parquet_bytes": parquet.stat().st_size,
            "parquet_sha256": _sha256(parquet),
            "parquet": parquet.relative_to(where).as_posix() if keep else None,
        }
        if reduce:
            began = time.monotonic()
            done = subprocess.run([*reduce, str(parquet)], capture_output=True, text=True)
            entry["reduce_s"] = round(time.monotonic() - began, 1)
            if done.returncode != 0:
                why = f"reduce exited {done.returncode}: {done.stderr.strip()[-2000:]}"
                return {**entry, "status": "failed", "error": why}
            if reported := _reported(done.stdout):
                entry["reduced"] = reported
        return {**entry, "status": "imported"}
    except BookError as error:
        return {**entry, "status": "failed", "error": str(error)}
    finally:
        archive.unlink(missing_ok=True)
        archive.with_name(archive.name + ".part").unlink(missing_ok=True)
        if run is not None and not keep:
            shutil.rmtree(run, ignore_errors=True)


def _reported(stdout: str) -> dict[str, Any] | None:
    """The reducer's record: its last line of output when that is a JSON object."""
    lines = stdout.strip().splitlines()
    try:
        record = json.loads(lines[-1]) if lines else None
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _download(address: str, archive: Path, open_url: Open) -> dict[str, Any] | None:
    """Streams the zip to `archive`; its bytes, SHA-256 and whether an MD5 ETag was checked.
    Raises BookError when the size or the MD5 is not the one the server gave."""
    response = open_url(address)
    if response is None:
        return None
    expected = response.headers.get("Content-Length")
    etag = _MD5.fullmatch(response.headers.get("ETag") or "")
    sha256, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    part = archive.with_name(archive.name + ".part")
    try:
        with open(part, "wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                sha256.update(chunk)
                md5.update(chunk)
                size += len(chunk)
    finally:
        response.close()
    if expected is None or int(expected) != size:
        raise BookError(f"download: {size} bytes, but the server said {expected}")
    if etag and etag.group(1) != md5.hexdigest():
        raise BookError(f"download: MD5 {md5.hexdigest()}, but the ETag says {etag.group(1)}")
    part.replace(archive)
    return {"bytes": size, "sha256": sha256.hexdigest(), "md5_checked": bool(etag)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def summary(entry: dict[str, Any]) -> str:
    if entry["status"] != "imported":
        return f"{entry['status']}: {entry.get('error', '')}"
    return (
        f"imported, {entry['bytes'] / 1e6:.0f} MB in {entry['download_s']:.0f} s, "
        f"imported in {entry['import_s']:.0f} s, {entry['samples']} samples, "
        f"{entry['parquet_bytes'] / 1e6:.1f} MB Parquet, peak {entry['peak_bytes'] / 1e6:.0f} MB, "
        f"id_gaps {entry['integrity']['id_gaps']}"
    )


def report(entries: dict[str, dict[str, Any]]) -> str:
    """One line per status, and the imported days whose integrity is not clean."""
    lines = []
    for status in ("imported", "missing", "failed"):
        keys = [k for k, e in entries.items() if e["status"] == status]
        lines.append(f"{status}: {len(keys)}" + ("" if status == "imported" else f" {keys}"))
    flagged = {
        key: {k: v for k, v in e["integrity"].items() if k in _FLAGS and v}
        for key, e in entries.items()
        if e["status"] == "imported" and any(e["integrity"].get(k) for k in _FLAGS)
    }
    lines.append(f"imported with integrity flags: {len(flagged)} {json.dumps(flagged)}")
    return "\n".join(lines)
