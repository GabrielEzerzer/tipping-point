"""Bybit's free order-book archive, fetched and imported a day at a time, keeping no raw data
(#30 M1).

One file per linear perpetual per UTC day, `<day>_<SYMBOL>_ob<N>.data.zip` under BASE_URL: a
snapshot then 100 ms deltas, 500 levels a side until 2025-08-20 and 200 from 2025-08-21.

The days are split by purpose into RANGES, each with its own folder `<out>/<range>/` and its own
manifest.json, so each range can be fetched at its own time, on its own machine. `<out>` is a
parameter (by default <data dir>/bybit-book); nothing else is written anywhere.

Per day, `fetch` downloads the zip into `<range>/staging/`, checks it, has the engine's
`store import` (engine/src/store/import.rs) write the day's `book_samples` Parquet (20 levels a
side on the 100 ms grid) seeded with the day before's closing book (below), deletes the zip, runs
the `reduce` command if one is given (its argv with the Parquet's path appended; a non-zero exit
fails the day), deletes the samples and records the day. With `keep_samples`, the samples are
imported into `<range>/samples/` and kept instead.
Whatever the day put in staging is deleted when the day ends, imported or failed (samples an
import wrote before it failed too), so at most one day's zip and samples exist at a time; the
manifest records each day's peak (`peak_bytes`: the zip and the run on disk together, after the
import). A file something (a virus scanner) still holds after a few tries stays: the day's
`leftover` lists it, relative to the range folder, and it is tried again after each later day and
when the next run starts (a retry of the day lists it again while it is there). Staging must be a
plain folder: a link there (a symlink or a junction) is refused before anything else happens, and
a link inside it is deleted as a link, never followed.

The day before's closing book (#170). A day's file opens with a snapshot a second or so after
midnight, so the engine seeds each day's import with the closing book the day before's import left
(its `close.jsonl`: the whole book as of that day's end, then its few records stamped past the
end), and the day's samples from 00:00:00.000 on come from the book carried over until the day's
own records take over. Each imported day's closing book is kept as
`<range>/state/<SYMBOL>/<day>.jsonl` (tens of kB; the entry's `state`) and seeds the next day.
When a day's predecessor has no closing book (the first day of a span or of the range, or a range
fetched before #170) and this run has not tried it yet, the predecessor is fetched and imported
first for its closing book only: its samples are not reduced and its manifest entry, if it has
one, is left as it is, so a run of one symbol's days in order needs nothing more. The day's `seed`
says which closing book seeded it (`state`) and whether the engine used it (`used`); when not,
`why` says why (no closing book for the day before, or one the engine refuses rather than sample a
stale book: one whose last update id does not run on to the day's first record, or whose book goes
over 2 s without a record; engine/src/store/import.rs), and the day is imported unseeded, its
first second or two then a gap as before. Samples kept from an import made before #170 (no closing
book in the run) or not given the closing book this one is are imported again, since the engine
returns a run it finished before as it is.

The reducer. When its last line of output is a JSON object, the manifest keeps it as the day's
`reduced` (tipping_point.book prints its feature file's path, relative to the range folder, its
SHA-256, rows, gap seconds and its own version and feature format). A first word `python` or
`python3` runs as this interpreter (sys.executable), so `--reduce "python -m tipping_point.book"`
finds the package in this environment whatever `python` means on the PATH. Any other program is
named by its path, written with forward slashes (the command is split by shell rules, which drop
backslashes). Before the first download the run calls the reducer once with `--version` appended
and refuses to start unless it exits 0; when that prints a JSON object with a `format`, a day
reduced with another format is not done.

Integrity (#23). Bybit publishes no .CHECKSUM, so a download must have the size its
Content-Length gave and, when the ETag is a plain MD5 (an S3 object uploaded in one part, as every
file probed so far is), that MD5. The engine then refuses a zip that does not hold exactly its one
member, fails its CRC or holds a malformed record, and counts Bybit's `u` breaks between snapshots
(`id_gaps`), resyncs, crossed books and sample rows written while unsynced. The manifest keeps the
zip's URL, bytes and SHA-256, the sample count, the samples' bytes and SHA-256, and the engine's
whole integrity record, whether or not the samples are kept.

A day already imported is skipped when it was imported with the same `reduce` command, the file
its `reduced` record names (if any) is present with the reducer's current format, if this run
keeps samples, its Parquet is present and, if no closing book was given for it, the day before
still has none (a day before that failed or was missing may have one since). A day that fails
(network, a size or MD5 mismatch, the engine or the reducer refusing it, a file the system will
not let go, a malformed record) is recorded with why and the range goes on; the next run tries it
again, as it does a day the archive does not have (404, `missing`). The manifest is rewritten
after every day, so a stopped run loses at most the day in hand.

The confirmatory range is refused unless `confirmatory` is set (#30 M7).
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import re
import shutil
import subprocess
import sys
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
CLOSE = "close.jsonl"  # the closing book an import leaves in its run, the next day's seed
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
# (zip, its URL, data dir, the day before's closing book or None) -> the run's folder
Importer = Callable[[Path, str, Path, Path | None], Path]


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

    def run(archive: Path, address: str, data_dir: Path, seed: Path | None) -> Path:
        argv = [str(store), "import", str(archive), "--data-dir", str(data_dir), "--url", address]
        argv += ["--seed", str(seed)] if seed else []
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
    staging = _plain(where / "staging")
    given = list(reduce) if reduce else None  # as recorded in the manifest
    reducer = _interpreter(given) if given else None  # as run
    version = _probe(reducer) if reducer else None
    shutil.rmtree(staging, ignore_errors=True)  # whatever a stopped run left
    staging.mkdir(parents=True, exist_ok=True)  # still there if a file in it would not go
    manifest_path = where / "manifest.json"
    manifest = (
        io.read_json(manifest_path)
        if manifest_path.exists()
        else {"format": 1, "source": BASE_URL, "range": name, "days": {}}
    )
    done: dict[str, dict[str, Any]] = {}
    tried: set[str] = set()  # the days this run fetched, as days of the range or for their book
    for symbol in symbols:
        for day in days:
            key = f"{symbol}/{day:%Y-%m-%d}"
            entry = manifest["days"].get(key)
            state = _state(where, symbol, day - timedelta(days=1))  # the day before's
            if entry and _done(entry, where, keep_samples, given, version, state):
                done[key] = entry
                log(f"{key}: imported before, skipped")
                continue
            earlier = set(entry.get("leftover", [])) if entry else set()  # this day's own
            data_dir = where / "samples" if keep_samples else staging
            before = _files_in(staging, where)  # what earlier days left, listed with those days
            seed = _seed(symbol, day, where, tried, open_url, importer)
            entry = _day(
                symbol, day, where, data_dir, open_url, importer, keep_samples, reducer, seed
            )
            tried.add(key)
            _clear(staging)  # all the day put there, imported or failed, named by a run or not
            left = _files_in(staging, where)
            if leftover := sorted((left - before) | (left & earlier)):
                entry["leftover"] = leftover
            entry["reduce"] = given
            done[key] = manifest["days"][key] = entry
            manifest["days"] = dict(sorted(manifest["days"].items()))
            io.write_json(manifest_path, manifest, replace=True)
            log(f"{key}: {summary(entry)}")
    shutil.rmtree(staging, ignore_errors=True)
    return done


def _state(where: Path, symbol: str, day: date) -> Path:
    """Where the range keeps a day's closing book."""
    return where / "state" / symbol / f"{day:%Y-%m-%d}.jsonl"


def _seed(
    symbol: str,
    day: date,
    where: Path,
    tried: set[str],
    open_url: Open,
    importer: Importer,
) -> dict[str, Any]:
    """The closing book of the day before `day` to seed it with: `state`, its path relative to the
    range folder, or None and `why`. A day before without one that this run has not tried yet
    (`tried`, keyed `<SYMBOL>/<day>`) is fetched and imported first, for its closing book only
    (module notes), and what it put in staging goes before `day` is fetched (a file that will not
    go is listed with `day`)."""
    before = day - timedelta(days=1)
    state = _state(where, symbol, before)
    key = f"{symbol}/{before:%Y-%m-%d}"
    why = ""
    if not state.is_file() and key not in tried:
        tried.add(key)
        got = _day(symbol, before, where, where / "staging", open_url, importer, False, None, None)
        _clear(where / "staging")
        why = "" if got["status"] == "imported" else f" ({got['status']}: {got.get('error', '')})"
    if state.is_file():
        return {"state": state.relative_to(where).as_posix()}
    return {"state": None, "why": f"no closing book for {before}{why}"}


def _interpreter(reduce: list[str]) -> list[str]:
    """The reducer's argv, a first word `python` or `python3` replaced by this interpreter."""
    return [sys.executable, *reduce[1:]] if reduce[0] in ("python", "python3") else reduce


def _probe(reduce: list[str]) -> dict[str, Any] | None:
    """Runs the reducer once with `--version`; its JSON record, if it prints one. Raises BookError
    when it cannot run or exits non-zero, before anything is downloaded."""
    try:
        done = subprocess.run([*reduce, "--version"], capture_output=True, text=True)
    except OSError as error:
        raise BookError(f"the reducer {reduce} cannot run: {error}") from error
    if done.returncode != 0:
        why = done.stderr.strip()[-2000:]
        raise BookError(f"the reducer {reduce} --version exited {done.returncode}: {why}")
    return _reported(done.stdout)


def _done(
    entry: dict[str, Any],
    where: Path,
    keep: bool,
    reduce: list[str] | None,
    version: dict[str, Any] | None,
    before: Path,
) -> bool:
    """Whether a day recorded before needs nothing: imported with the same reducer at the format
    the reducer now reports, with the file it reported still there, when this run keeps samples
    its Parquet still there and, when no closing book was given for it, still none at `before`
    (the day before's, _state): one the engine refused stays refused."""
    kept = entry.get("parquet") and (where / entry["parquet"]).is_file()
    reduced = entry.get("reduced")
    named = reduced.get("path") if isinstance(reduced, dict) else None
    output = named is None or (where / named).is_file()
    if version and "format" in version and isinstance(reduced, dict):
        output = output and reduced.get("format") == version["format"]
    same = entry["status"] == "imported" and entry.get("reduce") == reduce
    unseeded = not (entry.get("seed") or {}).get("state")  # an entry from before #170 too
    return same and output and (kept or not keep) and not (unseeded and before.is_file())


def _day(
    symbol: str,
    day: date,
    where: Path,
    data_dir: Path,
    open_url: Open,
    importer: Importer,
    keep: bool,
    reduce: list[str] | None,
    seed: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fetches and imports one day seeded as `seed` says (_seed), or, when None, as a day before
    for its closing book only; returns its manifest entry. A kept run with no closing book, or not
    given the one given now, is imported again (module notes)."""
    address = url(symbol, day)
    archive = where / "staging" / file_name(symbol, day)
    entry: dict[str, Any] = {
        "url": address,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reduce": reduce,
    }
    if seed is not None:
        entry["seed"] = dict(seed)
    try:
        began = time.monotonic()
        try:
            got = _download(address, archive, open_url)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as error:
            return {**entry, "status": "failed", "error": f"download: {error}"}
        if got is None:
            return {**entry, "status": "missing", "error": "the archive has no such file (404)"}
        entry |= got | {"download_s": round(time.monotonic() - began, 1)}
        began = time.monotonic()
        try:
            given = where / seed["state"] if seed and seed["state"] else None
            run = importer(archive, address, data_dir, given)
            complete = io.read_json(run / "complete.json")
            if keep and (not (run / CLOSE).is_file() or (given and "seed" not in complete)):
                # Kept from an import made before #170 or not given this closing book: the engine
                # returns a run it finished before as it is, so the run goes and the day is
                # imported again. shortcut: a run made just now whose book ends unsynced has no
                # closing book either and is imported twice; read the engine's "imported already"
                # line if such days turn up.
                shutil.rmtree(run)
                run = importer(archive, address, data_dir, given)
                complete = io.read_json(run / "complete.json")
        except (BookError, OSError, ValueError) as error:
            return {**entry, "status": "failed", "error": str(error)}
        entry["import_s"] = round(time.monotonic() - began, 1)
        if seed is not None:  # what the engine made of the seed: used, or why not
            said = complete.get("seed") if given else None
            entry["seed"] |= said or {"used": False}
        if (run / CLOSE).is_file():
            state = _state(where, symbol, day)
            state.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run / CLOSE, state)
            entry["state"] = state.relative_to(where).as_posix()
        entry["peak_bytes"] = archive.stat().st_size + _bytes(run)
        _unlink(archive)
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
    except (BookError, OSError, KeyError, ValueError) as error:
        return {**entry, "status": "failed", "error": f"{type(error).__name__}: {error}"}


def _unlink(path: Path, tries: int = 5, wait_s: float = 0.5) -> None:
    """Deletes `path` if it is there, trying again while something (a virus scanner) holds it."""
    for attempt in range(tries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(wait_s)


def _plain(folder: Path) -> Path:
    """`folder`, unless it is a link (a symlink or a junction): BookError, the link left as it is.
    What the run deletes in staging must not be another folder's contents."""
    if folder.is_symlink() or folder.is_junction():
        raise BookError(
            f"{folder} is a link (a symlink or a junction); staging must be a plain folder, since "
            "the run deletes everything in it. Remove the link; to stage on another drive, point "
            "--out there."
        )
    return folder


def _clear(folder: Path) -> None:
    """Deletes what `folder` holds, each file with _unlink's tries; what will not go stays.
    Nothing outside `folder` is deleted: `folder` itself is refused if it is a link (_plain), and
    Path.walk lists a link inside it (a junction too) as a file, which os.unlink deletes as a link
    without following it."""
    for parent, dirs, files in _plain(folder).walk(top_down=False):
        for name in files:
            with contextlib.suppress(OSError):  # it stays, and the caller lists it
                _unlink(parent / name)
        for name in dirs:
            with contextlib.suppress(OSError):  # not empty: a file in it stayed
                (parent / name).rmdir()


def _files_in(folder: Path, where: Path) -> set[str]:
    """The files under `folder` (a link counted as one), relative to `where`."""
    return {(p / f).relative_to(where).as_posix() for p, _, files in folder.walk() for f in files}


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
    left = f"; not deleted: {entry['leftover']}" if entry.get("leftover") else ""
    if entry["status"] != "imported":
        return f"{entry['status']}: {entry.get('error', '')}{left}"
    return (
        f"imported, {entry['bytes'] / 1e6:.0f} MB in {entry['download_s']:.0f} s, "
        f"imported in {entry['import_s']:.0f} s, {entry['samples']} samples, "
        f"{entry['parquet_bytes'] / 1e6:.1f} MB Parquet, peak {entry['peak_bytes'] / 1e6:.0f} MB, "
        f"id_gaps {entry['integrity']['id_gaps']}{left}"
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
