# The panel's archive fetched in parallel through archive.py's own per-file step, so checksums,
# refusals, 404s, resumes, row checks, folders and manifest.json are archive.fetch's and
# archive.collect reads them. Every request passes the held-back refusal first.

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from ..ising import io
from ..market import archive
from . import span

WORKERS = 16


def guarded(get: archive.Get, *, limit: date = span.HELD_BACK) -> archive.Get:
    return lambda url: get(span.refuse_url(url, limit=limit))


def fetch(
    symbols: Iterable[str],
    start: date,
    end: date,
    dest: Path,
    get: archive.Get,
    workers: int = WORKERS,
    log: Callable[[str], None] | None = None,
    *,
    limit: date = span.HELD_BACK,
) -> dict[str, Any]:
    span.refuse(start, end, limit=limit)
    symbols = span.symbols_of(symbols)
    span.held_back_root(limit, dest)
    files = span.manifest_files(dest, limit=limit)
    manifest_path = dest / "manifest.json"
    todo = archive.sources(symbols, start, end)
    safe = guarded(get, limit=limit)
    done: dict[str, Any] = {}
    with ThreadPoolExecutor(workers) as pool:
        jobs = pool.map(lambda s: archive._fetch(s, dest, safe, files.get(s.path)), todo)
        for i, (source, entry) in enumerate(zip(todo, jobs, strict=True), 1):
            done[source.path] = entry
            if log and (i % 2000 == 0 or i == len(todo)):
                log(f"{i} of {len(todo)} files")
    files.update(done)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "source": archive.BASE_URL,
        "licence": archive.LICENCE,
        **({"limit": limit.isoformat()} if limit != span.HELD_BACK else {}),
        "files": dict(sorted(files.items())),
    }
    io.write_json(manifest_path, manifest, replace=True)
    return done
