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


def guarded(get: archive.Get) -> archive.Get:
    return lambda url: get(span.refuse_url(url))


def fetch(
    symbols: Iterable[str],
    start: date,
    end: date,
    dest: Path,
    get: archive.Get,
    workers: int = WORKERS,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    span.refuse(start, end)
    manifest_path = dest / "manifest.json"
    files = io.read_json(manifest_path)["files"] if manifest_path.exists() else {}
    todo = archive.sources(list(dict.fromkeys(symbols)), start, end)
    safe = guarded(get)
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
        "files": dict(sorted(files.items())),
    }
    io.write_json(manifest_path, manifest, replace=True)
    return done
