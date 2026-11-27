# The study's days, and the refusal of every UTC day from a limit on, HELD_BACK (2024-01-01) unless
# a Python caller raises it: those days are held back as the test span, so no command downloads,
# reads or computes on one, and a run past HELD_BACK reads and writes only under held-back/.

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from ..ising import io

START = date(2021, 10, 1)
DEV_START = date(2022, 1, 1)
END = date(2023, 12, 31)
HELD_BACK = date(2024, 1, 1)
HELD_BACK_DIR = "held-back"
_DATE = re.compile(r"(?<!\d)(\d{4})-(?:(\d{2})(?:-(\d{2}))?(?!\d))?")


class HeldBackError(ValueError):
    pass


def refuse(*days: date, limit: date = HELD_BACK) -> None:
    for day in days:
        if day >= limit:
            raise HeldBackError(f"{day} is in the held-back test span (from {limit} on)")


def refuse_url(url: str, *, limit: date = HELD_BACK) -> str:
    for year, month, day in _DATE.findall(url):
        if date(int(year), int(month or 1), int(day or 1)) >= limit:
            raise HeldBackError(f"{url} names a day in the held-back test span")
    return url


def refuse_names(root: Path, names: Iterable[str], how: str, *, limit: date = HELD_BACK) -> None:
    for name in names:
        try:
            refuse_url(name, limit=limit)
        except HeldBackError:
            raise HeldBackError(
                f"{root} {how} {name}, which names a day in the held-back test span: the panel "
                f"reads no archive that holds one"
            ) from None


def refuse_archive(root: Path, *, limit: date = HELD_BACK) -> None:
    paths = root.rglob("*") if root.is_dir() else ()
    names = (path.relative_to(root).as_posix() for path in paths)
    refuse_names(root, names, "holds", limit=limit)


def manifest_files(root: Path, *, limit: date = HELD_BACK) -> dict[str, Any]:
    refuse_archive(root, limit=limit)
    path = root / "manifest.json"
    files = io.read_json(path)["files"] if path.exists() else {}
    refuse_names(root, files, "lists in its manifest", limit=limit)
    return files


def held_back_root(limit: date, *paths: Path) -> None:
    if limit <= HELD_BACK:
        return
    for path in paths:
        if HELD_BACK_DIR not in Path(path).resolve().parts:
            raise HeldBackError(
                f"{path} is not under a {HELD_BACK_DIR} folder: a run whose limit is past "
                f"{HELD_BACK} reads and writes only there"
            )
