# The study's days, and the refusal of every UTC day from a limit on, HELD_BACK (2024-01-01) unless
# a Python caller raises it: those days are held back as the test span, so no command downloads,
# reads or computes on one. Only a run past HELD_BACK reads and writes under held-back/.

from __future__ import annotations

import calendar
import os
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
_SYMBOL = re.compile(r"[A-Z0-9]+")


class HeldBackError(ValueError):
    pass


def refuse(*days: date, limit: date = HELD_BACK) -> None:
    for day in days:
        if day >= limit:
            raise HeldBackError(f"{day} is in the held-back test span (from {limit} on)")


def refuse_url(url: str, *, limit: date = HELD_BACK) -> str:
    for year, month, day in _DATE.findall(url):
        if last_day(int(year), month, day) >= limit:
            raise HeldBackError(f"{url} names a day in the held-back test span")
    return url


def last_day(year: int, month: str, day: str) -> date:
    if day:
        return date(year, int(month), int(day))
    if month:
        return date(year, int(month), calendar.monthrange(year, int(month))[1])
    return date(year, 12, 31)


def refuse_names(root: Path, names: Iterable[str], how: str, *, limit: date = HELD_BACK) -> None:
    for name in names:
        try:
            refuse_url(name, limit=limit)
        except HeldBackError:
            raise HeldBackError(
                f"{root} {how} {name}, which names a day in the held-back test span: the panel "
                f"reads no archive that holds one"
            ) from None


def symbols_of(symbols: Iterable[str]) -> list[str]:
    found = list(dict.fromkeys(symbols))
    for symbol in found:
        if not _SYMBOL.fullmatch(symbol):
            raise ValueError(
                f"{symbol!r} is not a coin symbol: capital letters and digits only, since it "
                f"names a folder under the archive root"
            )
    return found


def unlinked(root: Path, limit: date) -> list[Path]:
    found: list[Path] = []
    folders = [root]
    while folders:
        folder = folders.pop()
        with os.scandir(folder) as entries:
            listed = sorted(entries, key=lambda entry: entry.name)
        for entry in listed:
            path = folder / entry.name
            if entry.is_symlink() or entry.is_junction():
                raise HeldBackError(
                    f"{path} is a link: an archive root and a labels run hold none, so that "
                    f"every file is read where its name says it is"
                )
            if entry.name.casefold() == HELD_BACK_DIR:
                held_back_root(limit, path)
            found.append(path)
            if entry.is_dir(follow_symlinks=False):
                folders.append(path)
    return found


def refuse_archive(root: Path, *, limit: date = HELD_BACK) -> None:
    paths: list[Path] = []
    if root.is_dir():
        held_back_root(limit, root)
        paths = unlinked(root, limit)
    names = (path.relative_to(root).as_posix() for path in paths)
    refuse_names(root, names, "holds", limit=limit)


def manifest_files(root: Path, *, limit: date = HELD_BACK) -> dict[str, Any]:
    refuse_archive(root, limit=limit)
    path = root / "manifest.json"
    files = io.read_json(path)["files"] if path.exists() else {}
    refuse_names(root, files, "lists in its manifest", limit=limit)
    return files


def held_back_root(limit: date, *paths: Path) -> None:
    raised = limit > HELD_BACK
    for path in paths:
        inside = HELD_BACK_DIR in (part.casefold() for part in Path(path).resolve().parts)
        if raised and not inside:
            raise HeldBackError(
                f"{path} is not under a {HELD_BACK_DIR} folder: a run whose limit is past "
                f"{HELD_BACK} reads and writes only there"
            )
        if inside and not raised:
            raise HeldBackError(
                f"{path} is under a {HELD_BACK_DIR} folder, which holds the held-back test span: "
                f"a run whose limit is not past {HELD_BACK} neither reads nor writes there"
            )
