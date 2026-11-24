# The study's days, and the refusal of every UTC day from HELD_BACK (2024-01-01) on: those days
# are held back as the test span, so no command downloads, reads or computes on one, and an
# archive root whose file names or manifest records carry such a day is refused.

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
_DATE = re.compile(r"(?<!\d)(\d{4})-(?:(\d{2})(?:-(\d{2}))?(?!\d))?")


class HeldBackError(ValueError):
    pass


def refuse(*days: date) -> None:
    for day in days:
        if day >= HELD_BACK:
            raise HeldBackError(f"{day} is in the held-back test span (from {HELD_BACK} on)")


def refuse_url(url: str) -> str:
    for year, month, day in _DATE.findall(url):
        if date(int(year), int(month or 1), int(day or 1)) >= HELD_BACK:
            raise HeldBackError(f"{url} names a day in the held-back test span")
    return url


def refuse_names(root: Path, names: Iterable[str], how: str) -> None:
    for name in names:
        try:
            refuse_url(name)
        except HeldBackError:
            raise HeldBackError(
                f"{root} {how} {name}, which names a day in the held-back test span: the panel "
                f"reads no archive that holds one"
            ) from None


def refuse_archive(root: Path) -> None:
    paths = root.rglob("*") if root.is_dir() else ()
    refuse_names(root, (path.relative_to(root).as_posix() for path in paths), "holds")


def manifest_files(root: Path) -> dict[str, Any]:
    refuse_archive(root)
    path = root / "manifest.json"
    files = io.read_json(path)["files"] if path.exists() else {}
    refuse_names(root, files, "lists in its manifest")
    return files
