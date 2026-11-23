# The study's days, and the refusal of every UTC day from HELD_BACK (2024-01-01) on: those days
# are held back as the test span, so no command downloads, reads or computes on one.

from __future__ import annotations

import re
from datetime import date

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
