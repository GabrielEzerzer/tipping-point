"""Command line: python -m tipping_point.book <samples.parquet> [--out <range folder>]

Reduces one day's `book_samples` Parquet (an engine run's, with its manifest.json one folder up)
to `<out>/features/<SYMBOL>/<day>.parquet` (features.py) and prints the file's record (path, sha256,
bytes, rows, gap seconds) as one JSON line, which `market book` keeps in the range manifest under
the day's `reduced`. --out defaults to the range folder the samples sit in when `market book`
runs it as its reducer:

  python -m tipping_point.market book --range training --reduce "python -m tipping_point.book"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import features


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tipping_point.book", description=__doc__)
    parser.add_argument("samples", type=Path, help="a run's book_samples Parquet")
    parser.add_argument("--out", type=Path, help="default: the samples' `market book` range")
    args = parser.parse_args(argv)
    try:
        out = args.out or features.range_dir(args.samples)
        record = features.reduce_run(args.samples, out)
    except (features.FeatureError, OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
