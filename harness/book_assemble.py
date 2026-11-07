"""Merge `market book` range folders made on separate machines (one Kaggle kernel per symbol or
range) into one local data dir with one manifest per range, the layout jepa.days.ParquetSource
reads (#30 M4):

  <out>/<range>/manifest.json                    the sources' day entries merged, plus `assembled`
  <out>/<range>/features/<SYMBOL>/<day>.parquet  copied, sha256 checked against the entry

Every check runs before anything is written, and any problem refuses the whole merge:
  - a source folder without manifest.json, or whose manifest names another range than its folder;
  - a day of a range, for a symbol of market.book.SYMBOLS, that no source has imported (missing),
    or that two sources hold (duplicate, even with equal bytes: the kernels must split the days);
  - an entry not `imported`, reduced by another feature format than book.features.FORMAT, by a
    dirty tree or at another commit than --commit, or whose feature file is absent or does not
    hash to the entry's sha256.
It is idempotent: a feature file already in <out> with the entry's sha256 is kept, one with other
bytes is a problem, and a manifest equal to the one it would write is not rewritten. --dry-run
prints the plan and the problems and writes nothing.

  uv run python harness/book_assemble.py --commit fcd5b6f --out <data>/bybit-book-v3 \\
      <kaggle>/out-book-training-btc/bybit-book/training \\
      <kaggle>/out-book-training-eth/bybit-book/training \\
      <kaggle>/out-book-exploratory/bybit-book/exploratory [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from tipping_point.book.features import FORMAT
from tipping_point.ising import io
from tipping_point.market.book import BASE_URL, RANGES, SYMBOLS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Plan:
    manifests: dict[str, dict[str, Any]] = field(default_factory=dict)  # range -> merged manifest
    copies: list[tuple[Path, Path]] = field(default_factory=list)  # (source file, destination)
    kept: int = 0  # files already in place with the right bytes
    problems: list[str] = field(default_factory=list)


def plan(sources: Sequence[Path], out: Path, commit: str) -> Plan:
    """What merging `sources` into `out` would do, and every reason it must not (module notes)."""
    result = Plan()
    owner: dict[tuple[str, str], Path] = {}  # (range, key) -> the source holding it
    entries: dict[str, dict[str, Any]] = {}
    for src in map(Path, sources):
        path = src / "manifest.json"
        if not path.is_file():
            result.problems.append(f"{src}: no manifest.json (download incomplete?)")
            continue
        manifest = io.read_json(path)
        name = manifest.get("range")
        if name not in RANGES or name != src.name:
            result.problems.append(f"{src}: manifest range {name!r} is not the folder's")
            continue
        merged = result.manifests.setdefault(
            name,
            {
                "format": 1,
                "source": manifest.get("source", BASE_URL),
                "range": name,
                "days": {},
                "assembled": {"commit": commit, "feature_format": FORMAT, "from": []},
            },
        )
        merged["assembled"]["from"].append(
            {"folder": src.as_posix(), "manifest_sha256": _sha256(path)}
        )
        for key, entry in sorted(manifest.get("days", {}).items()):
            if (name, key) in owner:
                result.problems.append(f"{name} {key}: in both {owner[name, key]} and {src}")
                continue
            owner[name, key] = src
            entries[f"{name}/{key}"] = entry
            reduced = entry.get("reduced") or {}
            why = []
            if entry.get("status") != "imported":
                why.append(f"status {entry.get('status')!r}")
            if reduced.get("format") != FORMAT:
                why.append(f"feature format {reduced.get('format')}, not {FORMAT}")
            if not str(reduced.get("git_sha", "")).startswith(commit):
                why.append(f"reduced at {reduced.get('git_sha')}, not {commit}")
            if reduced.get("dirty") is not False:
                why.append("reduced from a dirty tree")
            if why:
                result.problems.append(f"{name} {key}: {'; '.join(why)}")
                continue
            file = src / reduced["path"]
            if not file.is_file() or _sha256(file) != reduced["sha256"]:
                result.problems.append(f"{name} {key}: {file} absent or not its sha256")
                continue
            target = out / name / reduced["path"]
            if target.is_file():
                if _sha256(target) == reduced["sha256"]:
                    result.kept += 1
                else:
                    result.problems.append(f"{name} {key}: {target} exists with other bytes")
                    continue
            else:
                result.copies.append((file, target))
            merged["days"][key] = entry
    for name in result.manifests:
        first, last = RANGES[name]
        wanted = [
            f"{s}/{first + timedelta(days=d):%Y-%m-%d}"
            for s in SYMBOLS
            for d in range((last - first).days + 1)
        ]
        missing = [k for k in wanted if (name, k) not in owner]
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            result.problems.append(f"{name}: {len(missing)} day(s) missing: {shown}")
        extra = sorted(k for (r, k) in owner if r == name and k not in set(wanted))
        if extra:
            result.problems.append(f"{name}: day(s) outside the range: {', '.join(extra[:5])}")
    return result


def apply(result: Plan, out: Path) -> list[str]:
    """Copies the planned files, checks each copy, and writes each range's manifest (unless an
    equal one is there). Returns what it did."""
    if result.problems:
        raise ValueError("refused: " + "; ".join(result.problems))
    done = []
    for file, target in result.copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".parquet.tmp")
        shutil.copyfile(file, part)
        if _sha256(part) != _sha256(file):
            part.unlink()
            raise OSError(f"copy of {file} does not match it")
        part.replace(target)
    done.append(f"copied {len(result.copies)}, kept {result.kept}")
    for name, manifest in sorted(result.manifests.items()):
        path = out / name / "manifest.json"
        if path.is_file() and io.read_json(path) == json.loads(json.dumps(manifest)):
            done.append(f"{name}: manifest unchanged, {len(manifest['days'])} days")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        io.write_json(path, manifest, replace=True)
        done.append(f"{name}: manifest written, {len(manifest['days'])} days")
    return done


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("sources", type=Path, nargs="+", help="range folders with manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="the merged data dir")
    parser.add_argument("--commit", required=True, help="the reducer's commit (a prefix)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args(argv)
    result = plan(args.sources, args.out, args.commit)
    for name, manifest in sorted(result.manifests.items()):
        print(f"{name}: {len(manifest['days'])} days pass their checks")
    print(f"to copy {len(result.copies)}, already in place {result.kept}")
    for problem in result.problems:
        print(f"problem: {problem}", file=sys.stderr)
    if args.dry_run:
        return 1 if result.problems else 0
    try:
        for line in apply(result, args.out):
            print(line)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
