"""A data dir that holds a Kaggle output's latents tables beside the local runs they were read from,
without touching either: <into>/ising/<run id>/ holds hard links to each exploratory run's files and
to the output's latents tables (a copy where links are refused), and to the tau_c records. A table
whose manifest names another complete.json than the local run's was read from other bytes, and is
refused here before it is ever scored.

Nothing writes into a data dir's files (io.write_file replaces them), so the links cannot carry a
change back. Delete the folder freely: a hard link is not a junction, and removing it leaves the
original alone. The scoring scripts take the folder as --data-dir.

  uv run --no-sync python harness/explore join_pilot --local <data dir> --output <kaggle out>
      --into <new folder>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from explore import common

RECORD = "pilot.json"


def _link(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _exploratory(run: Path) -> bool:
    """Whether a run folder is one this pipeline may read: its manifest says a seed from 1 to 5."""
    manifest = run / "manifest.json"
    if not manifest.is_file():
        return False
    return int(json.loads(manifest.read_text("utf-8"))["params"]["seed"]) in common.SEEDS


def join(local: Path, output: Path, into: Path) -> dict[str, int]:
    """Link the runs and tables into `into`, which must not exist yet; returns how many of each."""
    into.mkdir(parents=True)
    runs, tables, records = 0, 0, 0
    for source in sorted((local / "ising").iterdir()):
        target = into / "ising" / source.name
        if (source / RECORD).is_file():
            shutil.copytree(source, target, copy_function=_link)
            records += 1
        elif _exploratory(source):
            shutil.copytree(source, target, copy_function=_link)
            runs += 1
    for read in sorted((output / "ising").iterdir()):
        target = into / "ising" / read.name
        if not target.is_dir():
            continue  # a run this pipeline may not read, or one not on this machine
        digest = hashlib.sha256((target / "complete.json").read_bytes()).hexdigest()
        for table in sorted((read / "latents").iterdir()):
            manifest = json.loads((table / "manifest.json").read_text("utf-8"))
            if manifest["params"]["dataset"]["complete_sha256"] != digest:
                raise SystemExit(f"{read.name}/{table.name} was read from other bytes than local")
            shutil.copytree(table, target / "latents" / table.name, copy_function=_link)
            tables += 1
    return {"runs": runs, "tables": tables, "records": records}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, default=common.data_dir())
    p.add_argument("--output", type=Path, required=True, help="the kernel output's out/ folder")
    p.add_argument("--into", type=Path, required=True)
    args = p.parse_args(argv)
    print(join(args.local, args.output, args.into))
