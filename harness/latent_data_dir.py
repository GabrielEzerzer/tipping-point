# The data dir a latent pre-registration is scored from: its runs, their latents tables and its
# checkpoints, hard linked from data-dir-shaped folders and checked against their complete.json,
# with ews/ linked to the data dir whose trial log counts the file's budget.

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from explore.join_pilot import _link
from tipping_point import engine
from tipping_point.ews import ising, prereg
from tipping_point.ising import io
from tipping_point.jepa import extract, train

DESCRIPTION = """\
Assemble the data dir a latent pre-registration (a file with [latents]) is scored from.

<into>/ising/<run id>/ holds each run the file reads (its seeds and validation seed at its sizes,
and its pilot record), from the first --from folder holding it, with every latents table the file
pins from any --from folder; <into>/jepa/<training>/ holds each training it pins. Each --from is a
folder laid out as a data dir: a Kaggle kernel's out/, or a data dir. Files are hard linked, or
copied where a link is refused (across volumes). <into> must not exist yet.

<into>/ews is a link (a junction on Windows) to <ews-from>/ews: the trial log and the baseline
result stay those of that data dir, so a scoring from <into> charges the budget it is counted in.
Never give a scoring a data dir with an ews/ of its own: its empty trial log would let the budget
be spent a second time. To delete <into>, remove the ews link first (rmdir <into>/ews on Windows,
rm <into>/ews elsewhere): a delete that follows it would remove the trial log.

Every run and table is then checked: its files against its complete.json, each table's keys
against its run's frames (jepa.extract.verify), and each table against the complete.json it was
read over. Problems are listed and the exit code is 1. Then run harness/ising_experiment.py
--prereg <file> --data-dir <into> --check.
"""


def _wanted(doc: dict[str, Any], run: Path) -> bool:
    if run.name == doc["data"]["tau_c"]["pilot_run"]:
        return True
    if not (run / "manifest.json").is_file():
        return False
    params = io.read_json(run / "manifest.json")["params"]
    seeds = {*doc["data"]["seeds"], doc["latents"]["validation_seed"]}
    return params.get("seed") in seeds and params.get("L") in doc["data"]["L"]


def _tables_apart(run: Path) -> Callable[[str, list[str]], set[str]]:
    return lambda folder, _: {ising.LATENTS} if Path(folder) == run else set()


def _link_folder(target: Path, link: Path) -> None:
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
        )
    else:
        link.symlink_to(target, target_is_directory=True)


def assemble(path: Path, sources: Sequence[Path], into: Path, ews_from: Path) -> dict[str, int]:
    doc = prereg.load(path).doc
    if "latents" not in doc:
        raise prereg.PreregError(f"{path} has no [latents]: its runs need no data dir of their own")
    ews = Path(ews_from) / "ews"
    if not ews.is_dir():
        raise ising.IsingDataError(f"{ews} is no folder: the trial log must already live there")
    tables = {c.name for c in ising.checkpoints(doc)}
    trainings = {entry["training"] for entry in doc["latents"]["checkpoints"]}
    into.mkdir(parents=True)
    counts = {"runs": 0, "tables": 0, "trainings": 0}
    for source in sources:
        for run in sorted((source / "ising").glob("*")) if (source / "ising").is_dir() else []:
            target = into / "ising" / run.name
            if not run.is_dir() or not _wanted(doc, run):
                continue
            if not target.exists():
                shutil.copytree(run, target, copy_function=_link, ignore=_tables_apart(run))
                counts["runs"] += 1
            for name in sorted(tables):
                table, linked = run / ising.LATENTS / name, target / ising.LATENTS / name
                if table.is_dir() and not linked.exists():
                    shutil.copytree(table, linked, copy_function=_link)
                    counts["tables"] += 1
        for training in sorted(trainings):
            found, linked = source / "jepa" / training, into / "jepa" / training
            if found.is_dir() and not linked.exists():
                shutil.copytree(found, linked, copy_function=_link)
                counts["trainings"] += 1
    _link_folder(ews.resolve(), into / "ews")
    return counts


def problems(into: Path) -> list[str]:
    found = []
    for run in sorted((into / "ising").glob("*/complete.json")):
        damaged = train.verify(run.parent)
        found += [f"{run.parent.name}: {p}" for p in damaged]
        digest = ising.open_run(run.parent).complete_sha256
        for table in sorted((run.parent / ising.LATENTS).glob("*")):
            where = f"{run.parent.name}/{ising.LATENTS}/{table.name}"
            altered = train.verify(table)
            found += [f"{where}: {p}" for p in altered]
            read = io.read_json(table / "manifest.json")["params"]["dataset"]["complete_sha256"]
            if read != digest:
                found.append(f"{where}: read over another complete.json than this run's")
            elif not damaged and not altered:
                found += [f"{where}: {p}" for p in extract.verify(table)]
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument(
        "--from",
        dest="sources",
        type=Path,
        action="append",
        required=True,
        help="a folder laid out as a data dir; repeat it, the first holding a run gives its files",
    )
    parser.add_argument("--into", type=Path, required=True, help="the new data dir")
    parser.add_argument(
        "--ews-from",
        type=Path,
        default=engine.data_dir(),
        help="the data dir whose ews/ holds the trial log; default: the engine's own",
    )
    args = parser.parse_args(argv)
    try:
        counts = assemble(args.prereg, args.sources, args.into, args.ews_from)
    except (prereg.PreregError, ising.IsingDataError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.into}: {counts}; ews -> {(args.ews_from / 'ews').resolve()}")
    found = problems(args.into)
    for line in found:
        print(f"problem: {line}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
