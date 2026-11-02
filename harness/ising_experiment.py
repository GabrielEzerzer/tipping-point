"""The Ising experiment of a latent pre-registration (#28, slice 5): from a frozen file to the
result folder and its figures.

  uv run python harness/ising_experiment.py --prereg paper/prereg/ising-latent-v1.toml \
      [--data-dir D] [--seed 26] [--device cuda] [--allow-stale-pilot] [--no-figures]
  uv run python harness/ising_experiment.py --figures-only <result folder>

<data dir>/ews/<run_id>/ holds what `ews score` writes, in its layout, and more:
  auc surrogates alarms peaks fss   the scorer's tables, every row with its checkpoint
  probes    a ridge probe's R^2 per checkpoint, L, target and column, trained beside twin, on the
            frames the file's [probes] sentences name
  margins   trained minus twin by model seed (ews.margins.table); `checkpoint` lists what a row
            reads
  curves    per checkpoint, L and plateau: each latent column's mean and L^2 Var, and with
            checkpoint "" the classic |m|'s, on every saved frame: what the figures draw against T
  baseline  the rows of the result [baseline] names that are read here: |m|'s kappa and its
            matched-cadence AUC, the parity reference, never scored again
  csv/      every table above as CSV, written from the Parquet: what the figures read
  figures/  drawn by harness/ising_figures.py in an isolated environment (FIGURE_ENVIRONMENT), since
            matplotlib is no dependency of the project. A figure step that fails loses nothing: the
            tables are written and logged first, the run exits 2, and it says how to draw them again

--figures-only is for a run that died with its result complete but its CSV not: it writes csv/ from
the result's Parquet, over what is there, then draws the figures. It scores nothing, so it charges
nothing and logs no trial, and it refuses a folder with no complete.json, whose cells may be
unlogged.

Refused, before anything is read: a file not frozen, or short of a value (prereg.problems' strict
check), a scoring whose cells would take the file past its trial budget, and one that would score
past a torn result of this file (prereg.charge). Then what ising.admit refuses, a table extracted
under another rule than latents.extraction gives its role among it, and a baseline result that is
not the file's on these runs, and a checkpoint that is not the one it pins.
Probes and curves run before the cells, so that a fault in either costs minutes rather than a night.

--planted scores a file with no freeze behind it, for the harness alone: it refuses a data dir
holding any run the planted generator did not write (harness/ews_planted.py), so that it can never
read the confirmatory data early, and frames are embedded by ews_planted.planted_embedding.

Run time: see README, "The latent experiment". On the project's laptop the whole of ising-latent-v1
is projected at days, not a night, and most of it is the surrogate and alarm tables, which no
hypothesis judges, and the probes' embeddings on the CPU: --device cuda takes those to a GPU, if the
whole run goes there. There is no resume.
"""

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import ews_planted
import numpy as np
import pyarrow.parquet as pq
import torch
from tipping_point import engine
from tipping_point.ews import ising, margins, prereg, scoring
from tipping_point.ews.__main__ import score
from tipping_point.ising import io
from tipping_point.jepa import probes, readout, train

FIGURES = Path(__file__).with_name("ising_figures.py")
FIGURE_ENVIRONMENT = (
    "uv", "run", "--isolated", "--no-project", "--python", "3.13", "--with", "matplotlib==3.10.3",
)  # fmt: skip
SITES = 65_536  # sites per embedding batch, as jepa extract and probe take them
Row = dict[str, Any]
Embed = Callable[[ising.Checkpoint, Sequence[probes.Frames]], np.ndarray]


def checkpoint_embedder(
    data_dir: Path, passes: Sequence[ising.Checkpoint], device: str | None
) -> Embed:
    """The seam's real side: a checkpoint's embeddings of frames, from the files a training left in
    <data dir>/jepa/<training>/. Every file is held to the sha256 the prereg pins before anything is
    scored, and again as it is loaded."""
    dev = torch.device(device or train.default_device())
    train.deterministic(dev)

    def path(checkpoint: ising.Checkpoint) -> Path:
        training, step = checkpoint.name.rsplit("-step", 1)
        name = train.TWIN if int(step) == 0 else train.FINAL
        return Path(data_dir) / "jepa" / training / name

    bad = [
        str(path(c))
        for c in passes
        if not path(c).is_file() or train.sha256(path(c).read_bytes()) != c.sha256
    ]
    if bad:
        raise ising.IsingDataError(f"checkpoints missing, or not the sha256 the prereg pins: {bad}")

    def embed(checkpoint: ising.Checkpoint, sets: Sequence[probes.Frames]) -> np.ndarray:
        model = train.load_model(path(checkpoint), dev)
        if model.sha256 != checkpoint.sha256:
            raise ising.IsingDataError(f"{path(checkpoint)} changed since it was checked")
        return np.concatenate(
            [readout.embed(model.model, f.packed, f.L, dev, SITES).numpy() for f in sets]
        )

    return embed


def baseline_rows(
    doc: dict[str, Any], data_dir: Path, scored: Sequence[ising.Run], commit: str | None
) -> list[Row]:
    """What a latent file reads of its [baseline]: |m|'s static kappa and matched-cadence AUC at the
    headline cell, from the result it names, which must be that version's, on the runs scored."""
    baseline = doc["baseline"]
    folder = Path(data_dir) / "ews" / baseline["result"]
    if not (folder / "complete.json").is_file():
        raise ising.IsingDataError(f"baseline.result: no complete result at {folder}")
    params = io.read_json(folder / "manifest.json")["params"]
    stamp = params["prereg"]
    if f"{stamp['id']}-v{stamp['version']}" != baseline["prereg"] or (
        commit is not None and stamp["commit"] != commit
    ):
        raise ising.IsingDataError(
            f"baseline.result {folder.name} was scored under {stamp['id']}-v{stamp['version']} at "
            f"{stamp['commit']}, not under {baseline['prereg']}, whose runs this file reads"
        )
    read = {record["run_id"]: record["complete_sha256"] for record in params["inputs"]}
    foreign = [r.run_id for r in scored if read.get(r.run_id) != r.complete_sha256]
    if foreign:
        raise ising.IsingDataError(f"baseline.result {folder.name} did not read: {foreign}")

    def table(name: str) -> list[Row]:
        return pq.read_table(folder / name / f"{0:020d}.parquet").to_pylist()

    def row(quantity: str, L: int, h: float, value: Row, *names: str) -> Row:
        estimate, lo, hi = (float(value[name]) for name in names)
        return {"checkpoint": "", "quantity": quantity, "L": L, "h": h, "estimate": estimate,
                "lo": lo, "hi": hi}  # fmt: skip

    return [
        row(r["quantity"], 0, float("nan"), r, "estimate", "lo", "hi")
        for r in table("fss")
        if r["quantity"] == "l2_var_abs_m"
    ] + [
        row(f"auc_{r['indicator']}", r["L"], r["h"], r, "auc", "auc_lo", "auc_hi")
        for r in table("auc")
        if r["observable"] == "abs_m" and r["cadence"] == "matched" and r["headline"]
    ]


def curve_rows(
    doc: dict[str, Any], scored: Sequence[ising.Run], passes: Sequence[ising.Checkpoint]
) -> list[Row]:
    """Each staircase's plateaus, pooled over seeds: the mean and L^2 Var of each latent column per
    checkpoint on the frames its table holds, and of |m| on every saved frame, the classic chi."""
    rows = []
    for L in sorted(doc["data"]["L"]):
        runs = [r for r in scored if r.L == L and r.kind == "staircase"]
        for name in ("", *(c.name for c in passes)):
            found = [ising.plateaus(run, "matched", name or None)[0] for run in runs]
            for T in sorted({round(t, 9) for f in found for t in f}):
                parts = [p for f in found for t, p in f.items() if round(t, 9) == T]
                for column in sorted(parts[0].latents) if name else ["abs_m"]:
                    values = np.concatenate([p.observable(column).ravel() for p in parts])
                    rows.append(
                        {
                            "checkpoint": name,
                            "L": L,
                            "temperature": T,
                            "eps_L": float(runs[0].eps(T)),
                            "observable": column,
                            "mean": float(values.mean()),
                            "l2_var": float(L**2 * values.var(ddof=1)),
                            "frames": len(values),
                        }
                    )
    return rows


def _arms(doc: dict[str, Any]) -> dict[str, Row]:
    """Per checkpoint, the columns a probe row names it by: role, trained or twin, model seed."""
    return {
        ising.latents_name(entry["training"], step): {
            "role": entry["role"],
            "arm": arm,
            "model_seed": entry["model_seed"],
            "sha256": entry[pin],
        }
        for entry in doc["latents"]["checkpoints"]
        for step, arm, pin in (
            (int(doc["latents"]["steps"]), "trained", "trained_sha256"),
            (0, "twin", "twin_sha256"),
        )
    }


def probe_tables(
    doc: dict[str, Any],
    data_dir: Path,
    scored: Sequence[ising.Run],
    passes: Sequence[ising.Checkpoint],
    embed: Embed,
    seed: int,
) -> tuple[list[Row], dict[tuple[str, str, int, str], np.ndarray]]:
    """The probes table, and each quantity's data draws per (target, r2 or selectivity, L,
    checkpoint): fitted on the validation seed's staircase and scored on the scored seeds' at each
    L, on the frames probes.fit and probes.test state, every checkpoint on one draw of them. Each
    quantity resamples seeds, then replicas, from one RNG seed, so column b of the draws is the same
    resample for every checkpoint: the pairing the margin rule asks for."""
    limit, probe_seed = prereg.probe_protocol(doc)
    alpha, wanted = float(doc["probes"]["ridge_alpha"]), doc["probes"]["targets"]
    fit_seed, n = doc["latents"]["validation_seed"], int(doc["scoring"]["bootstrap"])
    complete, _ = ising.find_runs(data_dir)
    arms = _arms(doc)
    rows: list[Row] = []
    draws: dict[tuple[str, str, int, str], np.ndarray] = {}
    for L in sorted(doc["data"]["L"]):
        fit_runs = [r for r in complete if (r.kind, r.L, r.seed) == ("staircase", L, fit_seed)]
        if len(fit_runs) != 1:
            raise ising.IsingDataError(
                f"probes.fit: {len(fit_runs)} complete staircases of seed {fit_seed} at L = {L}, "
                "not one"
            )
        fit = [probes.frames(fit_runs[0], limit, probe_seed)]
        test = [
            probes.frames(r, limit, probe_seed)
            for r in scored
            if r.L == L and r.kind == "staircase"
        ]
        targets = probes.controls(fit, test, probe_seed)
        counts = {
            "n_fit": sum(len(f.replica) for f in fit),
            "n_test": sum(len(f.replica) for f in test),
        }
        for checkpoint in passes:
            x = embed(checkpoint, [*fit, *test])
            pred = probes.fitted(x[: counts["n_fit"]], x[counts["n_fit"] :], targets, alpha)
            head = {"checkpoint": checkpoint.name, **arms[checkpoint.name], "L": L}
            rows += [head | row | counts for row in probes.scored(pred, targets)
                     if row["target"] in wanted]  # fmt: skip
            for t, target in enumerate(wanted):
                raw = probes.replica_sums(test, targets[target, "raw"][1], pred[target, "raw"])
                shuffled = probes.replica_sums(
                    test, targets[target, "shuffle_within_T"][1], pred[target, "shuffle_within_T"]
                )
                both = [np.hstack(pair) for pair in zip(raw, shuffled, strict=True)]
                draws[target, "r2", L, checkpoint.name] = scoring.hierarchical_bootstrap(
                    probes.r2_of, [raw], n, np.random.default_rng([seed, L, t, 0])
                )
                draws[target, "selectivity", L, checkpoint.name] = scoring.hierarchical_bootstrap(
                    lambda a: probes.r2_of(a[:, :4]) - probes.r2_of(a[:, 4:]),
                    [both],
                    n,
                    np.random.default_rng([seed, L, t, 1]),
                )
    return rows, draws


def probe_quantities(
    doc: dict[str, Any], rows: Sequence[Row], draws: dict[tuple[str, str, int, str], np.ndarray]
) -> list[margins.Quantity]:
    """Each target's R^2 and selectivity, per L and as the mean over L that scoring.margins.sizes
    says is what a checkpoint's value is, draw by draw."""
    sizes = sorted(doc["data"]["L"])
    out = []
    for target in doc["probes"]["targets"]:
        for stat in ("r2", "selectivity"):
            values = {
                L: {
                    r["checkpoint"]: r[stat]
                    for r in rows
                    if (r["target"], r["column"], r["L"]) == (target, "raw", L)
                }
                for L in sizes
            }
            drawn = {L: {name: draws[target, stat, L, name] for name in values[L]} for L in sizes}
            name = f"{target}_{stat}"
            out += [margins.Quantity(name, f"L={L}", values[L], drawn[L]) for L in sizes]
            mean = {c: float(np.mean([values[L][c] for L in sizes])) for c in values[sizes[0]]}
            joined = {c: np.mean([drawn[L][c] for L in sizes], axis=0) for c in mean}
            out.append(margins.Quantity(name, margins.ALL, mean, joined))
    return out


def auc_quantities(
    doc: dict[str, Any], auc: Sequence[Row], baseline: Sequence[Row]
) -> list[margins.Quantity]:
    """Each judged pair's headline AUC, the mean over L and h per checkpoint, beside the classic |m|
    rows of the same indicator: parity, never judged, since AUC saturates for trained and twin."""
    out = []
    for observable, name in doc["indicators"]["pairs"]:
        per: dict[str, list[float]] = {}
        for r in auc:
            if r["headline"] and (r["observable"], r["indicator"]) == (observable, name):
                per.setdefault(r["checkpoint"], []).append(r["auc"])
        classic = [r["estimate"] for r in baseline if r["quantity"] == f"auc_{name}"]
        reference = float(np.mean(classic)) if classic else float("nan")
        values = {c: float(np.mean(v)) for c, v in per.items()}
        out.append(
            margins.Quantity(f"auc_{observable}_{name}", margins.ALL, values, reference=reference)
        )
    return out


def alignment_quantities(doc: dict[str, Any], curves: Sequence[Row]) -> list[margins.Quantity]:
    """The chi alignment, descriptive since locality predicts the twin aligns too: the Pearson
    correlation over the plateaus between L^2 Var_T(pc_1) and the classic L^2 Var_T(|m|), at each L
    and its mean over L. Undefined (NaN) under three plateaus, as in an ablation's tc-only table."""
    sizes = sorted(doc["data"]["L"])
    names = sorted({r["checkpoint"] for r in curves if r["checkpoint"]})

    def curve(name: str, column: str, L: int) -> dict[float, float]:
        return {
            r["temperature"]: r["l2_var"]
            for r in curves
            if (r["checkpoint"], r["observable"], r["L"]) == (name, column, L)
        }

    def pearson(name: str, L: int) -> float:
        latent, classic = curve(name, "pc_1", L), curve("", "abs_m", L)
        shared = sorted(latent.keys() & classic.keys())
        if len(shared) < 3:
            return float("nan")
        a, b = np.array([latent[T] for T in shared]), np.array([classic[T] for T in shared])
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")

    per_L = {L: {name: pearson(name, L) for name in names} for L in sizes}
    mean = {name: float(np.mean([per_L[L][name] for L in sizes])) for name in names}
    return [
        *(margins.Quantity("chi_alignment", f"L={L}", per_L[L]) for L in sizes),
        margins.Quantity("chi_alignment", margins.ALL, mean),
    ]


def run(
    path: Path,
    data_dir: Path,
    seed: int = 26,
    planted: bool = False,
    allow_stale: bool = False,
    embed: Embed | None = None,
    device: str | None = None,
) -> tuple[Path, list[Row], int]:
    """Everything but the figures; returns the result folder, its judged rows, the trials logged."""
    pre = prereg.load(path)
    if "latents" not in pre.doc:
        raise prereg.PreregError(f"{pre.name} has no [latents]: `ews score` scores it")
    doc = pre.doc
    found = prereg.problems(doc, strict=True)
    if found:
        raise prereg.PreregError(f"{pre.name} cannot be scored:\n  " + "\n  ".join(found))
    if planted:
        ref = prereg.Frozen("", "")
        real = [r.run_id for r in ising.find_runs(data_dir)[0]
                if r.manifest["producer"]["name"] != "planted"]  # fmt: skip
        if real:
            raise prereg.PreregError(f"--planted reads planted data alone, but found: {real}")
    else:
        ref = prereg.frozen(pre)
    planned = prereg.cells(doc)
    prereg.charge(data_dir, pre, planned)
    runs_commit = prereg.baseline_commit(pre) if ref.commit else None
    admitted = ising.admit(doc, data_dir, allow_stale=allow_stale, commit=ref.commit or None,
                           runs_commit=runs_commit)  # fmt: skip
    baseline = baseline_rows(doc, data_dir, admitted.scored, runs_commit)
    if embed is None:
        embed = (
            ews_planted.planted_embedding()
            if planted
            else checkpoint_embedder(data_dir, admitted.passes, device)
        )
    curves = curve_rows(doc, admitted.scored, admitted.passes)
    probe_rows, draws = probe_tables(doc, data_dir, admitted.scored, admitted.passes, embed, seed)

    def more(evaluation: ising.Evaluation) -> dict[str, list[Row]]:
        quantities = [
            *probe_quantities(doc, probe_rows, draws),
            *auc_quantities(doc, evaluation.tables["auc"], baseline),
            *alignment_quantities(doc, curves),
        ]
        table = margins.table(
            doc, quantities, evaluation.tables["fss"], np.random.default_rng([seed, 28])
        )
        return {"probes": probe_rows, "curves": curves, "baseline": baseline, "margins": table}

    run_dir, evaluation, logged = score(pre, ref, data_dir, seed, allow_stale, more)
    csv_tables(run_dir)
    table = _table(run_dir, "margins")  # none when no role has two model seeds
    judged = [r for r in pq.read_table(table).to_pylist() if r["judged"]] if table.is_file() else []
    failed = pipeline_faults(evaluation.tables["fss"])
    if failed:
        print(
            f"pipeline check not passed, read nothing else until it is: {failed}", file=sys.stderr
        )
    return run_dir, judged, logged + planned


def pipeline_faults(rows: Sequence[Row]) -> list[str]:
    """Which pipeline checks did not pass, static (role pipeline-check) or dynamic
    (dynamic-pipeline-check): hypothesis 1 says no other latent row is read until they do. A check
    judged on the twins alone is reported on a trained checkpoint, never passed or failed there."""
    return [
        f"{r['checkpoint']}: {r['quantity']} {r['verdict']}"
        for r in rows
        if r["role"].endswith("pipeline-check") and r["verdict"] not in ("pass", ising.REPORTED)
    ]


def _table(run_dir: Path, name: str) -> Path:
    return run_dir / name / f"{0:020d}.parquet"


def _text(value: Any) -> str:
    """One CSV field, the same bytes every time: a float's repr, a list as JSON, a null empty."""
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value)
    return repr(value) if isinstance(value, float) else str(value)


def csv_tables(run_dir: Path) -> None:
    """csv/<table>.csv for every table of a result, from its Parquet: what the figures read. It
    writes over whatever an earlier attempt left there."""
    (run_dir / "csv").mkdir(exist_ok=True)
    for name in io.read_json(run_dir / "manifest.json")["tables"]:
        table = pq.read_table(_table(run_dir, name))
        out = StringIO(newline="")
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(table.column_names)
        writer.writerows([_text(v) for v in row.values()] for row in table.to_pylist())
        io.write_file(
            run_dir / "csv" / f"{name}.csv",
            lambda f, o=out: f.write(o.getvalue().encode()),
            replace=True,
        )


def figure_command(run_dir: Path) -> list[str]:
    return [*FIGURE_ENVIRONMENT, "python", str(FIGURES), str(run_dir)]


def draw_figures(run_dir: Path) -> str | None:
    """The figures in a throwaway environment; None, or why they were not drawn. Nothing that goes
    wrong here touches the tables, which are already written."""
    try:
        done = subprocess.run(figure_command(run_dir), capture_output=True, text=True, check=False)
    except OSError as exc:
        return repr(exc)
    return None if done.returncode == 0 else (done.stderr or done.stdout)[-2000:]


def main(argv: Sequence[str] | None = None, embed: Embed | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--prereg", type=Path)
    entry.add_argument(
        "--figures-only",
        type=Path,
        metavar="RUN_DIR",
        help="a complete result's csv/, written again, then its figures: charges and logs nothing",
    )
    parser.add_argument("--data-dir", type=Path, default=engine.data_dir())
    parser.add_argument("--seed", type=int, default=26, help="every bootstrap and surrogate draw")
    parser.add_argument("--device", help="for the probes' embeddings; default: cuda if available")
    parser.add_argument("--allow-stale-pilot", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--planted", action="store_true", help="unfrozen, on planted data alone")
    args = parser.parse_args(argv)
    try:
        if args.figures_only:
            run_dir = args.figures_only
            if not (run_dir / "complete.json").is_file():  # its cells may be unlogged
                raise ising.IsingDataError(
                    f"{run_dir}: no complete.json (running or crashed); refused"
                )
            csv_tables(run_dir)
            print(f"{run_dir}  (csv/ written, no trial charged)")
        else:
            run_dir, judged, logged = run(
                args.prereg, args.data_dir, args.seed, args.planted, args.allow_stale_pilot, embed,
                args.device,
            )  # fmt: skip
            for row in judged:
                print(
                    f"margin {row['quantity']:<20} {row['margin']:+.3f} "
                    f"[{row['lo']:+.3f}, {row['hi']:+.3f}]  {row['verdict']}"
                )
            print(f"{run_dir}  ({logged} trials logged)")
    except (prereg.PreregError, ising.IsingDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.no_figures:
        return 0
    why = draw_figures(run_dir)
    if why is not None:
        command = " ".join(figure_command(run_dir))
        print(
            f"figures not drawn; the tables are safe. To draw them: {command}\n{why}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
