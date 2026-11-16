# harness/latent_data_dir.py on planted data: the data dir it links holds what the runner admits,
# its trial log is the source's, and a run, table or file the file needs that is missing, or not the
# bytes or rows its record names, is listed.

import json
import os
import shutil
from pathlib import Path

import ews_planted as planted
import ising_experiment as runner
import latent_data_dir
import pyarrow.parquet as pq
import pytest
from tipping_point.ews import ising, prereg
from tipping_point.ising import io
from tipping_point.jepa import train

TRAININGS = planted.experiment_trainings(ablations=0)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("latent-data-dir")
    data = planted.v1_dataset(root / "data", "fss")
    planted.add_fit_runs(data)
    planted.write_experiment_tables(data, TRAININGS)
    result = planted.write_baseline(data)
    path = root / "planted.toml"
    doc = planted.latent_doc(data, TRAININGS, result)
    path.write_text(planted.latent_toml(doc), encoding="utf-8", newline="\n")
    return path, data.root


def _files(folder: Path) -> list[Path]:
    return sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())


def _copied(root: Path, tmp_path: Path) -> Path:
    shutil.copytree(root / "ising", tmp_path / "copy" / "ising")
    return tmp_path / "copy"


def _scored_table(copy: Path) -> Path:
    return next(
        table
        for table in sorted((copy / "ising").glob(f"*/{ising.LATENTS}/*"))
        if io.read_json(table.parents[1] / "manifest.json")["params"]["seed"] != planted.FIT_SEED
    )


def _where(table: Path) -> str:
    return f"{table.parents[1].name}/{ising.LATENTS}/{table.name}"


def _problems(path: Path, source: Path, into: Path, ews_from: Path) -> list[str]:
    latent_data_dir.assemble(path, [source], into, ews_from)
    return latent_data_dir.problems(path, into)


def test_every_file_is_linked_and_the_runner_admits_the_data_dir(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    path, root = source
    into = tmp_path / "into"
    counts = latent_data_dir.assemble(path, [root], into, root)
    assert latent_data_dir.problems(path, into) == []
    assert counts["tables"] == len(list((into / "ising").glob(f"*/{ising.LATENTS}/*"))) > 0
    linked = _files(into / "ising")
    assert linked == _files(root / "ising")
    assert all(os.path.samefile(into / "ising" / p, root / "ising" / p) for p in linked)
    found = runner.ready(path, into, planted=True)
    assert found.logged_before == 0 and len(found.fits) == len(planted.V1_SIZES)


def test_the_trial_log_written_through_the_data_dir_is_the_sources(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    path, root = source
    home = tmp_path / "home"
    shutil.copytree(root / "ews", home / "ews")
    into = tmp_path / "into"
    latent_data_dir.assemble(path, [root], into, home)
    pre = prereg.load(path)
    before, theirs = prereg.trials(home, pre), prereg.trials(root, pre)
    prereg.log_trials(into, [{"prereg": {"id": pre.id, "version": pre.version}}])
    assert prereg.trials(home, pre) == before + 1 and prereg.trials(root, pre) == theirs


def test_runs_of_other_seeds_are_left_out_and_the_first_folder_gives_a_runs_files(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    path, root = source
    other = tmp_path / "other" / "ising" / "20261101T000000Z-L16-qs-s99"
    other.mkdir(parents=True)
    io.write_json(other / "manifest.json", {"params": {"seed": 99, "L": 16}})
    into = tmp_path / "into"
    latent_data_dir.assemble(path, [root, tmp_path / "other"], into, root)
    assert not (into / "ising" / other.name).exists()
    assert {p.name for p in (into / "ising").iterdir()} == {
        p.name for p in (root / "ising").iterdir()
    }


MISSING = {
    "a pinned table": "inventory: scored runs without a complete latents table",
    "a run's complete.json": "inventory: runs the prereg would score have no complete.json",
    "a fit staircase": f"inventory: probes.fit: 0 complete staircases of seed {planted.FIT_SEED}",
    "every run": "inventory: ",
}


@pytest.mark.parametrize(("gone", "said"), MISSING.items(), ids=MISSING.keys())
def test_what_the_file_needs_and_the_data_dir_lacks_is_listed(
    source: tuple[Path, Path], tmp_path: Path, gone: str, said: str
) -> None:
    path, root = source
    copy = _copied(root, tmp_path)
    table = _scored_table(copy)
    if gone == "a pinned table":
        shutil.rmtree(table)
    elif gone == "a run's complete.json":
        (table.parents[1] / "complete.json").unlink()
    elif gone == "a fit staircase":
        fit = next(
            run
            for run in (copy / "ising").glob(f"*-s{planted.FIT_SEED}")
            if io.read_json(run / "manifest.json")["params"]["L"] == planted.V1_SIZES[0]
        )
        shutil.rmtree(fit)
    else:
        shutil.rmtree(copy / "ising")
    found = _problems(path, copy, tmp_path / "into", root)
    assert any(line.startswith(said) for line in found), found


def test_a_chunk_that_is_not_its_recorded_bytes_is_listed(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    path, root = source
    copy = _copied(root, tmp_path)
    run = next(p for p in sorted((copy / "ising").glob("*")) if (p / "configs").is_dir())
    chunk = io.chunks(run, "configs")[0]
    chunk.write_bytes(chunk.read_bytes() + b"\0")
    found = _problems(path, copy, tmp_path / "into", root)
    assert f"{run.name}: sha256 differs: configs/{chunk.name}" in found


def test_a_table_file_that_is_not_its_recorded_bytes_is_listed(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    path, root = source
    copy = _copied(root, tmp_path)
    table = _scored_table(copy)
    chunk = sorted(table.glob("*.parquet"))[0]
    chunk.write_bytes(chunk.read_bytes() + b"\0")
    found = _problems(path, copy, tmp_path / "into", root)
    assert f"{_where(table)}: sha256 differs: {chunk.name}" in found


def test_a_table_short_of_a_row_is_listed_though_its_record_names_its_bytes(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    path, root = source
    copy = _copied(root, tmp_path)
    table = _scored_table(copy)
    chunk = sorted(table.glob("*.parquet"))[0]
    rows = pq.read_table(chunk)
    pq.write_table(rows.slice(0, rows.num_rows - 1), chunk)
    done = io.read_json(table / "complete.json")
    done["files"][chunk.name] = train.sha256(chunk.read_bytes())
    io.write_json(table / "complete.json", done, replace=True)
    found = _problems(path, copy, tmp_path / "into", root)
    where = _where(table)
    assert not [line for line in found if "sha256 differs" in line], found
    short = f"{where}: {chunk.name}: {rows.num_rows - 1} rows, not exactly the {rows.num_rows}"
    assert any(line.startswith(short) for line in found), found
    assert any(line.startswith(f"{where}: complete.json counts") for line in found), found


def test_a_table_with_no_manifest_is_listed_not_raised(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    path, root = source
    copy = _copied(root, tmp_path)
    table = _scored_table(copy)
    (table / "manifest.json").unlink()
    found = _problems(path, copy, tmp_path / "into", root)
    assert f"{_where(table)}: missing manifest.json" in found


def test_a_table_read_over_another_run_is_listed(source: tuple[Path, Path], tmp_path: Path) -> None:
    path, root = source
    copy = _copied(root, tmp_path)
    table = _scored_table(copy)
    manifest = json.loads((table / "manifest.json").read_text("utf-8"))
    manifest["params"]["dataset"]["complete_sha256"] = "0" * 64
    (table / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    found = _problems(path, copy, tmp_path / "into", root)
    assert f"{_where(table)}: read over another complete.json than this run's" in found


def test_an_existing_folder_is_refused(source: tuple[Path, Path], tmp_path: Path) -> None:
    path, root = source
    with pytest.raises(FileExistsError):
        latent_data_dir.assemble(path, [root], tmp_path, root)
