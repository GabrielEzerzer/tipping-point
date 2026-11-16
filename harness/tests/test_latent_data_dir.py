# harness/latent_data_dir.py on planted data: the data dir it links holds what the runner admits,
# its trial log is the source's, and a file or table not the bytes its record names is listed.

import json
import os
import shutil
from pathlib import Path

import ews_planted as planted
import ising_experiment as runner
import latent_data_dir
import pytest
from tipping_point.ews import ising, prereg
from tipping_point.ising import io

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


def test_every_file_is_linked_and_the_runner_admits_the_data_dir(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    path, root = source
    into = tmp_path / "into"
    counts = latent_data_dir.assemble(path, [root], into, root)
    assert latent_data_dir.problems(into) == []
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
    into = tmp_path / "into"
    latent_data_dir.assemble(path, [root], into, root)
    pre = prereg.load(path)
    before = prereg.trials(root, pre)
    prereg.log_trials(into, [{"prereg": {"id": pre.id, "version": pre.version}}])
    assert prereg.trials(root, pre) == before + 1


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


def test_a_chunk_that_is_not_its_recorded_bytes_is_listed(
    source: tuple[Path, Path], tmp_path: Path
) -> None:
    path, root = source
    copy = tmp_path / "copy"
    shutil.copytree(root / "ising", copy / "ising")
    run = next(p for p in sorted((copy / "ising").glob("*")) if (p / "configs").is_dir())
    chunk = io.chunks(run, "configs")[0]
    chunk.write_bytes(chunk.read_bytes() + b"\0")
    into = tmp_path / "into"
    latent_data_dir.assemble(path, [copy], into, root)
    assert f"{run.name}: sha256 differs: configs/{chunk.name}" in latent_data_dir.problems(into)


def test_a_table_read_over_another_run_is_listed(source: tuple[Path, Path], tmp_path: Path) -> None:
    path, root = source
    copy = tmp_path / "copy"
    shutil.copytree(root / "ising", copy / "ising")
    table = next(p for p in sorted((copy / "ising").glob(f"*/{ising.LATENTS}/*")))
    manifest = json.loads((table / "manifest.json").read_text("utf-8"))
    manifest["params"]["dataset"]["complete_sha256"] = "0" * 64
    (table / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    into = tmp_path / "into"
    latent_data_dir.assemble(path, [copy], into, root)
    where = f"{table.parents[1].name}/{ising.LATENTS}/{table.name}"
    assert f"{where}: read over another complete.json than this run's" in (
        latent_data_dir.problems(into)
    )


def test_an_existing_folder_is_refused(source: tuple[Path, Path], tmp_path: Path) -> None:
    path, root = source
    with pytest.raises(FileExistsError):
        latent_data_dir.assemble(path, [root], tmp_path, root)
