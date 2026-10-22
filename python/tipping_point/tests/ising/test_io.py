"""On-disk format: bit packing, integrity checks, schemas, manifest and the completion marker."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.ising import exact, gates, io, kernel, protocol, run


@pytest.mark.parametrize("L", [4, 6, 8, 16])
def test_pack_round_trips(L: int) -> None:
    """Catches a lost or reordered bit; L = 6 has 36 bits, so the last byte is padded."""
    spins = np.random.default_rng(L).choice(np.array([-1, 1], dtype=np.int8), size=(50, L, L))
    packed = io.pack(spins)
    assert packed.shape == (50, io.packed_width(L)) and packed.dtype == np.uint8
    assert np.array_equal(io.unpack(packed, L), spins)


def test_bit_convention_is_one_for_up_row_major_msb_first() -> None:
    """The convention the manifest promises other readers (the JEPA loader, DuckDB, arrow-rs)."""
    spins = -np.ones((3, 4, 4), dtype=np.int8)
    spins[0, 0, 0] = 1  # site 0: first byte's most significant bit
    spins[1, 1, 3] = 1  # site 7: first byte's least significant bit
    spins[2, 2, 0] = 1  # site 8: second byte's most significant bit
    assert io.pack(spins).tolist() == [[0x80, 0x00], [0x01, 0x00], [0x00, 0x80]]


def test_integrity_check_accepts_every_l4_state_and_rejects_corruption() -> None:
    """M even and E = -2 L^2 (mod 4) hold for every configuration; a flipped bit breaks them."""
    _, M, E = exact.enumerate_states(4)
    io.check_totals(M, E, 4)
    with pytest.raises(ValueError):
        io.check_totals(M + 1, E, 4)
    with pytest.raises(ValueError):
        io.check_totals(M, E + 2, 4)


def test_chunks_follow_the_documented_schema_and_key_order(
    validation: tuple[Path, list[gates.Check]],
) -> None:
    """Catches drifting column names or types, and rows out of (step, replica) order."""
    run_dir = validation[0]
    for name in io.TABLES:
        for path in io.chunks(run_dir, name):
            table = pq.read_table(path)
            assert table.schema.equals(io.schema(name, 4)), (path, table.schema)
            key = np.stack([table["step"].to_numpy(), table["replica"].to_numpy()], axis=1)
            order = np.lexsort((key[:, 1], key[:, 0]))
            assert np.array_equal(order, np.arange(len(key))), path
            assert path.stem == f"{key[0, 0]:020d}"  # named after the first step


def test_saved_configs_decode_to_their_stored_totals(
    validation: tuple[Path, list[gates.Check]],
) -> None:
    """Catches packing, row or replica misalignment between spins and their M and E columns."""
    configs = io.read_table(validation[0], "configs")
    spins = io.unpack_column(configs["spins"], 4)
    M, E = kernel.totals(torch.from_numpy(spins))
    assert np.array_equal(M.numpy(), configs["magnetisation"].to_numpy())
    assert np.array_equal(E.numpy(), configs["energy"].to_numpy())
    assert set(configs["frame"].to_numpy().tolist()) == set(range(8))


def test_manifest_carries_the_shared_convention_and_the_run(
    validation: tuple[Path, list[gates.Check]],
) -> None:
    """The fields #22's convention and the reproducibility story depend on."""
    m = io.read_json(validation[0] / "manifest.json")
    assert m["format"] == 1 and m["domain"] == "ising" and m["key"] == "step"
    assert m["run_id"] == validation[0].name and "=" not in m["run_id"]
    assert {"name", "version", "git_sha", "dirty"} <= m["producer"].keys()
    assert isinstance(m["started_ns"], int)
    assert m["tables"] == ["stats", "configs", "balance"]
    params = m["params"]
    for key in ("L", "J", "seed", "replicas", "blocks", "schedule", "algorithm", "rng", "encoding"):
        assert key in params, key
    assert params["L"] == 4 and params["schedule"]["kind"] == "fixed"
    assert {"device", "torch", "cuda", "python"} <= m["env"].keys()


def test_complete_json_only_after_a_clean_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a completion marker on a crashed run, or one whose rows and hashes lie."""
    real = kernel.sweep
    calls = 0

    def crash(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > 70:
            raise RuntimeError("injected crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(kernel, "sweep", crash)
    spec = run.Spec(L=8, replicas=2, seed=1, checkpoint_every=50)
    with pytest.raises(RuntimeError):
        run.start(spec, protocol.Fixed((2.0,), 20, 80, saves=2), tmp_path, "c", "cpu")
    monkeypatch.undo()
    (run_dir,) = (tmp_path / "ising").iterdir()
    assert not (run_dir / "complete.json").exists()
    run.resume(run_dir)
    done = io.read_json(run_dir / "complete.json")
    assert done["clean"] is True and done["last_key"] == 100
    assert not (run_dir / "checkpoint.pt").exists()
    for name, rows in done["rows"].items():
        assert io.read_table(run_dir, name).num_rows == rows
    for relative, digest in done["files"].items():
        assert hashlib.sha256((run_dir / relative).read_bytes()).hexdigest() == digest


def test_data_files_only_ever_appear(tmp_path: Path) -> None:
    """A second write to the same chunk name is refused; nothing is ever rewritten in place."""
    path = tmp_path / "x.parquet"
    io.write_file(path, lambda f: f.write(b"one"))
    with pytest.raises(FileExistsError):
        io.write_file(path, lambda f: f.write(b"two"))
    assert path.read_bytes() == b"one" and not path.with_name("x.parquet.tmp").exists()


def test_discard_after_removes_only_what_followed_the_checkpoint(tmp_path: Path) -> None:
    for name, first in (("stats", 1), ("stats", 101), ("configs", 150), ("stats", 201)):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / f"{first:020d}.parquet").write_bytes(b"")
    (tmp_path / "checkpoint.pt.tmp").write_bytes(b"")
    io.discard_after(tmp_path, 150)
    left = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert left == [
        f"configs/{150:020d}.parquet",
        f"stats/{1:020d}.parquet",
        f"stats/{101:020d}.parquet",
    ]


def test_runs_refuse_more_lattices_than_uint16_replica_ids_hold(tmp_path: Path) -> None:
    """Catches the research's Gate A size (65,536 per T x 3 T) crashing at the final flush, after
    the whole run: it is refused before anything is written. 65,536 lattices still fit."""
    too_many = run.Spec(L=4, replicas=65536, seed=1)
    with pytest.raises(ValueError, match="uint16"):
        run.start(too_many, protocol.Fixed(run.VALIDATION_TEMPERATURES, 0, 1), tmp_path, "x", "cpu")
    assert not (tmp_path / "ising").exists()
    full = run.start(too_many, protocol.Fixed((2.0,), 0, 1), tmp_path, "edge", "cpu")
    assert io.read_table(full, "stats")["replica"].to_numpy().max() == 65535


def test_spins_column_is_fixed_size_binary() -> None:
    """FIXED_LEN_BYTE_ARRAY in Parquet: what the research pinned for arrow-rs readers."""
    assert io.schema("configs", 128).field("spins").type == pa.binary(2048)
