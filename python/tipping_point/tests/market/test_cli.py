"""The two commands end to end on a fake archive (#29 S1-S2): fetch, then labels into a run in
#22's layout. Two runs give tables with identical sha256 (determinism), the manifest's sources
are only Binance klines, metrics and funding files (the static circularity check), and the
confirmatory span and unverified files are refused."""

import dataclasses
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from market_planted import vision
from tipping_point import engine
from tipping_point.market import archive, msvol
from tipping_point.market.__main__ import main

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
SPAN = ["--start", "2023-01-02", "--end", "2023-01-29"]  # four weeks from a Monday
LABELLER = ["--warmup-days", "14", "--baseline-days", "7", "--d-min", "6"]  # refits 01-16, 01-23
SOURCE = re.compile(
    re.escape(archive.BASE_URL)
    + r"/(daily/klines/(?P<k>[A-Z]+)/1m/(?P=k)-1m-\d{4}-\d\d-\d\d"
    + r"|daily/metrics/(?P<m>[A-Z]+)/(?P=m)-metrics-\d{4}-\d\d-\d\d"
    + r"|monthly/fundingRate/(?P<f>[A-Z]+)/(?P=f)-fundingRate-\d{4}-\d\d)\.zip"
)


@pytest.fixture(scope="module")
def fetched(tmp_path_factory: pytest.TempPathFactory) -> Path:
    fake = vision(SYMBOLS, date(2023, 1, 2), 28, seed=29)
    dest = tmp_path_factory.mktemp("archive")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(archive, "http_get", fake.get)
        assert main(["fetch", *SPAN, "--dest", str(dest)]) == 0
    return dest


def _labels(fetched: Path, data: Path, tag: str) -> Path:
    where = ["--archive", str(fetched), "--data-dir", str(data), "--tag", tag]
    assert main(["labels", *SPAN, *where, *LABELLER]) == 0
    (run_dir,) = (data / "labels").glob(f"*-{tag}")
    return run_dir


@pytest.fixture(scope="module")
def run(fetched: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _labels(fetched, tmp_path_factory.mktemp("data"), "a")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_labels_write_a_run_in_the_shared_layout(fetched: Path, run: Path) -> None:
    assert re.fullmatch(r"\d{8}T\d{6}Z-a", run.name)
    manifest, complete = _json(run / "manifest.json"), _json(run / "complete.json")
    assert (manifest["domain"], manifest["key"]) == ("labels", "ts_ns")
    assert manifest["run_id"] == run.name and manifest["tables"] == list(msvol.SCHEMAS)
    expected = msvol.Params(warmup_days=14, baseline_days=7, d_min=6)
    assert manifest["params"]["labeller"] == dataclasses.asdict(expected)
    assert manifest["params"]["symbols"] == SYMBOLS and manifest["missing_sources"] == []
    assert (manifest["licence"], manifest["attribution"]) == ("CC BY-NC-SA 4.0", "Binance Vision")
    fetch_manifest = _json(fetched / "manifest.json")["files"]
    assert len(manifest["sources"]) == len(fetch_manifest) == 2 * (28 + 28 + 1)
    recorded = {e["url"]: (e["bytes"], e["sha256"]) for e in fetch_manifest.values()}
    assert {s["url"]: (s["bytes"], s["sha256"]) for s in manifest["sources"]} == recorded
    assert complete["clean"]
    for name, schema in msvol.SCHEMAS.items():
        (path,) = (run / name).glob("*.parquet")
        table = pq.read_table(path)
        assert table.schema.equals(schema, check_metadata=False), name
        keys = table["ts_ns"].to_pylist()
        assert int(path.stem) == (keys[0] if keys else 0)
        pairs = list(zip(keys, table["symbol"].to_pylist(), strict=True))
        assert pairs == sorted(pairs) and complete["rows"][name] == table.num_rows
        relative = f"{name}/{path.name}"
        assert complete["files"][relative] == hashlib.sha256(path.read_bytes()).hexdigest()
    data = run.parent.parent
    bars = engine.read(data, "labels", run.name, "bars")
    assert bars.num_rows == 2 * 28 * 288 and complete["rows"]["labels"] == bars.num_rows
    first_label = archive.day_ms(date(2023, 1, 16)) * 1_000_000
    labels = engine.read(data, "labels", run.name, "labels", start=first_label)
    assert labels.num_rows == 2 * 14 * 288 and labels["p_high"].null_count == 0
    refits = engine.read(data, "labels", run.name, "refits").to_pylist()
    assert [r["ts_ns"] for r in refits[::2]] == [first_label, first_label + 7 * 86_400 * 10**9]
    assert len(refits) == 2 * 2 and not any(r["flagged"] for r in refits)


def test_the_labels_read_only_binance_klines_metrics_and_funding(run: Path) -> None:
    sources = _json(run / "manifest.json")["sources"]
    assert sources and all(SOURCE.fullmatch(s["url"]) for s in sources)


def test_two_runs_write_identical_tables(fetched: Path, run: Path) -> None:
    again = _labels(fetched, run.parent.parent, "b")
    files = [_json(r / "complete.json")["files"] for r in (run, again)]
    assert files[0] == files[1] and len(files[0]) == len(msvol.SCHEMAS)


def test_the_confirmatory_span_and_unverified_files_are_refused(
    fetched: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    late = ["labels", "--start", "2023-12-30", "--end", "2024-01-02", "--archive", str(fetched)]
    assert main([*late, "--data-dir", str(tmp_path)]) == 1
    assert "market-labels-v1" in capsys.readouterr().err
    spoilt = tmp_path / "archive"
    for path in fetched.rglob("*"):
        if path.is_file():
            target = spoilt / path.relative_to(fetched)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (spoilt / "klines/BTCUSDT/BTCUSDT-1m-2023-01-03.zip").write_bytes(b"tampered")
    data = tmp_path / "data"
    assert main(["labels", *SPAN, "--archive", str(spoilt), "--data-dir", str(data)]) == 1
    assert "SHA-256" in capsys.readouterr().err and not (data / "labels").exists()
