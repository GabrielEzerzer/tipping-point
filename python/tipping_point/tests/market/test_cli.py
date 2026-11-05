"""The commands end to end on a fake archive (#29 S1-S5): fetch, then labels into a run in #22's
layout. Two runs give tables with identical sha256 (determinism), the manifest's sources are only
Binance klines, metrics and funding files (the static circularity check), and the confirmatory
span and unverified files are refused. Then cascade into a run of its own, reports into the labels
run's reports/ from that run and the cascade run, refusing what does not match, and the floor's
entry point outside the package (harness/market_floor.py) on the run's own labeller."""

import dataclasses
import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path

import market_floor
import pyarrow.parquet as pq
import pytest
from market_planted import vision
from tipping_point import engine
from tipping_point.market import archive, msvol, reports
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


# ---------------------------------------------------------------- cascade, reports and the floor

# Every threshold at 0 fires on any window whose open interest falls while price and flow agree,
# so the fake archive's random walks give both symbols candidates for the reports to count.
LOOSE = ["--trailing-days", "7", "--a", "0", "--b", "0", "--c", "0", "--v", "0"]


def _cascade(fetched: Path, data: Path, tag: str, span: list[str]) -> Path:
    where = ["--archive", str(fetched), "--data-dir", str(data), "--tag", tag]
    assert main(["cascade", *span, *where, *LOOSE]) == 0
    (run_dir,) = (data / "labels").glob(f"*-{tag}")
    return run_dir


@pytest.fixture(scope="module")
def cascades(fetched: Path, run: Path) -> Path:
    return _cascade(fetched, run.parent.parent, "c", SPAN)


@pytest.fixture(scope="module")
def reported(fetched: Path, run: Path, cascades: Path) -> Path:
    argv = ["reports", "--run", str(run), "--cascades", str(cascades), "--archive", str(fetched)]
    assert main(argv) == 0
    return run / "reports"


def test_cascade_writes_its_pooled_list_as_a_run_of_its_own(cascades: Path) -> None:
    manifest, complete = _json(cascades / "manifest.json"), _json(cascades / "complete.json")
    assert (manifest["domain"], manifest["tables"]) == ("labels", ["cascades"])
    assert manifest["params"]["symbols"] == SYMBOLS and manifest["params"]["cascade"]["a"] == 0.0
    assert manifest["params"]["cascade"]["trailing_days"] == 7
    assert all(SOURCE.fullmatch(s["url"]) for s in manifest["sources"])
    rows = engine.read(cascades.parent.parent, "labels", cascades.name, "cascades").to_pylist()
    assert {r["symbol"] for r in rows} == set(SYMBOLS)
    assert complete["checks"]["asset_instances"] == len(rows) == complete["rows"]["cascades"]


def test_reports_land_in_the_labels_runs_reports_folder(reported: Path) -> None:
    for name, schemas in (("label_count", reports.LABEL_COUNT), ("stability", reports.STABILITY)):
        folder = reported / name
        files = sorted(p.name for p in folder.iterdir())
        assert files == sorted([*(f"{table}.parquet" for table in schemas), "summary.md"])
        for table, schema in schemas.items():
            assert pq.read_table(folder / f"{table}.parquet").schema.equals(schema), table
        assert (folder / "summary.md").read_text(encoding="utf-8").startswith("# ")
    stability = (reported / "stability" / "summary.md").read_text(encoding="utf-8")
    assert "The delay simulation drew from seed = 29." in stability.split("\n\n")[2]  # --seed


def test_reports_count_the_runs_labels_and_the_cascade_runs_list(
    run: Path, cascades: Path, reported: Path
) -> None:
    data = run.parent.parent
    episodes = engine.read(data, "labels", run.name, "episodes").to_pylist()
    monthly = pq.read_table(reported / "label_count" / "monthly.parquet").to_pylist()
    pooled = engine.read(data, "labels", cascades.name, "cascades").to_pylist()
    counted = pq.read_table(reported / "label_count" / "cascades.parquet").to_pylist()
    assert [row["symbol"] for row in counted] == SYMBOLS
    for symbol in SYMBOLS:
        events = [e for e in episodes if e["symbol"] == symbol and e["kind"] == "event"]
        assert sum(m["events"] for m in monthly if m["symbol"] == symbol) == len(events)
        (row,) = [row for row in counted if row["symbol"] == symbol]
        assert row["candidates"] == sum(r["symbol"] == symbol for r in pooled) > 0
        assert row["events"] == len(events)
    inputs = pq.read_table(reported / "label_count" / "inputs.parquet").to_pylist()
    assert {r["item"]: r["status"] for r in inputs} == {
        "releases": "supplied",  # the committed calendar, always
        "cascades": "supplied",
    }
    releases = pq.read_table(reported / "label_count" / "releases.parquet").to_pylist()
    kinds = ("CPI", "FOMC", "NFP")
    assert [(r["symbol"], r["kind"]) for r in releases] == [(s, k) for s in SYMBOLS for k in kinds]
    # labelled 01-16 to 01-29: CPI came on 01-12, payrolls on 01-06, the FOMC statement on 02-01
    assert all(r["releases_in_span"] == 0 for r in releases)
    agreement = pq.read_table(reported / "stability" / "agreement.parquet").to_pylist()
    assert sorted(r["symbol"] for r in agreement if r["against"] == "oracle") == SYMBOLS
    assert any(r["against"] == "parkinson" for r in agreement)
    refits = pq.read_table(reported / "stability" / "refits.parquet")
    assert refits.num_rows == engine.read(data, "labels", run.name, "refits").num_rows


def test_reports_refuse_runs_that_do_not_match(
    fetched: Path,
    run: Path,
    cascades: Path,
    reported: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    where = ["--archive", str(fetched)]
    before = sorted(reported.iterdir())
    assert main(["reports", "--run", str(cascades), *where]) == 1
    assert "is not a labels run" in capsys.readouterr().err
    assert main(["reports", "--run", str(run), "--cascades", str(run), *where]) == 1
    assert "is not a cascade run of" in capsys.readouterr().err
    btc = _cascade(fetched, run.parent.parent, "btc", [*SPAN, "--symbols", "BTCUSDT"])
    assert main(["reports", "--run", str(run), "--cascades", str(btc), *where]) == 1
    assert "is not a cascade run of" in capsys.readouterr().err
    # a day either side of the labels span: the manifest is all the check reads (#144)
    for key, day in (("start", "2023-01-01"), ("end", "2023-01-30")):
        wider = tmp_path / key / "labels" / cascades.name
        shutil.copytree(cascades, wider)
        manifest = _json(wider / "manifest.json")
        manifest["params"][key] = day
        (wider / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        assert main(["reports", "--run", str(run), "--cascades", str(wider), *where]) == 1
        assert "is not a cascade run of" in capsys.readouterr().err
    assert not (cascades / "reports").exists() and sorted(reported.iterdir()) == before


def test_reports_take_a_cascade_run_inside_the_labels_span(
    fetched: Path, run: Path, reported: Path, tmp_path: Path
) -> None:
    """The cascade rule's first days only fill its trailing medians, so its run starts later than
    the labels run (#144): taken, counted, and the summary names the narrower span. A cascade run
    of the labels run's own span gets no such sentence."""
    copy = tmp_path / "labels" / run.name
    shutil.copytree(run, copy, ignore=shutil.ignore_patterns("reports"))
    later = _cascade(fetched, tmp_path, "later", ["--start", "2023-01-09", "--end", "2023-01-29"])
    argv = ["reports", "--run", str(copy), "--cascades", str(later), "--archive", str(fetched)]
    assert main(argv) == 0
    counted = pq.read_table(copy / "reports" / "label_count" / "cascades.parquet").to_pylist()
    pooled = engine.read(tmp_path, "labels", later.name, "cascades").to_pylist()
    assert [row["symbol"] for row in counted] == SYMBOLS
    for row in counted:
        assert row["candidates"] == sum(r["symbol"] == row["symbol"] for r in pooled) > 0
    summary = (copy / "reports" / "label_count" / "summary.md").read_text(encoding="utf-8")
    note = (
        "The cascade candidates cover 2023-01-09 to 2023-01-29 only, inside the labels span of "
        "2023-01-02 to 2023-01-29; the events columns count the whole span."
    )
    cascade_section = summary.split("## Cascade candidates")[1].split("\n\n")
    assert cascade_section[1] == note
    full = (reported / "label_count" / "summary.md").read_text(encoding="utf-8")
    assert "only, inside the labels span" not in full


def test_reports_take_a_cascade_run_of_the_same_symbols_in_another_order(
    fetched: Path, run: Path, tmp_path: Path
) -> None:
    """The symbols are compared sorted (#140): a cascade run made with ETHUSDT before BTCUSDT
    covers the labels run's span and symbols, so it is taken, and counted per symbol."""
    copy = tmp_path / "labels" / run.name
    shutil.copytree(run, copy, ignore=shutil.ignore_patterns("reports"))
    reversed_symbols = [*SPAN, "--symbols", *SYMBOLS[::-1]]
    other = _cascade(fetched, tmp_path, "reversed", reversed_symbols)
    assert _json(other / "manifest.json")["params"]["symbols"] == SYMBOLS[::-1]
    argv = ["reports", "--run", str(copy), "--cascades", str(other), "--archive", str(fetched)]
    assert main(argv) == 0
    counted = pq.read_table(copy / "reports" / "label_count" / "cascades.parquet").to_pylist()
    pooled = engine.read(tmp_path, "labels", other.name, "cascades").to_pylist()
    assert [row["symbol"] for row in counted] == SYMBOLS
    for row in counted:
        assert row["candidates"] == sum(r["symbol"] == row["symbol"] for r in pooled) > 0


def test_reports_refuse_a_second_run_before_reading_the_archive(
    run: Path, reported: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run's reports exist: refused before the fetched files are read (none here)."""
    assert main(["reports", "--run", str(run), "--archive", str(tmp_path)]) == 1
    assert f"error: {reported / 'label_count'} exists" in capsys.readouterr().err


def test_reports_refuse_a_run_the_fetched_files_no_longer_give(
    fetched: Path, run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Relabelled with other parameters than the run's, the series is not the run's: refused, and
    nothing written."""
    copy = tmp_path / "labels" / run.name
    shutil.copytree(run, copy, ignore=shutil.ignore_patterns("reports"))
    manifest = _json(copy / "manifest.json")
    manifest["params"]["labeller"]["baseline_days"] = 14
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["reports", "--run", str(copy), "--archive", str(fetched)]) == 1
    assert "no longer give" in capsys.readouterr().err
    assert not (copy / "reports").exists()


def test_reports_take_an_unchanged_run_whose_refits_failed(fetched: Path, tmp_path: Path) -> None:
    """A refit with no valid start stores NaN in its theta, log-likelihood and shares. Written as
    null, labelling the same files again gives back the run's tables, so the reports go through."""
    where = ["--archive", str(fetched), "--data-dir", str(tmp_path), "--tag", "failed"]
    assert main(["labels", *SPAN, *where, "--warmup-days", "7", "--baseline-days", "14"]) == 0
    (run,) = (tmp_path / "labels").glob("*-failed")
    reasons = engine.read(tmp_path, "labels", run.name, "refits")["reason"].to_pylist()
    assert reasons.count("no valid start: too few observed bars") == 4  # 01-09 and 01-16
    assert main(["reports", "--run", str(run), "--archive", str(fetched)]) == 0
    assert sorted(p.name for p in (run / "reports").iterdir()) == ["label_count", "stability"]


FLOOR_ARGS = ["--omega", "1.14e-11", "--alpha", "0.005", "--beta", "0.9949", "--nu", "6"]


def test_the_floor_runs_the_labels_runs_own_labeller_into_its_reports(run: Path) -> None:
    assert market_floor.main(["--run", str(run), *FLOOR_ARGS, "--days", "28", "--paths", "2"]) == 0
    folder = run / "reports" / "floor"
    files = sorted(p.name for p in folder.iterdir())
    assert files == sorted([*(f"{table}.parquet" for table in reports.FLOOR), "summary.md"])
    values = pq.read_table(folder / "garch.parquet").to_pylist()
    assert values == [
        {"parameter": "omega", "value": 1.14e-11},
        {"parameter": "alpha", "value": 0.005},
        {"parameter": "beta", "value": 0.9949},
        {"parameter": "nu", "value": 6.0},
    ]
    paths = pq.read_table(folder / "paths.parquet").to_pylist()
    # the run's 14-day warm-up refits on days 14 and 21; the default 184 days would never refit
    assert [p["refits"] for p in paths] == [2, 2]
    cells = pq.read_table(folder / "cells.parquet").to_pylist()
    pairs = [(c["scorer"], c["horizon_bars"]) for c in cells]
    assert pairs == [(s, h) for s in ("variance", "ac") for h in market_floor.HORIZONS]
    assert pq.read_table(folder / "path_cells.parquet").num_rows == 2 * len(cells)
    first = (folder / "summary.md").read_text(encoding="utf-8").split("\n\n")[1]
    assert "seed = 29, paths = 2, days = 28" in first
    assert first.count("window=60, length=240") == len(market_floor.SCORERS)


def test_the_floor_refuses_a_run_whose_floor_exists_before_reading_it(tmp_path: Path) -> None:
    """A floor folder already there: refused before the manifest is read (#140), so a run with
    no manifest at all gets the refusal, not a missing file."""
    folder = tmp_path / "run" / "reports" / "floor"
    folder.mkdir(parents=True)
    with pytest.raises(SystemExit) as refused:
        market_floor.main(["--run", str(tmp_path / "run"), *FLOOR_ARGS, "--days", "28"])
    assert refused.value.code == f"error: {folder} exists"
    assert list(folder.iterdir()) == []


def test_the_floor_refuses_a_run_that_is_not_a_labels_run(cascades: Path) -> None:
    """A cascade run has no labeller to run the floor through (#140): refused, nothing written."""
    with pytest.raises(SystemExit) as refused:
        market_floor.main(["--run", str(cascades), *FLOOR_ARGS, "--days", "28"])
    assert refused.value.code == f"error: {cascades} is not a labels run"
    assert not (cascades / "reports").exists()
