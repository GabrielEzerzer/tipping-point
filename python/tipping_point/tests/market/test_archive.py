"""The archive fetcher and reader (#29 S1): files in each format parse, the row checks list gaps,
duplicates and unsorted rows, a checksum mismatch is refused, a re-run skips verified files, and
only verified or unpublished files reach the labeller. Synthetic files served by a fake archive
(harness/market_planted.py); no network."""

import hashlib
import json
import re
import subprocess
import urllib.error
from datetime import date, timedelta
from http.client import HTTPMessage
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from market_planted import funding_csv, kline_csv, metrics_csv, vision, zipped
from tipping_point.market import archive
from tipping_point.market.__main__ import main

DAY = date(2023, 1, 2)
REPO = Path(__file__).resolve().parents[4]


def _zip(tmp_path: Path, name: str, csv: bytes) -> Path:
    path = tmp_path / f"{name}.zip"
    path.write_bytes(zipped(f"{name}.csv", csv))
    return path


def test_klines_read_alike_with_and_without_their_header(tmp_path: Path) -> None:
    log_vol = np.full(1440, -8.0)
    with_header, _ = kline_csv(DAY, log_vol, np.random.default_rng(1), header=True)
    without, _ = kline_csv(DAY, log_vol, np.random.default_rng(1), header=False)
    a, header_a = archive.read("klines", _zip(tmp_path, "a", with_header))
    b, header_b = archive.read("klines", _zip(tmp_path, "b", without))
    assert header_a and not header_b
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    assert len(a["open_time"]) == 1440 and a["open_time"][0] == archive.day_ms(DAY)
    assert a["open"][0] == 20_000.0 and a["count"].dtype == np.int64


def test_metrics_and_funding_stamps_read_as_utc_ms(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    metrics, header = archive.read("metrics", _zip(tmp_path, "m", metrics_csv(DAY, "X", rng)))
    assert header and len(metrics["create_time"]) == 288
    assert metrics["create_time"][0] == archive.day_ms(DAY)
    assert np.all(np.diff(metrics["create_time"]) == archive.STEP_MS)
    funding, _ = archive.read("fundingRate", _zip(tmp_path, "f", funding_csv(DAY)))
    assert len(funding["calc_time"]) == 93  # January's 31 days, three settlements each
    assert funding["calc_time"][0] == archive.day_ms(date(2023, 1, 1))
    assert funding["last_funding_rate"][0] == 0.0001


def test_row_checks_list_gaps_duplicates_and_unsorted_rows(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    csv, _ = kline_csv(
        DAY, np.full(1440, -8.0), rng, drop=range(600, 660), duplicate=[100], swap=[200]
    )
    source = archive.Source("klines", "BTCUSDT", "2023-01-02")
    columns, header = archive.read("klines", _zip(tmp_path, "k", csv))
    assert archive.check(source, columns, header) == {
        "rows": 1440 - 60 + 1,
        "header": True,
        "unsorted": 1,
        "off_grid": 0,
        "duplicates": ["2023-01-02T01:40:00Z"],
        "gaps": [["2023-01-02T10:00:00Z", "2023-01-02T11:00:00Z"]],
    }
    csv = metrics_csv(DAY, "BTCUSDT", rng, drop=[0, 287], duplicate=[5], swap=[10])
    source = archive.Source("metrics", "BTCUSDT", "2023-01-02")
    columns, header = archive.read("metrics", _zip(tmp_path, "m", csv))
    found = archive.check(source, columns, header)
    assert (found["rows"], found["unsorted"], found["off_grid"]) == (287, 1, 0)
    assert found["duplicates"] == ["2023-01-02T00:25:00Z"]
    assert found["gaps"] == [
        ["2023-01-02T00:00:00Z", "2023-01-02T00:05:00Z"],
        ["2023-01-02T23:55:00Z", "2023-01-03T00:00:00Z"],
    ]
    clean = metrics_csv(DAY, "BTCUSDT", rng)
    columns, header = archive.read("metrics", _zip(tmp_path, "c", clean))
    found = archive.check(source, columns, header)
    assert (found["gaps"], found["duplicates"], found["unsorted"]) == ([], [], 0)


def test_fetch_keeps_verified_files_beside_their_checksums(tmp_path: Path) -> None:
    fake = vision(["BTCUSDT"], DAY, 2, seed=1)
    done = archive.fetch(["BTCUSDT"], DAY, DAY + timedelta(days=1), tmp_path, fake.get)
    assert sorted(done) == sorted(
        [
            "klines/BTCUSDT/BTCUSDT-1m-2023-01-02.zip",
            "klines/BTCUSDT/BTCUSDT-1m-2023-01-03.zip",
            "metrics/BTCUSDT/BTCUSDT-metrics-2023-01-02.zip",
            "metrics/BTCUSDT/BTCUSDT-metrics-2023-01-03.zip",
            "fundingRate/BTCUSDT/BTCUSDT-fundingRate-2023-01.zip",
        ]
    )
    for path, entry in done.items():
        data = (tmp_path / path).read_bytes()
        assert entry["status"] == "verified" and entry["fetched"]
        assert (
            entry["url"]
            == f"{archive.BASE_URL}/"
            + {
                "klines": f"daily/klines/BTCUSDT/1m/{Path(path).name}",
                "metrics": f"daily/metrics/BTCUSDT/{Path(path).name}",
                "fundingRate": f"monthly/fundingRate/BTCUSDT/{Path(path).name}",
            }[path.split("/")[0]]
        )
        assert entry["sha256"] == hashlib.sha256(data).hexdigest() and entry["bytes"] == len(data)
        assert (tmp_path / f"{path}.CHECKSUM").read_bytes() == fake.files[
            entry["url"] + ".CHECKSUM"
        ]
        assert entry["checks"]["unsorted"] == 0 and not entry["checks"].get("gaps")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == done and "CC BY-NC-SA 4.0" in manifest["licence"]


def test_a_rerun_skips_verified_files_and_refetches_only_what_is_gone(tmp_path: Path) -> None:
    fake = vision(["BTCUSDT"], DAY, 2, seed=1)
    end = DAY + timedelta(days=1)
    archive.fetch(["BTCUSDT"], DAY, end, tmp_path, fake.get)
    fake.requests.clear()
    again = archive.fetch(["BTCUSDT"], DAY, end, tmp_path, fake.get)
    assert fake.requests == [] and not any(e["fetched"] for e in again.values())
    gone = "metrics/BTCUSDT/BTCUSDT-metrics-2023-01-03.zip"
    (tmp_path / gone).unlink()
    third = archive.fetch(["BTCUSDT"], DAY, end, tmp_path, fake.get)
    url = again[gone]["url"]
    assert fake.requests == [url + ".CHECKSUM", url]
    assert [p for p, e in third.items() if e["fetched"]] == [gone]
    assert third[gone]["sha256"] == again[gone]["sha256"] and "previous_sha256" not in third[gone]


def test_a_checksum_mismatch_is_refused_and_never_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = vision(["BTCUSDT"], DAY, 1, seed=2)
    source = archive.Source("klines", "BTCUSDT", "2023-01-02")
    good = fake.files[source.url]
    fake.files[source.url] = good[:-1] + bytes([good[-1] ^ 1])  # one bit flipped in transit
    done = archive.fetch(["BTCUSDT"], DAY, DAY, tmp_path, fake.get)
    entry = done[source.path]
    assert entry["status"] == "refused" and entry["sha256"] != entry["expected"]
    assert entry["expected"] == hashlib.sha256(good).hexdigest()
    assert not (tmp_path / source.path).exists()
    assert not (tmp_path / f"{source.path}.CHECKSUM").exists()
    monkeypatch.setattr(archive, "http_get", fake.get)
    span = ["--start", "2023-01-02", "--end", "2023-01-02", "--symbols", "BTCUSDT"]
    assert main(["fetch", *span, "--dest", str(tmp_path)]) == 1
    fake.files[source.url] = good  # the archive serves the right bytes again: the next run heals
    assert main(["fetch", *span, "--dest", str(tmp_path)]) == 0


def test_a_file_spoilt_on_disk_is_fetched_again(tmp_path: Path) -> None:
    fake = vision(["BTCUSDT"], DAY, 1, seed=3)
    archive.fetch(["BTCUSDT"], DAY, DAY, tmp_path, fake.get)
    source = archive.Source("metrics", "BTCUSDT", "2023-01-02")
    (tmp_path / source.path).write_bytes(b"not a zip")
    with pytest.raises(archive.ArchiveError, match="SHA-256"):
        archive.verify(tmp_path / source.path)
    done = archive.fetch(["BTCUSDT"], DAY, DAY, tmp_path, fake.get)
    assert done[source.path]["fetched"] and done[source.path]["status"] == "verified"
    assert archive.verify(tmp_path / source.path) == done[source.path]["sha256"]


def test_a_changed_redownload_keeps_the_sha256_it_replaced(tmp_path: Path) -> None:
    fake = vision(["BTCUSDT"], DAY, 1, seed=4)
    first = archive.fetch(["BTCUSDT"], DAY, DAY, tmp_path, fake.get)
    source = archive.Source("fundingRate", "BTCUSDT", "2023-01")
    fake.add(source, funding_csv(DAY, rate=0.0002))  # Binance re-uploads the month
    (tmp_path / source.path).unlink()
    done = archive.fetch(["BTCUSDT"], DAY, DAY, tmp_path, fake.get)
    assert done[source.path]["previous_sha256"] == first[source.path]["sha256"]
    assert done[source.path]["sha256"] != first[source.path]["sha256"]
    later = archive.fetch(["BTCUSDT"], DAY, DAY, tmp_path, fake.get)  # and keeps saying so
    assert later[source.path]["previous_sha256"] == first[source.path]["sha256"]


def test_only_verified_or_unpublished_files_reach_the_labeller(tmp_path: Path) -> None:
    fake = vision(["BTCUSDT"], DAY, 2, seed=5)
    unpublished = archive.Source("metrics", "BTCUSDT", "2023-01-03")
    del fake.files[unpublished.url], fake.files[unpublished.url + ".CHECKSUM"]
    end = DAY + timedelta(days=1)
    done = archive.fetch(["BTCUSDT"], DAY, end, tmp_path, fake.get)
    assert done[unpublished.path]["status"] == "missing"
    span = archive.collect(["BTCUSDT"], DAY, end, tmp_path)
    assert span.missing == [unpublished.url] and len(span.sources) == 4
    assert {s["url"] for s in span.sources} == {
        e["url"] for e in done.values() if e["status"] == "verified"
    }
    assert len(span.columns[("BTCUSDT", "klines")]["open_time"]) == 2 * 1440
    assert len(span.columns[("BTCUSDT", "metrics")]["create_time"]) == 288
    with pytest.raises(archive.ArchiveError, match="not fetched"):
        archive.collect(["BTCUSDT"], DAY, end + timedelta(days=1), tmp_path)
    (tmp_path / "klines/BTCUSDT/BTCUSDT-1m-2023-01-02.zip").write_bytes(b"tampered")
    with pytest.raises(archive.ArchiveError, match="SHA-256"):
        archive.collect(["BTCUSDT"], DAY, end, tmp_path)


def test_requests_carry_a_plain_user_agent_and_404_means_unpublished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            return b"body"

    def urlopen(request: Any, timeout: float) -> Response:
        seen["headers"] = dict(request.header_items())
        if request.full_url.endswith("absent.zip"):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", HTTPMessage(), None)
        return Response()

    monkeypatch.setattr(archive.urllib.request, "urlopen", urlopen)
    assert archive.http_get(f"{archive.BASE_URL}/x.zip") == b"body"
    assert seen["headers"] == {"User-agent": archive.USER_AGENT}
    assert re.fullmatch(r"tipping-point/[0-9.]+", archive.USER_AGENT)
    assert archive.http_get(f"{archive.BASE_URL}/absent.zip") is None


def test_the_default_archive_folder_is_gitignored() -> None:
    target = REPO / "harness/fixtures/archives/binance/klines/BTCUSDT/BTCUSDT-1m-2023-01-02.zip"
    done = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(target)], check=False)
    assert done.returncode == 0
