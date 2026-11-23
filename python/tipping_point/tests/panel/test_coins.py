# The coin rule on a fake Binance Vision bucket: names, months and metrics days filter, volume
# ranks, listings page, checksums are checked, no request names a held-back day. No market data.

import hashlib
import urllib.parse
from datetime import date
from pathlib import Path

import pytest
from market_planted import checksum, zipped
from tipping_point.market import archive
from tipping_point.panel import coins, span
from tipping_point.panel.__main__ import main

PAGE = 3


class Bucket:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requests: list[str] = []

    def put(self, key: str, data: bytes) -> None:
        self.files[key] = data
        self.files[key + ".CHECKSUM"] = checksum(key.rsplit("/", 1)[-1], data)

    def get(self, url: str) -> bytes | None:
        self.requests.append(url)
        if url.startswith(coins.BUCKET + "?"):
            return self.list(urllib.parse.parse_qs(url.split("?", 1)[1]))
        return self.files.get(url.removeprefix("https://data.binance.vision/"))

    def list(self, query: dict[str, list[str]]) -> bytes:
        prefix, marker = query["prefix"][0], query.get("marker", [""])[0]
        entries: set[tuple[str, str]] = set()
        for key in self.files:
            if key.startswith(prefix):
                rest = key[len(prefix) :]
                if "/" in rest:
                    entries.add(("CommonPrefixes", prefix + rest.split("/")[0] + "/"))
                else:
                    entries.add(("Contents", key))
        ordered = sorted((e for e in entries if e[1] > marker), key=lambda e: e[1])
        page, more = ordered[:PAGE], len(ordered) > PAGE
        body = "".join(
            f"<CommonPrefixes><Prefix>{v}</Prefix></CommonPrefixes>"
            if kind == "CommonPrefixes"
            else f"<Contents><Key>{v}</Key></Contents>"
            for kind, v in page
        )
        tail = f"<NextMarker>{page[-1][1]}</NextMarker>" if more else ""
        return (
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>{tail}{body}"
            "</ListBucketResult>"
        ).encode()


def kline_file(symbol: str, month: str, daily: float, header: bool = True) -> bytes:
    first = date.fromisoformat(month + "-01")
    rows = ["open_time,open,high,low,close,volume,close_time,quote_volume,count"] if header else []
    for d in range(28):
        start = archive.day_ms(first) + d * 86_400_000
        rows.append(f"{start},1,1,1,1,10,{start + 86_399_999},{daily},5,5,{daily / 2},0")
    return zipped(f"{symbol}-1d-{month}.csv", ("\n".join(rows) + "\n").encode())


def publish(
    bucket: Bucket,
    symbol: str,
    daily: float,
    months: tuple[str, ...] = coins.MONTHS,
    metric_days: tuple[date, ...] = coins.METRIC_DAYS,
) -> None:
    for i, month in enumerate(months):
        key = f"{coins.KLINES}{symbol}/1d/{symbol}-1d-{month}.zip"
        bucket.put(key, kline_file(symbol, month, daily, header=i % 2 == 0))
    for day in metric_days:
        bucket.put(f"{coins.METRICS}{symbol}/{symbol}-metrics-{day}.zip", zipped("m.csv", b"x\n"))


@pytest.fixture
def bucket() -> Bucket:
    out = Bucket()
    publish(out, "BTCUSDT", 900.0)
    publish(out, "ETHUSDT", 50.0)
    publish(out, "AAAUSDT", 300.0)
    publish(out, "1000EEEUSDT", 80.0)
    publish(out, "BBBUSDT", 999.0, months=coins.MONTHS[:4] + coins.MONTHS[5:])
    publish(out, "CCCUSDT", 999.0, metric_days=coins.METRIC_DAYS[1:])
    publish(out, "BTCDOMUSDT", 999.0)
    publish(out, "ETHBUSD", 999.0)
    publish(out, "DDDUSDT_230331", 999.0)
    return out


def test_the_rule_keeps_named_symbols_with_every_month_and_both_metrics_days(
    bucket: Bucket,
) -> None:
    volumes = coins.derive(bucket.get, workers=4)
    assert volumes == {
        "AAAUSDT": 300.0 * 28 * 12,
        "BTCUSDT": 900.0 * 28 * 12,
        "ETHUSDT": 50.0 * 28 * 12,
        "1000EEEUSDT": 80.0 * 28 * 12,
    }
    assert coins.rank(volumes) == ["BTCUSDT", "ETHUSDT", "AAAUSDT", "1000EEEUSDT"]
    assert any("marker=" in url for url in bucket.requests), "the listing never paged"


def test_rank_puts_btc_and_eth_first_then_the_top_others_by_volume() -> None:
    volumes = {f"C{i:02d}USDT": float(i) for i in range(40)} | {"BTCUSDT": 0.0, "ETHUSDT": 1.0}
    volumes["TIEAUSDT"] = volumes["TIEBUSDT"] = 39.0
    panel = coins.rank(volumes)
    assert panel[:4] == ["BTCUSDT", "ETHUSDT", "C39USDT", "TIEAUSDT"]
    assert panel[4] == "TIEBUSDT" and len(panel) == 2 + coins.OTHERS
    assert panel[-1] == "C14USDT"
    with pytest.raises(archive.ArchiveError, match="ETHUSDT"):
        coins.rank({"BTCUSDT": 1.0})


def test_a_file_that_fails_its_checksum_is_refused(bucket: Bucket) -> None:
    key = f"{coins.KLINES}AAAUSDT/1d/AAAUSDT-1d-2023-03.zip"
    bucket.files[key + ".CHECKSUM"] = f"{hashlib.sha256(b'x').hexdigest()}  x.zip\n".encode()
    with pytest.raises(archive.ArchiveError, match=r"AAAUSDT-1d-2023-03\.zip"):
        coins.derive(bucket.get, workers=2)


def test_a_row_outside_its_month_is_refused() -> None:
    data = kline_file("AAAUSDT", "2023-02", 1.0)
    with pytest.raises(archive.ArchiveError, match="outside 2023-03"):
        coins.month_volume(data, "2023-03")


def test_no_request_names_a_held_back_day(bucket: Bucket, monkeypatch: pytest.MonkeyPatch) -> None:
    coins.derive(bucket.get, workers=4)
    for url in bucket.requests:
        span.refuse_url(url)
    patches = (
        ("YEAR", 2024),
        ("MONTHS", (*coins.MONTHS[1:], "2024-01")),
        ("METRIC_DAYS", (coins.METRIC_DAYS[0], date(2024, 1, 1))),
    )
    for name, value in patches:
        with monkeypatch.context() as patch:
            patch.setattr(coins, name, value)
            bucket.requests.clear()
            with pytest.raises(span.HeldBackError):
                coins.derive(bucket.get, workers=4)
            assert not any("2024" in url for url in bucket.requests)


def test_each_request_is_refused_on_a_held_back_day_past_the_span_check(
    bucket: Bucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(span, "refuse", lambda *days: None)
    patches = (("YEAR", 2024), ("METRIC_DAYS", (coins.METRIC_DAYS[0], date(2024, 1, 1))))
    for name, value in patches:
        with monkeypatch.context() as patch:
            patch.setattr(coins, name, value)
            bucket.requests.clear()
            with pytest.raises(span.HeldBackError):
                coins.derive(bucket.get, workers=4)
            assert bucket.requests, name
            assert not any("2024" in url for url in bucket.requests), name


def test_the_committed_panel_follows_the_rule() -> None:
    panel = coins.committed()
    assert len(panel) == len(set(panel)) == 2 + coins.OTHERS
    assert panel[:2] == list(coins.ALWAYS)
    assert all(coins.PATTERN.fullmatch(s) and s not in coins.EXCLUDED for s in panel)


def test_the_command_prints_the_ranking_and_checks_the_committed_list(
    bucket: Bucket,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(archive, "http_get", bucket.get)
    assert main(["coins", "--workers", "2"]) == 1
    assert "BTCUSDT ETHUSDT AAAUSDT 1000EEEUSDT" in capsys.readouterr().out
    held = tmp_path / "coins.txt"
    held.write_text("BTCUSDT\nETHUSDT\nAAAUSDT\n1000EEEUSDT\n", encoding="utf-8")
    monkeypatch.setattr(coins, "COMMITTED", held)
    assert main(["coins", "--workers", "2"]) == 0
