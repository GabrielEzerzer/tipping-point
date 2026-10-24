"""Writes the tiny synthetic archives the engine's import tests read, in each venue's own format,
to `engine/tests/fixtures/archives/`. Real archive days are never committed (republishing them is
a licensing risk); they stay local, gitignored, in `harness/fixtures/archives/`.

- `2026-10-20_BTCUSDT_ob200.data.zip`: Bybit's order-book archive. One member of CRLF-ended
  lines, each a message of the stream (`topic`, `type`, `ts`, `data` with `s`, `b`, `a`, `u`,
  `seq`, then `cts`), written with a data descriptor as Bybit's own zips are. A snapshot, deltas
  whose `u` runs on by one every 100 ms, one `u` skipped (a gap), deltas after it, Bybit's
  snapshot sent again, more deltas. Prices are written with two decimals and quantities with
  three, as Bybit writes BTCUSDT's; a quantity of "0" deletes.
- `BTCUSDT-aggTrades-2026-10-20.zip`: Binance's aggregated trades, with its header line, one
  aggregate id missing (and the trade ids it covered), and the `.CHECKSUM` Binance publishes
  beside each file.

Run from the repo root: `uv run python harness/engine-harness/tests/make_archive_fixtures.py`.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "engine" / "tests" / "fixtures" / "archives"
DAY_MS = 1_792_454_400_000  # 2026-10-20T00:00:00Z
STAMP = (2026, 10, 21, 3, 0, 0)  # the zip entries' time, fixed so the bytes repeat


class Unseekable(io.RawIOBase):
    """A sink zipfile cannot seek, so it writes sizes after the data, as Bybit's zips have them."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:  # type: ignore[override]
        self.buffer.extend(data)
        return len(data)


def zipped(member: str, data: bytes, *, descriptor: bool) -> bytes:
    info = zipfile.ZipInfo(member, date_time=STAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    if descriptor:
        sink = Unseekable()
        with zipfile.ZipFile(sink, "w") as archive:
            archive.writestr(info, data, compresslevel=6)
        return bytes(sink.buffer)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(info, data, compresslevel=6)
    return out.getvalue()


def price(ticks: int) -> str:
    """Ticks of 0.1 written with two decimals, as Bybit writes BTCUSDT's prices."""
    return f"{ticks // 10}.{ticks % 10}0"


def qty(lots: int) -> str:
    return "0" if lots == 0 else f"{lots // 1000}.{lots % 1000:03d}"


def bybit() -> None:
    rng = random.Random(23)
    bids = {819_007 - i: rng.randint(1, 4000) for i in range(5)}
    asks = {819_008 + i: rng.randint(1, 4000) for i in range(5)}
    lines: list[str] = []
    u = 5_788_995
    seq = 828_958_074_456

    def message(kind: str, ts: int, b: dict[int, int], a: dict[int, int]) -> str:
        data = {
            "s": "BTCUSDT",
            "b": [[price(p), qty(q)] for p, q in sorted(b.items(), reverse=True)],
            "a": [[price(p), qty(q)] for p, q in sorted(a.items())],
            "u": u,
            "seq": seq,
        }
        body = {"topic": "orderbook.200.BTCUSDT", "type": kind, "ts": ts, "data": data}
        body["cts"] = ts - 2
        return json.dumps(body, separators=(",", ":"))

    ts = DAY_MS + 495
    lines.append(message("snapshot", ts, bids, asks))
    for step in range(1, 35):
        ts += 100 + rng.randint(-3, 3)
        u += 2 if step == 14 else 1  # one message lost: u skips at step 14
        seq += rng.randint(100, 900)
        if step == 22:  # Bybit sends its snapshot again: the whole book as it is now
            lines.append(message("snapshot", ts, bids, asks))
            continue
        b: dict[int, int] = {}
        a: dict[int, int] = {}
        for _ in range(rng.randint(1, 3)):
            side, book = (b, bids) if rng.random() < 0.5 else (a, asks)
            level = max(bids) - rng.randint(0, 6) if book is bids else min(asks) + rng.randint(0, 6)
            lots = 0 if (level in book and rng.random() < 0.3) else rng.randint(1, 5000)
            side[level] = lots
            if lots:
                book[level] = lots
            else:
                book.pop(level, None)
        lines.append(message("delta", ts, b, a))
    data = ("\r\n".join(lines) + "\r\n").encode()
    name = "2026-10-20_BTCUSDT_ob200.data"
    (OUT / f"{name}.zip").write_bytes(zipped(name, data, descriptor=True))


def binance() -> None:
    rng = random.Random(51)
    rows = ["agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"]
    agg, trade, ts, ticks = 3_490_482_571, 8_187_372_393, DAY_MS + 7, 819_033
    for i in range(24):
        if i == 11:  # one aggregated trade missing, and the trades it covered
            agg += 1
            trade += 2
        covered = rng.randint(1, 3)
        ticks += rng.randint(-2, 2)
        lots = rng.randint(1, 900)
        maker = "true" if rng.random() < 0.5 else "false"
        low, high = trade, trade + covered - 1
        rows.append(f"{agg},{ticks // 10}.{ticks % 10},{qty(lots)},{low},{high},{ts},{maker}")
        agg += 1
        trade += covered
        ts += rng.randint(1, 400)
    data = ("\n".join(rows) + "\n").encode()
    name = "BTCUSDT-aggTrades-2026-10-20"
    archive = zipped(f"{name}.csv", data, descriptor=False)
    (OUT / f"{name}.zip").write_bytes(archive)
    digest = hashlib.sha256(archive).hexdigest()
    checksum = OUT / f"{name}.zip.CHECKSUM"
    checksum.write_text(f"{digest}  {name}.zip\n", encoding="utf-8", newline="")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bybit()
    binance()
    for path in sorted(OUT.iterdir()):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
