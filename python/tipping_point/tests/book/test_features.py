"""The 1-second book features (#30 M2): known answers on a hand-built book, gaps and unsynced
samples masked and never filled, no step across a gap, the day's returns adding up, causality by
perturbation (second s reads the samples in [s - 0.1, s + 1)), determinism, and the reducer wired
into `market book --reduce`. Synthetic books only; no network (conftest). The engine's own import
runs only when TP_STORE_BIN names a built store binary."""

import hashlib
import json
import math
import os
import shutil
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tipping_point.book import features
from tipping_point.book.__main__ import main as reduce_main
from tipping_point.market import book
from tipping_point.market.__main__ import main as market_main

REPO = Path(__file__).resolve().parents[4]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "book"
RUN = FIXTURES / "20261020T000000Z-bybit-linear-btcusdt"
ZIP = REPO / "engine/tests/fixtures/archives/2026-10-20_BTCUSDT_ob200.data.zip"
DAY0 = int(datetime(2023, 6, 10, tzinfo=UTC).timestamp()) * 1_000_000_000
UNIT = 1e-5  # notional per (price unit x lot) at price_decimals 2, qty_decimals 3

Level = tuple[int, int]
Row = tuple[int, bool, list[Level], list[Level]]  # (100 ms slot, synced, bids, asks)

# The deep book of second 1: six levels a side, prices in ticks, quantities in lots.
BIDS = [(1000, 3), (999, 1), (998, 1), (997, 1), (996, 1), (995, 4)]
ASKS = [(1001, 1), (1002, 1), (1003, 1), (1004, 1), (1005, 1), (1006, 1)]
# Second 0: one level a side, (bid, bid qty, ask, ask qty) per 100 ms sample.
TOP = [
    (1000, 3, 1001, 1),
    (1000, 5, 1001, 1),  # bid queue grows by 2:      e = +5 - 3      = +2
    (1000, 5, 1002, 4),  # ask moves up:              e = +1 (old ask) = +1
    (1001, 2, 1002, 4),  # bid moves up:              e = +2 (new bid) = +2
    (1001, 2, 1002, 1),  # ask queue shrinks by 3:    e = -1 + 4      = +3
    (1000, 6, 1002, 1),  # bid moves down:            e = -2 (old bid) = -2
    (1000, 6, 1001, 2),  # ask moves down:            e = -2 (new ask) = -2
    (1000, 6, 1001, 2),  # nothing:                   e = 0
    (1000, 1, 1001, 2),  # bid queue shrinks by 5:    e = 1 - 6       = -5
    (1000, 1, 1002, 7),  # ask moves up:              e = +2 (old ask) = +2
]


def _table(rows: list[Row], day0: int = DAY0) -> pa.Table:
    """book_samples rows in the engine's schema (engine/src/store/samples.rs)."""
    cols: dict[str, list[Any]] = {"sample_ts_ns": [], "seq": [], "synced": []}
    for prefix in ("bid_px", "bid_qty", "ask_px", "ask_qty"):
        for i in range(features.LEVELS):
            cols[f"{prefix}_{i}"] = []
    for n, (slot, synced, bids, asks) in enumerate(rows):
        cols["sample_ts_ns"].append(day0 + slot * features.STEP_NS)
        cols["seq"].append(n)
        cols["synced"].append(synced)
        for name, levels in (("bid", bids), ("ask", asks)):
            for i in range(features.LEVELS):
                level = levels[i] if synced and i < len(levels) else (None, None)
                cols[f"{name}_px_{i}"].append(level[0])
                cols[f"{name}_qty_{i}"].append(level[1])
    n = len(rows)
    fields = [
        ("sample_ts_ns", pa.array(cols.pop("sample_ts_ns"), pa.int64())),
        ("venue", pa.array(["bybit-linear"] * n)),
        ("symbol", pa.array(["ETHUSDT"] * n)),
        ("seq", pa.array(cols.pop("seq"), pa.int64())),
        ("last_update_id", pa.array([None] * n, pa.int64())),
        ("synced", pa.array(cols.pop("synced"), pa.bool_())),
    ]
    fields += [(k, pa.array(v, pa.int64())) for k, v in cols.items()]
    return pa.table(dict(fields))


def _shift(levels: list[Level], ticks: int) -> list[Level]:
    return [(p + ticks, q) for p, q in levels]


def _book() -> list[Row]:
    """Second 0 moves at the touch; second 1 is the deep book, still after the step into it from
    second 0's last sample; second 2 has three samples
    of the deep book, an unsynced one, one of the deep book a tick higher, and five absent;
    second 3 is unsynced throughout; nothing after. Two samples fall outside the day."""
    rows: list[Row] = [(-1, True, BIDS, ASKS)]
    rows += [(k, True, [(b, qb)], [(a, qa)]) for k, (b, qb, a, qa) in enumerate(TOP)]
    rows += [(10 + k, True, BIDS, ASKS) for k in range(10)]
    rows += [(20 + k, True, BIDS, ASKS) for k in range(3)]
    rows += [(23, False, [], []), (24, True, _shift(BIDS, 1), _shift(ASKS, 1))]
    rows += [(30 + k, False, [], []) for k in range(10)]
    rows += [(features.DAY * 10, True, BIDS, ASKS)]
    return rows


def _reduce(rows: list[Row]) -> dict[str, np.ndarray]:
    table = features.reduce(_table(rows), DAY0, 2, 3)
    return {name: table[name].to_numpy() for name in table.column_names}


def _deep(bids: list[Level], asks: list[Level]) -> dict[str, float]:
    """The deep book's per-sample values, written out by hand for BIDS and ASKS."""
    shift = bids[0][0] - 1000
    mid = 1000.5 + shift
    bid_notional = [p * q for p, q in bids]  # 3000, 999, 998, 997, 996, 3980 when unshifted
    ask_notional = [p * q for p, q in asks]  # 1001 ... 1006
    return {
        "micro_dev": 0.25,  # (1000 * 1 + 1001 * 3) / 4 - 1000.5
        "imb_1": 0.5,  # (3 - 1) / 4
        "imb_5": 2 / 12,  # (7 - 5) / 12
        "imb_20": 5 / 17,  # (11 - 6) / 17
        "depth_5": math.log((sum(bid_notional[:5]) + sum(ask_notional[:5])) * UNIT),
        "depth_20": math.log((sum(bid_notional) + sum(ask_notional)) * UNIT),
        "depth_asym": math.log(sum(bid_notional) / sum(ask_notional)),
        # Bid cumulative 3000, 3999, 4997, 5994 >= half of 10970 at the fourth level (997);
        # ask cumulative 1001, 2003, 3006, 4010 >= half of 6021 at the fourth level (1004).
        "slope_b": (mid - (997 + shift)) / mid * 1e4,
        "slope_a": ((1004 + shift) - mid) / mid * 1e4,
    }


def test_known_answers_on_a_hand_built_book() -> None:
    f = _reduce(_book())
    assert len(f["ret"]) == features.DAY == 86_400
    assert f["second_ts_ns"][0] == DAY0 and f["second_ts_ns"][-1] == DAY0 + 86_399 * 10**9
    approx = lambda x: pytest.approx(x, rel=1e-6, abs=1e-9)  # noqa: E731  float32 output

    # Second 0, the touch moving: mids in ticks 1000.5 1000.5 1001 1001.5 1001.5 1001 1000.5
    # 1000.5 1000.5 1001; five of the nine steps move. The sample just before the day gives no
    # step into 0.0 s: the day's first sample has none.
    mids = [(b + a) / 2 for b, _, a, _ in TOP]
    steps = [math.log(mids[k] / mids[k - 1]) for k in range(1, 10)]
    assert f["ret"][0] == approx(math.log(1001 / 1000.5))
    assert f["rv100"][0] == approx(sum(s * s for s in steps))
    assert f["rv100"][0] == approx(
        3 * math.log(1001 / 1000.5) ** 2 + 2 * math.log(1001.5 / 1001) ** 2
    )
    assert f["moves"][0] == 5
    assert f["wide"][0] == approx(0.3)  # spreads of 2 ticks at 0.2, 0.5 and 0.9 s
    # OFI: e = +2 +1 +2 +3 -2 -2 0 -5 +2 = +1 over a mean top-5 quantity of 62 / 10 lots.
    assert f["ofi"][0] == approx(1 / 6.2)
    micro = [(b * qa + a * qb) / (qb + qa) - (b + a) / 2 for b, qb, a, qa in TOP]
    assert f["micro_dev"][0] == approx(sum(micro) / 10)
    assert micro[0] == 0.25 and micro[2] == approx(1 / 9)  # (1000*4 + 1002*5) / 9 - 1001
    imb = [(qb - qa) / (qb + qa) for _, qb, _, qa in TOP]
    for name in ("imb_1", "imb_5", "imb_20"):  # one level a side
        assert f[name][0] == approx(sum(imb) / 10)
    assert f["depth_5"][0] == approx(
        sum(math.log((b * qb + a * qa) * UNIT) for b, qb, a, qa in TOP) / 10
    )
    assert f["depth_asym"][0] == approx(
        sum(math.log(b * qb / (a * qa)) for b, qb, a, qa in TOP) / 10
    )
    assert f["slope_b"][0] == approx(
        sum((m - b) / m * 1e4 for m, (b, *_) in zip(mids, TOP, strict=True)) / 10
    )
    assert f["slope_a"][0] == approx(
        sum((a - m) / m * 1e4 for m, (*_, a, _) in zip(mids, TOP, strict=True)) / 10
    )
    assert (f["samples"][0], f["gap"][0]) == (10, False)

    # Second 1, the deep book standing still, every level counted. Its one moving step is the
    # boundary step from second 0's last sample (1000 x 1 | 1002 x 7, mid 1001) to its first
    # (1000 x 3 | 1001 x 1, mid 1000.5): the bid stays, e = +3 - 1; the ask falls, e = -1. So
    # e = +1 over a mean top-5 quantity of 7 + 5 lots.
    for name, value in _deep(BIDS, ASKS).items():
        assert f[name][1] == approx(value), name
    assert f["ret"][1] == approx(math.log(1000.5 / 1001))
    assert f["rv100"][1] == approx(math.log(1000.5 / 1001) ** 2)
    assert (f["moves"][1], f["wide"][1]) == (1, 0)
    assert f["ofi"][1] == approx(1 / 12)
    assert (f["samples"][1], f["gap"][1]) == (10, False)


def test_ofi_is_positive_for_buying_pressure_and_negative_for_selling() -> None:
    def ofi(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
        rows = [(k, True, [(b, qb)], [(a, qa)]) for k, (b, qb, a, qa) in enumerate([first] * 5)]
        rows += [
            (5 + k, True, [(b, qb)], [(a, qa)]) for k, (b, qb, a, qa) in enumerate([second] * 5)
        ]
        return float(_reduce(rows)["ofi"][0])

    base = (1000, 4, 1001, 4)
    assert ofi(base, (1000, 6, 1001, 4)) == pytest.approx(2 / 9)  # bid queue grows
    assert ofi(base, (1001, 1, 1002, 4)) == pytest.approx((1 + 4) / 6.5)  # bid and ask move up
    assert ofi(base, (1000, 4, 1001, 7)) == pytest.approx(-3 / 9.5)  # ask queue grows
    assert ofi(base, (999, 2, 1000, 3)) == pytest.approx((-4 - 3) / 6.5)  # both move down


def test_a_partly_missing_second_is_masked_and_never_filled() -> None:
    f = _reduce(_book())
    # Second 2: samples at 0.0, 0.1, 0.2 and 0.4 s (0.3 unsynced, 0.5 to 0.9 absent). The book at
    # 0.4 s is a tick higher; no step bridges the unsynced sample, so the mid never "moved" (the
    # boundary step from second 1 is between two equal books).
    assert (f["samples"][2], f["gap"][2]) == (4, True)
    for name in ("ret", "rv100", "moves", "ofi"):
        assert f[name][2] == 0, name
    still, up = _deep(BIDS, ASKS), _deep(_shift(BIDS, 1), _shift(ASKS, 1))
    for name in still:
        assert f[name][2] == pytest.approx((3 * still[name] + up[name]) / 4, rel=1e-6), name
    # Second 3 is unsynced throughout and seconds 4 on hold no sample: NaN and masked.
    for s in (3, 4, 86_399):
        assert (f["samples"][s], f["gap"][s]) == (0, True)
        for name in features.FEATURES:
            assert np.isnan(f[name][s]), (s, name)
    # The samples just before and just after the day count nowhere.
    assert f["samples"].sum() == 10 + 10 + 4


def test_no_step_crosses_a_missing_second_or_an_unsynced_sample() -> None:
    def second(s: int, ticks: int) -> list[Row]:
        return [(10 * s + k, True, _shift(BIDS, ticks), _shift(ASKS, ticks)) for k in range(10)]

    rows = second(0, 0) + second(2, 5)  # second 1 missing
    rows += [(30, False, [], []), *second(3, 7)[1:]]  # second 3 starts unsynced
    rows += second(4, 8)  # a good boundary step, one tick up
    f = _reduce(rows)
    for s in (2, 3):
        assert (f["ret"][s], f["rv100"][s], f["moves"][s], f["ofi"][s]) == (0, 0, 0, 0), s
    assert np.isnan(f["ret"][1]) and f["gap"][1]
    assert f["ret"][4] == pytest.approx(math.log(1008.5 / 1007.5), rel=1e-6)
    assert f["moves"][4] == 1 and not f["gap"][4]
    # The boundary step: bid and ask both move up, e = +3 (new bid) + 1 (old ask), over 12 lots.
    assert f["ofi"][4] == pytest.approx(4 / 12, rel=1e-6)


def test_a_gap_free_days_returns_add_up_to_its_log_change() -> None:
    n = features.DAY * features.PER
    rng = np.random.default_rng(32)
    bid = 200_000 + np.cumsum(rng.integers(-1, 2, n))
    columns: dict[str, Any] = {
        "sample_ts_ns": DAY0 + np.arange(n, dtype=np.int64) * features.STEP_NS,
        "synced": np.ones(n, dtype=bool),
    }
    for name, values in (("bid", bid), ("ask", bid + 1 + (rng.random(n) < 0.1))):
        columns[f"{name}_px_0"] = values
        columns[f"{name}_qty_0"] = rng.integers(1, 1000, n)
        for i in range(1, features.LEVELS):
            columns[f"{name}_px_{i}"] = pa.nulls(n, pa.int64())
            columns[f"{name}_qty_{i}"] = pa.nulls(n, pa.int64())
    table = pa.table(columns)
    f = features.reduce(table, DAY0, 2, 3)
    assert not f["gap"].to_numpy().any()
    mid = (bid + columns["ask_px_0"]) / 2
    total = f["ret"].to_numpy().astype(np.float64).sum()
    assert total == pytest.approx(math.log(mid[-1] / mid[0]), abs=1e-7)
    assert f["moves"].to_numpy().sum() == np.count_nonzero(np.diff(mid))


def test_a_crossed_or_one_sided_sample_counts_as_missing() -> None:
    rows: list[Row] = [(k, True, BIDS, ASKS) for k in range(10)]
    rows[3] = (3, True, [(1002, 1)], ASKS)  # bid above the ask
    rows[6] = (6, True, [], ASKS)  # no bid
    f = _reduce(rows)
    assert (f["samples"][0], f["gap"][0]) == (8, True)
    assert f["imb_1"][0] == 0.5 and f["ret"][0] == 0


def _random_book(
    seconds: int, rng: np.random.Generator, absent: float = 0.05, unsynced: float = 0.05
) -> list[Row]:
    rows: list[Row] = []
    mid = 2000
    for slot in range(seconds * 10):
        if rng.random() < absent:
            continue  # absent
        mid += int(rng.integers(-1, 2))
        spread = 1 + int(rng.random() < 0.2)
        depth = int(rng.integers(1, 21))
        bids = [(mid - i, int(rng.integers(1, 500))) for i in range(depth)]
        asks = [(mid + spread + i, int(rng.integers(1, 500))) for i in range(depth)]
        rows.append((slot, bool(rng.random() >= unsynced), bids, asks))
    return rows


def _columns(table: pa.Table, s: slice) -> dict[str, bytes]:
    return {name: table[name].to_numpy()[s].tobytes() for name in table.column_names}


def test_features_of_a_second_depend_only_on_samples_from_its_boundary_on() -> None:
    """Second s reads [s - 0.1, s + 1): perturbing or cutting everything from s + 1 on leaves
    every second up to s identical; perturbing everything before s - 0.1 leaves s on identical."""
    rng = np.random.default_rng(30)
    rows = _random_book(8, rng)
    full = features.reduce(_table(rows), DAY0, 2, 3)
    cut = full
    for s in range(8):
        noise = _random_book(8, np.random.default_rng(100 + s))
        later = [r for r in rows if r[0] < (s + 1) * 10] + [
            r for r in noise if r[0] >= (s + 1) * 10
        ]
        boundary = s * 10 - 1  # the last sample of second s - 1
        earlier = [r for r in noise if r[0] < boundary] + [r for r in rows if r[0] >= boundary]
        cut = features.reduce(_table(later), DAY0, 2, 3)
        assert _columns(cut, slice(0, s + 1)) == _columns(full, slice(0, s + 1)), s
        truncated = features.reduce(_table([r for r in rows if r[0] < (s + 1) * 10]), DAY0, 2, 3)
        assert _columns(truncated, slice(0, s + 1)) == _columns(full, slice(0, s + 1)), s
        cut = features.reduce(_table(earlier), DAY0, 2, 3)
        assert _columns(cut, slice(s, None)) == _columns(full, slice(s, None)), s
    assert _columns(cut, slice(0, 1)) != _columns(full, slice(0, 1))  # the noise did change it


def test_the_last_sample_of_a_second_changes_that_second_and_the_next_only() -> None:
    rows = _random_book(6, np.random.default_rng(33), absent=0, unsynced=0)
    full = features.reduce(_table(rows), DAY0, 2, 3)
    for s in range(5):
        moved = list(rows)
        slot, synced, bids, asks = moved[s * 10 + 9]
        moved[s * 10 + 9] = (slot, synced, _shift(bids, 3), _shift(asks, 3))
        cut = features.reduce(_table(moved), DAY0, 2, 3)
        assert _changed(cut, full) == [s, s + 1], s


def _changed(a: pa.Table, b: pa.Table) -> list[int]:
    """The seconds where any column of two feature tables differs bitwise (NaN equal to NaN)."""
    rows = np.zeros(a.num_rows, dtype=bool)
    for name in a.column_names:
        x, y = a[name].to_numpy(), b[name].to_numpy()
        rows |= x.view(f"u{x.itemsize}") != y.view(f"u{y.itemsize}")
    return np.flatnonzero(rows).tolist()


def test_two_runs_write_the_same_bytes(tmp_path: Path) -> None:
    first = features.reduce_run(RUN / "book_samples" / book.TABLE.name, tmp_path / "a")
    second = features.reduce_run(RUN / "book_samples" / book.TABLE.name, tmp_path / "b")
    assert first == second
    path = tmp_path / "a" / first["path"]
    assert first["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # The committed synthetic day: every second it touches has a gap (it starts at 0.5 s, and
    # 1.9 to 2.6 s are unsynced).
    table = pq.read_table(path)
    assert first["path"] == "features/BTCUSDT/2026-10-20.parquet"
    assert first["rows"] == table.num_rows == 86_400
    assert first["gap_seconds"] == 86_400 and first["empty_seconds"] == 86_400 - 4
    assert table["samples"].to_numpy()[:5].tolist() == [5, 9, 3, 9, 0]
    meta = json.loads(table.schema.metadata[b"tp.book_features"])
    assert meta["symbol"] == "BTCUSDT" and meta["scales"] == {
        "price_decimals": 1,
        "qty_decimals": 3,
    }


def test_the_command_finds_the_range_folder_or_refuses(tmp_path: Path, capsys: Any) -> None:
    samples = tmp_path / "training" / "staging" / "market" / RUN.name / "book_samples"
    shutil.copytree(RUN, samples.parent)
    assert reduce_main([str(samples / book.TABLE.name)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert (tmp_path / "training" / record["path"]).is_file()
    assert reduce_main([str(RUN / "book_samples" / book.TABLE.name)]) == 1
    assert "name the folder with --out" in capsys.readouterr().err


class Archive:
    """Serves the committed synthetic zip at its URL."""

    def __init__(self) -> None:
        self.opened: list[str] = []

    def __call__(self, url: str) -> Any:
        self.opened.append(url)
        body = ZIP.read_bytes()
        response = Response(body)
        response.headers = {"Content-Length": str(len(body))}
        return response


class Response:
    def __init__(self, body: bytes) -> None:
        self.body, self.at = body, 0
        self.headers: dict[str, str] = {}

    def read(self, size: int) -> bytes:
        chunk = self.body[self.at : self.at + size]
        self.at += len(chunk)
        return chunk

    def close(self) -> None:
        pass


def _copy_importer(archive: Path, url: str, data_dir: Path) -> Path:
    """Stands in for `store import`: the committed run (samples, manifest, complete.json) that
    `store import` made of the same zip."""
    run = data_dir / "market" / RUN.name
    shutil.copytree(RUN, run)
    return run


def _wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, importer: Any) -> dict[str, Any]:
    day = date(2026, 10, 20)  # the committed day, outside every range: make it the training one
    monkeypatch.setitem(book.RANGES, "training", (day, day))
    archive = Archive()
    monkeypatch.setattr(book, "http_open", archive)
    monkeypatch.setattr(book, "engine_importer", lambda _: importer)
    store = tmp_path / "store.exe"
    store.write_bytes(b"")
    out = tmp_path / "out"
    reduce = f"{Path(sys.executable).as_posix()} -m tipping_point.book"
    argv = ["book", "--range", "training", "--symbols", "BTCUSDT", "--store", str(store)]
    assert market_main([*argv, "--out", str(out), "--reduce", reduce]) == 0
    kept = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert kept == ["training/features/BTCUSDT/2026-10-20.parquet", "training/manifest.json"]
    entry = json.loads((out / "training/manifest.json").read_text())["days"]["BTCUSDT/2026-10-20"]
    feature_file = out / "training" / entry["reduced"]["path"]
    assert entry["reduced"]["sha256"] == hashlib.sha256(feature_file.read_bytes()).hexdigest()
    assert entry["reduced"]["rows"] == 86_400 and entry["reduced"]["gap_seconds"] == 86_400
    # A re-run skips the day while its feature file is there, and makes it again once it is not.
    opened = len(archive.opened)
    assert market_main([*argv, "--out", str(out), "--reduce", reduce]) == 0
    assert len(archive.opened) == opened
    feature_file.unlink()
    assert market_main([*argv, "--out", str(out), "--reduce", reduce]) == 0
    assert len(archive.opened) == opened + 1 and feature_file.is_file()
    return entry


def test_market_book_reduce_keeps_only_the_feature_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _wired(tmp_path, monkeypatch, _copy_importer)
    direct = features.reduce_run(RUN / "book_samples" / book.TABLE.name, tmp_path / "direct")
    assert entry["reduced"] == direct


@pytest.mark.skipif(not os.environ.get("TP_STORE_BIN"), reason="TP_STORE_BIN names no binary")
def test_market_book_reduce_with_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    importer = book.engine_importer(Path(os.environ["TP_STORE_BIN"]))
    entry = _wired(tmp_path, monkeypatch, importer)
    direct = features.reduce_run(RUN / "book_samples" / book.TABLE.name, tmp_path / "direct")
    assert entry["reduced"]["sha256"] == direct["sha256"]  # the committed samples are the engine's


def test_batches_in_any_order_give_the_same_table() -> None:
    table = _table(_random_book(5, np.random.default_rng(31)))
    whole = features.reduce(table, DAY0, 2, 3)
    assert len(table.to_batches(7)) > 1
    split = features.reduce(table.to_batches(7)[::-1], DAY0, 2, 3)
    assert _columns(split, slice(None)) == _columns(whole, slice(None))  # bitwise, NaN too
    with pytest.raises(features.FeatureError, match="share a grid instant"):
        features.reduce([*table.to_batches(), *table.to_batches()], DAY0, 2, 3)
