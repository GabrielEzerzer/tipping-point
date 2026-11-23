# The panel's inputs on planted bars and planted archive files: no input moves when later data
# changes or goes, each group against a direct calculation, the 5/6 and carry rules, window-local
# classic statistics, the ratio reader, the market mean's 15 coins and the held-back refusals.

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from market_planted import checksum, vision, zipped
from scipy.stats import kendalltau
from tipping_point.ews import indicators
from tipping_point.market import archive, bars
from tipping_point.panel import inputs, labels, span, store
from tipping_point.panel.__main__ import main

DAY = bars.DAY_BARS
T0 = labels.ns(date(2022, 1, 1))
M, B = 4e-6, 0.02
SD = float(np.sqrt(inputs.NEAR_BARS * M))


def coin(
    columns: dict[str, np.ndarray], ratio: np.ndarray, stop: np.ndarray
) -> dict[str, np.ndarray]:
    return inputs.coin(columns, ratio, stop, np.full(len(stop), SD), np.full(len(stop), B))


def planted(seed: int = 5, days: int = 40) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    n = days * DAY
    columns = {
        "ts_ns": T0 + inputs.BAR_NS * np.arange(n, dtype=np.int64),
        "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))),
        "volume": 100 * np.exp(rng.normal(0, 0.3, n)),
        "taker_buy_share": rng.uniform(0.3, 0.7, n),
        "oi": 1e6 * np.exp(np.cumsum(rng.normal(0, 0.001, n))),
        "funding_rate": np.repeat(rng.normal(1e-4, 1e-4, n // 96 + 1), 96)[:n],
    }
    return columns, 1.5 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


def gapped(seed: int = 5, days: int = 40) -> tuple[dict[str, np.ndarray], np.ndarray]:
    columns, ratio = planted(seed, days)
    rng = np.random.default_rng(seed + 1)
    n = len(ratio)
    for name in ("close", "volume", "taker_buy_share"):
        columns[name][rng.choice(n, 100, replace=False)] = np.nan
    columns["close"][32 * DAY : 32 * DAY + 30] = np.nan
    columns["oi"][rng.choice(n, 300, replace=False)] = np.nan
    columns["oi"][rng.choice(n, 20, replace=False)] = 0.0
    columns["oi"][35 * DAY : 35 * DAY + 20] = np.nan
    ratio[rng.choice(n, 300, replace=False)] = np.nan
    ratio[34 * DAY : 34 * DAY + 40] = np.nan
    return columns, ratio


def ticked(seed: int = 5, days: int = 40) -> tuple[dict[str, np.ndarray], np.ndarray]:
    columns, ratio = gapped(seed, days)
    rng = np.random.default_rng(seed + 2)
    path = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(ratio))))
    columns["close"] = np.where(np.isfinite(columns["close"]), 0.1 * np.round(path / 0.1), np.nan)
    return columns, ratio


GAPS = ((32 * DAY, 30), (34 * DAY, 40), (35 * DAY, 20))
GAP_STOPS = np.array(sorted({s for g, n in GAPS for s in range(g, g + n + 25, 12)}))
STOPS = np.union1d(12 * np.arange(30 * 24, 40 * 24, 5), GAP_STOPS)
WILD = np.array([np.nan, 0.0, 1e15])


def scrambled(x: np.ndarray, later: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    out[later] = x[later] * np.exp(rng.normal(0, 0.2, later.size))
    wild = later[rng.random(later.size) < 0.3]
    out[wild] = rng.choice(WILD, wild.size)
    return out


def later_changed(
    columns: dict[str, np.ndarray], ratio: np.ndarray, k: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    changed = dict(columns)
    from_t, after_t = np.arange(k, len(ratio)), np.arange(k + 1, len(ratio))
    for name in ("close", "volume", "taker_buy_share", "funding_rate"):
        changed[name] = scrambled(columns[name], from_t, rng)
    changed["oi"] = scrambled(columns["oi"], after_t, rng)
    return coin(changed, scrambled(ratio, from_t, rng), np.array([k]))


def truncated(columns: dict[str, np.ndarray], ratio: np.ndarray, k: int) -> dict[str, np.ndarray]:
    cut = {name: values[: k + 1] for name, values in columns.items()}
    return coin(cut, ratio[: k + 1], np.array([k]))


def same(a: float, b: float) -> bool:
    return bool(np.isnan(a) and np.isnan(b)) or a == b


@pytest.mark.parametrize("series", [gapped, ticked])
def test_no_input_reads_a_bar_starting_at_or_after_t_or_a_row_stamped_after_t(
    series: Callable[[], tuple[dict[str, np.ndarray], np.ndarray]],
) -> None:
    columns, ratio = series()
    if series is ticked:
        returns = np.diff(np.log(columns["close"]))
        assert (returns == 0).mean() > 0.3
    base = coin(columns, ratio, STOPS)
    assert all(np.isfinite(base[name]).mean() > 0.5 for name in inputs.OWN)
    rng = np.random.default_rng(7)
    picked = set(range(0, len(STOPS), 9)) | set(np.flatnonzero(np.isin(STOPS, GAP_STOPS)).tolist())
    for i in sorted(picked):
        k = int(STOPS[i])
        for got in (later_changed(columns, ratio, k, rng), truncated(columns, ratio, k)):
            for name in inputs.OWN:
                assert same(got[name][0], base[name][i]), (name, k)


def test_the_volatility_scale_and_b_are_the_hours_own() -> None:
    columns, ratio = planted()
    stops = 12 * np.arange(30 * 24, 30 * 24 + 6)
    sd = SD * np.array([1.0, 2.0, np.nan, 0.5, 1.0, 3.0])
    b = B * np.array([1.0, 0.5, 2.0, np.nan, 4.0, 1.0])
    got = inputs.coin(columns, ratio, stops, sd, b)
    base = coin(columns, ratio, stops)
    for i in range(len(stops)):
        for name in inputs.WINDOWS:
            expected = base[f"vol_{name}"][i] - 2 * np.log(sd[i] / SD)
            assert got[f"vol_{name}"][i] == pytest.approx(expected, nan_ok=True)
        assert got["near_24h"][i] == pytest.approx(base["near_24h"][i] * B / b[i], nan_ok=True)
    assert np.isnan(got["vol_24h"][2]) and np.isfinite(got["near_24h"][2])
    assert np.isnan(got["near_24h"][3]) and np.isfinite(got["vol_24h"][3])
    for name in inputs.OWN:
        if not name.startswith("vol_") and name != "near_24h":
            assert np.array_equal(got[name], base[name], equal_nan=True), name


def test_the_latest_data_allowed_is_read() -> None:
    columns, ratio = planted()
    k = int(STOPS[40])
    base = coin(columns, ratio, np.array([k]))
    checks = [
        ("close", k - 1, ["ret_1h", "vol_1h", "near_24h", "dd_7d", *inputs.GROUPS["classic"]]),
        ("volume", k - 1, ["vsurge_1h", "volume_24h", "flow_1h"]),
        ("taker_buy_share", k - 1, ["flow_1h", "flow_24h"]),
        ("oi", k, ["oi_24h", "oi_7d", "oi_vs_28d"]),
        ("funding_rate", k - 1, ["funding", "funding_rel"]),
    ]
    for column, at, names in checks:
        changed = {name: values.copy() for name, values in columns.items()}
        changed[column][at] *= 1.5
        got = coin(changed, ratio, np.array([k]))
        for name in names:
            assert got[name][0] != base[name][0], (column, name)
    moved = ratio.copy()
    moved[k - 1] *= 1.5
    got = coin(columns, moved, np.array([k]))
    assert got["ls"][0] != base["ls"][0] and got["ls_rel"][0] != base["ls_rel"][0]


def test_vol_and_price_against_a_direct_calculation() -> None:
    columns, ratio = planted()
    k = 35 * DAY + 12 * 7
    got = {n: v[0] for n, v in coin(columns, ratio, np.array([k])).items()}
    close, volume = columns["close"], columns["volume"]
    r = np.diff(np.log(close))
    for name, w in inputs.WINDOWS.items():
        assert got[f"vol_{name}"] == pytest.approx(np.log(np.mean(r[k - 1 - w : k - 1] ** 2) / M))
        assert got[f"ret_{name}"] == pytest.approx(np.log(close[k - 1] / close[k - 1 - w]))
    q = [abs(np.log(close[j] / close[j - 6])) for j in range(k - 288, k)]
    assert got["near_24h"] == pytest.approx(max(q) / B)
    expected = np.log(np.mean(volume[k - 12 : k]) / np.median(volume[k - 8064 : k]))
    assert got["vsurge_1h"] == pytest.approx(expected)
    expected = np.log(np.mean(volume[k - 288 : k]) / np.mean(volume[k - 8064 : k]))
    assert got["volume_24h"] == pytest.approx(expected)
    assert got["aret_24h"] == abs(got["ret_24h"]) and got["aret_7d"] == abs(got["ret_7d"])
    assert got["dd_7d"] == pytest.approx(np.log(close[k - 1] / close[k - 2016 : k].max()))
    assert got["dd_28d"] == pytest.approx(np.log(close[k - 1] / close[k - 8064 : k].max()))
    assert got["ru_28d"] == pytest.approx(np.log(close[k - 1] / close[k - 8064 : k].min()))
    taker = columns["taker_buy_share"] * volume
    for name, w in (("flow_1h", 12), ("flow_24h", 288)):
        v, t = volume[k - w : k].sum(), taker[k - w : k].sum()
        assert got[name] == pytest.approx((2 * t - v) / v)


def test_classic_against_ews_indicators_and_scipy() -> None:
    columns, ratio = planted()
    k = 36 * DAY + 12 * 5
    got = {n: v[0] for n, v in coin(columns, ratio, np.array([k])).items()}
    r = np.diff(np.log(columns["close"]))
    variance = indicators.rolling_variance(r, 72)[k - 289 : k - 1]
    ac = indicators.rolling_autocorrelation(r, 72, 1)[k - 289 : k - 1]
    assert got["var_tau_24h"] == pytest.approx(kendalltau(np.arange(288), variance)[0])
    assert got["ac1_tau_24h"] == pytest.approx(kendalltau(np.arange(288), ac)[0])
    hourly = np.diff(np.log(columns["close"][11::12]))
    j = k // 12
    variance = indicators.rolling_variance(hourly, 84)[j - 169 : j - 1]
    ac = indicators.rolling_autocorrelation(hourly, 84, 1)[j - 169 : j - 1]
    assert got["var_tau_7d"] == pytest.approx(kendalltau(np.arange(168), variance)[0])
    assert got["ac1_tau_7d"] == pytest.approx(kendalltau(np.arange(168), ac)[0])


def test_lev_against_a_direct_calculation_and_the_carry_rule() -> None:
    columns, ratio = planted()
    k = 37 * DAY
    oi, funding = columns["oi"], columns["funding_rate"]
    got = {n: v[0] for n, v in coin(columns, ratio, np.array([k])).items()}
    assert got["oi_24h"] == pytest.approx(np.log(oi[k] / oi[k - 288]))
    assert got["oi_7d"] == pytest.approx(np.log(oi[k] / oi[k - 2016]))
    assert got["oi_vs_28d"] == pytest.approx(np.log(oi[k] / np.median(oi[k - 8063 : k + 1])))
    assert got["funding"] == funding[k - 1]
    assert got["funding_rel"] == pytest.approx(funding[k - 1] - np.mean(funding[k - 8053 : k : 12]))
    ls = np.log(ratio)
    assert got["ls"] == pytest.approx(ls[k - 1])
    assert got["ls_rel"] == pytest.approx(ls[k - 1] - np.mean(ls[k - 8064 : k]))

    held = {name: values.copy() for name, values in columns.items()}
    held["oi"][k - 11 : k + 1] = np.nan
    gap = ratio.copy()
    gap[k - 12 : k] = -1.0
    got = {n: v[0] for n, v in coin(held, gap, np.array([k])).items()}
    assert got["oi_24h"] == pytest.approx(np.log(oi[k - 12] / oi[k - 288]))
    assert got["ls"] == pytest.approx(ls[k - 13])
    held["oi"][k - 12] = 0.0
    gap[k - 13] = np.nan
    got = {n: v[0] for n, v in coin(held, gap, np.array([k])).items()}
    assert np.isnan(got["oi_24h"]) and np.isnan(got["oi_vs_28d"]) and np.isnan(got["ls"])


def test_the_28_day_baselines_count_only_observed_values() -> None:
    columns, ratio = planted()
    k = 37 * DAY
    sparse = np.ones(len(ratio), bool)
    sparse[k - 8064 : k + 1] = False
    sparse[k - 8064 : k + 1 : 6] = True
    sparse[k - 1 : k + 1] = True
    columns["oi"][~sparse] = np.nan
    ratio = np.where(sparse, ratio, np.nan)
    got = {n: v[0] for n, v in coin(columns, ratio, np.array([k])).items()}
    assert np.isfinite([got["oi_24h"], got["oi_7d"], got["ls"]]).all()
    assert np.isnan(got["oi_vs_28d"]) and np.isnan(got["ls_rel"])


def test_classic_values_depend_only_on_their_windows() -> None:
    columns, ratio = ticked(days=70)
    later = {name: values[30 * DAY :] for name, values in columns.items()}
    stops = 12 * np.arange(30 * 24, 40 * 24, 7)
    early = coin(columns, ratio, stops + 30 * DAY)
    late = coin(later, ratio[30 * DAY :], stops)
    for name in inputs.GROUPS["classic"]:
        assert np.isfinite(late[name]).mean() > 0.5, name
        assert np.array_equal(early[name], late[name], equal_nan=True), name
    x = np.random.default_rng(1).normal(size=400)
    x[100:172], x[200:272] = 0.0, 0.37
    variance, ac = inputs.rolling_variance(x, 72), inputs.rolling_autocorrelation(x, 72)
    for end in (171, 271):
        assert np.isnan(variance[end]) and np.isnan(ac[end])
        assert np.isfinite(variance[end - 1]) and np.isfinite(variance[end + 1])


def test_a_rolling_value_needs_five_sixths_of_its_window() -> None:
    columns, ratio = planted()
    k = 36 * DAY
    columns["close"][k - 5] = np.nan
    got = coin(columns, ratio, np.array([k]))
    r = np.diff(np.log(columns["close"]))[k - 13 : k - 1]
    assert np.isfinite(r).sum() == 10
    assert got["vol_1h"][0] == pytest.approx(np.log(np.nanmean(r**2) / M))
    columns["close"][k - 9] = np.nan
    assert np.isnan(coin(columns, ratio, np.array([k]))["vol_1h"][0])
    short = {name: values[: 25 * DAY] for name, values in columns.items()}
    got = coin(short, ratio[: 25 * DAY], np.array([20 * DAY]))
    assert np.isnan(got["vol_28d"][0]) and np.isfinite(got["vol_7d"][0])


def test_the_market_mean_needs_fifteen_coins() -> None:
    rng = np.random.default_rng(3)
    own = [{name: rng.normal(size=4) for name in inputs.OWN} for _ in range(16)]
    for i in range(16):
        own[i]["vol_1h"][1] = np.nan if i == 0 else own[i]["vol_1h"][1]
        own[i]["vol_1h"][2] = np.nan if i < 2 else own[i]["vol_1h"][2]
    got = inputs.market(own)
    column = np.array([o["vol_1h"] for o in own])
    assert got["m_vol_1h"][0] == pytest.approx(column[:, 0].mean())
    assert got["m_vol_1h"][1] == pytest.approx(column[1:, 1].mean())
    assert np.isnan(got["m_vol_1h"][2])
    assert set(got) == set(inputs.MARKET)


def metrics_row(minute: int, ratio: str, day: date = date(2022, 3, 1)) -> str:
    stamp = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(minutes=minute)
    return f"{stamp:%Y-%m-%d %H:%M:%S},XUSDT,100.0,2000.0,1.9,1.05,{ratio},0.95"


def metrics_file(path: Path, rows: list[str], header: bool = True) -> Path:
    head = ["create_time,symbol,a,b,c,d,count_long_short_ratio,e"] if header else []
    data = zipped(path.stem + ".csv", ("\n".join(head + rows) + "\n").encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.with_name(path.name + ".CHECKSUM").write_bytes(checksum(path.name, data))
    return path


def test_the_ratio_reader_takes_duplicate_unsorted_null_and_missing_rows(tmp_path: Path) -> None:
    rows = [
        metrics_row(0, "1.1"),
        metrics_row(10, "1.3"),
        metrics_row(5, "1.2"),
        metrics_row(15, "1.4"),
        metrics_row(15, "9.9"),
        metrics_row(20, ""),
        metrics_row(27, "5.0"),
    ]
    for header in (True, False):
        found = inputs.read_ratio(metrics_file(tmp_path / f"m{header}.zip", rows, header))
        grid = inputs.on_grid(found, archive.day_ms(date(2022, 3, 1)), 288)
        assert np.array_equal(grid[:6], [1.1, 1.2, 1.3, 1.4, np.nan, np.nan], equal_nan=True)
        assert np.isnan(grid[6:]).all()
    no_stamp = metrics_file(tmp_path / "n.zip", [metrics_row(0, "1.1"), ",XUSDT,1,2,3,4,1.2,5"])
    with pytest.raises(archive.ArchiveError, match="no create_time"):
        inputs.read_ratio(no_stamp)
    broken = tmp_path / "b.zip"
    data = bytearray(zipped("b.csv", ("\n".join(rows * 50) + "\n").encode()))
    data[60:90] = bytes(30)
    broken.write_bytes(bytes(data))
    with pytest.raises(archive.ArchiveError):
        inputs.read_ratio(broken)
    days = [date(2022, 3, 1) + timedelta(days=d) for d in range(3)]
    root = tmp_path / "root"
    for d in (days[0], days[2]):
        source = archive.Source("metrics", "XUSDT", f"{d:%Y-%m-%d}")
        metrics_file(root / source.path, [metrics_row(5, "2.0", d)])
    with pytest.raises(archive.ArchiveError, match="not fetched"):
        inputs.ratios(root, "XUSDT", days[0], days[2])
    gone = archive.Source("metrics", "XUSDT", f"{days[1]:%Y-%m-%d}").path
    (root / "manifest.json").write_text(
        json.dumps({"files": {gone: {"status": "missing"}}}), encoding="utf-8"
    )
    found = inputs.ratios(root, "XUSDT", days[0], days[2])
    grid = inputs.on_grid(found, archive.day_ms(days[0]), 3 * DAY)
    assert np.flatnonzero(np.isfinite(grid)).tolist() == [1, 2 * DAY + 1]


E2E_START, E2E_DEV, E2E_END = date(2022, 11, 1), date(2022, 12, 1), date(2022, 12, 10)
T = labels.ns(date(2022, 12, 6)) + 16 * labels.HOUR_NS
MINUTE_MS = 60_000


def raw_archive(seed: int = 3) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    days = (E2E_END - E2E_START).days + 1
    start = archive.day_ms(E2E_START)
    n, m = days * 1440, days * DAY
    close = 20_000 * np.exp(np.cumsum(rng.normal(0, 5e-4, n)))
    opens = np.concatenate([[20_000.0], close[:-1]])
    volume = rng.uniform(5.0, 50.0, n)
    return {
        "klines": {
            "open_time": start + MINUTE_MS * np.arange(n, dtype=np.int64),
            "open": opens,
            "high": np.maximum(opens, close) * (1 + rng.uniform(0, 1e-4, n)),
            "low": np.minimum(opens, close) * (1 - rng.uniform(0, 1e-4, n)),
            "close": close,
            "volume": volume,
            "count": rng.integers(50, 1000, n).astype(np.float64),
            "taker": volume * rng.uniform(0.3, 0.7, n),
        },
        "metrics": {
            "create_time": start + 5 * MINUTE_MS * np.arange(m, dtype=np.int64),
            "oi": 1e5 * np.exp(np.cumsum(rng.normal(0, 0.002, m))),
            "ratio": 1.5 * np.exp(np.cumsum(rng.normal(0, 0.01, m))),
        },
        "fundingRate": {
            "calc_time": start
            + 8 * 3_600_000 * np.arange(3 * days)
            + 4 * (np.arange(3 * days) % 3 == 1),
            "rate": rng.normal(1e-4, 1e-4, 3 * days),
        },
    }


def _value(v: float) -> str:
    return "" if np.isnan(v) else repr(float(v))


def kline_line(kl: dict[str, np.ndarray], i: int) -> str:
    stamp = int(kl["open_time"][i])
    prices = ",".join(_value(float(kl[k][i])) for k in ("open", "high", "low", "close", "volume"))
    taker = _value(float(kl["taker"][i]))
    return f"{stamp},{prices},{stamp + MINUTE_MS - 1},0,{int(kl['count'][i])},{taker},0,0"


def metrics_line(me: dict[str, np.ndarray], i: int) -> str:
    stamp = datetime.fromtimestamp(int(me["create_time"][i]) / 1000, UTC)
    oi, ratio = _value(float(me["oi"][i])), _value(float(me["ratio"][i]))
    return f"{stamp:%Y-%m-%d %H:%M:%S},XUSDT,{oi},1,1.9,1.05,{ratio},0.95"


def write_archive(root: Path, raw: dict[str, dict[str, np.ndarray]], last: date) -> Path:
    fu = raw["fundingRate"]
    stamped = {"klines": ("open_time", kline_line), "metrics": ("create_time", metrics_line)}
    missing = {}
    day = E2E_START
    while day <= E2E_END:
        lo, hi = archive.day_ms(day), archive.day_ms(day + timedelta(days=1))
        for kind, (stamp, line) in stamped.items():
            source = archive.Source(kind, "XUSDT", f"{day:%Y-%m-%d}")
            stamps = raw[kind][stamp]
            rows = np.flatnonzero((stamps >= lo) & (stamps < hi))
            if day > last or not rows.size:
                missing[source.path] = {"status": "missing"}
                continue
            lines = [line(raw[kind], int(i)) for i in rows]
            text = "\n".join([market_header(kind), *lines]) + "\n"
            put(root / source.path, source.name, text)
        day += timedelta(days=1)
    for month in (date(2022, 11, 1), date(2022, 12, 1)):
        source = archive.Source("fundingRate", "XUSDT", f"{month:%Y-%m}")
        lo, hi = archive.day_ms(month), archive.day_ms((month + timedelta(days=32)).replace(day=1))
        rows = np.flatnonzero((fu["calc_time"] >= lo) & (fu["calc_time"] < hi))
        lines = [f"{int(fu['calc_time'][i])},8,{_value(float(fu['rate'][i]))}" for i in rows]
        text = "\n".join([market_header("fundingRate"), *lines]) + "\n"
        put(root / source.path, source.name, text)
    (root / "manifest.json").write_text(json.dumps({"files": missing}), encoding="utf-8")
    return root


def market_header(kind: str) -> str:
    return ",".join(archive._COLUMNS[kind])


def put(path: Path, name: str, csv: str) -> None:
    data = zipped(name.removesuffix(".zip") + ".csv", csv.encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.with_name(path.name + ".CHECKSUM").write_bytes(checksum(name, data))


def daily_thresholds(end: date, scrambled_after: date | None = None) -> pa.Table:
    days = [E2E_START + timedelta(days=d) for d in range((end - E2E_START).days + 1)]
    sd = np.array([0.004 * (1 + 0.1 * np.sin(d)) for d in range(len(days))])
    if scrambled_after is not None:
        later = np.array([day > scrambled_after for day in days])
        sd = np.where(later, sd * 7.0, sd)
        sd[np.flatnonzero(later)[::2]] = np.nan
    return pa.table(
        {
            "ts_ns": pa.array([labels.ns(day) for day in days], pa.int64()),
            "symbol": ["XUSDT"] * len(days),
            "sd_ret": pa.array(sd, pa.float64(), mask=np.isnan(sd)),
            "b": pa.array(5.0 * sd, pa.float64(), mask=np.isnan(sd)),
        }
    )


def fake_labels(
    folder: Path, end: date, hours: np.ndarray, scrambled_after: date | None = None
) -> Path:
    grid = pa.table(
        {"ts_ns": pa.array(hours, pa.int64()), "symbol": pa.array(["XUSDT"] * len(hours))}
    )
    thresholds = daily_thresholds(end, scrambled_after)
    files = {}
    for name, table in (("grid/g.parquet", grid), ("thresholds/t.parquet", thresholds)):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, folder / name)
        files[name] = store.sha256(folder / name)
    params = {
        "start": E2E_START.isoformat(),
        "dev_start": E2E_DEV.isoformat(),
        "end": end.isoformat(),
        "symbols": ["XUSDT"],
    }
    manifest = {"run_id": folder.name, "params": params}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "complete.json").write_text(json.dumps({"files": files, "clean": True}), "utf-8")
    return folder


def grid_hours(end: date) -> np.ndarray:
    hours = np.arange(labels.ns(E2E_DEV), labels.ns(end + timedelta(days=1)), labels.HOUR_NS)
    return hours[hours <= T] if end < E2E_END else hours


def inputs_table(
    tmp_path: Path, name: str, raw: dict, last: date, end: date, scrambled_after: date | None = None
) -> pa.Table:
    root = write_archive(tmp_path / name / "archive", raw, last)
    folder = fake_labels(tmp_path / name / "labels" / name, end, grid_hours(end), scrambled_after)
    run, _ = inputs.run(folder, root, tmp_path / name / "data", "t")
    return pq.read_table(run / "inputs.parquet")


def row_at_t(
    tmp_path: Path, name: str, raw: dict, last: date, end: date, scrambled_after: date | None = None
) -> dict[str, float]:
    table = inputs_table(tmp_path, name, raw, last, end, scrambled_after)
    (i,) = np.flatnonzero(table["ts_ns"].to_numpy() == T)
    return {n: float(table[n][i].as_py()) for n in inputs.OWN}


def test_funding_stamped_late_or_on_the_hour_gives_the_same_inputs(tmp_path: Path) -> None:
    raw = raw_archive()
    exact = {kind: dict(cols) for kind, cols in raw.items()}
    exact["fundingRate"]["calc_time"] = raw["fundingRate"]["calc_time"] // 3_600_000 * 3_600_000
    late = {kind: dict(cols) for kind, cols in raw.items()}
    late["fundingRate"]["calc_time"] = exact["fundingRate"]["calc_time"] + 7
    first = inputs_table(tmp_path, "exact", exact, E2E_END, E2E_END)
    second = inputs_table(tmp_path, "late", late, E2E_END, E2E_END)
    assert np.isfinite(first["funding"].to_numpy()).all()
    for name in inputs.OWN:
        assert np.array_equal(first[name].to_numpy(), second[name].to_numpy(), equal_nan=True)


def test_no_input_through_the_command_reads_data_after_its_hour(tmp_path: Path) -> None:
    raw = raw_archive()
    t_ms = T // 1_000_000
    rng = np.random.default_rng(4)
    changed = {kind: dict(cols) for kind, cols in raw.items()}
    after = {
        "klines": raw["klines"]["open_time"] >= t_ms,
        "ratio": raw["metrics"]["create_time"] >= t_ms,
        "oi": raw["metrics"]["create_time"] > t_ms,
        "fundingRate": raw["fundingRate"]["calc_time"] >= t_ms,
    }
    for name in ("open", "high", "low", "close", "volume", "taker"):
        changed["klines"][name] = scrambled(
            raw["klines"][name], np.flatnonzero(after["klines"]), rng
        )
    for name in ("ratio", "oi"):
        changed["metrics"][name] = scrambled(raw["metrics"][name], np.flatnonzero(after[name]), rng)
    changed["fundingRate"]["rate"] = scrambled(
        raw["fundingRate"]["rate"], np.flatnonzero(after["fundingRate"]), rng
    )
    dropped = {
        "klines": {k: v[~after["klines"]] for k, v in raw["klines"].items()},
        "metrics": {k: v[~after["oi"]] for k, v in raw["metrics"].items()},
        "fundingRate": {k: v[~after["fundingRate"]] for k, v in raw["fundingRate"].items()},
    }
    dropped["metrics"]["ratio"] = np.where(
        dropped["metrics"]["create_time"] >= t_ms, np.nan, dropped["metrics"]["ratio"]
    )
    day_of_t = date(2022, 12, 6)
    base = row_at_t(tmp_path, "base", raw, E2E_END, E2E_END)
    assert np.isfinite(list(base.values())).all()
    variants = {
        "changed": (changed, E2E_END, E2E_END, day_of_t),
        "dropped": (dropped, day_of_t, E2E_END, None),
        "truncated": (dropped, day_of_t, day_of_t, None),
    }
    for name, (data, last, end, scrambled_after) in variants.items():
        got = row_at_t(tmp_path, name, data, last, end, scrambled_after)
        for column in inputs.OWN:
            assert same(got[column], base[column]), (name, column)


def test_the_command_refuses_unclean_altered_or_unbounded_labels_and_missing_funding(
    tmp_path: Path,
) -> None:
    raw = raw_archive()
    root = write_archive(tmp_path / "archive", raw, E2E_END)
    hours = grid_hours(E2E_END)

    def case(name: str) -> tuple[Path, dict]:
        folder = fake_labels(tmp_path / "labels" / name, E2E_END, hours)
        return folder, json.loads((folder / "complete.json").read_text(encoding="utf-8"))

    def attempt(folder: Path) -> None:
        inputs.run(folder, root, tmp_path / "data", "t")

    folder, complete = case("unclean")
    complete["clean"] = False
    (folder / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="clean"):
        attempt(folder)
    folder, complete = case("altered")
    complete["files"]["thresholds/t.parquet"] = "0" * 64
    (folder / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="sha256"):
        attempt(folder)
    folder, _ = case("reaching")
    late = np.append(hours, labels.ns(span.HELD_BACK) + labels.HOUR_NS)
    pq.write_table(
        pa.table({"ts_ns": pa.array(late, pa.int64()), "symbol": ["XUSDT"] * len(late)}),
        folder / "grid/g.parquet",
    )
    with pytest.raises(span.HeldBackError, match="reaches"):
        attempt(folder)
    folder, complete = case("unbounded")
    pq.write_table(
        pq.read_table(folder / "grid/g.parquet"), folder / "grid/g.parquet", write_statistics=False
    )
    complete["files"]["grid/g.parquet"] = store.sha256(folder / "grid/g.parquet")
    (folder / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(span.HeldBackError, match="no bounds"):
        attempt(folder)
    needed = labels.ns(E2E_DEV + timedelta(days=3))
    for name in ("short", "twice"):
        folder, complete = case(name)
        table = pq.read_table(folder / "thresholds/t.parquet")
        if name == "short":
            table = table.filter(pa.array(table["ts_ns"].to_numpy() != needed))
        else:
            table = pa.concat_tables([table, table.slice(3, 1)])
        pq.write_table(table, folder / "thresholds/t.parquet")
        complete["files"]["thresholds/t.parquet"] = store.sha256(folder / "thresholds/t.parquet")
        (folder / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
        with pytest.raises(ValueError, match="no row" if name == "short" else "twice"):
            attempt(folder)
    folder, _ = case("unfunded")
    gone = archive.Source("fundingRate", "XUSDT", "2022-12")
    (root / gone.path).unlink()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][gone.path] = {"status": "missing"}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(archive.ArchiveError, match="funding files missing"):
        attempt(folder)
    assert not (tmp_path / "data" / "panel-inputs").exists()


SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIRST = date(2022, 11, 28)


@pytest.fixture(scope="module")
def labelled(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    fake = vision(SYMBOLS, FIRST, 34, seed=11)
    dest = tmp_path_factory.mktemp("archive")
    archive.fetch(SYMBOLS, FIRST, date(2022, 12, 31), dest, fake.get)
    data = tmp_path_factory.mktemp("data")
    run, _ = labels.run(dest, data, SYMBOLS, FIRST, date(2022, 12, 29), date(2022, 12, 31), "t")
    complete = json.loads((run / "complete.json").read_text(encoding="utf-8"))
    (name,) = [f for f in complete["files"] if f.startswith("thresholds/")]
    once = pq.read_table(run / name).to_pylist()
    days = [labels.ns(FIRST + timedelta(days=d)) for d in range(34)]
    rows = [r | {"ts_ns": day} for r in once for day in days]
    rows[len(days) + 31]["sd_ret"] = None
    pq.write_table(pa.Table.from_pylist(rows, schema=labels.THRESHOLDS), run / name)
    complete["files"][name] = store.sha256(run / name)
    (run / "complete.json").write_text(json.dumps(complete), encoding="utf-8")
    return dest, run


def test_the_command_writes_an_inputs_run_on_the_grids_rows(
    labelled: tuple[Path, Path], tmp_path: Path
) -> None:
    dest, labels_dir = labelled
    where = ["--archive", str(dest), "--data-dir", str(tmp_path)]
    assert main(["inputs", "--labels", str(labels_dir), *where]) == 0
    (run,) = (tmp_path / "panel-inputs").glob("*-inputs")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["params"]["labels_run"] == labels_dir.name
    table = pq.read_table(run / "inputs.parquet")
    assert table.schema.equals(inputs.SCHEMA, check_metadata=False)
    grid = pq.read_table(next((labels_dir / "grid").glob("*.parquet")))
    assert table["ts_ns"].equals(grid["ts_ns"]) and table["symbol"].equals(grid["symbol"])
    assert set(json.loads((run / "complete.json").read_text(encoding="utf-8"))["files"]) == {
        "inputs.parquet"
    }
    thresholds = pq.read_table(next((labels_dir / "thresholds").glob("*.parquet")))
    got = archive.collect(["ETHUSDT"], FIRST, date(2022, 12, 31), dest)
    columns, _ = bars.bars(
        archive.day_ms(FIRST), 34, *(got.columns[("ETHUSDT", k)] for k in archive.KINDS)
    )
    ratio = inputs.on_grid(
        inputs.ratios(dest, "ETHUSDT", FIRST, date(2022, 12, 31)), archive.day_ms(FIRST), 34 * DAY
    )
    assert np.nanmax(ratio) == np.nanmin(ratio) == 2.1
    hours = np.unique(grid["ts_ns"].to_numpy())
    stop = (hours - labels.ns(FIRST)) // inputs.BAR_NS
    sd_ret, b = inputs.daily_scales(thresholds, "ETHUSDT", hours)
    expected = inputs.coin(columns, ratio, stop, sd_ret, b)
    eth = table.filter(np.array(table["symbol"].to_pylist()) == "ETHUSDT")
    for name in inputs.OWN:
        assert np.array_equal(eth[name].to_numpy(), expected[name], equal_nan=True), name
    unscaled = (hours - hours % inputs.DAY_NS) == labels.ns(date(2022, 12, 29))
    vol = eth["vol_28d"].to_numpy()
    assert unscaled.any() and np.isnan(vol[unscaled]).all() and np.isfinite(vol[~unscaled]).all()
    assert all(np.isnan(table[name].to_numpy()).all() for name in inputs.MARKET)


def test_the_command_refuses_a_labels_run_reaching_the_held_back_span(tmp_path: Path) -> None:
    folder = tmp_path / "labels" / "x"
    folder.mkdir(parents=True)
    params = {"start": "2023-10-01", "dev_start": "2023-12-01", "end": "2024-01-01"}
    (folder / "manifest.json").write_text(json.dumps({"params": params}), encoding="utf-8")
    assert main(["inputs", "--labels", str(folder), "--data-dir", str(tmp_path)]) == 1
    with pytest.raises(span.HeldBackError):
        inputs.labels_run(folder)
    with pytest.raises(span.HeldBackError):
        inputs.ratios(tmp_path, "BTCUSDT", date(2023, 12, 30), date(2024, 1, 1))
    assert not (tmp_path / "panel-inputs").exists()
