# The panel's inputs on planted bars: no input moves when data at or after its hour changes, each
# group against a direct calculation, the 5/6 and carry rules, the market mean's 15 coins, and the
# command on a fake archive with its held-back refusal.

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from market_planted import vision
from scipy.stats import kendalltau
from tipping_point.ews import indicators
from tipping_point.market import archive, bars
from tipping_point.panel import inputs, labels, span
from tipping_point.panel.__main__ import main

DAY = bars.DAY_BARS
T0 = labels.ns(date(2022, 1, 1))
M, B = 4e-6, 0.02


def planted(days: int = 40, seed: int = 5) -> tuple[dict[str, np.ndarray], np.ndarray]:
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


def gapped(seed: int = 5) -> tuple[dict[str, np.ndarray], np.ndarray]:
    columns, ratio = planted(seed=seed)
    rng = np.random.default_rng(seed + 1)
    n = len(ratio)
    for name in ("close", "volume", "taker_buy_share"):
        columns[name][rng.choice(n, 100, replace=False)] = np.nan
    columns["close"][32 * DAY : 32 * DAY + 30] = np.nan
    columns["oi"][rng.choice(n, 300, replace=False)] = np.nan
    columns["oi"][rng.choice(n, 20, replace=False)] = 0.0
    columns["oi"][35 * DAY : 35 * DAY + 20] = np.nan
    ratio[rng.choice(n, 300, replace=False)] = np.nan
    return columns, ratio


def ticked(seed: int = 5) -> tuple[dict[str, np.ndarray], np.ndarray]:
    columns, ratio = gapped(seed)
    rng = np.random.default_rng(seed + 2)
    path = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(ratio))))
    columns["close"] = np.where(np.isfinite(columns["close"]), 0.1 * np.round(path / 0.1), np.nan)
    return columns, ratio


STOPS = 12 * np.arange(30 * 24, 40 * 24, 5)


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
    base = inputs.coin(columns, ratio, STOPS, M, B)
    assert all(np.isfinite(base[name]).mean() > 0.5 for name in inputs.OWN)
    rng = np.random.default_rng(7)
    for i in range(0, len(STOPS), 9):
        k = int(STOPS[i])
        changed = {name: values.copy() for name, values in columns.items()}
        later = np.arange(k, len(ratio))
        for name in ("close", "volume"):
            changed[name][later] *= np.exp(rng.normal(0, 0.2, later.size))
        changed["taker_buy_share"][later] = rng.uniform(0, 1, later.size)
        changed["close"][later[::7]] = np.nan
        stamped_after = later[1:]
        changed["oi"][stamped_after] *= np.exp(rng.normal(0, 0.2, stamped_after.size))
        changed["funding_rate"][stamped_after] += 0.01
        moved = ratio.copy()
        moved[stamped_after] *= 3
        got = inputs.coin(changed, moved, np.array([k]), M, B)
        for name in inputs.OWN:
            assert same(got[name][0], base[name][i]), (name, k)


def test_the_latest_data_allowed_is_read() -> None:
    columns, ratio = planted()
    k = int(STOPS[40])
    base = inputs.coin(columns, ratio, np.array([k]), M, B)
    checks = [
        ("close", k - 1, ["ret_1h", "vol_1h", "near_24h", "dd_7d", "var_tau_24h", "var_tau_7d"]),
        ("volume", k - 1, ["vsurge_1h", "volume_24h", "flow_1h"]),
        ("oi", k, ["oi_24h", "oi_7d", "oi_vs_28d"]),
        ("funding_rate", k, ["funding", "funding_rel"]),
    ]
    for column, at, names in checks:
        changed = {name: values.copy() for name, values in columns.items()}
        changed[column][at] *= 1.5
        got = inputs.coin(changed, ratio, np.array([k]), M, B)
        for name in names:
            assert got[name][0] != base[name][0], (column, name)
    moved = ratio.copy()
    moved[k] *= 1.5
    got = inputs.coin(columns, moved, np.array([k]), M, B)
    assert got["ls"][0] != base["ls"][0] and got["ls_rel"][0] != base["ls_rel"][0]


def test_vol_and_price_against_a_direct_calculation() -> None:
    columns, ratio = planted()
    k = 35 * DAY + 12 * 7
    got = {n: v[0] for n, v in inputs.coin(columns, ratio, np.array([k]), M, B).items()}
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
    got = {n: v[0] for n, v in inputs.coin(columns, ratio, np.array([k]), M, B).items()}
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
    got = {n: v[0] for n, v in inputs.coin(columns, ratio, np.array([k]), M, B).items()}
    assert got["oi_24h"] == pytest.approx(np.log(oi[k] / oi[k - 288]))
    assert got["oi_7d"] == pytest.approx(np.log(oi[k] / oi[k - 2016]))
    assert got["oi_vs_28d"] == pytest.approx(np.log(oi[k] / np.median(oi[k - 8063 : k + 1])))
    assert got["funding"] == funding[k]
    assert got["funding_rel"] == pytest.approx(funding[k] - np.mean(funding[k - 8063 : k + 1]))
    ls = np.log(ratio)
    assert got["ls"] == pytest.approx(ls[k])
    assert got["ls_rel"] == pytest.approx(ls[k] - np.mean(ls[k - 8063 : k + 1]))

    held = {name: values.copy() for name, values in columns.items()}
    held["oi"][k - 11 : k + 1] = np.nan
    gap = ratio.copy()
    gap[k - 11 : k + 1] = -1.0
    got = {n: v[0] for n, v in inputs.coin(held, gap, np.array([k]), M, B).items()}
    assert got["oi_24h"] == pytest.approx(np.log(oi[k - 12] / oi[k - 288]))
    assert got["ls"] == pytest.approx(ls[k - 12])
    held["oi"][k - 12] = 0.0
    gap[k - 12] = np.nan
    got = {n: v[0] for n, v in inputs.coin(held, gap, np.array([k]), M, B).items()}
    assert np.isnan(got["oi_24h"]) and np.isnan(got["oi_vs_28d"]) and np.isnan(got["ls"])


def test_a_rolling_value_needs_five_sixths_of_its_window() -> None:
    columns, ratio = planted()
    k = 36 * DAY
    columns["close"][k - 5] = np.nan
    got = inputs.coin(columns, ratio, np.array([k]), M, B)
    r = np.diff(np.log(columns["close"]))[k - 13 : k - 1]
    assert np.isfinite(r).sum() == 10
    assert got["vol_1h"][0] == pytest.approx(np.log(np.nanmean(r**2) / M))
    columns["close"][k - 9] = np.nan
    assert np.isnan(inputs.coin(columns, ratio, np.array([k]), M, B)["vol_1h"][0])
    short = {name: values[: 25 * DAY] for name, values in columns.items()}
    got = inputs.coin(short, ratio[: 25 * DAY], np.array([20 * DAY]), M, B)
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


SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIRST = date(2022, 11, 28)


@pytest.fixture(scope="module")
def labelled(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    fake = vision(SYMBOLS, FIRST, 34, seed=11)
    dest = tmp_path_factory.mktemp("archive")
    archive.fetch(SYMBOLS, FIRST, date(2022, 12, 31), dest, fake.get)
    data = tmp_path_factory.mktemp("data")
    run, _ = labels.run(dest, data, SYMBOLS, FIRST, date(2022, 12, 29), date(2022, 12, 31), "t")
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
    constants = pq.read_table(run / "constants.parquet").to_pylist()
    thresholds = pq.read_table(next((labels_dir / "thresholds").glob("*.parquet"))).to_pylist()
    assert [c["b"] for c in constants] == [t["b"] for t in thresholds]
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
    expected = inputs.coin(columns, ratio, stop, constants[1]["mean_r2"], constants[1]["b"])
    eth = table.filter(np.array(table["symbol"].to_pylist()) == "ETHUSDT")
    for name in inputs.OWN:
        assert np.array_equal(eth[name].to_numpy(), expected[name], equal_nan=True), name
    assert np.isfinite(eth["vol_28d"].to_numpy()).all()
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
