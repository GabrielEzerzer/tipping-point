"""The same-input tier (#30, research pass 6.4 item 2): a feature day aggregates to known minute
values, a minute with a gapped second and a day with no file have no value, the confirmatory span
is refused before any file is opened, a file of another feature format is refused, and the
template's [same_input] is the code's."""

import json
import tomllib
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tipping_point.book import features
from tipping_point.ews import market_segments as ms
from tipping_point.ews import prereg, same_input
from tipping_point.market import archive

REPO = Path(__file__).resolve().parents[4]
TEMPLATE = REPO / "paper" / "prereg" / "TEMPLATE-market-ews.toml"
AGG = prereg.SAME_INPUT_AGGREGATION
PARTIAL = 5 * 60 + 17  # a gapped second with some samples: its features are finite
EMPTY = 900 * 60  # a gapped second with none: NaN, as the reducer leaves it


def _day() -> pa.Table:
    """A synthetic feature day: ret 0.001 and ofi -2 every second, imb_5 the second of the minute
    (0..59), depth_5 the minute index; minutes 5 and 900 each hold a gapped second."""
    second = np.arange(features.DAY)
    gap = np.zeros(features.DAY, bool)
    gap[[PARTIAL, EMPTY]] = True
    cols = {
        "gap": gap,
        "ret": np.full(features.DAY, 0.001),
        "imb_5": (second % 60).astype(float),
        "ofi": np.full(features.DAY, -2.0),
        "depth_5": (second // 60).astype(float),
    }
    for name in AGG:
        cols[name][EMPTY] = np.nan
        cols[name] = cols[name].astype(np.float32)
    return pa.table(cols)


def _write(root: Path, symbol: str, day: date, fmt: int = features.FORMAT) -> None:
    rng = "exploratory" if day >= date(2023, 10, 1) else "training"
    path = root / rng / "features" / symbol / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"symbol": symbol, "day": day.isoformat(), "format": fmt}
    table = _day().replace_schema_metadata({"tp.book_features": json.dumps(meta)})
    pq.write_table(table, path)


def test_a_day_aggregates_to_known_minutes() -> None:
    """sum adds the minute's seconds, mean averages them, and a minute holding a gapped second
    has no value in any feature."""
    got = same_input.aggregate(_day(), AGG)
    assert set(got) == set(AGG)
    gapped = np.zeros(same_input.MINUTES, bool)
    gapped[[5, 900]] = True
    for name, minutes in got.items():
        assert minutes.shape == (1440,)
        assert np.isnan(minutes[gapped]).all(), name
    ok = ~gapped
    np.testing.assert_allclose(got["ret"][ok], 0.06, rtol=1e-6)  # 60 x 0.001
    np.testing.assert_allclose(got["ofi"][ok], -120.0)
    np.testing.assert_allclose(got["imb_5"][ok], 29.5)  # the mean of 0..59
    np.testing.assert_allclose(got["depth_5"][ok], np.arange(1440)[ok])
    with pytest.raises(ValueError, match="aggregation must be one of"):
        same_input.aggregate(_day(), {"ret": "median"})


def test_a_value_not_finite_outside_gap_seconds_is_refused() -> None:
    """As jepa.days refuses such a day, rather than silently emptying the minute."""
    table = _day()
    ofi = table["ofi"].to_numpy().copy()
    ofi[10 * 60 + 3] = np.inf  # minute 10 holds no gapped second
    table = table.set_column(table.schema.get_field_index("ofi"), "ofi", pa.array(ofi))
    with pytest.raises(ms.SegmentError, match="ofi is not finite outside gap seconds"):
        same_input.aggregate(table, AGG)


def test_a_value_not_finite_beside_a_gap_second_is_refused_too() -> None:
    """The check covers every non-gap second, not only minutes with no gap (the reviewer's case)."""
    table = _day()
    gap = table["gap"].to_numpy().copy()
    gap[10 * 60] = True  # minute 10 has a gapped second 0 ...
    ofi = table["ofi"].to_numpy().copy()
    ofi[10 * 60 + 3] = np.inf  # ... and a corrupt value in its second 3
    table = table.set_column(table.schema.get_field_index("gap"), "gap", pa.array(gap))
    table = table.set_column(table.schema.get_field_index("ofi"), "ofi", pa.array(ofi))
    with pytest.raises(ms.SegmentError, match="ofi is not finite outside gap seconds"):
        same_input.aggregate(table, AGG)


def test_a_gapped_minute_and_a_missing_day_have_no_value(tmp_path: Path) -> None:
    """Over three days with the middle one's file absent: that day has no value in any minute and
    is listed, and the gapped minutes of the others have none either; the segment builder then
    excludes and counts any segment holding them."""
    first = date(2023, 3, 1)
    _write(tmp_path, "ETHUSDT", first)
    _write(tmp_path, "ETHUSDT", date(2023, 3, 3))
    got = same_input.read(tmp_path, "ETHUSDT", first, date(2023, 3, 3), AGG)
    assert got.missing_days == ["2023-03-02"]
    assert got.format == features.FORMAT
    for name, series in got.series.items():
        assert series.start_ms == archive.day_ms(first)
        days = series.values.reshape(3, 1440)
        assert np.isnan(days[1]).all(), name
        for d in (0, 2):
            assert (
                np.isnan(days[d, [5, 900]]).all()
                and np.isfinite(np.delete(days[d], [5, 900])).all()
            )
    # a segment of the same span reading a gapped minute is the builder's to exclude
    _, kept, _, outside, invalid = ms._cut(
        got.series["ret"],
        archive.day_ms(first) + ms.MINUTE_MS * np.array([0, 10, 700]),
        299,
        np.zeros(0, bool),
        0,
    )
    assert kept.tolist() == [False, True, False] and (invalid, outside) == (2, 0)


def test_the_confirmatory_span_is_never_opened(tmp_path: Path) -> None:
    """A span reaching 2024-01-01 is refused before any file is read: the confirmatory folder,
    here a file that is not even Parquet, is never opened. 2023-12-31 reads the exploratory one."""
    bad = tmp_path / "confirmatory" / "features" / "BTCUSDT" / "2024-01-01.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not parquet")
    with pytest.raises(ms.SegmentError, match="confirmatory"):
        same_input.read(tmp_path, "BTCUSDT", date(2023, 12, 31), date(2024, 1, 1), AGG)
    _write(tmp_path, "BTCUSDT", date(2023, 12, 31))
    got = same_input.read(tmp_path, "BTCUSDT", date(2023, 12, 31), date(2023, 12, 31), AGG)
    assert got.missing_days == []


def test_another_feature_format_or_day_is_refused(tmp_path: Path) -> None:
    day = date(2023, 6, 10)
    _write(tmp_path, "BTCUSDT", day, fmt=features.FORMAT + 1)
    with pytest.raises(ms.SegmentError, match=f"this code reads {features.FORMAT}"):
        same_input.read(tmp_path, "BTCUSDT", day, day, AGG)
    _write(tmp_path, "BTCUSDT", day)
    folder = tmp_path / "training" / "features"
    (folder / "ETHUSDT").mkdir()
    (folder / "BTCUSDT" / f"{day}.parquet").rename(folder / "ETHUSDT" / f"{day}.parquet")
    with pytest.raises(ms.SegmentError, match="holds BTCUSDT 2023-06-10"):
        same_input.read(tmp_path, "ETHUSDT", day, day, AGG)


def test_the_template_freezes_what_the_code_runs() -> None:
    """The frozen text must match the code: the template's features and aggregation are the ones
    the harness reads (prereg.SAME_INPUT_AGGREGATION), which are the reducer's features, and its
    rules the texts the validator holds."""
    tier = tomllib.loads(TEMPLATE.read_text("utf-8"))["same_input"]
    assert tier["features"] == list(AGG)
    assert tier["aggregation"] == AGG
    assert set(AGG) <= set(features.FEATURES)
    assert set(AGG.values()) <= set(prereg.AGGREGATIONS)
    assert tier["missing"] == prereg.SAME_INPUT_MISSING
    assert tier["series"] == prereg.SAME_INPUT_SERIES
    assert tier["seasonality"] == prereg.SAME_INPUT_SEASONALITY
    assert tier["counted"] is False
