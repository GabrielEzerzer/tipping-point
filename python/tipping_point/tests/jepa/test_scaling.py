"""The cross-day step of the market loader (#165; #30 research pass 2.2 and 4.3 test 2): the scaling
record is fitted on the training days alone, a feature that is mostly zero still gets a finite
scale, relative depth reads only earlier days, readouts reuse the frozen record, and the Parquet
source reads the files the feature reducer writes."""

import hashlib
import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tipping_point.book import features
from tipping_point.jepa import days
from tipping_point.jepa.days import DataError

SYMBOL = "BTCUSDT"
SPAN = days.Span(SYMBOL, date(2023, 1, 18), date(2023, 10, 1))
FEATURES = features.FEATURES
TRAIN = [date(2023, 3, 1) + timedelta(days=k) for k in range(6)]
VAL = [date(2023, 2, 27), date(2023, 3, 8)]  # one before the training days, one after
QUIET = 0.9  # the share of seconds with no mid move in the synthetic days


def book_day(day: date) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Synthetic book features under the reducer's names: ret, rv100 and moves are 0 in 90 % of
    seconds, wide in 98 %, depth is a log notional that drifts from day to day, slopes are positive,
    and a few seconds are gaps holding NaN."""
    rng = np.random.default_rng(day.toordinal())
    n = days.SECONDS
    moved = rng.random(n) >= QUIET
    ret = np.where(moved, rng.standard_t(3, n) * 2e-4, 0.0)
    level = 15.0 + 0.01 * (day - date(2023, 1, 1)).days  # a slow growth in notional
    cols = {
        "ret": ret,
        "rv100": np.where(moved, ret**2 * (1 + rng.random(n)), 0.0),
        "moves": np.where(moved, rng.integers(1, 4, n), 0).astype(float),
        "micro_dev": rng.uniform(-0.5, 0.5, n),
        "imb_1": rng.uniform(-1, 1, n),
        "imb_5": rng.uniform(-1, 1, n),
        "imb_20": rng.uniform(-1, 1, n),
        "ofi": rng.standard_t(3, n),
        "depth_5": level - 1.5 + 0.2 * rng.standard_normal(n),
        "depth_20": level + 0.2 * rng.standard_normal(n),
        "depth_asym": 0.3 * rng.standard_normal(n),
        "slope_b": np.exp(rng.normal(0.5, 0.6, n)),
        "slope_a": np.exp(rng.normal(0.5, 0.6, n)),
        "wide": np.where(rng.random(n) < 0.02, rng.uniform(0.1, 1, n), 0.0),
    }
    gap = np.zeros(n, bool)
    gap[[500, 40_000]] = True
    values = {k: v.astype(np.float32) for k, v in cols.items()}
    for v in values.values():
        v[gap] = np.nan
    return values, gap


class Source:
    """A days.DaySource of synthetic book days; `edit` changes the named days' values."""

    def __init__(self, edit: dict[date, Callable[[dict[str, np.ndarray]], None]] | None = None):
        self.edit = edit or {}

    def read(self, symbol: str, day: date) -> tuple[pa.Table, str]:
        values, gap = book_day(day)
        if day in self.edit:
            self.edit[day](values)
        table = pa.table({**{k: pa.array(values[k]) for k in FEATURES}, days.GAP: pa.array(gap)})
        digest = hashlib.sha256(b"".join(v.tobytes() for v in values.values())).hexdigest()
        return table, digest


def _load(source: Source, train: list[date] = TRAIN, val: list[date] = VAL) -> days.Dataset:
    return days.load(source, SPAN, SYMBOL, FEATURES, train, val)


@pytest.fixture(scope="module")
def plain() -> days.Dataset:
    return _load(Source())


def _louder(values: dict[str, np.ndarray]) -> None:
    for name in ("ret", "depth_20", "slope_a"):
        values[name] *= 1.5


def test_the_record_is_fitted_on_the_training_days_alone(plain: days.Dataset) -> None:
    """Leakage test 2: changing one training day changes the record; changing a validation day,
    even one inside the 7-day depth window before a training day, leaves it identical, and the
    record names the training span and days it was fitted on."""
    record = plain.sources["scaling"]
    assert record["features"] == list(FEATURES)
    assert record["span"] == ["2023-01-18", "2023-10-01"]
    assert record["days"] == [d.isoformat() for d in TRAIN]
    assert json.loads(json.dumps(record)) == record  # the manifest and resume see the same record
    for val_day in VAL:
        assert _load(Source({val_day: _louder})).sources["scaling"] == record
    changed = _load(Source({TRAIN[3]: _louder})).sources["scaling"]
    assert changed["days"] == record["days"]
    for name in ("ret", "depth_20", "slope_a"):
        k = FEATURES.index(name)
        assert changed["centre"][k] != record["centre"][k] or (
            changed["scale"][k] != record["scale"][k]
        ), name


def test_a_mostly_zero_feature_gets_a_finite_unit_scale(plain: days.Dataset) -> None:
    """ret, rv100 and moves are 0 in 90 % of seconds and wide in 98 %: their IQR is 0, the scale
    is finite and non-zero, and every transformed feature has a spread near one. A feature with
    one value in every training second is refused rather than divided by zero."""
    values = np.concatenate([book_day(d)[0]["rv100"] for d in TRAIN])
    assert np.nanpercentile(values, 25) == np.nanpercentile(values, 75) == 0  # the IQR divides by 0
    record = plain.sources["scaling"]
    assert all(np.isfinite(s) and s > 0 for s in record["scale"])
    frames, valid = plain.train.frames, plain.train.valid
    x = frames[valid].reshape(-1, len(FEATURES)).double()
    assert torch.isfinite(x).all()
    spread = x.std(0)
    for name in ("ret", "rv100", "moves", "wide"):
        k = FEATURES.index(name)
        assert (x[:, k] == 0).double().mean() > 0.85, name  # the zeros stay at the centre
        assert 0.05 < spread[k] < 3, (name, float(spread[k]))
    others = [k for k, n in enumerate(FEATURES) if n not in ("ret", "rv100", "moves", "wide")]
    assert ((spread[others] > 0.5) & (spread[others] < 2)).all(), spread

    def flat(values: dict[str, np.ndarray]) -> None:
        values["wide"][:] = 0.0

    with pytest.raises(DataError, match=r"wide is 0.0 in every valid training second"):
        _load(Source(dict.fromkeys(TRAIN, flat)))


def test_relative_depth_reads_only_earlier_days(plain: days.Dataset) -> None:
    """The first training day has no earlier day and trains on nothing; a later day's depth is
    relative to earlier days only: shifting a day's depth by 1 shifts its own relative depth by
    1 / scale (its own median is not in its baseline) and leaves every earlier day unchanged."""
    per_day = days.MINUTES
    assert not plain.train.valid[:per_day].any()  # 2023-03-01: a baseline for later days only
    assert plain.train.valid[per_day:].any()
    assert plain.sources["train"][0]["valid_minutes"] == 0

    def shifted(values: dict[str, np.ndarray]) -> None:
        values["depth_20"] += 1.0

    k = FEATURES.index("depth_20")

    def relative(data: days.Dataset, d: int) -> torch.Tensor:
        """Day d's relative depth_20 in log units, over its valid minutes."""
        record = data.sources["scaling"]
        span = slice(d * per_day, (d + 1) * per_day)
        x = data.train.frames[span][data.train.valid[span]][..., k].double()
        return x * record["scale"][k] + record["centre"][k]

    base = _load(Source(), val=[VAL[0]])
    moved = _load(Source({TRAIN[4]: shifted}), val=[VAL[0]])
    for d in range(1, 4):
        assert torch.allclose(relative(moved, d), relative(base, d), atol=1e-4), d
    diff = relative(moved, 4) - relative(base, 4)
    assert torch.allclose(diff, torch.ones_like(diff), atol=1e-4)


def test_readouts_reuse_the_frozen_record(plain: days.Dataset) -> None:
    """`frames` applies a training's record and never refits: the training days read through it
    are bitwise the training frames, a changed record changes them, and a record made under other
    transforms or a confirmatory day is refused."""
    record = plain.sources["scaling"]
    m, _ = days.frames(Source(), SYMBOL, TRAIN, record)
    assert torch.equal(m.frames, plain.train.frames) and torch.equal(m.valid, plain.train.valid)
    other = {**record, "centre": [c + 1.0 for c in record["centre"]]}
    assert not torch.equal(days.frames(Source(), SYMBOL, TRAIN, other)[0].frames, m.frames)
    with pytest.raises(DataError, match="transforms"):
        days.frames(Source(), SYMBOL, TRAIN, {**record, "transforms": ["none"] * len(FEATURES)})
    with pytest.raises(DataError, match="confirmatory"):
        days.frames(Source(), SYMBOL, [date(2024, 1, 1)], record)


def _write_day(root: Path, symbol: str, day: date, meta_day: date | None = None) -> Path:
    """A feature file as `reduce_run` writes it: the reducer's own schema (an empty day reduced),
    synthetic values, its metadata and write options."""
    start = int((day - date(1970, 1, 1)).total_seconds()) * features.SECOND_NS
    table = features.reduce([], start, 2, 3)
    values, gap = book_day(day)
    for name in FEATURES:
        table = table.set_column(table.schema.get_field_index(name), name, pa.array(values[name]))
    table = table.set_column(table.schema.get_field_index("gap"), "gap", pa.array(gap))
    meta = {"symbol": symbol, "day": (meta_day or day).isoformat(), "features": list(FEATURES)}
    table = table.replace_schema_metadata({"tp.book_features": json.dumps(meta)})
    path = root / "features" / symbol / f"{day.isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, **features._WRITE)
    return path


def test_the_parquet_source_reads_the_reducers_files(tmp_path: Path) -> None:
    """Reads `<root>/features/<SYMBOL>/<day>.parquet` with the sha256 of its bytes (as
    `reduce_run` records it), feeds `load`, and refuses a missing file, a file holding another
    day and any confirmatory day before touching the disk."""
    for day in [*TRAIN[:3], VAL[1]]:
        _write_day(tmp_path, SYMBOL, day)
    source = days.ParquetSource(tmp_path)
    path = tmp_path / "features" / SYMBOL / f"{TRAIN[0].isoformat()}.parquet"
    table, digest = source.read(SYMBOL, TRAIN[0])
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert table.num_rows == days.SECONDS and table["gap"].type == pa.bool_()
    data = days.load(source, SPAN, SYMBOL, FEATURES, TRAIN[:3], [VAL[1]])
    assert data.sources["train"][0]["sha256"] == digest
    assert data.train.frames.shape == (3 * days.MINUTES, days.MINUTE, len(FEATURES))
    assert torch.isfinite(data.train.frames).all()
    with pytest.raises(DataError, match="no feature file"):
        source.read(SYMBOL, TRAIN[4])
    with pytest.raises(DataError, match="no feature file"):
        source.read("ETHUSDT", TRAIN[0])
    _write_day(tmp_path, SYMBOL, TRAIN[5], meta_day=TRAIN[4])
    with pytest.raises(DataError, match="holds BTCUSDT 2023-03-05"):
        source.read(SYMBOL, TRAIN[5])
    with pytest.raises(DataError, match="confirmatory"):
        days.ParquetSource(tmp_path / "nowhere").read(SYMBOL, date(2024, 1, 2))
