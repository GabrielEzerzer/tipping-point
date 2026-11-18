# The market JEPA's latent extraction (#30 M8) on a market toy training and toy days: the gate's
# refusals, the folders and days a read may open, units that merge bitwise into one read, resume
# after a kill, and the merge's checks. No real data.

import copy
import hashlib
import json
import re
import subprocess
import tomllib
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import market_toy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tipping_point.book.features import FEATURES, FORMAT
from tipping_point.ews import prereg
from tipping_point.ising import io
from tipping_point.jepa import days, minutes, train
from tipping_point.jepa import market_extract as mx

SHORT = market_toy.config(steps=12, checkpoint_every=4, monitor_every=4, pca_frames=128)
SYMBOL = market_toy.SYMBOL
TEMPLATE = Path(__file__).resolve().parents[4] / "paper" / "prereg" / "TEMPLATE-market-latent.toml"
ROW_KEYS = (
    "run_id",
    "data_commit",
    "trained_sha256",
    "twin_sha256",
    "pca_sha256",
    "scaling_sha256",
)
SPANS = {
    "fit": (mx.TRAIN_START, mx.TRAIN_START + timedelta(days=5)),
    "exploratory": (date(2023, 10, 1), date(2023, 10, 6)),
    "confirmatory": (date(2024, 1, 1), date(2024, 1, 4)),
}


@pytest.fixture(scope="module")
def primaries(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("primaries")
    run = train.start(SHORT, market_toy.dataset(), root / "btc-s101", "BTCUSDT-S-s101", "cpu")
    trained = train.load_market(run / train.FINAL)
    assert trained.pca is not None
    pin = {
        "asset": "btc",
        "symbol": SYMBOL,
        "seed": 101,
        "role": "primary",
        "run_id": run.name,
        "data_commit": "abc1234",
        "trained_sha256": train.sha256((run / train.FINAL).read_bytes()),
        "twin_sha256": train.sha256((run / train.TWIN).read_bytes()),
        "pca_sha256": mx.pca_digest(trained.pca),
        "scaling_sha256": mx.digest(io.read_json(run / "manifest.json")["data"]["scaling"]),
    }
    pins = root / "pins.json"
    pins.write_text(json.dumps({"format": 1, "size": "S", "steps": 12, "checkpoints": [pin]}))
    return root, pins


@pytest.fixture
def short(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mx, "SPANS", SPANS)
    monkeypatch.setattr(mx, "BLOCK", 2)


def _book(until: date = days.CONFIRMATORY) -> mx.Book:
    source = market_toy.Source()
    return mx.Book({name: source for name, _, _ in mx.FOLDERS}, until)


def _loaded(
    primaries: tuple[Path, Path],
) -> tuple[dict[str, Any], list[mx.Model], dict[str, dict[str, Any]]]:
    root, pins_path = primaries
    pins, _ = mx.read_pins(pins_path)
    return pins, mx.load_models(root, pins, pins["checkpoints"]), mx.scalings(root, pins, [SYMBOL])


def _shas(folder: Path) -> dict[str, str]:
    found = sorted(folder.rglob("*.parquet"))
    return {p.relative_to(folder).as_posix(): mx.file_sha256(p) for p in found}


def test_units_split_across_shards_merge_into_bitwise_what_one_read_gives(
    primaries: tuple[Path, Path], short: None, tmp_path: Path
) -> None:
    _, models, scaling = _loaded(primaries)
    book, out = _book(), tmp_path / "units"
    todo = mx.tasks("exploratory", [SYMBOL])
    assert [len(t.days) for t in todo] == [2, 2, 1]
    shards = [mx.shard(todo, f"{k}/2") for k in range(2)]
    assert sorted(t.first for s in shards for t in s) == [t.first for t in todo]
    for part in shards:
        mx.extract(book, models, scaling, part, out, None)
    series = tmp_path / "series"
    merged = mx.merge([out], series, "exploratory", primaries[1], None, symbols=[SYMBOL])
    assert len(merged) == 2 and not mx.verify(series) and not mx.verify(out)

    wanted = mx.span_days("exploratory")
    eve = wanted[0] - timedelta(days=1)
    lead = [eve - timedelta(days=d) for d in range(days.BASELINE_DAYS, 0, -1)]
    got, _ = days.frames(book, SYMBOL, [eve, *wanted], scaling[SYMBOL]["record"], lead=lead)
    keep = slice(days.MINUTES, None)
    for m in models:
        z, s = minutes.readouts(m.ckpt.model, got.frames, got.valid)
        one = mx.table(wanted[0], got.valid[keep].numpy(), mx.project(m.ckpt, z[keep], s[keep]))
        path = series / m.name / f"{SYMBOL}-exploratory.parquet"
        read = pq.read_table(path)
        assert read.column_names == one.column_names
        for name in one.column_names:
            assert read[name].to_numpy().tobytes() == one[name].to_numpy().tobytes(), name
        record = io.read_json(path.with_suffix(".json"))
        assert record["pre_registered"] is False and record["gate"] is None
        assert record["checkpoint"]["sha256"] == m.ckpt.sha256
        assert record["minutes"] == len(wanted) * days.MINUTES
        assert record["invalid_minutes"] == len(wanted) * len(market_toy.GAP_MINUTES)
        listed = [
            (d["day"], d["folder"], bool(d.get("lead")), bool(d.get("context")))
            for d in record["days"]
        ]
        assert listed[days.BASELINE_DAYS] == (eve.isoformat(), "training", False, True)
        assert [d for d, f, lead_day, ctx in listed if not lead_day and not ctx] == [
            d.isoformat() for d in wanted
        ]


def test_a_killed_run_keeps_its_units_and_redoes_only_what_it_lacks(
    primaries: tuple[Path, Path], short: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, models, scaling = _loaded(primaries)
    todo = mx.tasks("exploratory", [SYMBOL])
    whole = tmp_path / "whole"
    assert mx.extract(_book(), models, scaling, todo, whole, None) == 6
    out = tmp_path / "killed"
    real, calls = io.write_json, []

    def killed(path: Path, record: Any, replace: bool = False) -> None:
        calls.append(path)
        if len(calls) > 3:
            raise RuntimeError("killed")
        real(path, record, replace)

    with monkeypatch.context() as m:
        m.setattr(io, "write_json", killed)
        with pytest.raises(RuntimeError, match="killed"):
            mx.extract(_book(), models, scaling, todo, out, None)
    kept = sorted(out.rglob("*.json"))
    assert len(kept) == 3 and len(list(out.rglob("*.parquet"))) == 4
    broken = kept[0].with_suffix(".parquet")
    broken.write_bytes(broken.read_bytes() + b"x")
    stamps = {p: p.stat().st_mtime_ns for p in kept[1:]}
    assert mx.extract(_book(), models, scaling, todo, out, None) == 4
    assert {p: p.stat().st_mtime_ns for p in kept[1:]} == stamps
    assert _shas(out) == _shas(whole)
    assert mx.extract(_book(), models, scaling, todo, out, None) == 0


def test_a_unit_from_another_run_is_redone_not_skipped(
    primaries: tuple[Path, Path], short: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, models, scaling = _loaded(primaries)
    out = tmp_path / "units"
    assert mx.extract(_book(), models, scaling, mx.tasks("exploratory", [SYMBOL]), out, None) == 6
    shorter = mx.tasks("exploratory", [SYMBOL], limit=1)
    assert mx.extract(_book(), models, scaling, shorter, out, None) == 2
    record = io.read_json(next(out.rglob("2023-10-01.json")))
    assert record["first"] == record["last"] == "2023-10-01"
    assert mx.extract(_book(), models, scaling, shorter, out, None) == 0
    monkeypatch.setattr(io, "code_version", lambda: {"git_sha": "e" * 40, "dirty": False})
    assert mx.extract(_book(), models, scaling, shorter, out, None) == 2


def test_without_a_gate_no_confirmatory_day_is_opened(
    primaries: tuple[Path, Path], short: None, tmp_path: Path
) -> None:
    root, pins = primaries
    _, _, scaling = _loaded(primaries)
    argv = ["run", "--pins", str(pins), "--primaries", str(root), "--span", "confirmatory"]
    argv += ["--data", str(tmp_path / "book"), "--out", str(tmp_path / "out")]
    assert mx.main(argv) == 1 and not (tmp_path / "out").exists()
    book = mx.open_book(tmp_path / "book", days.CONFIRMATORY)
    assert set(book.ranges) == {"training", "exploratory"}
    with pytest.raises(days.DataError, match="confirmatory span"):
        book.read(SYMBOL, date(2024, 1, 1))
    first = mx.Task(SYMBOL, "confirmatory", (date(2024, 1, 1),))
    with pytest.raises(days.DataError, match="confirmatory span"):
        mx.read_task(_book(), first, scaling[SYMBOL]["record"])
    with pytest.raises(mx.Refused, match="only under a gate"):
        mx.merge([tmp_path], tmp_path / "series", "confirmatory", pins, None, symbols=[SYMBOL])


def _day_file(root: Path, day: date) -> str:
    n = days.SECONDS
    columns = {name: pa.array(np.zeros(n, np.float32)) for name in FEATURES}
    meta = {"format": FORMAT, "symbol": SYMBOL, "day": day.isoformat(), "features": list(FEATURES)}
    t = pa.table({**columns, days.GAP: pa.array(np.zeros(n, bool))})
    t = t.replace_schema_metadata({"tp.book_features": json.dumps(meta)})
    path = root / "features" / SYMBOL / f"{day.isoformat()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _range_manifest(root: Path, name: str, files: dict[date, str]) -> None:
    listed = {f"{SYMBOL}/{d.isoformat()}": {"reduced": {"sha256": s}} for d, s in files.items()}
    record = {"range": name, "assembled": {"commit": "abc", "feature_format": FORMAT}}
    (root / "manifest.json").write_text(json.dumps({**record, "days": listed}))


def test_each_folder_opens_only_its_own_days_and_the_files_its_manifest_names(
    tmp_path: Path,
) -> None:
    late, first, end = date(2023, 12, 31), date(2024, 1, 1), mx.END
    exploratory, confirmatory = tmp_path / "exploratory", tmp_path / "confirmatory"
    _range_manifest(exploratory, "exploratory", {late: _day_file(exploratory, late)})
    shown = mx.Range(exploratory, "exploratory", mx.EXPLORATORY, days.CONFIRMATORY)
    assert shown.read(SYMBOL, late)[1] == mx.file_sha256(
        exploratory / "features" / SYMBOL / f"{late}.parquet"
    )
    with pytest.raises(days.DataError, match="confirmatory span"):
        shown.read(SYMBOL, first)
    closed = mx.Range(confirmatory, "confirmatory", days.CONFIRMATORY, end)
    with pytest.raises(days.DataError, match="before"):
        closed.read(SYMBOL, late)
    with pytest.raises(days.DataError, match="on or after 2026-10-01"):
        closed.read(SYMBOL, end)
    assert not confirmatory.exists()

    book = mx.Book({}, end)
    assert [book.folder(d) for d in (date(2023, 9, 30), late, first)] == [
        "training",
        "exploratory",
        "confirmatory",
    ]
    with pytest.raises(days.DataError, match="on or after 2026-10-01"):
        book.folder(end)
    with pytest.raises(days.DataError, match="outside the book data"):
        book.folder(date(2023, 1, 17))
    with pytest.raises(days.DataError, match="on or after 2026-10-01"):
        days.refuse(end, end)
    with pytest.raises(days.DataError, match="on or after 2023-10-01"):
        days.refuse(date(2023, 10, 5), mx.EXPLORATORY)

    other = date(2023, 12, 30)
    _day_file(exploratory, other)
    with pytest.raises(days.DataError, match="its manifest says None"):
        shown.read(SYMBOL, other)
    _range_manifest(exploratory, "confirmatory", {late: "0" * 64})
    with pytest.raises(days.DataError, match="'confirmatory' range, not exploratory"):
        mx.Range(exploratory, "exploratory", mx.EXPLORATORY, days.CONFIRMATORY).read(SYMBOL, late)


def test_a_checkpoint_basis_or_scaling_not_pinned_is_refused(primaries: tuple[Path, Path]) -> None:
    root, pins_path = primaries
    pins, _ = mx.read_pins(pins_path)
    for key in ("trained_sha256", "twin_sha256", "pca_sha256"):
        bad = copy.deepcopy(pins)
        bad["checkpoints"][0][key] = "0" * 64
        with pytest.raises(mx.Refused, match=r"sha256|PCA"):
            mx.load_models(root, bad, bad["checkpoints"])
    bad = copy.deepcopy(pins)
    bad["checkpoints"][0]["scaling_sha256"] = "0" * 64
    with pytest.raises(mx.Refused, match="scaling record"):
        mx.scalings(root, bad, [SYMBOL])
    with pytest.raises(mx.Refused, match="0 pinned scaling records"):
        mx.scalings(root, pins, ["ETHUSDT"])
    with pytest.raises(mx.Refused, match="not in pins"):
        mx.select(pins, ["no-such-run"])


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _filled(pin: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    text = TEMPLATE.read_text(encoding="utf-8")
    pins = []
    for i, row in enumerate(tomllib.loads(text)["latents"]["checkpoints"]):
        mine = all(row[k] == pin[k] for k in ("symbol", "seed", "role"))
        same = row["symbol"] == pin["symbol"]
        made = (
            pin
            if mine
            else {
                **{k: row[k] for k in ("symbol", "seed", "role")},
                "run_id": f"run-{i}",
                "data_commit": "abc1234",
                **{f"{k}_sha256": _sha(f"{k}{i}") for k in ("trained", "twin", "pca")},
                "scaling_sha256": pin["scaling_sha256"] if same else _sha(row["symbol"]),
            }
        )
        pins.append(made)
        for key in ROW_KEYS:
            text = re.sub(
                rf'^{key} = "<[^"]*>"', f'{key} = "{made[key]}"', text, count=1, flags=re.M
            )
    text = text.replace('id = "<lowercase-slug>"', 'id = "market-latent"', 1)
    for key, value in (
        ("sha256", f'"{"0" * 64}"'),
        ("judged", '"same-asset"'),
        ("against", '"each"'),
        ("tolerance", "0.02"),
    ):
        text = re.sub(rf'^{key} = "<[^"]*>"', f"{key} = {value}", text, flags=re.M)
    return re.sub(r'"<[^"]*>"', '"set"', text), pins


@pytest.fixture
def repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, primaries: tuple[Path, Path]
) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    src = root / "src" / "tipping_point"
    for rel in (*mx.HELD, *mx.CLEAN):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(f"{rel}\n")
    (root / "uv.lock").write_text("lock\n")
    toy, _ = mx.read_pins(primaries[1])
    text, rows = _filled(toy["checkpoints"][0])
    path = root / "paper" / "prereg" / "market-latent-v1.toml"
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode())
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({**toy, "checkpoints": rows}))
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "market-latent-v1")
    monkeypatch.setattr(mx, "SRC", src)
    monkeypatch.setattr(
        io,
        "code_version",
        lambda: {
            "git_sha": _git(root, "rev-parse", "HEAD"),
            "dirty": _git(root, "status", "--porcelain") != "",
        },
    )
    return root, path, pins


def _freeze(root: Path) -> None:
    _git(root, "tag", "-a", "prereg/market-latent-v1", "-m", "freeze")


def test_the_gate_refuses_an_unfrozen_or_edited_file_changed_code_or_a_dirty_tree(
    repo: tuple[Path, Path, Path],
) -> None:
    root, path, pins = repo
    with pytest.raises(prereg.PreregError, match="not frozen"):
        mx.gate(path, pins)
    _freeze(root)
    record = mx.gate(path, pins)
    assert record["prereg"]["tag"] == "prereg/market-latent-v1"
    rows = tomllib.loads(path.read_text(encoding="utf-8"))["latents"]["checkpoints"]
    assert record["checkpoints"] == sorted(rows, key=lambda r: r["run_id"])
    assert record["checkpoints"] == mx.pinned(mx.read_pins(pins)[0], sorted(rows[0]))
    assert record["prereg"]["path"] == "paper/prereg/market-latent-v1.toml"

    elsewhere = root / "paper" / "market-latent-v1.toml"
    elsewhere.write_bytes(path.read_bytes())
    with pytest.raises(prereg.PreregError, match="under its own name"):
        mx.gate(elsewhere, pins)
    elsewhere.unlink()

    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(prereg.PreregError, match="differs from the version frozen"):
        mx.gate(path, pins)
    _git(root, "checkout", "--", str(path))

    clean = root / "src" / "tipping_point" / mx.CLEAN[0]
    clean.write_text("edited\n")
    with pytest.raises(mx.Refused, match="not committed"):
        mx.gate(path, pins)
    _git(root, "checkout", "--", str(clean))

    (root / "notes.txt").write_text("untracked\n")
    with pytest.raises(mx.Refused, match="not clean"):
        mx.gate(path, pins)
    (root / "notes.txt").unlink()

    held = root / "src" / "tipping_point" / mx.HELD[0]
    held.write_text("changed after the freeze\n")
    _git(root, "commit", "-q", "-am", "change held code")
    with pytest.raises(mx.Refused, match="differ from prereg/market-latent-v1"):
        mx.gate(path, pins)


def test_the_gate_refuses_checkpoints_that_are_not_the_pins(
    repo: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    root, path, pins_path = repo
    _freeze(root)
    pins, _ = mx.read_pins(pins_path)
    for key, value in (
        ("twin_sha256", "1" * 64),
        ("data_commit", "def5678"),
        ("seed", 999),
        ("role", "spare"),
    ):
        other = copy.deepcopy(pins)
        other["checkpoints"][0][key] = value
        forged = tmp_path / f"pins-{key}.json"
        forged.write_text(json.dumps(other))
        with pytest.raises(mx.Refused, match="differ: only in market-latent-v1"):
            mx.gate(path, forged)
    other = copy.deepcopy(pins)
    del other["checkpoints"][-1]
    forged.write_text(json.dumps(other))
    with pytest.raises(mx.Refused, match="differ: only in market-latent-v1"):
        mx.gate(path, forged)


def test_the_gate_refuses_a_file_the_strict_check_refuses(repo: tuple[Path, Path, Path]) -> None:
    root, path, pins = repo
    text = path.read_text(encoding="utf-8").replace("tolerance = 0.02", 'tolerance = "TBD"', 1)
    path.write_bytes(text.encode())
    _git(root, "commit", "-q", "-am", "leave a placeholder")
    _freeze(root)
    with pytest.raises(prereg.PreregError, match="cannot gate a run"):
        mx.gate(path, pins)


def test_a_gated_run_reads_the_confirmatory_span_and_merges_under_its_gate(
    repo: tuple[Path, Path, Path],
    primaries: tuple[Path, Path],
    short: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, path, pins_path = repo
    _freeze(root)
    record = mx.gate(path, pins_path)
    _, models, scaling = _loaded(primaries)
    pins, pins_sha = mx.read_pins(pins_path)
    run = [mx.read_pins(primaries[1])[0]["checkpoints"][0]["run_id"]]
    mx.admit(record, pins, pins_sha)
    for broken, why in (
        ({**record, "pins_sha256": "0" * 64}, "pins.json is not the one"),
        ({**record, "prereg": {**record["prereg"], "text": "edited"}}, "not the one its stamp"),
        ({**record, "format": 0}, "not a gate record"),
        ({**record, "prereg": {**record["prereg"], "path": "market-latent-v1.toml"}}, "not at"),
    ):
        with pytest.raises(mx.Refused, match=why):
            mx.admit(broken, pins, pins_sha)
    assert "made again" in mx.admitted(record, pins_path)

    text = record["prereg"]["text"] + "# not the frozen file\n"
    forged_stamp = {
        **record["prereg"],
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    forged = {**record, "prereg": forged_stamp}
    mx.admit(forged, pins, pins_sha)
    with pytest.raises(mx.Refused, match="the gate made now is not the gate record's"):
        mx.admitted(forged, pins_path)

    def no_repository(path: Path) -> Path:
        raise prereg.PreregError("not a git repository")

    with monkeypatch.context() as m:
        m.setattr(prereg, "repo_root", no_repository)
        assert "no repository" in mx.admitted(record, pins_path)

    todo = mx.tasks("confirmatory", [SYMBOL], doc=mx.frozen_doc(record))
    out = tmp_path / "units"
    ungated = tmp_path / "ungated"
    assert mx.extract(_book(mx.END), models, scaling, todo, out, record) == 4
    assert mx.extract(_book(mx.END), models, scaling, todo, ungated, None) == 4
    series = tmp_path / "series"
    with pytest.raises(mx.Refused, match="made under another gate"):
        mx.merge([ungated], series, "confirmatory", pins_path, record, run, [SYMBOL])
    merged = mx.merge([out], series, "confirmatory", pins_path, record, run, [SYMBOL])
    written = io.read_json(merged[0].with_suffix(".json"))
    assert written["pre_registered"] is True
    assert written["gate"]["tag"] == "prereg/market-latent-v1"
    assert written["days"][days.BASELINE_DAYS]["day"] == "2023-12-31"
    assert written["days"][days.BASELINE_DAYS]["folder"] == "exploratory"
    assert {d["folder"] for d in written["days"][days.BASELINE_DAYS + 1 :]} == {"confirmatory"}
    pre = mx.extent("confirmatory", mx.frozen_doc(record))[2]
    assert pre == 514 and written["pre_minutes"] == pre
    assert written["days"][days.BASELINE_DAYS]["kept_minutes"] == pre

    eve, wanted = date(2023, 12, 31), mx.span_days("confirmatory", doc=mx.frozen_doc(record))
    lead = [eve - timedelta(days=d) for d in range(days.BASELINE_DAYS, 0, -1)]
    record_ = scaling[SYMBOL]["record"]
    got, _ = days.frames(_book(mx.END), SYMBOL, [eve, *wanted], record_, lead=lead, until=mx.END)
    keep = slice(days.MINUTES - pre, None)
    m = next(m for m in models if m.name == merged[0].parent.name)
    z, s = minutes.readouts(m.ckpt.model, got.frames, got.valid)
    one = mx.table(wanted[0], got.valid[keep].numpy(), mx.project(m.ckpt, z[keep], s[keep]), pre)
    read = pq.read_table(merged[0])
    start = np.datetime64("2024-01-01", "ns").astype(np.int64) - pre * mx.MINUTE_NS
    assert read["ts_ns"][0].as_py() == start and len(read) == len(wanted) * days.MINUTES + pre
    for name in one.column_names:
        assert read[name].to_numpy().tobytes() == one[name].to_numpy().tobytes(), name

    _git(root, "tag", "-d", "prereg/market-latent-v1")
    _git(root, "tag", "-a", "prereg/market-latent-v1", "-m", "frozen again")
    with pytest.raises(mx.Refused, match="the gate made now is not the gate record's"):
        mx.admitted(record, pins_path)
    _git(root, "tag", "-d", "prereg/market-latent-v1")
    _freeze(root)

    later = root / "src" / "tipping_point" / mx.HELD[0]
    later.write_text("changed after the units were made\n")
    _git(root, "commit", "-q", "-am", "change held code")
    with pytest.raises(mx.Refused, match="differ from prereg/market-latent-v1"):
        mx.merge([out], series, "confirmatory", pins_path, record, run, [SYMBOL])

    monkeypatch.setattr(io, "code_version", lambda: {"git_sha": "f" * 40, "dirty": False})
    with pytest.raises(mx.Refused, match="the gate record's"):
        mx.admit(record, pins, pins_sha)


def _amend(root: Path, path: Path, *edits: tuple[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    path.write_bytes(text.encode())
    _git(root, "commit", "-q", "-am", "amend the extent")


FIT = "fit = { start = 2023-01-18, end = 2023-09-30 }"


def test_the_fit_span_and_the_tail_are_the_frozen_file_s(
    repo: tuple[Path, Path, Path], primaries: tuple[Path, Path], short: None, tmp_path: Path
) -> None:
    root, path, pins_path = repo
    end = mx.TRAIN_START + timedelta(days=4)
    _amend(
        root,
        path,
        (FIT, f"fit = {{ start = {mx.TRAIN_START}, end = {end} }}"),
        ("\nwindow = 60\n", "\nwindow = 120\n"),
    )
    _freeze(root)
    record = mx.gate(path, pins_path)
    doc = mx.frozen_doc(record)
    assert mx.extent("fit", doc) == (mx.TRAIN_START, end + timedelta(days=1), 0)
    assert mx.extent("confirmatory", doc)[2] == 5 * 24 + 120 + 240 - 1 + 96 - 1 == 574
    with pytest.raises(mx.Refused, match="only under a gate"):
        mx.tasks("fit", [SYMBOL])
    _, models, scaling = _loaded(primaries)
    run = [mx.read_pins(primaries[1])[0]["checkpoints"][0]["run_id"]]
    book, out = _book(), tmp_path / "units"
    todo = mx.tasks("fit", [SYMBOL], doc=doc)
    assert [len(t.days) for t in todo] == [2, 2, 1] and todo[-1].days[-1] == end
    assert todo[0].first == mx.TRAIN_START and not todo[0].eve and not todo[0].lead
    assert todo[1].eve == [todo[0].days[-1]] and len(todo[1].lead) == 1
    assert mx.extract(book, models, scaling, todo, out, record) == 6
    series = tmp_path / "series"
    with pytest.raises(mx.Refused, match="only under a gate"):
        mx.merge([out], series, "fit", pins_path, None, run, [SYMBOL])
    merged = mx.merge([out], series, "fit", pins_path, record, run, [SYMBOL])
    assert len(merged) == 2 and not mx.verify(series)

    wanted = mx.span_days("fit", doc=doc)
    got, _ = days.frames(book, SYMBOL, wanted, scaling[SYMBOL]["record"], lead=[])
    for m in models:
        z, s = minutes.readouts(m.ckpt.model, got.frames, got.valid)
        one = mx.table(wanted[0], got.valid.numpy(), mx.project(m.ckpt, z, s))
        read = pq.read_table(series / m.name / f"{SYMBOL}-fit.parquet")
        assert len(read) == 5 * days.MINUTES
        for name in one.column_names:
            assert read[name].to_numpy().tobytes() == one[name].to_numpy().tobytes(), name
        assert not read["valid"].to_numpy()[: SHORT.model.context - 1].any()
        written = io.read_json(series / m.name / f"{SYMBOL}-fit.json")
        assert written["days"][0]["day"] == mx.TRAIN_START.isoformat()
        assert not any(d.get("lead") or d.get("context") for d in written["days"][:1])

    first = mx.tasks("confirmatory", [SYMBOL], doc=doc)[:1]
    assert first[0].pre == 574
    tail = tmp_path / "tail"
    assert mx.extract(_book(mx.END), models, scaling, first, tail, record) == 2
    unit = next(tail.rglob("2024-01-01.json"))
    assert io.read_json(unit)["days"][days.BASELINE_DAYS]["kept_minutes"] == 574
    table = pq.read_table(unit.with_suffix(".parquet"))
    start = np.datetime64("2024-01-01", "ns").astype(np.int64) - 574 * mx.MINUTE_NS
    assert len(table) == len(first[0].days) * days.MINUTES + 574
    assert table["ts_ns"][0].as_py() == start


@pytest.mark.parametrize(
    ("edit", "why"),
    [
        ((FIT, "fit = { start = 2023-02-01, end = 2023-09-30 }"), "the book's first day"),
        (("\nwindow = 60\n", "\nwindow = 1200\n"), "one day before its span"),
    ],
)
def test_the_gate_refuses_an_extent_a_read_cannot_take(
    repo: tuple[Path, Path, Path], edit: tuple[str, str], why: str
) -> None:
    root, path, pins = repo
    _amend(root, path, edit)
    _freeze(root)
    with pytest.raises(mx.Refused, match=why):
        mx.gate(path, pins)


def test_the_merge_refuses_a_missing_tampered_or_foreign_unit(
    primaries: tuple[Path, Path], short: None, tmp_path: Path
) -> None:
    _, models, scaling = _loaded(primaries)
    out = tmp_path / "units"
    mx.extract(_book(), models, scaling, mx.tasks("exploratory", [SYMBOL]), out, None)
    series = tmp_path / "series"
    units = sorted(out.rglob("*.json"))
    target = units[0].with_suffix(".parquet")
    saved = target.read_bytes()
    target.write_bytes(saved + b"x")
    with pytest.raises(mx.Refused, match="its parquet is not its record's"):
        mx.merge([out], series, "exploratory", primaries[1], None, symbols=[SYMBOL])
    assert mx.verify(out) == [f"{target}: not the file its record names"]
    target.write_bytes(saved)

    record = io.read_json(units[1])
    record["checkpoint"]["sha256"] = "2" * 64
    units[1].write_text(json.dumps(record))
    with pytest.raises(mx.Refused, match="is not pinned"):
        mx.merge([out], series, "exploratory", primaries[1], None, symbols=[SYMBOL])
    units[1].unlink()
    with pytest.raises(mx.Refused, match="no unit"):
        mx.merge([out], series, "exploratory", primaries[1], None, symbols=[SYMBOL])
    assert not series.exists()
