# Tipping Point

Open-source framework for detecting critical transitions in live data streams: a JEPA world model measures, Jev triages, and tool-calling agents investigate.


## Layout

- `engine/` — Rust core, a member of the root Cargo workspace (`Cargo.toml`).
- `python/tipping_point/` — the Python package (src layout), a member of the root uv workspace (`pyproject.toml`).
- `app/` — the Tauri desktop app (not yet started).
- `paper/` — the paper.
- `harness/` — supporting code that verifies the rest: the gate and its tests, plus `engine-harness/`,
  a Cargo workspace member holding the engine's recorder, soak and crash-injection binaries and the
  replay-equals-live checks (production code never depends on it).

## Binance ingest

`engine/src/binance/` keeps a correct local order book for BTCUSDT and ETHUSDT from the USD-M
depth stream (snapshot plus diffs, sequence-gap detection, resync, backpressure) and reads aggTrades.
The sync logic is a pure state machine (`sync.rs`) behind a multi-symbol `feed.rs`, and
`pipeline.rs` holds the one copy of how the engine reacts to every input; the network layer
(`ws.rs`, `rest.rs`, `live.rs`) is thin around it.

```sh
cargo run -p engine --bin ingest -- --seconds 60            # live: logs top-of-book and resyncs
cargo run -p engine --bin ingest -- --record                # ...and records a run to the data dir (Ctrl-C ends it cleanly)
cargo run -p engine-harness --bin record -- --seconds 60    # a run + the live transcript → harness/fixtures/recorded/ (gitignored)
cargo run -p engine-harness --bin soak -- --hours 24        # gaps/resyncs/reconnects/drops per hour → harness/out/
```

## Market data on disk

`engine/src/store/` is the storage. The live loop logs every input right before applying it —
depth and trade frames as received, snapshots, and control events (connects, disconnects, frames
dropped under backpressure, failed snapshots) — to an append-only binary log: 36-byte header, raw
payload, CRC-32C, one `write_all` per frame, `sync_data` every second, a new segment every UTC hour
or 1 GiB. A crash can only tear the tail; recovery cuts it back to the last good frame and refuses
damage in the middle. An offline step rotates closed segments into Parquet (`frames`,
`depth_updates`, `agg_trades`, `snapshots`, `book_samples`), and DuckDB queries and replays it
through the same pipeline the live loop runs.

Runs live in `<data dir>/market/<run_id>/` (`manifest.json`, `log/`, one Parquet file per table per
segment, `complete.json` after a clean end), the layout the Ising simulator shares. The data dir is
`$TP_DATA_DIR`, or else `%LOCALAPPDATA%\tipping-point` (`~/.local/share/tipping-point` elsewhere),
never the repository.

```sh
cargo run -p engine --bin store -- recover <run dir>              # after a crash: close the newest segment
cargo run -p engine --bin store -- rotate <run dir>               # closed segments → Parquet (safe to re-run)
cargo run -p engine --bin store -- replay <run dir> [--parquet]   # rebuild the books from the log or the Parquet
```

DuckDB is linked from its prebuilt library, which the build downloads once per target dir
(`DUCKDB_DOWNLOAD_LIB` in `.cargo/config.toml`), so the first build needs the network. On
Windows the build also copies `duckdb.dll` next to the binaries (`target/<profile>/`,
`engine/build.rs`), so `store.exe` and `serve.exe` start from any shell, not only under cargo;
a binary copied elsewhere needs that DLL beside it. Linux and macOS binaries find the library
through their rpath.

Recorded fixtures are never committed; the replay tests replay any present locally and skip
otherwise. The committed tests use synthetic streams and a golden log
(`engine/tests/fixtures/synthetic/`) and generated ones (property tests and fault injection in
`engine/tests/`, crash injection and replay-equals-live in `harness/engine-harness/tests/`).
`TP_CRASH_ROUNDS` lengthens the crash-injection run (24 kills by default).

## Ising simulator

`python/tipping_point/src/tipping_point/ising/` simulates the 2D Ising model with checkerboard
Metropolis in PyTorch: int8 spins, a batch of independent replicas, one seeded generator per run.
Runs follow the shared layout `<data dir>/ising/<run_id>/` (`manifest.json`, Parquet chunks under
`stats/`, `configs/`, `balance/`, and `complete.json` after a clean end). The data dir is
`--data-dir` or `$TP_DATA_DIR`, outside the repository; `--device` defaults to CUDA when available.

```sh
uv run python -m tipping_point.ising validate                        # Gates A-C on a small L = 4 run
uv run python -m tipping_point.ising pilot --L 16 --sweeps 50000      # tau_int per T -> tau.json
uv run python -m tipping_point.ising staircase --L 16 --tau <pilot run>/tau.json
uv run python -m tipping_point.ising ramp --L 16 --x 30               # Kibble-Zurek x: 30 fast, 0.3 slow
uv run python -m tipping_point.ising resume <run dir>                 # after a crash or a killed session
```

At Tc alone (seed 1) the L = 16 pilot reads tau(|m|) = 52.8 (binning 49.0) and tau(e) = 27.5 (#47).

## Running the gate

```sh
uv sync --extra cpu                      # or --extra cu126 on a CUDA machine (the PyTorch build)
uv run python harness/gate.py            # all legs
uv run python harness/gate.py --leg pytest
```

Needs `cargo` on `PATH`. Each leg (cargo fmt, clippy, test; ruff check, format; pyright; pytest) runs on its own and is GREEN only when its exit code is 0 and its own summary evidence is positive. The gate exits nonzero if any leg is RED; the full record lands in `harness/out/gate-<timestamp>.json`. The cargo legs cover only the root workspace's members: `harness/tests/test_gate.py` fails `engine/` or any crate under `harness/` that is outside it, so when `app/src-tauri` lands it must be added to `members` and to that check.
