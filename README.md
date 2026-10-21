# Tipping Point

Open-source framework for detecting critical transitions in live data streams: a JEPA world model measures, Jev triages, and tool-calling agents investigate.


## Layout

- `engine/` — Rust core, a member of the root Cargo workspace (`Cargo.toml`).
- `python/tipping_point/` — the Python package (src layout), a member of the root uv workspace (`pyproject.toml`).
- `app/` — the Tauri desktop app (not yet started).
- `paper/` — the paper.
- `harness/` — supporting code that verifies the rest: the gate and its tests, plus `engine-harness/`,
  a Cargo workspace member holding the engine's recorder and soak binaries (production code never
  depends on it).

## Binance ingest

`engine/src/binance/` keeps a correct local order book for BTCUSDT and ETHUSDT from the USD-M
depth stream (snapshot plus diffs, sequence-gap detection, resync, backpressure) and reads aggTrades.
The sync logic is a pure state machine (`sync.rs`) behind a multi-symbol `feed.rs`; the network
layer (`ws.rs`, `rest.rs`, `live.rs`) is thin around it.

```sh
cargo run -p engine --bin ingest -- --seconds 60            # live: logs top-of-book and resyncs
cargo run -p engine-harness --bin record -- --seconds 60    # raw frames + snapshots → harness/fixtures/recorded/ (gitignored)
cargo run -p engine-harness --bin soak -- --hours 24        # gaps/resyncs/reconnects/drops per hour → harness/out/
```

Recorded fixtures are never committed; `engine/tests/replay.rs` replays any present locally and skips
otherwise. The committed tests use synthetic streams (`engine/tests/fixtures/synthetic/`) and
generated ones (property tests and fault injection in `engine/tests/`).

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

## Running the gate

```sh
uv sync --extra cpu                      # or --extra cu126 on a CUDA machine (the PyTorch build)
uv run python harness/gate.py            # all legs
uv run python harness/gate.py --leg pytest
```

Needs `cargo` on `PATH`. Each leg (cargo fmt, clippy, test; ruff check, format; pyright; pytest) runs on its own and is GREEN only when its exit code is 0 and its own summary evidence is positive. The gate exits nonzero if any leg is RED; the full record lands in `harness/out/gate-<timestamp>.json`.
