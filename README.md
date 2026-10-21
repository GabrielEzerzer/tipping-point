# Tipping Point

Open-source framework for detecting critical transitions in live data streams: a JEPA world model measures, Jev triages, and tool-calling agents investigate.


## Layout

- `engine/` — Rust core, a member of the root Cargo workspace (`Cargo.toml`).
- `python/tipping_point/` — the Python package (src layout), a member of the root uv workspace (`pyproject.toml`).
- `app/` — the Tauri desktop app (not yet started).
- `paper/` — the paper.
- `harness/` — supporting code that verifies the rest: the gate and its tests. Outside both workspaces.

## Running the gate

```sh
uv sync
uv run python harness/gate.py            # all legs
uv run python harness/gate.py --leg pytest
```

Needs `cargo` on `PATH`. Each leg (cargo fmt, clippy, test; ruff check, format; pyright; pytest) runs on its own and is GREEN only when its exit code is 0 and its own summary evidence is positive. The gate exits nonzero if any leg is RED; the full record lands in `harness/out/gate-<timestamp>.json`.
