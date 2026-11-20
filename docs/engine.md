# Engine

The Rust engine (`engine/`) keeps local order books from exchange feeds, stores every input it
reads, rebuilds the books from storage, imports historical archive days, and serves runs over a
local HTTP API. Its test binaries and checks live in `harness/engine-harness/`, which production
code never depends on.

## Release builds, one heavy job at a time

Run the engine's binaries (`ingest`, `store` and `serve`, and the harness's `record` and `soak`)
from a release build: `cargo build --release -p engine -p engine-harness` once, then the
`cargo run --release ...` lines below, or `target/release/<bin>` directly. Plain `cargo run`
builds the dev profile, where rotate (binary log to Parquet) is roughly ten times slower.
`serve --record` rotates in the background as each segment closes, so in a dev build a long
recording falls behind and replay finds gaps. Tests stay in the dev profile, which compiles much
faster: `cargo test` and the gate build it.

On a laptop, run one heavy job at a time: a cargo build, `cargo test` or the gate, a recording
session or a by-hand `store rotate`, a simulator run, a training. They share the same cores, so
two at once each run slower, and a recording's live loop can fall behind the feed.

## Venues and the live feed

`engine/src/market/` is the market-data core every venue shares: integer-unit order books, a
multi-symbol `feed.rs` that turns a venue's raw frames into book updates, and `pipeline.rs`, the
one copy of how the engine reacts to every input; the network layer (`ws.rs`, `live.rs`) is thin
around it. What differs between venues (wire format, sync rule, scales, endpoints) lives in
`engine/src/venue/`, reached by matching on `enum Venue`. `venue/binance/` keeps a correct local
order book for BTCUSDT and ETHUSDT from the USD-M depth stream (REST snapshot plus diffs,
sequence-gap detection, resync, backpressure) and reads aggTrades; its sync rule is a pure state
machine (`sync.rs`). `venue/bybit/` reads Bybit's linear order book, which comes in band (a
snapshot, then deltas whose update id runs on by one; a skipped id drops the book until the next
snapshot); it has no live loop yet and is read from Bybit's archive. One run holds one venue, and
one `serve` hub takes several venues' loops at once.

A run keeps the scales (tick and lot size) it started with, from its manifest to its Parquet and
the API's `welcome`, so a tick or lot size made finer mid-run leaves that symbol's depth updates
unconvertible and its book down (`parse_errors` climbs). After ten such updates in a row the
Binance loop asks for the symbol's exchange info again, at most every ten minutes, and if the
scales did change it says so (`ScalesChanged`, printed by `ingest` and `soak`); a restart records
a new run at the new scales. A check the venue fails (a 429, a timeout) says so as well
(`ScalesCheckFailed`), so silence never stands for unchanged scales, and the symbols due a check
at once share one exchange-info download.

```sh
cargo run --release -p engine --bin ingest -- --seconds 60            # live: logs top-of-book and resyncs
cargo run --release -p engine --bin ingest -- --record                # ...and records a run to the data dir (Ctrl-C ends it cleanly)
cargo run --release -p engine-harness --bin record -- --seconds 60    # a run + the live transcript → harness/fixtures/recorded/ (gitignored)
cargo run --release -p engine-harness --bin soak -- --hours 24        # gaps/resyncs/reconnects/drops per hour → harness/out/
```

## Market data on disk

`engine/src/store/` is the storage. The live loop logs every input right before applying it
(depth and trade frames as received, snapshots, and control events: connects, disconnects, frames
dropped under backpressure, failed snapshots) to an append-only binary log: 36-byte header, raw
payload, CRC-32C, one `write_all` per frame, `sync_data` every second, a new segment every UTC hour
or 1 GiB. A crash can only tear the tail; recovery cuts it back to the last good frame and refuses
damage in the middle. An offline step rotates closed segments into Parquet (`frames`,
`depth_updates`, `trades`, `snapshots`, `book_samples`), and DuckDB queries and replays it
through the same pipeline the live loop runs.

Runs live in `<data dir>/market/<run_id>/` (`manifest.json`, `log/`, one Parquet file per table per
segment, `complete.json` after a clean end), the layout the Ising simulator shares. The data dir is
`$TP_DATA_DIR`, or else `%LOCALAPPDATA%\tipping-point` (`$XDG_DATA_HOME/tipping-point` or
`~/.local/share/tipping-point` elsewhere), never the repository. The Python CLIs (`ising`, `ews`,
`jepa`) resolve the same folder when `--data-dir` is not given (`tipping_point.engine.data_dir`),
so the engine and the simulator share one root with no configuration.

```sh
cargo run --release -p engine --bin store -- recover <run dir>              # after a crash: close the newest segment
cargo run --release -p engine --bin store -- rotate <run dir>               # closed segments → Parquet (safe to re-run)
cargo run --release -p engine --bin store -- replay <run dir> [--parquet]   # rebuild the books from the log or the Parquet
```

## Historical archives

`store import` turns one downloaded archive day into a run: Bybit's free linear order-book
archive (`<yyyy-mm-dd>_<SYMBOL>_ob200.data.zip`: 200 levels every 100 ms, 500 before 2025-08-21;
the source of book features) or Binance's USD-M aggregated trades
(`<SYMBOL>-aggTrades-<yyyy-mm-dd>.zip`). These trades are not the source of labels: the regime
labels and the cascade proxy read Binance's klines, metrics and funding through
`tipping_point.market` (see [market.md](market.md#regime-labels)), with taker flow from the klines.
Within `market/`, aggTrades appear only as the stability report's optional cross-check of the
variance estimator, and no code there reads these files yet. Each record goes through its venue's
adapter as the frame the live stream would have carried, into the same pipeline and sampler the
live loop and rotate run, and out as Parquet in
`<data dir>/market/<yyyymmdd>T000000Z-<venue>-<symbol>/`: book samples for a book and trades for
trades by default, `--tables` for more (`frames`, `depth_updates`, `snapshots`). There is no
binary log: the downloaded file is the lossless copy, named in the manifest with its URL, size and
SHA-256 (checked against Binance's `.CHECKSUM` when one sits beside it). An archive run runs on
the exchange's clock (`clock: exchange`), so join it to live runs on exchange time, never on
capture time. Its scales come from a pre-scan of the file, not from today's exchange info, and
`complete.json` keeps the integrity checks: records, gaps in the venue's id chain, snapshots,
times outside the day, and what the pipeline made of them, down to the book-sample rows written
while the book was not synced. Re-running is safe; another archive for the same run is refused.
Binance's archives are CC BY-NC-SA 4.0: anything derived from them and published carries that
licence and the "Binance Vision" attribution.

```sh
cargo run --release -p engine --bin store -- import <archive> [--data-dir DIR] [--tables T,...]
```

Downloaded days stay outside the repository, in `<data dir>/archives/`; with `TP_ARCHIVE_DAYS`
set, the import tests import every day found there (`TP_ARCHIVE_DAYS=1 cargo test --release -p
engine-harness --test archive_import local_archive_days`). The committed tests use tiny synthetic
archives in each venue's format (`engine/tests/fixtures/archives/`, made by
`harness/engine-harness/tests/make_archive_fixtures.py`).

## The DuckDB build

DuckDB is linked from its prebuilt library, which the build downloads once per target dir
(`DUCKDB_DOWNLOAD_LIB` in `.cargo/config.toml`), so the first build needs the network. On
Windows the build accepts only the archive whose SHA-256 is pinned in
`engine/build/duckdb_archive.rs` (x64's; Windows on Arm fails until its archive is pinned too)
and fails on any other: libduckdb-sys checks no hash, and a GitHub release's assets can be
replaced under the same tag. A DuckDB bump changes that pin in the same commit as the `duckdb`
and `libduckdb-sys` pins, taking the new release asset's digest from GitHub. On Windows the build
also copies `duckdb.dll` next to the binaries (`target/<profile>/`, `engine/build.rs`), so
`store.exe` and `serve.exe` start from any shell, not only under cargo; a binary copied elsewhere
needs that DLL beside it. Linux and macOS binaries find the library through their rpath.

## Test data

Recorded fixtures are never committed; `replay_equals_live` replays any present locally and,
with none, says so on a `SKIPPED:` line the gate counts rather than passing silently. A
recording whose `live-transcript.json` an older harness wrote is skipped the same way: its
`format` is below `FORMAT` (`harness/engine-harness/src/transcript.rs`, bumped whenever what a
transcript step records changes), so there is nothing to compare it with. Record it again
(`cargo run --release -p engine-harness --bin record -- --seconds 60`) and delete the old run
from `harness/fixtures/recorded/market/`. The committed tests use synthetic streams and a golden
log (`engine/tests/fixtures/synthetic/`) and generated ones (property tests and fault injection in
`engine/tests/`, crash injection and replay-equals-live in `harness/engine-harness/tests/`).
`TP_CRASH_ROUNDS` lengthens the crash-injection run (24 kills by default). With `TP_ROTATE_SOAK`
set, `cargo test --release -p engine-harness --test rotate_soak -- --nocapture` writes one full
1 GiB segment of synthetic frames, rotates it, and records the time and peak memory in
`harness/out/rotate-soak-<ms>.json` (`TP_ROTATE_SOAK_MB` for another size).

## Engine API

`serve` runs the engine as its own process: the live feed plus one HTTP server on 127.0.0.1
(`engine/src/api/`). Clients (a WebSocket viewer, the Python worker) are separate processes. It
prints `serve listening on http://127.0.0.1:<port>` once it accepts connections, so a parent
that passed `--port 0` reads the port from it.

```sh
cargo run --release -p engine --bin serve                                   # live Binance feed on 127.0.0.1:7731
cargo run --release -p engine --bin serve -- --record                       # ...recorded, closed segments rotated in the background
cargo run --release -p engine --bin serve -- --script <file.jsonl> --speed 10 --port 0   # a recorded script played back instead
```

| route | what |
|---|---|
| `GET /v1/health` | alive, protocol version, whether the live feed has started or ended, the run each venue records, and `errors`: samples that could not be made and live feeds refused because their venue had one, both never expected |
| `GET /v1/runs` | every run in the data dir, by domain, a market run with its venue |
| `GET /v1/live/book_samples` | Arrow IPC stream: live book samples, each batch one or more whole 100 ms instants (one per batch live; a backfill or an unpaced replay sends several); `?backfill_s=` first replays that much of the recent past from memory (no gap, no overlap), `?run=&speed=` replays a recorded run instead |
| `GET /v1/table/{domain}/{run}/{table}` | Arrow IPC stream: any table of a market or Ising run, `?start=&end=` a range of its key, read through DuckDB and cast back to the schema stored in the Parquet footer |
| `POST /v1/publish/{topic}` | a JSON result (`meters`, `pnl`, `ising`) relayed to WebSocket clients; `meters` and `pnl` keep the latest per `venue` and `symbol` the result names, `ising` keeps every message |
| `GET /v1/ws` | the WebSocket (protocol 2): JSON `book` (top 20 per venue and symbol), `trades`, `status` (per venue), and the relayed topics; replay and seek of recorded runs; binary frames carry only Ising lattices |

- **Live samples come from rotate's own sampler** (`store/samples.rs`), each row carrying the log
  seq of the last frame applied, which the live loop assigns: the live stream equals the
  `book_samples` table rotate writes later.
- **Ingest never waits on a client.** The live loop publishes into `watch` (latest value) and
  bounded `broadcast` channels. A WebSocket client is sent the newest book at most every 16 ms, a
  `gap` when it falls behind on trades, and close code 4002 after a 10 s stall; an Arrow stream is
  lossless or cut, its response aborted so a reader never takes it for a short table.
- **Times on the wire are milliseconds** (`ts_ms`), prices and quantities integer ticks and lots at
  the scales sent in `welcome`; storage keeps nanoseconds.
- **Venues:** two venues can share a symbol (Binance and Bybit both say `BTCUSDT`), so every
  message, table row and Arrow stream that names a symbol names its `venue` too; trade ids are
  strings (Bybit's are UUIDs). A market stream's `tp.runs` and `tp.scales` are keyed by venue; the
  market tables keyed by venue are at schema major 2.
- **Checks on every request:** `Host` must name the server (stops DNS rebinding) and a browser's
  `Origin` must be on the allowlist (the local origins listed in `engine/src/api/mod.rs`, or one
  given with `--allow-origin`).
- **Replay and scrub:** rotated hours come from Parquet through DuckDB, the hour not yet rotated
  from memory, as the same `book` and `trades` messages. The Parquet is read ten minutes of the
  run at a time, since each read pays a fixed cost (a DuckDB connection, every file's footer, an
  hour's row group decoded). `replay_speed` in `engine/src/api/replay.rs` measures it:
  `TP_REPLAY_HOURS=2 cargo test -p engine --lib replay_speed -- --ignored --nocapture`.

Training reads Parquet directly (`tipping_point.engine.read`, no engine); the engine's Arrow
streams serve live inference and replay tests of it (`live`, `table`, `publish`; stdlib and
pyarrow only). The protocol is pinned by golden files in `engine/tests/fixtures/api/`; a change
fails `api_contract` until they are regenerated in the same commit
(`TP_UPDATE_GOLDEN=1 cargo test -p engine-harness`). `api_load` drives ten times the live rate,
its depth frames as deep as Binance's (20 to 60 levels a side, one in a hundred 150 to 500, from
captured frames), into nine WebSocket clients, two Arrow clients and one stalled client of each
kind, and writes its measurements, the live path's time per depth frame among them, to
`harness/out/api-load-<ms>.json`; `TP_LOAD_SECONDS` runs it longer than its default 5 s. It fails
if any input takes the live path half the stall timeout or longer; the rates achieved are
recorded, not asserted, since on a shared machine they measure the CPU the test got. The report
also holds the cores, how late a 1 ms sleep woke through the load and the share of its target rate
the generator reached: the machine counts as loaded when the sleep's p99 passes a quarter of the
latency bound or the generator falls under half its target. The latency bound is checked on every
run; only on a loaded machine is a p99 over it excused, and only while it stays under the bound
plus three times the sleep's p99 (a message's two or three wakeups), with a `SKIPPED:` line the
gate counts, so a skip means the bound was missed under load. Past that it fails, loaded or not.
The Python round trip (`api_python`) runs the repository's `.venv` Python, or `$TP_PYTHON`.

## Reading runs from Python

`tipping_point.engine.read` follows the shared layout
`<root>/<domain>/<run_id>/<table>/<first_key:020>.parquet`, and `run_id` names a new run by the
engine's rule, so the engine can open it. Every Arrow stream keeps the stored schema, with
`tp.schema = "<table>/<major>"` in its metadata. A market stream says where its rows come from in
`tp.runs` and `tp.scales` (decoded by `tipping_point.engine.runs` and `scales`); an Ising stream
names its run in `tp.run`. A column the schema declares non-nullable never holds a null, so
`torch.from_dlpack(column)` shares its buffer; nullable book levels (past the book's depth, or
while unsynced) must be filled first, which copies.
