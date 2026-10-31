# Tipping Point

Tipping Point is a research project and an open-source framework built around one question: do
learned early-warning signals of critical transitions pass a falsification test based on
finite-size scaling? The 2D Ising model is where the ground truth is exact: its critical
temperature and exponents are known, so a signal that claims to detect its transition must change
with lattice size as finite-size scaling predicts. The project trains a JEPA (joint-embedding
predictive architecture) world model on raw Ising configurations, derives early-warning signals
from its latents and tests them against that prediction, alongside the classic indicators computed
on the raw signal. Crypto order books are the application: the same pre-registered protocol is run
on volatility-regime shifts in Bitcoin and Ether perpetual futures. For live streams the framework
also plans triage and an agent layer: Jev, a decision model that TypeSafe offers through a public
API, answers fixed questions about a JSON summary of the measurements, and a tool-calling agent
investigates only when its answer is confident. The desktop app and the agent layer are product,
not part of the paper's claim.

## Status as of 25 October 2026

- Built and tested: the Rust market-data engine, with live and archive venues, storage and replay,
  and the engine API; the GPU Ising simulator, validated against exact results; the classic
  early-warning scoring, with its pre-registration tooling; the JEPA world-model core.
- In progress: the pre-registered confirmatory datasets at L = 16, 32 and 64, generated only after
  the pre-registration is frozen as a git tag; JEPA training.
- Planned: the market experiments, the desktop app and the paper, aimed at an ICLR 2027 workshop.

The plan lives on the issue tracker, starting from the map in [#1](https://github.com/GabrielEzerzer/tipping-point/issues/1).
The pre-registration's design and the decisions behind it are in [#26](https://github.com/GabrielEzerzer/tipping-point/issues/26).

## Results so far (exploratory, not pre-registered)

- At Tc the Binder ratio is 1.1709 at L = 16 and 1.1625 at L = 32, and both 95% CIs contain the exact value, 1.1679.
- From L = 16 to 32, L^2 Var(|m|) scales as L^1.67 [1.57, 1.76] and the mean |m| as L^-0.113 [-0.125, -0.101]; the exact exponents, 7/4 and -1/8, lie inside these 95% CIs, -1/8 at the edge.
- Two sizes give a slope, not a test: the pre-registered test at L = 16, 32 and 64 is next ([full tables on #26](https://github.com/GabrielEzerzer/tipping-point/issues/26#issuecomment-6390788147)).

## Quick start

With [uv](https://docs.astral.sh/uv/) installed, run these from the repository root:

```sh
uv sync --extra cpu                             # the Python environment (--extra cu126 on a CUDA machine)
uv run python harness/gate.py                   # the gate: every Rust and Python check; needs cargo on PATH
uv run python -m tipping_point.ising validate   # a small L = 4 simulator run, checked against exact results
```

The sections below document the engine, the simulator, the scoring and the JEPA in detail.

## Licence

Tipping Point is released under the MIT licence; see [LICENSE](LICENSE).

## Layout

- `engine/` — Rust core, a member of the root Cargo workspace (`Cargo.toml`).
- `python/tipping_point/` — the Python package (src layout), a member of the root uv workspace (`pyproject.toml`).
- `app/` — the Tauri desktop app (not yet started).
- `notebooks/` — the Kaggle kernel runner (`kaggle_ising_runs.py`) and the plans it runs (`plans/`).
- `paper/` — the paper.
- `harness/` — supporting code that verifies the rest: the gate and its tests, plus `engine-harness/`,
  a Cargo workspace member holding the engine's recorder, soak and crash-injection binaries, the
  replay-equals-live checks and the API's contract, parity, Python and load checks (production code
  never depends on it).

## Release builds, one heavy job at a time

Run the engine's binaries (`ingest`, `store` and `serve`, and the harness's `record` and `soak`)
from a release build: `cargo build --release -p engine -p engine-harness` once, then the
`cargo run --release ...` lines below, or `target/release/<bin>` directly. Plain `cargo run`
builds the dev profile, where rotate (binary log to Parquet) is roughly ten times slower: 5.4-6.0
minutes per recorded hour against 35-37 s (#63). `serve --record` rotates in the background as
each segment closes, re-reading from the start, so in debug live-run replay finds gaps from about
the fifth hour, and Ctrl-C after a day of recording waits about an hour for the final rotate
(about 10 minutes in release). The app's Tauri sidecar, when it lands, ships the release binary.
Tests stay in the dev profile, which compiles much faster: `cargo test` and the gate build it.

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

A run keeps the scales it started with, from its manifest to its Parquet and the API's `welcome`,
so a tick or lot size made finer mid-run leaves that symbol's depth updates unconvertible and its
book down (`parse_errors` climbs). After ten such updates in a row the Binance loop asks for the
symbol's exchange info again, at most every ten minutes, and if the scales did change it says so
(`ScalesChanged`, printed by `ingest` and `soak`); a restart records a new run at the new scales
(#45). A check the venue fails (a 429, a timeout) says so as well (`ScalesCheckFailed`), so
silence never stands for unchanged scales, and the symbols due a check at once share one
exchange-info download (#100).

```sh
cargo run --release -p engine --bin ingest -- --seconds 60            # live: logs top-of-book and resyncs
cargo run --release -p engine --bin ingest -- --record                # ...and records a run to the data dir (Ctrl-C ends it cleanly)
cargo run --release -p engine-harness --bin record -- --seconds 60    # a run + the live transcript → harness/fixtures/recorded/ (gitignored)
cargo run --release -p engine-harness --bin soak -- --hours 24        # gaps/resyncs/reconnects/drops per hour → harness/out/
```

## Market data on disk

`engine/src/store/` is the storage. The live loop logs every input right before applying it —
depth and trade frames as received, snapshots, and control events (connects, disconnects, frames
dropped under backpressure, failed snapshots) — to an append-only binary log: 36-byte header, raw
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

### Historical archives

`store import` turns one downloaded archive day into a run: Bybit's free linear order-book
archive (`<yyyy-mm-dd>_<SYMBOL>_ob200.data.zip`: 200 levels every 100 ms, 500 before 2025-08-21;
the source of book features) or Binance's USD-M aggregated trades
(`<SYMBOL>-aggTrades-<yyyy-mm-dd>.zip`: the source of labels). Each record goes through its
venue's adapter as the frame the live stream would have carried, into the same pipeline and
sampler the live loop and rotate run, and out as Parquet in
`<data dir>/market/<yyyymmdd>T000000Z-<venue>-<symbol>/`: book samples for a book and trades for
trades by default, `--tables` for more (`frames`, `depth_updates`, `snapshots`). There is no
binary log: the downloaded file is the lossless copy, named in the manifest with its URL, size and
SHA-256 (checked against Binance's `.CHECKSUM` when one sits beside it). An archive run runs on
the exchange's clock (`clock: exchange`), so join it to live runs on exchange time, never on
capture time. Its scales come from a pre-scan of the file, not from today's exchange info, and
`complete.json` keeps the integrity checks: records, gaps in the venue's id chain, snapshots,
times outside the day, and what the pipeline made of them, down to the book-sample rows written
while the book was not synced. Re-running is safe; another archive
for the same run is refused. Binance's archives are CC BY-NC-SA 4.0: anything derived from them
and published carries that licence and the "Binance Vision" attribution.

```sh
cargo run --release -p engine --bin store -- import <archive> [--data-dir DIR] [--tables T,...]
```

Downloaded days stay local (`harness/fixtures/archives/`, gitignored); with `TP_ARCHIVE_DAYS`
set, the import tests import every day found there (`TP_ARCHIVE_DAYS=1 cargo test --release -p
engine-harness --test archive_import local_archive_days`). The committed tests use tiny synthetic archives in each venue's format
(`engine/tests/fixtures/archives/`, made by `harness/engine-harness/tests/make_archive_fixtures.py`).

DuckDB is linked from its prebuilt library, which the build downloads once per target dir
(`DUCKDB_DOWNLOAD_LIB` in `.cargo/config.toml`), so the first build needs the network. On
Windows the build accepts only the archive whose SHA-256 is pinned in
`engine/build/duckdb_archive.rs` (x64's; Windows on Arm fails until its archive is pinned too)
and fails on any other: libduckdb-sys checks no hash, and a GitHub release's assets can be
replaced under the same tag (#66). A DuckDB bump changes that pin in the same commit as the
`duckdb` and `libduckdb-sys` pins, taking the new release asset's digest from GitHub. On
Windows the build also copies `duckdb.dll` next to the binaries (`target/<profile>/`,
`engine/build.rs`), so `store.exe` and `serve.exe` start from any shell, not only under cargo;
a binary copied elsewhere needs that DLL beside it. Linux and macOS binaries find the library
through their rpath.

Recorded fixtures are never committed; `replay_equals_live` replays any present locally and,
with none, says so on a `SKIPPED:` line the gate counts rather than passing silently. A
recording whose `live-transcript.json` an older harness wrote is skipped the same way: its
`format` is below `FORMAT` (`harness/engine-harness/src/transcript.rs`, bumped whenever what a
transcript step records changes), so there is nothing to compare it with. Record it again
(`cargo run --release -p engine-harness --bin record -- --seconds 60`) and delete the old run
from `harness/fixtures/recorded/market/` (#101). The committed tests use synthetic streams and a golden log
(`engine/tests/fixtures/synthetic/`) and generated ones (property tests and fault injection in
`engine/tests/`, crash injection and replay-equals-live in `harness/engine-harness/tests/`).
`TP_CRASH_ROUNDS` lengthens the crash-injection run (24 kills by default). With `TP_ROTATE_SOAK`
set, `cargo test --release -p engine-harness --test rotate_soak -- --nocapture` writes one full
1 GiB segment of synthetic frames, rotates it, and records the time and peak memory in
`harness/out/rotate-soak-<ms>.json` (#52; `TP_ROTATE_SOAK_MB` for another size).

### Regime labels (#29)

The market experiment's labels come from Binance's free USD-M archive and need no engine
(`python/tipping_point/src/tipping_point/market/`, stdlib, numpy and pyarrow). `fetch` downloads a
span's 1-minute klines, 5-minute metrics and monthly funding into
`harness/fixtures/archives/binance/<kind>/<symbol>/` (gitignored), checks each file's SHA-256
against the `.CHECKSUM` beside it, skips files already verified, and lists every file in that
folder's `manifest.json` with its URL, size, sha256 and row checks: gaps in the minute or 5-minute
grid, duplicate stamps, rows out of order. `labels` turns a fetched span into 5-minute bars
(Garman-Klass variance, close, volume, taker-buy share, open interest, funding) and labels them:
a two-state Gaussian HMM on deseasonalised, baseline-relative log variance, refitted every Monday
by deterministic EM, read through as-of filtered probabilities and a hysteresis rule into
episodes, events, flickers and null stretches. A run lands in `<data dir>/labels/<run_id>/`:
`manifest.json` with every source file and parameter, Parquet tables `bars`, `labels`, `refits`,
`episodes` and `nulls` keyed by `ts_ns`, then `complete.json`. Every labeller parameter is an
option at the research pass's start value until the 2023 pilot sets it; days from 2024-01-01 are
refused until `market-labels-v1` is frozen.

```sh
uv run python -m tipping_point.market fetch --start 2022-07-01 --end 2023-12-31
uv run python -m tipping_point.market labels --start 2022-07-01 --end 2023-12-31
```

The cascade proxy, `market/cascade.py`, reads the same fetched files. No free source holds the
span's liquidations, so a cascade is inferred per symbol from a trailing 30-minute window: open
interest in contracts falls by at least `a` (as a log ratio), the price moves at least `b` one way,
taker flow leans at least `c` the same way, and volume reaches `v` times the median 30-minute
volume of the trailing 28 days. A long cascade falls and a short squeeze rises; a drop in open
interest alone fires neither. A window touching a missing open-interest step is not evaluated and
is counted. Candidates within 24 hours merge into the earliest, and both symbols firing within an
hour count as one market event with two asset instances; both counts are reported. The pooled list,
with each instance's onset and the bounds of its L2 window 12 hours either side, lands as the table
`cascades` of a run in `<data dir>/labels/<run_id>/`. `a`, `b`, `c` and `v` are options at their
start values (3 %, 2.5 %, 0.15, 3) until the 2023 pilot sets them, and days from 2024-01-01 are
refused as the labels' are. How open interest lines up with price is not an option, because the
feed's timing fixes it: a bar's row holds the open interest at its start and its close the price at
its end, so each stamp pairs with the klines before it and all the legs span the same 30 minutes (in
every month of 2023, on both symbols, a change in open interest correlates most with the 1-minute
returns between its two stamps). A `liquidations` table in a market run (#90) is read when there is
one; until then the run says `liquidations: not recorded`. The first 28 days only fill the trailing
medians, so this span lists 2023 alone:

```sh
uv run python -m tipping_point.market.cascade --start 2022-12-04 --end 2023-12-31
```

`market/documented_cascades.toml` lists documented cascades with their sources. With
`TP_CASCADE_RECALL` set, `uv run pytest python/tipping_point/tests/market/test_cascade.py -k
documented_cascades_on_real_days` runs the rule over each event found fetched, with the 29 days
before it; without it the test skips and says so. Five of the six events lie in the confirmatory
span, so the check gives only what the rule says inside each event's own days and reports recall
with no floor: nothing is to be tuned on it. Two more checks read 2023 days if they are fetched, and
otherwise skip and say so: the metrics stamp's timing, and 2023-08-17 at the start values.

`market/reports.py` writes two reports on labelled series, each as Parquet tables plus a Markdown
summary for the issue. The label-count report counts events, episodes, flickers, durations and time
high per asset and month, clocks the onsets by UTC hour and weekday, and counts onsets that coincide
across assets, the null hours and the 4-hour slices they hold. The stability report compares the
as-of labels with an oracle (the final theta, smoothed over the whole sample) and with each declared
variant and a plain threshold rule, follows theta over the refits with its flagged fits and the
jumps they make in P, and gives the filter's detection delay. The calendar of CPI, NFP and FOMC
releases, the cascade candidates and the aggTrades cross-check are optional inputs; a report without
one says "not supplied". `market/nulls.py` holds the GARCH(1,1)-t simulator, the delay simulation,
the event-time shift and the circularity floor. The floor simulates 10-second returns, builds
1-minute OHLC, runs the real bars and labeller, and scores 1-minute close-to-close returns, which
gives the AUC a clustering process with no critical dynamics produces by construction. Its GARCH
parameters, pipelines and `scoring.auc` are inputs: the parameters' fit belongs to the 2023 pilot,
and no module in `market/` may import the feature code (`tests/market/test_circularity.py`).

## Engine API

`serve` runs the engine as its own process: the live feed plus one HTTP server on 127.0.0.1
(`engine/src/api/`). The app is a WebSocket client and the Python worker a third process. It
prints `serve listening on http://127.0.0.1:<port>` once it accepts connections, so a parent
that passed `--port 0` reads the port from it.

```sh
cargo run --release -p engine --bin serve                                   # live Binance feed on 127.0.0.1:7731
cargo run --release -p engine --bin serve -- --record                       # ...recorded, closed segments rotated in the background
cargo run --release -p engine --bin serve -- --script <file.jsonl> --speed 10 --port 0   # a recorded script played back instead
```

| route | what |
|---|---|
| `GET /v1/health` | alive, protocol version, whether the live feed has started or ended, the run each venue records, and `errors`: samples that could not be made and live feeds refused because their venue had one, both never expected (#95) |
| `GET /v1/runs` | every run in the data dir, by domain, a market run with its venue |
| `GET /v1/live/book_samples` | Arrow IPC stream: live book samples, each batch one or more whole 100 ms instants (one per batch live; a backfill or an unpaced replay sends several); `?backfill_s=` first replays that much of the recent past from memory (no gap, no overlap), `?run=&speed=` replays a recorded run instead |
| `GET /v1/table/{domain}/{run}/{table}` | Arrow IPC stream: any table of a market or Ising run, `?start=&end=` a range of its key, read through DuckDB and cast back to the schema stored in the Parquet footer |
| `POST /v1/publish/{topic}` | a JSON result (`meters`, `jev`, `pnl`, `ising`) relayed to the app |
| `GET /v1/ws` | the app's WebSocket (protocol 2): JSON `book` (top 20 per venue and symbol), `trades`, `status` (per venue), and the relayed topics; replay and seek of recorded runs; binary frames carry only Ising lattices |

- **Live samples come from rotate's own sampler** (`store/samples.rs`), each row carrying the log
  seq of the last frame applied, which the live loop now assigns: the live stream equals the
  `book_samples` table rotate writes later.
- **Ingest never waits on a client.** The live loop publishes into `watch` (latest value) and
  bounded `broadcast` channels. An app client is sent the newest book at most every 16 ms, a `gap`
  when it falls behind on trades, and close code 4002 after a 10 s stall; an Arrow stream is
  lossless or cut, its response aborted so a reader never takes it for a short table.
- **Times on the wire are milliseconds** (`ts_ms`), prices and quantities integer ticks and lots at
  the scales sent in `welcome`; storage keeps nanoseconds.
- **Venues:** two venues can share a symbol (Binance and Bybit both say `BTCUSDT`), so every
  message, table row and Arrow stream that names a symbol names its `venue` too; trade ids are
  strings (Bybit's are UUIDs). A market stream's `tp.runs` and `tp.scales` are keyed by venue; the
  market tables keyed by venue are at schema major 2.
- **Checks on every request:** `Host` must name the server (stops DNS rebinding) and a browser's
  `Origin` must be on the allowlist (the Tauri webview, the Vite dev server, `--allow-origin`).
- **Replay and scrub:** rotated hours come from Parquet through DuckDB, the hour not yet rotated
  from memory, as the same `book` and `trades` messages. The Parquet is read ten minutes of the
  run at a time: each read pays a fixed cost (a DuckDB connection, every file's footer, an hour's
  row group decoded), so reading 10 s at a time took 46 s per recorded hour at speed 0 in a debug
  build, against 3.3 s now (#69). `replay_speed` in `engine/src/api/replay.rs` measures it:
  `TP_REPLAY_HOURS=2 cargo test -p engine --lib replay_speed -- --ignored --nocapture`.

Training reads Parquet directly (`tipping_point.engine.read`, no engine); the engine's Arrow
streams serve live inference and replay tests of it (`live`, `table`, `publish`; stdlib and
pyarrow only). The protocol is pinned by golden files in `engine/tests/fixtures/api/`, which the
app's mock replays; a change fails `api_contract` until they are regenerated in the same commit
(`TP_UPDATE_GOLDEN=1 cargo test -p engine-harness`). `api_load` drives ten times the live rate,
its depth frames as deep as Binance's (20 to 60 levels a side, one in a hundred 150 to 500, from
the captured frames, #83), into nine app clients, two Arrow clients and one stalled client of
each kind, and writes its measurements, the live path's time per depth frame among them, to
`harness/out/api-load-<ms>.json`; `TP_LOAD_SECONDS` runs it longer than its
default 5 s. It fails if any input takes the live path half the stall timeout or longer; the rates
achieved are recorded, not asserted, since on a shared machine they measure the CPU the test got.
The report also holds the cores and how late a 1 ms sleep woke through the load: past a quarter of
the latency bound at p99 the machine counts as loaded. The latency bound is checked on every run;
only on a loaded machine is a p99 over it excused, and only while it stays under the bound plus
three times that lateness (a message's two or three wakeups), with a `SKIPPED:` line the gate
counts, so a skip means the bound was missed under load. Past that it fails, loaded or not. The
Python round trip (`api_python`) runs the repository's `.venv` Python, or `$TP_PYTHON`.

## Ising simulator

`python/tipping_point/src/tipping_point/ising/` simulates the 2D Ising model with checkerboard
Metropolis in PyTorch: int8 spins, a batch of independent replicas, one seeded generator per run.
Runs follow the shared layout `<data dir>/ising/<run_id>/` (`manifest.json`, Parquet chunks under
`stats/`, `configs/`, `balance/`, and `complete.json` after a clean end). A run id is the UTC start
time plus a tag in the engine's charset (`tipping_point.engine.run_id`: ASCII letters, digits, `-`
and `_`), so a number's `.` is written `p`, as in `20261023T010000Z-L16-ramp-x0p02-s1`; the
engine's API opens no other name. The data dir is `--data-dir`, else the engine's (see Market data
on disk); `--device` defaults to CUDA when available.

```sh
uv run python -m tipping_point.ising validate                        # Gates A-C on a small L = 4 run
uv run python -m tipping_point.ising pilot --L 16 --sweeps 50000      # tau_int per T -> tau.json
uv run python -m tipping_point.ising tauc <pilot run> ...             # tau_c(L) at Tc -> pilot.json
uv run python -m tipping_point.ising staircase --L 16 --tau <pilot run>/tau.json
uv run python -m tipping_point.ising ramp --L 16 --x 0.02 --pilot <record>/pilot.json --eps-start 12 --eps-end -8  # x: 0.02 slow, 2 fast
uv run python -m tipping_point.ising null --L 16 --epsilon-L 4 --pilot <record>/pilot.json --seed 6   # epsilon_L 2, 4, 6; v1's seeds 6-10
uv run python -m tipping_point.ising resume <run dir>                 # after a crash or a killed session
```

### Which commit a run records (#109)

Every manifest names the code that wrote it in `producer.git_sha` and `producer.dirty`: Ising
runs, JEPA trainings and EWS results all take them from `io.code_version()`. Run from the
checkout, where `uv sync` installs the package editable, they are git's HEAD and whether the tree
has uncommitted changes, untracked files included. A wheel install sits in no checkout, so runs
made from a wheel are stamped from the build: the build hook (`python/tipping_point/hatch_build.py`)
writes `tipping_point/_build_info.py` (`commit`, `dirty`, `kernel_sha256` of `ising/kernel.py`,
`built_at` in UTC) into every sdist and wheel built from a git checkout, and `code_version` reads
it. `tauc` takes a stamped commit exactly as it takes one from git. An installed copy never reads
a repository that only encloses it, such as a venv inside another project. A build made where no
repository tracks the source records no commit, and `tauc` refuses its runs. The hook warns when it
makes one, but `pip` and `uv pip install` hide build output, so `pilot` refuses to start from such
an install rather than leave the refusal to `tauc`, hours later. To build a wheel for a GPU
machine, check out the commit, keep the tree clean, and run:

```sh
uv build --package tipping-point --wheel     # -> dist/tipping_point-0.1.0-py3-none-any.whl, stamped
uv run python harness/wheel_provenance.py --wheel dist/tipping_point-0.1.0-py3-none-any.whl
```

The second line checks a wheel end to end in throwaway venvs (about three minutes on a laptop
CPU): its runs must record the stamped commit, inside or outside another repository, and `tauc`
must accept them. `tauc` also needs every size from one commit and one kernel, so a record over
new sizes re-runs the old sizes from the same build. #50's pilots predate `kernel_sha256` in any
case, so #54 re-measured L = 16 and 32 beside 64 and 128.

### Which stream a run draws

A run's generator is seeded from its identity, not its seed alone: `run.stream_key` hashes the
seed, L, replicas, start and schedule. The manifest holds the result in `params.rng.seed` and names
the rule in `params.rng.derivation`, so the key can be checked from the manifest alone. Runs at one
seed therefore draw different numbers, and one run made twice draws the same whatever its tag,
which only names the folder. `--checkpoint-every` and Gate C's tallies stay out of it: they change
what is written, not what is drawn.

Runs made before this (the seeds 1-5 datasets, the grid pilots, #54's record) drew from the seed
alone, so two of one shape at one seed shared their start and all their numbers, and two holding
one temperature, as two ramps do during their equilibration, were bit-identical. The sharing does
not fade: one field of random numbers moves the uphill flips of every run on it together, even
after their lattices have parted. In a single cut-down CPU realisation on the old seeding, not the
exploratory data (null runs at eps_L 2, 4 and 6 at one seed: L = 16, 32 replicas, 6,000 sweeps
after 1,100; L = 32, 16 replicas, 4,000 sweeps after 4,500), energy correlates replica by replica
at r about 0.2-0.3 and |m| at about 0.01-0.05; a cooling ramp against a null run, sweep to sweep
(L = 16, the same lengths), at about 0.3 in energy and about 0.06 in |m|. Those runs were read
about 20 tau_c after the shared start, so its transient is not in these figures. With a stream per
run, all of them are within noise of zero. Signed M hides the effect. The coupling does touch one
earlier check: the surrogate calibration's binomial band (3.1-6.9%) treated its 480 slices as
independent while the three null runs at each of the five seeds shared a stream, so that pre-flight
check is somewhat weaker than stated; no freeze call rests on it.

### The pilot record: tau_c(L)

The time unit of everything downstream (ramp rates, null-run lengths, the EWS scorer's windows) is
tau_c(L) = tau_int(|m|) at exactly Tc, read from a pilot record and never from a constant. Run one
pilot per L at Tc, then `tauc` fits tau_c = A L^z and writes the record:

```sh
uv run python -m tipping_point.ising pilot --L 16 --T 2.269185314213022 --replicas 64 --therm 2000 --sweeps 50000
uv run python -m tipping_point.ising pilot --L 32 --T 2.269185314213022 --replicas 64 --therm 10000 --sweeps 100000
uv run python -m tipping_point.ising pilot --L 64 --T 2.269185314213022 --replicas 16 --therm 40000 --sweeps 300000
uv run python -m tipping_point.ising tauc <L16 run> <L32 run> <L64 run>    # -> <data dir>/ising/<id>-tauc-L16-32-64/pilot.json
```

`pilot.json` is `{"kernel_commit", "kernel_sha256", "per_L": {"16": {...}, ...}, "fit": {"A", "z",
"A_ci", "z_ci"}}`, each `per_L` entry `{"tau_c", "tau_c_binning", "tau_e", "n_over_tau", "replicas",
"sweeps", "seed"}` plus `"tau_c_ci"` (95%, optional for readers; field notes in `ising/pilot.py`).
A size missing from `per_L` reads tau_c = A L^z. `tauc` refuses a run that is incomplete, never
held exactly Tc, or holds fewer than 300 tau_c: subtracting each series' mean pulls the windowed
estimate about 10 / (N / tau) low, so a recorded tau_c is at most ~3% short (`test_autocorr` pins
the bias).

**A stale record is refused.** `kernel_sha256` is the sha256 of `ising/kernel.py` (the sweep and
its acceptance table) that the pilot runs ran; every run's manifest records its own as
`params.algorithm.kernel_sha256`. `ramp --x`, `null` and `ews score` refuse a record whose
`kernel_sha256` is not the running kernel's, since its tau_c would clock another kernel. The
commit cannot decide this: it moves with every merge, kernel or not. A record written before #55
has no `kernel_sha256` and counts as stale. `--allow-stale-pilot` uses a stale record anyway, with
a warning. Either way the run's manifest names the record it read as `params.pilot`:
`{"run_id", "sha256", "kernel_commit", "kernel_sha256", "stale"}`, and EWS results repeat that stamp
for each input run beside their own record's `kernel_sha256` and `stale` in `params.pilot`. The
record below measured the running kernel, so it reads as current; #50's record, which predates
the field, needs `--allow-stale-pilot`.

The record ising-ews-v1 pins (#54: run `20261025T194846Z-tauc-L16-32-64-128`, commit 655949a,
seed 1), committed byte for byte as `python/tipping_point/tests/fixtures/ising_pilot.json`, sha256
`7c5a4a296f2d8d1d55e223e4c775cd98267c43e6ae89f5e6ec594ac5c0581698`:

| L | replicas x sweeps | tau_c = tau(\|m\|) [95% CI] | binning | tau(e) | N / tau_c |
|---|---|---|---|---|---|
| 16 | 64 x 50,000 | 53.3 [51.4, 55.2] | 51.1 | 26.3 | 939 |
| 32 | 64 x 100,000 | 221.2 [209.9, 232.9] | 216.7 | 94.2 | 452 |
| 64 | 64 x 400,000 | 1043.5 [994.3, 1102.5] | 988.9 | 346.2 | 383 |
| 128 | 32 x 2,000,000 | 4619.5 [4246.6, 5006.8] | 4264.2 | 1342.5 | 433 |

Fit: A = 0.132 [0.114, 0.152], z = 2.155 [2.113, 2.197], 95% CIs from a bootstrap over replicas.
The interval holds the 2.17 of Liu, Polkovnikov and Sandvik's Kibble-Zurek collapse for
Metropolis; the local slopes between consecutive sizes are 2.05, 2.24 and 2.15. v1 reads L = 16,
32 and 64 only. #50's record on the same kernel (commit 5aac98e, CPU, 16 replicas at L = 64) read
52.1, 231.5 and 875.4, with z = 2.04 [1.96, 2.11]. The pre-#47 kernel read 21.3, 85.7 and 376
(A ~ 0.05): its fixed black-then-white order relayed level flips along domain walls (#50).

### Protocol (#19)

- **Staircase**, the quasi-static arm: the 40-point grid from 3.5 down to 1.5, each plateau
  thermalised for max(500, 20 tau) sweeps and measured for max(2000, 100 tau), with tau from a
  `pilot` run's tau.json at that temperature. The manifest records that file's sha256 as
  `params.schedule.params.tau_sha256`, null when `--tau` is a number. The plateau nearest Tc costs
  1,043 + 5,211 sweeps at L = 16.
- **Ramps**, the dynamic arm: linear cooling at r = tau_c(L) L / (x Tc) sweeps per unit T, so
  x = tau_c(L) / (sweeps per unit of epsilon_L = L (T - Tc) / Tc) at every L. tau_c(L) is the
  pilot record's measurement at L; the fit A L^z stands in only for a size it did not measure,
  and the manifest's `params.schedule.params.rate_source` says which (`measured` or `fit`, #56).
  The ramp runs from 3.5 to 1.5, or over an epsilon_L window: `--eps-start 12 --eps-end -8` sets
  T0 = Tc (1 + 12/L) and T1 = Tc (1 - 8/L), the window ising-ews-v1 scores at every L, and the
  manifest records it as `eps_L`. The slow ramp, x = 0.02, spends 50 tau_c per unit epsilon_L and
  crosses the critical window |epsilon_L| < 1 in 100 tau_c; the fast one, x = 2, crosses it in one
  tau_c, the Kibble-Zurek regime. Sweeps of cooling (equilibration and hold not counted) at the
  committed record's measured tau_c:

  | | L = 16 | L = 32 | L = 64 | L = 128 |
  |---|---|---|---|---|
  | x = 0.02: critical window | 5,326 | 22,120 | 104,351 | 461,954 |
  | x = 0.02: epsilon_L 12 to -8 | 53,258 | 221,199 | 1,043,507 | 4,619,544 |
  | x = 0.02: 3.5 to 1.5 | 37,552 | 311,934 | 2,943,102 | 26,057,880 |
  | x = 2: critical window | 53 | 221 | 1,044 | 4,620 |
  | x = 2: epsilon_L 12 to -8 | 533 | 2,212 | 10,435 | 46,195 |
  | x = 2: 3.5 to 1.5 | 376 | 3,120 | 29,432 | 260,578 |

  Before #50 the rate was L^3.17 / (x Tc), with no A, at nominal x = 0.3 and 30. Under this
  definition those were x = 0.016 and 1.6 on the pre-#47 kernel (A = 0.052), crossing the window
  in about 130 and 1.3 tau_c; on the fixed kernel, where tau_c is about 2.5 times longer, they
  are x = 0.030-0.038 and 3.0-3.8.

  Bursts are saved every 0.005 of cooling, except inside the EWS analysis band epsilon_L in
  [10, -5], where they come 28 per unit epsilon_L at every L (spaced evenly in T, L = 128
  got 3.5 there); the manifest's `schedule.saves` records the gaps and the burst counts.

- **Null runs**, the EWS scorer's negatives: `null --epsilon-L {2,4,6}` for each L, at least 5
  seeds each, holds T = Tc (1 + epsilon_L / L) for 20 tau_c of thermalisation and 700 tau_c of
  measurement, two scored slices at x = 0.02 (310-340 tau_c each), with a stats row every
  tau_c / 50 sweeps and bursts at the slow ramp's in-band cadence. The manifest is a `fixed`
  schedule whose params add `epsilon_L` and `tau_c`, and complete.json judges the run
  thermalised like a staircase plateau. With the committed record one run is 38,346 sweeps at
  L = 16, 159,264 at 32, 751,327 at 64 and 3,326,072 at 128. A size the record did not measure
  is refused, since the manifest could not tell a fitted tau_c from a measured one.

## Early-warning signals

`python/tipping_point/src/tipping_point/ews/` holds the classic indicators (rolling variance,
autocorrelation and skewness after Gaussian detrending), their Kendall-tau trend, surrogate tests,
the scoring protocol (AUC, alarms, lead times, hierarchical bootstrap) and finite-size-scaling
verdicts, on numpy alone. The definitions are ewstools 2.1.3's, pinned by a golden fixture that
`python/tipping_point/tests/fixtures/make_ews_golden.py` regenerates in an isolated environment.
Experiments are pre-registered in `paper/prereg/<id>-v<k>.toml` (start from `TEMPLATE.toml`) and
frozen as git tags; the scorer refuses an unfrozen or edited file, reads tau_c(L) from a pilot
run's `pilot.json`, never from the pre-registration, and scores only runs whose manifest names the
commit the freeze tag points at, from a clean tree: ramps and null runs made against the pilot
record the file pins, and staircases whose dwell was read from the tau.json it pins at that L
(`data.staircase.tau_sha256`), so a staircase given `--tau` as a number is refused.

```sh
uv run python -m tipping_point.ews prereg freeze ising-ews           # tags prereg/ising-ews-v<k>
uv run python -m tipping_point.ews score --prereg paper/prereg/ising-ews-v1.toml
uv run python -m tipping_point.ews calibrate                         # 2,000 nulls x 999 surrogates
```

Results land in `<data dir>/ews/<run_id>/`, and every evaluated cell is appended to
`<data dir>/ews/trials.jsonl`.

### Latent signals

A pre-registration with a `[latents]` section scores a JEPA's readouts instead of |m| (#28). It
names each training it scores, by its run id and the sha256 of `final.pt` and of its untrained twin,
and the scorer takes both through every table, once per checkpoint.

- **The join.** `ising.series(run, "matched", latents=<table>)` joins
  `<run>/latents/<training>-step<step>/`, which `jepa extract` writes, onto the run's saved frames
  with a plain Parquet reader, since `jepa.extract` imports the adapter. The six key columns must
  equal those of the frames the table's own selection rule keeps, so a subsampled staircase reads
  as the frames it holds. A key mismatch, a table without `complete.json`, or the dense cadence, for
  which no latents exist, is refused with an error naming the run and the table.
- **Two more indicators.** `level` (rolling mean) and `slope` (rolling OLS slope per sample) read
  the raw series: the rolling mean of residuals is zero by construction, so `Pipeline` skips the
  detrend for them. A level's rows carry the phase surrogate alone, and its bandwidth axis
  collapses to one cell per window. The extremum of `slope` is the dynamic statistic of a channel
  with no interior peak: finite-size scaling puts the steepest change at a fixed eps_L, 1/nu = 1,
  and a feature anchored to a fixed T gives 1/nu = 0. The harness plants both, and the second is the
  dynamic test's negative control.
- **Rows.** Every row of every table has a `checkpoint` column, the table's folder name, and so does
  each line of `trials.jsonl`; a classic row's is `""`. The file pins each checkpoint's sha256 and
  role, and the result's manifest gives the sha256 of each table's `complete.json` per run.
- **What a file holds.** `indicators.pairs` lists the (observable, indicator) pairs judged by AUC,
  `indicators.direction_by_observable` overrides a direction for one observable, and
  `fss.dynamic.statistics` lists the peak statistics judged by the dynamic test. A primary
  checkpoint and its twin take the whole grid; an ablation's take the headline cell, with
  `scoring.surrogates.n_ablation` surrogates. `prereg.cells` counts that. Static quantities are
  `l2_var_<column>`, `mean_<column>` and `one_minus_ac1_<column>`, read from the matched cadence.
- **The freeze.** A latent file reads the runs of the version its `[baseline]` names, so those runs
  must come from that tag's commit, and each table from the file's own tag and the checkpoint it
  pins: the freeze comes before any extraction on a scored seed. `ising-latent-v1` is not written
  yet; `python/tipping_point/tests/fixtures/ising_latent_draft.toml` is the draft the schema
  accepts, with placeholders for what the exploratory pass sets, and the freeze refuses them all.

Nothing here changes what ising-ews-v1 scores. `harness/ews_v1_golden.py` wrote every row and value
of a v1 scoring on planted data from the commit its freeze tag is cut at, and
`python/tipping_point/tests/ews/test_v1_unchanged.py` holds later commits to them, RNG stream
included.

## JEPA world model (Ising)

`python/tipping_point/src/tipping_point/jepa/` is the JEPA of #27: one per-frame encoder shared by
every L (2x2 patchify, three ConvNeXt blocks with circular depthwise 3x3 convolutions; 0.40M
parameters, a receptive field of 14 sites, below the smallest L), a convolutional predictor
(0.16M parameters, ~50 sites of range) and an EMA target encoder. It hides blocks of frame c (4-8
sites a side, 20-25% of the cells, a one-cell buffer cut from the context) and predicts their
LayerNorm'd target features in frame c + delta, delta 0-7 inside a burst; VICReg variance and
covariance terms on the visible tokens guard against collapse, and a detector with absolute floors
on feature spread watches for it. Bursts are flipped to positive magnetisation, D4 and rolls are
augmentations, and training reads staircase plateaus only, of dedicated simulator seeds: the
seeds 1-10 (ising-ews-v1's 6-10 and its exploratory pilot's 1-5), ramps and null runs are
refused.

```sh
uv run python -m tipping_point.jepa train --L 16 32 64 --train-seeds 101 102 --val-seed 103 \
    --seed 1 --pilot <pilot run>/pilot.json [--held-out] [--prereg paper/prereg/<id>.toml]
uv run python -m tipping_point.jepa resume <training>                # after a killed session
uv run python -m tipping_point.jepa extract <training>/final.pt <simulator run> [--select tc-plus]
uv run python -m tipping_point.jepa probe <training> --fit <val run> --test <scored run>
uv run python -m tipping_point.jepa verify <training> <run>/latents/<set> ...
uv run python -m tipping_point.jepa budget --quota 30                # GPU hours per ISO week
```

- **A training** lives in `<data dir>/jepa/<run_id>/`: the manifest (every input run with its
  `complete.json` sha256, the pilot record's sha256, the pre-registration's id, the seeds, the git
  commit), `train/` and `monitor/` Parquet, `ckpt/step_0000000.pt` (the untrained twin, saved
  before any update), `final.pt`, and `complete.json` with every file's sha256 and the collapse
  verdict. Both models carry a PCA basis fitted on training frames. Every session appends its
  hours to `<data dir>/jepa/gpu_hours.jsonl`.
- **Reproducibility:** one master seed gives each random stream its own generator;
  `use_deterministic_algorithms(True)` is a tripwire and `CUBLAS_WORKSPACE_CONFIG` is set before
  CUDA starts. Two CPU runs of a seed give bitwise-identical losses, and a resumed run continues
  bit for bit. A T4 trains in fp16 autocast, anything else in float32; never bf16, never
  `torch.compile`.
- **The latents table** (#28's input) is written beside the frames, in
  `<simulator run>/latents/<training>-step<step>/`: one row per `configs` row, keyed and ordered
  like `configs` (step, replica, temperature, segment, burst, frame), with `surprise` (the delta = 0
  prediction error over two fixed evaluation masks) and `pc_1`..`pc_8`, float32. Its manifest names
  the checkpoint and the run by sha256; `tipping_point.jepa.extract.read` returns it. `--select`
  subsamples staircases: `tc-plus` keeps the T = 2.269231 plateau whole and one burst in 8 (by
  burst index) at every other plateau, and refuses a staircase whose plateaus save a number of
  bursts 8 does not divide; `tc-only` keeps that plateau alone, for ablation checkpoints;
  ramps and null runs are always read whole. The manifest records the rule, and `verify` checks
  that the table holds exactly the rows it keeps.
- **Probes** read |m|, the energy and L^2 (|m| - <|m|>_T)^2 from frozen embeddings by ridge
  regression, each beside two control tasks: an i.i.d. Gaussian target (R^2 near 0) and the target
  shuffled among the frames of each plateau. Selectivity, R^2 minus the shuffle's, is what the
  embedding holds of a frame beyond its temperature.
- **The untrained twin is the null:** a local encoder's pooled readouts inherit Ising scaling with
  random weights, so learning is claimed only as a margin over each run's step-0 twin, which goes
  through every test. `tipping_point.ews.margins.paired` judges one: the CI of the mean margin
  over model seeds (nested with paired data-bootstrap draws) must exclude 0 in the predicted
  direction, and every seed's margin must carry the predicted sign.
- **Kaggle:** `notebooks/kaggle_ising_runs.py` runs a JSON plan from `notebooks/plans/` as a
  kernel: simulator runs, `jepa train` and `jepa extract`, one queue per T4. It installs the
  commit-stamped wheel from a wheel dataset (no clone and no token), stages the dataset runs a
  plan reads, checks every run with `jepa verify`, and resumes from the previous version's output.
  `--write-kernel <plan> <dir>` writes the folder `kaggle kernels push` takes; `--dry-run-jepa`
  rehearses training, resume and extraction on the CPU. Its docstring is the manual.

The harness trains a planted-stripe toy on the CPU (`harness/jepa_toy.py`: validation error 1.37
to 0.34 in 150 steps; probes cannot tell it from its twin, prediction error can), breaks the loss
to check that the detector fires, and runs the extract round trip on runs the simulator itself
writes (`harness/jepa_ising.py`).

## Running the gate

```sh
uv sync --extra cpu                      # or --extra cu126 on a CUDA machine (the PyTorch build)
uv run python harness/gate.py            # all legs
uv run python harness/gate.py --leg pytest
```

Needs `cargo` on `PATH`. Each leg (cargo fmt, clippy, test; ruff check, format; pyright; pytest) runs on its own and is GREEN only when its exit code is 0 and its own summary evidence is positive. The gate exits nonzero if any leg is RED, and its last line counts any check a leg skipped (`gate GREEN (1 skipped)`); the full record lands in `harness/out/gate-<timestamp>.json`, with each leg's last 200 lines of output and, for cargo test, every `test result` line and every failing test's whole section of output (from its `---- <name> ----` header to the next `failures:` line), each under the binary and exe that printed it, so a red leg can be read without a re-run however early it failed. `cargo test` runs with `--no-fail-fast`, so one failing binary does not hide the ones after it. As it starts and ends, the gate kills strays (`harness/strays.py`; `uv run python harness/strays.py --dry-run` only lists them): processes of `serve`, `store`, `ingest`, `record`, `soak`, `crash-writer` or the `api_python` test that run from a debug target dir of this repository or any of its worktrees, are older than ten minutes and whose owner is gone, as a killed test run leaves them. The owner is looked for up through cargo, bash, sh, cmd and other debug binaries, since a background shell killed without its tree leaves the bash it forked and its cargo running in between. A process with a live owner (a debug `serve --record` started from a terminal) is spared, and so is `api_load`, which holds its server in-process however long `TP_LOAD_SECONDS` runs it. Release builds, where a soak or a recording runs, are never touched. Cargo builds on half the logical cores unless `CARGO_BUILD_JOBS` is set, every leg runs at BelowNormal priority on Windows, and the record notes the cores and how busy the CPU was at the start, so a leg that goes red under load says so. The cargo legs cover only the root workspace's members: `harness/tests/test_gate.py` fails `engine/` or any crate under `harness/` that is outside it, so when `app/src-tauri` lands it must be added to `members` and to that check.
