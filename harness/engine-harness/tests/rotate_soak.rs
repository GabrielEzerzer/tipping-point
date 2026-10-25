//! Rotate on one full segment (#52). The rotate tests use small segments; this writes one
//! segment of the size the log writer closes at, 1 GiB, of synthetic Binance frames generated on
//! the fly (two symbols whose books sync, a diff every 100 ms each, a trade every few diffs),
//! rotates it into Parquet, and writes how long that took and the process's peak memory to
//! `harness/out/rotate-soak-<ms>.json` (gitignored). The run itself goes under cargo's scratch
//! dir and is deleted. A segment takes minutes and a few GiB of disk, so this runs only with
//! `TP_ROTATE_SOAK` set, in release, as `TP_ARCHIVE_DAYS` gates the real-day imports:
//! `TP_ROTATE_SOAK=1 cargo test --release -p engine-harness --test rotate_soak -- --nocapture`.
//! `TP_ROTATE_SOAK_MB` sets another segment size, for a quicker comparison.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use engine::market::Scales;
use engine::market::decimal::format_units;
use engine::market::pipeline::SnapshotPayload;
use engine::store::frame::{HEADER_LEN, Kind, Meta, TRAILER_LEN};
use engine::store::now_ns;
use engine::store::rotate::{TABLES, rotate};
use engine::store::run::{self, BookSample, Clock, DataSource, MarketParams};
use engine::store::segment::{self, LOG_DIR};
use engine::store::writer::{Entry, LogConfig, LogWriter};
use engine::venue::Venue;
use engine_harness::api::repo;
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::json;

type Outcome = Result<(), Box<dyn Error>>;

/// Where the capture clock starts (2027-01-15T08:00:00Z).
const T0_MS: u64 = 1_800_000_000_000;
/// A diff every 50 ms, the two symbols in turn, so each has its 100 ms stream.
const STEP_MS: u64 = 50;

/// Deletes the run, a GiB and more, however the test ends.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// xorshift64, so every run writes the same frames.
struct Random(u64);

impl Random {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// One symbol's streams: its name, scales, the mid its levels sit around (in ticks), and the
/// last update id it sent.
struct Stream {
    symbol: &'static str,
    scales: Scales,
    mid: i64,
    last_id: u64,
    trades: u64,
}

impl Stream {
    fn new(symbol: &'static str, price_decimals: u32, mid: i64) -> Self {
        Self {
            symbol,
            scales: Scales {
                price_decimals,
                qty_decimals: 3,
            },
            mid,
            last_id: 1000,
            trades: 0,
        }
    }

    fn pair(&self, price: i64, qty: i64) -> String {
        format!(
            "[\"{}\",\"{}\"]",
            format_units(price, self.scales.price_decimals),
            format_units(qty, self.scales.qty_decimals)
        )
    }

    /// `count` levels on one side of the mid (`side` -1 for bids, +1 for asks); a fifth of them
    /// remove their level.
    fn levels(&self, random: &mut Random, side: i64, count: u64) -> String {
        let levels: Vec<String> = (0..count)
            .map(|_| {
                let ticks = i64::try_from(1 + random.below(400)).unwrap_or(1);
                let qty = if random.below(5) == 0 {
                    0
                } else {
                    i64::try_from(1 + random.below(50_000)).unwrap_or(1)
                };
                self.pair(self.mid + side * ticks, qty)
            })
            .collect();
        format!("[{}]", levels.join(","))
    }

    /// The next diff, chained onto the last: 5 to 120 levels a side.
    fn depth(&mut self, random: &mut Random, t_ms: u64) -> String {
        let first = self.last_id + 1;
        let last = first + random.below(8);
        let (bid_count, ask_count) = (5 + random.below(116), 5 + random.below(116));
        let bids = self.levels(random, -1, bid_count);
        let asks = self.levels(random, 1, ask_count);
        let frame = format!(
            r#"{{"stream":"{}@depth@100ms","data":{{"e":"depthUpdate","E":{t_ms},"T":{},"s":"{}","U":{first},"u":{last},"pu":{},"b":{bids},"a":{asks}}}}}"#,
            self.symbol.to_lowercase(),
            t_ms - 2,
            self.symbol,
            self.last_id
        );
        self.last_id = last;
        frame
    }

    /// A REST snapshot the first buffered diff bridges onto: 500 levels a side, one a tick.
    fn snapshot(&self, random: &mut Random, t_ms: u64) -> Result<String, serde_json::Error> {
        let side = |random: &mut Random, side: i64| -> String {
            let levels: Vec<String> = (1..=500)
                .map(|ticks| {
                    let qty = i64::try_from(1 + random.below(50_000)).unwrap_or(1);
                    self.pair(self.mid + side * ticks, qty)
                })
                .collect();
            format!("[{}]", levels.join(","))
        };
        let body = format!(
            r#"{{"lastUpdateId":1001,"E":{t_ms},"T":{t_ms},"bids":{},"asks":{}}}"#,
            side(random, -1),
            side(random, 1)
        );
        serde_json::to_string(&SnapshotPayload {
            symbol: self.symbol.to_owned(),
            body,
        })
    }

    fn trade(&mut self, random: &mut Random, t_ms: u64) -> String {
        self.trades += 1;
        let id = self.trades;
        let price = self.mid + i64::try_from(random.below(5)).unwrap_or(0) - 2;
        let qty = i64::try_from(1 + random.below(2_000)).unwrap_or(1);
        format!(
            r#"{{"stream":"{}@aggTrade","data":{{"e":"aggTrade","E":{t_ms},"a":{id},"s":"{}","p":"{}","q":"{}","f":{},"l":{},"T":{t_ms},"m":{}}}}}"#,
            self.symbol.to_lowercase(),
            self.symbol,
            format_units(price, self.scales.price_decimals),
            format_units(qty, self.scales.qty_decimals),
            id * 3,
            id * 3 + 2,
            random.below(2) == 0
        )
    }
}

fn entry(kind: Kind, t_ms: u64, payload: String) -> Entry {
    let ns = i64::try_from(t_ms).unwrap_or(i64::MAX) * 1_000_000;
    Entry {
        kind,
        source_ts_ns: ns,
        capture_ts_ns: ns,
        payload,
    }
}

/// The process's peak resident memory so far, in bytes: the peak working set on Windows,
/// `VmHWM` on Linux; `None` when it cannot be read.
fn peak_memory() -> Option<u64> {
    if cfg!(windows) {
        let script = format!("(Get-Process -Id {}).PeakWorkingSet64", std::process::id());
        let output = Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .output()
            .ok()?;
        String::from_utf8_lossy(&output.stdout).trim().parse().ok()
    } else {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }
}

/// Writes one segment of at least `bytes` to a new run and closes it. Returns the entries
/// appended.
fn write_segment(run_dir: &Path, bytes: u64) -> Result<u64, Box<dyn Error>> {
    // 50 000.0 and 3 000.00, in ticks.
    let mut streams = [
        Stream::new("BTCUSDT", 1, 500_000),
        Stream::new("ETHUSDT", 2, 300_000),
    ];
    let params = MarketParams {
        venue: Venue::BinanceUsdm,
        symbols: streams.iter().map(|s| s.symbol.to_owned()).collect(),
        scales: streams
            .iter()
            .map(|s| (s.symbol.to_owned(), s.scales))
            .collect::<BTreeMap<_, _>>(),
        streams: Venue::BinanceUsdm
            .streams()
            .iter()
            .map(|&s| s.to_owned())
            .collect(),
        book_sample: BookSample::default(),
        clock: Clock::Capture,
        source: DataSource::Live,
    };
    let manifest = run::create(run_dir, params)?;
    // Only this loop ends the segment, once it holds `bytes`, and `finish` closes it: no hour
    // boundary of the wall clock may cut it short, and the writer, which counts the Meta too,
    // must not start another for the last entry.
    let config = LogConfig {
        max_segment_bytes: u64::MAX,
        rotate_every: Duration::from_hours(24 * 365 * 100),
        ..LogConfig::default()
    };
    let mut writer = LogWriter::start(run_dir, Meta::for_run(&manifest), config)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let mut random = Random(0x9E37_79B9_7F4A_7C15);
    let (mut written, mut appended, mut t_ms) = (0u64, 0u64, T0_MS);
    runtime.block_on(async {
        // Each book buffers its first diff, then its snapshot bridges it.
        let mut opening = Vec::new();
        for stream in &mut streams {
            opening.push(entry(Kind::Depth, t_ms, stream.depth(&mut random, t_ms)));
            t_ms += STEP_MS;
        }
        for stream in &streams {
            opening.push(entry(
                Kind::Snapshot,
                t_ms,
                stream.snapshot(&mut random, t_ms)?,
            ));
        }
        // Then the symbols in turn, every seventh entry a trade instead of a diff.
        let mut turn = 0u64;
        let mut entries = opening.into_iter();
        while written < bytes {
            let next = entries.next().unwrap_or_else(|| {
                turn += 1;
                let stream = &mut streams[usize::from(turn.is_multiple_of(2))];
                if turn.is_multiple_of(7) {
                    entry(Kind::Trade, t_ms, stream.trade(&mut random, t_ms))
                } else {
                    t_ms += STEP_MS;
                    entry(Kind::Depth, t_ms, stream.depth(&mut random, t_ms))
                }
            });
            written += (HEADER_LEN + next.payload.len() + TRAILER_LEN) as u64;
            writer.append(next).await?;
            appended += 1;
        }
        Ok::<(), Box<dyn Error>>(())
    })?;
    writer.finish()?;
    Ok(appended)
}

/// The rows of one Parquet file, from its footer.
fn parquet_rows(path: &Path) -> Result<i64, Box<dyn Error>> {
    let reader = SerializedFileReader::new(File::open(path)?)?;
    Ok(reader.metadata().file_metadata().num_rows())
}

#[test]
fn a_full_segment_rotates_and_its_time_and_memory_are_recorded() -> Outcome {
    if std::env::var_os("TP_ROTATE_SOAK").is_none() {
        println!("skipped: set TP_ROTATE_SOAK to rotate a full synthetic segment (minutes, GiBs)");
        return Ok(());
    }
    let megabytes: u64 = match std::env::var("TP_ROTATE_SOAK_MB") {
        Ok(value) => value.parse()?,
        Err(_) => 1024,
    };
    let scratch = Scratch(
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("rotate-soak-{}", std::process::id())),
    );
    let _ = fs::remove_dir_all(&scratch.0);
    let run_dir = scratch.0.join(run::MARKET).join("20270115T080000Z-soak");

    let started = Instant::now();
    let appended = write_segment(&run_dir, megabytes << 20)?;
    let generate = started.elapsed();
    let segments = segment::list(&run_dir.join(LOG_DIR))?;
    assert_eq!(segments.len(), 1, "one segment, closed by finish");
    let segment_bytes = fs::metadata(&segments[0].1)?.len();
    let peak_before = peak_memory();

    let started = Instant::now();
    let rotation = rotate(&run_dir)?;
    let rotate_time = started.elapsed();
    let peak_after = peak_memory();

    assert_eq!(rotation.segments, 1, "{rotation:?}");
    assert_eq!(rotation.written.len(), TABLES.len(), "{rotation:?}");
    assert_eq!((rotation.unparsed, rotation.open.as_ref()), (0, None));
    let mut tables = serde_json::Map::new();
    for path in &rotation.written {
        let table = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .ok_or("a table folder")?
            .to_owned();
        let rows = parquet_rows(path)?;
        tables.insert(
            table,
            json!({ "rows": rows, "bytes": fs::metadata(path)?.len() }),
        );
    }
    assert_eq!(
        tables["frames"]["rows"],
        json!(appended + 2),
        "every entry, the Meta and the end marker"
    );

    #[allow(clippy::cast_precision_loss)]
    let mib_per_s = segment_bytes as f64 / f64::from(1 << 20) / rotate_time.as_secs_f64();
    let report = json!({
        "segment_bytes": segment_bytes,
        "entries": appended,
        "generate_s": generate.as_secs_f64(),
        "rotate_s": rotate_time.as_secs_f64(),
        "rotate_mib_per_s": mib_per_s,
        "peak_memory_bytes_before_rotate": peak_before,
        "peak_memory_bytes_after_rotate": peak_after,
        "tables": tables,
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "engine": engine::version(),
    });
    let out = repo().join("harness/out");
    fs::create_dir_all(&out)?;
    let path = out.join(format!("rotate-soak-{}.json", now_ns() / 1_000_000));
    fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    println!("written to {}", path.display());
    Ok(())
}
