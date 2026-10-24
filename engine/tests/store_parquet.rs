//! Rotation into Parquet, and `DuckDB` over it: the Parquet holds exactly what the log holds, a
//! rotate can be re-run, survives its own crash and runs one at a time, and window queries and
//! replays see the rows they should. Two independent implementations meet here: arrow-rs
//! writes, `DuckDB` reads.

mod runs;

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

use arrow::array::{Array, Int64Array};
use arrow::record_batch::RecordBatch;
use duckdb::Connection;
use engine::market::live::LiveEvent;
use engine::market::pipeline::SnapshotPayload;
use engine::market::sync::BookSync;
use engine::market::{Scales, Symbol};
use engine::store::StoreError;
use engine::store::frame::{Frame, Kind, Meta};
use engine::store::query::{connect, for_each_frame, table_files, window};
use engine::store::replay::{Replay, replay_log, replay_parquet};
use engine::store::rotate::{LOCK_FILE, rotate};
use engine::store::segment::{self, LOG_DIR, recover_run};
use engine::store::writer::LogConfig;
use engine::venue::binance::messages::{Payload, parse_frame, parse_snapshot};
use runs::{fresh_dir, log_frames, synthetic, write_run};

type Outcome = Result<(), Box<dyn Error>>;

/// The script's first message, in nanoseconds.
const T0_NS: i64 = 1_700_000_000_000_000_000;
const MS: i64 = 1_000_000;

/// A `book_samples` row's sample time, symbol, seq, synced flag and top of each side.
type SampleRow = (
    i64,
    String,
    i64,
    bool,
    Option<i64>,
    Option<i64>,
    Option<i64>,
    Option<i64>,
);

/// Small segments, so the session spans several and every table gets several files.
fn session(name: &str) -> (PathBuf, PathBuf, u64) {
    let dir = fresh_dir(name);
    let config = LogConfig {
        max_segment_bytes: 2048,
        ..LogConfig::default()
    };
    let (run_dir, summary) = write_run(&synthetic("session-faults.jsonl"), &dir, config);
    assert!(summary.segments >= 3, "{summary:?}");
    (dir, run_dir, summary.segments)
}

/// A table's files as one `DuckDB` source.
fn source(run_dir: &Path, table: &str) -> String {
    let dir = run_dir.join(table).to_string_lossy().replace('\\', "/");
    format!("read_parquet('{dir}/*.parquet', union_by_name = true)")
}

fn count(conn: &Connection, run_dir: &Path, table: &str) -> Result<i64, duckdb::Error> {
    conn.query_row(
        &format!("SELECT count(*) FROM {}", source(run_dir, table)),
        [],
        |row| row.get(0),
    )
}

fn int64_column(batches: &[RecordBatch], name: &str) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|batch| {
            let column = batch.column_by_name(name).expect("the column");
            let ints = column.as_any().downcast_ref::<Int64Array>().expect("Int64");
            ints.values().to_vec()
        })
        .collect()
}

fn scales_of(log: &[Frame]) -> BTreeMap<Symbol, Scales> {
    serde_json::from_str::<Meta>(&log[0].payload)
        .expect("the run starts with its Meta")
        .scales
}

#[test]
fn rotated_parquet_holds_exactly_what_the_log_holds() -> Outcome {
    let (dir, run_dir, segments) = session("parquet");
    let log = log_frames(&run_dir);
    let rotation = rotate(&run_dir)?;
    assert_eq!(rotation.segments, segments);
    assert_eq!(rotation.written.len() as u64, 5 * segments);
    assert_eq!(
        (rotation.open, rotation.skipped, rotation.cleaned),
        (None, 0, 0)
    );

    // `frames`, read back file by file ORDER BY seq, is the log, field for field.
    let conn = connect()?;
    let mut parquet = Vec::new();
    for path in table_files(&run_dir, "frames")? {
        for_each_frame(&conn, &path, |frame| {
            parquet.push(frame);
            Ok(())
        })?;
    }
    assert_eq!(parquet, log);
    let n = i64::try_from(log.len())?;
    let keys: (i64, i64, i64) = conn.query_row(
        &format!(
            "SELECT count(*), count(DISTINCT seq), max(seq) FROM {}",
            source(&run_dir, "frames")
        ),
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    assert_eq!(
        keys,
        (n, n, n - 1),
        "count(*) = count(DISTINCT seq) = max(seq) + 1"
    );

    // The other tables hold the rows the log implies, nothing lost and nothing invented.
    let scales = scales_of(&log);
    let (mut depth_levels, mut trades, mut snapshot_levels, mut unparsed) = (0, 0, 0, 0);
    for frame in &log {
        match frame.kind {
            Kind::Depth | Kind::Trade => match parse_frame(&frame.payload).map(|f| f.data) {
                Ok(Payload::Depth(raw)) => {
                    match scales.get(&raw.symbol).map(|s| s.depth_update(&raw)) {
                        Some(Ok(update)) => depth_levels += update.bids.len() + update.asks.len(),
                        _ => unparsed += 1,
                    }
                }
                Ok(Payload::AggTrade(raw)) => {
                    match scales.get(&raw.symbol).map(|s| s.agg_trade(&raw)) {
                        Some(Ok(_)) => trades += 1,
                        _ => unparsed += 1,
                    }
                }
                Err(_) => unparsed += 1,
            },
            Kind::Snapshot => {
                let payload: SnapshotPayload = serde_json::from_str(&frame.payload)?;
                let converted = parse_snapshot(&payload.body)
                    .ok()
                    .and_then(|raw| scales[&payload.symbol].snapshot(&raw).ok());
                match converted {
                    Some(snapshot) => snapshot_levels += snapshot.bids.len() + snapshot.asks.len(),
                    None => unparsed += 1,
                }
            }
            Kind::Meta | Kind::Control | Kind::SegmentEnd => {}
        }
    }
    assert!(depth_levels > 0 && trades > 0 && snapshot_levels > 0);
    assert_eq!(
        count(&conn, &run_dir, "depth_updates")?,
        i64::try_from(depth_levels)?
    );
    assert_eq!(count(&conn, &run_dir, "trades")?, trades);
    assert_eq!(
        count(&conn, &run_dir, "snapshots")?,
        i64::try_from(snapshot_levels)?
    );
    assert_eq!(rotation.unparsed, unparsed, "the bad frames of the fixture");
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Book samples, checked against the book a replay holds at each grid instant: the top of each
/// side, the synced flag, and the seq of the last frame applied before the instant.
#[test]
fn book_samples_are_the_book_at_each_instant_of_the_capture_clock() -> Outcome {
    let (dir, run_dir, _) = session("samples");
    rotate(&run_dir)?;
    let log = log_frames(&run_dir);

    let mut expected: Vec<SampleRow> = Vec::new();
    let every = 100 * MS;
    let mut replay = Replay::new(None);
    let mut clock: Option<i64> = None;
    let mut next = 0;
    for frame in &log {
        if matches!(
            frame.kind,
            Kind::Depth | Kind::Trade | Kind::Snapshot | Kind::Control
        ) {
            let captured = frame.capture_ts_ns;
            match clock {
                None => {
                    next = captured.div_euclid(every) * every
                        + if captured % every == 0 { 0 } else { every }
                }
                Some(c) if captured > c => {
                    let feed = replay.pipeline().expect("built by the Meta").feed();
                    while next < captured {
                        for symbol in feed.symbols() {
                            let synced = feed.sync(symbol).is_some_and(BookSync::is_synced);
                            let book = feed.book(symbol).filter(|_| synced);
                            let bid = book.and_then(engine::market::book::Book::best_bid);
                            let ask = book.and_then(engine::market::book::Book::best_ask);
                            expected.push((
                                next,
                                symbol.clone(),
                                i64::try_from(frame.seq)? - 1,
                                synced,
                                bid.map(|l| l.price),
                                bid.map(|l| l.qty),
                                ask.map(|l| l.price),
                                ask.map(|l| l.qty),
                            ));
                        }
                        next += every;
                    }
                }
                Some(_) => {}
            }
            clock = Some(clock.map_or(captured, |c| c.max(captured)));
        }
        replay.push(frame, &mut |_| {})?;
    }
    assert!(expected.len() >= 30, "{} samples", expected.len());
    assert!(expected.iter().any(|row| row.3) && expected.iter().any(|row| !row.3));

    let conn = connect()?;
    let sql = format!(
        "SELECT sample_ts_ns, symbol, seq, synced, bid_px_0, bid_qty_0, ask_px_0, ask_qty_0 FROM {} ORDER BY sample_ts_ns, symbol",
        source(&run_dir, "book_samples")
    );
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get(3)?,
            row.get(4)?,
            row.get(5)?,
            row.get(6)?,
            row.get(7)?,
        ))
    })?;
    let actual: Vec<SampleRow> = rows.collect::<Result<_, _>>()?;
    assert_eq!(actual, expected);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn rotate_can_be_rerun_and_cleans_up_after_its_own_crash() -> Outcome {
    let (dir, run_dir, segments) = session("rerun");
    let first = rotate(&run_dir)?;
    assert!(first.unparsed > 0, "the fixture's bad frames: {first:?}");
    let again = rotate(&run_dir)?;
    assert_eq!(again.written, Vec::<PathBuf>::new());
    assert_eq!(again.skipped, 5 * segments);
    // A rotated segment is read again only for the replay, not for its rows (#68).
    assert_eq!(again.unparsed, 0, "{again:?}");

    // What a rotate killed halfway leaves: a temp file, and an output it never got to.
    let conn = connect()?;
    let rows = count(&conn, &run_dir, "depth_updates")?;
    let stray = run_dir
        .join("frames")
        .join("00000000000000000000.parquet.tmp");
    fs::write(&stray, b"half a file")?;
    let missing = run_dir
        .join("depth_updates")
        .join("00000000000000000000.parquet");
    fs::remove_file(&missing)?;
    let repaired = rotate(&run_dir)?;
    assert_eq!(repaired.cleaned, 1);
    assert_eq!(repaired.written, vec![missing]);
    assert!(!stray.exists());
    assert_eq!(count(&conn, &run_dir, "depth_updates")?, rows);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Every file under `dir` by its path relative to `dir`, with its bytes. Empty files (the lock
/// files) are listed but not read: Windows refuses reads of a file another handle has locked.
fn files(dir: &Path) -> std::io::Result<BTreeMap<PathBuf, Vec<u8>>> {
    let mut found = BTreeMap::new();
    let mut folders = vec![dir.to_owned()];
    while let Some(folder) = folders.pop() {
        for entry in fs::read_dir(&folder)? {
            let path = entry?.path();
            if path.is_dir() {
                folders.push(path);
                continue;
            }
            let bytes = if fs::metadata(&path)?.len() == 0 {
                Vec::new()
            } else {
                fs::read(&path)?
            };
            let relative = path.strip_prefix(dir).expect("found under dir").to_owned();
            found.insert(relative, bytes);
        }
    }
    Ok(found)
}

/// Two rotates of one run would write the same temp files and rename an interleaved one into
/// place for good. While one rotate holds the run, a second is refused at once and leaves every
/// file exactly as it was; once the first lets go, a rotate runs.
#[test]
fn a_second_rotate_of_a_run_is_refused_at_once_and_changes_nothing() -> Outcome {
    let (dir, run_dir, segments) = session("rotate-lock");
    let held = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(run_dir.join(LOCK_FILE))?;
    held.try_lock()?;
    let before = files(&run_dir)?;

    // On a thread with a deadline, so a rotate that waited for the lock fails here, not hangs.
    let (done, outcome) = std::sync::mpsc::channel();
    let second = run_dir.clone();
    std::thread::spawn(move || {
        let refused = rotate(&second).map(|_| ()).map_err(|error| {
            let in_use = matches!(error, StoreError::InUse(_));
            (in_use, error.to_string())
        });
        let _ = done.send(refused);
    });
    let refused = outcome
        .recv_timeout(std::time::Duration::from_secs(30))
        .map_err(|_| "the second rotate waited for the lock instead of failing")?;
    assert!(matches!(&refused, Err((true, _))), "{refused:?}");
    assert_eq!(
        files(&run_dir)?,
        before,
        "the refused rotate touched nothing"
    );

    drop(held);
    assert_eq!(rotate(&run_dir)?.segments, segments);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A crashed run's newest segment has no marker: rotate leaves it for later and replay says
/// so; once recovered, it rotates like the rest.
#[test]
fn an_unclosed_newest_segment_waits_for_recovery() -> Outcome {
    let (dir, run_dir, segments) = session("unclosed");
    let log_dir = run_dir.join(LOG_DIR);
    let newest = segment::list(&log_dir)?.pop().expect("segments");
    let len = fs::metadata(&newest.1)?.len();
    OpenOptions::new()
        .write(true)
        .open(&newest.1)?
        .set_len(len - 10)?;

    let rotation = rotate(&run_dir)?;
    assert_eq!(rotation.open, Some(newest.1.clone()));
    assert_eq!(rotation.segments, segments - 1);
    let name = format!("{:020}.parquet", newest.0);
    for table in ["frames", "book_samples"] {
        assert!(!run_dir.join(table).join(&name).exists());
        assert!(!run_dir.join(table).join(format!("{name}.tmp")).exists());
    }
    let replayed = replay_log(&run_dir, None, &mut |_| {})?;
    assert!(replayed.report.unclosed.is_some(), "{:?}", replayed.report);

    assert!(recover_run(&run_dir)?.is_some());
    let rotation = rotate(&run_dir)?;
    assert_eq!(rotation.segments, segments);
    assert_eq!(rotation.written.len(), 5);
    assert_eq!(
        replay_log(&run_dir, None, &mut |_| {})?.report.unclosed,
        None
    );
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn window_queries_return_the_rows_captured_in_range() -> Outcome {
    let (dir, run_dir, _) = session("window");
    rotate(&run_dir)?;
    let log = log_frames(&run_dir);
    let conn = connect()?;
    let range = T0_NS + 400 * MS..T0_NS + 1200 * MS;

    let seqs = int64_column(&window(&conn, &run_dir, "frames", range.clone())?, "seq");
    let expected: Vec<i64> = log
        .iter()
        .filter(|frame| range.contains(&frame.capture_ts_ns))
        .map(|frame| i64::try_from(frame.seq).expect("a small seq"))
        .collect();
    assert!(!expected.is_empty(), "the window holds frames");
    assert_eq!(seqs, expected);

    let times = int64_column(
        &window(&conn, &run_dir, "book_samples", range.clone())?,
        "sample_ts_ns",
    );
    assert!(!times.is_empty() && times.iter().all(|t| range.contains(t)));
    assert!(times.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!(matches!(
        window(&conn, &run_dir, "nope", range),
        Err(StoreError::UnknownTable(_))
    ));
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A window replay still starts at seq 0, so it ends in the same state as a full one, but it
/// reports only the inputs captured inside the window.
#[test]
fn a_window_replay_reports_only_inside_but_rebuilds_everything() -> Outcome {
    let (dir, run_dir, _) = session("window-replay");
    rotate(&run_dir)?;
    let log = log_frames(&run_dir);
    let range = T0_NS + 400 * MS..T0_NS + 1200 * MS;
    let mut reported = 0;
    let windowed = replay_parquet(&run_dir, Some(range.clone()), &mut |event| {
        if let LiveEvent::Applied { .. } = event {
            reported += 1;
        }
    })?;
    let inside = log
        .iter()
        .filter(|f| {
            !matches!(f.kind, Kind::Meta | Kind::SegmentEnd) && range.contains(&f.capture_ts_ns)
        })
        .count();
    assert!(inside > 0);
    assert_eq!(reported, inside);
    let full = replay_parquet(&run_dir, None, &mut |_| {})?;
    assert_eq!(
        windowed.pipeline.feed().stats(),
        full.pipeline.feed().stats()
    );
    assert_eq!(windowed.report, full.report);
    fs::remove_dir_all(&dir)?;
    Ok(())
}
