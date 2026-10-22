//! Replay equals live. The live loop's own path, driven from a script, logs every input and
//! applies it; a replay of that log, a replay of the Parquet rotate makes from it, and a second
//! replay of the log must each raise the same events, after every input, with the same books,
//! and end with the same counters. The committed session has a reconnect, a backpressure drop,
//! an unreadable frame and every other reaction the live loop has; recorded sessions, kept
//! locally, are checked against the transcript the live loop wrote while recording.

use std::error::Error;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use engine::binance::feed::Feed;
use engine::binance::live::{LiveEvent, LiveStats, Log, Observer, handle};
use engine::binance::pipeline::Pipeline;
use engine::binance::recording::{Record, read_records};
use engine::binance::sync::BookSync;
use engine::store::frame::{Kind, Meta};
use engine::store::replay::{replay_log, replay_parquet};
use engine::store::rotate::rotate;
use engine::store::run::{self, BookSample, MarketParams};
use engine::store::segment::{self, LOG_DIR, SegmentReader};
use engine::store::writer::{Entry, LogConfig, LogWriter};
use engine_harness::transcript::{FORMAT, Transcript};

type Outcome = Result<(), Box<dyn Error>>;

/// What the live loop showed its observer: every input with its seq, and the transcript.
#[derive(Default)]
struct Watch {
    inputs: Vec<(u64, Entry)>,
    transcript: Transcript,
}

impl Observer for Watch {
    fn input(&mut self, seq: u64, entry: &Entry, _feed: &Feed) {
        self.inputs.push((seq, entry.clone()));
    }

    fn event(&mut self, event: LiveEvent<'_>) {
        self.transcript.observe(&event);
    }
}

fn fresh_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

/// Every count the inputs drive must match. Connect failures and snapshot fetches are the
/// loop's own business: they change no book and are not in the log.
fn assert_same_end(live: &Pipeline, replayed: &Pipeline, what: &str) {
    let (ours, theirs) = (live.feed(), replayed.feed());
    assert_eq!(ours.stats(), theirs.stats(), "{what}: feed counters");
    for symbol in ours.symbols() {
        assert_eq!(
            ours.book(symbol),
            theirs.book(symbol),
            "{what}: {symbol} book"
        );
        assert_eq!(
            ours.sync(symbol).map(BookSync::last_id),
            theirs.sync(symbol).map(BookSync::last_id),
            "{what}: {symbol} last update id"
        );
    }
    let inputs_only = |stats: &LiveStats| LiveStats {
        connect_failures: 0,
        snapshot_fetches: 0,
        ..*stats
    };
    assert_eq!(
        inputs_only(live.stats()),
        inputs_only(replayed.stats()),
        "{what}: loop counters"
    );
}

#[tokio::test]
async fn a_session_with_every_fault_replays_exactly_as_it_ran() -> Outcome {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../engine/tests/fixtures/synthetic/session-faults.jsonl");
    let mut records = read_records(BufReader::new(File::open(script)?))?.into_iter();
    let Some(Record::Scales { scales }) = records.next() else {
        panic!("the script starts with its scales");
    };
    let dir = fresh_dir("replay-equals-live");
    let run_dir = dir.join(run::MARKET).join("20231114T221320Z-session");
    let params = MarketParams {
        symbols: scales.keys().cloned().collect(),
        scales: scales.clone(),
        streams: vec!["depth@100ms".to_owned(), "aggTrade".to_owned()],
        book_sample: BookSample::default(),
    };
    let manifest = run::create(&run_dir, params)?;
    // Small segments, so the replays also cross segment boundaries.
    let config = LogConfig {
        max_segment_bytes: 2048,
        ..LogConfig::default()
    };
    let mut log = Log::recorded(LogWriter::start(
        &run_dir,
        Meta::for_run(&manifest),
        config,
    )?);

    // Live: exactly what run_live does with each message, network aside.
    let mut pipeline = Pipeline::new(Feed::new(scales));
    let mut live = Watch::default();
    for record in records {
        let message = record
            .into_message()
            .expect("only the first line holds scales");
        handle(message, &mut pipeline, &mut log, &mut live).await?;
    }
    let summary = tokio::task::spawn_blocking(move || log.finish())
        .await??
        .expect("a recorded log");
    assert!(summary.segments >= 3, "{summary:?}");
    run::complete(&run_dir, &summary)?;

    // The loop knew each input's seq as it applied it: the seq it is stored under, across every
    // segment boundary (#53).
    let stored: Vec<(u64, Entry)> = segment::list(&run_dir.join(LOG_DIR))?
        .iter()
        .flat_map(|(_, path)| {
            let mut reader = SegmentReader::open(path).expect("a segment");
            std::iter::from_fn(move || reader.next_frame().expect("a good frame"))
        })
        .filter(|frame| !matches!(frame.kind, Kind::Meta | Kind::SegmentEnd))
        .map(|frame| {
            let entry = Entry {
                kind: frame.kind,
                source_ts_ns: frame.source_ts_ns,
                capture_ts_ns: frame.capture_ts_ns,
                payload: frame.payload,
            };
            (frame.seq, entry)
        })
        .collect();
    assert_eq!(
        live.inputs, stored,
        "the seqs the loop saw are the stored seqs"
    );
    let live = live.transcript;

    // The session does exercise what replay used to miss.
    for reaction in [
        "Reconnect",
        "Backpressure",
        "BadFrame",
        "SequenceGap",
        "SnapshotFailed",
        "Disconnected { stream: Trades",
    ] {
        assert!(live.raised(reaction), "the session never raised {reaction}");
    }

    let mut from_log = Transcript::default();
    let replayed = replay_log(&run_dir, None, &mut |event| from_log.observe(&event))?;
    assert_eq!(live.first_difference(&from_log), None, "log replay");
    assert_same_end(&pipeline, &replayed.pipeline, "log replay");
    assert_eq!(replayed.report.unclosed, None);
    assert!(replayed.report.errors > 0, "bad inputs are reported");

    let rotation = rotate(&run_dir)?;
    assert_eq!(rotation.segments, summary.segments);
    let mut from_parquet = Transcript::default();
    let from_duckdb = replay_parquet(&run_dir, None, &mut |event| from_parquet.observe(&event))?;
    assert_eq!(
        live.first_difference(&from_parquet),
        None,
        "DuckDB frames replay"
    );
    assert_same_end(&pipeline, &from_duckdb.pipeline, "DuckDB frames replay");
    assert_eq!(from_duckdb.report, replayed.report);

    let mut again = Transcript::default();
    replay_log(&run_dir, None, &mut |event| again.observe(&event))?;
    assert_eq!(from_log.first_difference(&again), None, "second log replay");
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// The recordings `record` made, if any: each run's log replays (and its Parquet replays)
/// exactly as the live loop ran while recording it. Skips, and says so, when there are none,
/// and passes over (saying so) a run whose transcript an older harness wrote.
#[test]
fn recorded_sessions_replay_exactly_as_they_ran_live() -> Outcome {
    let recorded = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures/recorded")
        .join(run::MARKET);
    let runs: Vec<PathBuf> = run::runs(&recorded)?
        .into_iter()
        .filter(|run| run.join("live-transcript.json").is_file())
        .collect();
    if runs.is_empty() {
        println!(
            "skipped: no recorded runs with a live transcript in {}",
            recorded.display()
        );
        return Ok(());
    }
    for original in runs {
        let live = Transcript::load(&original.join("live-transcript.json"))?;
        if live.format != FORMAT {
            println!(
                "passed over {}: its transcript is format {}, this harness writes {FORMAT}; record it again",
                original.display(),
                live.format
            );
            continue;
        }
        // A copy, so rotating never leaves Parquet from older code next to the recording.
        let run_dir = fresh_dir("recorded").join(original.file_name().expect("a run id"));
        fs::create_dir_all(run_dir.join(LOG_DIR))?;
        fs::copy(original.join(run::MANIFEST), run_dir.join(run::MANIFEST))?;
        for (_, path) in segment::list(&original.join(LOG_DIR))? {
            fs::copy(
                &path,
                run_dir
                    .join(LOG_DIR)
                    .join(path.file_name().expect("a segment name")),
            )?;
        }

        let mut from_log = Transcript::default();
        replay_log(&run_dir, None, &mut |event| from_log.observe(&event))?;
        assert_eq!(
            live.first_difference(&from_log),
            None,
            "{}: log replay",
            original.display()
        );

        rotate(&run_dir)?;
        let mut from_parquet = Transcript::default();
        replay_parquet(&run_dir, None, &mut |event| from_parquet.observe(&event))?;
        assert_eq!(
            live.first_difference(&from_parquet),
            None,
            "{}: DuckDB replay",
            original.display()
        );
        println!(
            "{}: {} inputs replay exactly as live",
            original.display(),
            live.steps.len()
        );
        fs::remove_dir_all(run_dir.parent().expect("a scratch dir"))?;
    }
    Ok(())
}
