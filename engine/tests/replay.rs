//! Fixture replay: the same pipeline the live loop runs, driven from JSON-lines scripts.
//!
//! The committed fixtures are small hand-built synthetic streams. Depth-only JSON-lines
//! recordings from before the binary log, if any, live in the gitignored
//! `harness/fixtures/recorded/` and are replayed only when present.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use engine::binance::pipeline::Replayed;
use engine::binance::recording::{RecordingError, read_records, replay};

fn replay_file(path: &Path) -> Result<Replayed, RecordingError> {
    let records = read_records(BufReader::new(File::open(path)?))?;
    replay(records)
}

fn synthetic(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/synthetic")
        .join(name)
}

fn recorded_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../harness/fixtures/recorded")
}

#[test]
fn synthetic_clean_stream_syncs_with_no_resync() -> Result<(), RecordingError> {
    let replayed = replay_file(&synthetic("depth-clean.jsonl"))?;
    assert_eq!(replayed.report.errors, 0);
    let feed = replayed.pipeline.feed();
    let stats = feed.stats();
    assert_eq!(stats.resyncs, 0);
    assert_eq!(stats.parse_errors, 0);
    assert_eq!(
        (stats.buffered, stats.dropped_stale, stats.applied),
        (2, 1, 2)
    );
    assert_eq!(stats.crossed, 0);

    let sync = feed.sync("BTCUSDT").expect("tracked symbol");
    assert!(sync.is_synced());
    assert_eq!(sync.last_id(), Some(106));
    let book = feed.book("BTCUSDT").expect("synced book");
    assert_eq!(book.depth(), (2, 2));
    assert_eq!(
        book.best_bid().map(|l| (l.price, l.qty)),
        Some((499_990, 500))
    );
    assert_eq!(
        book.best_ask().map(|l| (l.price, l.qty)),
        Some((500_010, 1500))
    );
    assert!(!book.is_crossed());
    Ok(())
}

#[test]
fn synthetic_gap_stream_resyncs_once_and_recovers() -> Result<(), RecordingError> {
    let replayed = replay_file(&synthetic("depth-gap.jsonl"))?;
    assert_eq!(replayed.report.errors, 0);
    let feed = replayed.pipeline.feed();
    let stats = feed.stats();
    assert_eq!((stats.resyncs, stats.sequence_gaps), (1, 1));
    assert_eq!(
        stats.applied, 4,
        "bridge, bridge after the gap, and the two after it"
    );
    assert_eq!(stats.snapshots, 2);

    let sync = feed.sync("BTCUSDT").expect("tracked symbol");
    assert!(sync.is_synced());
    assert_eq!(sync.last_id(), Some(113));
    let book = feed.book("BTCUSDT").expect("synced book");
    assert_eq!(book.depth(), (3, 2));
    assert_eq!(
        book.best_bid().map(|l| (l.price, l.qty)),
        Some((499_990, 500))
    );
    assert_eq!(
        book.best_ask().map(|l| (l.price, l.qty)),
        Some((500_010, 2000))
    );
    Ok(())
}

/// Replays every local recording; skips (and says so) when there is none.
#[test]
fn recorded_fixtures_replay_with_no_unexpected_resync() -> Result<(), RecordingError> {
    let dir = recorded_dir();
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "jsonl"))
            .collect(),
        Err(_) => Vec::new(),
    };
    paths.sort();
    if paths.is_empty() {
        println!("skipped: no recordings in {}", dir.display());
        return Ok(());
    }

    for path in paths {
        let replayed = replay_file(&path)?;
        let feed = replayed.pipeline.feed();
        let stats = feed.stats();
        println!(
            "{}: frames={} applied={} buffered={} stale={} snapshots={} resyncs={} crossed={}",
            path.display(),
            stats.frames,
            stats.applied,
            stats.buffered,
            stats.dropped_stale,
            stats.snapshots,
            stats.resyncs,
            stats.crossed
        );
        assert_eq!(stats.parse_errors, 0, "{}", path.display());
        assert_eq!(stats.resyncs, 0, "{}: unexpected resync", path.display());
        assert_eq!(stats.crossed, 0, "{}: crossed book", path.display());
        assert!(stats.applied > 0, "{}: nothing applied", path.display());
        for symbol in feed.symbols() {
            let sync = feed.sync(symbol).expect("tracked symbol");
            assert!(
                sync.is_synced(),
                "{}: {symbol} did not sync",
                path.display()
            );
            let book = sync.book().expect("synced book");
            assert!(!book.is_crossed(), "{}: {symbol} crossed", path.display());
            assert!(
                book.bids().chain(book.asks()).all(|l| l.qty > 0),
                "{}: {symbol} holds a zero level",
                path.display()
            );
        }
    }
    Ok(())
}
