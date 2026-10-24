//! Fixture replay: the same pipeline the live loop runs, driven from JSON-lines scripts.
//!
//! The committed fixtures are small hand-built synthetic streams. Real recorded sessions are
//! binary-log runs, which `replay_equals_live` in the harness replays against the transcript the
//! live loop wrote while recording them (#71).

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use engine::market::pipeline::Replayed;
use engine::market::recording::{RecordingError, read_records, replay};
use engine::market::sync::BookSync;

fn replay_file(path: &Path) -> Result<Replayed, RecordingError> {
    let records = read_records(BufReader::new(File::open(path)?))?;
    replay(records)
}

fn synthetic(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/synthetic")
        .join(name)
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

/// A Bybit session, its books sent on the stream: every reaction the live path has, each
/// counted once. Its snapshots are frames of the depth stream, never REST fetches.
#[test]
fn a_bybit_session_keeps_its_books_from_the_stream() -> Result<(), RecordingError> {
    let mut replayed = replay_file(&synthetic("bybit-session.jsonl"))?;
    let feed = replayed.pipeline.feed();
    assert_eq!(feed.venue(), engine::venue::Venue::BybitLinear);
    let stats = feed.stats();
    assert_eq!(
        (
            stats.frames,
            stats.depth_events,
            stats.snapshots,
            stats.applied
        ),
        (26, 25, 7, 11)
    );
    assert_eq!(
        (stats.skipped, stats.buffered, stats.dropped_stale),
        (4, 0, 0)
    );
    assert_eq!(
        (
            stats.resyncs,
            stats.sequence_gaps,
            stats.backpressure_resyncs,
            stats.bad_frame_resyncs,
            stats.reconnect_resyncs
        ),
        (5, 1, 2, 2, 0),
        "a gap, a drop that took both books, a bad frame and an unreadable one; the reconnect          found no book left to drop"
    );
    assert_eq!((stats.parse_errors, replayed.report.errors), (3, 3));
    let live = replayed.pipeline.stats();
    assert_eq!((live.depth_reconnects, live.depth_frames_dropped), (1, 2));

    let btc = feed.book("BTCUSDT").expect("BTCUSDT synced again");
    assert_eq!(feed.sync("BTCUSDT").and_then(BookSync::last_id), Some(13));
    let side = |levels: Vec<engine::market::sync::Level>| -> Vec<(i64, i64)> {
        levels.into_iter().map(|l| (l.price, l.qty)).collect()
    };
    assert_eq!(
        side(btc.bids().collect()),
        vec![(500_001, 1100), (500_000, 2000), (499_995, 600)]
    );
    assert_eq!(
        side(btc.asks().collect()),
        vec![(500_002, 1500), (500_010, 3000)]
    );
    let eth = feed.book("ETHUSDT").expect("ETHUSDT synced again");
    assert_eq!(feed.sync("ETHUSDT").and_then(BookSync::last_id), Some(1011));
    assert_eq!(
        side(eth.bids().collect()),
        vec![(300_015, 50), (300_010, 500), (300_005, 150)]
    );
    assert!(
        replayed.pipeline.feed_mut().snapshots_due().is_empty(),
        "a Bybit book never waits on a REST fetch"
    );
    Ok(())
}
