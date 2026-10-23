//! Property tests on Bybit's sync rule over generated streams, as `book_properties` and
//! `fault_injection` do for Binance's. A Bybit stream is a snapshot, then one delta per message
//! whose update id is one more than the last. A valid stream never resyncs and the book is the
//! exchange's after every delta, never crossed, every level positive. A dropped, duplicated or
//! reordered delta drops the book at the first affected delta, the deltas after it are skipped,
//! and the next snapshot brings the book back to the exchange's. A restart (a whole book at
//! `u = 1`) replaces the book mid-stream. The same streams, written as Bybit's JSON, go through
//! the multi-symbol feed too, so its reading and routing are covered as well.

mod common;

use std::collections::BTreeMap;

use common::{Exchange, all_levels_positive, op};
use engine::market::Scales;
use engine::market::book::Book;
use engine::market::decimal::format_units;
use engine::market::feed::{Event, Feed};
use engine::market::sync::{DepthUpdate, Level, Outcome, ResyncReason, Snapshot};
use engine::venue::Venue;
use engine::venue::bybit::sync::BookSync;
use proptest::prelude::*;

/// A valid Bybit stream from the reference exchange: its snapshot, every delta after it, the
/// exchange's book after each delta, and its snapshot at the end.
#[derive(Debug, Clone)]
struct Stream {
    snapshot: Snapshot,
    deltas: Vec<DepthUpdate>,
    books: Vec<Book>,
    last: Snapshot,
}

/// One delta per message: span 0, so `first = last = u` and `prev = u - 1`, Bybit's chain.
fn stream(min: usize, max: usize) -> impl Strategy<Value = Stream> {
    prop::collection::vec(prop::collection::vec(op(), 1..=4), min..=max).prop_map(|raw| {
        let mut exchange = Exchange::seeded(5_000);
        let snapshot = exchange.snapshot();
        let mut deltas = Vec::with_capacity(raw.len());
        let mut books = Vec::with_capacity(raw.len());
        for ops in raw {
            deltas.push(exchange.step(&ops, 0));
            books.push(exchange.book());
        }
        Stream {
            snapshot,
            deltas,
            books,
            last: exchange.snapshot(),
        }
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    Drop,
    Duplicate,
    Reorder,
}

fn fault() -> impl Strategy<Value = Fault> {
    prop_oneof![
        Just(Fault::Drop),
        Just(Fault::Duplicate),
        Just(Fault::Reorder)
    ]
}

/// Applies the fault at `k` and returns the index of the first delta that cannot follow.
fn inject(fed: &mut Vec<DepthUpdate>, fault: Fault, k: usize) -> usize {
    match fault {
        Fault::Drop => {
            fed.remove(k);
            k
        }
        Fault::Duplicate => {
            fed.insert(k + 1, fed[k].clone());
            k + 1
        }
        Fault::Reorder => {
            fed.swap(k, k + 1);
            k
        }
    }
}

/// `levels` as Bybit's `[price, qty]` decimal strings.
fn pairs(levels: &[Level], scales: Scales) -> String {
    let pairs: Vec<String> = levels
        .iter()
        .map(|l| {
            format!(
                r#"["{}","{}"]"#,
                format_units(l.price, scales.price_decimals),
                format_units(l.qty, scales.qty_decimals)
            )
        })
        .collect();
    format!("[{}]", pairs.join(","))
}

/// A Bybit order-book message for BTCUSDT.
fn message(kind: &str, u: u64, bids: &[Level], asks: &[Level], scales: Scales) -> String {
    format!(
        r#"{{"topic":"orderbook.200.BTCUSDT","type":"{kind}","ts":{u},"data":{{"s":"BTCUSDT","b":{},"a":{},"u":{u},"seq":{u}}},"cts":{u}}}"#,
        pairs(bids, scales),
        pairs(asks, scales)
    )
}

const SCALES: Scales = Scales {
    price_decimals: 1,
    qty_decimals: 3,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// A valid stream never resyncs; after every delta the book is the exchange's, never
    /// crossed, every level positive; the last update id is the last delta's.
    #[test]
    fn a_valid_stream_keeps_exactly_the_exchanges_book(s in stream(1, 40)) {
        let mut sync = BookSync::new();
        prop_assert_eq!(sync.on_snapshot(&s.snapshot).outcome, Outcome::Applied);
        for (i, (delta, reference)) in s.deltas.iter().zip(&s.books).enumerate() {
            prop_assert_eq!(sync.on_event(delta), Outcome::Applied, "delta {}", i);
            let book = sync.book().expect("synced");
            prop_assert_eq!(book, reference, "delta {}", i);
            prop_assert!(!book.is_crossed() && all_levels_positive(book), "delta {}", i);
        }
        prop_assert_eq!(sync.last_id(), s.deltas.last().map(|d| d.last_id));
    }

    /// Joining mid-stream: deltas before the first snapshot are skipped, never applied, and the
    /// snapshot then syncs the book at its update id.
    #[test]
    fn deltas_before_the_first_snapshot_are_skipped(s in stream(2, 40), at_raw in 0usize..40) {
        let at = at_raw % s.deltas.len();
        let mut sync = BookSync::new();
        for delta in &s.deltas[..at] {
            prop_assert_eq!(sync.on_event(delta), Outcome::Skipped);
            prop_assert!(sync.book().is_none());
        }
        let joined = if at == 0 { s.snapshot.clone() } else {
            Snapshot {
                last_update_id: s.deltas[at - 1].last_id,
                bids: s.books[at - 1].bids().collect(),
                asks: s.books[at - 1].asks().collect(),
            }
        };
        prop_assert_eq!(sync.on_snapshot(&joined).outcome, Outcome::Applied);
        for delta in &s.deltas[at..] {
            prop_assert_eq!(sync.on_event(delta), Outcome::Applied);
        }
        prop_assert_eq!(sync.book(), s.books.last());
    }

    /// A lost, repeated or swapped delta drops the book at the first delta that cannot follow,
    /// never applies past it, skips the rest, and the next snapshot restores the exchange's book.
    #[test]
    fn a_gap_drops_the_book_until_the_next_snapshot(
        s in stream(3, 40),
        fault in fault(),
        fault_raw in 0usize..40,
    ) {
        let mut fed = s.deltas.clone();
        let k = 1 + fault_raw % (fed.len() - 2);
        let expect_resync_at = inject(&mut fed, fault, k);
        let mut sync = BookSync::new();
        sync.on_snapshot(&s.snapshot);
        let mut first_resync = None;
        for (i, delta) in fed.iter().enumerate() {
            let outcome = sync.on_event(delta);
            match (first_resync, outcome) {
                (None, Outcome::Applied) => {
                    prop_assert!(i < expect_resync_at, "applied past the fault at {}", i);
                }
                (None, Outcome::NeedsResync(reason)) => {
                    prop_assert!(matches!(reason, ResyncReason::SequenceGap { .. }), "{:?}", reason);
                    first_resync = Some(i);
                }
                (Some(_), Outcome::Skipped) => {}
                (_, other) => prop_assert!(false, "delta {}: unexpected {:?}", i, other),
            }
        }
        prop_assert_eq!(first_resync, Some(expect_resync_at));
        prop_assert!(sync.book().is_none(), "a book survived the gap");
        prop_assert!(!sync.needs_snapshot(), "Bybit's books come on the stream");
        prop_assert_eq!(sync.on_snapshot(&s.last).outcome, Outcome::Applied);
        prop_assert_eq!(sync.book(), s.books.last());
    }

    /// A restart mid-stream: a whole book at `u = 1` replaces whatever the book held, and the
    /// deltas after it chain on from 1.
    #[test]
    fn a_restart_at_u_one_replaces_the_book(s in stream(2, 40), cut_raw in 0usize..40) {
        let cut = 1 + cut_raw % (s.deltas.len() - 1);
        let mut sync = BookSync::new();
        sync.on_snapshot(&s.snapshot);
        for delta in &s.deltas[..cut] {
            sync.on_event(delta);
        }
        let restart = Snapshot {
            last_update_id: 1,
            bids: s.books[cut - 1].bids().collect(),
            asks: s.books[cut - 1].asks().collect(),
        };
        prop_assert_eq!(sync.on_snapshot(&restart).outcome, Outcome::Applied);
        prop_assert_eq!(sync.last_id(), Some(1));
        for (n, delta) in (2u64..).zip(&s.deltas[cut..]) {
            let renumbered = DepthUpdate { first_id: n, last_id: n, prev_last_id: n - 1, ..delta.clone() };
            prop_assert_eq!(sync.on_event(&renumbered), Outcome::Applied);
        }
        prop_assert_eq!(sync.book(), s.books.last());
    }

    /// The same streams as Bybit's JSON through a Bybit feed: the book after every message is
    /// the exchange's, and a dropped message is one sequence gap, then skipped deltas until the
    /// next snapshot message.
    #[test]
    fn the_feed_reads_bybit_messages_into_the_same_books(
        s in stream(3, 30),
        drop_raw in 0usize..30,
    ) {
        let mut feed = Feed::new(Venue::BybitLinear, BTreeMap::from([("BTCUSDT".to_owned(), SCALES)]));
        let first = &s.snapshot;
        let text = message("snapshot", first.last_update_id, &first.bids, &first.asks, SCALES);
        let installed = matches!(feed.on_text(&text), Ok(Event::Depth { outcome: Outcome::Applied, .. }));
        prop_assert!(installed, "the snapshot message installs the book");
        let dropped = 1 + drop_raw % (s.deltas.len() - 2);
        for (i, (delta, reference)) in s.deltas.iter().zip(&s.books).enumerate() {
            if i == dropped {
                continue;
            }
            let text = message("delta", delta.last_id, &delta.bids, &delta.asks, SCALES);
            let event = feed.on_text(&text).expect("a Bybit message");
            let Event::Depth { outcome, .. } = event else {
                return Err(TestCaseError::fail("a depth event"));
            };
            if i < dropped {
                prop_assert_eq!(outcome, Outcome::Applied);
                prop_assert_eq!(feed.book("BTCUSDT"), Some(reference));
            } else if i == dropped + 1 {
                let gap = matches!(outcome, Outcome::NeedsResync(ResyncReason::SequenceGap { .. }));
                prop_assert!(gap, "the delta after the dropped one is a gap: {:?}", outcome);
            } else {
                prop_assert_eq!(outcome, Outcome::Skipped);
            }
        }
        let last = &s.last;
        let text = message("snapshot", last.last_update_id, &last.bids, &last.asks, SCALES);
        feed.on_text(&text).expect("a Bybit snapshot");
        prop_assert_eq!(feed.book("BTCUSDT"), s.books.last());
        let stats = feed.stats();
        prop_assert_eq!((stats.sequence_gaps, stats.resyncs), (1, 1));
        let skipped = u64::try_from(s.deltas.len() - dropped - 2).expect("a small count");
        prop_assert_eq!(stats.skipped, skipped);
        prop_assert_eq!(stats.snapshots, 2);
    }
}
