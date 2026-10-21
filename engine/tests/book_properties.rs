//! Property tests on the sync state machine over generated valid streams.

mod common;

use common::{Scenario, all_levels_positive, scenario, wild_stream};
use engine::binance::sync::{BookSync, Outcome};
use proptest::prelude::*;

/// Buffers the first `buffered` events, installs `snapshots[snapshot_index]`, then feeds the
/// rest, returning every outcome in the order the events were fed (the snapshot's outcome is
/// returned separately).
fn drive(
    sync: &mut BookSync,
    scenario: &Scenario,
    buffered: usize,
    snapshot_index: usize,
    mut inspect: impl FnMut(&BookSync, usize),
) -> (Outcome, Vec<Outcome>) {
    let mut outcomes = Vec::with_capacity(scenario.len());
    for (i, event) in scenario.events[..buffered].iter().enumerate() {
        outcomes.push(sync.on_event(event.clone()));
        inspect(sync, i);
    }
    let snapshot_outcome = sync
        .on_snapshot(&scenario.snapshots[snapshot_index])
        .outcome;
    inspect(sync, buffered);
    for (i, event) in scenario.events[buffered..].iter().enumerate() {
        outcomes.push(sync.on_event(event.clone()));
        inspect(sync, buffered + i);
    }
    (snapshot_outcome, outcomes)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// A valid stream never asks for a resync, however the snapshot and buffer line up, and
    /// the machine ends synced with exactly the exchange's book.
    #[test]
    fn valid_stream_syncs_without_resync_and_matches_reference(
        scenario in scenario(1, 40),
        buffered_raw in 0usize..=40,
        snapshot_raw in 0usize..=40,
    ) {
        let n = scenario.len();
        let buffered = buffered_raw % (n + 1);
        let snapshot_index = snapshot_raw % n;
        let mut sync = BookSync::new();
        let (snapshot_outcome, outcomes) = drive(&mut sync, &scenario, buffered, snapshot_index, |_, _| {});

        prop_assert_ne!(snapshot_outcome, Outcome::DroppedStale);
        prop_assert!(!matches!(snapshot_outcome, Outcome::NeedsResync(_)), "{snapshot_outcome:?}");
        for (i, outcome) in outcomes.iter().enumerate() {
            prop_assert!(!matches!(outcome, Outcome::NeedsResync(_)), "event {i}: {outcome:?}");
        }
        prop_assert!(sync.is_synced());
        prop_assert_eq!(sync.last_id(), Some(scenario.events[n - 1].last_id));
        prop_assert_eq!(sync.book(), Some(&scenario.final_book));
    }

    /// With the snapshot installed first (so every apply is visible), the applied updates are
    /// exactly the ones from the bridge on, with strictly increasing `u` and each `pu` equal to
    /// the previous applied `u`: the chain is unbroken from the bridge to the end.
    #[test]
    fn applied_ids_are_monotonic_with_an_unbroken_pu_chain(
        scenario in scenario(1, 40),
        snapshot_raw in 0usize..=40,
    ) {
        let n = scenario.len();
        let snapshot_index = snapshot_raw % n;
        let mut sync = BookSync::new();
        let (_, outcomes) = drive(&mut sync, &scenario, 0, snapshot_index, |_, _| {});

        let applied: Vec<&engine::binance::sync::DepthUpdate> = scenario
            .events
            .iter()
            .zip(&outcomes)
            .filter(|(_, outcome)| **outcome == Outcome::Applied)
            .map(|(event, _)| event)
            .collect();
        let applied_ids: Vec<u64> = applied.iter().map(|e| e.last_id).collect();
        let expected: Vec<u64> = scenario.events[snapshot_index..].iter().map(|e| e.last_id).collect();
        prop_assert_eq!(&applied_ids, &expected);
        prop_assert!(applied_ids.windows(2).all(|w| w[0] < w[1]), "{applied_ids:?}");
        for pair in applied.windows(2) {
            prop_assert_eq!(pair[1].prev_last_id, pair[0].last_id);
        }
        let bridge = applied[0];
        let snapshot_id = scenario.snapshots[snapshot_index].last_update_id;
        prop_assert!(bridge.first_id <= snapshot_id && snapshot_id <= bridge.last_id);
    }

    /// Once both sides are populated the book is never crossed, and every level held has a
    /// positive quantity (a zero quantity removed it).
    #[test]
    fn book_is_never_crossed_and_every_level_positive(
        scenario in scenario(1, 40),
        buffered_raw in 0usize..=40,
        snapshot_raw in 0usize..=40,
    ) {
        let n = scenario.len();
        let buffered = buffered_raw % (n + 1);
        let snapshot_index = snapshot_raw % n;
        let mut sync = BookSync::new();
        let mut crossed_at = None;
        let mut nonpositive_at = None;
        drive(&mut sync, &scenario, buffered, snapshot_index, |sync, step| {
            if let Some(book) = sync.book() {
                if book.is_crossed() {
                    crossed_at.get_or_insert(step);
                }
                if !all_levels_positive(book) {
                    nonpositive_at.get_or_insert(step);
                }
            }
        });
        prop_assert_eq!(crossed_at, None);
        prop_assert_eq!(nonpositive_at, None);
    }

    /// Every zero-quantity level in an applied update is absent from the book afterwards, and
    /// every positive one is present with exactly that quantity.
    #[test]
    fn applied_levels_land_exactly(scenario in scenario(1, 40), snapshot_raw in 0usize..=40) {
        let n = scenario.len();
        let snapshot_index = snapshot_raw % n;
        let mut sync = BookSync::new();
        prop_assert_eq!(sync.on_event(scenario.events[0].clone()), Outcome::Buffered);
        sync.on_snapshot(&scenario.snapshots[snapshot_index]);
        for event in &scenario.events[1..] {
            let outcome = sync.on_event(event.clone());
            if outcome != Outcome::Applied {
                continue;
            }
            let book = sync.book().expect("applied means a book");
            let bids: Vec<_> = book.bids().collect();
            let asks: Vec<_> = book.asks().collect();
            for level in &event.bids {
                let held = bids.iter().find(|l| l.price == level.price).map(|l| l.qty);
                prop_assert_eq!(held, (level.qty > 0).then_some(level.qty));
            }
            for level in &event.asks {
                let held = asks.iter().find(|l| l.price == level.price).map(|l| l.qty);
                prop_assert_eq!(held, (level.qty > 0).then_some(level.qty));
            }
        }
    }

    /// Arbitrary levels with a valid id chain never panic and never trigger a resync; a cross
    /// is something the book reports, not something the machine rejects.
    #[test]
    fn arbitrary_levels_never_panic_and_a_cross_is_only_reported(events in wild_stream(40)) {
        let mut sync = BookSync::new();
        let first = &events[0];
        prop_assert_eq!(sync.on_event(first.clone()), Outcome::Buffered);
        let snapshot = engine::binance::sync::Snapshot {
            last_update_id: first.first_id,
            bids: vec![],
            asks: vec![],
        };
        prop_assert_eq!(sync.on_snapshot(&snapshot).outcome, Outcome::Applied);
        prop_assert!(sync.is_synced());
        for event in &events[1..] {
            prop_assert_eq!(sync.on_event(event.clone()), Outcome::Applied);
            let book = sync.book().expect("synced");
            let reported = book.is_crossed();
            let computed = match (book.best_bid(), book.best_ask()) {
                (Some(bid), Some(ask)) => bid.price >= ask.price,
                _ => false,
            };
            prop_assert_eq!(reported, computed);
            prop_assert!(all_levels_positive(book));
        }
    }
}
