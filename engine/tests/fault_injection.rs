//! Fault injection: a dropped, duplicated or reordered update must make the machine ask for a
//! resync at the first affected update, never apply a wrong book, and a fresh snapshot must
//! bring it back to the no-fault reference.

mod common;

use common::scenario;
use engine::market::sync::{DepthUpdate, Outcome, ResyncReason};
use engine::venue::binance::sync::BookSync;
use proptest::prelude::*;

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

/// Applies the fault at index `k` of `fed` and returns the index at which the machine must
/// first report a resync.
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn faults_force_a_resync_then_a_fresh_snapshot_recovers(
        scenario in scenario(3, 40),
        fault in fault(),
        snapshot_raw in 0usize..=40,
        fault_raw in 0usize..=40,
    ) {
        let n = scenario.len();
        // Sync at `snapshot_index`, then fault one of the later events but never the very last
        // one (a drop or swap there would leave nothing after it to reveal the fault).
        let snapshot_index = snapshot_raw % (n - 2);
        let mut fed: Vec<DepthUpdate> = scenario.events[snapshot_index..].to_vec();
        let k = 1 + fault_raw % (fed.len() - 2);
        let expect_resync_at = inject(&mut fed, fault, k);

        let mut sync = BookSync::new();
        prop_assert_eq!(sync.on_snapshot(&scenario.snapshots[snapshot_index]).outcome, Outcome::Applied);

        let mut first_resync = None;
        for (i, event) in fed.iter().enumerate() {
            let outcome = sync.on_event(event.clone());
            match (first_resync, outcome) {
                (None, Outcome::Applied) => {
                    prop_assert!(i < expect_resync_at, "applied past the fault at {i}: {outcome:?}");
                }
                (None, Outcome::NeedsResync(reason)) => {
                    prop_assert!(matches!(reason, ResyncReason::SequenceGap { .. }), "{reason:?}");
                    first_resync = Some(i);
                }
                (Some(_), Outcome::Buffered) => {}
                (_, other) => prop_assert!(false, "event {i}: unexpected {other:?}"),
            }
        }
        prop_assert_eq!(first_resync, Some(expect_resync_at));
        prop_assert!(sync.book().is_none(), "a book survived the fault");

        // Recovery, as the live loop does it: fetch a fresh snapshot until the machine accepts
        // it, then let the stream bridge it. Two rounds suffice: the first drains the junk the
        // fault left in the buffer, the second (if needed) sees only stale updates.
        let fresh = &scenario.snapshots[n - 1];
        let mut accepted = sync.on_snapshot(fresh).outcome;
        if matches!(accepted, Outcome::NeedsResync(_)) {
            accepted = sync.on_snapshot(fresh).outcome;
        }
        prop_assert_eq!(accepted, Outcome::Applied);
        if !sync.is_synced() {
            prop_assert_eq!(sync.on_event(scenario.events[n - 1].clone()), Outcome::Applied);
        }
        prop_assert!(sync.is_synced());
        prop_assert_eq!(sync.book(), Some(&scenario.final_book));
    }

    /// Invalidating for a reason the stream cannot show (reconnect, backpressure) behaves like
    /// a gap: the book is gone until a fresh snapshot bridges again.
    #[test]
    fn invalidation_recovers_like_a_gap(scenario in scenario(2, 40), snapshot_raw in 0usize..=40) {
        let n = scenario.len();
        let snapshot_index = snapshot_raw % (n - 1);
        let mut sync = BookSync::new();
        sync.on_snapshot(&scenario.snapshots[snapshot_index]);
        for event in &scenario.events[snapshot_index..n - 1] {
            prop_assert_eq!(sync.on_event(event.clone()), Outcome::Applied);
        }
        prop_assert_eq!(
            sync.invalidate(ResyncReason::Backpressure),
            Outcome::NeedsResync(ResyncReason::Backpressure)
        );
        prop_assert!(sync.book().is_none());
        prop_assert_eq!(sync.on_event(scenario.events[n - 1].clone()), Outcome::Buffered);
        prop_assert!(sync.needs_snapshot());
        prop_assert_eq!(sync.on_snapshot(&scenario.snapshots[n - 1]).outcome, Outcome::Applied);
        prop_assert!(sync.is_synced());
        prop_assert_eq!(sync.book(), Some(&scenario.final_book));
    }
}
