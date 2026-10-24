//! Binance's order-book sync rule for one symbol: a pure, IO-free state machine, fed with
//! snapshots and depth updates, and answering with what happened.
//!
//! The rules are Binance's USD-M "how to manage a local order book correctly":
//! 1. buffer depth updates until a snapshot arrives;
//! 2. drop buffered updates whose `u < lastUpdateId`;
//! 3. the first update applied must satisfy `U <= lastUpdateId <= u`;
//! 4. every later update's `pu` must equal the previous update's `u`, else resync.

use std::collections::VecDeque;

use crate::market::book::Book;
use crate::market::sync::{DepthUpdate, Outcome, ResyncReason, Snapshot, SnapshotOutcome};

/// Buffered updates beyond this are the oldest, least useful ones; dropping them at worst makes
/// the next snapshot look too old, which the machine already handles with a resync.
const MAX_BUFFERED: usize = 5_000;

#[derive(Debug)]
enum State {
    /// No trustworthy book; updates are buffered until a snapshot arrives.
    Buffering(VecDeque<DepthUpdate>),
    /// Snapshot installed; waiting for the update whose id range covers `snapshot_id`.
    Bridging { snapshot_id: u64 },
    /// Updates apply as long as the `pu` chain holds.
    Synced { last_id: u64 },
}

/// Sync state and book for one symbol.
#[derive(Debug)]
pub struct BookSync {
    state: State,
    book: Book,
}

impl Default for BookSync {
    fn default() -> Self {
        Self::new()
    }
}

impl BookSync {
    /// Starts out buffering, with no book.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: State::Buffering(VecDeque::new()),
            book: Book::default(),
        }
    }

    /// Feeds one depth update.
    pub fn on_event(&mut self, update: DepthUpdate) -> Outcome {
        match &mut self.state {
            State::Buffering(buffer) => {
                if buffer.len() >= MAX_BUFFERED {
                    buffer.pop_front();
                }
                buffer.push_back(update);
                Outcome::Buffered
            }
            State::Bridging { snapshot_id } => {
                let snapshot_id = *snapshot_id;
                if update.last_id < snapshot_id {
                    Outcome::DroppedStale
                } else if update.first_id <= snapshot_id {
                    self.apply(&update)
                } else {
                    let reason = ResyncReason::SnapshotTooOld {
                        snapshot_id,
                        first_id: update.first_id,
                    };
                    self.resync(reason, Some(update))
                }
            }
            State::Synced { last_id } => {
                if update.prev_last_id == *last_id {
                    self.apply(&update)
                } else {
                    let reason = ResyncReason::SequenceGap {
                        expected: *last_id,
                        got: update.prev_last_id,
                    };
                    self.resync(reason, Some(update))
                }
            }
        }
    }

    /// Installs a snapshot and replays whatever was buffered on top of it.
    pub fn on_snapshot(&mut self, snapshot: &Snapshot) -> SnapshotOutcome {
        let State::Buffering(buffer) = &mut self.state else {
            return SnapshotOutcome {
                outcome: Outcome::DroppedStale,
                drained: Vec::new(),
            };
        };
        let buffered = std::mem::take(buffer);
        self.book = Book::from_snapshot(snapshot);
        self.state = State::Bridging {
            snapshot_id: snapshot.last_update_id,
        };
        let drained: Vec<(Outcome, bool)> = buffered
            .into_iter()
            .map(|u| {
                let outcome = self.on_event(u);
                (outcome, self.book().is_some_and(Book::is_crossed))
            })
            .collect();
        let first_failure = drained.iter().find_map(|(o, _)| match o {
            Outcome::NeedsResync(reason) => Some(*reason),
            _ => None,
        });
        SnapshotOutcome {
            outcome: first_failure.map_or(Outcome::Applied, Outcome::NeedsResync),
            drained,
        }
    }

    /// Declares the book untrustworthy for a reason the stream itself cannot show (frames
    /// dropped under backpressure, a reconnect). Clears the book and the buffer.
    pub fn invalidate(&mut self, reason: ResyncReason) -> Outcome {
        self.resync(reason, None)
    }

    /// True when a snapshot should be fetched: buffering, and at least one update has arrived
    /// so the stream is known to be open (a snapshot fetched before that cannot be bridged).
    #[must_use]
    pub fn needs_snapshot(&self) -> bool {
        matches!(&self.state, State::Buffering(buffer) if !buffer.is_empty())
    }

    /// True once an update has been applied on top of the snapshot.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        matches!(self.state, State::Synced { .. })
    }

    /// The book, present only while it is trustworthy (snapshot installed, chain unbroken).
    #[must_use]
    pub fn book(&self) -> Option<&Book> {
        match self.state {
            State::Buffering(_) => None,
            State::Bridging { .. } | State::Synced { .. } => Some(&self.book),
        }
    }

    /// The `u` of the last applied update, once synced.
    #[must_use]
    pub fn last_id(&self) -> Option<u64> {
        match self.state {
            State::Synced { last_id } => Some(last_id),
            _ => None,
        }
    }

    fn apply(&mut self, update: &DepthUpdate) -> Outcome {
        self.book.apply(update);
        self.state = State::Synced {
            last_id: update.last_id,
        };
        Outcome::Applied
    }

    fn resync(&mut self, reason: ResyncReason, carry: Option<DepthUpdate>) -> Outcome {
        self.book = Book::default();
        self.state = State::Buffering(carry.into_iter().collect());
        Outcome::NeedsResync(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::sync::Level;

    fn update(first: u64, last: u64, prev: u64, bid: (i64, i64)) -> DepthUpdate {
        DepthUpdate {
            first_id: first,
            last_id: last,
            prev_last_id: prev,
            bids: vec![Level {
                price: bid.0,
                qty: bid.1,
            }],
            asks: vec![],
        }
    }

    fn snapshot(id: u64) -> Snapshot {
        Snapshot {
            last_update_id: id,
            bids: vec![Level { price: 100, qty: 1 }],
            asks: vec![Level { price: 101, qty: 1 }],
        }
    }

    #[test]
    fn buffers_until_snapshot_then_bridges_and_syncs() {
        let mut sync = BookSync::new();
        assert!(!sync.needs_snapshot());
        assert_eq!(sync.on_event(update(1, 3, 0, (99, 1))), Outcome::Buffered);
        assert!(sync.needs_snapshot());
        assert_eq!(sync.on_event(update(4, 6, 3, (98, 2))), Outcome::Buffered);
        assert_eq!(sync.on_event(update(7, 9, 6, (97, 3))), Outcome::Buffered);
        assert!(sync.book().is_none());

        // Snapshot id 5 sits inside the second update: first is stale, second bridges.
        assert_eq!(sync.on_snapshot(&snapshot(5)).outcome, Outcome::Applied);
        assert!(sync.is_synced());
        assert_eq!(sync.last_id(), Some(9));
        let book = sync.book().expect("synced book");
        assert_eq!(book.depth(), (3, 1));
        assert!(book.bids().all(|l| l.price != 99));

        assert_eq!(sync.on_event(update(10, 10, 9, (97, 0))), Outcome::Applied);
        assert_eq!(sync.book().expect("book").depth(), (2, 1));
    }

    #[test]
    fn snapshot_newer_than_buffer_waits_for_the_bridge() {
        let mut sync = BookSync::new();
        assert_eq!(sync.on_event(update(1, 3, 0, (99, 1))), Outcome::Buffered);
        assert_eq!(sync.on_snapshot(&snapshot(10)).outcome, Outcome::Applied);
        assert!(!sync.is_synced());
        assert!(sync.book().is_some());
        assert_eq!(
            sync.on_event(update(4, 6, 3, (98, 1))),
            Outcome::DroppedStale
        );
        assert_eq!(sync.on_event(update(7, 12, 6, (98, 1))), Outcome::Applied);
        assert!(sync.is_synced());
    }

    #[test]
    fn gap_after_snapshot_asks_for_a_resync_and_keeps_the_update() {
        let mut sync = BookSync::new();
        assert_eq!(
            sync.on_event(update(20, 22, 19, (99, 1))),
            Outcome::Buffered
        );
        assert_eq!(
            sync.on_snapshot(&snapshot(10)).outcome,
            Outcome::NeedsResync(ResyncReason::SnapshotTooOld {
                snapshot_id: 10,
                first_id: 20
            })
        );
        assert!(sync.book().is_none());
        assert!(
            sync.needs_snapshot(),
            "the gap update is kept for the next snapshot"
        );
        assert_eq!(sync.on_snapshot(&snapshot(21)).outcome, Outcome::Applied);
        assert!(sync.is_synced());
    }

    #[test]
    fn sequence_gap_clears_the_book() {
        let mut sync = BookSync::new();
        assert_eq!(sync.on_event(update(1, 3, 0, (99, 1))), Outcome::Buffered);
        assert_eq!(sync.on_snapshot(&snapshot(2)).outcome, Outcome::Applied);
        assert_eq!(
            sync.on_event(update(7, 9, 6, (97, 3))),
            Outcome::NeedsResync(ResyncReason::SequenceGap {
                expected: 3,
                got: 6
            })
        );
        assert!(sync.book().is_none());
        assert!(!sync.is_synced());
    }

    #[test]
    fn invalidate_drops_everything_and_stale_snapshots_are_ignored() {
        let mut sync = BookSync::new();
        assert_eq!(sync.on_event(update(1, 3, 0, (99, 1))), Outcome::Buffered);
        assert_eq!(sync.on_snapshot(&snapshot(2)).outcome, Outcome::Applied);
        assert_eq!(
            sync.on_snapshot(&snapshot(2)).outcome,
            Outcome::DroppedStale
        );
        assert_eq!(
            sync.invalidate(ResyncReason::Reconnect),
            Outcome::NeedsResync(ResyncReason::Reconnect)
        );
        assert!(!sync.needs_snapshot(), "no update seen since the reconnect");
        assert_eq!(
            sync.on_event(update(50, 51, 49, (99, 1))),
            Outcome::Buffered
        );
        assert!(sync.needs_snapshot());
    }

    #[test]
    fn buffer_is_bounded() {
        let mut sync = BookSync::new();
        for i in 0..(MAX_BUFFERED as u64 + 10) {
            sync.on_event(update(i, i, i.wrapping_sub(1), (1, 1)));
        }
        let State::Buffering(buffer) = &sync.state else {
            panic!("still buffering");
        };
        assert_eq!(buffer.len(), MAX_BUFFERED);
        assert_eq!(buffer.front().map(|u| u.first_id), Some(10));
    }
}
