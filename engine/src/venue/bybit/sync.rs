//! Bybit's order-book sync rule for one symbol: a pure, IO-free state machine, fed with
//! snapshots and deltas, and answering with what happened.
//!
//! Bybit sends its book in band (v5 `orderbook.{depth}.{symbol}`, and the same lines in its
//! order-book archive):
//! 1. a `snapshot` replaces the book: on subscribing, whenever Bybit sends one again, and with
//!    `u = 1` after its service restarts;
//! 2. every delta after it carries the next update id, `u + 1`; anything else is loss, so the
//!    book is dropped;
//! 3. there is no REST snapshot to fetch and nothing to buffer: without a book, deltas are skipped
//!    until the next snapshot (live, the loop reconnects to get one).
//!
//! Its deltas reach the rule as [`DepthUpdate`]s with `prev_last_id = u - 1`, so "follows the
//! last one" is the same test Binance's `pu` chain makes.

use crate::market::book::Book;
use crate::market::sync::{DepthUpdate, Outcome, ResyncReason, Snapshot, SnapshotOutcome};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// No trustworthy book; deltas are skipped until a snapshot arrives.
    Waiting,
    /// A snapshot is in place and every delta since followed the one before.
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
    /// Starts out waiting for a snapshot, with no book.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: State::Waiting,
            book: Book::default(),
        }
    }

    /// Feeds one delta.
    pub fn on_event(&mut self, update: &DepthUpdate) -> Outcome {
        match self.state {
            State::Waiting => Outcome::Skipped,
            State::Synced { last_id } if update.prev_last_id == last_id => {
                self.book.apply(update);
                self.state = State::Synced {
                    last_id: update.last_id,
                };
                Outcome::Applied
            }
            State::Synced { last_id } => self.resync(ResyncReason::SequenceGap {
                expected: last_id,
                got: update.prev_last_id,
            }),
        }
    }

    /// Installs a snapshot from the stream, whatever came before it: Bybit's snapshot is the
    /// whole book as of its update id.
    pub fn on_snapshot(&mut self, snapshot: &Snapshot) -> SnapshotOutcome {
        self.book = Book::from_snapshot(snapshot);
        self.state = State::Synced {
            last_id: snapshot.last_update_id,
        };
        SnapshotOutcome {
            outcome: Outcome::Applied,
            drained: Vec::new(),
        }
    }

    /// Declares the book untrustworthy for a reason the stream itself cannot show (frames
    /// dropped under backpressure, a reconnect).
    pub fn invalidate(&mut self, reason: ResyncReason) -> Outcome {
        self.resync(reason)
    }

    /// Never: Bybit's snapshots come on the stream, not from a REST fetch.
    #[must_use]
    pub fn needs_snapshot(&self) -> bool {
        false
    }

    /// True while a snapshot is in place and the update ids have run on without a break.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        matches!(self.state, State::Synced { .. })
    }

    /// The book, present only while it is trustworthy.
    #[must_use]
    pub fn book(&self) -> Option<&Book> {
        match self.state {
            State::Waiting => None,
            State::Synced { .. } => Some(&self.book),
        }
    }

    /// The update id of the last snapshot or delta applied, while synced.
    #[must_use]
    pub fn last_id(&self) -> Option<u64> {
        match self.state {
            State::Synced { last_id } => Some(last_id),
            State::Waiting => None,
        }
    }

    fn resync(&mut self, reason: ResyncReason) -> Outcome {
        self.book = Book::default();
        self.state = State::Waiting;
        Outcome::NeedsResync(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::sync::Level;

    fn delta(u: u64, bid: (i64, i64)) -> DepthUpdate {
        DepthUpdate {
            first_id: u,
            last_id: u,
            prev_last_id: u - 1,
            bids: vec![Level {
                price: bid.0,
                qty: bid.1,
            }],
            asks: vec![],
        }
    }

    fn snapshot(u: u64) -> Snapshot {
        Snapshot {
            last_update_id: u,
            bids: vec![Level { price: 100, qty: 1 }],
            asks: vec![Level { price: 101, qty: 1 }],
        }
    }

    #[test]
    fn deltas_wait_for_a_snapshot_then_chain_on_u() {
        let mut sync = BookSync::new();
        assert_eq!(sync.on_event(&delta(7, (99, 1))), Outcome::Skipped);
        assert!(sync.book().is_none() && !sync.needs_snapshot());
        assert_eq!(sync.on_snapshot(&snapshot(10)).outcome, Outcome::Applied);
        assert_eq!(sync.last_id(), Some(10));
        assert_eq!(sync.on_event(&delta(11, (99, 2))), Outcome::Applied);
        assert_eq!(sync.on_event(&delta(12, (100, 0))), Outcome::Applied);
        let book = sync.book().expect("synced");
        assert_eq!(book.best_bid(), Some(Level { price: 99, qty: 2 }));
        assert_eq!(sync.last_id(), Some(12));
    }

    #[test]
    fn a_skipped_u_drops_the_book_until_the_next_snapshot() {
        let mut sync = BookSync::new();
        sync.on_snapshot(&snapshot(10));
        assert_eq!(
            sync.on_event(&delta(12, (99, 1))),
            Outcome::NeedsResync(ResyncReason::SequenceGap {
                expected: 10,
                got: 11
            })
        );
        assert!(sync.book().is_none() && !sync.is_synced());
        assert_eq!(sync.on_event(&delta(13, (99, 1))), Outcome::Skipped);
        assert_eq!(sync.on_snapshot(&snapshot(20)).outcome, Outcome::Applied);
        assert_eq!(sync.on_event(&delta(21, (98, 1))), Outcome::Applied);
    }

    #[test]
    fn a_new_snapshot_replaces_the_book_even_mid_chain() {
        let mut sync = BookSync::new();
        sync.on_snapshot(&snapshot(10));
        sync.on_event(&delta(11, (90, 5)));
        let restart = Snapshot {
            last_update_id: 1,
            bids: vec![Level { price: 50, qty: 1 }],
            asks: vec![],
        };
        assert_eq!(sync.on_snapshot(&restart).outcome, Outcome::Applied);
        assert_eq!(sync.last_id(), Some(1));
        assert_eq!(
            sync.book().map(|b| b.bids().collect::<Vec<_>>()),
            Some(vec![Level { price: 50, qty: 1 }])
        );
        assert_eq!(sync.on_event(&delta(2, (51, 1))), Outcome::Applied);
    }

    #[test]
    fn invalidating_drops_the_book() {
        let mut sync = BookSync::new();
        sync.on_snapshot(&snapshot(10));
        assert_eq!(
            sync.invalidate(ResyncReason::Reconnect),
            Outcome::NeedsResync(ResyncReason::Reconnect)
        );
        assert!(sync.book().is_none());
        assert_eq!(sync.on_event(&delta(11, (99, 1))), Outcome::Skipped);
    }
}
