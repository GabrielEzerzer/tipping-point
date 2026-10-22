//! What every venue's sync rule shares: price levels and depth changes in integer units, what a
//! rule did with an input, why a book stopped being trustworthy, and [`BookSync`], one symbol's
//! book kept by its venue's rule. The rules themselves live with their venues
//! ([`crate::venue::binance::sync`]).
//!
//! Every rule keeps the same promises: a book is trusted only from a snapshot to the first
//! evidence of loss; a level holds an absolute quantity and a zero removes it; and a book is
//! thrown away for anything the stream itself cannot show (a reconnect, frames dropped under
//! backpressure, an unreadable frame).

use super::book::Book;
use crate::venue::{Venue, binance};

/// One price level in integer units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Level {
    pub price: i64,
    pub qty: i64,
}

/// A depth change in integer units, with the ids its venue chains changes by: the first and last
/// update id it covers and the last id of the change it follows (Binance's `U`, `u` and `pu`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthUpdate {
    pub first_id: u64,
    pub last_id: u64,
    pub prev_last_id: u64,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
}

/// A whole book in integer units, as of update id `last_update_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub last_update_id: u64,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
}

/// What a sync rule did with an input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The book changed (an update applied, or a snapshot installed).
    Applied,
    /// Held until a snapshot arrives.
    Buffered,
    /// Older than what the book already holds; ignored.
    DroppedStale,
    /// The book is no longer trustworthy and was cleared; fetch a fresh snapshot.
    NeedsResync(ResyncReason),
}

/// What installing a snapshot did: its own outcome plus the outcome of every buffered update
/// replayed on top of it, so a caller can count those too.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotOutcome {
    /// `Applied` when the snapshot is in place (synced, or bridging until the covering update
    /// arrives), `NeedsResync` when the buffered updates could not be joined to it, and
    /// `DroppedStale` when the rule was not waiting for a snapshot.
    pub outcome: Outcome,
    /// One `(outcome, crossed)` per buffered update, in the order they were replayed; `crossed`
    /// is whether the book was crossed right after that update.
    pub drained: Vec<(Outcome, bool)>,
}

/// Why a resync became necessary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResyncReason {
    /// An update did not follow the last one applied: something was lost or reordered.
    SequenceGap { expected: u64, got: u64 },
    /// The snapshot predates every update we hold, so the gap can never be bridged.
    SnapshotTooOld { snapshot_id: u64, first_id: u64 },
    /// The reader dropped frames because the channel was full.
    Backpressure,
    /// The socket reconnected, so frames were missed.
    Reconnect,
    /// A depth update could not be read, so it was lost and the update chain is broken.
    BadFrame,
}

/// One symbol's sync state and book, kept by its venue's rule.
#[derive(Debug)]
pub enum BookSync {
    Binance(binance::sync::BookSync),
}

impl BookSync {
    /// No book yet, under `venue`'s rule.
    #[must_use]
    pub fn new(venue: Venue) -> Self {
        match venue {
            Venue::BinanceUsdm => Self::Binance(binance::sync::BookSync::new()),
        }
    }

    /// Feeds one depth update.
    pub fn on_event(&mut self, update: DepthUpdate) -> Outcome {
        match self {
            Self::Binance(sync) => sync.on_event(update),
        }
    }

    /// Installs a snapshot.
    pub fn on_snapshot(&mut self, snapshot: &Snapshot) -> SnapshotOutcome {
        match self {
            Self::Binance(sync) => sync.on_snapshot(snapshot),
        }
    }

    /// Declares the book untrustworthy for a reason the stream itself cannot show.
    pub fn invalidate(&mut self, reason: ResyncReason) -> Outcome {
        match self {
            Self::Binance(sync) => sync.invalidate(reason),
        }
    }

    /// True when a snapshot should be fetched from the venue.
    #[must_use]
    pub fn needs_snapshot(&self) -> bool {
        match self {
            Self::Binance(sync) => sync.needs_snapshot(),
        }
    }

    /// True once the book is kept up to date by the stream.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        match self {
            Self::Binance(sync) => sync.is_synced(),
        }
    }

    /// The book, present only while it is trustworthy.
    #[must_use]
    pub fn book(&self) -> Option<&Book> {
        match self {
            Self::Binance(sync) => sync.book(),
        }
    }

    /// The last update id applied, once synced.
    #[must_use]
    pub fn last_id(&self) -> Option<u64> {
        match self {
            Self::Binance(sync) => sync.last_id(),
        }
    }
}
