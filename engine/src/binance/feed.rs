//! The multi-symbol feed: raw frame text in, book outcomes out. Pure, so the live loop, a
//! recording and a synthetic test stream all drive exactly the same code.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use super::book::Book;
use super::decimal::DecimalError;
use super::messages::{AggTrade, MessageError, Payload, parse_frame, parse_snapshot};
use super::sync::{BookSync, Outcome, ResyncReason};
use super::{Scales, Symbol};

/// Counters over everything the feed has seen.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub frames: u64,
    pub parse_errors: u64,
    pub depth_events: u64,
    pub trades: u64,
    pub applied: u64,
    pub buffered: u64,
    pub dropped_stale: u64,
    pub snapshots: u64,
    /// Times a trustworthy book was thrown away, by reason.
    pub resyncs: u64,
    pub sequence_gaps: u64,
    pub snapshots_too_old: u64,
    pub backpressure_resyncs: u64,
    pub reconnect_resyncs: u64,
    /// Applied updates that left the book crossed (logged, never asserted).
    pub crossed: u64,
}

/// What one frame turned out to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Depth {
        symbol: Symbol,
        outcome: Outcome,
        crossed: bool,
    },
    Trade(AggTrade),
}

/// Why a frame or snapshot could not be fed.
#[derive(Debug, Error)]
pub enum FeedError {
    #[error(transparent)]
    Message(#[from] MessageError),
    #[error(transparent)]
    Decimal(#[from] DecimalError),
    #[error("no scales for symbol {0:?}")]
    UnknownSymbol(String),
}

/// One [`BookSync`] per symbol, plus the scales to parse with and the running counters.
#[derive(Debug)]
pub struct Feed {
    scales: BTreeMap<Symbol, Scales>,
    syncs: BTreeMap<Symbol, BookSync>,
    pending_snapshot: BTreeSet<Symbol>,
    stats: Stats,
}

impl Feed {
    /// A feed for exactly the symbols given scales.
    #[must_use]
    pub fn new(scales: BTreeMap<Symbol, Scales>) -> Self {
        let syncs = scales
            .keys()
            .cloned()
            .map(|s| (s, BookSync::new()))
            .collect();
        Self {
            scales,
            syncs,
            pending_snapshot: BTreeSet::new(),
            stats: Stats::default(),
        }
    }

    /// The symbols this feed tracks.
    pub fn symbols(&self) -> impl Iterator<Item = &Symbol> {
        self.scales.keys()
    }

    /// The scales a symbol is parsed with.
    #[must_use]
    pub fn scales(&self, symbol: &str) -> Option<Scales> {
        self.scales.get(symbol).copied()
    }

    /// The sync machine for a symbol.
    #[must_use]
    pub fn sync(&self, symbol: &str) -> Option<&BookSync> {
        self.syncs.get(symbol)
    }

    /// A symbol's book, only while it is trustworthy.
    #[must_use]
    pub fn book(&self, symbol: &str) -> Option<&Book> {
        self.syncs.get(symbol).and_then(BookSync::book)
    }

    /// The running counters.
    #[must_use]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Feeds the text of one stream frame.
    ///
    /// # Errors
    /// Returns an error when the frame cannot be parsed or names a symbol without scales; the
    /// parse-error counter is bumped either way.
    pub fn on_text(&mut self, text: &str) -> Result<Event, FeedError> {
        self.stats.frames += 1;
        let frame = match parse_frame(text) {
            Ok(frame) => frame,
            Err(error) => {
                self.stats.parse_errors += 1;
                return Err(error.into());
            }
        };
        match frame.data {
            Payload::Depth(raw) => {
                self.stats.depth_events += 1;
                let scales = self.scales_for(&raw.symbol)?;
                let update = scales.depth_update(&raw)?;
                let sync = self.syncs.entry(raw.symbol.clone()).or_default();
                let outcome = sync.on_event(update);
                let crossed = sync.book().is_some_and(Book::is_crossed);
                self.count(outcome, crossed);
                Ok(Event::Depth {
                    symbol: raw.symbol,
                    outcome,
                    crossed,
                })
            }
            Payload::AggTrade(raw) => {
                self.stats.trades += 1;
                let scales = self.scales_for(&raw.symbol)?;
                Ok(Event::Trade(scales.agg_trade(&raw)?))
            }
        }
    }

    /// Feeds the body of a REST snapshot for a symbol.
    ///
    /// # Errors
    /// Returns an error when the body cannot be parsed or the symbol has no scales.
    pub fn on_snapshot(&mut self, symbol: &str, body: &str) -> Result<Outcome, FeedError> {
        self.pending_snapshot.remove(symbol);
        self.stats.snapshots += 1;
        let scales = self.scales_for(symbol)?;
        let snapshot = scales.snapshot(&parse_snapshot(body)?)?;
        let sync = self.syncs.entry(symbol.to_owned()).or_default();
        let installed = sync.on_snapshot(&snapshot);
        let crossed = sync.book().is_some_and(Book::is_crossed);
        for outcome in installed.drained {
            self.count(outcome, crossed);
        }
        // The install itself is already counted under `snapshots`; only its failure counts.
        if let Outcome::NeedsResync(_) = installed.outcome {
            self.count(installed.outcome, false);
        }
        Ok(installed.outcome)
    }

    /// Records that a requested snapshot never arrived, so it may be requested again.
    pub fn snapshot_failed(&mut self, symbol: &str) {
        self.pending_snapshot.remove(symbol);
    }

    /// Symbols that need a snapshot and do not already have one on the way. Each is marked
    /// pending until [`Self::on_snapshot`] or [`Self::snapshot_failed`] clears it.
    pub fn snapshots_due(&mut self) -> Vec<Symbol> {
        let due: Vec<Symbol> = self
            .syncs
            .iter()
            .filter(|(symbol, sync)| {
                sync.needs_snapshot() && !self.pending_snapshot.contains(*symbol)
            })
            .map(|(symbol, _)| symbol.clone())
            .collect();
        self.pending_snapshot.extend(due.iter().cloned());
        due
    }

    /// Throws away every book for a reason the stream cannot show (reconnect, backpressure).
    /// Returns the symbols that actually lost a book.
    pub fn invalidate_all(&mut self, reason: ResyncReason) -> Vec<Symbol> {
        let mut lost = Vec::new();
        for (symbol, sync) in &mut self.syncs {
            let had_book = sync.book().is_some();
            let _ = sync.invalidate(reason);
            if had_book {
                lost.push(symbol.clone());
            }
        }
        for _ in &lost {
            self.count(Outcome::NeedsResync(reason), false);
        }
        lost
    }

    fn scales_for(&self, symbol: &str) -> Result<Scales, FeedError> {
        self.scales
            .get(symbol)
            .copied()
            .ok_or_else(|| FeedError::UnknownSymbol(symbol.to_owned()))
    }

    fn count(&mut self, outcome: Outcome, crossed: bool) {
        match outcome {
            Outcome::Applied => {
                self.stats.applied += 1;
                if crossed {
                    self.stats.crossed += 1;
                }
            }
            Outcome::Buffered => self.stats.buffered += 1,
            Outcome::DroppedStale => self.stats.dropped_stale += 1,
            Outcome::NeedsResync(reason) => {
                self.stats.resyncs += 1;
                match reason {
                    ResyncReason::SequenceGap { .. } => self.stats.sequence_gaps += 1,
                    ResyncReason::SnapshotTooOld { .. } => self.stats.snapshots_too_old += 1,
                    ResyncReason::Backpressure => self.stats.backpressure_resyncs += 1,
                    ResyncReason::Reconnect => self.stats.reconnect_resyncs += 1,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed() -> Feed {
        Feed::new(BTreeMap::from([(
            "BTCUSDT".to_owned(),
            Scales {
                price_decimals: 1,
                qty_decimals: 3,
            },
        )]))
    }

    fn depth(first: u64, last: u64, prev: u64, bids: &str, asks: &str) -> String {
        format!(
            r#"{{"stream":"btcusdt@depth@100ms","data":{{"e":"depthUpdate","E":1,"T":1,"s":"BTCUSDT","U":{first},"u":{last},"pu":{prev},"b":{bids},"a":{asks}}}}}"#
        )
    }

    #[test]
    fn routes_frames_requests_one_snapshot_and_syncs() -> Result<(), FeedError> {
        let mut feed = feed();
        assert_eq!(
            feed.snapshots_due(),
            Vec::<String>::new(),
            "nothing seen yet"
        );

        let event = feed.on_text(&depth(10, 12, 9, r#"[["100.0","1.000"]]"#, "[]"))?;
        assert!(matches!(
            event,
            Event::Depth {
                outcome: Outcome::Buffered,
                ..
            }
        ));
        assert_eq!(feed.snapshots_due(), vec!["BTCUSDT".to_owned()]);
        assert_eq!(
            feed.snapshots_due(),
            Vec::<String>::new(),
            "already pending"
        );

        let body = r#"{"lastUpdateId":11,"bids":[["99.0","2.000"]],"asks":[["101.0","1.000"]]}"#;
        assert_eq!(feed.on_snapshot("BTCUSDT", body)?, Outcome::Applied);
        let book = feed.book("BTCUSDT").expect("synced");
        assert_eq!(book.best_bid().map(|l| l.price), Some(1000));
        assert!(feed.sync("BTCUSDT").is_some_and(BookSync::is_synced));
        assert_eq!(
            feed.stats().applied,
            1,
            "the bridge update drained from the buffer"
        );
        assert_eq!(feed.stats().snapshots, 1);
        assert_eq!(feed.snapshots_due(), Vec::<String>::new());

        let gap = feed.on_text(&depth(20, 21, 19, "[]", "[]"))?;
        assert!(matches!(
            gap,
            Event::Depth {
                outcome: Outcome::NeedsResync(ResyncReason::SequenceGap { .. }),
                ..
            }
        ));
        assert_eq!(feed.stats().sequence_gaps, 1);
        assert_eq!(feed.snapshots_due(), vec!["BTCUSDT".to_owned()]);
        Ok(())
    }

    #[test]
    fn trades_are_counted_and_scaled() -> Result<(), FeedError> {
        let mut feed = feed();
        let text = r#"{"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1,"a":7,"s":"BTCUSDT","p":"100.5","q":"0.010","f":1,"l":1,"T":1,"m":false}}"#;
        let Event::Trade(trade) = feed.on_text(text)? else {
            panic!("expected a trade");
        };
        assert_eq!((trade.price, trade.qty), (1005, 10));
        assert_eq!(feed.stats().trades, 1);
        Ok(())
    }

    #[test]
    fn bad_frames_and_unknown_symbols_are_errors_not_panics() {
        let mut feed = feed();
        assert!(feed.on_text("not json").is_err());
        assert_eq!(feed.stats().parse_errors, 1);
        let other = depth(1, 1, 0, "[]", "[]").replace("BTCUSDT", "DOGEUSDT");
        assert!(matches!(
            feed.on_text(&other),
            Err(FeedError::UnknownSymbol(s)) if s == "DOGEUSDT"
        ));
    }

    #[test]
    fn invalidate_all_only_counts_books_that_existed() -> Result<(), FeedError> {
        let mut feed = feed();
        assert_eq!(
            feed.invalidate_all(ResyncReason::Reconnect),
            Vec::<String>::new()
        );
        assert_eq!(feed.stats().resyncs, 0);

        feed.on_text(&depth(10, 12, 9, "[]", "[]"))?;
        feed.on_snapshot("BTCUSDT", r#"{"lastUpdateId":11,"bids":[],"asks":[]}"#)?;
        assert_eq!(
            feed.invalidate_all(ResyncReason::Backpressure),
            vec!["BTCUSDT".to_owned()]
        );
        assert_eq!(feed.stats().backpressure_resyncs, 1);
        assert!(feed.book("BTCUSDT").is_none());
        Ok(())
    }

    #[test]
    fn a_crossed_apply_is_counted() -> Result<(), FeedError> {
        let mut feed = feed();
        feed.on_text(&depth(10, 12, 9, "[]", "[]"))?;
        feed.on_snapshot(
            "BTCUSDT",
            r#"{"lastUpdateId":11,"bids":[["100.0","1.000"]],"asks":[["101.0","1.000"]]}"#,
        )?;
        let event = feed.on_text(&depth(13, 13, 12, r#"[["102.0","1.000"]]"#, "[]"))?;
        assert!(matches!(event, Event::Depth { crossed: true, .. }));
        assert_eq!(feed.stats().crossed, 1);
        Ok(())
    }
}
