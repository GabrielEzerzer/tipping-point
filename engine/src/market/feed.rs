//! The multi-symbol feed of one venue: raw frame text in, book outcomes out. Pure, so the live
//! loop, a recording and a synthetic test stream all drive exactly the same code. The venue reads
//! the text and its rule keeps each book; the counting and the reactions are shared.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use super::book::Book;
use super::decimal::DecimalError;
use super::sync::{BookSync, Outcome, ResyncReason};
use super::{Scales, Symbol};
use crate::venue::{Depth, MessageError, Parsed, Venue};

/// One trade in integer units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trade {
    pub symbol: String,
    /// The venue's id for it, as text: Binance numbers its aggregated trades, Bybit uses UUIDs.
    pub trade_id: String,
    pub price: i64,
    pub qty: i64,
    pub trade_time_ms: u64,
    /// True when the buyer was the maker (the seller crossed the spread), whatever the venue
    /// calls it.
    pub buyer_is_maker: bool,
}

/// Counters over everything the feed has seen.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub frames: u64,
    /// Frames [`Feed::on_text`] returned an error for, whatever the reason.
    pub parse_errors: u64,
    pub depth_events: u64,
    pub trades: u64,
    pub applied: u64,
    pub buffered: u64,
    pub dropped_stale: u64,
    /// Updates dropped because no book was there and none was coming but the venue's next
    /// snapshot (Bybit's rule).
    pub skipped: u64,
    /// Snapshots fed: REST bodies, and whole books sent on the stream.
    pub snapshots: u64,
    /// Times a trustworthy book was thrown away, by reason.
    pub resyncs: u64,
    pub sequence_gaps: u64,
    pub snapshots_too_old: u64,
    pub backpressure_resyncs: u64,
    pub reconnect_resyncs: u64,
    pub bad_frame_resyncs: u64,
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
    Trade(Trade),
}

/// Why a frame or snapshot could not be fed.
#[derive(Debug, Error)]
pub enum FeedError {
    #[error(transparent)]
    Message(#[from] MessageError),
    #[error(transparent)]
    Decimal(#[from] DecimalError),
    /// A depth update did not convert at its symbol's scales. It is lost, which breaks the
    /// symbol's update chain, so the symbol's buffer was dropped, and its book with it when it had
    /// one (`book_lost`: the only case counted as a resync).
    #[error("{symbol} depth update lost{}: {error}", if *.book_lost { ", book invalidated" } else { "" })]
    BadDepth {
        symbol: Symbol,
        error: DecimalError,
        book_lost: bool,
    },
    #[error("no scales for symbol {0:?}")]
    UnknownSymbol(String),
}

/// One venue's [`BookSync`] per symbol, plus the scales to parse with and the running counters.
#[derive(Debug)]
pub struct Feed {
    venue: Venue,
    scales: BTreeMap<Symbol, Scales>,
    syncs: BTreeMap<Symbol, BookSync>,
    pending_snapshot: BTreeSet<Symbol>,
    stats: Stats,
}

impl Feed {
    /// A feed of `venue`'s books for exactly the symbols given scales.
    #[must_use]
    pub fn new(venue: Venue, scales: BTreeMap<Symbol, Scales>) -> Self {
        let syncs = scales
            .keys()
            .cloned()
            .map(|s| (s, BookSync::new(venue)))
            .collect();
        Self {
            venue,
            scales,
            syncs,
            pending_snapshot: BTreeSet::new(),
            stats: Stats::default(),
        }
    }

    /// The venue whose streams this feed reads.
    #[must_use]
    pub fn venue(&self) -> Venue {
        self.venue
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
    /// A depth update that does not convert at its symbol's scales is lost, which breaks that
    /// symbol's update chain, so its buffer and any book are dropped ([`ResyncReason::BadFrame`])
    /// before the error comes back, saying whether there was a book to lose.
    ///
    /// # Errors
    /// Returns an error when the frame cannot be parsed or converted, or names a symbol without
    /// scales. Every error bumps the parse-error counter.
    pub fn on_text(&mut self, text: &str) -> Result<Event, FeedError> {
        let parsed = self.venue.parse(text);
        self.on_parsed(parsed)
    }

    /// [`Self::on_text`] for a frame its caller has already read by the venue's rules
    /// ([`Venue::parse`]) for a use of its own, so that it is read only once: rotate reads every
    /// frame for its table rows too (#68).
    ///
    /// # Errors
    /// As [`Self::on_text`]; a frame that did not read is an error here too.
    pub fn on_parsed(&mut self, parsed: Result<Parsed, MessageError>) -> Result<Event, FeedError> {
        self.stats.frames += 1;
        let event = parsed
            .map_err(FeedError::from)
            .and_then(|parsed| self.route(parsed));
        if event.is_err() {
            self.stats.parse_errors += 1;
        }
        event
    }

    fn route(&mut self, parsed: Parsed) -> Result<Event, FeedError> {
        match parsed {
            Parsed::Depth(raw) => {
                self.stats.depth_events += 1;
                let scales = self.scales_for(raw.symbol())?;
                let converted = raw.convert(scales);
                let symbol = raw.into_symbol();
                let depth = match converted {
                    Ok(depth) => depth,
                    Err(error) => {
                        let book_lost = self.invalidate(&symbol, ResyncReason::BadFrame);
                        return Err(FeedError::BadDepth {
                            symbol,
                            error,
                            book_lost,
                        });
                    }
                };
                let Some(sync) = self.syncs.get_mut(&symbol) else {
                    return Err(FeedError::UnknownSymbol(symbol));
                };
                let (outcome, crossed) = match depth {
                    Depth::Update(update) => {
                        let outcome = sync.on_event(update);
                        let crossed = sync.book().is_some_and(Book::is_crossed);
                        self.count(outcome, crossed);
                        (outcome, crossed)
                    }
                    // Counted as a snapshot, as a REST one is; its install is no applied update.
                    Depth::Book(snapshot) => {
                        let installed = sync.on_snapshot(&snapshot);
                        let crossed = sync.book().is_some_and(Book::is_crossed);
                        self.stats.snapshots += 1;
                        for (outcome, crossed) in installed.drained {
                            self.count(outcome, crossed);
                        }
                        (installed.outcome, crossed)
                    }
                };
                Ok(Event::Depth {
                    symbol,
                    outcome,
                    crossed,
                })
            }
            Parsed::Trade(raw) => {
                self.stats.trades += 1;
                let scales = self.scales_for(raw.symbol())?;
                Ok(Event::Trade(raw.convert(scales)?))
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
        let snapshot = self.venue.parse_snapshot(body)?.convert(scales)?;
        let Some(sync) = self.syncs.get_mut(symbol) else {
            return Err(FeedError::UnknownSymbol(symbol.to_owned()));
        };
        let installed = sync.on_snapshot(&snapshot);
        // The install is counted under `snapshots`; a failed install is the resync among the
        // replayed updates, so counting those counts it exactly once.
        for (outcome, crossed) in installed.drained {
            self.count(outcome, crossed);
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
        let symbols: Vec<Symbol> = self.syncs.keys().cloned().collect();
        symbols
            .into_iter()
            .filter(|symbol| self.invalidate(symbol, reason))
            .collect()
    }

    /// Throws away one symbol's book and buffer. True when there was a book to lose, which is
    /// the only case counted as a resync.
    fn invalidate(&mut self, symbol: &str, reason: ResyncReason) -> bool {
        let Some(sync) = self.syncs.get_mut(symbol) else {
            return false;
        };
        let had_book = sync.book().is_some();
        let _ = sync.invalidate(reason);
        if had_book {
            self.count(Outcome::NeedsResync(reason), false);
        }
        had_book
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
            Outcome::Skipped => self.stats.skipped += 1,
            Outcome::NeedsResync(reason) => {
                self.stats.resyncs += 1;
                match reason {
                    ResyncReason::SequenceGap { .. } => self.stats.sequence_gaps += 1,
                    ResyncReason::SnapshotTooOld { .. } => self.stats.snapshots_too_old += 1,
                    ResyncReason::Backpressure => self.stats.backpressure_resyncs += 1,
                    ResyncReason::Reconnect => self.stats.reconnect_resyncs += 1,
                    ResyncReason::BadFrame => self.stats.bad_frame_resyncs += 1,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed() -> Feed {
        Feed::new(
            Venue::BinanceUsdm,
            BTreeMap::from([(
                "BTCUSDT".to_owned(),
                Scales {
                    price_decimals: 1,
                    qty_decimals: 3,
                },
            )]),
        )
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
        assert_eq!(feed.stats().parse_errors, 2);
    }

    /// Synced at `u = 12` with a bid of 100.0 x 1.
    fn synced_feed() -> Result<Feed, FeedError> {
        let mut feed = feed();
        feed.on_text(&depth(10, 12, 9, "[]", "[]"))?;
        feed.on_snapshot(
            "BTCUSDT",
            r#"{"lastUpdateId":11,"bids":[["100.0","1.000"]],"asks":[["101.0","1.000"]]}"#,
        )?;
        assert!(feed.sync("BTCUSDT").is_some_and(BookSync::is_synced));
        Ok(feed)
    }

    #[test]
    fn a_depth_frame_that_does_not_convert_drops_the_book() -> Result<(), FeedError> {
        let mut feed = synced_feed()?;
        // Deletes the 100.0 bid, but 100.05 is finer than the one-decimal tick: unconvertible.
        let bad = depth(
            13,
            13,
            12,
            r#"[["100.0","0.000"],["100.05","1.000"]]"#,
            "[]",
        );
        let error = feed.on_text(&bad).err();
        assert!(
            matches!(
                &error,
                Some(FeedError::BadDepth { symbol, book_lost: true, .. }) if symbol == "BTCUSDT"
            ),
            "{error:?}"
        );
        assert!(
            feed.book("BTCUSDT").is_none(),
            "the stale bid must not survive"
        );
        assert!(!feed.sync("BTCUSDT").is_some_and(BookSync::is_synced));
        let stats = feed.stats();
        assert_eq!(stats.parse_errors, 1);
        assert_eq!((stats.resyncs, stats.bad_frame_resyncs), (1, 1));
        Ok(())
    }

    #[test]
    fn a_depth_frame_that_does_not_convert_with_no_book_says_none_was_lost() {
        let mut feed = feed();
        let finer_than_the_tick = depth(10, 12, 9, r#"[["100.05","1.000"]]"#, "[]");
        let error = feed.on_text(&finer_than_the_tick).err();
        assert!(
            matches!(
                &error,
                Some(FeedError::BadDepth {
                    book_lost: false,
                    ..
                })
            ),
            "{error:?}"
        );
        let stats = feed.stats();
        assert_eq!((stats.parse_errors, stats.resyncs), (1, 0));
    }

    #[test]
    fn a_too_old_snapshot_counts_one_resync() -> Result<(), FeedError> {
        let mut feed = feed();
        feed.on_text(&depth(20, 22, 19, "[]", "[]"))?;
        let outcome = feed.on_snapshot("BTCUSDT", r#"{"lastUpdateId":10,"bids":[],"asks":[]}"#)?;
        assert!(matches!(
            outcome,
            Outcome::NeedsResync(ResyncReason::SnapshotTooOld { .. })
        ));
        let stats = feed.stats();
        assert_eq!((stats.resyncs, stats.snapshots_too_old), (1, 1));
        Ok(())
    }

    #[test]
    fn crosses_are_counted_per_replayed_update() -> Result<(), FeedError> {
        let mut feed = feed();
        feed.on_text(&depth(10, 12, 9, "[]", "[]"))?; // bridge
        feed.on_text(&depth(13, 13, 12, r#"[["102.0","1.000"]]"#, "[]"))?; // crosses 101.0
        feed.on_text(&depth(14, 14, 13, r#"[["102.0","0.000"]]"#, "[]"))?; // clears
        feed.on_text(&depth(15, 15, 14, r#"[["101.5","1.000"]]"#, "[]"))?; // crosses again
        feed.on_snapshot(
            "BTCUSDT",
            r#"{"lastUpdateId":11,"bids":[["100.0","1.000"]],"asks":[["101.0","1.000"]]}"#,
        )?;
        assert_eq!(feed.stats().applied, 4);
        assert_eq!(feed.stats().crossed, 2);
        Ok(())
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

    fn bybit_feed() -> Feed {
        Feed::new(
            Venue::BybitLinear,
            BTreeMap::from([(
                "BTCUSDT".to_owned(),
                Scales {
                    price_decimals: 1,
                    qty_decimals: 3,
                },
            )]),
        )
    }

    /// A Bybit order-book message, as its stream and archive send one.
    fn bybit(kind: &str, u: u64, bids: &str, asks: &str) -> String {
        format!(
            r#"{{"topic":"orderbook.200.BTCUSDT","type":"{kind}","ts":1,"data":{{"s":"BTCUSDT","b":{bids},"a":{asks},"u":{u},"seq":7}},"cts":1}}"#
        )
    }

    #[test]
    fn a_bybit_book_comes_on_the_stream_and_chains_on_u() -> Result<(), FeedError> {
        let mut feed = bybit_feed();
        let early = feed.on_text(&bybit("delta", 41, r#"[["100.0","1.000"]]"#, "[]"))?;
        assert!(matches!(
            early,
            Event::Depth {
                outcome: Outcome::Skipped,
                ..
            }
        ));
        assert_eq!(
            feed.snapshots_due(),
            Vec::<String>::new(),
            "nothing to fetch"
        );
        let book = r#"[["100.0","1.000"],["99.0","2.000"]]"#;
        feed.on_text(&bybit("snapshot", 42, book, r#"[["101.0","1.000"]]"#))?;
        assert!(feed.sync("BTCUSDT").is_some_and(BookSync::is_synced));
        feed.on_text(&bybit("delta", 43, r#"[["100.0","0"]]"#, "[]"))?;
        assert_eq!(
            feed.book("BTCUSDT")
                .and_then(Book::best_bid)
                .map(|l| l.price),
            Some(990)
        );
        let gap = feed.on_text(&bybit("delta", 45, "[]", "[]"))?;
        assert!(matches!(
            gap,
            Event::Depth {
                outcome: Outcome::NeedsResync(ResyncReason::SequenceGap {
                    expected: 43,
                    got: 44
                }),
                ..
            }
        ));
        feed.on_text(&bybit("delta", 46, "[]", "[]"))?;
        // A restart: the stream starts over at u = 1 with a whole book.
        feed.on_text(&bybit("snapshot", 1, book, "[]"))?;
        feed.on_text(&bybit("delta", 2, r#"[["98.0","1.000"]]"#, "[]"))?;
        assert_eq!(feed.sync("BTCUSDT").and_then(BookSync::last_id), Some(2));
        let stats = feed.stats();
        assert_eq!(
            (
                stats.snapshots,
                stats.applied,
                stats.skipped,
                stats.sequence_gaps
            ),
            (2, 2, 2, 1)
        );
        assert!(matches!(
            feed.on_snapshot("BTCUSDT", "{}"),
            Err(FeedError::Message(MessageError::NoRestSnapshots(
                Venue::BybitLinear
            )))
        ));
        Ok(())
    }
}
