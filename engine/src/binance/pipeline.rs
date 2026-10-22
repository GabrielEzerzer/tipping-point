//! What the engine does with each input, written once so that the live loop and a replay of
//! its log build the same books and raise the same events.
//!
//! The live loop turns everything it receives into log entries ([`entries`]), logs each one,
//! then applies it ([`Pipeline::apply`]). A replay reads the same entries back from the log and
//! applies them the same way. Nothing reaches the feed by any other path, so whatever changed
//! a book live is in the log: the frames and snapshots, and also the reconnects, the frames
//! dropped under backpressure and the failed snapshot fetches.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::Symbol;
use super::feed::{Event, Feed, FeedError};
use super::live::{LiveEvent, LiveStats, Stream};
use super::sync::{BookSync, Outcome, ResyncReason};
use super::ws::Inbound;
use crate::store::frame::Kind;
use crate::store::ms_to_ns;
use crate::store::writer::Entry;

/// Something a socket or the REST client did that the feed reacts to although it is no frame.
/// Logged as a Control frame: JSON tagged by `type`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Control {
    /// A socket connected; `attempt` counts connections since its reader started.
    Connected { stream: Stream, attempt: u32 },
    /// A socket's connection ended; `dropped_before` counts frames lost just before.
    Disconnected {
        stream: Stream,
        reason: String,
        dropped_before: u64,
    },
    /// A reader dropped frames on a full channel.
    FramesDropped { stream: Stream, count: u64 },
    /// A snapshot fetch failed, so the symbol may be asked for again.
    SnapshotFailed { symbol: Symbol, error: String },
}

/// A Snapshot frame's payload: the symbol and the REST body exactly as received.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotPayload {
    pub symbol: Symbol,
    pub body: String,
}

/// Everything the live loop receives that can become input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// What a socket reader sent.
    Socket { stream: Stream, inbound: Inbound },
    /// A finished snapshot fetch, stamped when its reply arrived; a failure as its message.
    Snapshot {
        symbol: Symbol,
        capture_ns: i64,
        result: Result<String, String>,
    },
}

/// The log entries a message stands for, in the order the live loop applies them. A drop count
/// arrives with the next frame but invalidates the books before that frame is applied, so it
/// becomes its own entry first. A failed connection attempt changes no book and stands for
/// nothing. Pure: the same message always gives the same entries.
#[must_use]
pub fn entries(message: Message) -> Vec<Entry> {
    match message {
        Message::Socket { stream, inbound } => match inbound {
            Inbound::Connected {
                attempt,
                capture_ns,
            } => vec![control(&Control::Connected { stream, attempt }, capture_ns)],
            Inbound::ConnectFailed { .. } => Vec::new(),
            Inbound::Disconnected {
                reason,
                dropped_before,
                capture_ns,
            } => {
                let disconnected = Control::Disconnected {
                    stream,
                    reason,
                    dropped_before,
                };
                vec![control(&disconnected, capture_ns)]
            }
            Inbound::Text {
                text,
                dropped_before,
                capture_ns,
            } => {
                let mut out = Vec::with_capacity(2);
                if dropped_before > 0 {
                    let dropped = Control::FramesDropped {
                        stream,
                        count: dropped_before,
                    };
                    out.push(control(&dropped, capture_ns));
                }
                out.push(Entry {
                    kind: stream.kind(),
                    source_ts_ns: frame_event_time_ns(&text),
                    capture_ts_ns: capture_ns,
                    payload: text,
                });
                out
            }
        },
        Message::Snapshot {
            symbol,
            capture_ns,
            result: Ok(body),
        } => vec![Entry {
            kind: Kind::Snapshot,
            source_ts_ns: snapshot_event_time_ns(&body),
            capture_ts_ns: capture_ns,
            payload: to_json(&SnapshotPayload { symbol, body }),
        }],
        Message::Snapshot {
            symbol,
            capture_ns,
            result: Err(error),
        } => vec![control(
            &Control::SnapshotFailed { symbol, error },
            capture_ns,
        )],
    }
}

fn control(control: &Control, capture_ts_ns: i64) -> Entry {
    Entry {
        kind: Kind::Control,
        source_ts_ns: 0,
        capture_ts_ns,
        payload: to_json(control),
    }
}

/// Control and snapshot payloads are plain structs of strings and numbers: they always
/// serialize.
fn to_json(value: &impl Serialize) -> String {
    serde_json::to_string(value).expect("a Control or Snapshot payload always serializes")
}

/// Just the exchange event time `E`, wherever it sits.
#[derive(Deserialize)]
struct EventTime {
    #[serde(rename = "E")]
    event_time_ms: Option<u64>,
}

#[derive(Deserialize)]
struct Combined {
    data: Option<EventTime>,
}

/// The `E` of a combined-stream frame in nanoseconds, 0 when it has none.
fn frame_event_time_ns(text: &str) -> i64 {
    serde_json::from_str::<Combined>(text)
        .ok()
        .and_then(|frame| frame.data)
        .and_then(|data| data.event_time_ms)
        .map_or(0, ms_to_ns)
}

/// The `E` of a REST snapshot body in nanoseconds, 0 when it has none.
fn snapshot_event_time_ns(body: &str) -> i64 {
    serde_json::from_str::<EventTime>(body)
        .ok()
        .and_then(|body| body.event_time_ms)
        .map_or(0, ms_to_ns)
}

/// What one input did to the feed.
#[derive(Debug)]
pub enum Applied {
    /// A depth or trade frame, and what the feed made of it.
    Frame(Result<Event, FeedError>),
    /// A REST snapshot, and what installing it did.
    Snapshot(Result<Outcome, FeedError>),
    /// A control event.
    Control,
    /// A Snapshot or Control payload that does not parse; nothing was applied.
    BadPayload(serde_json::Error),
    /// Meta or `SegmentEnd`: bookkeeping, not an input.
    NotAnInput,
}

impl Applied {
    /// The error the input met, if any.
    #[must_use]
    pub fn error(&self) -> Option<String> {
        match self {
            Self::Frame(Err(error)) | Self::Snapshot(Err(error)) => Some(error.to_string()),
            Self::BadPayload(error) => Some(format!("bad payload: {error}")),
            Self::Frame(Ok(_)) | Self::Snapshot(Ok(_)) | Self::Control | Self::NotAnInput => None,
        }
    }
}

/// The live loop's state around one feed: the feed, the counters the sockets drive, and which
/// symbols were last reported synced.
#[derive(Debug)]
pub struct Pipeline {
    feed: Feed,
    stats: LiveStats,
    synced: BTreeSet<Symbol>,
}

impl Pipeline {
    #[must_use]
    pub fn new(feed: Feed) -> Self {
        Self {
            feed,
            stats: LiveStats::default(),
            synced: BTreeSet::new(),
        }
    }

    #[must_use]
    pub fn feed(&self) -> &Feed {
        &self.feed
    }

    /// For the live loop's snapshot scheduling ([`Feed::snapshots_due`]), which marks requests
    /// pending and changes no book.
    pub fn feed_mut(&mut self) -> &mut Feed {
        &mut self.feed
    }

    #[must_use]
    pub fn stats(&self) -> &LiveStats {
        &self.stats
    }

    /// For the counters the live loop keeps outside any input: connect failures and snapshot
    /// fetches.
    pub fn stats_mut(&mut self) -> &mut LiveStats {
        &mut self.stats
    }

    /// Applies one log entry, given its kind and payload, reacting exactly as the live loop
    /// always has: an unconvertible depth update resyncs its symbol, an unreadable depth frame
    /// drops every book, a depth reconnect or drop drops every book, and anything else the feed
    /// cannot use is skipped. Raises the events that follow, then [`LiveEvent::Applied`];
    /// bookkeeping frames (Meta, `SegmentEnd`) are skipped silently.
    pub fn apply(
        &mut self,
        kind: Kind,
        payload: &str,
        on_event: &mut impl FnMut(LiveEvent<'_>),
    ) -> Applied {
        let applied = match kind {
            Kind::Depth => Applied::Frame(self.depth(payload, on_event)),
            // Whatever the trade socket's text turns out to be, live only lets the feed count it.
            Kind::Trade => Applied::Frame(self.feed.on_text(payload)),
            Kind::Snapshot => match serde_json::from_str::<SnapshotPayload>(payload) {
                Ok(snapshot) => {
                    Applied::Snapshot(self.snapshot(&snapshot.symbol, &snapshot.body, on_event))
                }
                Err(error) => Applied::BadPayload(error),
            },
            Kind::Control => match serde_json::from_str::<Control>(payload) {
                Ok(control) => {
                    self.control(control, on_event);
                    Applied::Control
                }
                Err(error) => Applied::BadPayload(error),
            },
            Kind::Meta | Kind::SegmentEnd => return Applied::NotAnInput,
        };
        on_event(LiveEvent::Applied {
            kind,
            applied: &applied,
            feed: &self.feed,
        });
        applied
    }

    fn depth(
        &mut self,
        text: &str,
        on_event: &mut impl FnMut(LiveEvent<'_>),
    ) -> Result<Event, FeedError> {
        let result = self.feed.on_text(text);
        match &result {
            Ok(Event::Depth {
                symbol, outcome, ..
            }) => {
                if let Outcome::NeedsResync(reason) = outcome {
                    on_event(LiveEvent::Resync {
                        symbol,
                        reason: *reason,
                    });
                }
                self.note_sync_change(symbol, on_event);
            }
            // The feed already dropped this symbol's book.
            Err(FeedError::BadDepth { symbol, .. }) => {
                on_event(LiveEvent::Resync {
                    symbol,
                    reason: ResyncReason::BadFrame,
                });
                self.note_sync_change(symbol, on_event);
            }
            // Unreadable, so it may have been any symbol's update: trust no book.
            Err(FeedError::Message(_)) => self.invalidate(ResyncReason::BadFrame, on_event),
            Ok(Event::Trade(_)) | Err(FeedError::Decimal(_) | FeedError::UnknownSymbol(_)) => {}
        }
        result
    }

    fn snapshot(
        &mut self,
        symbol: &str,
        body: &str,
        on_event: &mut impl FnMut(LiveEvent<'_>),
    ) -> Result<Outcome, FeedError> {
        let result = self.feed.on_snapshot(symbol, body);
        match &result {
            Ok(Outcome::NeedsResync(reason)) => on_event(LiveEvent::Resync {
                symbol,
                reason: *reason,
            }),
            Ok(_) => {}
            Err(_) => self.stats.snapshot_failures += 1,
        }
        self.note_sync_change(symbol, on_event);
        result
    }

    fn control(&mut self, control: Control, on_event: &mut impl FnMut(LiveEvent<'_>)) {
        match control {
            Control::Connected { stream, attempt } => {
                on_event(LiveEvent::Connected { stream, attempt });
            }
            Control::Disconnected {
                stream,
                reason,
                dropped_before,
            } => {
                self.note_dropped(stream, dropped_before, on_event);
                match stream {
                    Stream::Depth => self.stats.depth_reconnects += 1,
                    Stream::Trades => self.stats.trade_reconnects += 1,
                }
                on_event(LiveEvent::Disconnected {
                    stream,
                    reason: &reason,
                });
                // Depth frames were missed, so no book can be trusted; trades carry no book.
                if stream == Stream::Depth {
                    self.invalidate(ResyncReason::Reconnect, on_event);
                }
            }
            Control::FramesDropped { stream, count } => {
                if self.note_dropped(stream, count, on_event) && stream == Stream::Depth {
                    self.invalidate(ResyncReason::Backpressure, on_event);
                }
            }
            Control::SnapshotFailed { symbol, error } => {
                self.stats.snapshot_failures += 1;
                self.feed.snapshot_failed(&symbol);
                on_event(LiveEvent::SnapshotFailed {
                    symbol: &symbol,
                    error: &error,
                });
                self.note_sync_change(&symbol, on_event);
            }
        }
    }

    /// Counts and reports frames a reader dropped. True when there were any.
    fn note_dropped(
        &mut self,
        stream: Stream,
        count: u64,
        on_event: &mut impl FnMut(LiveEvent<'_>),
    ) -> bool {
        if count == 0 {
            return false;
        }
        match stream {
            Stream::Depth => self.stats.depth_frames_dropped += count,
            Stream::Trades => self.stats.trade_frames_dropped += count,
        }
        on_event(LiveEvent::FramesDropped { stream, count });
        true
    }

    fn invalidate(&mut self, reason: ResyncReason, on_event: &mut impl FnMut(LiveEvent<'_>)) {
        for symbol in self.feed.invalidate_all(reason) {
            self.synced.remove(&symbol);
            on_event(LiveEvent::Resync {
                symbol: &symbol,
                reason,
            });
        }
    }

    fn note_sync_change(&mut self, symbol: &str, on_event: &mut impl FnMut(LiveEvent<'_>)) {
        let is_synced = self.feed.sync(symbol).is_some_and(BookSync::is_synced);
        if is_synced && self.synced.insert(symbol.to_owned()) {
            on_event(LiveEvent::Synced { symbol });
        } else if !is_synced {
            self.synced.remove(symbol);
        }
    }
}

/// What a replay made of its inputs. A replay carries on past every error, as the live loop
/// does, and reports them here rather than stopping at the first.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Report {
    /// Frames read, bookkeeping included.
    pub frames: u64,
    /// Frames that were inputs.
    pub inputs: u64,
    /// Inputs the feed could not use: unreadable or unconvertible frames, unknown symbols, bad
    /// snapshots and payloads.
    pub errors: u64,
    /// The first [`FIRST_ERRORS`] of them, by position (the seq, in a log), with what was wrong.
    pub first_errors: Vec<(u64, String)>,
    /// How the input ended when it did not end cleanly: a log whose newest segment is open (still
    /// being written, or the writer died between frames) or torn.
    pub unclosed: Option<String>,
}

/// How many errors a [`Report`] keeps in full.
pub const FIRST_ERRORS: usize = 32;

impl Report {
    /// Counts one frame and what applying it did.
    pub fn note(&mut self, position: u64, applied: &Applied) {
        self.frames += 1;
        if matches!(applied, Applied::NotAnInput) {
            return;
        }
        self.inputs += 1;
        if let Some(error) = applied.error() {
            self.errors += 1;
            if self.first_errors.len() < FIRST_ERRORS {
                self.first_errors.push((position, error));
            }
        }
    }
}

/// A finished replay: the pipeline in its final state, and what it made of the inputs.
#[derive(Debug)]
pub struct Replayed {
    pub pipeline: Pipeline,
    pub report: Report,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::binance::Scales;

    const SYMBOL: &str = "BTCUSDT";

    fn text(first: u64, last: u64, prev: u64, dropped_before: u64) -> Message {
        Message::Socket {
            stream: Stream::Depth,
            inbound: Inbound::Text {
                text: format!(
                    r#"{{"stream":"btcusdt@depth@100ms","data":{{"e":"depthUpdate","E":1,"T":1,"s":"BTCUSDT","U":{first},"u":{last},"pu":{prev},"b":[["100.0","1.000"]],"a":[]}}}}"#
                ),
                dropped_before,
                capture_ns: 5,
            },
        }
    }

    /// A pipeline driven the way the live loop drives it, plus every resync it reported.
    struct Live {
        pipeline: Pipeline,
        resyncs: Vec<(Symbol, ResyncReason)>,
    }

    impl Live {
        /// BTCUSDT synced at `u = 12`, reached the way the loop reaches it.
        fn synced() -> Self {
            let scales = Scales {
                price_decimals: 1,
                qty_decimals: 3,
            };
            let mut live = Self {
                pipeline: Pipeline::new(Feed::new(BTreeMap::from([(SYMBOL.to_owned(), scales)]))),
                resyncs: Vec::new(),
            };
            live.send(text(10, 12, 9, 0));
            live.send(Message::Snapshot {
                symbol: SYMBOL.to_owned(),
                capture_ns: 6,
                result: Ok(
                    r#"{"lastUpdateId":11,"bids":[["99.0","1.000"]],"asks":[["101.0","1.000"]]}"#
                        .to_owned(),
                ),
            });
            assert!(live.has_book());
            live
        }

        fn send(&mut self, message: Message) {
            let resyncs = &mut self.resyncs;
            let mut on_event = |event: LiveEvent<'_>| {
                if let LiveEvent::Resync { symbol, reason } = event {
                    resyncs.push((symbol.to_owned(), reason));
                }
            };
            for entry in entries(message) {
                self.pipeline
                    .apply(entry.kind, &entry.payload, &mut on_event);
            }
        }

        fn has_book(&self) -> bool {
            self.pipeline.feed().book(SYMBOL).is_some()
        }
    }

    #[test]
    fn frames_dropped_under_backpressure_drop_the_book() {
        let mut live = Live::synced();
        // This update chains (pu = 12), so only the drop count says frames were lost.
        live.send(text(13, 13, 12, 1));
        assert!(!live.has_book());
        assert_eq!(live.pipeline.stats().depth_frames_dropped, 1);
        assert_eq!(
            live.resyncs,
            vec![(SYMBOL.to_owned(), ResyncReason::Backpressure)]
        );
        assert!(live.pipeline.synced.is_empty());
    }

    #[test]
    fn a_reconnect_drops_the_book_and_asks_for_a_fresh_snapshot() {
        let mut live = Live::synced();
        live.send(Message::Socket {
            stream: Stream::Depth,
            inbound: Inbound::Disconnected {
                reason: "no frames for 30s".to_owned(),
                dropped_before: 2,
                capture_ns: 7,
            },
        });
        assert!(!live.has_book());
        let stats = live.pipeline.stats();
        assert_eq!(stats.depth_reconnects, 1);
        assert_eq!(stats.depth_frames_dropped, 2, "drops just before it count");
        assert_eq!(
            live.resyncs,
            vec![(SYMBOL.to_owned(), ResyncReason::Reconnect)]
        );

        // On the new connection even an update that would have chained waits for a snapshot.
        live.send(text(13, 13, 12, 0));
        assert!(!live.has_book());
        assert_eq!(
            live.pipeline.feed_mut().snapshots_due(),
            vec![SYMBOL.to_owned()]
        );
    }

    #[test]
    fn a_trade_reconnect_leaves_the_books_alone() {
        let mut live = Live::synced();
        live.send(Message::Socket {
            stream: Stream::Trades,
            inbound: Inbound::Disconnected {
                reason: "stream ended".to_owned(),
                dropped_before: 0,
                capture_ns: 7,
            },
        });
        assert!(live.has_book());
        assert_eq!(live.pipeline.stats().trade_reconnects, 1);
        assert_eq!(live.resyncs, Vec::new());
    }

    #[test]
    fn a_depth_frame_that_does_not_convert_unsyncs_its_symbol() {
        let mut live = Live::synced();
        let finer_than_the_tick = r#"{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1,"T":1,"s":"BTCUSDT","U":13,"u":13,"pu":12,"b":[["100.05","1.000"]],"a":[]}}"#;
        live.send(Message::Socket {
            stream: Stream::Depth,
            inbound: Inbound::Text {
                text: finer_than_the_tick.to_owned(),
                dropped_before: 0,
                capture_ns: 7,
            },
        });
        assert!(!live.has_book());
        assert!(live.pipeline.synced.is_empty());
        assert_eq!(
            live.resyncs,
            vec![(SYMBOL.to_owned(), ResyncReason::BadFrame)]
        );
    }

    #[test]
    fn an_unreadable_depth_frame_drops_every_book() {
        let mut live = Live::synced();
        live.send(Message::Socket {
            stream: Stream::Depth,
            inbound: Inbound::Text {
                text: "not json".to_owned(),
                dropped_before: 0,
                capture_ns: 7,
            },
        });
        assert!(!live.has_book());
        assert_eq!(
            live.resyncs,
            vec![(SYMBOL.to_owned(), ResyncReason::BadFrame)]
        );
    }

    #[test]
    fn an_unreadable_trade_frame_is_counted_and_nothing_else() {
        let mut live = Live::synced();
        live.send(Message::Socket {
            stream: Stream::Trades,
            inbound: Inbound::Text {
                text: "not json".to_owned(),
                dropped_before: 3,
                capture_ns: 7,
            },
        });
        assert!(live.has_book());
        assert_eq!(live.resyncs, Vec::new());
        assert_eq!(live.pipeline.stats().trade_frames_dropped, 3);
        assert_eq!(live.pipeline.feed().stats().parse_errors, 1);
    }

    #[test]
    fn a_failed_snapshot_frees_the_symbol_to_be_asked_again() {
        let mut live = Live::synced();
        live.send(text(20, 21, 19, 0)); // a gap: the book is gone and a snapshot is due
        assert_eq!(
            live.pipeline.feed_mut().snapshots_due(),
            vec![SYMBOL.to_owned()]
        );
        assert!(
            live.pipeline.feed_mut().snapshots_due().is_empty(),
            "pending"
        );
        live.send(Message::Snapshot {
            symbol: SYMBOL.to_owned(),
            capture_ns: 8,
            result: Err("http: timed out".to_owned()),
        });
        assert_eq!(live.pipeline.stats().snapshot_failures, 1);
        assert_eq!(
            live.pipeline.feed_mut().snapshots_due(),
            vec![SYMBOL.to_owned()]
        );
    }

    #[test]
    fn messages_become_entries_in_the_order_they_apply() {
        let entries = entries(text(13, 13, 12, 4));
        let kinds: Vec<Kind> = entries.iter().map(|e| e.kind).collect();
        assert_eq!(kinds, vec![Kind::Control, Kind::Depth]);
        assert_eq!(
            serde_json::from_str::<Control>(&entries[0].payload).ok(),
            Some(Control::FramesDropped {
                stream: Stream::Depth,
                count: 4
            })
        );
        assert_eq!(entries[1].source_ts_ns, 1_000_000, "E = 1 ms");
        assert_eq!(entries[1].capture_ts_ns, 5);

        let snapshot = super::entries(Message::Snapshot {
            symbol: SYMBOL.to_owned(),
            capture_ns: 9,
            result: Ok(r#"{"lastUpdateId":1,"E":1700000000123,"bids":[],"asks":[]}"#.to_owned()),
        });
        assert_eq!(snapshot[0].source_ts_ns, 1_700_000_000_123_000_000);
        let payload: SnapshotPayload =
            serde_json::from_str(&snapshot[0].payload).expect("a snapshot payload");
        assert_eq!(payload.symbol, SYMBOL);

        let failed = Message::Socket {
            stream: Stream::Depth,
            inbound: Inbound::ConnectFailed {
                error: "refused".to_owned(),
            },
        };
        assert_eq!(super::entries(failed), Vec::new());
    }

    #[test]
    fn control_payloads_are_tagged_json() {
        let control = Control::Disconnected {
            stream: Stream::Trades,
            reason: "server closed".to_owned(),
            dropped_before: 1,
        };
        assert_eq!(
            to_json(&control),
            r#"{"type":"disconnected","stream":"trades","reason":"server closed","dropped_before":1}"#
        );
    }

    #[test]
    fn a_report_keeps_counting_past_errors() {
        let mut feed = Feed::new(BTreeMap::new());
        let mut report = Report::default();
        report.note(0, &Applied::NotAnInput);
        report.note(1, &Applied::Frame(feed.on_text("not json")));
        report.note(2, &Applied::Control);
        assert_eq!((report.frames, report.inputs, report.errors), (3, 2, 1));
        assert_eq!(report.first_errors.len(), 1);
        assert_eq!(report.first_errors[0].0, 1);
    }
}
