//! What every live loop shares, whichever venue it reads: the sockets' counters and events, the
//! [`Observer`] that watches a loop, the [`Log`] that numbers (and, recording, keeps) its inputs,
//! and [`handle`], what a loop does with each message, network aside. A venue's own loop (such as
//! [`crate::venue::binance::live`]) adds its endpoints and its way of fetching snapshots.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::Scales;
use super::feed::Feed;
use super::pipeline::{Applied, Message, Pipeline, entries};
use super::sync::ResyncReason;
use super::ws::Inbound;
use crate::store::LogError;
use crate::store::frame::Kind;
use crate::store::writer::{Entry, LogWriter, Summary};

/// Which socket an event is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stream {
    Depth,
    Trades,
}

impl Stream {
    /// The kind of frame this socket's text is logged as.
    #[must_use]
    pub fn kind(self) -> Kind {
        match self {
            Self::Depth => Kind::Depth,
            Self::Trades => Kind::Trade,
        }
    }
}

/// Counters the sockets and REST produce (the feed keeps its own).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LiveStats {
    pub depth_reconnects: u64,
    pub trade_reconnects: u64,
    pub connect_failures: u64,
    pub depth_frames_dropped: u64,
    pub trade_frames_dropped: u64,
    pub snapshot_fetches: u64,
    pub snapshot_failures: u64,
}

/// What the loop tells its caller as it goes.
#[derive(Debug)]
pub enum LiveEvent<'a> {
    Connected {
        stream: Stream,
        attempt: u32,
    },
    Disconnected {
        stream: Stream,
        reason: &'a str,
    },
    ConnectFailed {
        stream: Stream,
        error: &'a str,
    },
    FramesDropped {
        stream: Stream,
        count: u64,
    },
    Resync {
        symbol: &'a str,
        reason: ResyncReason,
    },
    Synced {
        symbol: &'a str,
    },
    SnapshotFailed {
        symbol: &'a str,
        error: &'a str,
    },
    /// The venue now gives `symbol` other scales than its feed reads it at (`was`): its tick or
    /// lot size changed, so its depth updates stopped converting (#45). Every price and
    /// quantity of a run is at one scale, so the feed keeps `was` and the symbol's book stays
    /// down until a restart records a new run at `now`. The loop's own business, like
    /// [`LiveEvent::Report`]: no input raised it, and no replay raises it again.
    ScalesChanged {
        symbol: &'a str,
        was: Scales,
        now: Scales,
    },
    /// A check of `symbol`'s scales could not be made: the venue's exchange info did not come
    /// (a 429, a timeout, a 5xx), did not read, or no longer lists the symbol (#104). Said, so
    /// that the lack of a [`LiveEvent::ScalesChanged`] is never taken for unchanged scales while
    /// the symbol's updates fail to convert; the loop asks again at its next check (#100). The
    /// loop's own business, like `ScalesChanged`.
    ScalesCheckFailed {
        symbol: &'a str,
        error: &'a str,
    },
    Report {
        elapsed: Duration,
        feed: &'a Feed,
        stats: &'a LiveStats,
    },
    /// One input was applied: what it did, and the feed and the loop's counters right after.
    /// Fires for every frame, so loggers usually skip it; transcripts, replay checks and the
    /// API's status are built from it.
    Applied {
        kind: Kind,
        applied: &'a Applied,
        feed: &'a Feed,
        stats: &'a LiveStats,
    },
}

/// Whoever watches the live loop: it sees every input right before the loop applies it, and
/// every event the loop raises. A closure over [`LiveEvent`]s is an observer that skips inputs.
pub trait Observer {
    /// The loop is about to start, with this feed (its symbols and scales) and, when recording,
    /// this run id.
    fn started(&mut self, feed: &Feed, run_id: Option<&str>) {
        let _ = (feed, run_id);
    }

    /// One input, with the seq its log gave it, right before it is applied to `feed`.
    fn input(&mut self, seq: u64, entry: &Entry, feed: &Feed) {
        let _ = (seq, entry, feed);
    }

    /// One event, as it happens.
    fn event(&mut self, event: LiveEvent<'_>);
}

impl<F: FnMut(LiveEvent<'_>)> Observer for F {
    fn event(&mut self, event: LiveEvent<'_>) {
        self(event);
    }
}

/// Where the live loop numbers its inputs: the run's log when recording, or else a count that
/// numbers them as a one-segment log would (its Meta at seq 0, then one seq per input), so an
/// input has a seq either way.
#[derive(Debug)]
pub enum Log {
    Recorded(Box<LogWriter>),
    Unrecorded { next_seq: u64 },
}

impl Log {
    /// The log of a recorded run.
    #[must_use]
    pub fn recorded(writer: LogWriter) -> Self {
        Self::Recorded(Box::new(writer))
    }

    /// A count for a loop that records nothing.
    #[must_use]
    pub fn unrecorded() -> Self {
        Self::Unrecorded { next_seq: 1 }
    }

    /// Logs one entry, or only numbers it, and returns its seq.
    ///
    /// # Errors
    /// Returns the log's error once the writer has stopped.
    pub async fn append(&mut self, entry: Entry) -> Result<u64, LogError> {
        match self {
            Self::Recorded(writer) => writer.append(entry).await,
            Self::Unrecorded { next_seq } => {
                let seq = *next_seq;
                *next_seq += 1;
                Ok(seq)
            }
        }
    }

    /// Lets a recorded log end its segment when the hour is over, with nothing to append.
    ///
    /// # Errors
    /// Returns the log's error once the writer has stopped.
    pub async fn tick(&mut self) -> Result<(), LogError> {
        if let Self::Recorded(writer) = self {
            writer.tick().await?;
        }
        Ok(())
    }

    /// The writer, when recording.
    #[must_use]
    pub fn writer(&self) -> Option<&LogWriter> {
        match self {
            Self::Recorded(writer) => Some(writer),
            Self::Unrecorded { .. } => None,
        }
    }

    /// Closes a recorded log (see [`LogWriter::finish`]); `None` when nothing was recorded.
    ///
    /// # Errors
    /// Returns the error that stopped the writer.
    pub fn finish(self) -> Result<Option<Summary>, LogError> {
        match self {
            Self::Recorded(writer) => writer.finish().map(Some),
            Self::Unrecorded { .. } => Ok(None),
        }
    }
}

/// What the live loop does with each message, network aside: it logs every input the message
/// stands for, shows it to the observer with its seq, then applies it, so the log holds exactly
/// what the feed saw, in order. A failed connection attempt is counted, not logged: it changes
/// no book. Public so that a test can drive the live path from a script.
///
/// # Errors
/// Returns the log's error once the writer has stopped.
pub async fn handle(
    message: Message,
    pipeline: &mut Pipeline,
    log: &mut Log,
    observer: &mut impl Observer,
) -> Result<(), LogError> {
    if let Message::Socket {
        stream,
        inbound: Inbound::ConnectFailed { error },
    } = &message
    {
        pipeline.stats_mut().connect_failures += 1;
        observer.event(LiveEvent::ConnectFailed {
            stream: *stream,
            error,
        });
        return Ok(());
    }
    for entry in entries(pipeline.feed().venue(), message) {
        let seq = log.append(entry.clone()).await?;
        observer.input(seq, &entry, pipeline.feed());
        pipeline.apply(entry.kind, &entry.payload, &mut |event| {
            observer.event(event);
        });
    }
    Ok(())
}

/// Picks the process-wide TLS crypto provider once. With one provider compiled in rustls would
/// pick it anyway; pinning it here keeps a future second provider from turning into a panic.
/// Call it before building any REST client or socket outside a venue's live loop.
pub fn install_crypto_provider() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
}

/// Wall-clock `HH:MM:SSZ` for log lines.
#[must_use]
pub fn utc_clock() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!(
        "{:02}:{:02}:{:02}Z",
        secs / 3600 % 24,
        secs / 60 % 60,
        secs % 60
    )
}
