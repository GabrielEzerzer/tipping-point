//! The live loop: two socket readers and a REST client around one [`Pipeline`], recording
//! every input it applies to a run's log when asked to. Every input gets the seq its log gives
//! it, recorded or not, and an [`Observer`] sees each one, numbered, right before it is applied.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::{Instant, interval, sleep, sleep_until};

use super::feed::Feed;
use super::pipeline::{Applied, Message, Pipeline, entries};
use super::rest::{Rest, RestError};
use super::sync::ResyncReason;
use super::ws::{self, Inbound};
use super::{Scales, Symbol};
use crate::store::frame::{Kind, Meta};
use crate::store::run::{self, BookSample, MarketParams};
use crate::store::writer::{Entry, LogConfig, LogWriter, Summary};
use crate::store::{LogError, now_ns};

/// The depth stream the loop subscribes to.
pub const DEPTH_STREAM: &str = "depth@100ms";
/// The trade stream the loop subscribes to.
pub const TRADE_STREAM: &str = "aggTrade";

/// Knobs for a live run.
#[derive(Debug, Clone)]
pub struct LiveConfig {
    pub symbols: Vec<Symbol>,
    /// Stop after this long; `None` runs until Ctrl-C or until the task is dropped.
    pub duration: Option<Duration>,
    /// How often a [`LiveEvent::Report`] fires.
    pub report_every: Duration,
    /// Reader-to-loop channel depth per connection; full means frames drop and books resync.
    pub channel_capacity: usize,
    /// Floor between two snapshot fetches for one symbol, to respect REST weight on a flapping
    /// stream. The first fetch after a gap is still immediate.
    pub min_snapshot_interval: Duration,
    pub max_connection_age: Duration,
    /// Record every input to a new run, or `None` to record nothing.
    pub record: Option<Recording>,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            symbols: vec!["BTCUSDT".to_owned(), "ETHUSDT".to_owned()],
            duration: None,
            report_every: Duration::from_secs(1),
            channel_capacity: 1024,
            min_snapshot_interval: Duration::from_secs(2),
            max_connection_age: ws::DEFAULT_MAX_CONNECTION_AGE,
            record: None,
        }
    }
}

/// Where and how a live run records.
#[derive(Debug, Clone)]
pub struct Recording {
    /// The run folder to create, normally `<data dir>/market/<run_id>/`; it must not exist.
    pub run_dir: PathBuf,
    pub log: LogConfig,
}

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

/// A finished snapshot fetch: the symbol, when its reply arrived, and the body or the error.
type Fetched = (Symbol, i64, Result<String, RestError>);

/// Why the loop could not start or record.
#[derive(Debug, Error)]
pub enum LiveError {
    #[error(transparent)]
    Rest(#[from] RestError),
    #[error("recording: {0}")]
    Log(#[from] LogError),
    #[error("recording to {}: {source}", .path.display())]
    Run { path: PathBuf, source: io::Error },
}

/// Runs the pipeline live until `config.duration` elapses or Ctrl-C, telling `observer` as it
/// goes. When recording, the run folder gets its manifest at the start and `complete.json`
/// after a clean end.
///
/// # Errors
/// Returns an error when the REST client cannot be built, the scales cannot be fetched, or the
/// recording cannot be created or written.
pub async fn run_live(config: LiveConfig, mut observer: impl Observer) -> Result<(), LiveError> {
    install_crypto_provider();
    let rest = Rest::new()?;
    let scales = rest.fetch_scales(&config.symbols).await?;
    let mut log = match &config.record {
        Some(recording) => Log::recorded(start_recording(recording, &config.symbols, &scales)?),
        None => Log::unrecorded(),
    };
    let mut pipeline = Pipeline::new(Feed::new(scales));
    let run_id = config
        .record
        .as_ref()
        .and_then(|recording| recording.run_dir.file_name())
        .and_then(|name| name.to_str());
    observer.started(pipeline.feed(), run_id);

    let (depth_tx, mut depth_rx) = mpsc::channel(config.channel_capacity);
    let (trade_tx, mut trade_rx) = mpsc::channel(config.channel_capacity);
    let depth_url = ws::stream_url(ws::DEPTH_HOST, &config.symbols, DEPTH_STREAM);
    let trade_url = ws::stream_url(ws::TRADE_HOST, &config.symbols, TRADE_STREAM);
    let depth_reader = tokio::spawn(ws::run_reader(
        depth_url,
        config.max_connection_age,
        depth_tx,
    ));
    let trade_reader = tokio::spawn(ws::run_reader(
        trade_url,
        config.max_connection_age,
        trade_tx,
    ));

    let (snapshot_tx, mut snapshot_rx) = mpsc::channel::<Fetched>(16);
    let mut next_fetch_allowed: BTreeMap<Symbol, Instant> = BTreeMap::new();

    let start = Instant::now();
    // tokio caps a very long sleep rather than overflowing, so "forever" is just a long sleep.
    let stop = sleep(
        config
            .duration
            .unwrap_or(Duration::from_secs(u64::from(u32::MAX))),
    );
    tokio::pin!(stop);
    // A ctrl_c future that fails to register resolves at once; it is then disarmed, never
    // polled again, and only the duration stops the loop.
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let mut ctrl_c_armed = true;
    let mut report = interval(config.report_every);
    report.tick().await; // the first tick fires immediately; skip it

    let ended: Result<(), LiveError> = loop {
        let message = tokio::select! {
            Some(inbound) = depth_rx.recv() => Message::Socket { stream: Stream::Depth, inbound },
            Some(inbound) = trade_rx.recv() => Message::Socket { stream: Stream::Trades, inbound },
            Some((symbol, capture_ns, result)) = snapshot_rx.recv() => Message::Snapshot {
                symbol,
                capture_ns,
                result: result.map_err(|error| error.to_string()),
            },
            _ = report.tick() => {
                observer.event(LiveEvent::Report { elapsed: start.elapsed(), feed: pipeline.feed(), stats: pipeline.stats() });
                if let Err(error) = log.tick().await {
                    break Err(error.into());
                }
                continue;
            }
            () = &mut stop => break Ok(()),
            pressed = &mut ctrl_c, if ctrl_c_armed => {
                ctrl_c_armed = false;
                if pressed.is_ok() {
                    break Ok(());
                }
                continue;
            }
        };
        if let Err(error) = handle(message, &mut pipeline, &mut log, &mut observer).await {
            break Err(error.into());
        }

        for symbol in pipeline.feed_mut().snapshots_due() {
            let now = Instant::now();
            let at = next_fetch_allowed
                .get(&symbol)
                .copied()
                .unwrap_or(now)
                .max(now);
            next_fetch_allowed.insert(symbol.clone(), at + config.min_snapshot_interval);
            pipeline.stats_mut().snapshot_fetches += 1;
            let rest = rest.clone();
            let tx = snapshot_tx.clone();
            tokio::spawn(async move {
                sleep_until(at).await;
                let result = rest.fetch_snapshot(&symbol).await;
                let _ = tx.send((symbol, now_ns(), result)).await;
            });
        }
    };

    depth_reader.abort();
    trade_reader.abort();
    let recorded = match (log, &config.record) {
        (Log::Recorded(log), Some(recording)) => {
            finish_recording(*log, &recording.run_dir, ended.is_ok()).await
        }
        _ => Ok(()),
    };
    // The writer's own error explains a loop that stopped because the writer did.
    recorded.and(ended)
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
    for entry in entries(message) {
        let seq = log.append(entry.clone()).await?;
        observer.input(seq, &entry, pipeline.feed());
        pipeline.apply(entry.kind, &entry.payload, &mut |event| {
            observer.event(event);
        });
    }
    Ok(())
}

/// Creates the run folder and its manifest, then starts the log writer.
fn start_recording(
    recording: &Recording,
    symbols: &[Symbol],
    scales: &BTreeMap<Symbol, Scales>,
) -> Result<LogWriter, LiveError> {
    let params = MarketParams {
        symbols: symbols.to_vec(),
        scales: scales.clone(),
        streams: vec![DEPTH_STREAM.to_owned(), TRADE_STREAM.to_owned()],
        book_sample: BookSample::default(),
    };
    let manifest = run::create(&recording.run_dir, params).map_err(|source| LiveError::Run {
        path: recording.run_dir.clone(),
        source,
    })?;
    let log = LogWriter::start(
        &recording.run_dir,
        Meta::for_run(&manifest),
        recording.log.clone(),
    )?;
    Ok(log)
}

/// Closes the log, then writes `complete.json` if the loop itself ended cleanly: a run cut short
/// by an error stays incomplete even though its log closed properly.
async fn finish_recording(log: LogWriter, run_dir: &Path, clean: bool) -> Result<(), LiveError> {
    let summary = tokio::task::spawn_blocking(move || log.finish())
        .await
        .map_err(|_| LogError::WriterPanicked)??;
    if !clean {
        return Ok(());
    }
    run::complete(run_dir, &summary).map_err(|source| LiveError::Run {
        path: run_dir.to_owned(),
        source,
    })
}

/// Picks the process-wide TLS crypto provider once. With one provider compiled in rustls would
/// pick it anyway; pinning it here keeps a future second provider from turning into a panic.
/// Call it before building any [`Rest`] client or socket outside [`run_live`].
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
