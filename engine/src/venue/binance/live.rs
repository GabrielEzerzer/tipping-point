//! Binance's live loop: two combined-stream sockets (depth on `/public`, aggregated trades on
//! `/market`) and a REST client around one [`Pipeline`], recording every input it applies to a
//! run's log when asked to. Every input gets the seq its log gives it, recorded or not, and an
//! [`Observer`] sees each one, numbered, right before it is applied.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::{Instant, interval, sleep, sleep_until};

use super::rest::{Rest, RestError};
use crate::market::feed::Feed;
use crate::market::live::{LiveEvent, Log, Observer, Stream, handle, install_crypto_provider};
use crate::market::pipeline::{Message, Pipeline};
use crate::market::ws;
use crate::market::{Scales, Symbol};
use crate::store::frame::Meta;
use crate::store::run::{self, BookSample, MarketParams};
use crate::store::writer::{LogConfig, LogWriter};
use crate::store::{LogError, now_ns};
use crate::venue::Venue;

/// The depth stream the loop subscribes to.
pub const DEPTH_STREAM: &str = "depth@100ms";
/// The trade stream the loop subscribes to.
pub const TRADE_STREAM: &str = "aggTrade";

/// Routed path that serves `@depth@100ms`.
pub const DEPTH_HOST: &str = "wss://fstream.binance.com/public/stream";
/// Routed path that serves `@aggTrade` (it is not delivered on `/public`).
pub const TRADE_HOST: &str = "wss://fstream.binance.com/market/stream";

/// Binance closes a connection after 24 h; leaving before that keeps the reconnect on our terms.
pub const DEFAULT_MAX_CONNECTION_AGE: Duration = Duration::from_hours(23);

/// Combined-stream URL for the given symbols and stream suffix, e.g. `depth@100ms`.
#[must_use]
pub fn stream_url(host: &str, symbols: &[Symbol], suffix: &str) -> String {
    let streams: Vec<String> = symbols
        .iter()
        .map(|s| format!("{}@{suffix}", s.to_lowercase()))
        .collect();
    format!("{host}?streams={}", streams.join("/"))
}

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
            max_connection_age: DEFAULT_MAX_CONNECTION_AGE,
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
    let mut pipeline = Pipeline::new(Feed::new(Venue::BinanceUsdm, scales));
    let run_id = config
        .record
        .as_ref()
        .and_then(|recording| recording.run_dir.file_name())
        .and_then(|name| name.to_str());
    observer.started(pipeline.feed(), run_id);

    let (depth_tx, mut depth_rx) = mpsc::channel(config.channel_capacity);
    let (trade_tx, mut trade_rx) = mpsc::channel(config.channel_capacity);
    let depth_url = stream_url(DEPTH_HOST, &config.symbols, DEPTH_STREAM);
    let trade_url = stream_url(TRADE_HOST, &config.symbols, TRADE_STREAM);
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

/// Creates the run folder and its manifest, then starts the log writer.
fn start_recording(
    recording: &Recording,
    symbols: &[Symbol],
    scales: &BTreeMap<Symbol, Scales>,
) -> Result<LogWriter, LiveError> {
    let params = MarketParams {
        venue: Venue::BinanceUsdm,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_combined_stream_urls() {
        let symbols = vec!["BTCUSDT".to_owned(), "ETHUSDT".to_owned()];
        assert_eq!(
            stream_url(DEPTH_HOST, &symbols, "depth@100ms"),
            "wss://fstream.binance.com/public/stream?streams=btcusdt@depth@100ms/ethusdt@depth@100ms"
        );
        assert_eq!(
            stream_url(TRADE_HOST, &symbols[..1], "aggTrade"),
            "wss://fstream.binance.com/market/stream?streams=btcusdt@aggTrade"
        );
    }
}
