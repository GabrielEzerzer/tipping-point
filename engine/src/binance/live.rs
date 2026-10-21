//! The live loop: two socket readers and a REST client around one [`Feed`].

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::{Instant, interval, sleep, sleep_until};

use super::Symbol;
use super::feed::{Event, Feed};
use super::rest::{Rest, RestError};
use super::sync::{Outcome, ResyncReason};
use super::ws::{self, Inbound};

/// Knobs for a live run.
#[derive(Debug, Clone)]
pub struct LiveConfig {
    pub symbols: Vec<Symbol>,
    /// Stop after this long; `None` runs until the task is dropped.
    pub duration: Option<Duration>,
    /// How often a [`LiveEvent::Report`] fires.
    pub report_every: Duration,
    /// Reader-to-loop channel depth per connection; full means frames drop and books resync.
    pub channel_capacity: usize,
    /// Floor between two snapshot fetches for one symbol, to respect REST weight on a flapping
    /// stream. The first fetch after a gap is still immediate.
    pub min_snapshot_interval: Duration,
    pub max_connection_age: Duration,
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
        }
    }
}

/// Which socket an event is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Depth,
    Trades,
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
        error: &'a RestError,
    },
    Report {
        elapsed: Duration,
        feed: &'a Feed,
        stats: &'a LiveStats,
    },
}

/// Why the loop could not start.
#[derive(Debug, Error)]
pub enum LiveError {
    #[error(transparent)]
    Rest(#[from] RestError),
}

/// Runs the pipeline live until `config.duration` elapses, calling `on_event` as it goes.
///
/// # Errors
/// Returns an error when the REST client cannot be built or the scales cannot be fetched.
pub async fn run_live(
    config: LiveConfig,
    mut on_event: impl FnMut(LiveEvent<'_>),
) -> Result<(), LiveError> {
    install_crypto_provider();
    let rest = Rest::new()?;
    let mut feed = Feed::new(rest.fetch_scales(&config.symbols).await?);
    let mut stats = LiveStats::default();
    let mut synced: BTreeSet<Symbol> = BTreeSet::new();

    let (depth_tx, mut depth_rx) = mpsc::channel(config.channel_capacity);
    let (trade_tx, mut trade_rx) = mpsc::channel(config.channel_capacity);
    let depth_url = ws::stream_url(ws::DEPTH_HOST, &config.symbols, "depth@100ms");
    let trade_url = ws::stream_url(ws::TRADE_HOST, &config.symbols, "aggTrade");
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

    let (snapshot_tx, mut snapshot_rx) = mpsc::channel::<(Symbol, Result<String, RestError>)>(16);
    let mut next_fetch_allowed: BTreeMap<Symbol, Instant> = BTreeMap::new();

    let start = Instant::now();
    // tokio caps a very long sleep rather than overflowing, so "forever" is just a long sleep.
    let stop = sleep(
        config
            .duration
            .unwrap_or(Duration::from_secs(u64::from(u32::MAX))),
    );
    tokio::pin!(stop);
    let mut report = interval(config.report_every);
    report.tick().await; // the first tick fires immediately; skip it

    loop {
        tokio::select! {
            Some(inbound) = depth_rx.recv() => {
                handle_depth(inbound, &mut feed, &mut stats, &mut synced, &mut on_event);
            }
            Some(inbound) = trade_rx.recv() => {
                handle_trades(inbound, &mut feed, &mut stats, &mut on_event);
            }
            Some((symbol, result)) = snapshot_rx.recv() => {
                match result {
                    Ok(body) => match feed.on_snapshot(&symbol, &body) {
                        Ok(Outcome::NeedsResync(reason)) => on_event(LiveEvent::Resync { symbol: &symbol, reason }),
                        Ok(_) => {}
                        Err(_) => stats.snapshot_failures += 1,
                    },
                    Err(error) => {
                        stats.snapshot_failures += 1;
                        feed.snapshot_failed(&symbol);
                        on_event(LiveEvent::SnapshotFailed { symbol: &symbol, error: &error });
                    }
                }
                note_sync_change(&symbol, &feed, &mut synced, &mut on_event);
            }
            _ = report.tick() => {
                on_event(LiveEvent::Report { elapsed: start.elapsed(), feed: &feed, stats: &stats });
            }
            () = &mut stop => break,
        }

        for symbol in feed.snapshots_due() {
            let now = Instant::now();
            let at = next_fetch_allowed
                .get(&symbol)
                .copied()
                .unwrap_or(now)
                .max(now);
            next_fetch_allowed.insert(symbol.clone(), at + config.min_snapshot_interval);
            stats.snapshot_fetches += 1;
            let rest = rest.clone();
            let tx = snapshot_tx.clone();
            tokio::spawn(async move {
                sleep_until(at).await;
                let result = rest.fetch_snapshot(&symbol).await;
                let _ = tx.send((symbol, result)).await;
            });
        }
    }

    depth_reader.abort();
    trade_reader.abort();
    Ok(())
}

fn handle_depth(
    inbound: Inbound,
    feed: &mut Feed,
    stats: &mut LiveStats,
    synced: &mut BTreeSet<Symbol>,
    on_event: &mut impl FnMut(LiveEvent<'_>),
) {
    match inbound {
        Inbound::Connected { attempt } => on_event(LiveEvent::Connected {
            stream: Stream::Depth,
            attempt,
        }),
        Inbound::ConnectFailed { error } => {
            stats.connect_failures += 1;
            on_event(LiveEvent::ConnectFailed {
                stream: Stream::Depth,
                error: &error,
            });
        }
        Inbound::Disconnected { reason } => {
            stats.depth_reconnects += 1;
            on_event(LiveEvent::Disconnected {
                stream: Stream::Depth,
                reason: &reason,
            });
            invalidate(feed, ResyncReason::Reconnect, synced, on_event);
        }
        Inbound::Text {
            text,
            dropped_before,
        } => {
            if dropped_before > 0 {
                stats.depth_frames_dropped += dropped_before;
                on_event(LiveEvent::FramesDropped {
                    stream: Stream::Depth,
                    count: dropped_before,
                });
                invalidate(feed, ResyncReason::Backpressure, synced, on_event);
            }
            if let Ok(Event::Depth {
                symbol, outcome, ..
            }) = feed.on_text(&text)
            {
                if let Outcome::NeedsResync(reason) = outcome {
                    on_event(LiveEvent::Resync {
                        symbol: &symbol,
                        reason,
                    });
                }
                note_sync_change(&symbol, feed, synced, on_event);
            }
        }
    }
}

fn handle_trades(
    inbound: Inbound,
    feed: &mut Feed,
    stats: &mut LiveStats,
    on_event: &mut impl FnMut(LiveEvent<'_>),
) {
    match inbound {
        Inbound::Connected { attempt } => on_event(LiveEvent::Connected {
            stream: Stream::Trades,
            attempt,
        }),
        Inbound::ConnectFailed { error } => {
            stats.connect_failures += 1;
            on_event(LiveEvent::ConnectFailed {
                stream: Stream::Trades,
                error: &error,
            });
        }
        Inbound::Disconnected { reason } => {
            stats.trade_reconnects += 1;
            on_event(LiveEvent::Disconnected {
                stream: Stream::Trades,
                reason: &reason,
            });
        }
        Inbound::Text {
            text,
            dropped_before,
        } => {
            if dropped_before > 0 {
                stats.trade_frames_dropped += dropped_before;
                on_event(LiveEvent::FramesDropped {
                    stream: Stream::Trades,
                    count: dropped_before,
                });
            }
            let _ = feed.on_text(&text);
        }
    }
}

fn invalidate(
    feed: &mut Feed,
    reason: ResyncReason,
    synced: &mut BTreeSet<Symbol>,
    on_event: &mut impl FnMut(LiveEvent<'_>),
) {
    for symbol in feed.invalidate_all(reason) {
        synced.remove(&symbol);
        on_event(LiveEvent::Resync {
            symbol: &symbol,
            reason,
        });
    }
}

fn note_sync_change(
    symbol: &str,
    feed: &Feed,
    synced: &mut BTreeSet<Symbol>,
    on_event: &mut impl FnMut(LiveEvent<'_>),
) {
    let is_synced = feed
        .sync(symbol)
        .is_some_and(super::sync::BookSync::is_synced);
    if is_synced && synced.insert(symbol.to_owned()) {
        on_event(LiveEvent::Synced { symbol });
    } else if !is_synced {
        synced.remove(symbol);
    }
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
