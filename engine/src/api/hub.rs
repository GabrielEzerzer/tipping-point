//! The publish side. The live loop hands the hub every book sample, trade and status change
//! through a [`LiveTap`], and Python's results arrive over HTTP; readers take them from here.
//! Nothing a reader does can hold the loop up: latest-wins topics are `watch` channels (the
//! newest value replaces the last) and append topics bounded `broadcast` channels (a reader that
//! falls behind loses the oldest and is told how many), and neither ever blocks the sender.
//!
//! The hub also keeps the recent past in memory, book samples and trades, so a reader can
//! start with a backfill and a replay can cover the hour the rotate step has not reached yet.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::{broadcast, watch};

use super::ipc::{Encoder, Provenance, tagged};
use super::messages::{Status, StreamStatus, Streams, SymbolStatus, Topic, TradeMessage};
use crate::market::feed::{Event, Feed};
use crate::market::live::{LiveEvent, LiveStats, Observer, Stream};
use crate::market::pipeline::Applied;
use crate::market::sync::BookSync;
use crate::market::{Scales, Symbol};
use crate::store::run::{BookSample, MARKET};
use crate::store::samples::{Sample, SampleRows, Sampler};
use crate::store::writer::Entry;

/// How much the hub keeps and how far a reader may fall behind.
#[derive(Debug, Clone)]
pub struct HubConfig {
    /// The book sampling: levels a side and the grid of the capture clock.
    pub book_sample: BookSample,
    /// How far back book samples and trades are kept in memory.
    pub ring: Duration,
    /// Grid instants an Arrow subscriber may fall behind before it is cut.
    pub samples_capacity: usize,
    /// Trades a WebSocket client may fall behind before it is sent a gap.
    pub trades_capacity: usize,
    /// Relayed `jev` and `ising` messages a client may fall behind by.
    pub relay_capacity: usize,
    /// Harness only: also keep every message in one channel in publish order, so a lossless
    /// client sees all of them, uncoalesced. `false` in production.
    pub lossless: bool,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            book_sample: BookSample::default(),
            ring: Duration::from_mins(75),
            samples_capacity: 1024,
            trades_capacity: 16 * 1024,
            relay_capacity: 1024,
            lossless: false,
        }
    }
}

/// What the live feed is, known once it starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    pub scales: BTreeMap<Symbol, Scales>,
    /// The run being recorded, if any.
    pub run: Option<String>,
}

/// One grid instant as the live feed published it.
#[derive(Debug)]
pub struct Published {
    pub sample: Arc<Sample>,
    /// The instant as one Arrow IPC record batch message, shared by every Arrow subscriber.
    pub ipc: Vec<u8>,
    /// The capture time of the input that moved the clock past the instant: the moment the
    /// sample could first be made.
    pub captured_ns: i64,
}

/// What an Arrow subscriber receives.
#[derive(Debug, Clone)]
pub enum LiveItem {
    Sample(Arc<Published>),
    /// The live feed stopped: the stream ends cleanly.
    End,
}

/// Every message, in publish order, for lossless clients.
#[derive(Debug, Clone)]
pub enum Ordered {
    Sample(Arc<Published>),
    Trade(Arc<TradeMessage>),
    Status(Arc<Status>),
    Relay(Topic, Arc<Value>),
    End,
}

/// Latest-wins relayed results, by the `symbol` they carry ("" for none).
pub type Latest = Arc<BTreeMap<String, Arc<Value>>>;

/// Counts for instruments.
#[derive(Debug, Default)]
pub struct Counters {
    pub samples: AtomicU64,
    pub trades: AtomicU64,
    /// Samples that could not be made or encoded; never expected.
    pub errors: AtomicU64,
    pub ws_clients: AtomicUsize,
}

/// The recent past.
#[derive(Debug, Default)]
struct Ring {
    samples: VecDeque<Arc<Sample>>,
    /// Each with its capture time, the clock samples are on.
    trades: VecDeque<(i64, Arc<TradeMessage>)>,
    /// The run's first sample instant, kept after the ring lets that sample go.
    first_sample: Option<i64>,
}

/// Book samples and trades from memory, each trade with its capture time.
#[derive(Debug)]
pub struct Recent {
    pub samples: Vec<Arc<Sample>>,
    pub trades: Vec<(i64, Arc<TradeMessage>)>,
    /// The run's first sample instant, whether or not memory still holds it.
    pub first_sample: Option<i64>,
}

/// What a reader starting with a backfill gets: the recent past, and receivers that carry on
/// right after it, with no gap and no overlap.
#[derive(Debug)]
pub struct Backfill {
    pub samples: Vec<Arc<Sample>>,
    pub trades: Vec<Arc<TradeMessage>>,
}

/// The publish side's channels. See the module docs.
#[derive(Debug)]
pub struct Hub {
    config: HubConfig,
    source: watch::Sender<Option<Arc<Source>>>,
    ring: Mutex<Ring>,
    samples: broadcast::Sender<LiveItem>,
    book: watch::Sender<Option<Arc<Published>>>,
    trades: broadcast::Sender<Arc<TradeMessage>>,
    status: watch::Sender<Option<Arc<Status>>>,
    meters: watch::Sender<Latest>,
    pnl: watch::Sender<Latest>,
    jev: broadcast::Sender<Arc<Value>>,
    ising: broadcast::Sender<Arc<Value>>,
    ordered: Option<broadcast::Sender<Ordered>>,
    /// Bumped on every publish, so a connection waits on one thing.
    changed: watch::Sender<u64>,
    ended: watch::Sender<bool>,
    pub counters: Counters,
}

impl Hub {
    #[must_use]
    pub fn new(config: HubConfig) -> Arc<Self> {
        let ordered = config.lossless.then(|| broadcast::channel(1 << 16).0);
        Arc::new(Self {
            samples: broadcast::channel(config.samples_capacity.max(1)).0,
            trades: broadcast::channel(config.trades_capacity.max(1)).0,
            jev: broadcast::channel(config.relay_capacity.max(1)).0,
            ising: broadcast::channel(config.relay_capacity.max(1)).0,
            config,
            source: watch::channel(None).0,
            ring: Mutex::new(Ring::default()),
            book: watch::channel(None).0,
            status: watch::channel(None).0,
            meters: watch::channel(Latest::default()).0,
            pnl: watch::channel(Latest::default()).0,
            ordered,
            changed: watch::channel(0).0,
            ended: watch::channel(false).0,
            counters: Counters::default(),
        })
    }

    #[must_use]
    pub fn config(&self) -> &HubConfig {
        &self.config
    }

    fn ring(&self) -> MutexGuard<'_, Ring> {
        self.ring.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn changed(&self) {
        self.changed.send_modify(|n| *n = n.wrapping_add(1));
    }

    fn order(&self, item: Ordered) {
        if let Some(ordered) = &self.ordered {
            let _ = ordered.send(item);
        }
    }

    fn ring_ns(&self) -> i64 {
        i64::try_from(self.config.ring.as_nanos()).unwrap_or(i64::MAX)
    }

    /// The live feed has started.
    pub fn start(&self, source: Source) {
        self.source.send_replace(Some(Arc::new(source)));
        self.changed();
    }

    /// The live feed, once it has started.
    pub async fn source(&self) -> Arc<Source> {
        let mut rx = self.source.subscribe();
        loop {
            if let Some(source) = rx.borrow_and_update().clone() {
                return source;
            }
            if rx.changed().await.is_err() {
                std::future::pending::<()>().await;
            }
        }
    }

    /// The live feed, if it has started.
    #[must_use]
    pub fn started(&self) -> Option<Arc<Source>> {
        self.source.borrow().clone()
    }

    pub fn publish_sample(&self, published: Published) {
        let published = Arc::new(published);
        {
            let mut ring = self.ring();
            let t = published.sample.sample_ts_ns;
            ring.first_sample.get_or_insert(t);
            ring.samples.push_back(Arc::clone(&published.sample));
            let oldest = t.saturating_sub(self.ring_ns());
            while ring
                .samples
                .front()
                .is_some_and(|s| s.sample_ts_ns < oldest)
            {
                ring.samples.pop_front();
            }
            // Under the ring's lock, so a subscriber's backfill ends exactly where it starts.
            let _ = self.samples.send(LiveItem::Sample(Arc::clone(&published)));
        }
        self.counters.samples.fetch_add(1, Ordering::Relaxed);
        self.book.send_replace(Some(Arc::clone(&published)));
        self.order(Ordered::Sample(published));
        self.changed();
    }

    pub fn publish_trade(&self, captured_ns: i64, trade: TradeMessage) {
        let trade = Arc::new(trade);
        {
            let mut ring = self.ring();
            ring.trades.push_back((captured_ns, Arc::clone(&trade)));
            let oldest = captured_ns.saturating_sub(self.ring_ns());
            while ring.trades.front().is_some_and(|(t, _)| *t < oldest) {
                ring.trades.pop_front();
            }
            let _ = self.trades.send(Arc::clone(&trade));
        }
        self.counters.trades.fetch_add(1, Ordering::Relaxed);
        self.order(Ordered::Trade(trade));
        self.changed();
    }

    pub fn publish_status(&self, status: Status) {
        let status = Arc::new(status);
        self.status.send_replace(Some(Arc::clone(&status)));
        self.order(Ordered::Status(status));
        self.changed();
    }

    /// A result from the Python side, for the app.
    pub fn relay(&self, topic: Topic, data: Value) {
        let data = Arc::new(data);
        let key = data
            .get("symbol")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        match topic {
            Topic::Meters => self.meters.send_modify(|latest| {
                Arc::make_mut(latest).insert(key, Arc::clone(&data));
            }),
            Topic::Pnl => self.pnl.send_modify(|latest| {
                Arc::make_mut(latest).insert(key, Arc::clone(&data));
            }),
            Topic::Jev => {
                let _ = self.jev.send(Arc::clone(&data));
            }
            Topic::Ising => {
                let _ = self.ising.send(Arc::clone(&data));
            }
            Topic::Book | Topic::Trades | Topic::Status | Topic::Lattice => return,
        }
        self.order(Ordered::Relay(topic, data));
        self.changed();
    }

    /// The live feed stopped: Arrow streams end cleanly and clients are told.
    pub fn end(&self) {
        {
            let _ring = self.ring();
            self.ended.send_replace(true);
            let _ = self.samples.send(LiveItem::End);
        }
        self.order(Ordered::End);
        self.changed();
    }

    #[must_use]
    pub fn has_ended(&self) -> bool {
        *self.ended.borrow()
    }

    /// Live samples for an Arrow subscriber, after those of the last `backfill` before the
    /// newest one, and whether the feed had already ended (then no [`LiveItem::End`] follows).
    #[must_use]
    pub fn subscribe_samples(
        &self,
        backfill: Duration,
    ) -> (Vec<Arc<Sample>>, broadcast::Receiver<LiveItem>, bool) {
        let ring = self.ring();
        let rx = self.samples.subscribe();
        let samples = backfill_of(&ring.samples, backfill, |s| s.sample_ts_ns)
            .cloned()
            .collect();
        (samples, rx, *self.ended.borrow())
    }

    /// Trades from now on, after a backfill of the last `backfill` of samples and trades.
    #[must_use]
    pub fn subscribe_trades(
        &self,
        backfill: Duration,
    ) -> (Backfill, broadcast::Receiver<Arc<TradeMessage>>) {
        let ring = self.ring();
        let rx = self.trades.subscribe();
        let samples = backfill_of(&ring.samples, backfill, |s| s.sample_ts_ns)
            .cloned()
            .collect();
        let trades = backfill_of(&ring.trades, backfill, |(t, _)| *t)
            .map(|(_, trade)| Arc::clone(trade))
            .collect();
        (Backfill { samples, trades }, rx)
    }

    /// Book samples and trades still in memory from after `after_ns`, in time order: the part
    /// of the live run the rotate step has not reached.
    #[must_use]
    pub fn recent(&self, after_ns: i64) -> Recent {
        let ring = self.ring();
        Recent {
            samples: ring
                .samples
                .iter()
                .filter(|s| s.sample_ts_ns > after_ns)
                .cloned()
                .collect(),
            trades: ring
                .trades
                .iter()
                .filter(|(t, _)| *t > after_ns)
                .cloned()
                .collect(),
            first_sample: ring.first_sample,
        }
    }

    /// Grid instants and trades held in memory.
    #[must_use]
    pub fn ring_len(&self) -> (usize, usize) {
        let ring = self.ring();
        (ring.samples.len(), ring.trades.len())
    }

    #[must_use]
    pub fn arrow_subscribers(&self) -> usize {
        self.samples.receiver_count()
    }

    #[must_use]
    pub fn watch_changed(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    #[must_use]
    pub fn watch_book(&self) -> watch::Receiver<Option<Arc<Published>>> {
        self.book.subscribe()
    }

    #[must_use]
    pub fn watch_status(&self) -> watch::Receiver<Option<Arc<Status>>> {
        self.status.subscribe()
    }

    #[must_use]
    pub fn watch_relayed(&self, topic: Topic) -> Option<watch::Receiver<Latest>> {
        match topic {
            Topic::Meters => Some(self.meters.subscribe()),
            Topic::Pnl => Some(self.pnl.subscribe()),
            _ => None,
        }
    }

    #[must_use]
    pub fn subscribe_relayed(&self, topic: Topic) -> Option<broadcast::Receiver<Arc<Value>>> {
        match topic {
            Topic::Jev => Some(self.jev.subscribe()),
            Topic::Ising => Some(self.ising.subscribe()),
            _ => None,
        }
    }

    /// Every message from now on, in publish order; `None` unless the hub is lossless.
    #[must_use]
    pub fn subscribe_ordered(&self) -> Option<broadcast::Receiver<Ordered>> {
        self.ordered.as_ref().map(broadcast::Sender::subscribe)
    }
}

/// The items of `ring` within `backfill` of its newest, oldest first.
fn backfill_of<T>(
    ring: &VecDeque<T>,
    backfill: Duration,
    time: impl Fn(&T) -> i64,
) -> impl Iterator<Item = &T> {
    let newest = ring.back().map(&time);
    let span = i64::try_from(backfill.as_nanos()).unwrap_or(i64::MAX);
    let start = newest.map_or(ring.len(), |newest| {
        let from = newest.saturating_sub(span);
        ring.partition_point(|item| time(item) <= from)
    });
    ring.iter().skip(start)
}

/// The live loop's observer for the API: it samples the book before each input with rotate's
/// sampler, so the live stream holds exactly the `book_samples` rows rotate writes later, and
/// publishes trades and every change of status.
#[derive(Debug)]
pub struct LiveTap {
    hub: Arc<Hub>,
    sampler: Sampler,
    rows: SampleRows,
    encoder: Option<Encoder>,
    /// The seq and capture time of the input being applied.
    input: Option<(u64, i64)>,
    depth_connected: bool,
    trades_connected: bool,
    last_status: Option<Status>,
}

impl LiveTap {
    #[must_use]
    pub fn new(hub: Arc<Hub>) -> Self {
        let sample = hub.config().book_sample;
        Self {
            hub,
            sampler: Sampler::new(sample),
            rows: SampleRows::new(sample.levels),
            encoder: None,
            input: None,
            depth_connected: false,
            trades_connected: false,
            last_status: None,
        }
    }

    fn publish_sample(&mut self, sample: Sample, captured_ns: i64) {
        let Some(encoder) = self.encoder.as_mut() else {
            return;
        };
        self.rows.push(&sample);
        let ipc = self
            .rows
            .take()
            .ok()
            .and_then(|batch| encoder.batch(&batch).ok());
        let Some(ipc) = ipc else {
            self.hub.counters.errors.fetch_add(1, Ordering::Relaxed);
            return;
        };
        self.hub.publish_sample(Published {
            sample: Arc::new(sample),
            ipc,
            captured_ns,
        });
    }

    fn status(&self, feed: &Feed, stats: &LiveStats, ts_ms: i64) -> Status {
        let counters = feed.stats();
        Status {
            ts_ms,
            streams: Streams {
                depth: StreamStatus {
                    connected: self.depth_connected,
                    reconnects: stats.depth_reconnects,
                    frames_dropped: stats.depth_frames_dropped,
                },
                trades: StreamStatus {
                    connected: self.trades_connected,
                    reconnects: stats.trade_reconnects,
                    frames_dropped: stats.trade_frames_dropped,
                },
            },
            symbols: feed
                .symbols()
                .map(|symbol| {
                    let synced = feed.sync(symbol).is_some_and(BookSync::is_synced);
                    (symbol.clone(), SymbolStatus { synced })
                })
                .collect(),
            resyncs: counters.resyncs,
            sequence_gaps: counters.sequence_gaps,
            snapshot_failures: stats.snapshot_failures,
            parse_errors: counters.parse_errors,
        }
    }
}

/// The schema of the live `book_samples` stream for a source.
#[must_use]
pub fn live_schema(source: &Source, levels: usize) -> arrow::datatypes::SchemaRef {
    tagged(
        &SampleRows::schema(levels),
        &Provenance {
            table: "book_samples",
            domain: MARKET,
            run: source.run.as_deref().unwrap_or_default(),
            scales: Some(&source.scales),
        },
    )
}

impl Observer for LiveTap {
    fn started(&mut self, feed: &Feed, run_id: Option<&str>) {
        let scales = feed
            .symbols()
            .filter_map(|symbol| Some((symbol.clone(), feed.scales(symbol)?)))
            .collect();
        let source = Source {
            scales,
            run: run_id.map(str::to_owned),
        };
        let schema = live_schema(&source, self.sampler.levels());
        self.encoder = Encoder::new(schema).ok().map(|(encoder, _header)| encoder);
        self.hub.start(source);
    }

    fn input(&mut self, seq: u64, entry: &Entry, feed: &Feed) {
        self.input = Some((seq, entry.capture_ts_ns));
        let instants = self.sampler.instants(entry.capture_ts_ns);
        let before = i64::try_from(seq).unwrap_or(i64::MAX) - 1;
        for t in instants {
            match Sample::of_feed(feed, t, before, self.sampler.levels()) {
                Ok(sample) => self.publish_sample(sample, entry.capture_ts_ns),
                Err(_) => {
                    self.hub.counters.errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    fn event(&mut self, event: LiveEvent<'_>) {
        match event {
            LiveEvent::Connected { stream, .. } => self.set_connected(stream, true),
            LiveEvent::Disconnected { stream, .. } => self.set_connected(stream, false),
            LiveEvent::Applied {
                applied,
                feed,
                stats,
                ..
            } => {
                let Some((seq, captured_ns)) = self.input else {
                    return;
                };
                if let Applied::Frame(Ok(Event::Trade(trade))) = applied {
                    self.hub.publish_trade(
                        captured_ns,
                        TradeMessage {
                            symbol: trade.symbol.clone(),
                            seq,
                            ts_ms: trade.trade_time_ms,
                            id: trade.agg_trade_id,
                            px: trade.price,
                            qty: trade.qty,
                            buyer_is_maker: trade.buyer_is_maker,
                        },
                    );
                }
                let status = self.status(feed, stats, captured_ns.div_euclid(1_000_000));
                if !self
                    .last_status
                    .as_ref()
                    .is_some_and(|last| last.same_as(&status))
                {
                    self.last_status = Some(status.clone());
                    self.hub.publish_status(status);
                }
            }
            _ => {}
        }
    }
}

impl LiveTap {
    fn set_connected(&mut self, stream: Stream, connected: bool) {
        match stream {
            Stream::Depth => self.depth_connected = connected,
            Stream::Trades => self.trades_connected = connected,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backfill_is_the_tail_within_its_span() {
        let ring: VecDeque<i64> = (0..10).map(|i| i * 100).collect();
        let tail = |span: u64| -> Vec<i64> {
            backfill_of(&ring, Duration::from_nanos(span), |t| *t)
                .copied()
                .collect()
        };
        assert_eq!(tail(0), Vec::<i64>::new());
        assert_eq!(tail(250), vec![700, 800, 900]);
        assert_eq!(tail(10_000), ring.iter().copied().collect::<Vec<_>>());
        let empty: VecDeque<i64> = VecDeque::new();
        assert_eq!(
            backfill_of(&empty, Duration::from_secs(1), |t| *t).count(),
            0
        );
    }

    #[test]
    fn relayed_results_keep_the_latest_per_symbol() {
        let hub = Hub::new(HubConfig::default());
        let meters = hub
            .watch_relayed(Topic::Meters)
            .expect("a latest-wins topic");
        hub.relay(
            Topic::Meters,
            serde_json::json!({"symbol": "BTCUSDT", "v": 1}),
        );
        hub.relay(
            Topic::Meters,
            serde_json::json!({"symbol": "BTCUSDT", "v": 2}),
        );
        hub.relay(
            Topic::Meters,
            serde_json::json!({"symbol": "ETHUSDT", "v": 3}),
        );
        let latest = meters.borrow().clone();
        assert_eq!(latest.len(), 2);
        assert_eq!(latest["BTCUSDT"]["v"], 2);
    }
}
