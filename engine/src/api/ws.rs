//! The app's WebSocket. The upgrade runs over hyper with tokio-tungstenite; each connection is
//! one task that reads the client's messages and writes what it subscribed to.
//!
//! A connection sends at most every 16 ms. Between sends, latest-wins topics keep only their
//! newest value, so a slow client skips book samples rather than queueing them; an append topic
//! it falls behind on yields a `gap` with the count lost (`jev`, never dropped, closes it with
//! 4002 instead). A client whose socket takes nothing for the stall timeout is closed with 4002
//! when the close can still be sent, and dropped otherwise. None of this reaches the live loop:
//! the hub never waits on a reader.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use hyper::upgrade::OnUpgrade;
use hyper_util::rt::TokioIo;
use serde_json::Value;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::broadcast::error::TryRecvError;
use tokio::sync::{broadcast, watch};
use tokio::time::{Instant, Interval, MissedTickBehavior, interval, sleep_until, timeout};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::handshake::derive_accept_key;
use tokio_tungstenite::tungstenite::protocol::frame::CloseFrame;
use tokio_tungstenite::tungstenite::protocol::{Role, WebSocketConfig};
use tokio_tungstenite::tungstenite::{Error as WsError, Message};

use super::Api;
use super::hub::{Latest, Ordered, Published};
use super::messages::{
    BookMessage, ClientMessage, PROTOCOL, ServerMessage, SourceMessage, Status, Topic,
    TradeMessage, Welcome, close,
};
use super::replay::{Item, Replay};
use super::table::TableError;
use crate::store::samples::Sample;
use crate::venue::Venue;

/// Client messages are small; a bigger one is a bad request.
const MAX_MESSAGE: usize = 64 * 1024;
/// A connection sends at most this often.
const FLUSH_EVERY: Duration = Duration::from_millis(16);
/// Status goes out at least this often, unchanged if need be, so the app sees the engine is alive.
const HEARTBEAT: Duration = Duration::from_secs(1);
/// A client gets this long to say hello.
const HELLO_WITHIN: Duration = Duration::from_secs(10);
/// Messages a client may send before the welcome, waiting for the live feed to start.
const EARLY: usize = 16;

/// `GET /v1/ws`: checks the handshake, answers 101 and hands the upgraded socket to a task.
pub async fn upgrade(State(api): State<Arc<Api>>, mut request: Request) -> Response {
    let headers = request.headers();
    let wants_websocket = request.method() == Method::GET
        && has_token(headers, &header::UPGRADE, "websocket")
        && has_token(headers, &header::CONNECTION, "upgrade")
        && headers
            .get(header::SEC_WEBSOCKET_VERSION)
            .is_some_and(|version| version == "13");
    let key = headers.get(header::SEC_WEBSOCKET_KEY).cloned();
    let (Some(key), true) = (key, wants_websocket) else {
        return (StatusCode::BAD_REQUEST, "not a WebSocket upgrade").into_response();
    };
    let Some(on_upgrade) = request.extensions_mut().remove::<OnUpgrade>() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "this connection cannot upgrade",
        )
            .into_response();
    };
    tokio::spawn(async move {
        let Ok(upgraded) = on_upgrade.await else {
            return;
        };
        let config = WebSocketConfig::default()
            .max_message_size(Some(MAX_MESSAGE))
            .max_frame_size(Some(MAX_MESSAGE));
        let ws =
            WebSocketStream::from_raw_socket(TokioIo::new(upgraded), Role::Server, Some(config))
                .await;
        Connection::new(api, ws).run().await;
    });
    let accept = HeaderValue::from_str(&derive_accept_key(key.as_bytes()))
        .expect("a base64 accept key is a valid header value");
    let mut response = Response::new(Body::empty());
    *response.status_mut() = StatusCode::SWITCHING_PROTOCOLS;
    let headers = response.headers_mut();
    headers.insert(header::UPGRADE, HeaderValue::from_static("websocket"));
    headers.insert(header::CONNECTION, HeaderValue::from_static("upgrade"));
    headers.insert(header::SEC_WEBSOCKET_ACCEPT, accept);
    response
}

/// True when the comma-separated header holds `token`, ignoring case.
fn has_token(headers: &HeaderMap, name: &header::HeaderName, token: &str) -> bool {
    headers
        .get_all(name)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|part| part.trim().eq_ignore_ascii_case(token))
}

/// Why a connection ended.
#[derive(Debug)]
enum Closing {
    /// Close with this code and reason.
    Code(u16, String),
    /// The client left, or its socket failed.
    Gone,
}

fn bad(reason: impl Into<String>) -> Closing {
    Closing::Code(close::BAD_REQUEST, reason.into())
}

/// What the client subscribed to.
#[derive(Debug)]
struct Sub {
    topics: BTreeSet<Topic>,
    /// `None` for every venue.
    venues: Option<BTreeSet<Venue>>,
    /// `None` for every symbol.
    symbols: Option<BTreeSet<String>>,
    levels: usize,
}

impl Sub {
    fn has(&self, topic: Topic) -> bool {
        self.topics.contains(&topic)
    }

    /// True when the client takes `venue`'s `symbol`: both filters let it through.
    fn wants(&self, venue: Venue, symbol: &str) -> bool {
        self.venues
            .as_ref()
            .is_none_or(|venues| venues.contains(&venue))
            && self
                .symbols
                .as_ref()
                .is_none_or(|symbols| symbols.contains(symbol))
    }
}

/// Where the connection's book and trades come from.
#[derive(Debug)]
enum Mode {
    Live,
    Replay {
        replay: Replay,
        run: String,
        from_ms: Option<i64>,
        speed: f64,
        paused: bool,
    },
    /// A replay ran out or failed, or the live feed stopped: nothing comes until `replay` or
    /// `live`. After a replay it is kept here, so a seek starts it again.
    Ended(Option<LastReplay>),
}

/// A connection's replay once it is over: its run and speed, and why it failed, if it did.
#[derive(Debug)]
struct LastReplay {
    run: String,
    speed: f64,
    error: Option<String>,
}

struct Connection<S> {
    api: Arc<Api>,
    ws: WebSocketStream<S>,
    sub: Sub,
    mode: Mode,
    outbox: Vec<Message>,
    changed: watch::Receiver<u64>,
    book: watch::Receiver<Latest<Venue, Published>>,
    status: watch::Receiver<Option<Arc<Status>>>,
    meters: watch::Receiver<Latest>,
    pnl: watch::Receiver<Latest>,
    sent_meters: Latest,
    sent_pnl: Latest,
    trades: Option<broadcast::Receiver<Arc<TradeMessage>>>,
    jev: Option<broadcast::Receiver<Arc<Value>>>,
    ising: Option<broadcast::Receiver<Arc<Value>>>,
    ordered: Option<broadcast::Receiver<Ordered>>,
    /// The newest book instant sent, per venue.
    last_book: BTreeMap<Venue, i64>,
    status_due: bool,
    replay_book: Option<Arc<Sample>>,
    replay_trades: Vec<Arc<TradeMessage>>,
}

impl<S: AsyncRead + AsyncWrite + Unpin> Connection<S> {
    fn new(api: Arc<Api>, ws: WebSocketStream<S>) -> Self {
        let hub = Arc::clone(&api.hub);
        hub.counters.ws_clients.fetch_add(1, Ordering::Relaxed);
        let levels = hub.config().book_sample.levels;
        Self {
            ordered: hub.subscribe_ordered(),
            changed: hub.watch_changed(),
            book: hub.watch_book(),
            status: hub.watch_status(),
            meters: hub
                .watch_relayed(Topic::Meters)
                .expect("meters is a latest-wins topic"),
            pnl: hub
                .watch_relayed(Topic::Pnl)
                .expect("pnl is a latest-wins topic"),
            sent_meters: Latest::default(),
            sent_pnl: Latest::default(),
            api,
            ws,
            sub: Sub {
                topics: BTreeSet::new(),
                venues: None,
                symbols: None,
                levels,
            },
            mode: Mode::Live,
            outbox: Vec::new(),
            trades: None,
            jev: None,
            ising: None,
            last_book: BTreeMap::new(),
            status_due: false,
            replay_book: None,
            replay_trades: Vec::new(),
        }
    }

    async fn run(mut self) {
        if let Err(Closing::Code(code, mut reason)) = self.serve().await {
            // A close frame is a control frame: 125 bytes, two of them the code.
            let mut end = reason.len().min(123);
            while !reason.is_char_boundary(end) {
                end -= 1;
            }
            reason.truncate(end);
            let frame = CloseFrame {
                code: code.into(),
                reason: reason.into(),
            };
            let _ = timeout(Duration::from_secs(1), self.ws.close(Some(frame))).await;
        }
        self.api
            .hub
            .counters
            .ws_clients
            .fetch_sub(1, Ordering::Relaxed);
    }

    async fn serve(&mut self) -> Result<(), Closing> {
        self.hello().await?;
        let hub = Arc::clone(&self.api.hub);
        // The welcome needs the feed's symbols. A client may send its next messages (a `sub`)
        // without waiting for it; they are kept and handled right after the welcome.
        let mut early = Vec::new();
        let source = loop {
            tokio::select! {
                biased;
                source = hub.source() => break source,
                incoming = self.ws.next() => match incoming {
                    Some(Ok(Message::Text(text))) if early.len() < EARLY => early.push(text),
                    Some(Ok(Message::Text(_))) => return Err(bad("too many messages before welcome")),
                    Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {}
                    Some(Ok(Message::Binary(_))) => {
                        return Err(bad("the engine takes text messages only"));
                    }
                    Some(Ok(Message::Close(_)) | Err(_)) | None => return Err(Closing::Gone),
                },
            }
        };
        let config = hub.config();
        let welcome = Welcome::new(
            &source,
            config.book_sample.levels,
            config.book_sample.every_ns,
        );
        self.queue(&ServerMessage::Welcome(&welcome));
        // A feed that ended before this client came: it is told so before anything it sent is
        // answered, and stays ended (#102). A lossless client would not hear it at all, as its
        // `Ordered::End` went out before it subscribed. A feed that ends after this is caught at
        // the next send, or at `Ordered::End`.
        if hub.has_ended() {
            self.mode = Mode::Ended(None);
            self.queue_ended();
        }
        for text in early {
            self.on_text(&text).await?;
        }
        self.flush().await?;
        if self.ordered.is_some() {
            self.lossless().await
        } else {
            self.coalescing().await
        }
    }

    async fn hello(&mut self) -> Result<(), Closing> {
        let first = timeout(HELLO_WITHIN, self.ws.next())
            .await
            .map_err(|_| bad("no hello"))?;
        let text = match first {
            Some(Ok(Message::Text(text))) => text,
            Some(Ok(Message::Close(_)) | Err(_)) | None => return Err(Closing::Gone),
            Some(Ok(_)) => return Err(bad("the first message must be hello")),
        };
        match serde_json::from_str::<ClientMessage>(&text) {
            Ok(ClientMessage::Hello { v }) if v == PROTOCOL => Ok(()),
            Ok(ClientMessage::Hello { v }) => Err(Closing::Code(
                close::VERSION_MISMATCH,
                format!("protocol {v} asked; this engine speaks {PROTOCOL}"),
            )),
            _ => Err(bad("the first message must be hello")),
        }
    }

    /// Normal clients: send at most every [`FLUSH_EVERY`], newest values only.
    async fn coalescing(&mut self) -> Result<(), Closing> {
        let mut heartbeat = interval(HEARTBEAT);
        heartbeat.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut next_flush = Instant::now();
        loop {
            tokio::select! {
                biased;
                incoming = self.ws.next() => self.on_incoming(incoming).await?,
                // The heartbeat and the replay before the hub's wake-up: a hub publishing faster
                // than this connection sends keeps the wake-up ready every turn, so an arm after
                // it was never polled (#85). The wake-up losing a turn costs nothing: `collect`
                // reads every channel on every turn.
                _ = heartbeat.tick() => self.status_due = true,
                item = next_item(&mut self.mode) => self.on_replay(item),
                changed = self.changed.changed() => {
                    if changed.is_err() {
                        return Ok(());
                    }
                }
            }
            sleep_until(next_flush).await;
            self.collect(&mut heartbeat)?;
            if !self.outbox.is_empty() {
                self.flush().await?;
                next_flush = Instant::now() + FLUSH_EVERY;
            }
        }
    }

    /// Harness clients of a lossless hub: every message, in publish order, as it comes.
    async fn lossless(&mut self) -> Result<(), Closing> {
        loop {
            let Some(ordered) = self.ordered.as_mut() else {
                return Ok(());
            };
            tokio::select! {
                biased;
                incoming = self.ws.next() => self.on_incoming(incoming).await?,
                item = ordered.recv() => match item {
                    Ok(item) => self.on_ordered(item),
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        return Err(Closing::Code(close::TOO_SLOW, "fell behind a lossless feed".to_owned()));
                    }
                    Err(broadcast::error::RecvError::Closed) => return Ok(()),
                },
            }
            if !self.outbox.is_empty() {
                self.flush().await?;
            }
        }
    }

    fn queue(&mut self, message: &ServerMessage<'_>) {
        self.outbox.push(Message::text(message.to_text()));
    }

    fn queue_book(&mut self, sample: &Sample) {
        for row in &sample.rows {
            if self.sub.wants(sample.venue, &row.symbol) {
                let book = BookMessage::new(sample, row, self.sub.levels);
                self.outbox
                    .push(Message::text(ServerMessage::Book(&book).to_text()));
            }
        }
    }

    fn queue_trades(&mut self, trades: &[Arc<TradeMessage>]) {
        let wanted: Vec<TradeMessage> = trades
            .iter()
            .filter(|trade| self.sub.wants(trade.venue, &trade.symbol))
            .map(|trade| TradeMessage::clone(trade))
            .collect();
        if !wanted.is_empty() {
            self.queue(&ServerMessage::Trades { trades: &wanted });
        }
    }

    /// A `gap` on an append topic, when the client lost any of it.
    fn queue_gap(&mut self, topic: Topic, lost: u64) {
        if lost > 0 {
            self.queue(&ServerMessage::Gap { topic, lost });
        }
    }

    fn queue_source(&mut self, mode: &'static str) {
        let message = match &self.mode {
            Mode::Replay {
                run,
                from_ms,
                speed,
                ..
            } => SourceMessage {
                mode,
                run: Some(run.clone()),
                from_ms: *from_ms,
                speed: Some(*speed),
                error: None,
            },
            Mode::Live | Mode::Ended(_) => SourceMessage {
                mode,
                run: None,
                from_ms: None,
                speed: None,
                error: None,
            },
        };
        self.queue(&ServerMessage::Source(&message));
    }

    /// The source of a connection whose replay or live feed is over: `ended`, or `failed` and
    /// why, naming the replay's run.
    fn queue_ended(&mut self) {
        let message = match &self.mode {
            Mode::Ended(Some(last)) => SourceMessage {
                mode: if last.error.is_some() {
                    "failed"
                } else {
                    "ended"
                },
                run: Some(last.run.clone()),
                from_ms: None,
                speed: None,
                error: last.error.clone(),
            },
            Mode::Live | Mode::Replay { .. } | Mode::Ended(None) => SourceMessage {
                mode: "ended",
                run: None,
                from_ms: None,
                speed: None,
                error: None,
            },
        };
        self.queue(&ServerMessage::Source(&message));
    }

    /// Sends everything queued, giving the socket the stall timeout to take it.
    async fn flush(&mut self) -> Result<(), Closing> {
        let messages = std::mem::take(&mut self.outbox);
        let ws = &mut self.ws;
        let sent = timeout(self.api.config.ws_stall, async move {
            for message in messages {
                ws.feed(message).await?;
            }
            ws.flush().await
        })
        .await;
        match sent {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err(Closing::Gone),
            Err(_) => Err(Closing::Code(
                close::TOO_SLOW,
                "the client stopped reading".to_owned(),
            )),
        }
    }

    async fn on_incoming(
        &mut self,
        incoming: Option<Result<Message, WsError>>,
    ) -> Result<(), Closing> {
        match incoming {
            Some(Ok(Message::Text(text))) => self.on_text(&text).await,
            Some(Ok(Message::Binary(_))) => Err(bad("the engine takes text messages only")),
            Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => Ok(()),
            Some(Err(WsError::Capacity(_))) => Err(bad("message too large")),
            Some(Ok(Message::Close(_)) | Err(_)) | None => Err(Closing::Gone),
        }
    }

    async fn on_text(&mut self, text: &str) -> Result<(), Closing> {
        let message = serde_json::from_str::<ClientMessage>(text)
            .map_err(|error| bad(format!("unreadable message: {error}")))?;
        self.on_client(message).await
    }

    async fn on_client(&mut self, message: ClientMessage) -> Result<(), Closing> {
        match message {
            ClientMessage::Hello { .. } => Err(bad("hello was already said")),
            ClientMessage::Sub {
                topics,
                venues,
                symbols,
                levels,
                backfill_s,
            } => self.subscribe(
                topics,
                venues.map(|venues| venues.into_iter().collect()),
                symbols.map(|symbols| symbols.into_iter().collect()),
                levels,
                backfill_s,
            ),
            ClientMessage::Replay {
                run,
                from_ms,
                speed,
            } => self.replay(run, from_ms, speed).await,
            ClientMessage::Seek { to_ms } => {
                // A replay that is over can still be sought in: it starts again from there.
                let (Mode::Replay { run, speed, .. }
                | Mode::Ended(Some(LastReplay { run, speed, .. }))) = &self.mode
                else {
                    return Err(bad("seek needs a replay"));
                };
                let (run, speed) = (run.clone(), *speed);
                self.replay(run, Some(to_ms), speed).await
            }
            ClientMessage::Pause | ClientMessage::Play => {
                let play = matches!(message, ClientMessage::Play);
                let (replay, paused) = match &mut self.mode {
                    Mode::Replay { replay, paused, .. } => (replay, paused),
                    // Sent as the replay ran out, before the client heard: it is over, and the
                    // client is told so again.
                    Mode::Ended(_) => {
                        self.queue_ended();
                        return Ok(());
                    }
                    Mode::Live => return Err(bad("pause and play need a replay")),
                };
                if play {
                    replay.play();
                } else {
                    replay.pause();
                }
                *paused = !play;
                if !play {
                    // What came before the pause goes out before it; anything the replay had
                    // already queued waits for play.
                    self.queue_replayed();
                }
                self.queue_source(if play { "replay" } else { "paused" });
                Ok(())
            }
            ClientMessage::Live => {
                self.mode = Mode::Live;
                self.last_book.clear();
                self.book.mark_changed();
                self.trades = self
                    .sub
                    .has(Topic::Trades)
                    .then(|| self.api.hub.subscribe_trades(Duration::ZERO).1);
                self.queue_source("live");
                Ok(())
            }
        }
    }

    fn subscribe(
        &mut self,
        topics: Vec<Topic>,
        venues: Option<BTreeSet<Venue>>,
        symbols: Option<BTreeSet<String>>,
        levels: Option<usize>,
        backfill_s: Option<f64>,
    ) -> Result<(), Closing> {
        let backfill = super::backfill(backfill_s).map_err(|super::ApiError(_, why)| bad(why))?;
        let hub = Arc::clone(&self.api.hub);
        let most = hub.config().book_sample.levels;
        self.sub = Sub {
            topics: topics.into_iter().collect(),
            venues,
            symbols,
            levels: levels.unwrap_or(most).clamp(1, most),
        };
        self.jev = self
            .sub
            .has(Topic::Jev)
            .then(|| hub.subscribe_relayed(Topic::Jev))
            .flatten();
        self.ising = self
            .sub
            .has(Topic::Ising)
            .then(|| hub.subscribe_relayed(Topic::Ising))
            .flatten();
        self.sent_meters = Latest::default();
        self.sent_pnl = Latest::default();
        self.meters.mark_changed();
        self.pnl.mark_changed();
        self.status.mark_changed();
        if matches!(self.mode, Mode::Live) && self.ordered.is_none() {
            self.last_book.clear();
            let (past, trades) = hub.subscribe_trades(backfill);
            self.trades = self.sub.has(Topic::Trades).then_some(trades);
            if self.sub.has(Topic::Book) {
                for sample in &past.samples {
                    self.queue_book(sample);
                    let last = self.last_book.entry(sample.venue).or_insert(i64::MIN);
                    *last = (*last).max(sample.sample_ts_ns);
                }
            }
            if self.sub.has(Topic::Trades) {
                self.queue_trades(&past.trades);
            }
            self.book.mark_changed();
        }
        match self.mode {
            Mode::Live => self.queue_source("live"),
            Mode::Replay { paused: true, .. } => self.queue_source("paused"),
            Mode::Replay { .. } => self.queue_source("replay"),
            Mode::Ended(_) => self.queue_ended(),
        }
        Ok(())
    }

    async fn replay(
        &mut self,
        run: String,
        from_ms: Option<i64>,
        speed: f64,
    ) -> Result<(), Closing> {
        if !super::replay::valid_speed(speed) {
            return Err(bad(format!(
                "speed is 0 (as fast as possible) or at least {}",
                super::replay::MIN_SPEED
            )));
        }
        let replay = Replay::start(
            &self.api.config.data_dir,
            Arc::clone(&self.api.hub),
            &run,
            from_ms,
            speed,
            self.api.config.open_delay,
        )
        .await
        .map_err(|error| match error {
            TableError::BadName(why) | TableError::NotFound(why) => bad(why),
            TableError::Store(error) => bad(format!("cannot replay {run}: {error}")),
        })?;
        self.mode = Mode::Replay {
            replay,
            run,
            from_ms,
            speed,
            paused: false,
        };
        self.replay_book = None;
        self.replay_trades.clear();
        self.trades = None;
        self.queue_source("replay");
        Ok(())
    }

    fn on_replay(&mut self, first: Option<Item>) {
        let mut next = first;
        loop {
            match next {
                Some(Item::Sample(sample)) => self.replay_book = Some(sample),
                Some(Item::Trade(trade)) => self.replay_trades.push(trade),
                Some(Item::BookGap(lost)) => {
                    // Everything before the gap goes out before it.
                    self.queue_replayed();
                    if self.sub.has(Topic::Book) {
                        self.queue(&ServerMessage::Gap {
                            topic: Topic::Book,
                            lost,
                        });
                    }
                }
                Some(Item::End | Item::Failed(_)) | None => {
                    // What is left still goes out, then the source says the run is over, or
                    // that the replay failed and why.
                    let error = match next {
                        Some(Item::Failed(reason)) => Some(reason),
                        None => Some("the replay stopped".to_owned()),
                        _ => None,
                    };
                    self.queue_replayed();
                    let last = match std::mem::replace(&mut self.mode, Mode::Ended(None)) {
                        Mode::Replay { run, speed, .. } => Some(LastReplay { run, speed, error }),
                        Mode::Live | Mode::Ended(_) => None,
                    };
                    self.mode = Mode::Ended(last);
                    self.queue_ended();
                    return;
                }
            }
            let Mode::Replay { replay, .. } = &mut self.mode else {
                return;
            };
            match replay.items.try_recv() {
                Ok(item) => next = Some(item),
                Err(_) => return,
            }
        }
    }

    fn on_ordered(&mut self, item: Ordered) {
        match item {
            Ordered::Sample(published) => {
                if self.sub.has(Topic::Book) {
                    self.queue_book(&published.sample);
                }
            }
            Ordered::Trade(trade) => {
                if self.sub.has(Topic::Trades) {
                    self.queue_trades(&[trade]);
                }
            }
            Ordered::Status(status) => {
                if self.sub.has(Topic::Status) {
                    self.queue(&ServerMessage::Status(&status));
                }
            }
            Ordered::Relay(topic, data) => {
                if self.sub.has(topic) {
                    self.queue(&ServerMessage::relay(topic, &data));
                }
            }
            Ordered::End => {
                self.mode = Mode::Ended(None);
                self.queue_ended();
            }
        }
    }

    /// Moves what changed since the last send into the outbox, newest values only. A status sent
    /// starts the `heartbeat`'s second again.
    fn collect(&mut self, heartbeat: &mut Interval) -> Result<(), Closing> {
        match &self.mode {
            Mode::Live => self.collect_live(),
            Mode::Replay { paused: false, .. } => self.queue_replayed(),
            Mode::Replay { paused: true, .. } | Mode::Ended(_) => {}
        }
        if self.status.has_changed().unwrap_or(false) || self.status_due {
            self.status_due = false;
            let latest = self.status.borrow_and_update().clone();
            if let Some(status) = latest
                && self.sub.has(Topic::Status)
            {
                self.queue(&ServerMessage::Status(&status));
                // Unchanged, it goes again a second from now, not on this one's heels: a client
                // subscribing at once got its status twice, 16 ms apart, the second from the
                // heartbeat's first tick (#102).
                heartbeat.reset();
            }
        }
        self.collect_latest(Topic::Meters);
        self.collect_latest(Topic::Pnl);
        if let Some(jev) = self.jev.as_mut() {
            let (fresh, lost) = drain(jev);
            if lost > 0 {
                return Err(Closing::Code(
                    close::TOO_SLOW,
                    "fell behind on jev, which is never dropped".to_owned(),
                ));
            }
            for data in fresh {
                self.queue(&ServerMessage::relay(Topic::Jev, &data));
            }
        }
        if let Some(ising) = self.ising.as_mut() {
            let (fresh, lost) = drain(ising);
            self.queue_gap(Topic::Ising, lost);
            for data in fresh {
                self.queue(&ServerMessage::relay(Topic::Ising, &data));
            }
        }
        Ok(())
    }

    /// The replay's newest book and its trades since the last send.
    fn queue_replayed(&mut self) {
        if let Some(sample) = self.replay_book.take()
            && self.sub.has(Topic::Book)
        {
            self.queue_book(&sample);
        }
        let trades = std::mem::take(&mut self.replay_trades);
        if self.sub.has(Topic::Trades) {
            self.queue_trades(&trades);
        }
    }

    fn collect_live(&mut self) {
        // Read first: the feed ends after its last book and trade, so once this says ended,
        // everything drained below is everything there was, and `ended` goes out after it.
        let ended = self.api.hub.has_ended();
        // Only while subscribed, so the newest book is still unsent when a subscription starts.
        // Each venue's newest book goes out unless it went already.
        if self.sub.has(Topic::Book) && self.book.has_changed().unwrap_or(false) {
            let latest = self.book.borrow_and_update().clone();
            let fresh: Vec<&Arc<Published>> = latest
                .iter()
                .filter(|(venue, published)| {
                    let last = self.last_book.entry(**venue).or_insert(i64::MIN);
                    let newer = published.sample.sample_ts_ns > *last;
                    *last = (*last).max(published.sample.sample_ts_ns);
                    newer
                })
                .map(|(_, published)| published)
                .collect();
            for published in fresh {
                self.queue_book(&published.sample);
            }
        }
        if let Some(trades) = self.trades.as_mut() {
            let (fresh, lost) = drain(trades);
            self.queue_gap(Topic::Trades, lost);
            self.queue_trades(&fresh);
        }
        if ended {
            // Ended as a lossless connection is at `Ordered::End`, so a `sub` after this is
            // answered `ended`, not `live` (#102).
            self.mode = Mode::Ended(None);
            self.queue_ended();
        }
    }

    fn collect_latest(&mut self, topic: Topic) {
        let (rx, sent) = match topic {
            Topic::Meters => (&mut self.meters, &mut self.sent_meters),
            _ => (&mut self.pnl, &mut self.sent_pnl),
        };
        if !rx.has_changed().unwrap_or(false) {
            return;
        }
        let latest = rx.borrow_and_update().clone();
        let fresh: Vec<Arc<Value>> = latest
            .iter()
            .filter(|(key, value)| !sent.get(*key).is_some_and(|old| Arc::ptr_eq(old, value)))
            .map(|(_, value)| Arc::clone(value))
            .collect();
        *sent = latest;
        if self.sub.has(topic) {
            for data in fresh {
                self.queue(&ServerMessage::relay(topic, &data));
            }
        }
    }
}

/// Everything waiting on an append topic's channel, and how many items the client lost by falling
/// behind: one drain for jev, ising and trades, which differ only in what a loss means (#87).
fn drain<T: Clone>(rx: &mut broadcast::Receiver<T>) -> (Vec<T>, u64) {
    let (mut fresh, mut lost) = (Vec::new(), 0);
    loop {
        match rx.try_recv() {
            Ok(item) => fresh.push(item),
            Err(TryRecvError::Lagged(n)) => lost += n,
            Err(TryRecvError::Empty | TryRecvError::Closed) => return (fresh, lost),
        }
    }
}

/// The next replay item, or never when the connection is not replaying.
async fn next_item(mode: &mut Mode) -> Option<Item> {
    match mode {
        Mode::Replay { replay, .. } => replay.items.recv().await,
        Mode::Live | Mode::Ended(_) => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upgrade_tokens_match_in_any_case_and_list() {
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONNECTION,
            HeaderValue::from_static("keep-alive, Upgrade"),
        );
        assert!(has_token(&headers, &header::CONNECTION, "upgrade"));
        assert!(!has_token(&headers, &header::UPGRADE, "websocket"));
    }
}
