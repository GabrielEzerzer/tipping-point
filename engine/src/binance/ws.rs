//! The WebSocket reader: one combined-stream connection, kept alive and reconnected, handing
//! raw text frames to a bounded channel without ever blocking on it.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc::{self, error::TrySendError};
use tokio::time::{Instant, sleep, sleep_until, timeout};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{WebSocketStream, connect_async};

use super::Symbol;

/// Routed path that serves `@depth@100ms`.
pub const DEPTH_HOST: &str = "wss://fstream.binance.com/public/stream";
/// Routed path that serves `@aggTrade` (it is not delivered on `/public`).
pub const TRADE_HOST: &str = "wss://fstream.binance.com/market/stream";

/// Binance closes a connection after 24 h; leaving before that keeps the reconnect on our terms.
pub const DEFAULT_MAX_CONNECTION_AGE: Duration = Duration::from_hours(23);

const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Both streams send several frames a second, so this long with nothing at all means the
/// connection is dead even if TCP never said so (a half-open socket after a Wi-Fi drop or NAT
/// expiry, with no FIN or RST): treat it as a disconnect.
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);

/// A drop mid-handshake leaves the peer silent rather than closed, so a connect (TCP, TLS and
/// the WebSocket upgrade) still unfinished after this long is given up and retried.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// What the reader sends down its channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inbound {
    /// A connection is open; `attempt` counts connections since the reader started.
    Connected { attempt: u32 },
    /// A connection attempt failed; the reader will back off and retry.
    ConnectFailed { error: String },
    /// A text frame. `dropped_before` is how many frames were discarded on a full channel
    /// since the last frame that got through.
    Text { text: String, dropped_before: u64 },
    /// The connection ended; the reader will reconnect. Any book is now stale. `dropped_before`
    /// counts frames discarded since the last one that got through, as on [`Inbound::Text`].
    Disconnected { reason: String, dropped_before: u64 },
}

/// Combined-stream URL for the given symbols and stream suffix, e.g. `depth@100ms`.
#[must_use]
pub fn stream_url(host: &str, symbols: &[Symbol], suffix: &str) -> String {
    let streams: Vec<String> = symbols
        .iter()
        .map(|s| format!("{}@{suffix}", s.to_lowercase()))
        .collect();
    format!("{host}?streams={}", streams.join("/"))
}

/// Reads `url` forever, reconnecting with backoff, until the receiver is dropped.
///
/// Frames go out with `try_send`: when the channel is full the frame is dropped and counted
/// rather than waited on, because a reader stuck on a full channel stops answering pings and
/// loses the connection for everyone.
pub async fn run_reader(url: String, max_age: Duration, tx: mpsc::Sender<Inbound>) {
    let mut backoff = INITIAL_BACKOFF;
    let mut attempt = 0;
    loop {
        attempt += 1;
        let connected = match timeout(CONNECT_TIMEOUT, connect_async(&url)).await {
            Ok(result) => result.map_err(|error| error.to_string()),
            Err(_) => Err(format!(
                "no handshake within {}s",
                CONNECT_TIMEOUT.as_secs()
            )),
        };
        let socket = match connected {
            Ok((socket, _response)) => socket,
            Err(error) => {
                if tx.send(Inbound::ConnectFailed { error }).await.is_err() {
                    return;
                }
                sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
                continue;
            }
        };
        backoff = INITIAL_BACKOFF;
        if tx.send(Inbound::Connected { attempt }).await.is_err() {
            return;
        }
        let Some(disconnected) = read_until_closed(socket, max_age, &tx).await else {
            return;
        };
        if tx.send(disconnected).await.is_err() {
            return;
        }
        sleep(backoff).await;
    }
}

/// Pumps one connection. Returns the [`Inbound::Disconnected`] notice saying how it ended, or
/// `None` when the receiver is gone.
async fn read_until_closed<S>(
    mut socket: WebSocketStream<S>,
    max_age: Duration,
    tx: &mpsc::Sender<Inbound>,
) -> Option<Inbound>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let retire_at = Instant::now() + max_age;
    let mut dropped: u64 = 0;
    let reason = loop {
        let message = tokio::select! {
            message = timeout(IDLE_TIMEOUT, socket.next()) => message,
            () = sleep_until(retire_at) => {
                let _ = socket.close(None).await;
                break "max connection age reached".to_owned();
            }
        };
        let Ok(message) = message else {
            break format!("no frames for {}s", IDLE_TIMEOUT.as_secs());
        };
        match message {
            Some(Ok(Message::Text(text))) => {
                let frame = Inbound::Text {
                    text: text.to_string(),
                    dropped_before: dropped,
                };
                match tx.try_send(frame) {
                    Ok(()) => dropped = 0,
                    Err(TrySendError::Full(_)) => dropped += 1,
                    Err(TrySendError::Closed(_)) => return None,
                }
            }
            Some(Ok(Message::Ping(payload))) => {
                if let Err(error) = socket.send(Message::Pong(payload)).await {
                    break format!("pong failed: {error}");
                }
            }
            Some(Ok(Message::Close(frame))) => break format!("server closed: {frame:?}"),
            Some(Ok(Message::Binary(_) | Message::Pong(_) | Message::Frame(_))) => {}
            Some(Err(error)) => break format!("socket error: {error}"),
            None => break "stream ended".to_owned(),
        }
    };
    Some(Inbound::Disconnected {
        reason,
        dropped_before: dropped,
    })
}

#[cfg(test)]
mod tests {
    use tokio::io::{DuplexStream, duplex};
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::protocol::Role;

    use super::*;

    /// Our end of a socket and the exchange's end, joined in memory: no network.
    async fn socket_pair() -> (WebSocketStream<DuplexStream>, WebSocketStream<DuplexStream>) {
        let (ours, theirs) = duplex(1 << 16);
        (
            WebSocketStream::from_raw_socket(ours, Role::Client, None).await,
            WebSocketStream::from_raw_socket(theirs, Role::Server, None).await,
        )
    }

    fn text(text: &str, dropped_before: u64) -> Inbound {
        Inbound::Text {
            text: text.to_owned(),
            dropped_before,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn frames_keep_a_connection_open_and_silence_ends_it() {
        let (socket, mut exchange) = socket_pair().await;
        let (tx, mut rx) = mpsc::channel(16);
        let start = Instant::now();
        let reader =
            tokio::spawn(
                async move { read_until_closed(socket, Duration::from_hours(1), &tx).await },
            );

        // A frame every 20 s for a minute, then silence with the socket still open: half-open.
        for i in 0..3 {
            let frame = Message::text(format!("frame {i}"));
            exchange.send(frame).await.expect("in-memory send");
            sleep(Duration::from_secs(20)).await;
        }
        let ended = reader.await.expect("reader task");

        assert_eq!(
            ended,
            Some(Inbound::Disconnected {
                reason: "no frames for 30s".to_owned(),
                dropped_before: 0
            })
        );
        assert_eq!(
            start.elapsed().as_secs(),
            70,
            "30 s after the last frame, at 40 s"
        );
        for i in 0..3 {
            assert_eq!(rx.try_recv().ok(), Some(text(&format!("frame {i}"), 0)));
        }
        drop(exchange);
    }

    #[tokio::test]
    async fn frames_dropped_just_before_a_disconnect_are_reported() {
        let (socket, mut exchange) = socket_pair().await;
        for i in 0..3 {
            let frame = Message::text(format!("frame {i}"));
            exchange.send(frame).await.expect("in-memory send");
        }
        drop(exchange);
        // Room for one frame, so the other two are dropped and only the disconnect can say so.
        let (tx, mut rx) = mpsc::channel(1);
        let ended = read_until_closed(socket, Duration::from_hours(1), &tx).await;

        assert!(
            matches!(
                ended,
                Some(Inbound::Disconnected {
                    dropped_before: 2,
                    ..
                })
            ),
            "{ended:?}"
        );
        assert_eq!(rx.try_recv().ok(), Some(text("frame 0", 0)));
    }

    #[tokio::test]
    async fn a_peer_that_accepts_but_never_answers_the_handshake_fails_the_connect() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let url = format!("ws://{}/", listener.local_addr().expect("local address"));
        let (tx, mut rx) = mpsc::channel(16);
        let start = Instant::now();
        let reader = tokio::spawn(run_reader(url, Duration::from_hours(1), tx));

        // The TCP connection is made, then the peer stays silent: no HTTP 101, and no close.
        let (_silent_peer, _) = timeout(Duration::from_secs(10), listener.accept())
            .await
            .expect("the reader dials within 10 s")
            .expect("accept");
        tokio::time::pause();
        // Its own bound makes a missing connect timeout fail here instead of hanging (#42).
        let first = timeout(4 * CONNECT_TIMEOUT, rx.recv())
            .await
            .expect("still connecting at 4x the connect timeout");

        assert_eq!(
            first,
            Some(Inbound::ConnectFailed {
                error: format!("no handshake within {}s", CONNECT_TIMEOUT.as_secs())
            })
        );
        let elapsed = start.elapsed();
        assert!(
            (CONNECT_TIMEOUT..CONNECT_TIMEOUT + Duration::from_secs(1)).contains(&elapsed),
            "gave up after {elapsed:?}"
        );
        reader.abort();
    }

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
