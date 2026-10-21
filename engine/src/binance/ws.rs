//! The WebSocket reader: one combined-stream connection, kept alive and reconnected, handing
//! raw text frames to a bounded channel without ever blocking on it.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc::{self, error::TrySendError};
use tokio::time::{Instant, sleep, sleep_until};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use super::Symbol;

/// Routed path that serves `@depth@100ms`.
pub const DEPTH_HOST: &str = "wss://fstream.binance.com/public/stream";
/// Routed path that serves `@aggTrade` (it is not delivered on `/public`).
pub const TRADE_HOST: &str = "wss://fstream.binance.com/market/stream";

/// Binance closes a connection after 24 h; leaving before that keeps the reconnect on our terms.
pub const DEFAULT_MAX_CONNECTION_AGE: Duration = Duration::from_hours(23);

const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

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
    /// The connection ended; the reader will reconnect. Any book is now stale.
    Disconnected { reason: String },
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
        let socket = match connect_async(&url).await {
            Ok((socket, _response)) => socket,
            Err(error) => {
                let notice = Inbound::ConnectFailed {
                    error: error.to_string(),
                };
                if tx.send(notice).await.is_err() {
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
        let Some(reason) = read_until_closed(socket, max_age, &tx).await else {
            return;
        };
        if tx.send(Inbound::Disconnected { reason }).await.is_err() {
            return;
        }
        sleep(backoff).await;
    }
}

type Socket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Pumps one connection. Returns why it ended, or `None` when the receiver is gone.
async fn read_until_closed(
    mut socket: Socket,
    max_age: Duration,
    tx: &mpsc::Sender<Inbound>,
) -> Option<String> {
    let retire_at = Instant::now() + max_age;
    let mut dropped: u64 = 0;
    loop {
        let message = tokio::select! {
            message = socket.next() => message,
            () = sleep_until(retire_at) => {
                let _ = socket.close(None).await;
                return Some("max connection age reached".to_owned());
            }
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
                    return Some(format!("pong failed: {error}"));
                }
            }
            Some(Ok(Message::Close(frame))) => {
                return Some(format!("server closed: {frame:?}"));
            }
            Some(Ok(Message::Binary(_) | Message::Pong(_) | Message::Frame(_))) => {}
            Some(Err(error)) => return Some(format!("socket error: {error}")),
            None => return Some("stream ended".to_owned()),
        }
    }
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
