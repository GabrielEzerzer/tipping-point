//! Binance USD-M futures ingest: live depth and trade streams kept as a correct local order book.
//!
//! The sync logic is a pure state machine ([`sync::BookSync`]) wrapped by a multi-symbol
//! [`feed::Feed`] that turns raw frames into book updates, and [`pipeline`] holds the one copy
//! of how the engine reacts to every input. Everything that touches the network ([`ws`],
//! [`rest`], [`live`]) is a thin layer around it, so the same pipeline runs on a live socket, a
//! replayed log, a recording, or a synthetic test stream.

pub mod book;
pub mod decimal;
pub mod feed;
pub mod live;
pub mod messages;
pub mod pipeline;
pub mod recording;
pub mod rest;
pub mod sync;
pub mod ws;

/// An exchange symbol in Binance's upper-case form, e.g. `BTCUSDT`.
pub type Symbol = String;

/// How a symbol's decimal strings map to integer units: a price of `"12345.6"` with one price
/// decimal is `123456` ticks, a quantity of `"0.001"` with three quantity decimals is `1` lot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Scales {
    pub price_decimals: u32,
    pub qty_decimals: u32,
}
