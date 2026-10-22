//! The venues the engine reads, and everything that differs between them: the wire format of
//! their streams, the rule that keeps a local book right, where scales come from, their ids,
//! clocks and trade conventions, and their endpoints. Everything else (books, the feed, the
//! pipeline, storage, the API) is shared, and reaches a venue's own code by matching on
//! [`Venue`].

pub mod binance;

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::market::decimal::DecimalError;
use crate::market::feed::AggTrade;
use crate::market::sync::{DepthUpdate, Snapshot};
use crate::market::{Scales, Symbol};

/// An exchange market the engine reads. Its id names it in run tags, manifests, tables and
/// messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Venue {
    /// Binance USD-M perpetual futures.
    #[serde(rename = "binance-usdm")]
    BinanceUsdm,
}

impl Venue {
    /// Every venue, in id order.
    pub const ALL: [Self; 1] = [Self::BinanceUsdm];

    /// The venue's id: lower case, digits and `-`, safe in a run tag.
    #[must_use]
    pub fn id(self) -> &'static str {
        match self {
            Self::BinanceUsdm => "binance-usdm",
        }
    }

    /// Reads one frame of this venue's depth or trade stream.
    ///
    /// # Errors
    /// Returns an error when the frame is not one this venue sends.
    pub fn parse(self, text: &str) -> Result<Parsed, MessageError> {
        match self {
            Self::BinanceUsdm => binance::messages::parse(text),
        }
    }

    /// The exchange time a stream frame carries, in nanoseconds since the epoch; 0 when it has
    /// none.
    #[must_use]
    pub fn event_time_ns(self, text: &str) -> i64 {
        match self {
            Self::BinanceUsdm => binance::messages::frame_event_time_ns(text),
        }
    }

    /// Reads the body of a snapshot fetched from the venue's REST API.
    ///
    /// # Errors
    /// Returns an error when the body is not a snapshot this venue sends.
    pub fn parse_snapshot(self, body: &str) -> Result<RawSnapshot, MessageError> {
        match self {
            Self::BinanceUsdm => binance::messages::parse_snapshot(body).map(RawSnapshot::Binance),
        }
    }

    /// The exchange time of a REST snapshot body, in nanoseconds since the epoch; 0 when it has
    /// none.
    #[must_use]
    pub fn snapshot_time_ns(self, body: &str) -> i64 {
        match self {
            Self::BinanceUsdm => binance::messages::snapshot_event_time_ns(body),
        }
    }
}

impl fmt::Display for Venue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl FromStr for Venue {
    type Err = String;

    fn from_str(id: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|venue| venue.id() == id)
            .ok_or_else(|| {
                let ids: Vec<&str> = Self::ALL.iter().map(|venue| venue.id()).collect();
                format!("no venue {id:?} (one of {})", ids.join(", "))
            })
    }
}

/// Why a frame could not be understood.
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("malformed json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("bad decimal: {0}")]
    Decimal(#[from] DecimalError),
}

/// One frame of a venue's stream, read by that venue's rules: what it is and whose, its prices
/// and quantities still decimal text.
#[derive(Debug, Clone)]
pub enum Parsed {
    /// A change to one symbol's book.
    Depth(RawDepth),
    /// One trade.
    Trade(RawTrade),
}

/// A depth frame as its venue sent it.
#[derive(Debug, Clone)]
pub enum RawDepth {
    Binance(binance::messages::RawDepthUpdate),
}

/// What a depth frame does to a book, in integer units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Depth {
    /// A change to chain onto the book.
    Update(DepthUpdate),
}

/// What a `depth_updates` row holds besides the levels: the frame's exchange times and the ids
/// its venue chains updates by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthIds {
    pub event_time_ms: u64,
    pub transaction_time_ms: u64,
    pub first_update_id: u64,
    pub last_update_id: u64,
    pub prev_last_update_id: u64,
}

impl RawDepth {
    /// The symbol whose book it changes.
    #[must_use]
    pub fn symbol(&self) -> &str {
        match self {
            Self::Binance(raw) => &raw.symbol,
        }
    }

    /// The symbol, taken out of the frame.
    #[must_use]
    pub fn into_symbol(self) -> Symbol {
        match self {
            Self::Binance(raw) => raw.symbol,
        }
    }

    /// The frame in integer units at `scales`.
    ///
    /// # Errors
    /// Returns an error when a price or quantity is not representable at this scale.
    pub fn convert(&self, scales: Scales) -> Result<Depth, DecimalError> {
        match self {
            Self::Binance(raw) => scales.depth_update(raw).map(Depth::Update),
        }
    }

    /// Its exchange times and update ids.
    #[must_use]
    pub fn ids(&self) -> DepthIds {
        match self {
            Self::Binance(raw) => DepthIds {
                event_time_ms: raw.event_time_ms,
                transaction_time_ms: raw.transaction_time_ms,
                first_update_id: raw.first_update_id,
                last_update_id: raw.last_update_id,
                prev_last_update_id: raw.prev_last_update_id,
            },
        }
    }
}

/// A REST snapshot body as its venue sent it.
#[derive(Debug, Clone)]
pub enum RawSnapshot {
    Binance(binance::messages::RawSnapshot),
}

impl RawSnapshot {
    /// The snapshot in integer units at `scales`.
    ///
    /// # Errors
    /// Returns an error when a price or quantity is not representable at this scale.
    pub fn convert(&self, scales: Scales) -> Result<Snapshot, DecimalError> {
        match self {
            Self::Binance(raw) => scales.snapshot(raw),
        }
    }
}

/// A trade frame as its venue sent it.
#[derive(Debug, Clone)]
pub enum RawTrade {
    Binance(binance::messages::RawAggTrade),
}

/// What a trades row holds besides the trade itself: the frame's exchange event time and, for
/// an aggregated trade, the first and last trade ids it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TradeIds {
    pub event_time_ms: u64,
    pub first_trade_id: Option<u64>,
    pub last_trade_id: Option<u64>,
}

impl RawTrade {
    /// The symbol traded.
    #[must_use]
    pub fn symbol(&self) -> &str {
        match self {
            Self::Binance(raw) => &raw.symbol,
        }
    }

    /// The trade in integer units at `scales`.
    ///
    /// # Errors
    /// Returns an error when the price or quantity is not representable at this scale.
    pub fn convert(&self, scales: Scales) -> Result<AggTrade, DecimalError> {
        match self {
            Self::Binance(raw) => scales.agg_trade(raw),
        }
    }

    /// Its exchange event time and the trade ids it covers.
    #[must_use]
    pub fn ids(&self) -> TradeIds {
        match self {
            Self::Binance(raw) => TradeIds {
                event_time_ms: raw.event_time_ms,
                first_trade_id: raw.first_trade_id,
                last_trade_id: raw.last_trade_id,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_venue_is_named_by_its_id_everywhere() -> Result<(), String> {
        for venue in Venue::ALL {
            assert_eq!(venue.to_string(), venue.id());
            assert_eq!(venue.id().parse::<Venue>()?, venue);
            assert_eq!(
                serde_json::to_string(&venue).map_err(|e| e.to_string())?,
                format!("\"{}\"", venue.id())
            );
            let ok = venue
                .id()
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
            assert!(ok, "{venue} is not safe in a run tag");
        }
        assert!("binance".parse::<Venue>().is_err());
        Ok(())
    }
}
