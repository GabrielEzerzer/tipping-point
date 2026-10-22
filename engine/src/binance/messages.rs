//! Wire shapes of the Binance USD-M streams and REST replies, and their conversion to integer
//! units. Prices and quantities arrive as decimal strings and are parsed exactly.

use serde::Deserialize;
use thiserror::Error;

use super::Scales;
use super::decimal::{DecimalError, parse_units};
use super::sync::{DepthUpdate, Level, Snapshot};

/// One message from a combined stream: `{"stream": "btcusdt@depth@100ms", "data": {...}}`.
#[derive(Debug, Clone, Deserialize)]
pub struct CombinedFrame {
    pub stream: String,
    pub data: Payload,
}

/// The `data` of a combined frame, picked by its `e` event type.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "e")]
pub enum Payload {
    #[serde(rename = "depthUpdate")]
    Depth(RawDepthUpdate),
    #[serde(rename = "aggTrade")]
    AggTrade(RawAggTrade),
}

/// A `depthUpdate` event as sent: ids plus `[price, qty]` string pairs.
#[derive(Debug, Clone, Deserialize)]
pub struct RawDepthUpdate {
    #[serde(rename = "E")]
    pub event_time_ms: u64,
    #[serde(rename = "T")]
    pub transaction_time_ms: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub first_update_id: u64,
    #[serde(rename = "u")]
    pub last_update_id: u64,
    #[serde(rename = "pu")]
    pub prev_last_update_id: u64,
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
}

/// An `aggTrade` event as sent.
#[derive(Debug, Clone, Deserialize)]
pub struct RawAggTrade {
    #[serde(rename = "E")]
    pub event_time_ms: u64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "a")]
    pub agg_trade_id: u64,
    #[serde(rename = "p")]
    pub price: String,
    #[serde(rename = "q")]
    pub qty: String,
    /// The first and last trade ids aggregated; kept for the `agg_trades` table, absent
    /// fields read as `None` so they never fail a frame.
    #[serde(rename = "f", default)]
    pub first_trade_id: Option<u64>,
    #[serde(rename = "l", default)]
    pub last_trade_id: Option<u64>,
    #[serde(rename = "T")]
    pub trade_time_ms: u64,
    #[serde(rename = "m")]
    pub buyer_is_maker: bool,
}

/// The REST `/fapi/v1/depth` reply.
#[derive(Debug, Clone, Deserialize)]
pub struct RawSnapshot {
    #[serde(rename = "lastUpdateId")]
    pub last_update_id: u64,
    pub bids: Vec<[String; 2]>,
    pub asks: Vec<[String; 2]>,
}

/// An aggregated trade in integer units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggTrade {
    pub symbol: String,
    pub agg_trade_id: u64,
    pub price: i64,
    pub qty: i64,
    pub trade_time_ms: u64,
    pub buyer_is_maker: bool,
}

/// Why a message could not be understood.
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("malformed json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("bad decimal: {0}")]
    Decimal(#[from] DecimalError),
}

/// Parses the text of one combined-stream frame.
///
/// # Errors
/// Returns an error when the frame is not valid JSON of a known event type.
pub fn parse_frame(text: &str) -> Result<CombinedFrame, MessageError> {
    Ok(serde_json::from_str(text)?)
}

/// Parses the body of a REST depth snapshot.
///
/// # Errors
/// Returns an error when the body is not a valid snapshot.
pub fn parse_snapshot(body: &str) -> Result<RawSnapshot, MessageError> {
    Ok(serde_json::from_str(body)?)
}

impl Scales {
    /// Converts a raw depth event into integer units.
    ///
    /// # Errors
    /// Returns an error when a price or quantity is not representable at this scale.
    pub fn depth_update(&self, raw: &RawDepthUpdate) -> Result<DepthUpdate, DecimalError> {
        Ok(DepthUpdate {
            first_id: raw.first_update_id,
            last_id: raw.last_update_id,
            prev_last_id: raw.prev_last_update_id,
            bids: self.levels(&raw.bids)?,
            asks: self.levels(&raw.asks)?,
        })
    }

    /// Converts a raw snapshot into integer units.
    ///
    /// # Errors
    /// Returns an error when a price or quantity is not representable at this scale.
    pub fn snapshot(&self, raw: &RawSnapshot) -> Result<Snapshot, DecimalError> {
        Ok(Snapshot {
            last_update_id: raw.last_update_id,
            bids: self.levels(&raw.bids)?,
            asks: self.levels(&raw.asks)?,
        })
    }

    /// Converts a raw aggregated trade into integer units.
    ///
    /// # Errors
    /// Returns an error when the price or quantity is not representable at this scale.
    pub fn agg_trade(&self, raw: &RawAggTrade) -> Result<AggTrade, DecimalError> {
        Ok(AggTrade {
            symbol: raw.symbol.clone(),
            agg_trade_id: raw.agg_trade_id,
            price: parse_units(&raw.price, self.price_decimals)?,
            qty: parse_units(&raw.qty, self.qty_decimals)?,
            trade_time_ms: raw.trade_time_ms,
            buyer_is_maker: raw.buyer_is_maker,
        })
    }

    fn levels(self, pairs: &[[String; 2]]) -> Result<Vec<Level>, DecimalError> {
        pairs
            .iter()
            .map(|[price, qty]| {
                Ok(Level {
                    price: parse_units(price, self.price_decimals)?,
                    qty: parse_units(qty, self.qty_decimals)?,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEPTH: &str = r#"{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1700000000123,"T":1700000000120,"s":"BTCUSDT","U":100,"u":105,"pu":99,"b":[["50000.10","1.500"],["49999.90","0.000"]],"a":[["50000.20","0.250"]]}}"#;
    const TRADE: &str = r#"{"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1700000000123,"a":42,"s":"BTCUSDT","p":"50000.10","q":"0.003","f":1,"l":2,"T":1700000000120,"m":true}}"#;
    const SNAPSHOT: &str = r#"{"lastUpdateId":104,"E":1,"T":1,"bids":[["50000.00","2.000"]],"asks":[["50000.30","1.000"]]}"#;

    const BTC: Scales = Scales {
        price_decimals: 1,
        qty_decimals: 3,
    };

    #[test]
    fn parses_a_depth_frame_into_units() -> Result<(), MessageError> {
        let frame = parse_frame(DEPTH)?;
        assert_eq!(frame.stream, "btcusdt@depth@100ms");
        let Payload::Depth(raw) = frame.data else {
            panic!("expected a depth payload");
        };
        let update = BTC.depth_update(&raw)?;
        assert_eq!(
            (update.first_id, update.last_id, update.prev_last_id),
            (100, 105, 99)
        );
        assert_eq!(
            update.bids,
            vec![
                Level {
                    price: 500_001,
                    qty: 1500
                },
                Level {
                    price: 499_999,
                    qty: 0
                }
            ]
        );
        assert_eq!(
            update.asks,
            vec![Level {
                price: 500_002,
                qty: 250
            }]
        );
        Ok(())
    }

    #[test]
    fn parses_an_agg_trade_frame() -> Result<(), MessageError> {
        let frame = parse_frame(TRADE)?;
        let Payload::AggTrade(raw) = frame.data else {
            panic!("expected an aggTrade payload");
        };
        assert_eq!((raw.first_trade_id, raw.last_trade_id), (Some(1), Some(2)));
        let trade = BTC.agg_trade(&raw)?;
        assert_eq!(trade.agg_trade_id, 42);
        assert_eq!(trade.price, 500_001);
        assert_eq!(trade.qty, 3);
        assert!(trade.buyer_is_maker);
        Ok(())
    }

    #[test]
    fn parses_a_snapshot() -> Result<(), MessageError> {
        let snapshot = BTC.snapshot(&parse_snapshot(SNAPSHOT)?)?;
        assert_eq!(snapshot.last_update_id, 104);
        assert_eq!(snapshot.bids[0].price, 500_000);
        assert_eq!(snapshot.asks[0].qty, 1000);
        Ok(())
    }

    #[test]
    fn rejects_unknown_event_types_and_bad_precision() {
        let unknown = r#"{"stream":"x","data":{"e":"bookTicker","s":"BTCUSDT"}}"#;
        assert!(matches!(parse_frame(unknown), Err(MessageError::Json(_))));

        let frame = parse_frame(DEPTH).expect("fixture parses");
        let Payload::Depth(raw) = frame.data else {
            panic!("expected a depth payload");
        };
        let coarse = Scales {
            price_decimals: 0,
            qty_decimals: 3,
        };
        assert!(coarse.depth_update(&raw).is_err());
    }
}
