//! Wire shapes of the Binance USD-M streams and REST replies, and their conversion to integer
//! units. Prices and quantities arrive as decimal strings and are parsed exactly.

use std::borrow::Cow;
use std::fmt;

use serde::Deserialize;
use serde::de::{self, Deserializer, Visitor};

use crate::market::Scales;
use crate::market::decimal::{DecimalError, parse_units};
use crate::market::feed::Trade;
use crate::market::sync::{DepthUpdate, Snapshot};
use crate::store::ms_to_ns;
use crate::venue::{MessageError, Parsed, RawDepth, RawTrade};

/// One message from a combined stream: `{"stream": "btcusdt@depth@100ms", "data": {...}}`.
#[derive(Debug, Clone)]
pub struct CombinedFrame {
    pub stream: String,
    pub data: Payload,
}

/// The `data` of a combined frame, picked by its `e` event type.
#[derive(Debug, Clone)]
pub enum Payload {
    Depth(RawDepthUpdate),
    AggTrade(RawAggTrade),
}

/// The event types [`parse_frame`] reads.
const EVENTS: &[&str] = &["depthUpdate", "aggTrade"];

/// A decimal string as sent, kept inline: a price or a quantity is a few characters and a depth
/// frame holds hundreds of them, so a heap allocation for each was most of what reading a frame
/// cost (#68).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Decimal {
    len: u8,
    bytes: [u8; Decimal::MAX],
}

impl Decimal {
    /// The longest decimal string read; Binance's run to a dozen or so characters.
    pub const MAX: usize = 31;

    /// The text as sent.
    #[must_use]
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.len)]).unwrap_or_default()
    }
}

impl AsRef<str> for Decimal {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Debug for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<'de> Deserialize<'de> for Decimal {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Text;
        impl Visitor<'_> for Text {
            type Value = Decimal;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a decimal string of at most {} bytes", Decimal::MAX)
            }

            fn visit_str<E: de::Error>(self, text: &str) -> Result<Decimal, E> {
                let mut bytes = [0; Decimal::MAX];
                bytes
                    .get_mut(..text.len())
                    .ok_or_else(|| E::invalid_length(text.len(), &self))?
                    .copy_from_slice(text.as_bytes());
                let len =
                    u8::try_from(text.len()).map_err(|_| E::invalid_length(text.len(), &self))?;
                Ok(Decimal { len, bytes })
            }
        }
        deserializer.deserialize_str(Text)
    }
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
    pub bids: Vec<[Decimal; 2]>,
    #[serde(rename = "a")]
    pub asks: Vec<[Decimal; 2]>,
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
    pub bids: Vec<[Decimal; 2]>,
    pub asks: Vec<[Decimal; 2]>,
}

/// A combined frame's stream and event type, borrowed, the rest of it skipped.
#[derive(Deserialize)]
struct Peek<'a> {
    #[serde(borrow)]
    stream: Cow<'a, str>,
    #[serde(borrow)]
    data: EventType<'a>,
}

#[derive(Deserialize)]
struct EventType<'a> {
    #[serde(borrow)]
    e: Cow<'a, str>,
}

/// A combined frame whose data is known to be a `T`.
#[derive(Deserialize)]
struct CombinedOf<T> {
    data: T,
}

/// Parses the text of one combined-stream frame. Its event type is read first and the data
/// then decoded straight into that event's type: a tagged enum would copy the whole object
/// into a buffer of its own before decoding it a second time (#68).
///
/// # Errors
/// Returns an error when the frame is not valid JSON of a known event type.
pub fn parse_frame(text: &str) -> Result<CombinedFrame, MessageError> {
    let peek: Peek = serde_json::from_str(text)?;
    let data = match peek.data.e.as_ref() {
        "depthUpdate" => Payload::Depth(serde_json::from_str::<CombinedOf<_>>(text)?.data),
        "aggTrade" => Payload::AggTrade(serde_json::from_str::<CombinedOf<_>>(text)?.data),
        other => {
            let error = <serde_json::Error as de::Error>::unknown_variant(other, EVENTS);
            return Err(MessageError::Json(error));
        }
    };
    Ok(CombinedFrame {
        stream: peek.stream.into_owned(),
        data,
    })
}

/// Reads one combined-stream frame as a venue frame.
///
/// # Errors
/// Returns an error when the frame is not valid JSON of a known event type.
pub(crate) fn parse(text: &str) -> Result<Parsed, MessageError> {
    Ok(match parse_frame(text)?.data {
        Payload::Depth(raw) => Parsed::Depth(RawDepth::Binance(raw)),
        Payload::AggTrade(raw) => Parsed::Trade(RawTrade::Binance(raw)),
    })
}

/// Parses the body of a REST depth snapshot.
///
/// # Errors
/// Returns an error when the body is not a valid snapshot.
pub fn parse_snapshot(body: &str) -> Result<RawSnapshot, MessageError> {
    Ok(serde_json::from_str(body)?)
}

/// Just the exchange event time `E`, wherever it sits.
#[derive(Deserialize)]
struct EventTime {
    #[serde(rename = "E")]
    event_time_ms: Option<u64>,
}

#[derive(Deserialize)]
struct Combined {
    data: Option<EventTime>,
}

/// The `E` of a combined-stream frame in nanoseconds, 0 when it has none.
pub(crate) fn frame_event_time_ns(text: &str) -> i64 {
    serde_json::from_str::<Combined>(text)
        .ok()
        .and_then(|frame| frame.data)
        .and_then(|data| data.event_time_ms)
        .map_or(0, ms_to_ns)
}

/// The `E` of a REST snapshot body in nanoseconds, 0 when it has none.
pub(crate) fn snapshot_event_time_ns(body: &str) -> i64 {
    serde_json::from_str::<EventTime>(body)
        .ok()
        .and_then(|body| body.event_time_ms)
        .map_or(0, ms_to_ns)
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

    /// Converts a raw aggregated trade into integer units; its id becomes the trade id.
    ///
    /// # Errors
    /// Returns an error when the price or quantity is not representable at this scale.
    pub fn agg_trade(&self, raw: &RawAggTrade) -> Result<Trade, DecimalError> {
        Ok(Trade {
            symbol: raw.symbol.clone(),
            trade_id: raw.agg_trade_id.to_string(),
            price: parse_units(&raw.price, self.price_decimals)?,
            qty: parse_units(&raw.qty, self.qty_decimals)?,
            trade_time_ms: raw.trade_time_ms,
            buyer_is_maker: raw.buyer_is_maker,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::sync::Level;

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

    /// Decimals are kept inline (#68): an escaped one still reads, and one too long for the
    /// buffer is a bad frame, never a panic or a truncated price.
    #[test]
    fn decimals_read_inline_escaped_or_not_and_overlong_ones_are_refused()
    -> Result<(), MessageError> {
        // The 0 of 50000.10 as the JSON escape backslash-u-0030.
        let backslash = char::from(0x5c_u8);
        let escaped = DEPTH.replace("\"50000.10\"", &format!("\"5000{backslash}u0030.10\""));
        assert_ne!(escaped, DEPTH);
        let Payload::Depth(raw) = parse_frame(&escaped)?.data else {
            panic!("expected a depth payload");
        };
        assert_eq!(raw.bids[0][0].as_str(), "50000.10");
        assert_eq!(format!("{:?}", raw.bids[0]), r#"["50000.10", "1.500"]"#);
        let overlong = format!("\"{}\"", "1".repeat(Decimal::MAX + 1));
        let overlong = DEPTH.replace("\"50000.10\"", &overlong);
        assert!(matches!(parse_frame(&overlong), Err(MessageError::Json(_))));
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
        assert_eq!(trade.trade_id, "42");
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
