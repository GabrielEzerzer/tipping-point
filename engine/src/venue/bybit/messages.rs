//! Wire shapes of Bybit's v5 public order-book stream (`orderbook.{depth}.{symbol}`), whose
//! messages are also, line for line, what its order-book archive holds, and their conversion to
//! integer units. Prices and quantities arrive as decimal strings and are parsed exactly; a
//! quantity of `"0"` deletes the level.

use serde::Deserialize;

use crate::market::Scales;
use crate::market::decimal::DecimalError;
use crate::market::sync::{DepthUpdate, Snapshot};
use crate::store::ms_to_ns;
use crate::venue::{Depth, MessageError, Parsed, RawDepth};

/// One order-book message:
/// `{"topic":"orderbook.200.BTCUSDT","type":"delta","ts":…,"data":{"s","b","a","u","seq"},"cts":…}`.
#[derive(Debug, Clone, Deserialize)]
pub struct RawBook {
    pub topic: String,
    #[serde(rename = "type")]
    pub kind: BookKind,
    /// When Bybit's system made the message, in milliseconds since the epoch.
    pub ts: u64,
    /// When the matching engine made the change, in milliseconds since the epoch.
    #[serde(default)]
    pub cts: Option<u64>,
    pub data: BookData,
}

/// Whether a message is a whole book or a change to one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BookKind {
    Snapshot,
    Delta,
}

/// A message's book data: `[price, qty]` string pairs and the ids.
#[derive(Debug, Clone, Deserialize)]
pub struct BookData {
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "b")]
    pub bids: Vec<[String; 2]>,
    #[serde(rename = "a")]
    pub asks: Vec<[String; 2]>,
    /// The update id: one more with every message, back to 1 when Bybit's service restarts.
    #[serde(rename = "u")]
    pub update_id: u64,
    /// The cross sequence, kept for reference; books chain on `u`.
    #[serde(default)]
    pub seq: Option<u64>,
}

impl RawBook {
    /// A whole book rather than a change to one: a snapshot, or the `u = 1` that Bybit sends,
    /// as a whole book, after its service restarts.
    #[must_use]
    pub fn is_whole(&self) -> bool {
        self.kind == BookKind::Snapshot || self.data.update_id == 1
    }

    /// The message in integer units at `scales`: a whole book, or a change that follows update
    /// `u - 1`.
    ///
    /// # Errors
    /// Returns an error when a price or quantity is not representable at this scale.
    pub fn convert(&self, scales: Scales) -> Result<Depth, DecimalError> {
        let bids = scales.levels(&self.data.bids)?;
        let asks = scales.levels(&self.data.asks)?;
        let u = self.data.update_id;
        Ok(if self.is_whole() {
            Depth::Book(Snapshot {
                last_update_id: u,
                bids,
                asks,
            })
        } else {
            Depth::Update(DepthUpdate {
                first_id: u,
                last_id: u,
                prev_last_id: u.saturating_sub(1),
                bids,
                asks,
            })
        })
    }
}

/// Reads one order-book message as a venue frame.
///
/// # Errors
/// Returns an error when the text is not an order-book message.
pub(crate) fn parse(text: &str) -> Result<Parsed, MessageError> {
    let raw: RawBook = serde_json::from_str(text)?;
    Ok(Parsed::Depth(RawDepth::Bybit(raw)))
}

/// Just the message time `ts`.
#[derive(Deserialize)]
struct MessageTime {
    ts: Option<u64>,
}

/// The `ts` of a message in nanoseconds, 0 when it has none.
pub(crate) fn frame_event_time_ns(text: &str) -> i64 {
    serde_json::from_str::<MessageTime>(text)
        .ok()
        .and_then(|message| message.ts)
        .map_or(0, ms_to_ns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::sync::Level;

    // Shaped like the archive's lines, which end in CRLF; the line reader strips the CR.
    const SNAPSHOT: &str = r#"{"topic":"orderbook.200.BTCUSDT","type":"snapshot","ts":1792454400495,"data":{"s":"BTCUSDT","b":[["81900.70","3.914"],["81900.60","0.551"]],"a":[["81900.80","1.206"]],"u":5788995,"seq":828958074456},"cts":1792454400393}"#;
    const DELTA: &str = r#"{"topic":"orderbook.200.BTCUSDT","type":"delta","ts":1792454400594,"data":{"s":"BTCUSDT","b":[["81900.60","0"]],"a":[["81907.60","0.013"]],"u":5788996,"seq":828958074787},"cts":1792454400494}"#;

    const BTC: Scales = Scales {
        price_decimals: 1,
        qty_decimals: 3,
    };

    fn read(text: &str) -> RawBook {
        match parse(text) {
            Ok(Parsed::Depth(RawDepth::Bybit(raw))) => raw,
            other => panic!("not a Bybit book message: {other:?}"),
        }
    }

    #[test]
    fn a_snapshot_is_a_whole_book_at_its_update_id() -> Result<(), DecimalError> {
        let raw = read(SNAPSHOT);
        assert_eq!(
            (raw.ts, raw.cts, raw.data.seq),
            (
                1_792_454_400_495,
                Some(1_792_454_400_393),
                Some(828_958_074_456)
            )
        );
        let Depth::Book(book) = raw.convert(BTC)? else {
            panic!("a snapshot is a whole book");
        };
        assert_eq!(book.last_update_id, 5_788_995);
        assert_eq!(
            book.bids,
            vec![
                Level {
                    price: 819_007,
                    qty: 3914
                },
                Level {
                    price: 819_006,
                    qty: 551
                }
            ]
        );
        assert_eq!(
            book.asks,
            vec![Level {
                price: 819_008,
                qty: 1206
            }]
        );
        assert_eq!(frame_event_time_ns(SNAPSHOT), 1_792_454_400_495_000_000);
        Ok(())
    }

    #[test]
    fn a_delta_follows_the_update_before_it_and_zero_deletes() -> Result<(), DecimalError> {
        let Depth::Update(update) = read(DELTA).convert(BTC)? else {
            panic!("a delta is a change");
        };
        assert_eq!(
            (update.first_id, update.last_id, update.prev_last_id),
            (5_788_996, 5_788_996, 5_788_995)
        );
        assert_eq!(
            update.bids,
            vec![Level {
                price: 819_006,
                qty: 0
            }]
        );
        Ok(())
    }

    #[test]
    fn update_id_one_is_a_whole_book_whatever_its_type() -> Result<(), DecimalError> {
        let restart = DELTA.replace(r#""u":5788996"#, r#""u":1"#);
        assert!(
            matches!(read(&restart).convert(BTC)?, Depth::Book(book) if book.last_update_id == 1)
        );
        Ok(())
    }

    #[test]
    fn other_messages_and_bad_precision_are_errors() {
        let ack = r#"{"success":true,"ret_msg":"subscribe","conn_id":"x","op":"subscribe"}"#;
        assert!(matches!(parse(ack), Err(MessageError::Json(_))));
        let trades = r#"{"topic":"publicTrade.BTCUSDT","type":"snapshot","ts":1,"data":[]}"#;
        assert!(parse(trades).is_err());
        let coarse = Scales {
            price_decimals: 0,
            qty_decimals: 3,
        };
        assert!(read(DELTA).convert(coarse).is_err());
    }
}
