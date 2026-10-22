//! The WebSocket protocol, version [`PROTOCOL`]: JSON text frames tagged by `t`, plus one binary
//! frame, the Ising lattice. Golden files under `engine/tests/fixtures/api/` pin every message;
//! a change here fails the contract tests until they are regenerated in the same commit.
//!
//! Units: prices are integer ticks and quantities integer lots at each symbol's scales, sent
//! once in `welcome`; times are integer milliseconds since the epoch (`ts_ms`), because
//! nanoseconds since the epoch are past 2^53 and JavaScript would round them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::binance::Scales;
use crate::binance::sync::Level;
use crate::store::samples::{BookRow, Sample};

/// The protocol version a client must say hello with.
pub const PROTOCOL: u32 = 1;

/// Close codes the engine ends a connection with.
pub mod close {
    /// A message that does not parse, or one not allowed at that point.
    pub const BAD_REQUEST: u16 = 4000;
    /// A hello with another protocol version.
    pub const VERSION_MISMATCH: u16 = 4001;
    /// The client stopped reading: its socket took nothing for the stall timeout, or it fell
    /// behind on `jev`, which is never dropped.
    pub const TOO_SLOW: u16 = 4002;
}

/// What a client can subscribe to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Topic {
    /// Top-N book per symbol at each 100 ms grid instant; the latest wins.
    Book,
    /// Aggregated trades, batched per flush; every one, or a `gap` saying how many were lost.
    Trades,
    /// Sockets, sync and counters; the latest wins, sent on change and every second.
    Status,
    /// Warning meters from the model; the latest per symbol wins.
    Meters,
    /// Jev labels; every one, never dropped.
    Jev,
    /// Paper P&L; the latest per symbol wins.
    Pnl,
    /// Ising lattices, binary; the latest wins.
    Lattice,
    /// Ising measurements per step; every one, or a `gap`.
    Ising,
}

impl Topic {
    /// Every topic, in the order `welcome` lists them.
    pub const ALL: [Self; 8] = [
        Self::Book,
        Self::Trades,
        Self::Status,
        Self::Meters,
        Self::Jev,
        Self::Pnl,
        Self::Lattice,
        Self::Ising,
    ];

    /// The topics a result can be published to over HTTP: the engine relays them to the app.
    #[must_use]
    pub fn relayed(name: &str) -> Option<Self> {
        match name {
            "meters" => Some(Self::Meters),
            "jev" => Some(Self::Jev),
            "pnl" => Some(Self::Pnl),
            "ising" => Some(Self::Ising),
            _ => None,
        }
    }

    /// The name on the wire.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Book => "book",
            Self::Trades => "trades",
            Self::Status => "status",
            Self::Meters => "meters",
            Self::Jev => "jev",
            Self::Pnl => "pnl",
            Self::Lattice => "lattice",
            Self::Ising => "ising",
        }
    }
}

/// What a client sends.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum ClientMessage {
    /// First, always: the protocol version the client speaks.
    Hello { v: u32 },
    /// Replaces the subscription. `symbols` defaults to all, `levels` to the sampled depth, and
    /// `backfill_s` first sends that many seconds of book samples and trades the engine still
    /// holds.
    Sub {
        topics: Vec<Topic>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        symbols: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        levels: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        backfill_s: Option<f64>,
    },
    /// Switches this connection to a recorded market run from `from_ms` (its start when absent),
    /// at `speed` times real time (0 for as fast as the client takes it).
    Replay {
        run: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        from_ms: Option<i64>,
        #[serde(default = "one")]
        speed: f64,
    },
    /// Jumps the replay to `to_ms`.
    Seek { to_ms: i64 },
    /// Holds the replay where it is.
    Pause,
    /// Carries the replay on.
    Play,
    /// Back to the live feed.
    Live,
}

fn one() -> f64 {
    1.0
}

/// A symbol and its scales, as `welcome` lists them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SymbolScales {
    pub symbol: String,
    pub price_decimals: u32,
    pub qty_decimals: u32,
}

/// The engine's answer to `hello`: everything the app needs to read what follows.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Welcome {
    pub v: u32,
    /// `live`: the connection starts on the live feed.
    pub mode: &'static str,
    /// The run the live feed is recording, if any.
    pub run: Option<String>,
    /// Levels each side of a `book` message holds at most.
    pub levels: usize,
    pub sample_every_ms: i64,
    pub symbols: Vec<SymbolScales>,
    pub topics: Vec<Topic>,
}

impl Welcome {
    #[must_use]
    pub fn new(
        scales: &BTreeMap<String, Scales>,
        run: Option<String>,
        levels: usize,
        sample_every_ns: i64,
    ) -> Self {
        Self {
            v: PROTOCOL,
            mode: "live",
            run,
            levels,
            sample_every_ms: sample_every_ns / 1_000_000,
            symbols: scales
                .iter()
                .map(|(symbol, scales)| SymbolScales {
                    symbol: symbol.clone(),
                    price_decimals: scales.price_decimals,
                    qty_decimals: scales.qty_decimals,
                })
                .collect(),
            topics: Topic::ALL.to_vec(),
        }
    }
}

/// One symbol's book at one grid instant: a `book_samples` row, as JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BookMessage<'a> {
    pub symbol: &'a str,
    pub ts_ms: i64,
    /// The last log frame applied before the instant.
    pub seq: i64,
    pub synced: bool,
    pub last_update_id: Option<i64>,
    /// `[price, qty]` pairs, best first, at most the subscription's levels.
    pub bids: Vec<[i64; 2]>,
    pub asks: Vec<[i64; 2]>,
}

impl<'a> BookMessage<'a> {
    #[must_use]
    pub fn new(sample: &Sample, row: &'a BookRow, levels: usize) -> Self {
        let pairs = |side: &[Level]| -> Vec<[i64; 2]> {
            side.iter()
                .take(levels)
                .map(|level| [level.price, level.qty])
                .collect()
        };
        Self {
            symbol: &row.symbol,
            ts_ms: sample.sample_ts_ns.div_euclid(1_000_000),
            seq: sample.seq,
            synced: row.synced,
            last_update_id: row.last_update_id,
            bids: pairs(&row.bids),
            asks: pairs(&row.asks),
        }
    }
}

/// One aggregated trade.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TradeMessage {
    pub symbol: String,
    /// The log frame that carried it.
    pub seq: u64,
    /// The exchange's trade time.
    pub ts_ms: u64,
    pub id: u64,
    pub px: i64,
    pub qty: i64,
    pub buyer_is_maker: bool,
}

/// One socket's state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct StreamStatus {
    pub connected: bool,
    pub reconnects: u64,
    pub frames_dropped: u64,
}

/// Both sockets.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Streams {
    pub depth: StreamStatus,
    pub trades: StreamStatus,
}

/// One symbol's state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SymbolStatus {
    pub synced: bool,
}

/// What the ingest is doing. `ts_ms` is the capture time of the last input it reflects, so a
/// client tells stale data from its own clock.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Status {
    pub ts_ms: i64,
    pub streams: Streams,
    pub symbols: BTreeMap<String, SymbolStatus>,
    pub resyncs: u64,
    pub sequence_gaps: u64,
    pub snapshot_failures: u64,
    pub parse_errors: u64,
}

impl Status {
    /// Equal but for the time: a change worth sending.
    #[must_use]
    pub fn same_as(&self, other: &Self) -> bool {
        Self {
            ts_ms: other.ts_ms,
            ..self.clone()
        } == *other
    }
}

/// Where a connection's book and trades come from, sent whenever that changes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceMessage {
    /// `live`, `replay`, `paused`, or `ended` (a replay ran out, or the live feed stopped).
    pub mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}

/// Every text frame the engine sends.
#[derive(Debug, Serialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum ServerMessage<'a> {
    Welcome(&'a Welcome),
    Book(&'a BookMessage<'a>),
    Trades {
        trades: &'a [TradeMessage],
    },
    Status(&'a Status),
    /// `lost` messages of an append topic were dropped for this client.
    Gap {
        topic: Topic,
        lost: u64,
    },
    Source(&'a SourceMessage),
    /// A result relayed from the Python side, as it was published.
    Meters {
        data: &'a Value,
    },
    Jev {
        data: &'a Value,
    },
    Pnl {
        data: &'a Value,
    },
    Ising {
        data: &'a Value,
    },
}

impl ServerMessage<'_> {
    /// The frame's text.
    ///
    /// # Panics
    /// Never: every message is plain data and always serializes.
    #[must_use]
    pub fn to_text(&self) -> String {
        serde_json::to_string(self).expect("engine messages always serialize")
    }

    /// A relayed result on its topic.
    #[must_use]
    pub fn relay(topic: Topic, data: &Value) -> ServerMessage<'_> {
        match topic {
            Topic::Jev => ServerMessage::Jev { data },
            Topic::Pnl => ServerMessage::Pnl { data },
            Topic::Ising => ServerMessage::Ising { data },
            _ => ServerMessage::Meters { data },
        }
    }
}

/// The header of a lattice frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatticeHeader {
    pub t: String,
    pub run: String,
    pub replica: u32,
    pub step: i64,
    #[serde(rename = "L")]
    pub l: u32,
    #[serde(rename = "T")]
    pub temperature: f64,
}

/// The one binary frame: `u32` little-endian header length, the JSON header, then the spins as
/// the `configs` table stores them, `ceil(L^2 / 8)` bytes, one bit per site, 1 = +1, row-major,
/// most significant bit first.
///
/// # Panics
/// Never: the header is plain data of a few dozen bytes.
#[must_use]
pub fn lattice_frame(header: &LatticeHeader, spins: &[u8]) -> Vec<u8> {
    let json = serde_json::to_vec(header).expect("a lattice header always serializes");
    let len = u32::try_from(json.len()).expect("a lattice header is a few dozen bytes");
    let mut frame = Vec::with_capacity(4 + json.len() + spins.len());
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(&json);
    frame.extend_from_slice(spins);
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_messages_are_tagged_by_t() {
        let parse = |text: &str| serde_json::from_str::<ClientMessage>(text).ok();
        assert_eq!(
            parse(r#"{"t":"hello","v":1}"#),
            Some(ClientMessage::Hello { v: 1 })
        );
        assert_eq!(
            parse(r#"{"t":"replay","run":"r"}"#),
            Some(ClientMessage::Replay {
                run: "r".to_owned(),
                from_ms: None,
                speed: 1.0
            })
        );
        assert_eq!(parse(r#"{"t":"sub","topics":["nope"]}"#), None);
        assert_eq!(parse(r#"{"t":"bye"}"#), None);
    }

    #[test]
    fn a_status_change_ignores_the_time() {
        let a = Status::default();
        let b = Status {
            ts_ms: 5,
            ..Status::default()
        };
        assert!(a.same_as(&b));
        let c = Status {
            resyncs: 1,
            ..Status::default()
        };
        assert!(!a.same_as(&c));
    }

    #[test]
    fn a_lattice_frame_is_length_header_spins() {
        let header = LatticeHeader {
            t: "lattice".to_owned(),
            run: "r".to_owned(),
            replica: 0,
            step: 1,
            l: 4,
            temperature: 2.5,
        };
        let frame = lattice_frame(&header, &[0x80, 0x01]);
        let len = u32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
        let parsed: LatticeHeader =
            serde_json::from_slice(&frame[4..4 + len]).expect("the header parses");
        assert_eq!(parsed, header);
        assert_eq!(&frame[4 + len..], &[0x80, 0x01]);
    }
}
