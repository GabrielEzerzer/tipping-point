//! A script of what a live loop received, one JSON line per message: the format of the
//! hand-written synthetic fixtures and of the depth recordings made before the binary log
//! existed. Replaying a script drives the same [`Pipeline`] the live loop runs, without the
//! network. The engine's own recordings are binary logs ([`crate::store`]).

use std::collections::BTreeMap;
use std::io::{self, BufRead};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::feed::Feed;
use super::live::Stream;
use super::pipeline::{Message, Pipeline, Replayed, Report, entries};
use super::ws::Inbound;
use super::{Scales, Symbol};
use crate::store::ms_to_ns;
use crate::venue::Venue;

/// One line of a script.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    /// Always first: the venue whose streams the script holds (Binance's, for scripts written
    /// before there were others) and the scales its frames must be parsed with.
    Scales {
        #[serde(default = "binance")]
        venue: Venue,
        scales: BTreeMap<Symbol, Scales>,
    },
    /// The raw text of one socket frame: the depth socket's unless `stream` says otherwise,
    /// with how many frames the reader dropped just before it.
    Frame {
        t_ms: u64,
        text: String,
        #[serde(default = "depth")]
        stream: Stream,
        #[serde(default)]
        dropped_before: u64,
    },
    /// The raw body of a REST snapshot, at the point in the stream it arrived.
    Snapshot {
        t_ms: u64,
        symbol: Symbol,
        body: String,
    },
    /// A snapshot fetch that failed.
    SnapshotFailed {
        t_ms: u64,
        symbol: Symbol,
        error: String,
    },
    /// A socket connected.
    Connected {
        t_ms: u64,
        stream: Stream,
        attempt: u32,
    },
    /// A socket's connection ended.
    Disconnected {
        t_ms: u64,
        stream: Stream,
        reason: String,
        #[serde(default)]
        dropped_before: u64,
    },
}

/// Scripts from before the trade socket was recorded hold depth frames only.
fn depth() -> Stream {
    Stream::Depth
}

/// Scripts from before there were venues are Binance's.
fn binance() -> Venue {
    Venue::BinanceUsdm
}

impl Record {
    /// When the loop received it, in milliseconds since the epoch; `None` for the scales.
    #[must_use]
    pub fn t_ms(&self) -> Option<u64> {
        match self {
            Self::Scales { .. } => None,
            Self::Frame { t_ms, .. }
            | Self::Snapshot { t_ms, .. }
            | Self::SnapshotFailed { t_ms, .. }
            | Self::Connected { t_ms, .. }
            | Self::Disconnected { t_ms, .. } => Some(*t_ms),
        }
    }

    /// The live-loop message this line stands for; `None` for the scales.
    #[must_use]
    pub fn into_message(self) -> Option<Message> {
        let message = match self {
            Self::Scales { .. } => return None,
            Self::Frame {
                t_ms,
                text,
                stream,
                dropped_before,
            } => Message::Socket {
                stream,
                inbound: Inbound::Text {
                    text,
                    dropped_before,
                    capture_ns: ms_to_ns(t_ms),
                },
            },
            Self::Snapshot { t_ms, symbol, body } => Message::Snapshot {
                symbol,
                capture_ns: ms_to_ns(t_ms),
                result: Ok(body),
            },
            Self::SnapshotFailed {
                t_ms,
                symbol,
                error,
            } => Message::Snapshot {
                symbol,
                capture_ns: ms_to_ns(t_ms),
                result: Err(error),
            },
            Self::Connected {
                t_ms,
                stream,
                attempt,
            } => Message::Socket {
                stream,
                inbound: Inbound::Connected {
                    attempt,
                    capture_ns: ms_to_ns(t_ms),
                },
            },
            Self::Disconnected {
                t_ms,
                stream,
                reason,
                dropped_before,
            } => Message::Socket {
                stream,
                inbound: Inbound::Disconnected {
                    reason,
                    dropped_before,
                    capture_ns: ms_to_ns(t_ms),
                },
            },
        };
        Some(message)
    }
}

/// Why a script could not be read or replayed.
#[derive(Debug, Error)]
pub enum RecordingError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("line {0}: {1}")]
    Malformed(usize, serde_json::Error),
    #[error("the first record must carry the scales")]
    MissingScales,
    #[error("record {0}: only the first record carries the scales")]
    MisplacedScales(u64),
}

/// Reads every record of a JSON-lines script.
///
/// # Errors
/// Returns an error for an unreadable source or a line that is not a record.
pub fn read_records(input: impl BufRead) -> Result<Vec<Record>, RecordingError> {
    let mut records = Vec::new();
    for (index, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record =
            serde_json::from_str(&line).map_err(|e| RecordingError::Malformed(index + 1, e))?;
        records.push(record);
    }
    Ok(records)
}

/// Drives a fresh pipeline through a script exactly as the live loop would have, and reports
/// what it made of every input, by record index (the scales are record 0). Like the live loop
/// it never stops at an input it cannot use.
///
/// # Errors
/// Returns an error when the scales are missing or appear again later.
pub fn replay(records: impl IntoIterator<Item = Record>) -> Result<Replayed, RecordingError> {
    let mut records = records.into_iter();
    let Some(Record::Scales { venue, scales }) = records.next() else {
        return Err(RecordingError::MissingScales);
    };
    let mut pipeline = Pipeline::new(Feed::new(venue, scales));
    let mut report = Report::default();
    for (index, record) in (1u64..).zip(records) {
        let message = record
            .into_message()
            .ok_or(RecordingError::MisplacedScales(index))?;
        for entry in entries(venue, message) {
            let applied = pipeline.apply(entry.kind, &entry.payload, &mut |_| {});
            report.note(index, &applied);
        }
    }
    Ok(Replayed { pipeline, report })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scales() -> Record {
        Record::Scales {
            venue: Venue::BinanceUsdm,
            scales: BTreeMap::from([(
                "BTCUSDT".to_owned(),
                Scales {
                    price_decimals: 1,
                    qty_decimals: 3,
                },
            )]),
        }
    }

    #[test]
    fn records_round_trip_through_json_lines() -> Result<(), Box<dyn std::error::Error>> {
        let records = vec![
            scales(),
            Record::Frame {
                t_ms: 5,
                text: "{}".to_owned(),
                stream: Stream::Depth,
                dropped_before: 0,
            },
            Record::Frame {
                t_ms: 5,
                text: "{}".to_owned(),
                stream: Stream::Trades,
                dropped_before: 3,
            },
            Record::Snapshot {
                t_ms: 6,
                symbol: "BTCUSDT".to_owned(),
                body: "{}".to_owned(),
            },
            Record::SnapshotFailed {
                t_ms: 7,
                symbol: "BTCUSDT".to_owned(),
                error: "timed out".to_owned(),
            },
            Record::Connected {
                t_ms: 8,
                stream: Stream::Trades,
                attempt: 2,
            },
            Record::Disconnected {
                t_ms: 9,
                stream: Stream::Depth,
                reason: "stream ended".to_owned(),
                dropped_before: 1,
            },
        ];
        let mut bytes = Vec::new();
        for record in &records {
            serde_json::to_writer(&mut bytes, record)?;
            bytes.push(b'\n');
        }
        assert_eq!(
            String::from_utf8_lossy(&bytes).lines().count(),
            records.len()
        );
        assert_eq!(read_records(bytes.as_slice())?, records);
        Ok(())
    }

    #[test]
    fn a_frame_line_without_stream_or_drops_is_a_depth_frame() -> Result<(), RecordingError> {
        let line = b"{\"kind\":\"frame\",\"t_ms\":1,\"text\":\"x\"}\n";
        assert_eq!(
            read_records(&line[..])?,
            vec![Record::Frame {
                t_ms: 1,
                text: "x".to_owned(),
                stream: Stream::Depth,
                dropped_before: 0
            }]
        );
        Ok(())
    }

    #[test]
    fn replay_requires_scales_first_and_only_first() {
        let frame = Record::Frame {
            t_ms: 1,
            text: "{}".to_owned(),
            stream: Stream::Depth,
            dropped_before: 0,
        };
        assert!(matches!(
            replay(vec![frame]),
            Err(RecordingError::MissingScales)
        ));
        assert!(matches!(
            replay(vec![scales(), scales()]),
            Err(RecordingError::MisplacedScales(1))
        ));
    }

    #[test]
    fn replay_reports_bad_inputs_and_keeps_going() -> Result<(), RecordingError> {
        let frame = |text: &str| Record::Frame {
            t_ms: 1,
            text: text.to_owned(),
            stream: Stream::Depth,
            dropped_before: 0,
        };
        let good = r#"{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1,"T":1,"s":"BTCUSDT","U":1,"u":2,"pu":0,"b":[],"a":[]}}"#;
        let replayed = replay(vec![scales(), frame("not json"), frame(good)])?;
        let report = replayed.report;
        assert_eq!((report.inputs, report.errors), (2, 1));
        assert_eq!(report.first_errors[0].0, 1);
        assert_eq!(
            replayed.pipeline.feed().stats().depth_events,
            1,
            "the good frame still ran"
        );
        Ok(())
    }

    #[test]
    fn malformed_lines_name_their_line_number() {
        let input = b"{\"kind\":\"frame\",\"t_ms\":1,\"text\":\"x\"}\n\nnot json\n";
        assert!(matches!(
            read_records(&input[..]),
            Err(RecordingError::Malformed(3, _))
        ));
    }
}
