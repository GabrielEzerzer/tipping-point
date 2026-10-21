//! The recording format: one JSON line per input the feed received, in the order it received
//! them. A recorder writes it from a live socket; a replay drives the same [`Feed`] from it.

use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::feed::{Feed, FeedError};
use super::{Scales, Symbol};

/// One line of a recording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    /// Always first: the scales the frames must be parsed with.
    Scales { scales: BTreeMap<Symbol, Scales> },
    /// The raw text of one stream frame.
    Frame { t_ms: u64, text: String },
    /// The raw body of a REST snapshot, at the point in the stream it arrived.
    Snapshot {
        t_ms: u64,
        symbol: Symbol,
        body: String,
    },
}

/// Why a recording could not be read or replayed.
#[derive(Debug, Error)]
pub enum RecordingError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("line {0}: {1}")]
    Malformed(usize, serde_json::Error),
    #[error("the first record must carry the scales")]
    MissingScales,
    #[error("record {0}: {1}")]
    Feed(usize, FeedError),
}

/// Milliseconds since the Unix epoch, for stamping records.
#[must_use]
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// Writes one record as a JSON line.
///
/// # Errors
/// Returns the underlying write error.
pub fn write_record(out: &mut impl Write, record: &Record) -> io::Result<()> {
    let line = serde_json::to_string(record).map_err(io::Error::other)?;
    out.write_all(line.as_bytes())?;
    out.write_all(b"\n")
}

/// Reads every record of a JSON-lines recording.
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

/// Drives a fresh [`Feed`] through the records and returns it, counters and all.
///
/// # Errors
/// Returns an error when the scales are missing or a record cannot be fed.
pub fn replay(records: impl IntoIterator<Item = Record>) -> Result<Feed, RecordingError> {
    let mut records = records.into_iter().enumerate();
    let Some((_, Record::Scales { scales })) = records.next() else {
        return Err(RecordingError::MissingScales);
    };
    let mut feed = Feed::new(scales);
    for (index, record) in records {
        let result = match &record {
            Record::Scales { .. } => Err(RecordingError::MissingScales),
            Record::Frame { text, .. } => feed
                .on_text(text)
                .map(|_| ())
                .map_err(|e| RecordingError::Feed(index + 1, e)),
            Record::Snapshot { symbol, body, .. } => feed
                .on_snapshot(symbol, body)
                .map(|_| ())
                .map_err(|e| RecordingError::Feed(index + 1, e)),
        };
        result?;
    }
    Ok(feed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip_through_json_lines() -> Result<(), RecordingError> {
        let records = vec![
            Record::Scales {
                scales: BTreeMap::from([(
                    "BTCUSDT".to_owned(),
                    Scales {
                        price_decimals: 1,
                        qty_decimals: 3,
                    },
                )]),
            },
            Record::Frame {
                t_ms: 5,
                text: "{}".to_owned(),
            },
            Record::Snapshot {
                t_ms: 6,
                symbol: "BTCUSDT".to_owned(),
                body: "{}".to_owned(),
            },
        ];
        let mut bytes = Vec::new();
        for record in &records {
            write_record(&mut bytes, record)?;
        }
        assert_eq!(String::from_utf8_lossy(&bytes).lines().count(), 3);
        assert_eq!(read_records(bytes.as_slice())?, records);
        Ok(())
    }

    #[test]
    fn replay_requires_scales_first() {
        let records = vec![Record::Frame {
            t_ms: 1,
            text: "{}".to_owned(),
        }];
        assert!(matches!(
            replay(records),
            Err(RecordingError::MissingScales)
        ));
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
