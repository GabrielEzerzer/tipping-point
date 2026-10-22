//! A transcript of what a pipeline did, input by input: what each input was and what the feed
//! made of it, the events it raised, and a digest of every book right after it. The live loop
//! and every replay of its log emit the same [`LiveEvent`]s, so their transcripts must be equal.

use std::fs;
use std::io;
use std::path::Path;

use engine::binance::feed::Feed;
use engine::binance::live::LiveEvent;
use engine::binance::sync::BookSync;
use engine::store::frame::crc32c;
use serde::{Deserialize, Serialize};

/// One applied input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    /// The frame kind.
    pub kind: String,
    /// What the feed made of it.
    pub applied: String,
    /// The events it raised, in order.
    pub events: Vec<String>,
    /// [`books_digest`] right after it.
    pub books: u32,
}

/// Every applied input of a run, in order.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transcript {
    pub steps: Vec<Step>,
    #[serde(skip)]
    pending: Vec<String>,
}

impl Transcript {
    /// Takes in one event. Events raised while applying an input come before its
    /// [`LiveEvent::Applied`], which closes the step. Reports and failed connection attempts are
    /// the loop's own business, not effects of an input, and are left out.
    pub fn observe(&mut self, event: &LiveEvent<'_>) {
        match event {
            LiveEvent::Applied {
                kind,
                applied,
                feed,
            } => {
                let events = std::mem::take(&mut self.pending);
                self.steps.push(Step {
                    kind: format!("{kind:?}"),
                    applied: format!("{applied:?}"),
                    events,
                    books: books_digest(feed),
                });
            }
            LiveEvent::Report { .. } | LiveEvent::ConnectFailed { .. } => {}
            other => self.pending.push(format!("{other:?}")),
        }
    }

    /// Where two transcripts first part, readable enough to debug from; `None` when equal.
    #[must_use]
    pub fn first_difference(&self, other: &Self) -> Option<String> {
        for (index, (ours, theirs)) in self.steps.iter().zip(&other.steps).enumerate() {
            if ours != theirs {
                return Some(format!("step {index}:\n  {ours:?}\n  {theirs:?}"));
            }
        }
        (self.steps.len() != other.steps.len())
            .then(|| format!("{} steps against {}", self.steps.len(), other.steps.len()))
    }

    /// True when any step raised an event whose description contains `needle`.
    #[must_use]
    pub fn raised(&self, needle: &str) -> bool {
        self.steps
            .iter()
            .any(|step| step.events.iter().any(|event| event.contains(needle)))
    }

    /// Writes the transcript as JSON.
    ///
    /// # Errors
    /// Returns the I/O error.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let text = serde_json::to_vec(self).map_err(io::Error::other)?;
        fs::write(path, text)
    }

    /// Reads a transcript written by [`Self::save`].
    ///
    /// # Errors
    /// Returns the I/O error, or `InvalidData` when the file is not a transcript.
    pub fn load(path: &Path) -> io::Result<Self> {
        let text = fs::read(path)?;
        serde_json::from_slice(&text)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
    }
}

/// CRC-32C over every symbol's state: its name, whether it is synced, its last applied update
/// id, then, while it has a book, each level's price and quantity as little-endian i64 (bids
/// high to low, then asks low to high).
#[must_use]
pub fn books_digest(feed: &Feed) -> u32 {
    let mut bytes = Vec::new();
    for symbol in feed.symbols() {
        bytes.extend_from_slice(symbol.as_bytes());
        let sync = feed.sync(symbol);
        bytes.push(u8::from(sync.is_some_and(BookSync::is_synced)));
        let last_id = sync.and_then(BookSync::last_id).unwrap_or(u64::MAX);
        bytes.extend_from_slice(&last_id.to_le_bytes());
        match feed.book(symbol) {
            Some(book) => {
                let (bids, asks) = book.depth();
                bytes.extend_from_slice(&(bids as u64).to_le_bytes());
                bytes.extend_from_slice(&(asks as u64).to_le_bytes());
                for level in book.bids().chain(book.asks()) {
                    bytes.extend_from_slice(&level.price.to_le_bytes());
                    bytes.extend_from_slice(&level.qty.to_le_bytes());
                }
            }
            None => bytes.extend_from_slice(&u64::MAX.to_le_bytes()),
        }
    }
    crc32c(&bytes)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use engine::binance::Scales;
    use engine::binance::sync::ResyncReason;

    use super::*;

    fn feed() -> Feed {
        let scales = Scales {
            price_decimals: 1,
            qty_decimals: 3,
        };
        Feed::new(BTreeMap::from([("BTCUSDT".to_owned(), scales)]))
    }

    #[test]
    fn a_book_change_changes_the_digest() {
        let mut feed = feed();
        let empty = books_digest(&feed);
        let depth = r#"{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1,"T":1,"s":"BTCUSDT","U":10,"u":12,"pu":9,"b":[],"a":[]}}"#;
        feed.on_text(depth).expect("a depth frame");
        assert_eq!(books_digest(&feed), empty, "buffering changes no book");
        let snapshot =
            r#"{"lastUpdateId":11,"bids":[["99.0","1.000"]],"asks":[["101.0","1.000"]]}"#;
        feed.on_snapshot("BTCUSDT", snapshot).expect("a snapshot");
        assert_ne!(books_digest(&feed), empty);
    }

    #[test]
    fn events_attach_to_the_input_that_raised_them() {
        let mut transcript = Transcript::default();
        transcript.observe(&LiveEvent::Resync {
            symbol: "BTCUSDT",
            reason: ResyncReason::Reconnect,
        });
        let feed = feed();
        let applied = engine::binance::pipeline::Applied::Control;
        transcript.observe(&LiveEvent::Applied {
            kind: engine::store::frame::Kind::Control,
            applied: &applied,
            feed: &feed,
        });
        assert_eq!(transcript.steps.len(), 1);
        assert!(transcript.raised("Reconnect"));
        assert_eq!(transcript.first_difference(&transcript.clone()), None);
        assert!(
            transcript
                .first_difference(&Transcript::default())
                .is_some()
        );
    }
}
