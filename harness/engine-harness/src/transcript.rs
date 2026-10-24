//! A transcript of what a pipeline did, input by input: what each input was and what the feed
//! made of it, the events it raised, and a digest of every book right after it. The live loop
//! and every replay of its log emit the same [`LiveEvent`]s, so their transcripts must be equal.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use engine::market::book::Book;
use engine::market::feed::Feed;
use engine::market::live::LiveEvent;
use engine::market::sync::BookSync;
use engine::store::frame::crc32c;
use serde::{Deserialize, Serialize};

/// What tells one state of a symbol's book from another cheaply: whether it is synced, its last
/// applied update id, and its depth. The engine changes a book only by applying an update
/// (which moves the last applied id) or by installing or dropping one (which moves the sync
/// state), so a symbol whose key is what it was at the last step still has the same book.
type BookKey = (bool, Option<u64>, Option<(usize, usize)>);

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

/// Bumped whenever what a step records changes, so that a transcript saved by an older harness
/// is recognised as such rather than read as a replay that differs. 2: trades carry their id as
/// a string (venues). 3: a depth update that does not convert raises a resync only when its
/// symbol had a book, and its error says whether it did (#41); a Binance frame's event type is
/// read before the rest of it, so an unreadable frame's error can read differently (#68).
pub const FORMAT: u32 = 3;

/// Every applied input of a run, in order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    /// The [`FORMAT`] it was made in; 0 for one saved before formats were numbered.
    #[serde(default)]
    pub format: u32,
    pub steps: Vec<Step>,
    #[serde(skip)]
    pending: Vec<String>,
    /// Each symbol's key and [`book_digest`] at the last step: a real book holds thousands of
    /// levels, so only the books an input changed are hashed again.
    #[serde(skip)]
    books: BTreeMap<String, (BookKey, u32)>,
}

impl Default for Transcript {
    fn default() -> Self {
        Self {
            format: FORMAT,
            steps: Vec::new(),
            pending: Vec::new(),
            books: BTreeMap::new(),
        }
    }
}

impl Transcript {
    /// Takes in one event. Events raised while applying an input come before its
    /// [`LiveEvent::Applied`], which closes the step. Reports, failed connection attempts and
    /// changed scales found on the venue are the loop's own business, not effects of an input,
    /// and are left out.
    pub fn observe(&mut self, event: &LiveEvent<'_>) {
        match event {
            LiveEvent::Applied {
                kind,
                applied,
                feed,
                ..
            } => {
                let events = std::mem::take(&mut self.pending);
                let books = self.books_now(feed);
                self.steps.push(Step {
                    kind: format!("{kind:?}"),
                    applied: format!("{applied:?}"),
                    events,
                    books,
                });
            }
            LiveEvent::Report { .. }
            | LiveEvent::ConnectFailed { .. }
            | LiveEvent::ScalesChanged { .. }
            | LiveEvent::ScalesCheckFailed { .. } => {}
            other => self.pending.push(format!("{other:?}")),
        }
    }

    /// [`books_digest`] of the feed, hashing again only the books whose key moved.
    fn books_now(&mut self, feed: &Feed) -> u32 {
        let mut digests = Vec::new();
        for symbol in feed.symbols() {
            let sync = feed.sync(symbol);
            let key = (
                sync.is_some_and(BookSync::is_synced),
                sync.and_then(BookSync::last_id),
                feed.book(symbol).map(Book::depth),
            );
            let digest = match self.books.get(symbol) {
                Some((cached, digest)) if *cached == key => *digest,
                _ => {
                    let digest = book_digest(feed, symbol);
                    self.books.insert(symbol.clone(), (key, digest));
                    digest
                }
            };
            digests.extend_from_slice(&digest.to_le_bytes());
        }
        crc32c(&digests)
    }

    /// Where two transcripts first part, readable enough to debug from; `None` when equal.
    #[must_use]
    pub fn first_difference(&self, other: &Self) -> Option<String> {
        if self.format != other.format {
            return Some(format!(
                "transcript formats {} and {} cannot be compared",
                self.format, other.format
            ));
        }
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

/// CRC-32C over every symbol's [`book_digest`], in symbol order.
#[must_use]
pub fn books_digest(feed: &Feed) -> u32 {
    let digests: Vec<u8> = feed
        .symbols()
        .flat_map(|symbol| book_digest(feed, symbol).to_le_bytes())
        .collect();
    crc32c(&digests)
}

/// CRC-32C over one symbol's state: its name, whether it is synced, its last applied update id,
/// then, while it has a book, each level's price and quantity as little-endian i64 (bids high
/// to low, then asks low to high).
#[must_use]
pub fn book_digest(feed: &Feed, symbol: &str) -> u32 {
    let mut bytes = symbol.as_bytes().to_vec();
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
    crc32c(&bytes)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use engine::market::Scales;
    use engine::market::sync::ResyncReason;
    use engine::venue::Venue;

    use super::*;

    fn feed() -> Feed {
        let scales = Scales {
            price_decimals: 1,
            qty_decimals: 3,
        };
        Feed::new(
            Venue::BinanceUsdm,
            BTreeMap::from([("BTCUSDT".to_owned(), scales)]),
        )
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

    /// The cached digest a transcript keeps is what hashing every book afresh gives, at every
    /// step, through each way a book changes: buffered, installed, bridged, applied, a gap, a
    /// reconnect and a reinstall.
    #[test]
    fn the_cached_digest_matches_hashing_every_book_afresh() {
        use engine::market::live::Stream;
        use engine::market::pipeline::{Message, Pipeline, entries};
        use engine::market::ws::Inbound;

        let depth = |first: u64, last: u64, prev: u64, bid: &str| Message::Socket {
            stream: Stream::Depth,
            inbound: Inbound::Text {
                text: format!(
                    r#"{{"stream":"btcusdt@depth@100ms","data":{{"e":"depthUpdate","E":1,"T":1,"s":"BTCUSDT","U":{first},"u":{last},"pu":{prev},"b":[["{bid}","1.000"]],"a":[]}}}}"#
                ),
                dropped_before: 0,
                capture_ns: 1,
            },
        };
        let snapshot = |id: u64| Message::Snapshot {
            symbol: "BTCUSDT".to_owned(),
            capture_ns: 2,
            result: Ok(format!(
                r#"{{"lastUpdateId":{id},"bids":[["99.0","1.000"]],"asks":[["101.0","1.000"]]}}"#
            )),
        };
        let disconnect = Message::Socket {
            stream: Stream::Depth,
            inbound: Inbound::Disconnected {
                reason: "gone".to_owned(),
                dropped_before: 0,
                capture_ns: 3,
            },
        };
        let script = [
            depth(10, 12, 9, "98.0"),
            snapshot(11),
            depth(13, 13, 12, "97.0"),
            depth(14, 15, 13, "96.0"),
            depth(20, 21, 19, "95.0"), // a gap
            snapshot(20),
            disconnect,
            depth(30, 30, 29, "94.0"),
            snapshot(30),
            depth(31, 31, 30, "93.0"),
        ];
        let mut pipeline = Pipeline::new(feed());
        let mut transcript = Transcript::default();
        let mut fresh = Vec::new();
        for message in script {
            for entry in entries(Venue::BinanceUsdm, message) {
                pipeline.apply(entry.kind, &entry.payload, &mut |event| {
                    if let LiveEvent::Applied { feed, .. } = &event {
                        fresh.push(books_digest(feed));
                    }
                    transcript.observe(&event);
                });
            }
        }
        let cached: Vec<u32> = transcript.steps.iter().map(|step| step.books).collect();
        assert_eq!(cached, fresh);
        assert!(transcript.raised("SequenceGap") && transcript.raised("Reconnect"));
    }

    /// Changed scales the live loop found on the venue, or a check of them that failed, are no
    /// effect of an input: a replay never raises them, so a live transcript must not hold them
    /// (#45, #100).
    #[test]
    fn scales_found_changed_are_left_out() {
        let mut transcript = Transcript::default();
        let was = Scales {
            price_decimals: 1,
            qty_decimals: 3,
        };
        let now = Scales {
            price_decimals: 2,
            ..was
        };
        transcript.observe(&LiveEvent::ScalesChanged {
            symbol: "BTCUSDT",
            was,
            now,
        });
        transcript.observe(&LiveEvent::ScalesCheckFailed {
            symbol: "ETHUSDT",
            error: "/fapi/v1/exchangeInfo answered 429: Too many requests",
        });
        let feed = feed();
        let applied = engine::market::pipeline::Applied::Control;
        transcript.observe(&LiveEvent::Applied {
            kind: engine::store::frame::Kind::Control,
            applied: &applied,
            feed: &feed,
            stats: &engine::market::live::LiveStats::default(),
        });
        assert_eq!(transcript.steps[0].events, Vec::<String>::new());
    }

    #[test]
    fn events_attach_to_the_input_that_raised_them() {
        let mut transcript = Transcript::default();
        transcript.observe(&LiveEvent::Resync {
            symbol: "BTCUSDT",
            reason: ResyncReason::Reconnect,
        });
        let feed = feed();
        let applied = engine::market::pipeline::Applied::Control;
        transcript.observe(&LiveEvent::Applied {
            kind: engine::store::frame::Kind::Control,
            applied: &applied,
            feed: &feed,
            stats: &engine::market::live::LiveStats::default(),
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
