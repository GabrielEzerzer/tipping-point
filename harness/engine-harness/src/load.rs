//! Synthetic market load for the API's load test: valid Binance frames at any rate, so both books
//! stay synced and every input takes the live path it would take live, depth frames as deep as
//! Binance's. Plus the instrument's small helpers: percentiles and the process's peak memory.

use std::collections::BTreeMap;
use std::time::Duration;

use engine::market::decimal::format_units;
use engine::market::live::Stream;
use engine::market::pipeline::Message;
use engine::market::ws::Inbound;
use engine::market::{Scales, Symbol};

/// Levels each side of a snapshot holds.
const LEVELS: i64 = 20;
const QTY_DECIMALS: u32 = 3;
/// Levels a side of a depth update, as the captured Binance frames carry them (#83:
/// `harness/out/step0-23/dl-live`, BTC and ETH depth@100ms, 40 and 122.5 levels a frame on
/// average, 20 to 60 a side, a p99 of 115 to 954 and at most 1,246): 20 to 60 a side, and one
/// update in a hundred a burst of 150 to 500 a side, as a fast market sends. The one level a side
/// the generator sent before left out most of a real frame's decoding cost.
const SIDE_LEVELS: (u64, u64) = (20, 60);
const BURST_LEVELS: (u64, u64) = (150, 500);
const BURST_EVERY: u64 = 100;

/// One symbol's side of the generator: the update chain and where its book sits.
#[derive(Debug)]
struct Book {
    symbol: String,
    stream: String,
    price_decimals: u32,
    /// Mid price in ticks.
    mid: i64,
    last_id: u64,
    bridged: bool,
    step: i64,
}

impl Book {
    fn new(symbol: &str, price_decimals: u32, mid: i64) -> Self {
        Self {
            symbol: symbol.to_owned(),
            stream: symbol.to_lowercase(),
            price_decimals,
            mid,
            last_id: 1000,
            bridged: false,
            step: 0,
        }
    }

    fn price(&self, ticks: i64) -> String {
        format_units(ticks, self.price_decimals)
    }

    fn snapshot(&self, ms: i64) -> String {
        let side = |sign: i64| -> Vec<String> {
            (1..=LEVELS)
                .map(|i| format!("[\"{}\",\"1.000\"]", self.price(self.mid + sign * i)))
                .collect()
        };
        format!(
            "{{\"lastUpdateId\":{},\"E\":{ms},\"T\":{ms},\"bids\":[{}],\"asks\":[{}]}}",
            self.last_id,
            side(-1).join(","),
            side(1).join(",")
        )
    }

    /// The next update of the chain, `bids` and `asks` levels a side: the first one bridges the
    /// snapshot (`U <= lastUpdateId <= u`), every later one follows the last (`pu` = the last
    /// `u`). The levels sit on their own ticks below and above the mid, so the book never
    /// crosses, and one in seven has quantity 0, a level removed, as a real stream sends.
    fn update(&mut self, ms: i64, bids: usize, asks: usize) -> String {
        let (first, last, prev) = if self.bridged {
            (self.last_id + 1, self.last_id + 1, self.last_id)
        } else {
            self.bridged = true;
            (self.last_id, self.last_id + 1, self.last_id - 1)
        };
        self.last_id = last;
        self.step += 1;
        let side = |sign: i64, count: usize| -> String {
            let levels: Vec<String> = (1..=i64::try_from(count).unwrap_or(0))
                .map(|k| {
                    let qty = format_units((self.step + k) % 7, QTY_DECIMALS);
                    format!("[\"{}\",\"{qty}\"]", self.price(self.mid + sign * k))
                })
                .collect();
            levels.join(",")
        };
        format!(
            "{{\"stream\":\"{}@depth@100ms\",\"data\":{{\"e\":\"depthUpdate\",\"E\":{ms},\"T\":{ms},\"s\":\"{}\",\"U\":{first},\"u\":{last},\"pu\":{prev},\"b\":[{}],\"a\":[{}]}}}}",
            self.stream,
            self.symbol,
            side(-1, bids),
            side(1, asks)
        )
    }
}

/// Makes the frames: an opening (both sockets connected, a snapshot per symbol), then depth
/// updates and aggregated trades in any mix.
#[derive(Debug)]
pub struct Generator {
    books: Vec<Book>,
    turn: usize,
    trade_id: u64,
    /// xorshift64 state, so every run sends the same frames.
    random: u64,
    /// Depth updates made, the levels they held (both sides), and the most in one.
    depth_levels: (u64, u64, u64),
}

impl Default for Generator {
    fn default() -> Self {
        Self {
            books: vec![
                Book::new("BTCUSDT", 1, 500_000),
                Book::new("ETHUSDT", 2, 300_000),
            ],
            turn: 0,
            trade_id: 0,
            random: 0x9E37_79B9_7F4A_7C15,
            depth_levels: (0, 0, 0),
        }
    }
}

impl Generator {
    /// The scales the frames are written at.
    #[must_use]
    pub fn scales(&self) -> BTreeMap<Symbol, Scales> {
        self.books
            .iter()
            .map(|book| {
                let scales = Scales {
                    price_decimals: book.price_decimals,
                    qty_decimals: QTY_DECIMALS,
                };
                (book.symbol.clone(), scales)
            })
            .collect()
    }

    /// Both sockets connected, then a snapshot per symbol.
    #[must_use]
    pub fn opening(&self, capture_ns: i64) -> Vec<Message> {
        let connected = |stream| Message::Socket {
            stream,
            inbound: Inbound::Connected {
                attempt: 1,
                capture_ns,
            },
        };
        let mut out = vec![connected(Stream::Depth), connected(Stream::Trades)];
        for book in &self.books {
            out.push(Message::Snapshot {
                symbol: book.symbol.clone(),
                capture_ns,
                result: Ok(book.snapshot(capture_ns / 1_000_000)),
            });
        }
        out
    }

    /// The next depth update, the symbols in turn, as deep as [`SIDE_LEVELS`] says.
    pub fn depth(&mut self, capture_ns: i64) -> Message {
        let (bids, asks) = (self.side_levels(), self.side_levels());
        let (frames, levels, most) = &mut self.depth_levels;
        *frames += 1;
        *levels += bids + asks;
        *most = (*most).max(bids + asks);
        let count = self.books.len();
        let book = &mut self.books[self.turn % count];
        self.turn += 1;
        let (bids, asks) = (
            usize::try_from(bids).unwrap_or(0),
            usize::try_from(asks).unwrap_or(0),
        );
        Message::Socket {
            stream: Stream::Depth,
            inbound: Inbound::Text {
                text: book.update(capture_ns / 1_000_000, bids, asks),
                dropped_before: 0,
                capture_ns,
            },
        }
    }

    /// Depth updates made so far, the levels they held on both sides, and the most one held.
    #[must_use]
    pub fn depth_levels(&self) -> (u64, u64, u64) {
        self.depth_levels
    }

    /// Levels a side of the next depth update: [`SIDE_LEVELS`], or once in [`BURST_EVERY`]
    /// [`BURST_LEVELS`].
    fn side_levels(&mut self) -> u64 {
        self.random ^= self.random << 13;
        self.random ^= self.random >> 7;
        self.random ^= self.random << 17;
        let (low, high) = if self.random.is_multiple_of(BURST_EVERY) {
            BURST_LEVELS
        } else {
            SIDE_LEVELS
        };
        low + self.random / BURST_EVERY % (high - low + 1)
    }

    /// The next aggregated trade, the symbols in turn.
    pub fn trade(&mut self, capture_ns: i64) -> Message {
        self.trade_id += 1;
        let count = self.books.len();
        let book = &self.books[usize::try_from(self.trade_id).unwrap_or(0) % count];
        let ms = capture_ns / 1_000_000;
        let id = self.trade_id;
        let text = format!(
            "{{\"stream\":\"{}@aggTrade\",\"data\":{{\"e\":\"aggTrade\",\"E\":{ms},\"a\":{id},\"s\":\"{}\",\"p\":\"{}\",\"q\":\"0.010\",\"f\":{id},\"l\":{id},\"T\":{ms},\"m\":{}}}}}",
            book.stream,
            book.symbol,
            book.price(book.mid),
            id.is_multiple_of(2)
        );
        Message::Socket {
            stream: Stream::Trades,
            inbound: Inbound::Text {
                text,
                dropped_before: 0,
                capture_ns,
            },
        }
    }
}

/// The `p`-th percentile (0 to 100) of durations, nearest rank; zero for none.
#[must_use]
pub fn percentile(durations: &mut [Duration], p: f64) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    durations.sort_unstable();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let rank = ((p / 100.0) * durations.len() as f64).ceil() as usize;
    durations[rank.clamp(1, durations.len()) - 1]
}

/// The process's peak memory in bytes, when the platform says: the peak working set on Windows
/// (asked of PowerShell), `VmHWM` on Linux.
#[must_use]
pub fn peak_memory_bytes() -> Option<u64> {
    if cfg!(windows) {
        let pid = std::process::id();
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!("(Get-Process -Id {pid}).PeakWorkingSet64"),
            ])
            .output()
            .ok()?;
        String::from_utf8_lossy(&output.stdout).trim().parse().ok()
    } else {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
        let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kib * 1024)
    }
}

#[cfg(test)]
mod tests {
    use engine::market::feed::Feed;
    use engine::market::pipeline::{Pipeline, entries};
    use engine::venue::Venue;

    use super::*;

    /// The frames keep both books synced however many follow, and every one converts.
    #[test]
    fn generated_frames_keep_both_books_synced() {
        let mut generator = Generator::default();
        let mut pipeline = Pipeline::new(Feed::new(Venue::BinanceUsdm, generator.scales()));
        let mut messages = generator.opening(1_000_000_000);
        for i in 0..200 {
            messages.push(generator.depth(1_000_000_000 + i));
            messages.push(generator.trade(1_000_000_000 + i));
        }
        for message in messages {
            for entry in entries(Venue::BinanceUsdm, message) {
                let applied = pipeline.apply(entry.kind, &entry.payload, &mut |_| {});
                assert_eq!(applied.error(), None, "{}", entry.payload);
            }
        }
        let feed = pipeline.feed();
        for symbol in ["BTCUSDT", "ETHUSDT"] {
            assert!(
                feed.sync(symbol)
                    .is_some_and(engine::market::sync::BookSync::is_synced)
            );
            let book = feed.book(symbol).expect("a synced book");
            assert!(!book.is_crossed(), "{symbol}");
            assert!(book.depth().0 > 50, "{symbol}: {:?}", book.depth());
        }
        assert_eq!(feed.stats().resyncs, 0);
        assert_eq!(feed.stats().trades, 200);
    }

    /// Depth updates are as deep as the captured Binance frames (#83): 20 to 60 levels a side,
    /// now and then hundreds, never the one level a side that hid most of a frame's cost.
    #[test]
    fn depth_updates_carry_as_many_levels_as_binances() {
        let mut generator = Generator::default();
        let mut deepest = 0;
        for i in 0..2_000 {
            let Message::Socket {
                inbound: Inbound::Text { text, .. },
                ..
            } = generator.depth(i)
            else {
                panic!("a depth update is a text frame");
            };
            let frame: serde_json::Value = serde_json::from_str(&text).expect("a JSON frame");
            for side in ["b", "a"] {
                let levels = frame["data"][side].as_array().map_or(0, Vec::len);
                assert!(levels >= 20, "{levels} levels a side");
                deepest = deepest.max(levels);
            }
        }
        let (frames, levels, most) = generator.depth_levels();
        assert_eq!(frames, 2_000);
        let mean = levels / frames;
        assert!(
            (60..=120).contains(&mean),
            "{mean} levels a frame on average"
        );
        assert!(
            most >= 300 && deepest >= 150,
            "no burst: at most {most} a frame"
        );
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let mut values: Vec<Duration> = (1..=100).map(Duration::from_millis).collect();
        assert_eq!(percentile(&mut values, 50.0), Duration::from_millis(50));
        assert_eq!(percentile(&mut values, 99.0), Duration::from_millis(99));
        assert_eq!(percentile(&mut [], 99.0), Duration::ZERO);
    }
}
