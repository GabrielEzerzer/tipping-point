//! Book samples: the book of each of a venue's symbols on a fixed grid of the capture clock, the
//! rows of the `book_samples` table. Rotate makes them from the log and the live API makes them
//! in the live loop with this same code, so the live stream holds exactly the rows rotate writes
//! later. The grid is absolute (multiples of the spacing since the epoch), so two venues' samples
//! join exactly on `sample_ts_ns`.
//!
//! A sample at instant `t` is the book after every input captured at or before `t`, the capture
//! clock taken as a running maximum so that a clock step back cannot reorder anything. Its
//! `seq` is the last frame before the input that moved the clock past `t`. Levels are null past
//! the book's depth and while the symbol is not synced.

use std::sync::Arc;

use arrow::array::{ArrayBuilder, ArrayRef, BooleanBuilder, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;

use super::StoreError;
use super::run::BookSample;
use crate::market::feed::Feed;
use crate::market::sync::{BookSync, Level};
use crate::venue::Venue;

/// A capture-clock jump longer than this many grid instants is skipped rather than filled with
/// samples of an unchanged book (a day at the default 100 ms).
pub const MAX_GAP_SAMPLES: i64 = 864_000;

/// The first instant of the grid `every_ns` apart (multiples of it since the epoch) at or after
/// `t`. The one place the grid is rounded to, for the sampler and for the replay that counts the
/// instants it lost (#87).
pub(crate) fn grid_ceil(t: i64, every_ns: i64) -> i64 {
    t.div_euclid(every_ns)
        .saturating_add(i64::from(t.rem_euclid(every_ns) != 0))
        .saturating_mul(every_ns)
}

/// One symbol's book as a sample holds it: the top levels of each side, empty while unsynced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookRow {
    pub symbol: String,
    pub synced: bool,
    /// The last update id applied, while synced.
    pub last_update_id: Option<i64>,
    /// Best first: bids high to low, asks low to high.
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
}

impl BookRow {
    /// The symbol's book in `feed` as it stands, at most `levels` a side.
    ///
    /// # Errors
    /// Returns [`StoreError::TooBig`] for an update id past the Int64 column's range.
    pub fn from_feed(feed: &Feed, symbol: &str, levels: usize) -> Result<Self, StoreError> {
        let sync = feed.sync(symbol);
        let synced = sync.is_some_and(BookSync::is_synced);
        let last_update_id = match sync.and_then(BookSync::last_id) {
            Some(id) if synced => Some(i64::try_from(id).map_err(|_| StoreError::TooBig(id))?),
            _ => None,
        };
        let book = feed.book(symbol).filter(|_| synced);
        Ok(Self {
            symbol: symbol.to_owned(),
            synced,
            last_update_id,
            bids: book
                .map(|b| b.bids().take(levels).collect())
                .unwrap_or_default(),
            asks: book
                .map(|b| b.asks().take(levels).collect())
                .unwrap_or_default(),
        })
    }
}

/// Every one of a venue's symbols' [`BookRow`] at one grid instant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    pub sample_ts_ns: i64,
    /// The last frame applied before the instant, in the venue's run.
    pub seq: i64,
    pub venue: Venue,
    /// One per symbol, in symbol order.
    pub rows: Vec<BookRow>,
}

impl Sample {
    /// Every symbol of `feed` as it stands, at `levels` a side.
    ///
    /// # Errors
    /// As [`BookRow::from_feed`].
    pub fn of_feed(
        feed: &Feed,
        sample_ts_ns: i64,
        seq: i64,
        levels: usize,
    ) -> Result<Self, StoreError> {
        let rows = feed
            .symbols()
            .map(|symbol| BookRow::from_feed(feed, symbol, levels))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            sample_ts_ns,
            seq,
            venue: feed.venue(),
            rows,
        })
    }
}

/// Walks the capture clock and says which grid instants an input moves it past.
#[derive(Debug, Clone)]
pub struct Sampler {
    levels: usize,
    every_ns: i64,
    /// The running maximum of input capture times.
    clock: Option<i64>,
    /// The next grid instant not yet sampled.
    next: i64,
    skipped: u64,
}

impl Sampler {
    #[must_use]
    pub fn new(sample: BookSample) -> Self {
        Self {
            levels: sample.levels,
            every_ns: sample.every_ns.max(1),
            clock: None,
            next: 0,
            skipped: 0,
        }
    }

    /// Levels kept per side.
    #[must_use]
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Grid instants left unsampled because the clock jumped further than [`MAX_GAP_SAMPLES`].
    #[must_use]
    pub fn skipped(&self) -> u64 {
        self.skipped
    }

    /// Called for every input, before it is applied, with its capture time: the grid instants
    /// the clock now passes, each to be sampled from the book as it stands, after every earlier
    /// input and before this one. None when the clock does not move past an instant, and none
    /// for a jump of more than [`MAX_GAP_SAMPLES`], which only counts them as skipped.
    pub fn instants(&mut self, captured: i64) -> Instants {
        let none = Instants {
            next: 0,
            end: 0,
            every_ns: self.every_ns,
        };
        let Some(clock) = self.clock else {
            self.clock = Some(captured);
            self.next = grid_ceil(captured, self.every_ns);
            return none;
        };
        if captured <= clock {
            return none;
        }
        self.clock = Some(captured);
        let count = (captured - self.next).max(0).div_euclid(self.every_ns);
        if count > MAX_GAP_SAMPLES {
            self.skipped += u64::try_from(count).unwrap_or(0);
            self.next = grid_ceil(captured, self.every_ns);
            return none;
        }
        let instants = Instants {
            next: self.next,
            end: captured,
            every_ns: self.every_ns,
        };
        self.next = grid_ceil(captured, self.every_ns);
        instants
    }
}

/// Grid instants in `[next, end)`, `every_ns` apart.
#[derive(Debug, Clone)]
pub struct Instants {
    next: i64,
    end: i64,
    every_ns: i64,
}

impl Iterator for Instants {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        (self.next < self.end).then(|| {
            let t = self.next;
            self.next += self.every_ns;
            t
        })
    }
}

/// `book_samples` rows being gathered: one row per venue, symbol and grid instant, the top
/// `levels` of each side in wide columns (`bid_px_0..`, `bid_qty_0..`, `ask_px_0..`,
/// `ask_qty_0..`).
#[derive(Debug)]
pub struct SampleRows {
    levels: usize,
    sample_ts_ns: Int64Builder,
    venue: StringBuilder,
    symbol: StringBuilder,
    seq: Int64Builder,
    last_update_id: Int64Builder,
    synced: BooleanBuilder,
    /// `levels` builders each for bid prices, bid quantities, ask prices, ask quantities.
    sides: [Vec<Int64Builder>; 4],
}

impl SampleRows {
    #[must_use]
    pub fn new(levels: usize) -> Self {
        let side = || (0..levels).map(|_| Int64Builder::new()).collect::<Vec<_>>();
        Self {
            levels,
            sample_ts_ns: Int64Builder::new(),
            venue: StringBuilder::new(),
            symbol: StringBuilder::new(),
            seq: Int64Builder::new(),
            last_update_id: Int64Builder::new(),
            synced: BooleanBuilder::new(),
            sides: [side(), side(), side(), side()],
        }
    }

    /// The table's schema at `levels` a side.
    #[must_use]
    pub fn schema(levels: usize) -> SchemaRef {
        let mut fields = vec![
            Field::new("sample_ts_ns", DataType::Int64, false),
            Field::new("venue", DataType::Utf8, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("seq", DataType::Int64, false),
            Field::new("last_update_id", DataType::Int64, true),
            Field::new("synced", DataType::Boolean, false),
        ];
        for prefix in ["bid_px", "bid_qty", "ask_px", "ask_qty"] {
            fields.extend(
                (0..levels).map(|i| Field::new(format!("{prefix}_{i}"), DataType::Int64, true)),
            );
        }
        Arc::new(Schema::new(fields))
    }

    /// Rows gathered so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends every row of a sample.
    pub fn push(&mut self, sample: &Sample) {
        for row in &sample.rows {
            self.sample_ts_ns.append_value(sample.sample_ts_ns);
            self.venue.append_value(sample.venue.id());
            self.symbol.append_value(&row.symbol);
            self.seq.append_value(sample.seq);
            self.last_update_id.append_option(row.last_update_id);
            self.synced.append_value(row.synced);
            let [bid_px, bid_qty, ask_px, ask_qty] = &mut self.sides;
            for i in 0..self.levels {
                bid_px[i].append_option(row.bids.get(i).map(|l| l.price));
                bid_qty[i].append_option(row.bids.get(i).map(|l| l.qty));
                ask_px[i].append_option(row.asks.get(i).map(|l| l.price));
                ask_qty[i].append_option(row.asks.get(i).map(|l| l.qty));
            }
        }
    }

    /// Takes the rows gathered so far as a batch, leaving the builders empty.
    ///
    /// # Errors
    /// Returns the Arrow error, which a batch of this schema never gives.
    pub fn take(&mut self) -> Result<RecordBatch, StoreError> {
        let column = |array: arrow::array::Int64Array| -> ArrayRef { Arc::new(array) };
        let mut columns: Vec<ArrayRef> = vec![
            column(self.sample_ts_ns.finish()),
            Arc::new(self.venue.finish()),
            Arc::new(self.symbol.finish()),
            column(self.seq.finish()),
            column(self.last_update_id.finish()),
            Arc::new(self.synced.finish()),
        ];
        for side in &mut self.sides {
            columns.extend(side.iter_mut().map(|builder| column(builder.finish())));
        }
        Ok(RecordBatch::try_new(Self::schema(self.levels), columns)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler() -> Sampler {
        Sampler::new(BookSample {
            levels: 1,
            every_ns: 100,
        })
    }

    #[test]
    fn grid_instants_round_up() {
        assert_eq!(grid_ceil(0, 100), 0);
        assert_eq!(grid_ceil(1, 100), 100);
        assert_eq!(grid_ceil(100, 100), 100);
        assert_eq!(grid_ceil(-1, 100), 0);
        assert_eq!(
            grid_ceil(i64::MAX, 100),
            i64::MAX,
            "saturates, never overflows"
        );
    }

    #[test]
    fn an_input_yields_the_instants_its_capture_time_passes() {
        let mut sampler = sampler();
        assert_eq!(
            sampler.instants(150).count(),
            0,
            "the first input starts the clock"
        );
        assert_eq!(sampler.instants(200).collect::<Vec<_>>(), Vec::<i64>::new());
        assert_eq!(sampler.instants(201).collect::<Vec<_>>(), vec![200]);
        assert_eq!(
            sampler.instants(180).count(),
            0,
            "a clock step back changes nothing"
        );
        assert_eq!(sampler.instants(500).collect::<Vec<_>>(), vec![300, 400]);
        let far = 500 + 100 * (MAX_GAP_SAMPLES + 2);
        assert_eq!(sampler.instants(far).count(), 0);
        assert!(sampler.skipped() > 0);
    }

    #[test]
    fn the_schema_is_keys_then_wide_levels() {
        let samples = SampleRows::schema(2);
        assert_eq!(samples.fields().len(), 6 + 4 * 2);
        assert_eq!(samples.field(1).name(), "venue");
        assert_eq!(samples.field(6).name(), "bid_px_0");
        assert_eq!(samples.field(8).name(), "bid_qty_0");
        assert_eq!(samples.field(13).name(), "ask_qty_1");
    }
}
