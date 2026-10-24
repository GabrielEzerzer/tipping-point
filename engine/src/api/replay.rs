//! One WebSocket connection's replay of a recorded market run. The rotated hours come from the
//! run's Parquet through `DuckDB`, a slice of the capture clock at a time (a seek is a query on
//! `book_samples`); when the run is the one being recorded live, the replay lists its files
//! again each time it reaches their end, so it takes in the hours rotated while it played, and
//! the hour the rotate step has not reached comes from the hub's memory. Book samples and trades
//! go out in capture order at `speed` times real time, as the same `book` and `trades` messages
//! the live feed sends.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::SchemaRef;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep_until};

use super::hub::Hub;
use super::messages::TradeMessage;
use super::table::{Table, TableError, locate};
use crate::market::sync::Level;
use crate::store::run::MARKET;
use crate::store::samples::{BookRow, Sample};
use crate::store::{StoreError, query};
use crate::venue::Venue;

/// The capture-clock span read from Parquet at a time.
const SLICE_NS: i64 = 10_000_000_000;
/// Items queued for the connection; the replay waits when it is full.
const QUEUE: usize = 256;

/// What a replay sends its connection.
#[derive(Debug, Clone)]
pub enum Item {
    Sample(Arc<Sample>),
    Trade(Arc<TradeMessage>),
    /// This many grid instants are missing before the next sample: memory no longer reaches back
    /// to where the rotated hours end, or to the requested start.
    BookGap(u64),
    /// The run is over.
    End,
    /// The replay could not go on (a damaged or vanished Parquet file, a `DuckDB` error): not
    /// the end of the run.
    Failed(String),
}

/// A running replay. Dropping it stops it.
#[derive(Debug)]
pub struct Replay {
    pub items: mpsc::Receiver<Item>,
    paused: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl Drop for Replay {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// The run's two replayed tables, those that have Parquet yet.
#[derive(Debug, Clone)]
struct Tables {
    samples: Option<(Table, SchemaRef)>,
    trades: Option<(Table, SchemaRef)>,
}

impl Tables {
    /// The tables of `run` as its files are now. While the run is being recorded, rotate may be
    /// writing a segment: it writes the segment's `book_samples` file last, under the name its
    /// other files have, so only the trades files whose `book_samples` file is listed (listed
    /// first) are kept, and a segment comes in whole or not at all.
    fn open(data_dir: &Path, run: &str, recording: bool) -> Result<Self, TableError> {
        let open = |name: &str| -> Result<Option<(Table, SchemaRef)>, TableError> {
            match locate(data_dir, MARKET, run, name) {
                Ok(table) => {
                    let schema = table.schema()?;
                    Ok(Some((table, schema)))
                }
                Err(TableError::NotFound(_)) => Ok(None),
                Err(error) => Err(error),
            }
        };
        let samples = open("book_samples")?;
        // A run rotated before there were venues keeps its trades in `agg_trades`.
        let mut trades = match open("trades")? {
            Some(trades) => Some(trades),
            None => open("agg_trades")?,
        };
        if recording {
            let rotated: BTreeSet<OsString> = samples
                .iter()
                .flat_map(|(table, _)| &table.files)
                .filter_map(|file| file.file_name().map(ToOwned::to_owned))
                .collect();
            if let Some((table, _)) = &mut trades {
                table
                    .files
                    .retain(|file| file.file_name().is_some_and(|name| rotated.contains(name)));
            }
            trades = trades.filter(|(table, _)| !table.files.is_empty());
        }
        Ok(Self { samples, trades })
    }

    /// Only the files `before` did not list; `None` when it listed them all.
    fn added_since(&self, before: &Self) -> Option<Self> {
        let added = |now: &Option<(Table, SchemaRef)>, then: &Option<(Table, SchemaRef)>| {
            let (mut table, schema) = now.clone()?;
            let listed: &[PathBuf] = then.as_ref().map_or(&[], |(table, _)| &table.files);
            table.files.retain(|file| !listed.contains(file));
            (!table.files.is_empty()).then_some((table, schema))
        };
        let added = Self {
            samples: added(&self.samples, &before.samples),
            trades: added(&self.trades, &before.trades),
        };
        (added.samples.is_some() || added.trades.is_some()).then_some(added)
    }
}

/// The run a replay plays while it is being recorded live: the hub whose memory carries it on
/// after its rotated hours, the venue recording it, and where its files are, to list them again.
#[derive(Debug)]
struct Recording {
    hub: Arc<Hub>,
    venue: Venue,
    data_dir: PathBuf,
    run: String,
}

impl Replay {
    /// Starts replaying `run` from `from_ms` (its start when `None`).
    ///
    /// # Errors
    /// Returns [`TableError`] when the run is neither recorded with book samples nor the run
    /// being recorded live.
    pub fn start(
        data_dir: &Path,
        hub: Arc<Hub>,
        run: &str,
        from_ms: Option<i64>,
        speed: f64,
    ) -> Result<Self, TableError> {
        // The venue whose live feed records this run, if one does: memory carries the run on.
        let live = hub.started().and_then(|source| source.venue_recording(run));
        let tables = Tables::open(data_dir, run, live.is_some())?;
        if tables.samples.is_none() && live.is_none() {
            return Err(TableError::NotFound(format!(
                "market run {run:?} has no book samples to replay"
            )));
        }
        let (tx, items) = mpsc::channel(QUEUE);
        let (paused, paused_rx) = watch::channel(false);
        let start_ns = from_ms.map_or(i64::MIN, |ms| ms.saturating_mul(1_000_000));
        let recording = live.map(|venue| Recording {
            hub,
            venue,
            data_dir: data_dir.to_owned(),
            run: run.to_owned(),
        });
        let task = tokio::spawn(feed(tables, recording, start_ns, speed, tx, paused_rx));
        Ok(Self {
            items,
            paused,
            task,
        })
    }

    pub fn pause(&self) {
        self.paused.send_replace(true);
    }

    pub fn play(&self) {
        self.paused.send_replace(false);
    }
}

/// Runs `work` on a blocking thread, as `DuckDB` and the file system need; its error as text.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, TableError> + Send + 'static,
) -> Result<T, String> {
    match tokio::task::spawn_blocking(work).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(error.to_string()),
        Err(stopped) => Err(format!("the replay stopped: {stopped}")),
    }
}

/// Sends the run's items from `from_ns` on, paced, then [`Item::End`]. `live` is set when the
/// run is being recorded live.
async fn feed(
    mut tables: Tables,
    live: Option<Recording>,
    start_ns: i64,
    speed: f64,
    tx: mpsc::Sender<Item>,
    mut paused: watch::Receiver<bool>,
) {
    let mut pacer = Pacer::new(speed);
    let mut cursor = start_ns;
    // Where the Parquet left off: the last sample instant and the last trade sent from it.
    let mut last_sample = i64::MIN;
    let mut last_trade: Option<u64> = None;
    // Where the Parquet had left off when its files were last listed again: the replay goes back
    // for the new files' first rows, and skips what it sent before.
    let mut sent: (i64, Option<u64>) = (i64::MIN, None);
    loop {
        let slice = tables.clone();
        let found = match blocking(move || Ok(slice_or_next(&slice, cursor)?)).await {
            Ok(found) => found,
            Err(error) => {
                let _ = tx.send(Item::Failed(error)).await;
                return;
            }
        };
        match found {
            Found::Items(items, next) => {
                for (t, item) in items {
                    match &item {
                        Item::Sample(sample) if sample.sample_ts_ns <= sent.0 => continue,
                        Item::Trade(trade) if sent.1.is_some_and(|seq| trade.seq <= seq) => {
                            continue;
                        }
                        Item::Sample(sample) => last_sample = last_sample.max(sample.sample_ts_ns),
                        Item::Trade(trade) => last_trade = last_trade.max(Some(trade.seq)),
                        Item::BookGap(_) | Item::End | Item::Failed(_) => {}
                    }
                    if !pacer.wait(t, &mut paused).await || tx.send(item).await.is_err() {
                        return;
                    }
                }
                cursor = next;
            }
            Found::Skip(next) => cursor = next,
            Found::Done => {
                let Some(live) = &live else { break };
                // Rotate turns each closed hour of the run into Parquet: list the files again
                // and play on through any it added, before memory takes over.
                let (before, data_dir, run) =
                    (tables.clone(), live.data_dir.clone(), live.run.clone());
                match blocking(move || more_files(&before, &data_dir, &run, start_ns)).await {
                    Ok(Some((more, first))) => {
                        tables = more;
                        cursor = cursor.min(first);
                        sent = (last_sample, last_trade);
                    }
                    Ok(None) => break,
                    Err(error) => {
                        let _ = tx.send(Item::Failed(error)).await;
                        return;
                    }
                }
            }
        }
    }
    if let Some(Recording { hub, venue, .. }) = live {
        // The rotated tables are a prefix of the run: memory carries on after them, from the
        // requested start, and says so when it no longer reaches back that far.
        let recent = hub.recent(venue, start_ns.saturating_sub(1));
        let samples: Vec<Arc<Sample>> = recent
            .samples
            .into_iter()
            .filter(|sample| sample.sample_ts_ns > last_sample)
            .collect();
        let every = hub.config().book_sample.every_ns;
        let first = samples.first().map(|sample| sample.sample_ts_ns);
        if let Some(lost) =
            missing_instants(last_sample, start_ns, recent.first_sample, first, every)
            && tx.send(Item::BookGap(lost)).await.is_err()
        {
            return;
        }
        let trades = recent
            .trades
            .into_iter()
            .filter(|(_, trade)| last_trade.is_none_or(|last| trade.seq > last))
            .collect();
        for (t, item) in merge(samples, trades) {
            if !pacer.wait(t, &mut paused).await || tx.send(item).await.is_err() {
                return;
            }
        }
    }
    let _ = tx.send(Item::End).await;
}

/// Grid instants missing between where the replay is (the last sample sent, or else the
/// requested start, but never before the run's first sample `first_ever`, where a replay with no
/// start begins) and the first sample memory still holds; `None` when they meet or when there is
/// nothing to compare. Samples are one grid step apart however quiet the market, so a longer step
/// is samples lost to the replay.
fn missing_instants(
    last_sample: i64,
    start_ns: i64,
    first_ever: Option<i64>,
    first_held: Option<i64>,
    every_ns: i64,
) -> Option<u64> {
    let every_ns = every_ns.max(1);
    let first = first_held?;
    let expected = if last_sample != i64::MIN {
        last_sample.checked_add(every_ns)?
    } else if start_ns != i64::MIN {
        // The first grid instant at or after the start; a run's first sample comes a second or
        // two after the run starts, and there is nothing before it to lose.
        let start = start_ns
            .div_euclid(every_ns)
            .checked_add(i64::from(start_ns.rem_euclid(every_ns) != 0))?
            .checked_mul(every_ns)?;
        first_ever.map_or(start, |first_ever| start.max(first_ever))
    } else {
        first_ever?
    };
    let lost = u64::try_from((first - expected).div_euclid(every_ns)).ok()?;
    (lost > 0).then_some(lost)
}

/// The run's tables listed again, and the first time at or after `from` in the files they
/// gained; `None` when they gained none with rows from then on. A segment's rows come after the
/// last segment's, but the slice the replay last read can reach past where those ended, so the
/// replay goes back to the new files' first row.
fn more_files(
    before: &Tables,
    data_dir: &Path,
    run: &str,
    from: i64,
) -> Result<Option<(Tables, i64)>, TableError> {
    let now = Tables::open(data_dir, run, true)?;
    let Some(added) = now.added_since(before) else {
        return Ok(None);
    };
    let conn = query::connect()?;
    Ok(first(&added, &conn, from)?.map(|t| (now, t)))
}

/// What one look at the Parquet found.
enum Found {
    /// The items of the slice from the cursor, and where the next slice starts.
    Items(Vec<(i64, Item)>, i64),
    /// Nothing in the slice; the next data starts here.
    Skip(i64),
    /// Nothing at or after the cursor.
    Done,
}

/// The slice `[cursor, cursor + SLICE_NS)` of both tables, or where to look next.
fn slice_or_next(tables: &Tables, cursor: i64) -> Result<Found, StoreError> {
    let conn = query::connect()?;
    let start = if cursor == i64::MIN {
        match first(tables, &conn, i64::MIN)? {
            Some(t) => t,
            None => return Ok(Found::Done),
        }
    } else {
        cursor
    };
    let end = start.saturating_add(SLICE_NS);
    let mut samples = Vec::new();
    if let Some((table, schema)) = &tables.samples {
        for batch in table.window(&conn, schema, "sample_ts_ns", (start, end))? {
            samples.extend(samples_of(&batch, run_venue(table))?);
        }
    }
    let mut trades = Vec::new();
    if let Some((table, schema)) = &tables.trades {
        for batch in table.window(&conn, schema, "capture_ts_ns", (start, end))? {
            trades.extend(trades_of(&batch, run_venue(table))?);
        }
    }
    if samples.is_empty() && trades.is_empty() {
        return Ok(match first(tables, &conn, end)? {
            Some(next) => Found::Skip(next),
            None => Found::Done,
        });
    }
    Ok(Found::Items(merge(samples, trades), end))
}

/// The earliest sample or trade time at or after `from`.
fn first(tables: &Tables, conn: &duckdb::Connection, from: i64) -> Result<Option<i64>, StoreError> {
    let mut earliest: Option<i64> = None;
    if let Some((table, _)) = &tables.samples
        && let Some(t) = table.next_at(conn, "sample_ts_ns", from)?
    {
        earliest = Some(earliest.map_or(t, |e| e.min(t)));
    }
    if let Some((table, _)) = &tables.trades
        && let Some(t) = table.next_at(conn, "capture_ts_ns", from)?
    {
        earliest = Some(earliest.map_or(t, |e| e.min(t)));
    }
    Ok(earliest)
}

/// Samples and trades in capture order. A trade captured at a sample's instant was applied
/// before the sample was taken, so it goes first.
fn merge(samples: Vec<Arc<Sample>>, trades: Vec<(i64, Arc<TradeMessage>)>) -> Vec<(i64, Item)> {
    let mut items: Vec<(i64, u8, Item)> = trades
        .into_iter()
        .map(|(t, trade)| (t, 0, Item::Trade(trade)))
        .chain(
            samples
                .into_iter()
                .map(|sample| (sample.sample_ts_ns, 1, Item::Sample(sample))),
        )
        .collect();
    items.sort_by_key(|(t, rank, _)| (*t, *rank));
    items.into_iter().map(|(t, _, item)| (t, item)).collect()
}

fn int64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, StoreError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| missing(name))
}

fn missing(name: &str) -> StoreError {
    StoreError::Arrow(arrow::error::ArrowError::SchemaError(format!(
        "no column {name} of the stored type"
    )))
}

fn strings<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray, StoreError> {
    batch
        .column_by_name(name)
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| missing(name))
}

/// The venue a market run holds (Binance's, for runs recorded before runs said).
fn run_venue(table: &Table) -> Venue {
    table.venue.unwrap_or(Venue::BinanceUsdm)
}

/// The venues of a batch's rows: its `venue` column, or, in a table rotated before there were
/// venues, which has none, the run's venue for every row.
enum Venues<'a> {
    Column(&'a StringArray),
    Run(Venue),
}

impl Venues<'_> {
    fn of(batch: &RecordBatch, run: Venue) -> Result<Venues<'_>, StoreError> {
        Ok(match batch.column_by_name("venue") {
            Some(_) => Venues::Column(strings(batch, "venue")?),
            None => Venues::Run(run),
        })
    }

    fn at(&self, row: usize) -> Result<Venue, StoreError> {
        match self {
            Self::Column(venues) => venues.value(row).parse().map_err(|error: String| {
                StoreError::Arrow(arrow::error::ArrowError::SchemaError(error))
            }),
            Self::Run(venue) => Ok(*venue),
        }
    }
}

/// The trade ids of a batch's rows: `trade_id` as stored, or a pre-venue `agg_trades` table's
/// numeric `agg_trade_id`.
enum TradeIds<'a> {
    Text(&'a StringArray),
    Number(&'a Int64Array),
}

impl TradeIds<'_> {
    fn of(batch: &RecordBatch) -> Result<TradeIds<'_>, StoreError> {
        Ok(match batch.column_by_name("trade_id") {
            Some(_) => TradeIds::Text(strings(batch, "trade_id")?),
            None => TradeIds::Number(int64(batch, "agg_trade_id")?),
        })
    }

    fn at(&self, row: usize) -> String {
        match self {
            Self::Text(ids) => ids.value(row).to_owned(),
            Self::Number(ids) => ids.value(row).to_string(),
        }
    }
}

/// `book_samples` rows back into samples, one per venue and instant; `venue` is the run's, for a
/// table rotated before rows named theirs.
fn samples_of(batch: &RecordBatch, venue: Venue) -> Result<Vec<Arc<Sample>>, StoreError> {
    let times = int64(batch, "sample_ts_ns")?;
    let seqs = int64(batch, "seq")?;
    let ids = int64(batch, "last_update_id")?;
    let venues = Venues::of(batch, venue)?;
    let symbols = strings(batch, "symbol")?;
    let synced = batch
        .column_by_name("synced")
        .and_then(|column| column.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| missing("synced"))?;
    let levels = batch
        .schema()
        .fields()
        .iter()
        .filter(|field| field.name().starts_with("bid_px_"))
        .count();
    let side = |prefix: &str| -> Result<Vec<(&Int64Array, &Int64Array)>, StoreError> {
        (0..levels)
            .map(|i| {
                Ok((
                    int64(batch, &format!("{prefix}_px_{i}"))?,
                    int64(batch, &format!("{prefix}_qty_{i}"))?,
                ))
            })
            .collect()
    };
    let (bids, asks) = (side("bid")?, side("ask")?);
    let levels_at = |columns: &[(&Int64Array, &Int64Array)], row: usize| -> Vec<Level> {
        columns
            .iter()
            .take_while(|(px, qty)| px.is_valid(row) && qty.is_valid(row))
            .map(|(px, qty)| Level {
                price: px.value(row),
                qty: qty.value(row),
            })
            .collect()
    };
    let mut out: Vec<Sample> = Vec::new();
    for row in 0..batch.num_rows() {
        let book = BookRow {
            symbol: symbols.value(row).to_owned(),
            synced: synced.value(row),
            last_update_id: ids.is_valid(row).then(|| ids.value(row)),
            bids: levels_at(&bids, row),
            asks: levels_at(&asks, row),
        };
        let (t, venue) = (times.value(row), venues.at(row)?);
        match out.last_mut() {
            Some(sample) if sample.sample_ts_ns == t && sample.venue == venue => {
                sample.rows.push(book);
            }
            _ => out.push(Sample {
                sample_ts_ns: t,
                seq: seqs.value(row),
                venue,
                rows: vec![book],
            }),
        }
    }
    Ok(out.into_iter().map(Arc::new).collect())
}

/// `trades` rows (or a pre-venue run's `agg_trades` rows) back into trade messages, each with its
/// capture time; `venue` is the run's, for a table rotated before rows named theirs.
fn trades_of(
    batch: &RecordBatch,
    venue: Venue,
) -> Result<Vec<(i64, Arc<TradeMessage>)>, StoreError> {
    let seqs = int64(batch, "seq")?;
    let traded = int64(batch, "trade_ts_ns")?;
    let captured = int64(batch, "capture_ts_ns")?;
    let prices = int64(batch, "price")?;
    let qtys = int64(batch, "qty")?;
    let venues = Venues::of(batch, venue)?;
    let symbols = strings(batch, "symbol")?;
    let ids = TradeIds::of(batch)?;
    let makers = batch
        .column_by_name("buyer_is_maker")
        .and_then(|column| column.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| missing("buyer_is_maker"))?;
    (0..batch.num_rows())
        .map(|row| {
            let trade = TradeMessage {
                venue: venues.at(row)?,
                symbol: symbols.value(row).to_owned(),
                seq: u64::try_from(seqs.value(row)).unwrap_or_default(),
                ts_ms: u64::try_from(traded.value(row).div_euclid(1_000_000)).unwrap_or_default(),
                id: ids.at(row),
                px: prices.value(row),
                qty: qtys.value(row),
                buyer_is_maker: makers.value(row),
            };
            Ok((captured.value(row), Arc::new(trade)))
        })
        .collect()
}

/// The slowest replay a client may ask for, besides 0 (as fast as possible): a thousandth of
/// real time, a 100 ms instant every 100 s.
pub const MIN_SPEED: f64 = 1e-3;

/// A replay speed a client may ask for: 0, or a finite number at least [`MIN_SPEED`].
#[must_use]
pub fn valid_speed(speed: f64) -> bool {
    speed.is_finite() && speed >= 0.0 && !(0.0 < speed && speed < MIN_SPEED)
}

/// The wait until an item `elapsed_ns` after the first on the capture clock is due, at `speed`
/// times real time. Capped at a day, so no speed and no gap in a run can overflow the clock.
pub(crate) fn delay(elapsed_ns: i64, speed: f64) -> Duration {
    const LONGEST: Duration = Duration::from_hours(24);
    #[allow(clippy::cast_precision_loss)]
    let seconds = elapsed_ns.max(0) as f64 / 1e9 / speed;
    Duration::try_from_secs_f64(seconds).map_or(LONGEST, |wait| wait.min(LONGEST))
}

/// Spaces items out at `speed` times the capture clock, and holds them while paused.
struct Pacer {
    speed: f64,
    /// Wall time and capture time of the item the pacing counts from.
    origin: Option<(Instant, i64)>,
}

impl Pacer {
    fn new(speed: f64) -> Self {
        Self {
            speed,
            origin: None,
        }
    }

    /// Waits until the item at `t` is due. False when the replay should stop.
    async fn wait(&mut self, t: i64, paused: &mut watch::Receiver<bool>) -> bool {
        loop {
            if *paused.borrow_and_update() {
                if paused.wait_for(|paused| !paused).await.is_err() {
                    return false;
                }
                // Count again from here: the pause takes no capture time.
                self.origin = None;
            }
            if self.speed <= 0.0 {
                return true;
            }
            let (wall, t0) = *self.origin.get_or_insert((Instant::now(), t));
            tokio::select! {
                () = sleep_until(wall + delay(t - t0, self.speed)) => return true,
                changed = paused.changed() => {
                    if changed.is_err() {
                        return false;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use arrow::array::ArrayRef;
    use arrow::datatypes::{DataType, Field, Schema};

    use super::super::hub::{HubConfig, Published, VenueSource};
    use super::*;
    use crate::store::run::MANIFEST;
    use crate::store::samples::SampleRows;

    const RUN: &str = "20261024T100000Z-live";
    const T0: i64 = 1_700_000_000_000_000_000;
    const STEP: i64 = 100_000_000;

    /// The book sample of grid instant `i`.
    fn sample_at(i: i64) -> Sample {
        Sample {
            sample_ts_ns: T0 + i * STEP,
            seq: 2 * i,
            venue: Venue::BinanceUsdm,
            rows: vec![BookRow {
                symbol: "BTCUSDT".to_owned(),
                synced: true,
                last_update_id: Some(i),
                bids: vec![Level {
                    price: 1000,
                    qty: 1 + i,
                }],
                asks: vec![Level {
                    price: 1001,
                    qty: 1,
                }],
            }],
        }
    }

    /// The trade captured 10 ms after grid instant `i`, with its capture time.
    fn trade_at(i: i64) -> (i64, TradeMessage) {
        let captured = T0 + i * STEP + 10_000_000;
        let trade = TradeMessage {
            venue: Venue::BinanceUsdm,
            symbol: "BTCUSDT".to_owned(),
            seq: u64::try_from(2 * i + 1).expect("a positive seq"),
            ts_ms: u64::try_from(captured / 1_000_000).expect("a positive time"),
            id: i.to_string(),
            px: 1000,
            qty: 1,
            buyer_is_maker: false,
        };
        (captured, trade)
    }

    fn write_parquet(path: &Path, batch: &RecordBatch) {
        fs::create_dir_all(path.parent().expect("a table dir")).expect("the table dir");
        let file = fs::File::create(path).expect("a Parquet file");
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None)
            .expect("a Parquet writer");
        writer.write(batch).expect("rows written");
        writer.close().expect("a footer");
    }

    /// The segment of grid instants `instants` as rotate leaves it: its `trades` file (the trades
    /// after `trades`' instants), then its `book_samples` file, both named after the segment.
    fn write_segment(run_dir: &Path, instants: std::ops::Range<i64>, trades: &[i64]) {
        let name = format!("{:020}.parquet", instants.start);
        let trades: Vec<(i64, TradeMessage)> = trades.iter().map(|&i| trade_at(i)).collect();
        let int64 = |value: &dyn Fn(&(i64, TradeMessage)) -> i64| -> ArrayRef {
            Arc::new(Int64Array::from(
                trades.iter().map(value).collect::<Vec<_>>(),
            ))
        };
        let text = |value: &dyn Fn(&(i64, TradeMessage)) -> String| -> ArrayRef {
            Arc::new(StringArray::from(
                trades.iter().map(value).collect::<Vec<_>>(),
            ))
        };
        let field = |name: &str, kind: DataType| Field::new(name, kind, false);
        let schema = Arc::new(Schema::new(vec![
            field("seq", DataType::Int64),
            field("trade_ts_ns", DataType::Int64),
            field("capture_ts_ns", DataType::Int64),
            field("venue", DataType::Utf8),
            field("symbol", DataType::Utf8),
            field("trade_id", DataType::Utf8),
            field("price", DataType::Int64),
            field("qty", DataType::Int64),
            field("buyer_is_maker", DataType::Boolean),
        ]));
        let columns = vec![
            int64(&|(_, trade)| i64::try_from(trade.seq).expect("a seq")),
            int64(&|(t, _)| *t),
            int64(&|(t, _)| *t),
            text(&|(_, trade)| trade.venue.id().to_owned()),
            text(&|(_, trade)| trade.symbol.clone()),
            text(&|(_, trade)| trade.id.clone()),
            int64(&|(_, trade)| trade.px),
            int64(&|(_, trade)| trade.qty),
            Arc::new(BooleanArray::from(vec![false; trades.len()])) as ArrayRef,
        ];
        let batch = RecordBatch::try_new(schema, columns).expect("a trades batch");
        write_parquet(&run_dir.join("trades").join(&name), &batch);
        let mut rows = SampleRows::new(1);
        for i in instants {
            rows.push(&sample_at(i));
        }
        let samples = rows.take().expect("a book_samples batch");
        write_parquet(&run_dir.join("book_samples").join(&name), &samples);
    }

    /// The live run replayed from its start while rotate turns another hour into Parquet: the
    /// replay plays the new hour from its files, then memory, every instant and trade once and no
    /// gap. Memory reaches back only into that hour, so before, the replay went from the first
    /// hour's files to memory and said instants were lost that were on disk.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_replay_of_the_live_run_takes_in_the_hours_rotated_while_it_plays() {
        let dir = std::env::temp_dir().join(format!("tp-replay-rotated-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let run_dir = dir.join(MARKET).join(RUN);
        fs::create_dir_all(&run_dir).expect("the run dir");
        let manifest =
            r#"{"key":"seq","tables":["trades","book_samples"],"params":{"venue":"binance-usdm"}}"#;
        fs::write(run_dir.join(MANIFEST), manifest).expect("the manifest");
        // 650 instants: more than the replay queues, and the hour ends inside a slice.
        write_segment(&run_dir, 0..650, &[25, 125, 225, 325, 425, 525, 625]);
        let hub = Hub::new(HubConfig::default());
        let feed = VenueSource {
            scales: BTreeMap::new(),
            run: Some(RUN.to_owned()),
        };
        assert!(hub.start(Venue::BinanceUsdm, feed));
        // Memory holds the hour rotate has not reached and the end of the one before it.
        for i in 1000..1500 {
            hub.publish_sample(Published {
                sample: Arc::new(sample_at(i)),
                ipc: Vec::new(),
                captured_ns: T0 + i * STEP,
            });
        }
        for i in [1025, 1125, 1225, 1325, 1425] {
            let (captured, trade) = trade_at(i);
            hub.publish_trade(captured, trade);
        }

        let mut replay =
            Replay::start(&dir, Arc::clone(&hub), RUN, None, 0.0).expect("the live run replays");
        // The replay waits on its full queue inside the first file while rotate adds the next.
        write_segment(&run_dir, 650..1300, &[725, 825, 925, 1025, 1125, 1225]);
        let (mut instants, mut trades) = (Vec::new(), Vec::new());
        loop {
            match replay.items.recv().await.expect("the replay ends with End") {
                Item::Sample(sample) => instants.push((sample.sample_ts_ns - T0) / STEP),
                Item::Trade(trade) => trades.push(trade.seq),
                Item::BookGap(lost) => panic!("a gap of {lost} instants, all on disk or in memory"),
                Item::End => break,
                Item::Failed(why) => panic!("the replay failed: {why}"),
            }
        }
        let first_wrong = instants.iter().zip(0..).find(|&(got, want)| *got != want);
        assert!(
            instants.len() == 1500 && first_wrong.is_none(),
            "every instant once, in order: {} sent, first wrong (sent, due) {first_wrong:?}",
            instants.len()
        );
        let every_trade: Vec<u64> = (0..15).map(|k| 2 * (100 * k + 25) + 1).collect();
        assert_eq!(trades, every_trade, "every trade once");

        // A segment rotate is still writing, its trades file there and its book_samples file not
        // yet, stays out of a recording run's tables until it is whole.
        let partial = run_dir.join("trades").join(format!("{:020}.parquet", 1300));
        fs::copy(
            run_dir.join("trades").join(format!("{:020}.parquet", 650)),
            partial,
        )
        .expect("a trades file");
        let trades_files = |recording| {
            Tables::open(&dir, RUN, recording)
                .expect("the tables")
                .trades
                .map(|(table, _)| table.files.len())
        };
        assert_eq!(trades_files(true), Some(2));
        assert_eq!(trades_files(false), Some(3));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_step_longer_than_the_grid_is_instants_lost() {
        let lost = |last_sample, start_ns, first_ever, first_held| {
            missing_instants(last_sample, start_ns, first_ever, first_held, 100)
        };
        // After the rotated hours: the next sample is due one step after the last one sent.
        assert_eq!(lost(1_000, i64::MIN, Some(0), Some(1_100)), None);
        assert_eq!(lost(1_000, i64::MIN, Some(0), Some(1_200)), Some(1));
        assert_eq!(lost(1_000, i64::MIN, Some(0), Some(1_400)), Some(3));
        assert_eq!(lost(1_000, i64::MIN, Some(0), Some(1_050)), None);
        // Nothing rotated: memory must start at the first instant of the requested start...
        assert_eq!(lost(i64::MIN, 950, Some(1_000), Some(1_000)), None);
        assert_eq!(lost(i64::MIN, 1_000, Some(1_000), Some(1_500)), Some(5));
        // ...but never before the run's first sample: a fresh run replayed from its start (the
        // first sample 1.5 s later) lost nothing, until memory lets its first samples go.
        assert_eq!(lost(i64::MIN, 0, Some(1_500), Some(1_500)), None);
        assert_eq!(lost(i64::MIN, 0, Some(1_500), Some(2_000)), Some(5));
        assert_eq!(lost(i64::MIN, 1_750, Some(1_500), Some(2_000)), Some(2));
        // No start: from the run's first sample.
        let no_start = |first_held| lost(i64::MIN, i64::MIN, Some(1_500), Some(first_held));
        assert_eq!(no_start(1_500), None);
        assert_eq!(no_start(2_000), Some(5), "memory let the first samples go");
        assert_eq!(lost(1_000, i64::MIN, Some(0), None), None, "memory empty");
    }

    #[test]
    fn any_speed_gives_a_finite_wait() {
        assert_eq!(delay(1_000_000_000, 1.0), Duration::from_secs(1));
        assert_eq!(delay(1_000_000_000, 10.0), Duration::from_millis(100));
        assert_eq!(delay(-5, 1.0), Duration::ZERO);
        let day = Duration::from_hours(24);
        assert_eq!(delay(1_000_000_000, 1e-300), day, "no panic, capped");
        assert_eq!(delay(i64::MAX, f64::MIN_POSITIVE), day);
        assert_eq!(
            delay(1, 0.0),
            day,
            "callers never pace at 0, but it must not panic"
        );
        assert!(valid_speed(0.0) && valid_speed(MIN_SPEED) && valid_speed(1e9));
        assert!(!valid_speed(1e-300) && !valid_speed(-1.0) && !valid_speed(f64::NAN));
        assert!(!valid_speed(f64::INFINITY));
    }

    #[test]
    fn a_trade_at_a_sample_instant_goes_first() {
        let sample = Arc::new(Sample {
            sample_ts_ns: 100,
            seq: 1,
            venue: Venue::BinanceUsdm,
            rows: Vec::new(),
        });
        let trade = |seq: u64| {
            Arc::new(TradeMessage {
                venue: Venue::BinanceUsdm,
                symbol: "BTCUSDT".to_owned(),
                seq,
                ts_ms: 0,
                id: seq.to_string(),
                px: 1,
                qty: 1,
                buyer_is_maker: false,
            })
        };
        let merged = merge(vec![sample], vec![(100, trade(2)), (50, trade(1))]);
        let order: Vec<(i64, bool)> = merged
            .iter()
            .map(|(t, item)| (*t, matches!(item, Item::Sample(_))))
            .collect();
        assert_eq!(order, vec![(50, false), (100, false), (100, true)]);
    }
}
