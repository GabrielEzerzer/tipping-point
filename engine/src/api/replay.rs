//! One WebSocket connection's replay of a recorded market run. The rotated hours come from the
//! run's Parquet through `DuckDB`, a slice of the capture clock at a time (a seek is a query on
//! `book_samples`); when the run is the one being recorded live, the hour the rotate step has
//! not reached comes from the hub's memory. Book samples and trades go out in capture order at
//! `speed` times real time, as the same `book` and `trades` messages the live feed sends.

use std::path::Path;
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
use crate::binance::sync::Level;
use crate::store::run::MARKET;
use crate::store::samples::{BookRow, Sample};
use crate::store::{StoreError, query};

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
        let live = hub
            .started()
            .and_then(|source| source.run.clone())
            .is_some_and(|live| live == run);
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
        let tables = Tables {
            samples: open("book_samples")?,
            trades: open("agg_trades")?,
        };
        if tables.samples.is_none() && !live {
            return Err(TableError::NotFound(format!(
                "market run {run:?} has no book samples to replay"
            )));
        }
        let (tx, items) = mpsc::channel(QUEUE);
        let (paused, paused_rx) = watch::channel(false);
        let start_ns = from_ms.map_or(i64::MIN, |ms| ms.saturating_mul(1_000_000));
        let task = tokio::spawn(feed(
            tables,
            live.then_some(hub),
            start_ns,
            speed,
            tx,
            paused_rx,
        ));
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

/// Sends the run's items from `from_ns` on, paced, then [`Item::End`].
async fn feed(
    tables: Tables,
    live: Option<Arc<Hub>>,
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
    loop {
        let slice = tables.clone();
        let Ok(Ok(found)) =
            tokio::task::spawn_blocking(move || slice_or_next(&slice, cursor)).await
        else {
            break;
        };
        match found {
            Found::Items(items, next) => {
                for (t, item) in items {
                    match &item {
                        Item::Sample(sample) => last_sample = last_sample.max(sample.sample_ts_ns),
                        Item::Trade(trade) => last_trade = last_trade.max(Some(trade.seq)),
                        Item::BookGap(_) | Item::End => {}
                    }
                    if !pacer.wait(t, &mut paused).await || tx.send(item).await.is_err() {
                        return;
                    }
                }
                cursor = next;
            }
            Found::Skip(next) => cursor = next,
            Found::Done => break,
        }
    }
    if let Some(hub) = live {
        // The rotated tables are a prefix of the run: memory carries on after them, from the
        // requested start, and says so when it no longer reaches back that far.
        let recent = hub.recent(start_ns.saturating_sub(1));
        let samples: Vec<Arc<Sample>> = recent
            .samples
            .into_iter()
            .filter(|sample| sample.sample_ts_ns > last_sample)
            .collect();
        let every = hub.config().book_sample.every_ns;
        let first = samples.first().map(|sample| sample.sample_ts_ns);
        if let Some(lost) = missing_instants(last_sample, start_ns, first, every)
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
/// requested start) and the first sample memory still holds; `None` when they meet or when there
/// is nothing to compare. Samples are one grid step apart however quiet the market, so a longer
/// step is samples lost to the replay.
fn missing_instants(
    last_sample: i64,
    start_ns: i64,
    first_held: Option<i64>,
    every_ns: i64,
) -> Option<u64> {
    let every_ns = every_ns.max(1);
    let first = first_held?;
    let expected = if last_sample != i64::MIN {
        last_sample.checked_add(every_ns)?
    } else if start_ns != i64::MIN {
        // The first grid instant at or after the start.
        start_ns
            .div_euclid(every_ns)
            .checked_add(i64::from(start_ns.rem_euclid(every_ns) != 0))?
            .checked_mul(every_ns)?
    } else {
        return None;
    };
    let lost = u64::try_from((first - expected).div_euclid(every_ns)).ok()?;
    (lost > 0).then_some(lost)
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
            samples.extend(samples_of(&batch)?);
        }
    }
    let mut trades = Vec::new();
    if let Some((table, schema)) = &tables.trades {
        for batch in table.window(&conn, schema, "capture_ts_ns", (start, end))? {
            trades.extend(trades_of(&batch)?);
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

/// `book_samples` rows back into samples, one per instant.
fn samples_of(batch: &RecordBatch) -> Result<Vec<Arc<Sample>>, StoreError> {
    let times = int64(batch, "sample_ts_ns")?;
    let seqs = int64(batch, "seq")?;
    let ids = int64(batch, "last_update_id")?;
    let symbols = batch
        .column_by_name("symbol")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| missing("symbol"))?;
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
        let t = times.value(row);
        match out.last_mut() {
            Some(sample) if sample.sample_ts_ns == t => sample.rows.push(book),
            _ => out.push(Sample {
                sample_ts_ns: t,
                seq: seqs.value(row),
                rows: vec![book],
            }),
        }
    }
    Ok(out.into_iter().map(Arc::new).collect())
}

/// `agg_trades` rows back into trade messages, each with its capture time.
fn trades_of(batch: &RecordBatch) -> Result<Vec<(i64, Arc<TradeMessage>)>, StoreError> {
    let seqs = int64(batch, "seq")?;
    let traded = int64(batch, "trade_ts_ns")?;
    let captured = int64(batch, "capture_ts_ns")?;
    let ids = int64(batch, "agg_trade_id")?;
    let prices = int64(batch, "price")?;
    let qtys = int64(batch, "qty")?;
    let symbols = batch
        .column_by_name("symbol")
        .and_then(|column| column.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| missing("symbol"))?;
    let makers = batch
        .column_by_name("buyer_is_maker")
        .and_then(|column| column.as_any().downcast_ref::<BooleanArray>())
        .ok_or_else(|| missing("buyer_is_maker"))?;
    Ok((0..batch.num_rows())
        .map(|row| {
            let trade = TradeMessage {
                symbol: symbols.value(row).to_owned(),
                seq: u64::try_from(seqs.value(row)).unwrap_or_default(),
                ts_ms: u64::try_from(traded.value(row).div_euclid(1_000_000)).unwrap_or_default(),
                id: u64::try_from(ids.value(row)).unwrap_or_default(),
                px: prices.value(row),
                qty: qtys.value(row),
                buyer_is_maker: makers.value(row),
            };
            (captured.value(row), Arc::new(trade))
        })
        .collect())
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
    use super::*;

    #[test]
    fn a_step_longer_than_the_grid_is_instants_lost() {
        // After the rotated hours: the next sample is due one step after the last one sent.
        assert_eq!(missing_instants(1_000, i64::MIN, Some(1_100), 100), None);
        assert_eq!(missing_instants(1_000, i64::MIN, Some(1_400), 100), Some(3));
        // Nothing rotated: memory must start at the first instant of the requested start.
        assert_eq!(missing_instants(i64::MIN, 950, Some(1_000), 100), None);
        assert_eq!(missing_instants(i64::MIN, 1_000, Some(1_500), 100), Some(5));
        assert_eq!(
            missing_instants(i64::MIN, i64::MIN, Some(1_500), 100),
            None,
            "no start"
        );
        assert_eq!(
            missing_instants(1_000, i64::MIN, None, 100),
            None,
            "memory empty"
        );
        assert_eq!(missing_instants(1_000, i64::MIN, Some(1_050), 100), None);
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
            rows: Vec::new(),
        });
        let trade = |seq: u64| {
            Arc::new(TradeMessage {
                symbol: "BTCUSDT".to_owned(),
                seq,
                ts_ms: 0,
                id: seq,
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
