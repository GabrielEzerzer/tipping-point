//! Recorded runs through `DuckDB`, as Arrow IPC streams: any table of a market or Ising run
//! under the shared layout (`<data dir>/<domain>/<run_id>/<table>/<first_key:020>.parquet`),
//! rows in a range of the run's key in the order stored, and a market run's book samples
//! replayed as if live. Every batch is cast back to the schema stored in the Parquet footer, so
//! a read through the engine equals a direct `pyarrow.dataset` read of the same files.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use duckdb::params_from_iter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;

use super::ipc::{Encoder, Origin, Provenance, conform, tagged};
use super::replay::delay;
use super::stream::Chunks;
use crate::market::{Scales, Symbol};
use crate::store::run::{self, MANIFEST};
use crate::store::{StoreError, query};
use crate::venue::Venue;

/// The domains a data dir holds.
pub const DOMAINS: [&str; 2] = [run::MARKET, "ising"];

/// Why a table cannot be served.
#[derive(Debug, thiserror::Error)]
pub enum TableError {
    #[error("{0}")]
    BadName(String),
    #[error("{0}")]
    NotFound(String),
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// What the engine reads from any run's manifest.
#[derive(Debug, Deserialize)]
struct Manifest {
    key: String,
    tables: Vec<String>,
    #[serde(default)]
    params: Params,
}

#[derive(Debug, Default, Deserialize)]
struct Params {
    #[serde(default)]
    venue: Option<Venue>,
    #[serde(default)]
    scales: Option<std::collections::BTreeMap<Symbol, Scales>>,
}

/// A table of a run, checked to exist.
#[derive(Debug, Clone)]
pub struct Table {
    pub domain: String,
    pub run: String,
    pub name: String,
    pub run_dir: PathBuf,
    /// The run's key column: `seq` (market) or `step` (Ising).
    pub key: String,
    /// Its Parquet files in key order.
    pub files: Vec<PathBuf>,
    /// A market run's venue (Binance's, for runs recorded before runs said), `None` elsewhere.
    pub venue: Option<Venue>,
    pub scales: Option<std::collections::BTreeMap<Symbol, Scales>>,
}

/// A run id or table name: letters, digits, `-` and `_` only, so it can never leave the data dir.
fn safe(name: &str, what: &str) -> Result<(), TableError> {
    let ok = !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if ok {
        Ok(())
    } else {
        Err(TableError::BadName(format!(
            "{what} {name:?} is not a valid name"
        )))
    }
}

/// Finds a table: the domain is known, the run has a manifest that lists the table, and the
/// table has files.
///
/// # Errors
/// Returns [`TableError::BadName`] or [`TableError::NotFound`].
pub fn locate(data_dir: &Path, domain: &str, run: &str, name: &str) -> Result<Table, TableError> {
    if !DOMAINS.contains(&domain) {
        return Err(TableError::NotFound(format!("no domain {domain:?}")));
    }
    safe(run, "run")?;
    safe(name, "table")?;
    let run_dir = data_dir.join(domain).join(run);
    let manifest: Manifest = run::read_json(&run_dir.join(MANIFEST))
        .map_err(|_| TableError::NotFound(format!("no {domain} run {run:?}")))?;
    if !manifest.tables.iter().any(|table| table == name) {
        return Err(TableError::NotFound(format!(
            "run {run:?} has no table {name:?}"
        )));
    }
    let files = query::table_files(&run_dir, name)?;
    if files.is_empty() {
        return Err(TableError::NotFound(format!(
            "table {name:?} of run {run:?} has no Parquet yet"
        )));
    }
    let venue =
        (domain == run::MARKET).then(|| manifest.params.venue.unwrap_or(Venue::BinanceUsdm));
    Ok(Table {
        domain: domain.to_owned(),
        run: run.to_owned(),
        name: name.to_owned(),
        run_dir,
        key: manifest.key,
        files,
        venue,
        scales: manifest.params.scales,
    })
}

impl Table {
    /// The schema the table was stored with (its first file's footer), with the engine's
    /// metadata.
    ///
    /// # Errors
    /// Returns the I/O or Parquet error from reading the footer.
    pub fn schema(&self) -> Result<SchemaRef, StoreError> {
        let path = &self.files[0];
        let file = File::open(path).map_err(StoreError::io(path))?;
        let stored = ParquetRecordBatchReaderBuilder::try_new(file)?
            .schema()
            .clone();
        let origin = match self.venue {
            Some(venue) => Origin::Market {
                runs: BTreeMap::from([(venue, self.run.as_str())]),
                scales: self.scales.iter().map(|scales| (venue, scales)).collect(),
            },
            None => Origin::Run(&self.run),
        };
        Ok(tagged(
            &stored,
            &Provenance {
                table: &self.name,
                domain: &self.domain,
                origin,
            },
        ))
    }

    /// The query for rows whose key is in `[start, end)`, the stored columns in stored order.
    fn sql(&self, schema: &SchemaRef, start: Option<i64>, end: Option<i64>) -> String {
        self.select(schema, &self.key, start.is_some(), end.is_some())
    }

    /// The stored columns in stored order, where `column` is at least and below the bounds
    /// given as parameters. No ORDER BY: files go in key order and `DuckDB` keeps insertion
    /// order, which is the stored order, so a whole range is never sorted.
    fn select(&self, schema: &SchemaRef, column: &str, from: bool, to: bool) -> String {
        let columns: Vec<String> = schema
            .fields()
            .iter()
            .map(|field| identifier(field.name()))
            .collect();
        let column = identifier(column);
        let mut conditions = Vec::new();
        if from {
            conditions.push(format!("{column} >= ?"));
        }
        if to {
            conditions.push(format!("{column} < ?"));
        }
        let filter = if conditions.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", conditions.join(" AND "))
        };
        format!(
            "SELECT {} FROM {}{filter}",
            columns.join(", "),
            self.source()
        )
    }

    fn source(&self) -> String {
        let files: Vec<String> = self.files.iter().map(|path| literal(path)).collect();
        format!("read_parquet([{}], union_by_name = true)", files.join(", "))
    }

    /// The rows whose `column` is in `[from, to)`, in stored order and `schema`.
    ///
    /// # Errors
    /// Returns the `DuckDB` or Arrow error.
    pub fn window(
        &self,
        conn: &duckdb::Connection,
        schema: &SchemaRef,
        column: &str,
        (from, to): (i64, i64),
    ) -> Result<Vec<RecordBatch>, StoreError> {
        let mut statement = conn.prepare(&self.select(schema, column, true, true))?;
        drop(statement.stream_arrow([from, to])?);
        let mut batches = Vec::new();
        while let Some(chunk) = statement.step()? {
            batches.push(conform(&RecordBatch::from(&chunk), schema)?);
        }
        Ok(batches)
    }

    /// The smallest `column` at or after `from`, if any row has one.
    ///
    /// # Errors
    /// Returns the `DuckDB` error.
    pub fn next_at(
        &self,
        conn: &duckdb::Connection,
        column: &str,
        from: i64,
    ) -> Result<Option<i64>, StoreError> {
        let column = identifier(column);
        let sql = format!(
            "SELECT min({column}) FROM {} WHERE {column} >= ?",
            self.source()
        );
        Ok(conn.query_row(&sql, [from], |row| row.get(0))?)
    }

    /// Streams the rows whose key is in `[start, end)` into `chunks`, in `schema` (from
    /// [`Self::schema`]): the schema message, one batch per `DuckDB` chunk, then the end marker.
    /// Runs on a blocking thread. An error, or a panic, aborts the response.
    ///
    /// # Errors
    /// Returns the error that stopped the stream; the response has been aborted by then.
    pub fn stream(
        &self,
        schema: &SchemaRef,
        (start, end): (Option<i64>, Option<i64>),
        mut chunks: Chunks,
        runtime: &tokio::runtime::Handle,
    ) -> Result<(), StoreError> {
        let streamed = self.stream_inner(schema, start, end, &mut chunks, runtime);
        finish(streamed, chunks, runtime)
    }

    /// The batches, then the end-of-stream chunk to finish with; `None` when the client left or
    /// was cut.
    fn stream_inner(
        &self,
        schema: &SchemaRef,
        start: Option<i64>,
        end: Option<i64>,
        chunks: &mut Chunks,
        runtime: &tokio::runtime::Handle,
    ) -> Result<Option<Vec<u8>>, StoreError> {
        let (mut encoder, header) = Encoder::new(Arc::clone(schema))?;
        if chunks.send_blocking(runtime, header).is_err() {
            return Ok(None);
        }
        let conn = query::connect()?;
        conn.execute_batch("SET preserve_insertion_order = true")?;
        let mut statement = conn.prepare(&self.sql(schema, start, end))?;
        drop(statement.stream_arrow(params_from_iter(start.into_iter().chain(end)))?);
        while let Some(chunk) = statement.step()? {
            let batch = conform(&RecordBatch::from(&chunk), schema)?;
            if batch.num_rows() == 0 {
                continue;
            }
            if chunks
                .send_blocking(runtime, encoder.batch(&batch)?)
                .is_err()
            {
                return Ok(None);
            }
        }
        Ok(Some(encoder.finish()?))
    }

    /// A market run's book samples as if live, paced at `speed` times the capture clock. Every
    /// batch holds whole instants: one per batch when paced, as many as a `DuckDB` chunk holds
    /// at speed 0 (no pacing).
    ///
    /// # Errors
    /// Returns the error that stopped the stream; the response has been aborted by then.
    pub fn replay_samples(
        &self,
        schema: &SchemaRef,
        speed: f64,
        mut chunks: Chunks,
        runtime: &tokio::runtime::Handle,
    ) -> Result<(), StoreError> {
        let streamed = self.replay_inner(schema, speed, &mut chunks, runtime);
        finish(streamed, chunks, runtime)
    }

    fn replay_inner(
        &self,
        schema: &SchemaRef,
        speed: f64,
        chunks: &mut Chunks,
        runtime: &tokio::runtime::Handle,
    ) -> Result<Option<Vec<u8>>, StoreError> {
        let (mut encoder, header) = Encoder::new(Arc::clone(schema))?;
        if chunks.send_blocking(runtime, header).is_err() {
            return Ok(None);
        }
        let conn = query::connect()?;
        conn.execute_batch("SET preserve_insertion_order = true")?;
        let mut statement = conn.prepare(&self.sql(schema, None, None))?;
        drop(statement.stream_arrow([])?);
        let mut out = Instants {
            speed,
            clock: None,
            encoder: &mut encoder,
            chunks,
            runtime,
        };
        // An instant's rows can straddle two `DuckDB` chunks (with three symbols, 2,048 rows end
        // mid-instant): the last instant of each chunk waits for the next one.
        let mut carried: Option<RecordBatch> = None;
        while let Some(chunk) = statement.step()? {
            let mut batch = conform(&RecordBatch::from(&chunk), schema)?;
            if let Some(head) = carried.take() {
                batch = concat_batches(schema, [&head, &batch])?;
            }
            let last = last_instant_start(&batch)?;
            carried = Some(batch.slice(last, batch.num_rows() - last));
            if !out.send(&batch.slice(0, last))? {
                return Ok(None);
            }
        }
        if let Some(rest) = carried
            && !out.send(&rest)?
        {
            return Ok(None);
        }
        Ok(Some(encoder.finish()?))
    }
}

/// Sends whole instants of book samples, paced or not.
struct Instants<'a> {
    speed: f64,
    clock: Option<(Instant, i64)>,
    encoder: &'a mut Encoder,
    chunks: &'a mut Chunks,
    runtime: &'a tokio::runtime::Handle,
}

impl Instants<'_> {
    /// Sends `batch`, whole instants only: one batch per instant when paced, all at once when
    /// not. False once the client left or was cut.
    fn send(&mut self, batch: &RecordBatch) -> Result<bool, StoreError> {
        if batch.num_rows() == 0 {
            return Ok(true);
        }
        let pieces = if self.speed > 0.0 {
            instants(batch)?
        } else {
            vec![batch.clone()]
        };
        for piece in pieces {
            if self.speed > 0.0 {
                pace(&mut self.clock, first_time(&piece)?, self.speed);
            }
            let message = self.encoder.batch(&piece)?;
            if self.chunks.send_blocking(self.runtime, message).is_err() {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Where the last instant of a batch of book samples starts: its rows run to the end.
fn last_instant_start(batch: &RecordBatch) -> Result<usize, StoreError> {
    let times = sample_times(batch)?;
    let mut start = times.len();
    while start > 0 && times.value(start - 1) == times.value(times.len() - 1) {
        start -= 1;
    }
    Ok(start)
}

/// Ends a stream as it went: finished with its end marker when it ran to the end, aborted on an
/// error. A client that left or was cut needs nothing more: dropping `chunks` aborts.
fn finish(
    streamed: Result<Option<Vec<u8>>, StoreError>,
    mut chunks: Chunks,
    runtime: &tokio::runtime::Handle,
) -> Result<(), StoreError> {
    match streamed {
        Ok(Some(end)) => {
            let _ = chunks.finish_blocking(runtime, end);
            Ok(())
        }
        Ok(None) => Ok(()),
        Err(error) => {
            chunks.abort(&error.to_string());
            Err(error)
        }
    }
}

/// A batch of book samples split at every change of `sample_ts_ns`: one piece per instant.
fn instants(batch: &RecordBatch) -> Result<Vec<RecordBatch>, StoreError> {
    let times = sample_times(batch)?;
    let mut pieces = Vec::new();
    let mut start = 0;
    for i in 1..=times.len() {
        if i == times.len() || times.value(i) != times.value(start) {
            pieces.push(batch.slice(start, i - start));
            start = i;
        }
    }
    Ok(pieces)
}

fn sample_times(batch: &RecordBatch) -> Result<&Int64Array, StoreError> {
    batch
        .column_by_name("sample_ts_ns")
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| {
            StoreError::Arrow(arrow::error::ArrowError::SchemaError(
                "book_samples has no Int64 sample_ts_ns".to_owned(),
            ))
        })
}

fn first_time(batch: &RecordBatch) -> Result<i64, StoreError> {
    let times = sample_times(batch)?;
    Ok(if times.is_empty() { 0 } else { times.value(0) })
}

/// Sleeps until `t` on the capture clock comes round at `speed` times real time, the clock
/// starting at the first instant.
fn pace(clock: &mut Option<(Instant, i64)>, t: i64, speed: f64) {
    let (wall, t0) = *clock.get_or_insert((Instant::now(), t));
    if let Some(wait) = (wall + delay(t - t0, speed)).checked_duration_since(Instant::now()) {
        std::thread::sleep(wait);
    }
}

/// A column name as a `DuckDB` identifier.
fn identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// A path as a `DuckDB` string literal.
fn literal(path: &Path) -> String {
    let text = path
        .to_string_lossy()
        .replace('\\', "/")
        .replace('\'', "''");
    format!("'{text}'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_that_could_leave_the_data_dir_are_refused() {
        for bad in ["", "..", "a/b", "a\\b", "a b", "a.b", "a=b"] {
            assert!(safe(bad, "run").is_err(), "{bad:?}");
        }
        assert!(safe("20261021T140512Z-btceth", "run").is_ok());
        assert!(matches!(
            locate(Path::new("."), "elsewhere", "r", "t"),
            Err(TableError::NotFound(_))
        ));
    }

    #[test]
    fn identifiers_and_literals_are_quoted() {
        assert_eq!(identifier("a\"b"), "\"a\"\"b\"");
        assert_eq!(literal(Path::new(r"C:\x\it's")), "'C:/x/it''s'");
    }
}
