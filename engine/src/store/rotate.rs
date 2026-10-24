//! The offline rotate step: every closed log segment becomes one Parquet file per table, named
//! after the segment's first seq, so each file traces back to exactly one segment. What it does
//! with each frame ([`Ingest`]) is also what an archive import does with each archive record, so
//! an archive's books and samples come from the same code as a log's.
//!
//! It is safe to re-run. Outputs that exist are skipped, temp files an interrupted run left are
//! deleted first, and a file only appears by renaming a synced temp file to a new name. One
//! rotate at a time: a second rotate of a run while one is running is refused, because both
//! would write the same temp files. Only this step writes Parquet: a live writer that crashed
//! would leave a file without its footer.
//!
//! | table | one row per |
//! |---|---|
//! | `frames` | log frame, a lossless copy |
//! | `depth_updates` | price level in a depth update |
//! | `trades` | trade |
//! | `snapshots` | price level in a snapshot (Binance's REST, Bybit's in-band) |
//! | `book_samples` | symbol and instant on a grid of the capture clock |
//!
//! Every row but a frame's says its `venue`: one run holds one venue, but a symbol names a book
//! only together with its venue. Times are Int64 nanoseconds since the epoch, update ids are
//! Int64 (each venue's continuity id is `last_update_id`; the ids only some venues send are
//! nullable), trade ids are strings (Bybit's are UUIDs), and prices and quantities are integer
//! units at the run's scales (kept in its manifest and every Meta). Rows go in seq order. Book
//! samples are made by [`super::samples`], the code the live API samples with too.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayBuilder, ArrayRef, BooleanBuilder, Int8Builder, Int32Builder, Int64Builder, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use super::frame::{Frame, Kind, Meta};
use super::replay::Replay;
use super::run::{self, BookSample, MANIFEST, Manifest};
use super::samples::{Sample, SampleRows, Sampler};
use super::segment::{self, Ending, LOG_DIR, SegmentReader};
use super::{StoreError, ms_to_ns};
use crate::market::live::LiveEvent;
use crate::market::pipeline::{Replayed, SnapshotPayload};
use crate::market::sync::{DepthUpdate, Level, Snapshot};
use crate::market::{Scales, Symbol};
use crate::venue::{Depth, Parsed, RawDepth, RawTrade, Venue};

/// Every table rotate writes, in the order the manifest lists them.
pub const TABLES: [&str; 5] = [
    "frames",
    "depth_updates",
    "trades",
    "snapshots",
    "book_samples",
];

/// The file in the run folder a rotate holds a lock on for as long as it runs.
pub const LOCK_FILE: &str = "rotate.lock";

/// Rows per Parquet row group: several per hourly file, and one group is what the writer holds
/// in memory.
const ROW_GROUP: usize = 256 * 1024;
/// Rows gathered in Arrow builders before they go to the writer.
const BATCH: usize = 16 * 1024;
/// Text gathered before a batch goes out whatever its row count, so string offsets stay small.
const BATCH_BYTES: usize = 64 << 20;

/// What a rotate did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Rotation {
    /// Closed segments read.
    pub segments: u64,
    /// Files written this time.
    pub written: Vec<PathBuf>,
    /// Files that were already there and were left alone.
    pub skipped: u64,
    /// Temp files an interrupted rotate left, deleted.
    pub cleaned: u64,
    /// Depth, trade and snapshot frames that gave no rows because they did not parse or convert
    /// (the `frames` table still holds their text).
    pub unparsed: u64,
    /// Grid instants left unsampled because the capture clock jumped further than a day.
    pub skipped_samples: u64,
    /// The newest segment, not rotated because it has no end marker yet.
    pub open: Option<PathBuf>,
}

/// Rotates every closed segment of a run into Parquet. Reads all closed segments in order
/// through one pipeline, because a book sample depends on everything before it; writes only
/// the files that are missing.
///
/// # Errors
/// Returns [`StoreError::InUse`] at once, touching nothing, while another rotate holds the run;
/// otherwise an error for a missing manifest, a damaged or unclosed segment that is not the
/// newest, a seq that skips, or a failed write.
pub fn rotate(run_dir: &Path) -> Result<Rotation, StoreError> {
    let lock_path = run_dir.join(LOCK_FILE);
    let _lock = super::try_lock(&lock_path)
        .map_err(StoreError::io(&lock_path))?
        .ok_or_else(|| StoreError::InUse(run_dir.to_owned()))?;
    let manifest_path = run_dir.join(MANIFEST);
    let manifest: Manifest =
        run::read_json(&manifest_path).map_err(StoreError::io(&manifest_path))?;
    let mut rotation = Rotation {
        cleaned: clean_temp_files(run_dir)?,
        ..Rotation::default()
    };
    let segments = segment::list(&run_dir.join(LOG_DIR))?;
    let params = manifest.params;
    let mut ingest = Ingest::new(params.venue, params.scales, params.book_sample);
    for (index, (first_seq, path)) in segments.iter().enumerate() {
        let mut tables = Tables::open(
            run_dir,
            *first_seq,
            ingest.levels(),
            &TABLES,
            &mut rotation.skipped,
        )?;
        let mut reader = SegmentReader::open(path)?;
        while let Some(frame) = reader.next_frame()? {
            ingest.frame(&frame, &mut tables, &mut |_| {})?;
        }
        if let Some(Ending::Closed(_)) = reader.ending() {
            rotation.segments += 1;
            tables.finish(&mut rotation.written)?;
        } else {
            tables.abandon();
            if index + 1 < segments.len() {
                return Err(StoreError::Unclosed(path.clone()));
            }
            rotation.open = Some(path.clone());
        }
    }
    rotation.unparsed = ingest.unparsed;
    rotation.skipped_samples = ingest.sampler.skipped();
    Ok(rotation)
}

/// What rotate and an archive import do with every frame, in seq order from the run's Meta at
/// seq 0: its table rows, the book samples its capture time passes (taken before it is applied:
/// each instant it passes is the book without it), then the frame itself through the pipeline
/// a replay runs, which is the one the live loop runs.
pub(super) struct Ingest {
    replay: Replay,
    sampler: Sampler,
    venue: Venue,
    scales: BTreeMap<Symbol, Scales>,
    /// Depth, trade and snapshot frames that gave no rows because they did not parse or convert.
    pub(super) unparsed: u64,
    /// Book-sample rows written while their book was not synced.
    pub(super) unsynced: u64,
}

impl Ingest {
    /// Before the first frame: the venue and scales to read table rows with until a Meta says
    /// otherwise, and the book sampling.
    pub(super) fn new(venue: Venue, scales: BTreeMap<Symbol, Scales>, sample: BookSample) -> Self {
        Self {
            replay: Replay::new(None),
            sampler: Sampler::new(sample),
            venue,
            scales,
            unparsed: 0,
            unsynced: 0,
        }
    }

    /// Levels a book sample keeps per side.
    pub(super) fn levels(&self) -> usize {
        self.sampler.levels()
    }

    /// Takes the next frame.
    ///
    /// # Errors
    /// As [`Replay::push`], or when a row does not fit its table.
    pub(super) fn frame(
        &mut self,
        frame: &Frame,
        tables: &mut Tables,
        on_event: &mut impl FnMut(LiveEvent<'_>),
    ) -> Result<(), StoreError> {
        if frame.kind == Kind::Meta
            && let Ok(meta) = serde_json::from_str::<Meta>(&frame.payload)
        {
            self.venue = meta.venue;
            self.scales = meta.scales;
        }
        if tables.frames.enabled {
            tables.frames.rows.push(frame)?;
        }
        if !tables.push_rows(frame, self.venue, &self.scales)? {
            self.unparsed += 1;
        }
        if is_input(frame.kind) {
            let instants = self.sampler.instants(frame.capture_ts_ns);
            if let Some(pipeline) = self.replay.pipeline() {
                let seq = to_i64(frame.seq)? - 1;
                for t in instants {
                    let sample = Sample::of_feed(pipeline.feed(), t, seq, self.sampler.levels())?;
                    if tables.samples.enabled {
                        tables.samples.rows.push(&sample);
                        let unsynced = sample.rows.iter().filter(|row| !row.synced).count();
                        self.unsynced += unsynced as u64;
                    }
                }
            }
        }
        self.replay.push(frame, on_event)?;
        tables.flush_full()
    }

    /// The pipeline in its final state with what it made of the inputs, and the grid instants
    /// left unsampled across clock jumps of over a day.
    ///
    /// # Errors
    /// Returns [`StoreError::NoFrames`] when no frame was ever taken.
    pub(super) fn finish(self) -> Result<(Replayed, u64), StoreError> {
        let skipped = self.sampler.skipped();
        Ok((self.replay.finish()?, skipped))
    }
}

/// Frames the feed takes; Meta and `SegmentEnd` are bookkeeping.
fn is_input(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::Depth | Kind::Trade | Kind::Snapshot | Kind::Control
    )
}

/// Deletes `*.parquet.tmp` files a crashed rotate or import left in the table folders.
pub(super) fn clean_temp_files(run_dir: &Path) -> Result<u64, StoreError> {
    let mut cleaned = 0;
    for table in TABLES {
        let dir = run_dir.join(table);
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(StoreError::io(&dir)(error)),
        };
        for entry in entries {
            let path = entry.map_err(StoreError::io(&dir))?.path();
            let stem_is_parquet = path
                .file_stem()
                .map(Path::new)
                .and_then(Path::extension)
                .is_some_and(|ext| ext == "parquet");
            if stem_is_parquet && path.extension().is_some_and(|ext| ext == "tmp") {
                fs::remove_file(&path).map_err(StoreError::io(&path))?;
                cleaned += 1;
            }
        }
    }
    Ok(cleaned)
}

fn to_i64(value: u64) -> Result<i64, StoreError> {
    i64::try_from(value).map_err(|_| StoreError::TooBig(value))
}

fn int64(name: &str) -> Field {
    Field::new(name, DataType::Int64, false)
}

fn column(array: impl arrow::array::Array + 'static) -> ArrayRef {
    Arc::new(array)
}

/// One table's rows being gathered.
trait Rows {
    fn len(&self) -> usize;
    /// Text bytes gathered, for tables with long strings.
    fn bytes(&self) -> usize {
        0
    }
    /// Takes the rows gathered so far as a batch, leaving the builders empty.
    fn take(&mut self) -> Result<RecordBatch, StoreError>;
}

/// A table for one segment: whether the run has it at all, its rows, where they go (`None` when
/// its file already exists, or the run has no such table), and how many went.
struct Table<R> {
    enabled: bool,
    rows: R,
    output: Option<Output>,
    written: u64,
}

impl<R: Rows> Table<R> {
    fn new(enabled: bool, rows: R, output: Option<Output>) -> Self {
        Self {
            enabled,
            rows,
            output,
            written: 0,
        }
    }

    fn flush(&mut self) -> Result<(), StoreError> {
        if self.rows.len() == 0 {
            return Ok(());
        }
        let batch = self.rows.take()?;
        if let Some(output) = &mut self.output {
            output.writer.write(&batch)?;
            self.written += batch.num_rows() as u64;
        }
        Ok(())
    }

    fn flush_full(&mut self) -> Result<(), StoreError> {
        if self.rows.len() >= BATCH || self.rows.bytes() >= BATCH_BYTES {
            self.flush()?;
        }
        Ok(())
    }

    /// Writes what is left and makes the file appear; the rows it holds, when it was written.
    fn finish(mut self, written: &mut Vec<PathBuf>) -> Result<Option<u64>, StoreError> {
        self.flush()?;
        match self.output {
            Some(output) => {
                written.push(output.finish()?);
                Ok(Some(self.written))
            }
            None => Ok(None),
        }
    }
}

/// A Parquet file being written to its temp name.
struct Output {
    tmp: PathBuf,
    path: PathBuf,
    writer: ArrowWriter<File>,
}

impl Output {
    /// Starts `path` unless it exists.
    fn create(
        path: PathBuf,
        schema: SchemaRef,
        skipped: &mut u64,
    ) -> Result<Option<Self>, StoreError> {
        if path.exists() {
            *skipped += 1;
            return Ok(None);
        }
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(StoreError::io(dir))?;
        }
        let tmp = run::tmp_path(&path);
        let file = File::create(&tmp).map_err(StoreError::io(&tmp))?;
        let properties = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .set_max_row_group_row_count(Some(ROW_GROUP))
            .build();
        let writer = ArrowWriter::try_new(file, schema, Some(properties))?;
        Ok(Some(Self { tmp, path, writer }))
    }

    /// Footer, sync, then the rename that makes the file appear.
    fn finish(self) -> Result<PathBuf, StoreError> {
        let file = self.writer.into_inner()?;
        file.sync_all().map_err(StoreError::io(&self.tmp))?;
        drop(file);
        run::rename_new(&self.tmp, &self.path).map_err(StoreError::io(&self.path))?;
        Ok(self.path)
    }

    fn abandon(self) {
        drop(self.writer);
        let _ = fs::remove_file(&self.tmp);
    }
}

/// The tables of one segment (one archive, in an import): the five a log rotates into, only
/// those of `enabled` written.
pub(super) struct Tables {
    frames: Table<FrameRows>,
    depth: Table<DepthRows>,
    trades: Table<TradeRows>,
    snapshots: Table<SnapshotRows>,
    samples: Table<SampleRows>,
}

impl Tables {
    /// Starts each enabled table's file for the segment starting at `first_seq`, unless it
    /// exists (counted in `skipped`).
    pub(super) fn open(
        run_dir: &Path,
        first_seq: u64,
        levels: usize,
        enabled: &[&str],
        skipped: &mut u64,
    ) -> Result<Self, StoreError> {
        let name = format!("{first_seq:020}.parquet");
        let mut table =
            |table: &str, schema: SchemaRef| -> Result<(bool, Option<Output>), StoreError> {
                if !enabled.contains(&table) {
                    return Ok((false, None));
                }
                let output = Output::create(run_dir.join(table).join(&name), schema, skipped)?;
                Ok((true, output))
            };
        let (on, frames) = table(TABLES[0], FrameRows::schema())?;
        let frames = Table::new(on, FrameRows::default(), frames);
        let (on, depth) = table(TABLES[1], DepthRows::schema())?;
        let depth = Table::new(on, DepthRows::default(), depth);
        let (on, trades) = table(TABLES[2], TradeRows::schema())?;
        let trades = Table::new(on, TradeRows::default(), trades);
        let (on, snapshots) = table(TABLES[3], SnapshotRows::schema())?;
        let snapshots = Table::new(on, SnapshotRows::default(), snapshots);
        let (on, samples) = table(TABLES[4], SampleRows::schema(levels))?;
        let samples = Table::new(on, SampleRows::new(levels), samples);
        Ok(Self {
            frames,
            depth,
            trades,
            snapshots,
            samples,
        })
    }

    /// The table rows a frame's content gives, read by `venue`'s rules. False when a depth,
    /// trade or snapshot frame did not parse or convert. Content decides the table, as it decides
    /// how the feed routes text.
    fn push_rows(
        &mut self,
        frame: &Frame,
        venue: Venue,
        scales: &BTreeMap<Symbol, Scales>,
    ) -> Result<bool, StoreError> {
        match frame.kind {
            Kind::Depth | Kind::Trade => {
                if !(self.depth.enabled || self.trades.enabled || self.snapshots.enabled) {
                    return Ok(true);
                }
                match venue.parse(&frame.payload) {
                    Ok(Parsed::Depth(raw)) => {
                        let Some(depth) = scales
                            .get(raw.symbol())
                            .and_then(|scales| raw.convert(*scales).ok())
                        else {
                            return Ok(false);
                        };
                        match depth {
                            Depth::Update(update) if self.depth.enabled => {
                                self.depth.rows.push(frame, venue, &raw, &update)
                            }
                            Depth::Book(snapshot) if self.snapshots.enabled => self
                                .snapshots
                                .rows
                                .push(frame, venue, raw.symbol(), &snapshot),
                            Depth::Update(_) | Depth::Book(_) => Ok(true),
                        }
                    }
                    Ok(Parsed::Trade(raw)) if self.trades.enabled => {
                        self.trades.rows.push(frame, venue, &raw, scales)
                    }
                    Ok(Parsed::Trade(_)) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
            Kind::Snapshot if self.snapshots.enabled => {
                match serde_json::from_str::<SnapshotPayload>(&frame.payload) {
                    Ok(snapshot) => self
                        .snapshots
                        .rows
                        .push_rest(frame, venue, &snapshot, scales),
                    Err(_) => Ok(false),
                }
            }
            Kind::Snapshot | Kind::Meta | Kind::Control | Kind::SegmentEnd => Ok(true),
        }
    }

    fn flush_full(&mut self) -> Result<(), StoreError> {
        self.frames.flush_full()?;
        self.depth.flush_full()?;
        self.trades.flush_full()?;
        self.snapshots.flush_full()?;
        self.samples.flush_full()
    }

    /// Writes every enabled table's file that was missing. Returns the rows each file written
    /// holds, by table.
    pub(super) fn finish(
        self,
        written: &mut Vec<PathBuf>,
    ) -> Result<BTreeMap<String, u64>, StoreError> {
        let rows = [
            (TABLES[0], self.frames.finish(written)?),
            (TABLES[1], self.depth.finish(written)?),
            (TABLES[2], self.trades.finish(written)?),
            (TABLES[3], self.snapshots.finish(written)?),
            (TABLES[4], self.samples.finish(written)?),
        ];
        Ok(rows
            .into_iter()
            .filter_map(|(table, rows)| Some((table.to_owned(), rows?)))
            .collect())
    }

    /// Drops every file being written, leaving nothing behind.
    pub(super) fn abandon(self) {
        let outputs = [
            self.frames.output,
            self.depth.output,
            self.trades.output,
            self.snapshots.output,
            self.samples.output,
        ];
        for output in outputs.into_iter().flatten() {
            output.abandon();
        }
    }
}

/// `frames`: seq, kind, `source_ts_ns`, `capture_ts_ns`, payload.
#[derive(Default)]
struct FrameRows {
    seq: Int64Builder,
    kind: Int8Builder,
    source_ts_ns: Int64Builder,
    capture_ts_ns: Int64Builder,
    payload: StringBuilder,
    bytes: usize,
}

impl FrameRows {
    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            int64("seq"),
            Field::new("kind", DataType::Int8, false),
            int64("source_ts_ns"),
            int64("capture_ts_ns"),
            Field::new("payload", DataType::Utf8, false),
        ]))
    }

    fn push(&mut self, frame: &Frame) -> Result<(), StoreError> {
        self.seq.append_value(to_i64(frame.seq)?);
        self.kind.append_value(frame.kind.code().cast_signed());
        self.source_ts_ns.append_value(frame.source_ts_ns);
        self.capture_ts_ns.append_value(frame.capture_ts_ns);
        self.payload.append_value(&frame.payload);
        self.bytes += frame.payload.len();
        Ok(())
    }
}

impl Rows for FrameRows {
    fn len(&self) -> usize {
        self.seq.len()
    }

    fn bytes(&self) -> usize {
        self.bytes
    }

    fn take(&mut self) -> Result<RecordBatch, StoreError> {
        self.bytes = 0;
        let columns = vec![
            column(self.seq.finish()),
            column(self.kind.finish()),
            column(self.source_ts_ns.finish()),
            column(self.capture_ts_ns.finish()),
            column(self.payload.finish()),
        ];
        Ok(RecordBatch::try_new(Self::schema(), columns)?)
    }
}

/// Price levels as rows: `level_idx` is the level's place in its side's list, side +1 for a bid
/// and -1 for an ask.
#[derive(Default)]
struct LevelColumns {
    level_idx: Int32Builder,
    side: Int8Builder,
    price: Int64Builder,
    qty: Int64Builder,
}

impl LevelColumns {
    fn fields() -> [Field; 4] {
        [
            Field::new("level_idx", DataType::Int32, false),
            Field::new("side", DataType::Int8, false),
            int64("price"),
            int64("qty"),
        ]
    }

    /// Appends one level and calls `row` to fill the other columns of the same row.
    fn push(&mut self, bids: &[Level], asks: &[Level], mut row: impl FnMut()) {
        for (side, levels) in [(1, bids), (-1, asks)] {
            for (idx, level) in levels.iter().enumerate() {
                self.level_idx
                    .append_value(i32::try_from(idx).unwrap_or(i32::MAX));
                self.side.append_value(side);
                self.price.append_value(level.price);
                self.qty.append_value(level.qty);
                row();
            }
        }
    }

    fn take(&mut self) -> [ArrayRef; 4] {
        [
            column(self.level_idx.finish()),
            column(self.side.finish()),
            column(self.price.finish()),
            column(self.qty.finish()),
        ]
    }
}

/// `depth_updates`: one row per level of a depth update (qty 0 removes the level).
#[derive(Default)]
struct DepthRows {
    seq: Int64Builder,
    source_ts_ns: Int64Builder,
    transaction_ts_ns: Int64Builder,
    capture_ts_ns: Int64Builder,
    venue: StringBuilder,
    symbol: StringBuilder,
    first_update_id: Int64Builder,
    last_update_id: Int64Builder,
    prev_last_update_id: Int64Builder,
    levels: LevelColumns,
}

impl DepthRows {
    fn schema() -> SchemaRef {
        let mut fields = vec![
            int64("seq"),
            int64("source_ts_ns"),
            Field::new("transaction_ts_ns", DataType::Int64, true),
            int64("capture_ts_ns"),
            Field::new("venue", DataType::Utf8, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("first_update_id", DataType::Int64, true),
            int64("last_update_id"),
            Field::new("prev_last_update_id", DataType::Int64, true),
        ];
        fields.extend(LevelColumns::fields());
        Arc::new(Schema::new(fields))
    }

    fn push(
        &mut self,
        frame: &Frame,
        venue: Venue,
        raw: &RawDepth,
        update: &DepthUpdate,
    ) -> Result<bool, StoreError> {
        let ids = raw.ids();
        let id = |value: Option<u64>| value.map(i64::try_from).transpose();
        let (Ok(first), Ok(last), Ok(prev)) = (
            id(ids.first_update_id),
            i64::try_from(ids.last_update_id),
            id(ids.prev_last_update_id),
        ) else {
            return Ok(false);
        };
        let seq = to_i64(frame.seq)?;
        let (source, transaction) = (
            ms_to_ns(ids.event_time_ms),
            ids.transaction_time_ms.map(ms_to_ns),
        );
        let Self {
            levels,
            seq: seqs,
            source_ts_ns,
            transaction_ts_ns,
            capture_ts_ns,
            venue: venues,
            symbol,
            first_update_id,
            last_update_id,
            prev_last_update_id,
        } = self;
        levels.push(&update.bids, &update.asks, || {
            seqs.append_value(seq);
            source_ts_ns.append_value(source);
            transaction_ts_ns.append_option(transaction);
            capture_ts_ns.append_value(frame.capture_ts_ns);
            venues.append_value(venue.id());
            symbol.append_value(raw.symbol());
            first_update_id.append_option(first);
            last_update_id.append_value(last);
            prev_last_update_id.append_option(prev);
        });
        Ok(true)
    }
}

impl Rows for DepthRows {
    fn len(&self) -> usize {
        self.seq.len()
    }

    fn take(&mut self) -> Result<RecordBatch, StoreError> {
        let mut columns = vec![
            column(self.seq.finish()),
            column(self.source_ts_ns.finish()),
            column(self.transaction_ts_ns.finish()),
            column(self.capture_ts_ns.finish()),
            column(self.venue.finish()),
            column(self.symbol.finish()),
            column(self.first_update_id.finish()),
            column(self.last_update_id.finish()),
            column(self.prev_last_update_id.finish()),
        ];
        columns.extend(self.levels.take());
        Ok(RecordBatch::try_new(Self::schema(), columns)?)
    }
}

/// `trades`: one row per trade. `first_trade_id` and `last_trade_id` are the trades an
/// aggregated trade covers (Binance's); null for a venue that sends trades one by one.
#[derive(Default)]
struct TradeRows {
    seq: Int64Builder,
    source_ts_ns: Int64Builder,
    trade_ts_ns: Int64Builder,
    capture_ts_ns: Int64Builder,
    venue: StringBuilder,
    symbol: StringBuilder,
    trade_id: StringBuilder,
    first_trade_id: Int64Builder,
    last_trade_id: Int64Builder,
    price: Int64Builder,
    qty: Int64Builder,
    buyer_is_maker: BooleanBuilder,
}

impl TradeRows {
    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            int64("seq"),
            int64("source_ts_ns"),
            int64("trade_ts_ns"),
            int64("capture_ts_ns"),
            Field::new("venue", DataType::Utf8, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("trade_id", DataType::Utf8, false),
            Field::new("first_trade_id", DataType::Int64, true),
            Field::new("last_trade_id", DataType::Int64, true),
            int64("price"),
            int64("qty"),
            Field::new("buyer_is_maker", DataType::Boolean, false),
        ]))
    }

    fn push(
        &mut self,
        frame: &Frame,
        venue: Venue,
        raw: &RawTrade,
        scales: &BTreeMap<Symbol, Scales>,
    ) -> Result<bool, StoreError> {
        let Some(trade) = scales
            .get(raw.symbol())
            .and_then(|scales| raw.convert(*scales).ok())
        else {
            return Ok(false);
        };
        let ids = raw.ids();
        let id = |value: Option<u64>| value.map(i64::try_from).transpose();
        let (Ok(first), Ok(last)) = (id(ids.first_trade_id), id(ids.last_trade_id)) else {
            return Ok(false);
        };
        self.seq.append_value(to_i64(frame.seq)?);
        self.source_ts_ns.append_value(ms_to_ns(ids.event_time_ms));
        self.trade_ts_ns.append_value(ms_to_ns(trade.trade_time_ms));
        self.capture_ts_ns.append_value(frame.capture_ts_ns);
        self.venue.append_value(venue.id());
        self.symbol.append_value(&trade.symbol);
        self.trade_id.append_value(&trade.trade_id);
        self.first_trade_id.append_option(first);
        self.last_trade_id.append_option(last);
        self.price.append_value(trade.price);
        self.qty.append_value(trade.qty);
        self.buyer_is_maker.append_value(trade.buyer_is_maker);
        Ok(true)
    }
}

impl Rows for TradeRows {
    fn len(&self) -> usize {
        self.seq.len()
    }

    fn take(&mut self) -> Result<RecordBatch, StoreError> {
        let columns = vec![
            column(self.seq.finish()),
            column(self.source_ts_ns.finish()),
            column(self.trade_ts_ns.finish()),
            column(self.capture_ts_ns.finish()),
            column(self.venue.finish()),
            column(self.symbol.finish()),
            column(self.trade_id.finish()),
            column(self.first_trade_id.finish()),
            column(self.last_trade_id.finish()),
            column(self.price.finish()),
            column(self.qty.finish()),
            column(self.buyer_is_maker.finish()),
        ];
        Ok(RecordBatch::try_new(Self::schema(), columns)?)
    }
}

/// `snapshots`: one row per level of a snapshot: a REST body (Binance) or a whole book sent on
/// the stream (Bybit).
#[derive(Default)]
struct SnapshotRows {
    seq: Int64Builder,
    source_ts_ns: Int64Builder,
    capture_ts_ns: Int64Builder,
    venue: StringBuilder,
    symbol: StringBuilder,
    last_update_id: Int64Builder,
    levels: LevelColumns,
}

impl SnapshotRows {
    fn schema() -> SchemaRef {
        let mut fields = vec![
            int64("seq"),
            int64("source_ts_ns"),
            int64("capture_ts_ns"),
            Field::new("venue", DataType::Utf8, false),
            Field::new("symbol", DataType::Utf8, false),
            int64("last_update_id"),
        ];
        fields.extend(LevelColumns::fields());
        Arc::new(Schema::new(fields))
    }

    /// The rows of a REST snapshot frame.
    fn push_rest(
        &mut self,
        frame: &Frame,
        venue: Venue,
        payload: &SnapshotPayload,
        scales: &BTreeMap<Symbol, Scales>,
    ) -> Result<bool, StoreError> {
        let converted = scales.get(&payload.symbol).and_then(|scales| {
            let raw = venue.parse_snapshot(&payload.body).ok()?;
            raw.convert(*scales).ok()
        });
        let Some(snapshot) = converted else {
            return Ok(false);
        };
        self.push(frame, venue, &payload.symbol, &snapshot)
    }

    fn push(
        &mut self,
        frame: &Frame,
        venue: Venue,
        symbol_name: &str,
        snapshot: &Snapshot,
    ) -> Result<bool, StoreError> {
        let Ok(update_id) = i64::try_from(snapshot.last_update_id) else {
            return Ok(false);
        };
        let seq = to_i64(frame.seq)?;
        let Self {
            levels,
            seq: seqs,
            source_ts_ns,
            capture_ts_ns,
            venue: venues,
            symbol,
            last_update_id,
        } = self;
        levels.push(&snapshot.bids, &snapshot.asks, || {
            seqs.append_value(seq);
            source_ts_ns.append_value(frame.source_ts_ns);
            capture_ts_ns.append_value(frame.capture_ts_ns);
            venues.append_value(venue.id());
            symbol.append_value(symbol_name);
            last_update_id.append_value(update_id);
        });
        Ok(true)
    }
}

impl Rows for SnapshotRows {
    fn len(&self) -> usize {
        self.seq.len()
    }

    fn take(&mut self) -> Result<RecordBatch, StoreError> {
        let mut columns = vec![
            column(self.seq.finish()),
            column(self.source_ts_ns.finish()),
            column(self.capture_ts_ns.finish()),
            column(self.venue.finish()),
            column(self.symbol.finish()),
            column(self.last_update_id.finish()),
        ];
        columns.extend(self.levels.take());
        Ok(RecordBatch::try_new(Self::schema(), columns)?)
    }
}

/// `book_samples` rows go through the same table plumbing as every other table.
impl Rows for SampleRows {
    fn len(&self) -> usize {
        SampleRows::len(self)
    }

    fn take(&mut self) -> Result<RecordBatch, StoreError> {
        SampleRows::take(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_schema_starts_with_its_key() {
        for schema in [
            FrameRows::schema(),
            DepthRows::schema(),
            TradeRows::schema(),
            SnapshotRows::schema(),
        ] {
            assert_eq!(schema.field(0).name(), "seq");
        }
    }
}
