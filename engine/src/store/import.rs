//! Archive import: one downloaded archive day of one venue's symbol becomes a run. Its records go,
//! one frame each (see [`crate::venue::archive`]), through rotate's own [`Ingest`]: the same
//! pipeline the live loop runs and the same sampler, written to Parquet by rotate's table writers.
//! There is no binary log: the downloaded file is the lossless copy, and the manifest names it
//! with its URL, size and SHA-256 (checked against a `.CHECKSUM` beside it when there is one).
//!
//! An archive run is stamped with the exchange's clock (`clock: exchange` in its manifest): a
//! frame's capture time is its record's own exchange time, so book samples sit on the exchange
//! clock's grid and archive runs join live runs on exchange time, never on capture time.
//!
//! The run is `<data dir>/market/<yyyymmdd>T000000Z-<venue>-<symbol>/`: named, like every run, by
//! when its data starts, then which venue and symbol it holds. Two passes read the archive: the
//! first finds the scales the file needs and checks its integrity, the second makes the frames.
//! The zip's CRC is checked each time a pass reaches the end of the file.
//!
//! Seeding (#170). Bybit's archive is one file per UTC day, and a day's file opens with a
//! snapshot a second or so into the day (2023-06-15 BTCUSDT: at 00:00:01.451), so on its own an
//! import has no book for the day's first instants. Every order-book import therefore leaves the
//! day's closing book in its run, [`CLOSE`]: a snapshot message of the whole book as it stood
//! after the last record stamped before the day's end, then, verbatim, every record from the
//! first one stamped at or after the end. A file holds a second or so past its end, and ends
//! with the very snapshot the next file opens with (2023-06-14 BTCUSDT: deltas to u 49365365 at
//! 00:00:01.447, then the snapshot of u 49365365 at 01.451 that opens 2023-06-15). Given to the
//! next day's import as its seed, those lines go through the pipeline before the archive's, so
//! the instants from midnight to the day's own first record are sampled as one import of both
//! files would sample them. A seed is used only when it is this symbol's book, opens with a whole
//! book stamped before the day, and runs on to the day's first record:
//! - that record's `u` is the seed's last `u` (that same message again, or a snapshot of the same
//!   book), or the next one when the record is a delta, which the sync rule then applies to the
//!   seed's book. A snapshot takes no `u` of its own (its `u` is the last delta's, as above), so a
//!   snapshot of the next `u` took in an update made at a time neither file gives, and the seed's
//!   book may be stale at any instant before it;
//! - its book never goes over 2 s without a record, from the closing book through the day's first
//!   record ([`SEED_GAP_NS`]). The reducer masks a book that went 2 s without a record as stale
//!   (`STALE_SAMPLES`, book/features.py) but counts from midnight only, so a silence that began
//!   before midnight would leave instants counted that one import of both files masks.
//!
//! Otherwise the day is imported unseeded, and `complete.json` says why (`seed`), so no instant
//! is sampled from a book that missed an update, or counted from one gone stale.
//!
//! Safe to re-run: an import of a run that is complete changes nothing, a crashed one is finished
//! (its temp files deleted, the tables it had written kept), and an archive other than the one a
//! run was made from, or another set of tables, is refused. One import or rotate of a run at a
//! time. A run complete already is done whatever seed it is given again.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::frame::{Frame, Kind, Meta};
use super::rotate::{Ingest, LOCK_FILE, TABLES, Tables, clean_temp_files};
use super::run::{
    self, ArchiveFile, BookSample, COMPLETE, Clock, Complete, DAY_NS, DataSource, FORMAT,
    Integrity, KEY, MANIFEST, MARKET, Manifest, MarketParams, Producer, Seed,
};
use super::{StoreError, now_ns};
use crate::market::decimal::format_units;
use crate::market::feed::Feed;
use crate::market::live::LiveEvent;
use crate::market::sync::Level;
use crate::market::{Scales, Symbol};
use crate::venue::archive::{ArchiveFormat, ArchiveFrame, ArchiveName, Scan};

/// The day's closing book an order-book import leaves in its run, the next day's seed (module
/// notes): one message per line.
pub const CLOSE: &str = "close.jsonl";

/// The longest a seed's book may go without a record (module notes): the reducer's stale window,
/// `STALE_SAMPLES` instants of 100 ms in book/features.py, which moves with it.
const SEED_GAP_NS: i64 = 2_000_000_000;

/// How an import goes.
#[derive(Debug, Clone)]
pub struct ImportConfig {
    /// Where runs live: `<data dir>/market/<run_id>/`.
    pub data_dir: PathBuf,
    /// Where the archive was downloaded from, when not the venue's own address for it.
    pub url: Option<String>,
    /// The tables to write; the archive format's own when `None` (book samples for an order
    /// book, trades for trades).
    pub tables: Option<Vec<String>>,
    /// The day before's [`CLOSE`], to sample the day's book from midnight (module notes).
    pub seed: Option<PathBuf>,
}

/// What an import did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Imported {
    pub run_dir: PathBuf,
    /// Files written this time.
    pub written: Vec<PathBuf>,
    /// The run was complete already, so nothing was done.
    pub done_before: bool,
    /// Rows in each of the run's tables.
    pub rows: BTreeMap<String, u64>,
    pub integrity: Integrity,
    /// What became of the seed, when one was given.
    pub seed: Option<Seed>,
}

/// Why an archive could not be imported.
#[derive(Debug, Error)]
pub enum ImportError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(
        "{}: not an archive the engine imports (Bybit's <yyyy-mm-dd>_<SYMBOL>_ob<N>.data.zip or Binance's <SYMBOL>-aggTrades-<yyyy-mm-dd>.zip)",
        .0.display()
    )]
    NotAnArchive(PathBuf),
    #[error("{}: {why}", .path.display())]
    BadArchive { path: PathBuf, why: String },
    #[error("{}: its SHA-256 is {actual}, but the .CHECKSUM beside it says {expected}", .path.display())]
    Checksum {
        path: PathBuf,
        actual: String,
        expected: String,
    },
    #[error(
        "{}: imported from another archive ({theirs}); this one is {ours}. Remove the run to import it again.",
        .run_dir.display()
    )]
    OtherArchive {
        run_dir: PathBuf,
        theirs: String,
        ours: String,
    },
    #[error(
        "{}: imported with the tables {theirs:?}, not {ours:?}. Remove the run to import it with others.",
        .run_dir.display()
    )]
    OtherTables {
        run_dir: PathBuf,
        theirs: Vec<String>,
        ours: Vec<String>,
    },
    #[error("{0:?} is not a table an import writes (one of {TABLES:?})")]
    UnknownTable(String),
}

impl ImportError {
    fn bad(path: &Path, why: impl Into<String>) -> Self {
        Self::BadArchive {
            path: path.to_owned(),
            why: why.into(),
        }
    }
}

fn archive_name(archive: &Path) -> Result<ArchiveName, ImportError> {
    archive
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(ArchiveName::parse)
        .ok_or_else(|| ImportError::NotAnArchive(archive.to_owned()))
}

/// `<yyyymmdd>T000000Z-<venue>-<symbol>`.
fn run_id(name: &ArchiveName) -> String {
    let tag = format!("{}-{}", name.venue(), name.symbol.to_lowercase());
    run::run_id(name.day.start_ns(), &tag).unwrap_or(tag)
}

/// Imports one archive into its run, raising `on_event` for every input as a replay does.
///
/// # Errors
/// Returns why the archive was refused (see [`ImportError`]), or the I/O, Parquet or pipeline
/// error that stopped the import; a stopped import leaves no partial file, only what a re-run
/// finishes.
pub fn import(
    archive: &Path,
    config: &ImportConfig,
    on_event: &mut impl FnMut(LiveEvent<'_>),
) -> Result<Imported, ImportError> {
    let name = archive_name(archive)?;
    let tables = tables(&name, config.tables.as_deref())?;
    let (sha256, bytes) = sha256(archive)?;
    if let Some(expected) = checksum_beside(archive)?
        && expected != sha256
    {
        return Err(ImportError::Checksum {
            path: archive.to_owned(),
            actual: sha256,
            expected,
        });
    }
    let run_dir = config.data_dir.join(MARKET).join(run_id(&name));
    // A run already done needs nothing. Its files only ever appear, so reading them needs no lock.
    if let Some(done) = done_before(&run_dir, &sha256, &tables)? {
        return Ok(done);
    }
    // The whole archive is read before anything is written: a broken one leaves no run behind.
    let scan = prescan(archive, &name)?;
    let (seed_lines, seed, found) = seeded(config.seed.as_deref(), &name, &scan);
    let scales = BTreeMap::from([(name.symbol.clone(), found)]);

    fs::create_dir_all(&run_dir).map_err(StoreError::io(&run_dir))?;
    let lock_path = run_dir.join(LOCK_FILE);
    let _lock = super::try_lock(&lock_path)
        .map_err(StoreError::io(&lock_path))?
        .ok_or_else(|| StoreError::InUse(run_dir.clone()))?;
    // Under the lock, what another import may have done meanwhile counts.
    if let Some(done) = done_before(&run_dir, &sha256, &tables)? {
        return Ok(done);
    }
    let manifest_path = run_dir.join(MANIFEST);
    let manifest = if manifest_path.is_file() {
        // A crashed import of this archive: finish it.
        let manifest: Manifest =
            run::read_json(&manifest_path).map_err(StoreError::io(&manifest_path))?;
        if manifest.params.scales != scales {
            return Err(ImportError::bad(
                archive,
                "its scales are not the ones the run was made with",
            ));
        }
        manifest
    } else {
        let file = ArchiveFile {
            name: name.file_name.clone(),
            url: config.url.clone().unwrap_or_else(|| name.url()),
            sha256: sha256.clone(),
            bytes,
        };
        let manifest = new_manifest(&name, file, scales, tables.clone());
        run::write_json_once(&manifest_path, &manifest).map_err(StoreError::io(&manifest_path))?;
        manifest
    };
    clean_temp_files(&run_dir)?;

    let (written, last_seq, piped) = ingest(
        archive,
        &name,
        &run_dir,
        &manifest,
        &tables,
        &seed_lines,
        on_event,
    )?;
    let integrity = merge(scan.integrity(), &piped);
    let rows = table_rows(&run_dir, &tables)?;
    let complete_path = run_dir.join(COMPLETE);
    if !complete_path.is_file() {
        let complete = Complete {
            ended_ns: now_ns(),
            last_key: last_seq,
            rows: rows.clone(),
            clean: true,
            integrity: Some(integrity.clone()),
            seed: seed.clone(),
        };
        run::write_json_once(&complete_path, &complete).map_err(StoreError::io(&complete_path))?;
    }
    Ok(Imported {
        run_dir,
        written,
        done_before: false,
        rows,
        integrity,
        seed,
    })
}

/// The seed's lines to feed before the archive's, what became of it, and the scales the archive
/// and the seed need (the seed's book must convert exactly too, though a day's own records
/// nearly always need as many decimals already).
fn seeded(
    path: Option<&Path>,
    name: &ArchiveName,
    scan: &Scan,
) -> (Vec<String>, Option<Seed>, Scales) {
    let found = scan.scales();
    match path.map(|path| read_seed(path, name, scan)) {
        None => (Vec::new(), None, found),
        Some(Ok((lines, needs))) => {
            let scales = Scales {
                price_decimals: found.price_decimals.max(needs.price_decimals),
                qty_decimals: found.qty_decimals.max(needs.qty_decimals),
            };
            let used = Seed {
                used: true,
                why: None,
            };
            (lines, Some(used), scales)
        }
        Some(Err(why)) => {
            let unused = Seed {
                used: false,
                why: Some(why),
            };
            (Vec::new(), Some(unused), found)
        }
    }
}

/// The seed's lines and the scales its prices and quantities need, or why it cannot seed the
/// day whose archive the pre-scan read into `day` (module notes).
fn read_seed(path: &Path, name: &ArchiveName, day: &Scan) -> Result<(Vec<String>, Scales), String> {
    if name.format != ArchiveFormat::BybitOrderbook {
        return Err("only an order book is seeded".to_owned());
    }
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let lines: Vec<String> = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect();
    let mut scan = Scan::new(name.format);
    let mut times = Vec::with_capacity(lines.len() + 1); // each record's, then the day's first
    for (index, line) in lines.iter().enumerate() {
        name.scan(&mut scan, line, false)
            .map_err(|why| format!("line {}: {why}", index + 1))?;
        if index == 0 && scan.integrity().snapshots != Some(1) {
            return Err("it does not open with a whole book".to_owned());
        }
        times.extend(scan.integrity().last_ts_ns);
    }
    let (seed, first) = (scan.integrity(), day.integrity());
    if seed.first_ts_ns.is_none_or(|ts| ts >= name.day.start_ns()) {
        return Err("it holds no book from before the day".to_owned());
    }
    // The day's first record repeats the seed's last u (the file's overlap, or a snapshot of the
    // same book) or is the delta of the next u. A snapshot of the next u holds an update made at
    // a time neither file gives. A repeated delta would drop the book in the sync rule, never
    // apply twice.
    let next = seed.last_id.and_then(|u| u.checked_add(1));
    let follows = first.first_id == next && day.opens_whole() == Some(false);
    if first.first_id.is_none() || (first.first_id != seed.last_id && !follows) {
        let record = if day.opens_whole() == Some(true) {
            "a whole book"
        } else {
            "a delta"
        };
        return Err(format!(
            "it does not chain: its last u is {}, and the day opens with {record} of u {}",
            seed.last_id.unwrap_or_default(),
            first.first_id.unwrap_or_default()
        ));
    }
    times.extend(first.first_ts_ns);
    let mut gaps = times.windows(2).map(|pair| pair[1].saturating_sub(pair[0]));
    if let Some(gap) = gaps.find(|&gap| gap > SEED_GAP_NS) {
        return Err(format!(
            "its book goes {} ms without a record, over the {} ms after which a book is stale",
            gap / 1_000_000,
            SEED_GAP_NS / 1_000_000
        ));
    }
    Ok((lines, scan.scales()))
}

/// Writes the closing book's lines to the run's [`CLOSE`], replacing one a crashed import left.
fn write_close(run_dir: &Path, lines: &[String]) -> Result<PathBuf, ImportError> {
    let path = run_dir.join(CLOSE);
    let temp = run_dir.join(format!("{CLOSE}.tmp"));
    let mut text = lines.join("\n");
    text.push('\n');
    fs::write(&temp, text).map_err(StoreError::io(&temp))?;
    fs::rename(&temp, &path).map_err(StoreError::io(&path))?;
    Ok(path)
}

/// The symbol's whole book in `feed` as a Bybit snapshot message stamped `time_ns`, the first
/// line of a [`CLOSE`]; none while the book is not synced.
fn closing_book(feed: &Feed, name: &ArchiveName, time_ns: i64) -> Option<String> {
    let sync = feed.sync(&name.symbol)?;
    let (book, u) = (sync.book()?, sync.last_id()?);
    let scales = feed.scales(&name.symbol)?;
    let side = |levels: &mut dyn Iterator<Item = Level>| -> Vec<[String; 2]> {
        levels
            .map(|level| {
                [
                    format_units(level.price, scales.price_decimals),
                    format_units(level.qty, scales.qty_decimals),
                ]
            })
            .collect()
    };
    let stream = name.streams().into_iter().next().unwrap_or_default();
    let message = serde_json::json!({
        "topic": format!("{stream}.{}", name.symbol),
        "type": "snapshot",
        "ts": time_ns.div_euclid(1_000_000),
        "data": {"s": name.symbol, "b": side(&mut book.bids()), "a": side(&mut book.asks()), "u": u},
    });
    Some(message.to_string())
}

/// The manifest of a new archive run: one venue and symbol at the scales the file needs, on the
/// exchange clock, from this archive file.
fn new_manifest(
    name: &ArchiveName,
    file: ArchiveFile,
    scales: BTreeMap<Symbol, Scales>,
    tables: Vec<String>,
) -> Manifest {
    Manifest {
        format: FORMAT,
        domain: MARKET.to_owned(),
        run_id: run_id(name),
        producer: Producer::engine(),
        started_ns: name.day.start_ns(),
        key: KEY.to_owned(),
        tables,
        params: MarketParams {
            venue: name.venue(),
            symbols: vec![name.symbol.clone()],
            scales,
            streams: name.streams(),
            book_sample: BookSample::default(),
            clock: Clock::Exchange,
            source: DataSource::Archive {
                format: name.format.id().to_owned(),
                files: vec![file],
            },
        },
    }
}

/// The pre-scan's own findings, with what the pipeline made of the records.
fn merge(scanned: &Integrity, piped: &Integrity) -> Integrity {
    Integrity {
        errors: piped.errors,
        resyncs: piped.resyncs,
        crossed: piped.crossed,
        skipped_samples: piped.skipped_samples,
        unsynced_samples: piped.unsynced_samples,
        ..scanned.clone()
    }
}

/// The tables asked for, or the format's own; each must be a table rotate writes.
fn tables(name: &ArchiveName, asked: Option<&[String]>) -> Result<Vec<String>, ImportError> {
    let asked: Vec<String> = match asked {
        Some(asked) => asked.to_vec(),
        None => name
            .format
            .default_tables()
            .iter()
            .map(|&table| table.to_owned())
            .collect(),
    };
    for table in &asked {
        if !TABLES.contains(&table.as_str()) {
            return Err(ImportError::UnknownTable(table.clone()));
        }
    }
    // In the manifest's order, each once.
    Ok(TABLES
        .iter()
        .filter(|table| asked.iter().any(|asked| asked == *table))
        .map(|&table| table.to_owned())
        .collect())
}

/// The run, when an import of this archive with these tables has finished it already; an error
/// when the run was made from another archive or with other tables.
fn done_before(
    run_dir: &Path,
    sha256: &str,
    tables: &[String],
) -> Result<Option<Imported>, ImportError> {
    let manifest_path = run_dir.join(MANIFEST);
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest: Manifest =
        run::read_json(&manifest_path).map_err(StoreError::io(&manifest_path))?;
    same_run(run_dir, &manifest, sha256, tables)?;
    let complete_path = run_dir.join(COMPLETE);
    let done = complete_path.is_file() && tables.iter().all(|t| table_file(run_dir, t).is_file());
    if !done {
        return Ok(None);
    }
    let complete: Complete =
        run::read_json(&complete_path).map_err(StoreError::io(&complete_path))?;
    Ok(Some(Imported {
        run_dir: run_dir.to_owned(),
        written: Vec::new(),
        done_before: true,
        rows: complete.rows,
        integrity: complete.integrity.unwrap_or_default(),
        seed: complete.seed,
    }))
}

/// A run made earlier must be this archive's, with these tables.
fn same_run(
    run_dir: &Path,
    manifest: &Manifest,
    sha256: &str,
    tables: &[String],
) -> Result<(), ImportError> {
    let theirs = match &manifest.params.source {
        DataSource::Archive { files, .. } => files.first().map(|file| file.sha256.clone()),
        DataSource::Live => None,
    }
    .unwrap_or_else(|| "a live recording".to_owned());
    if theirs != sha256 {
        return Err(ImportError::OtherArchive {
            run_dir: run_dir.to_owned(),
            theirs,
            ours: sha256.to_owned(),
        });
    }
    if manifest.tables != tables {
        return Err(ImportError::OtherTables {
            run_dir: run_dir.to_owned(),
            theirs: manifest.tables.clone(),
            ours: tables.to_vec(),
        });
    }
    Ok(())
}

/// The second pass: the run's Meta at seq 0, then a frame per line of the seed and per record,
/// through rotate's ingest into the tables, and an order book's closing book into [`CLOSE`]
/// (module notes; none while the book is not synced). Returns the files written, the last seq,
/// and what the pipeline made of it.
fn ingest(
    archive: &Path,
    name: &ArchiveName,
    run_dir: &Path,
    manifest: &Manifest,
    tables: &[String],
    seed: &[String],
    on_event: &mut impl FnMut(LiveEvent<'_>),
) -> Result<(Vec<PathBuf>, u64, Integrity), ImportError> {
    let params = &manifest.params;
    let enabled: Vec<&str> = tables.iter().map(String::as_str).collect();
    let mut ingest = Ingest::new(params.venue, params.scales.clone(), params.book_sample);
    let mut skipped = 0;
    let mut out = Tables::open(run_dir, 0, ingest.levels(), &enabled, &mut skipped)?;
    let mut seq = 0;
    let closes = name.format == ArchiveFormat::BybitOrderbook;
    let day_end = name.day.start_ns().saturating_add(DAY_NS);
    // The closing book once a record reaches the day's end, then the records from that one on.
    let mut close: Option<(Option<String>, Vec<String>)> = None;
    let mut last_ns = None;
    let fed = (|| -> Result<(), ImportError> {
        let meta = Meta::for_run(manifest);
        let meta = Frame {
            seq,
            kind: Kind::Meta,
            source_ts_ns: 0,
            capture_ts_ns: manifest.started_ns,
            payload: serde_json::to_string(&meta)
                .map_err(|error| ImportError::bad(archive, error.to_string()))?,
        };
        ingest.frame(&meta, &mut out, &mut *on_event)?;
        let mut push = |ingest: &mut Ingest, out: &mut Tables, record: ArchiveFrame, at: i64| {
            seq += 1;
            let frame = Frame {
                seq,
                kind: record.kind,
                source_ts_ns: record.time_ns,
                capture_ts_ns: at,
                payload: record.payload,
            };
            ingest.frame(&frame, out, &mut *on_event)
        };
        // The seed's book stands until the record after it, so it is taken in just before
        // midnight: the grid starts at the day's first instant, not at instants of the day before.
        let day_start = name.day.start_ns();
        for (index, line) in seed.iter().enumerate() {
            let number = index + 1;
            let record = name
                .frame(line)
                .map_err(|why| ImportError::bad(archive, format!("seed line {number}: {why}")))?;
            if let Some(record) = record {
                let at = record.time_ns.max(day_start - 1);
                push(&mut ingest, &mut out, record, at)?;
            }
        }
        each_line(archive, name, &mut |number, line| {
            let record = name
                .frame(line)
                .map_err(|why| ImportError::bad(archive, format!("line {number}: {why}")))?;
            if let Some(record) = record {
                if closes && close.is_none() && record.time_ns >= day_end {
                    let book = ingest
                        .feed()
                        .zip(last_ns)
                        .and_then(|(feed, ts)| closing_book(feed, name, ts));
                    close = Some((book, Vec::new()));
                }
                if let Some((_, tail)) = close.as_mut() {
                    tail.push(line.to_owned());
                }
                last_ns = Some(record.time_ns);
                let at = record.time_ns;
                push(&mut ingest, &mut out, record, at)?;
            }
            Ok(())
        })
    })();
    if let Err(error) = fed {
        out.abandon();
        return Err(error);
    }
    let close = match close {
        Some((book, tail)) => book.map(|book| [vec![book], tail].concat()),
        None if closes => ingest
            .feed()
            .zip(last_ns)
            .and_then(|(feed, ts)| closing_book(feed, name, ts))
            .map(|book| vec![book]),
        None => None,
    };
    let mut written = Vec::new();
    out.finish(&mut written)?;
    if let Some(close) = close {
        written.push(write_close(run_dir, &close)?);
    }
    let unsynced_samples = ingest.unsynced;
    let (replayed, skipped_samples) = ingest.finish()?;
    let stats = replayed.pipeline.feed().stats();
    let integrity = Integrity {
        errors: replayed.report.errors,
        resyncs: stats.resyncs,
        crossed: stats.crossed,
        skipped_samples,
        unsynced_samples,
        ..Integrity::default()
    };
    Ok((written, seq, integrity))
}

/// The first pass: every record checked, the scales found.
fn prescan(archive: &Path, name: &ArchiveName) -> Result<Scan, ImportError> {
    let mut scan = Scan::new(name.format);
    each_line(archive, name, &mut |number, line| {
        name.scan(&mut scan, line, number == 1)
            .map_err(|why| ImportError::bad(archive, format!("line {number}: {why}")))
    })?;
    if scan.integrity().records == 0 {
        return Err(ImportError::bad(archive, "it holds no records"));
    }
    Ok(scan)
}

/// Calls `each` with every line of the archive's one file, numbered from 1, its line ending
/// (`\n` or `\r\n`) taken off; blank lines are skipped. The zip's CRC is checked at the end.
fn each_line(
    archive: &Path,
    name: &ArchiveName,
    each: &mut impl FnMut(u64, &str) -> Result<(), ImportError>,
) -> Result<(), ImportError> {
    let file = File::open(archive).map_err(StoreError::io(archive))?;
    let mut zip =
        zip::ZipArchive::new(file).map_err(|error| ImportError::bad(archive, error.to_string()))?;
    if zip.len() != 1 {
        return Err(ImportError::bad(
            archive,
            format!("it holds {} files, not one", zip.len()),
        ));
    }
    let member = zip
        .by_index(0)
        .map_err(|error| ImportError::bad(archive, error.to_string()))?;
    let held = member
        .name()
        .map_err(|error| ImportError::bad(archive, error.to_string()))?
        .into_owned();
    if held != name.member() {
        return Err(ImportError::bad(
            archive,
            format!("it holds {held:?}, not {:?}", name.member()),
        ));
    }
    let mut reader = BufReader::with_capacity(1 << 20, member);
    let mut buffer = Vec::new();
    let mut number = 0;
    loop {
        buffer.clear();
        let read = reader
            .read_until(b'\n', &mut buffer)
            .map_err(|error| ImportError::bad(archive, format!("after line {number}: {error}")))?;
        if read == 0 {
            return Ok(());
        }
        number += 1;
        let mut line: &[u8] = &buffer;
        line = line.strip_suffix(b"\n").unwrap_or(line);
        line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let line = std::str::from_utf8(line)
            .map_err(|_| ImportError::bad(archive, format!("line {number} is not UTF-8")))?;
        each(number, line)?;
    }
}

/// The SHA-256 of a file, in lower-case hex, and its size.
fn sha256(path: &Path) -> Result<(String, u64), ImportError> {
    let mut file = File::open(path).map_err(StoreError::io(path))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    let mut bytes = 0;
    loop {
        let read = file.read(&mut buffer).map_err(StoreError::io(path))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes += read as u64;
    }
    let hex = hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        });
    Ok((hex, bytes))
}

/// The SHA-256 a `<file>.CHECKSUM` beside the archive gives (Binance publishes one per file:
/// `<sha256>  <name>`), if there is one.
fn checksum_beside(archive: &Path) -> Result<Option<String>, ImportError> {
    let mut name = archive.as_os_str().to_owned();
    name.push(".CHECKSUM");
    let path = PathBuf::from(name);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(StoreError::io(&path)(error).into()),
    };
    let hash = text
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_lowercase();
    Ok(Some(hash))
}

/// An import's one file of `table`, named after seq 0.
fn table_file(run_dir: &Path, table: &str) -> PathBuf {
    run_dir.join(table).join(format!("{:020}.parquet", 0))
}

/// Rows in each of the run's tables, from their Parquet footers.
fn table_rows(run_dir: &Path, tables: &[String]) -> Result<BTreeMap<String, u64>, ImportError> {
    let mut rows = BTreeMap::new();
    for table in tables {
        let path = table_file(run_dir, table);
        let file = File::open(&path).map_err(StoreError::io(&path))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file).map_err(StoreError::from)?;
        let count = u64::try_from(reader.metadata().file_metadata().num_rows()).unwrap_or(0);
        rows.insert(table.clone(), count);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_run_is_named_by_its_day_then_venue_and_symbol() -> Result<(), ImportError> {
        let run_of = |file: &str| archive_name(Path::new(file)).map(|name| run_id(&name));
        assert_eq!(
            run_of("dl/2026-10-20_BTCUSDT_ob200.data.zip")?,
            "20261020T000000Z-bybit-linear-btcusdt"
        );
        assert_eq!(
            run_of("ETHUSDT-aggTrades-2023-06-15.zip")?,
            "20230615T000000Z-binance-usdm-ethusdt"
        );
        assert!(matches!(
            run_of("notes.zip"),
            Err(ImportError::NotAnArchive(_))
        ));
        Ok(())
    }

    #[test]
    fn tables_are_the_formats_own_or_known_ones_in_manifest_order() -> Result<(), ImportError> {
        let name = ArchiveName::parse("2026-10-20_BTCUSDT_ob200.data.zip")
            .ok_or(ImportError::NotAnArchive(PathBuf::new()))?;
        assert_eq!(tables(&name, None)?, vec!["book_samples".to_owned()]);
        let asked = vec![
            "book_samples".to_owned(),
            "frames".to_owned(),
            "frames".to_owned(),
        ];
        assert_eq!(
            tables(&name, Some(&asked))?,
            vec!["frames".to_owned(), "book_samples".to_owned()]
        );
        assert!(matches!(
            tables(&name, Some(&["agg_trades".to_owned()])),
            Err(ImportError::UnknownTable(_))
        ));
        Ok(())
    }

    /// A scratch folder of its own for each test.
    fn scratch(test: &str) -> Result<PathBuf, ImportError> {
        let dir = std::env::temp_dir().join(format!("tp-import-{test}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).map_err(StoreError::io(&dir))?;
        Ok(dir)
    }

    /// A Bybit book message of BTCUSDT at `ms`, its levels a side as `px/qty`, best first and
    /// comma-separated (empty for none).
    fn message(kind: &str, ms: i64, u: u64, bids: &str, asks: &str) -> String {
        let side = |levels: &str| {
            let levels: Vec<String> = levels
                .split(',')
                .filter_map(|level| level.split_once('/'))
                .map(|(px, qty)| format!(r#"["{px}","{qty}"]"#))
                .collect();
            format!("[{}]", levels.join(","))
        };
        format!(
            r#"{{"topic":"orderbook.200.BTCUSDT","type":"{kind}","ts":{ms},"data":{{"s":"BTCUSDT","b":{},"a":{},"u":{u},"seq":1}}}}"#,
            side(bids),
            side(asks)
        )
    }

    /// A closing book's lines.
    fn lines_of(close: &Path) -> Result<Vec<String>, ImportError> {
        let text = fs::read_to_string(close).map_err(StoreError::io(close))?;
        Ok(text.lines().map(str::to_owned).collect())
    }

    /// A closing book's first line, its whole book.
    fn book_of(close: &Path) -> Result<serde_json::Value, ImportError> {
        let first = lines_of(close)?.into_iter().next().unwrap_or_default();
        serde_json::from_str(&first).map_err(|e| ImportError::bad(close, e.to_string()))
    }

    /// Why `read_seed` refuses `lines` as the seed of 2026-10-20 whose archive opens with the
    /// message `first`; empty when it takes them.
    fn refusal(dir: &Path, lines: &[String], first: &str) -> Result<String, ImportError> {
        let close = write_close(dir, lines)?;
        let name = archive_name(Path::new("2026-10-20_BTCUSDT_ob200.data.zip"))?;
        let mut day = Scan::new(name.format);
        name.scan(&mut day, first, true)
            .map_err(|why| ImportError::bad(&close, why))?;
        Ok(read_seed(&close, &name, &day).err().unwrap_or_default())
    }

    /// Writes Bybit's archive of `day` holding `lines`, and returns its path.
    fn archive_of(dir: &Path, day: &str, lines: &[String]) -> Result<PathBuf, ImportError> {
        use std::io::Write as _;
        let name = format!("{day}_BTCUSDT_ob200.data.zip");
        let path = dir.join(&name);
        let file = File::create(&path).map_err(StoreError::io(&path))?;
        let mut zip = zip::ZipWriter::new(file);
        let bad = |error: &dyn std::fmt::Display| ImportError::bad(&path, error.to_string());
        zip.start_file(
            name.trim_end_matches(".zip"),
            zip::write::SimpleFileOptions::default(),
        )
        .map_err(|e| bad(&e))?;
        for line in lines {
            zip.write_all(format!("{line}\r\n").as_bytes())
                .map_err(|e| bad(&e))?;
        }
        zip.finish().map_err(|e| bad(&e))?;
        Ok(path)
    }

    /// The `sample_ts_ns` and `ask_px_0` of a run's book samples.
    fn samples(run_dir: &Path) -> Result<Vec<(i64, Option<i64>)>, ImportError> {
        use arrow::array::{Array, Int64Array};
        let path = table_file(run_dir, "book_samples");
        let file = File::open(&path).map_err(StoreError::io(&path))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .and_then(ParquetRecordBatchReaderBuilder::build)
            .map_err(StoreError::from)?;
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch.map_err(|e| ImportError::bad(&path, e.to_string()))?;
            let column = |name: &str| {
                batch
                    .column_by_name(name)
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .cloned()
                    .ok_or_else(|| ImportError::bad(&path, format!("no {name}")))
            };
            let (ts, ask) = (column("sample_ts_ns")?, column("ask_px_0")?);
            rows.extend(
                (0..ts.len()).map(|i| (ts.value(i), ask.is_valid(i).then(|| ask.value(i)))),
            );
        }
        Ok(rows)
    }

    #[test]
    fn a_day_seeded_with_the_day_befores_closing_book_is_sampled_from_midnight()
    -> Result<(), ImportError> {
        let dir = scratch("seeded")?;
        let midnight = 1_792_454_400_000; // 2026-10-20, in ms
        let day_before = [
            message("snapshot", midnight - 86_399_500, 10, "100.0/1", "100.5/1"),
            message("delta", midnight - 200, 11, "100.1/2", ""),
            // Stamped past its day's end, as Bybit's files have a few: the closing book's tail.
            message("delta", midnight + 50, 12, "", "100.4/3"),
        ];
        let day = [
            // The book as of the seed's last u, as Bybit's next file opens, then its changes.
            message(
                "snapshot",
                midnight + 1_450,
                12,
                "100.1/2,100.0/1",
                "100.4/3,100.5/1",
            ),
            message("delta", midnight + 1_480, 13, "", "100.3/1"),
            message("delta", midnight + 1_550, 14, "100.2/1", ""),
        ];
        let config = |seed: Option<PathBuf>| ImportConfig {
            data_dir: dir.join("data"),
            url: None,
            tables: None,
            seed,
        };
        let before = import(
            &archive_of(&dir, "2026-10-19", &day_before)?,
            &config(None),
            &mut |_| {},
        )?;
        assert_eq!(before.seed, None);
        let close = before.run_dir.join(CLOSE);
        // The book after the last record before midnight, stamped with it, then the tail as is.
        let book = book_of(&close)?;
        assert_eq!(book["type"], "snapshot");
        assert_eq!(book["ts"], midnight - 200);
        assert_eq!(book["data"]["u"], 11);
        assert_eq!(
            book["data"]["b"],
            serde_json::json!([["100.1", "2"], ["100.0", "1"]])
        );
        assert_eq!(book["data"]["a"], serde_json::json!([["100.5", "1"]]));
        assert_eq!(lines_of(&close)?[1..], day_before[2..]);

        let ns = |ms: i64| ms * 1_000_000;
        let archive = archive_of(&dir, "2026-10-20", &day)?;
        let seeded = import(&archive, &config(Some(close.clone())), &mut |_| {})?;
        assert_eq!(seeded.seed.as_ref().map(|s| s.used), Some(true));
        let rows = samples(&seeded.run_dir)?;
        // Midnight's sample is the closing book; the tail's record moves the ask at 0.05 s, and
        // the day's own records take over from 1.45 s.
        assert_eq!(rows[0], (ns(midnight), Some(1005)));
        assert_eq!(rows[1], (ns(midnight + 100), Some(1004)));
        assert_eq!(rows[14], (ns(midnight + 1_400), Some(1004)));
        assert_eq!(rows[15], (ns(midnight + 1_500), Some(1003)));
        assert_eq!(rows.len(), 16);
        // No record of the day is stamped past its end: it closes with the book after its last.
        let closed = seeded.run_dir.join(CLOSE);
        let book = book_of(&closed)?;
        assert_eq!(lines_of(&closed)?.len(), 1);
        assert_eq!(book["ts"], midnight + 1_550);
        assert_eq!(book["data"]["u"], 14);
        assert_eq!(
            book["data"]["b"],
            serde_json::json!([["100.2", "1"], ["100.1", "2"], ["100.0", "1"]])
        );
        assert_eq!(
            book["data"]["a"],
            serde_json::json!([["100.3", "1"], ["100.4", "3"], ["100.5", "1"]])
        );

        // Unseeded, the day's samples start after its first record, as before.
        fs::remove_dir_all(&seeded.run_dir).map_err(StoreError::io(&seeded.run_dir))?;
        let unseeded = import(&archive, &config(None), &mut |_| {})?;
        assert_eq!(samples(&unseeded.run_dir)?, rows[15..]);
        fs::remove_dir_all(&dir).map_err(StoreError::io(&dir))?;
        Ok(())
    }

    #[test]
    fn a_seed_that_does_not_chain_to_the_day_is_not_used() -> Result<(), ImportError> {
        let dir = scratch("unchained")?;
        let midnight = 1_792_454_400_000;
        let snapshot = message("snapshot", midnight + 1_450, 13, "101.0/1", "101.5/1");
        let config = ImportConfig {
            data_dir: dir.join("data"),
            url: None,
            tables: None,
            seed: Some(dir.join(CLOSE)),
        };
        // The day opens with a snapshot of the u after the seed's last: update 13 came at a time
        // neither file gives, so the seed's book may be stale at any instant before the snapshot.
        // The import goes on unseeded, saying why, and samples nothing from that book.
        let chained = message("snapshot", midnight - 200, 12, "100.0/1", "100.5/1");
        write_close(&dir, std::slice::from_ref(&chained))?;
        let archive = archive_of(&dir, "2026-10-20", std::slice::from_ref(&snapshot))?;
        let imported = import(&archive, &config, &mut |_| {})?;
        let why = imported.seed.and_then(|seed| seed.why).unwrap_or_default();
        assert!(why.contains("does not chain"), "{why}");
        let complete: Complete = run::read_json(&imported.run_dir.join(COMPLETE))
            .map_err(StoreError::io(&imported.run_dir))?;
        assert_eq!(complete.seed.map(|s| s.used), Some(false));
        assert_eq!(samples(&imported.run_dir)?, vec![]);

        let refused = |lines: &[String], first: &str| refusal(&dir, lines, first);
        let one = std::slice::from_ref(&chained);
        // The next u runs on only as a delta, which the sync rule applies to the seed's book.
        let delta = message("delta", midnight + 1_450, 13, "", "101.5/1");
        assert_eq!(refused(one, &delta)?, "");
        assert!(refused(one, &snapshot)?.contains("whole book of u 13"));
        // The same u runs on either way: a snapshot of the same book, or that very message again,
        // as Bybit's files overlap.
        let same = message("snapshot", midnight - 200, 13, "100.0/1", "100.5/1");
        assert_eq!(refused(&[same], &snapshot)?, "");
        assert_eq!(
            refused(&[chained.clone(), snapshot.clone()], &snapshot)?,
            ""
        );
        let behind = message("snapshot", midnight - 200, 11, "100.0/1", "100.5/1");
        assert!(refused(std::slice::from_ref(&behind), &delta)?.contains("does not chain"));
        assert!(refused(&[behind], &snapshot)?.contains("does not chain"));
        let opens_delta = message("delta", midnight - 200, 12, "100.0/1", "");
        assert!(refused(&[opens_delta], &delta)?.contains("whole book"));
        let late = message("snapshot", midnight, 12, "100.0/1", "100.5/1");
        assert!(refused(&[late], &delta)?.contains("before the day"));
        let eth = chained.replace(r#""s":"BTCUSDT""#, r#""s":"ETHUSDT""#);
        assert!(refused(&[eth], &delta)?.contains("ETHUSDT"));
        fs::remove_dir_all(&dir).map_err(StoreError::io(&dir))?;
        Ok(())
    }

    #[test]
    fn a_day_opening_with_the_next_delta_applies_it_to_the_seeds_book() -> Result<(), ImportError> {
        let dir = scratch("next-delta")?;
        let midnight = 1_792_454_400_000;
        let seed = [message(
            "snapshot",
            midnight - 200,
            12,
            "100.0/1",
            "100.5/1",
        )];
        let day = [
            message("delta", midnight + 1_450, 13, "", "100.4/2"),
            message("delta", midnight + 1_550, 14, "100.1/1", ""),
        ];
        let config = ImportConfig {
            data_dir: dir.join("data"),
            url: None,
            tables: None,
            seed: Some(write_close(&dir, &seed)?),
        };
        let imported = import(&archive_of(&dir, "2026-10-20", &day)?, &config, &mut |_| {})?;
        assert_eq!(imported.seed.map(|s| s.used), Some(true));
        let ns = |ms: i64| ms * 1_000_000;
        let rows = samples(&imported.run_dir)?;
        assert_eq!(rows.len(), 16);
        assert_eq!(rows[14], (ns(midnight + 1_400), Some(1005)));
        assert_eq!(rows[15], (ns(midnight + 1_500), Some(1004))); // update 13 on the seed's book
        let integrity = &imported.integrity;
        assert_eq!((integrity.resyncs, integrity.unsynced_samples), (0, 0));
        fs::remove_dir_all(&dir).map_err(StoreError::io(&dir))?;
        Ok(())
    }

    #[test]
    fn a_seed_whose_book_goes_over_2_s_without_a_record_is_not_used() -> Result<(), ImportError> {
        let dir = scratch("quiet")?;
        let midnight = 1_792_454_400_000;
        let first = |ms: i64| message("snapshot", ms, 13, "101.0/1", "101.5/1");
        // The reducer counts a book's silence from midnight only (STALE_SAMPLES in
        // book/features.py): seeded with a book last changed at 23:59:55, the day's instants up to
        // its first record would count, where one import of both files masks them as stale.
        let quiet = message("snapshot", midnight - 5_000, 13, "100.0/1", "100.5/1");
        let why = refusal(&dir, &[quiet], &first(midnight + 1_450))?;
        assert!(why.contains("goes 6450 ms without a record"), "{why}");
        // After midnight too, up to the day's first record; 2 s exactly is not over.
        let seed = [
            message("snapshot", midnight - 100, 12, "100.0/1", "100.5/1"),
            message("delta", midnight + 50, 13, "", "100.4/1"),
        ];
        let why = refusal(&dir, &seed, &first(midnight + 2_051))?;
        assert!(why.contains("goes 2001 ms without a record"), "{why}");
        assert_eq!(refusal(&dir, &seed, &first(midnight + 2_050))?, "");
        fs::remove_dir_all(&dir).map_err(StoreError::io(&dir))?;
        Ok(())
    }

    #[test]
    fn the_sha256_is_the_published_check_value() -> Result<(), ImportError> {
        let dir = std::env::temp_dir().join(format!("tp-import-sha-{}", std::process::id()));
        fs::create_dir_all(&dir).map_err(StoreError::io(&dir))?;
        let path = dir.join("abc");
        fs::write(&path, b"abc").map_err(StoreError::io(&path))?;
        assert_eq!(
            sha256(&path)?,
            (
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_owned(),
                3
            )
        );
        fs::remove_dir_all(&dir).map_err(StoreError::io(&dir))?;
        Ok(())
    }
}
