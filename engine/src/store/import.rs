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
//! Safe to re-run: an import of a run that is complete changes nothing, a crashed one is finished
//! (its temp files deleted, the tables it had written kept), and an archive other than the one a
//! run was made from, or another set of tables, is refused. One import or rotate of a run at a
//! time.

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
    self, ArchiveFile, BookSample, COMPLETE, Clock, Complete, DataSource, FORMAT, Integrity, KEY,
    MANIFEST, MARKET, Manifest, MarketParams, Producer,
};
use super::{StoreError, now_ns};
use crate::market::live::LiveEvent;
use crate::market::{Scales, Symbol};
use crate::venue::archive::{ArchiveName, Scan};

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

/// The run an archive becomes, under `data_dir`.
///
/// # Errors
/// Returns [`ImportError::NotAnArchive`] when the file name is not an archive's.
pub fn run_dir_of(archive: &Path, data_dir: &Path) -> Result<PathBuf, ImportError> {
    let name = archive_name(archive)?;
    Ok(data_dir.join(MARKET).join(run_id(&name)))
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
    let scales = BTreeMap::from([(name.symbol.clone(), scan.scales())]);

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

    let (written, last_seq, piped) =
        ingest(archive, &name, &run_dir, &manifest, &tables, on_event)?;
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
        };
        run::write_json_once(&complete_path, &complete).map_err(StoreError::io(&complete_path))?;
    }
    Ok(Imported {
        run_dir,
        written,
        done_before: false,
        rows,
        integrity,
    })
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

/// The second pass: the run's Meta at seq 0, then a frame per record, through rotate's ingest
/// into the tables. Returns the files written, the last seq, and what the pipeline made of it.
fn ingest(
    archive: &Path,
    name: &ArchiveName,
    run_dir: &Path,
    manifest: &Manifest,
    tables: &[String],
    on_event: &mut impl FnMut(LiveEvent<'_>),
) -> Result<(Vec<PathBuf>, u64, Integrity), ImportError> {
    let params = &manifest.params;
    let enabled: Vec<&str> = tables.iter().map(String::as_str).collect();
    let mut ingest = Ingest::new(params.venue, params.scales.clone(), params.book_sample);
    let mut skipped = 0;
    let mut out = Tables::open(run_dir, 0, ingest.levels(), &enabled, &mut skipped)?;
    let mut seq = 0;
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
        each_line(archive, name, &mut |number, line| {
            let frame = name
                .frame(line)
                .map_err(|why| ImportError::bad(archive, format!("line {number}: {why}")))?;
            if let Some(frame) = frame {
                seq += 1;
                let frame = Frame {
                    seq,
                    kind: frame.kind,
                    source_ts_ns: frame.time_ns,
                    capture_ts_ns: frame.time_ns,
                    payload: frame.payload,
                };
                ingest.frame(&frame, &mut out, &mut *on_event)?;
            }
            Ok(())
        })
    })();
    if let Err(error) = fed {
        out.abandon();
        return Err(error);
    }
    let mut written = Vec::new();
    out.finish(&mut written)?;
    let (replayed, skipped_samples) = ingest.finish()?;
    let stats = replayed.pipeline.feed().stats();
    let integrity = Integrity {
        errors: replayed.report.errors,
        resyncs: stats.resyncs,
        crossed: stats.crossed,
        skipped_samples,
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
        let data = Path::new("data");
        assert_eq!(
            run_dir_of(Path::new("dl/2026-10-20_BTCUSDT_ob200.data.zip"), data)?,
            data.join("market")
                .join("20261020T000000Z-bybit-linear-btcusdt")
        );
        assert_eq!(
            run_dir_of(Path::new("ETHUSDT-aggTrades-2023-06-15.zip"), data)?,
            data.join("market")
                .join("20230615T000000Z-binance-usdm-ethusdt")
        );
        assert!(matches!(
            run_dir_of(Path::new("notes.zip"), data),
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
