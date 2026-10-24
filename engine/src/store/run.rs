//! Run directories, laid out the way the Ising simulator lays out its own:
//!
//! ```text
//! <data dir>/market/<run_id>/
//!   manifest.json               written once, at the start
//!   complete.json               written once, at a clean end; absent = running or crashed
//!   log/<first_seq:020>.tplog   the binary log (a live run; an archive run has none)
//!   <table>/<first_seq:020>.parquet
//! ```
//!
//! A market run holds one venue. A live run's `capture_ts_ns` is the local clock when an input
//! arrived; an archive run's is the exchange's own time (its manifest says `clock`), and its
//! manifest names the archive files it came from, each the lossless copy of what the run holds.
//!
//! Files only ever appear: each is written to a temp name, synced, then renamed to a name that
//! did not exist. Nothing is rewritten.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::now_ns;
use super::rotate::TABLES;
use super::writer::Summary;
use crate::market::{Scales, Symbol};
use crate::venue::Venue;

/// The environment variable naming the data dir.
pub const DATA_DIR_ENV: &str = "TP_DATA_DIR";
/// The manifest format this code writes: 2 added the venue, the clock and the data's source.
pub const FORMAT: u32 = 2;
/// Nanoseconds in a UTC day.
pub const DAY_NS: i64 = 86_400_000_000_000;
/// The market domain's folder under the data dir.
pub const MARKET: &str = "market";
/// The run's key column: every table carries it, rows in key order.
pub const KEY: &str = "seq";
/// Written once when a run starts.
pub const MANIFEST: &str = "manifest.json";
/// Written once when a run ends cleanly.
pub const COMPLETE: &str = "complete.json";

/// `TP_DATA_DIR`, or else a per-user data dir outside the repository (and outside any synced
/// Desktop): `%LOCALAPPDATA%\tipping-point` on Windows, `$XDG_DATA_HOME/tipping-point` or
/// `~/.local/share/tipping-point` elsewhere. A day of recording is gigabytes, so it must never
/// land under git, the vault's map rebuild or a cloud-synced folder.
#[must_use]
pub fn data_dir() -> PathBuf {
    data_dir_from(|name| std::env::var_os(name))
}

fn data_dir_from(var: impl Fn(&str) -> Option<OsString>) -> PathBuf {
    let set = |name: &str| var(name).filter(|value| !value.is_empty());
    if let Some(dir) = set(DATA_DIR_ENV) {
        return PathBuf::from(dir);
    }
    let base = set("LOCALAPPDATA")
        .or_else(|| set("XDG_DATA_HOME"))
        .map(PathBuf::from)
        .or_else(|| set("HOME").map(|home| PathBuf::from(home).join(".local").join("share")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("tipping-point")
}

/// A run id: the UTC start time plus a tag, e.g. `20261021T140512Z-btceth`. Sorts by start
/// time; the tag may hold only ASCII letters, digits, `-` and `_` (an `=` would make `DuckDB`
/// read the folder as a hive partition).
///
/// # Errors
/// Returns the tag back when it is empty or holds anything else.
pub fn run_id(started_ns: i64, tag: &str) -> Result<String, String> {
    let ok = !tag.is_empty()
        && tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !ok {
        return Err(format!(
            "run tag {tag:?} must be non-empty ASCII letters, digits, '-' or '_'"
        ));
    }
    let secs = started_ns.div_euclid(1_000_000_000);
    let (year, month, day) = civil_from_days(secs.div_euclid(86_400));
    let in_day = secs.rem_euclid(86_400);
    Ok(format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z-{tag}",
        in_day / 3600,
        in_day / 60 % 60,
        in_day % 60
    ))
}

/// A proleptic Gregorian (year, month, day) to days since 1970-01-01: Howard Hinnant's
/// `days_from_civil`, the inverse of [`civil_from_days`].
#[must_use]
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let (month, day) = (i64::from(month), i64::from(day));
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year.rem_euclid(400);
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Days since 1970-01-01 to a proleptic Gregorian (year, month, day): Howard Hinnant's
/// `civil_from_days`, valid for any day count an `i64` of nanoseconds can reach.
#[must_use]
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Who wrote a run: the producer's name and version, and the commit it was built from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Producer {
    pub name: String,
    pub version: String,
    /// `None` when git is unavailable, e.g. a binary run away from its checkout.
    pub git_sha: Option<String>,
    /// Whether the checkout had uncommitted edits.
    pub dirty: Option<bool>,
}

impl Producer {
    /// This engine build, with the commit read from git in the crate's own checkout.
    #[must_use]
    pub fn engine() -> Self {
        let git = |args: &[&str]| {
            let output = Command::new("git")
                .args(args)
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        };
        Self {
            name: "engine".to_owned(),
            version: crate::version().to_owned(),
            git_sha: git(&["rev-parse", "HEAD"]),
            dirty: git(&["status", "--porcelain"]).map(|status| !status.is_empty()),
        }
    }
}

/// How the offline rotate step samples the book.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookSample {
    /// Levels kept per side.
    pub levels: usize,
    /// Grid spacing on the capture clock.
    pub every_ns: i64,
}

impl Default for BookSample {
    fn default() -> Self {
        Self {
            levels: 20,
            every_ns: 100_000_000,
        }
    }
}

/// What a market run recorded and how.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketParams {
    /// The venue whose streams the run holds; one run per venue. Runs recorded before there
    /// were venues are Binance's.
    #[serde(default = "binance")]
    pub venue: Venue,
    pub symbols: Vec<Symbol>,
    pub scales: BTreeMap<Symbol, Scales>,
    pub streams: Vec<String>,
    pub book_sample: BookSample,
    /// What the run's `capture_ts_ns` is; runs from before manifests said were live.
    #[serde(default)]
    pub clock: Clock,
    /// Where the run's data came from; runs from before manifests said were live.
    #[serde(default)]
    pub source: DataSource,
}

/// What a run's `capture_ts_ns` is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Clock {
    /// The local wall clock when each input arrived: a live run.
    #[default]
    Capture,
    /// The exchange's own time for each record: an archive run, which joins live runs on
    /// exchange time, never capture time.
    Exchange,
}

/// Where a run's data came from.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DataSource {
    /// Recorded from the venue's live streams (or a script played through the live path).
    #[default]
    Live,
    /// Imported from archive files of `format`.
    Archive {
        format: String,
        files: Vec<ArchiveFile>,
    },
}

/// One archive file a run was imported from, the lossless copy of what the run holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchiveFile {
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
}

/// What an archive import checked about the archive and made of it, kept in `complete.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Integrity {
    /// Records the archive holds (a CSV header is no record).
    pub records: u64,
    /// The first and last exchange time among them.
    pub first_ts_ns: Option<i64>,
    pub last_ts_ns: Option<i64>,
    /// Records whose exchange time is outside the archive's day.
    pub outside_day: u64,
    /// The venue's continuity id over the file (Bybit's `u`, Binance's aggregate trade id), and
    /// the places it did not run on by one (for Bybit, only between snapshots).
    pub first_id: Option<u64>,
    pub last_id: Option<u64>,
    pub id_gaps: u64,
    /// Binance: aggregated trades whose first trade id did not follow the last one's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trade_id_gaps: Option<u64>,
    /// Bybit: whole books in the file, the first and every one Bybit sent again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshots: Option<u64>,
    /// What the pipeline made of the records: inputs it could not use, books it threw away, and
    /// updates that left a book crossed.
    pub errors: u64,
    pub resyncs: u64,
    pub crossed: u64,
    /// Book-sample instants left unsampled across exchange-clock jumps of over a day.
    pub skipped_samples: u64,
    /// Book-sample rows written while their book was not synced: instants with no trustworthy
    /// book, before the first snapshot or after a gap until the next one. A day that lost one
    /// message and never got a snapshot again shows here, not as an error.
    #[serde(default)]
    pub unsynced_samples: u64,
}

/// The venue of every run recorded before manifests said which venue they were.
fn binance() -> Venue {
    Venue::BinanceUsdm
}

/// `manifest.json`: written once, when the run starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub domain: String,
    pub run_id: String,
    pub producer: Producer,
    pub started_ns: i64,
    pub key: String,
    pub tables: Vec<String>,
    pub params: MarketParams,
}

/// `complete.json`: written once, at a clean end. A run without it is running or crashed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Complete {
    pub ended_ns: i64,
    pub last_key: u64,
    /// Rows per table at the end of the run. The log is the only table a live run writes, so
    /// a live market run counts its frames; rotate later writes exactly that many `frames` rows.
    /// An archive run counts the rows of each table it wrote.
    pub rows: BTreeMap<String, u64>,
    pub clean: bool,
    /// An archive run's checks of its archive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub integrity: Option<Integrity>,
}

/// A new market run's folder, `<data dir>/market/<run_id>`, with the run id stamped now.
///
/// # Errors
/// Returns the message when the tag is not a valid run tag.
pub fn market_run_dir(data_dir: &Path, tag: &str) -> Result<PathBuf, String> {
    Ok(data_dir.join(MARKET).join(run_id(now_ns(), tag)?))
}

/// Creates a market run folder and writes its manifest. The folder must not exist yet; its
/// name is the run id.
///
/// # Errors
/// Returns the I/O error, `AlreadyExists` when the folder exists, or `InvalidInput` when it
/// has no usable name.
pub fn create(run_dir: &Path, params: MarketParams) -> io::Result<Manifest> {
    let run_id = run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "a run folder needs a UTF-8 name",
            )
        })?
        .to_owned();
    if let Some(parent) = run_dir.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(run_dir)?;
    let manifest = Manifest {
        format: FORMAT,
        domain: MARKET.to_owned(),
        run_id,
        producer: Producer::engine(),
        started_ns: now_ns(),
        key: KEY.to_owned(),
        tables: TABLES.iter().map(|&table| table.to_owned()).collect(),
        params,
    };
    write_json_once(&run_dir.join(MANIFEST), &manifest)?;
    Ok(manifest)
}

/// Writes `complete.json` for a run whose log closed cleanly.
///
/// # Errors
/// Returns the I/O error, or `AlreadyExists` when the run is already complete.
pub fn complete(run_dir: &Path, log: &Summary) -> io::Result<()> {
    let complete = Complete {
        ended_ns: now_ns(),
        last_key: log.last_seq,
        rows: BTreeMap::from([("frames".to_owned(), log.frames)]),
        clean: true,
        integrity: None,
    };
    write_json_once(&run_dir.join(COMPLETE), &complete)
}

/// Writes `value` as pretty JSON to `path`: to `path.tmp` first, synced, then renamed. Refuses
/// to replace a file that exists.
///
/// # Errors
/// Returns the I/O error, or `AlreadyExists` when `path` is there already.
pub fn write_json_once(path: &Path, value: &impl Serialize) -> io::Result<()> {
    let mut text = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    text.push(b'\n');
    write_once(path, |file| file.write_all(&text))
}

/// Creates `path` by writing `path.tmp` with `write`, syncing it and renaming it into place.
/// Refuses to replace a file that exists.
///
/// # Errors
/// Returns the I/O error, or `AlreadyExists` when `path` is there already.
pub fn write_once(path: &Path, write: impl FnOnce(&mut File) -> io::Result<()>) -> io::Result<()> {
    if path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("{} exists; files are never rewritten", path.display()),
        ));
    }
    let tmp = tmp_path(path);
    let mut file = File::create(&tmp)?;
    write(&mut file)?;
    file.sync_all()?;
    drop(file);
    rename_new(&tmp, path)
}

/// `path` with `.tmp` appended to its file name.
#[must_use]
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(OsString::from).unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Renames a fresh temp file to a new name, retrying while Windows reports the file busy (a
/// scanner or indexer holding it briefly).
///
/// # Errors
/// Returns the last error once the retries run out.
pub fn rename_new(from: &Path, to: &Path) -> io::Result<()> {
    let mut wait = Duration::from_millis(25);
    for _ in 0..6 {
        match fs::rename(from, to) {
            Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {
                sleep(wait);
                wait *= 2;
            }
            done => return done,
        }
    }
    fs::rename(from, to)
}

/// Reads a JSON file.
///
/// # Errors
/// Returns the I/O error, or an `InvalidData` error naming the file when it does not parse.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> io::Result<T> {
    let text = fs::read(path)?;
    serde_json::from_slice(&text).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {error}", path.display()),
        )
    })
}

/// The run dirs under `<data dir>/<domain>/`, in run-id (start time) order: every folder with a
/// `manifest.json`.
///
/// # Errors
/// Returns the I/O error from listing the domain folder; a missing folder is no runs.
pub fn runs(domain_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(domain_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut runs = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.join("manifest.json").is_file() {
            runs.push(path);
        }
    }
    runs.sort();
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_ids_are_utc_start_plus_tag() {
        // 2026-10-21T14:05:12Z
        let started = 1_792_591_512_000_000_000;
        assert_eq!(
            run_id(started, "btceth").as_deref(),
            Ok("20261021T140512Z-btceth")
        );
        assert_eq!(run_id(0, "x").as_deref(), Ok("19700101T000000Z-x"));
        // 2000-02-29T23:59:59Z, a leap day.
        assert_eq!(
            run_id(951_868_799_000_000_000, "leap").as_deref(),
            Ok("20000229T235959Z-leap")
        );
    }

    #[test]
    fn days_from_civil_inverts_civil_from_days() {
        for days in [-719_468, -1, 0, 1, 11_016, 20_746, 2_932_896] {
            let (year, month, day) = civil_from_days(days);
            let (month, day) = (u32::try_from(month).ok(), u32::try_from(day).ok());
            let (Some(month), Some(day)) = (month, day) else {
                panic!("a month and a day");
            };
            assert_eq!(
                days_from_civil(year, month, day),
                days,
                "{year}-{month}-{day}"
            );
        }
        assert_eq!(
            days_from_civil(2026, 10, 20) * DAY_NS,
            1_792_454_400_000_000_000
        );
    }

    #[test]
    fn a_manifest_from_before_venues_reads_as_a_live_binance_run() -> io::Result<()> {
        let old = r#"{"symbols":["BTCUSDT"],"scales":{"BTCUSDT":{"price_decimals":1,"qty_decimals":3}},"streams":["depth@100ms"],"book_sample":{"levels":20,"every_ns":100000000}}"#;
        let params: MarketParams = serde_json::from_str(old).map_err(io::Error::other)?;
        assert_eq!(params.venue, Venue::BinanceUsdm);
        assert_eq!(
            (params.clock, params.source),
            (Clock::Capture, DataSource::Live)
        );
        Ok(())
    }

    #[test]
    fn run_tags_that_would_confuse_a_reader_are_refused() {
        for tag in ["", "a=b", "a/b", "a b", "a.b"] {
            assert!(run_id(0, tag).is_err(), "{tag:?}");
        }
    }

    #[test]
    fn the_data_dir_is_the_env_var_or_else_outside_the_repository() {
        let from = |vars: &[(&str, &str)]| {
            let vars: Vec<(String, OsString)> = vars
                .iter()
                .map(|(k, v)| ((*k).to_owned(), OsString::from(v)))
                .collect();
            data_dir_from(move |name| vars.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone()))
        };
        assert_eq!(
            from(&[("TP_DATA_DIR", "D:/data"), ("LOCALAPPDATA", "C:/l")]),
            PathBuf::from("D:/data")
        );
        assert_eq!(
            from(&[("TP_DATA_DIR", ""), ("LOCALAPPDATA", "C:/l")]),
            Path::new("C:/l").join("tipping-point")
        );
        assert_eq!(
            from(&[("HOME", "/home/g")]),
            Path::new("/home/g/.local/share").join("tipping-point")
        );

        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("engine/ sits in the repo root");
        let real_default = data_dir_from(|name| {
            (name != DATA_DIR_ENV)
                .then(|| std::env::var_os(name))
                .flatten()
        });
        assert!(
            !real_default.starts_with(repo),
            "{} is inside the repository",
            real_default.display()
        );
    }

    #[test]
    fn json_files_are_written_once() -> io::Result<()> {
        let dir = std::env::temp_dir().join(format!("tp-run-test-{}", std::process::id()));
        fs::create_dir_all(&dir)?;
        let path = dir.join("complete.json");
        let _ = fs::remove_file(&path);
        let complete = Complete {
            ended_ns: 5,
            last_key: 9,
            rows: BTreeMap::from([("frames".to_owned(), 10)]),
            clean: true,
            integrity: None,
        };
        write_json_once(&path, &complete)?;
        assert_eq!(read_json::<Complete>(&path)?, complete);
        assert_eq!(
            write_json_once(&path, &complete).map_err(|e| e.kind()),
            Err(io::ErrorKind::AlreadyExists)
        );
        assert!(!tmp_path(&path).exists());
        fs::remove_dir_all(&dir)
    }
}
