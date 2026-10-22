//! Run directories, laid out the way the Ising simulator lays out its own:
//!
//! ```text
//! <data dir>/market/<run_id>/
//!   manifest.json               written once, at the start
//!   complete.json               written once, at a clean end; absent = running or crashed
//!   log/<first_seq:020>.tplog   the binary log
//!   <table>/<first_seq:020>.parquet
//! ```
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
use crate::binance::{Scales, Symbol};

/// The environment variable naming the data dir.
pub const DATA_DIR_ENV: &str = "TP_DATA_DIR";
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

/// Days since 1970-01-01 to a proleptic Gregorian (year, month, day): Howard Hinnant's
/// `civil_from_days`, valid for any day count an `i64` of nanoseconds can reach.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
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
    pub symbols: Vec<Symbol>,
    pub scales: BTreeMap<Symbol, Scales>,
    pub streams: Vec<String>,
    pub book_sample: BookSample,
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
    /// a market run counts its frames; rotate later writes exactly that many `frames` rows.
    pub rows: BTreeMap<String, u64>,
    pub clean: bool,
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
        format: 1,
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
