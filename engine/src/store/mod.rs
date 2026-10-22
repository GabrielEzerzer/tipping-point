//! Market data on disk.
//!
//! The live loop writes every input it applies to an append-only binary log ([`writer`],
//! [`frame`]), which a crash can tear only at its tail ([`segment`]). An offline step rotates
//! closed log segments into Parquet ([`rotate`]); `DuckDB` queries that Parquet and replays it
//! ([`query`], [`replay`]) through the same pipeline the live loop runs, so a replay rebuilds
//! the books the live loop had. Run folders follow the layout shared with the Ising simulator
//! ([`run`]).

pub mod frame;
pub mod query;
pub mod replay;
pub mod rotate;
pub mod run;
pub mod segment;
pub mod writer;

use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

use frame::{FrameError, Kind};

/// Nanoseconds since the Unix epoch on the wall clock, for capture times. The wall clock can
/// step back; order comes from seq, never from this.
#[must_use]
pub fn now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| {
            i64::try_from(since.as_nanos()).unwrap_or(i64::MAX)
        })
}

/// Binance's millisecond times as nanoseconds, the unit of every `_ns` column; 0 when out of
/// range.
#[must_use]
pub fn ms_to_ns(ms: u64) -> i64 {
    i64::try_from(ms)
        .ok()
        .and_then(|ms| ms.checked_mul(1_000_000))
        .unwrap_or(0)
}

/// Why the log could not be written, read or recovered.
#[derive(Debug, Error)]
pub enum LogError {
    #[error("{}: {source}", .path.display())]
    Io { path: PathBuf, source: io::Error },
    #[error(
        "{}: frame {index} at byte {offset} is bad ({cause}) and valid frames follow it, so it is not a torn tail; nothing was changed",
        .path.display()
    )]
    Damaged {
        path: PathBuf,
        offset: u64,
        index: u64,
        cause: FrameError,
    },
    #[error("{}: not named like a segment (<first_seq:020>.tplog)", .0.display())]
    BadName(PathBuf),
    #[error("{}: a log writer still holds this run", .0.display())]
    InUse(PathBuf),
    #[error("{}: already holds a log; a new run never appends to an old one", .0.display())]
    NotEmpty(PathBuf),
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error("the log writer has stopped")]
    WriterGone,
    #[error("the log writer panicked")]
    WriterPanicked,
}

impl LogError {
    /// Wraps an I/O error with the path it happened on.
    pub(crate) fn io(path: &Path) -> impl FnOnce(io::Error) -> Self + '_ {
        move |source| Self::Io {
            path: path.to_owned(),
            source,
        }
    }
}

/// Why a run could not be rotated, queried or replayed.
#[derive(Debug, Error)]
pub enum StoreError {
    #[error(transparent)]
    Log(#[from] LogError),
    #[error("{}: {source}", .path.display())]
    Io { path: PathBuf, source: io::Error },
    #[error("parquet: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("duckdb: {0}")]
    Duck(#[from] duckdb::Error),
    #[error("bad {what} payload at seq {seq}: {error}")]
    Payload {
        what: &'static str,
        seq: u64,
        error: serde_json::Error,
    },
    #[error("a run starts with a Meta frame at seq 0, not {kind:?} at seq {seq}")]
    MissingMeta { kind: Kind, seq: u64 },
    #[error("no frames to replay")]
    NoFrames,
    #[error(
        "{0:?} is not a table (one of frames, depth_updates, agg_trades, snapshots, book_samples)"
    )]
    UnknownTable(String),
    #[error("seq {got} follows seq {previous}: frames are missing or out of order")]
    SeqGap { previous: u64, got: u64 },
    #[error("{}: segment has no end marker and is not the newest; recover the run first", .0.display())]
    Unclosed(PathBuf),
    #[error("{0} does not fit an Int64 column")]
    TooBig(u64),
    #[error("kind code {0} in Parquet is not a frame kind")]
    BadKind(i64),
    #[error("seq {0} in Parquet is negative")]
    BadSeq(i64),
}

impl StoreError {
    /// Wraps an I/O error with the path it happened on.
    pub(crate) fn io(path: &Path) -> impl FnOnce(io::Error) -> Self + '_ {
        move |source| Self::Io {
            path: path.to_owned(),
            source,
        }
    }
}
