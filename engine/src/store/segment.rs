//! Reading one segment file, and recovering the newest segment of a crashed run.
//!
//! The reader goes frame by frame and stops at the first bad one: a short header; a wrong
//! magic, version, kind or reserved field; a length over the limit or past the end of the
//! file; a CRC mismatch; or a seq that is not the previous one plus one. What it does next
//! depends on what follows. If no valid frame starts anywhere at or after that point, the bad
//! bytes are a torn write (a crash cut the last frame short or left junk) and everything before
//! them is good. If a valid frame does follow, the damage is in the middle of the file: nothing
//! is cut, and the reader reports which frame and byte it is.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::frame::{Frame, FrameError, HEADER_LEN, Header, Kind, MAGIC, SegmentEnd, TRAILER_LEN};
use super::{LogError, now_ns};

/// Segment files are `<first_seq:020>.tplog`.
pub const EXTENSION: &str = "tplog";
/// A run's log folder, inside its run dir.
pub const LOG_DIR: &str = "log";
/// The file the writer holds a lock on for as long as it writes a run.
pub const LOCK_FILE: &str = "writer.lock";

/// The file name of the segment that starts at `first_seq`.
#[must_use]
pub fn file_name(first_seq: u64) -> String {
    format!("{first_seq:020}.{EXTENSION}")
}

/// The first seq a segment file is named after, if `path` is named like a segment.
#[must_use]
pub fn first_seq_of(path: &Path) -> Option<u64> {
    if path.extension()? != EXTENSION {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    if stem.len() != 20 || !stem.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    stem.parse().ok()
}

/// The segment files in a log folder, in seq order. Other files (the lock) are ignored.
///
/// # Errors
/// Returns the I/O error from listing the folder.
pub fn list(log_dir: &Path) -> Result<Vec<(u64, PathBuf)>, LogError> {
    let mut segments = Vec::new();
    for entry in fs::read_dir(log_dir).map_err(LogError::io(log_dir))? {
        let path = entry.map_err(LogError::io(log_dir))?.path();
        if let Some(first_seq) = first_seq_of(&path) {
            segments.push((first_seq, path));
        }
    }
    segments.sort();
    Ok(segments)
}

/// How a segment ends, once its last good frame has been read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ending {
    /// A `SegmentEnd` marker, then nothing.
    Closed(SegmentEnd),
    /// The file ends right after a good frame and has no marker: the writer stopped between
    /// frames, or is still writing.
    Open,
    /// The first bad frame starts at byte `at` and no valid frame follows it: a torn write,
    /// safe to cut off.
    Torn { at: u64, cause: FrameError },
}

/// Reads a segment file frame by frame. After an error the reader is spent.
#[derive(Debug)]
pub struct SegmentReader {
    path: PathBuf,
    file: BufReader<File>,
    len: u64,
    offset: u64,
    frames: u64,
    next_seq: u64,
    last_seq: Option<u64>,
    marker: Option<SegmentEnd>,
    ending: Option<Ending>,
}

impl SegmentReader {
    /// Opens a segment; its first frame must carry the seq the file is named after.
    ///
    /// # Errors
    /// Returns an error when the file cannot be opened or is not named like a segment.
    pub fn open(path: &Path) -> Result<Self, LogError> {
        let first_seq = first_seq_of(path).ok_or_else(|| LogError::BadName(path.to_owned()))?;
        let file = File::open(path).map_err(LogError::io(path))?;
        let len = file.metadata().map_err(LogError::io(path))?.len();
        Ok(Self {
            path: path.to_owned(),
            file: BufReader::with_capacity(1 << 16, file),
            len,
            offset: 0,
            frames: 0,
            next_seq: first_seq,
            last_seq: None,
            marker: None,
            ending: None,
        })
    }

    /// The next good frame, or `None` once there are no more; [`Self::ending`] then says how
    /// the segment ended.
    ///
    /// # Errors
    /// Returns [`LogError::Damaged`] when a bad frame has valid frames after it, and I/O errors.
    pub fn next_frame(&mut self) -> Result<Option<Frame>, LogError> {
        if self.ending.is_some() {
            return Ok(None);
        }
        if let Some(marker) = &self.marker {
            if self.offset == self.len {
                self.ending = Some(Ending::Closed(marker.clone()));
                return Ok(None);
            }
            return Err(self.damaged(FrameError::AfterEnd));
        }
        if self.offset == self.len {
            self.ending = Some(Ending::Open);
            return Ok(None);
        }
        let (frame, size) = match self.read_one().map_err(LogError::io(&self.path))? {
            Ok(read) => read,
            Err(cause) => return self.bad(cause),
        };
        if frame.seq != self.next_seq {
            let cause = FrameError::SeqGap {
                expected: self.next_seq,
                got: frame.seq,
            };
            return self.bad(cause);
        }
        if frame.kind == Kind::SegmentEnd {
            match serde_json::from_str::<SegmentEnd>(&frame.payload) {
                Ok(marker) => self.marker = Some(marker),
                Err(error) => return self.bad(FrameError::BadMarker(error.to_string())),
            }
        }
        self.offset += size;
        self.frames += 1;
        self.next_seq += 1;
        self.last_seq = Some(frame.seq);
        Ok(Some(frame))
    }

    /// How the segment ended, once [`Self::next_frame`] has returned `None`.
    #[must_use]
    pub fn ending(&self) -> Option<&Ending> {
        self.ending.as_ref()
    }

    /// Good frames read so far.
    #[must_use]
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// The seq of the last good frame read.
    #[must_use]
    pub fn last_seq(&self) -> Option<u64> {
        self.last_seq
    }

    /// The seq the next frame must carry.
    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// The segment's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reads the frame at the current offset. The outer error is I/O, the inner one says why
    /// the bytes are not a frame.
    fn read_one(&mut self) -> io::Result<Result<(Frame, u64), FrameError>> {
        let available = self.len - self.offset;
        if available < HEADER_LEN as u64 {
            return Ok(Err(FrameError::Truncated {
                needed: HEADER_LEN as u64,
                available,
            }));
        }
        let mut head = [0; HEADER_LEN];
        self.file.read_exact(&mut head)?;
        let header = match Header::parse(&head) {
            Ok(header) => header,
            Err(cause) => return Ok(Err(cause)),
        };
        let total = (HEADER_LEN + header.body_len()) as u64;
        if available < total {
            return Ok(Err(FrameError::Truncated {
                needed: total,
                available,
            }));
        }
        let mut body = vec![0; header.body_len()];
        self.file.read_exact(&mut body)?;
        Ok(header.finish(&head, body).map(|frame| (frame, total)))
    }

    /// A bad frame at the current offset: a torn tail when nothing valid follows, else damage.
    fn bad(&mut self, cause: FrameError) -> Result<Option<Frame>, LogError> {
        let valid_after = valid_frame_from(&self.path, self.offset, self.len)
            .map_err(LogError::io(&self.path))?;
        if valid_after {
            return Err(self.damaged(cause));
        }
        self.ending = Some(Ending::Torn {
            at: self.offset,
            cause,
        });
        Ok(None)
    }

    fn damaged(&self, cause: FrameError) -> LogError {
        LogError::Damaged {
            path: self.path.clone(),
            offset: self.offset,
            index: self.frames,
            cause,
        }
    }
}

/// True when a frame with a good CRC (whatever its seq) starts anywhere in `[from, len)`.
/// Scans for the magic in chunks and checks each candidate in full.
fn valid_frame_from(path: &Path, from: u64, len: u64) -> io::Result<bool> {
    const CHUNK: usize = 1 << 16;
    let mut file = File::open(path)?;
    let mut buf = vec![0; CHUNK + MAGIC.len() - 1];
    let mut start = from;
    while start < len {
        // The extra bytes past CHUNK let a magic that straddles two chunks be seen whole.
        let want = usize::try_from(len - start).map_or(buf.len(), |left| left.min(buf.len()));
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut buf[..want])?;
        for i in 0..want.min(CHUNK) {
            if buf[i..want].starts_with(&MAGIC) && crc_valid_at(&mut file, start + i as u64, len)? {
                return Ok(true);
            }
        }
        start += CHUNK as u64;
    }
    Ok(false)
}

/// True when the bytes at `at` are a whole frame with a good CRC.
fn crc_valid_at(file: &mut File, at: u64, len: u64) -> io::Result<bool> {
    let available = len - at;
    if available < (HEADER_LEN + TRAILER_LEN) as u64 {
        return Ok(false);
    }
    file.seek(SeekFrom::Start(at))?;
    let mut head = [0; HEADER_LEN];
    file.read_exact(&mut head)?;
    let Ok(header) = Header::parse(&head) else {
        return Ok(false);
    };
    if available < (HEADER_LEN + header.body_len()) as u64 {
        return Ok(false);
    }
    let mut body = vec![0; header.body_len()];
    file.read_exact(&mut body)?;
    Ok(header.verify(&head, &body).is_ok())
}

/// What recovery did to a segment a crash left without its end marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recovery {
    pub path: PathBuf,
    /// Good frames kept, not counting the marker recovery appended.
    pub frames: u64,
    pub last_seq: Option<u64>,
    /// The seq of the appended marker: the last good seq plus one.
    pub marker_seq: u64,
    /// Bytes cut off the end; 0 when the file ended between frames.
    pub cut: u64,
    /// Why the cut bytes were not a frame, when there were any.
    pub cause: Option<FrameError>,
}

/// Closes a segment a crash left open: cuts a torn tail back to the last good frame, then
/// appends a `SegmentEnd` marker with `clean: false` and syncs. A closed segment is left alone
/// (`Ok(None)`). Damage in the middle is refused and the file is left exactly as it was.
///
/// Only call this on a segment no writer has open: [`recover_run`] checks the writer's lock.
///
/// # Errors
/// Returns [`LogError::Damaged`] for mid-file damage, and I/O errors.
pub fn recover(path: &Path) -> Result<Option<Recovery>, LogError> {
    let mut reader = SegmentReader::open(path)?;
    let ending = loop {
        if reader.next_frame()?.is_none() {
            break reader.ending().cloned();
        }
    };
    let (good_end, cause) = match ending {
        Some(Ending::Closed(_)) => return Ok(None),
        Some(Ending::Torn { at, cause }) => (at, Some(cause)),
        Some(Ending::Open) | None => (reader.len, None),
    };
    let marker = SegmentEnd {
        last_seq: reader.last_seq(),
        frames: reader.frames(),
        clean: false,
    };
    let frame = Frame {
        seq: reader.next_seq(),
        kind: Kind::SegmentEnd,
        source_ts_ns: 0,
        capture_ts_ns: now_ns(),
        payload: serde_json::to_string(&marker)
            .map_err(|e| LogError::io(path)(io::Error::other(e)))?,
    };
    let bytes = frame.encode()?;
    let mut file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(LogError::io(path))?;
    file.set_len(good_end).map_err(LogError::io(path))?;
    file.seek(SeekFrom::Start(good_end))
        .map_err(LogError::io(path))?;
    file.write_all(&bytes).map_err(LogError::io(path))?;
    file.sync_all().map_err(LogError::io(path))?;
    Ok(Some(Recovery {
        path: path.to_owned(),
        frames: reader.frames(),
        last_seq: reader.last_seq(),
        marker_seq: frame.seq,
        cut: reader.len - good_end,
        cause,
    }))
}

/// Takes the run's writer lock, failing with [`LogError::InUse`] while a writer holds it. The
/// lock lasts as long as the returned file is open.
///
/// # Errors
/// Returns [`LogError::InUse`] or the I/O error.
pub fn lock(log_dir: &Path) -> Result<File, LogError> {
    let path = log_dir.join(LOCK_FILE);
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&path)
        .map_err(LogError::io(&path))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(LogError::InUse(log_dir.to_owned())),
        Err(TryLockError::Error(error)) => Err(LogError::io(&path)(error)),
    }
}

/// Recovers a crashed run: only its newest segment can be missing its marker. Refuses while a
/// writer still holds the run.
///
/// # Errors
/// Returns [`LogError::InUse`], [`LogError::Damaged`] or the I/O error.
pub fn recover_run(run_dir: &Path) -> Result<Option<Recovery>, LogError> {
    let log_dir = run_dir.join(LOG_DIR);
    let _lock = lock(&log_dir)?;
    match list(&log_dir)?.last() {
        Some((_, newest)) => recover(newest),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_names_carry_their_first_seq() {
        assert_eq!(file_name(42), "00000000000000000042.tplog");
        assert_eq!(
            first_seq_of(Path::new("log/00000000000000000042.tplog")),
            Some(42)
        );
        for other in [
            "log/writer.lock",
            "log/42.tplog",
            "log/0000000000000000004x.tplog",
            "log/00000000000000000042.tplog.tmp",
            "log/00000000000000000042.parquet",
        ] {
            assert_eq!(first_seq_of(Path::new(other)), None, "{other}");
        }
    }
}
