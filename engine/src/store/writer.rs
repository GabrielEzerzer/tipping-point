//! The log writer. One thread owns the open segment file. It writes each frame with a single
//! `write_all` (no `BufWriter`, whose buffer would die with the process), calls `sync_data` at
//! most a second after any write, and moves to a new segment at every UTC hour or once a
//! segment reaches 1 GiB. The live loop hands it entries through a bounded queue and waits when
//! the queue is full, so a stalled disk slows the loop instead of growing memory.
//!
//! The writer assigns each entry its seq in arrival order. The live loop is the only sender and
//! logs each input right before applying it, so seq order is the order the feed changed in.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use super::frame::{Frame, FrameError, Kind, MAX_PAYLOAD, Meta, SegmentEnd};
use super::segment::{self, LOG_DIR};
use super::{LogError, now_ns};

/// When the writer syncs and rotates.
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// A segment holding at least this many bytes is closed before the next frame.
    pub max_segment_bytes: u64,
    /// Segments end at every multiple of this since the Unix epoch: an hour is every UTC hour.
    pub rotate_every: Duration,
    /// The longest a written frame waits for `sync_data`.
    pub sync_every: Duration,
    /// Entries the queue holds before the sender waits.
    pub queue: usize,
    /// Harness only: write every frame in two halves with this pause between, so that a kill
    /// can land in the middle of a frame. `None` in production.
    pub split_frames: Option<Duration>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            max_segment_bytes: 1 << 30,
            rotate_every: Duration::from_hours(1),
            sync_every: Duration::from_secs(1),
            queue: 4096,
            split_frames: None,
        }
    }
}

/// One frame waiting for its seq.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub kind: Kind,
    pub source_ts_ns: i64,
    pub capture_ts_ns: i64,
    pub payload: String,
}

/// What a finished log holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    pub segments: u64,
    /// Every frame written, Meta and markers included: seqs `0..frames`.
    pub frames: u64,
    pub last_seq: u64,
}

/// The handle the live loop holds: a queue into the writer thread.
#[derive(Debug)]
pub struct LogWriter {
    tx: mpsc::Sender<Entry>,
    thread: JoinHandle<Result<Summary, LogError>>,
}

impl LogWriter {
    /// Creates `<run_dir>/log/`, takes the run's writer lock, writes the first segment's Meta
    /// frame and starts the writer thread. A run starts at seq 0 and is never appended to by a
    /// later process.
    ///
    /// # Errors
    /// Returns [`LogError::InUse`] when another writer holds the run, [`LogError::NotEmpty`]
    /// when the run already has a log, and I/O errors.
    pub fn start(run_dir: &Path, meta: Meta, config: LogConfig) -> Result<Self, LogError> {
        let log_dir = run_dir.join(LOG_DIR);
        fs::create_dir_all(&log_dir).map_err(LogError::io(&log_dir))?;
        let lock = segment::lock(&log_dir)?;
        if !segment::list(&log_dir)?.is_empty() {
            return Err(LogError::NotEmpty(log_dir));
        }
        let queue = config.queue.max(1);
        let writer = Writer::open(log_dir, meta, config, lock)?;
        let (tx, rx) = mpsc::channel(queue);
        let thread = thread::Builder::new()
            .name("tplog-writer".to_owned())
            .spawn(move || writer.run(rx))
            .map_err(LogError::io(run_dir))?;
        Ok(Self { tx, thread })
    }

    /// Queues one entry, waiting while the queue is full.
    ///
    /// # Errors
    /// Returns [`LogError::WriterGone`] once the writer has stopped (call [`Self::finish`] for
    /// its error), or [`FrameError::TooLong`] for a payload over the limit.
    pub async fn append(&self, entry: Entry) -> Result<(), LogError> {
        check(&entry)?;
        self.tx.send(entry).await.map_err(|_| LogError::WriterGone)
    }

    /// [`Self::append`] for callers outside an async runtime.
    ///
    /// # Errors
    /// As [`Self::append`].
    pub fn append_blocking(&self, entry: Entry) -> Result<(), LogError> {
        check(&entry)?;
        self.tx
            .blocking_send(entry)
            .map_err(|_| LogError::WriterGone)
    }

    /// Lets the writer drain the queue, close the last segment with a clean marker and sync
    /// it, then returns what the log holds. Blocks until then.
    ///
    /// # Errors
    /// Returns the error that stopped the writer.
    pub fn finish(self) -> Result<Summary, LogError> {
        drop(self.tx);
        self.thread.join().map_err(|_| LogError::WriterPanicked)?
    }
}

fn check(entry: &Entry) -> Result<(), LogError> {
    if entry.payload.len() > MAX_PAYLOAD as usize {
        return Err(FrameError::TooLong(entry.payload.len() as u64).into());
    }
    Ok(())
}

/// The writer thread's state: the open segment and the run's counters.
#[derive(Debug)]
struct Writer {
    log_dir: PathBuf,
    meta: Meta,
    config: LogConfig,
    path: PathBuf,
    file: File,
    next_seq: u64,
    segment_frames: u64,
    segment_bytes: u64,
    segments: u64,
    rotate_at_ns: i64,
    unsynced_since: Option<Instant>,
    _lock: File,
}

impl Writer {
    fn open(log_dir: PathBuf, meta: Meta, config: LogConfig, lock: File) -> Result<Self, LogError> {
        let (path, file) = create_segment(&log_dir, 0)?;
        let mut writer = Self {
            log_dir,
            meta,
            rotate_at_ns: next_boundary(now_ns(), config.rotate_every),
            config,
            path,
            file,
            next_seq: 0,
            segment_frames: 0,
            segment_bytes: 0,
            segments: 1,
            unsynced_since: None,
            _lock: lock,
        };
        writer.put_meta()?;
        Ok(writer)
    }

    /// Writes entries as they come until every sender is gone, syncing and rotating on time in
    /// between. It runs a single-threaded tokio runtime of its own only so that it can wait for
    /// whichever comes first, the next entry or the next deadline.
    fn run(mut self, mut rx: mpsc::Receiver<Entry>) -> Result<Summary, LogError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .map_err(LogError::io(&self.log_dir))?;
        runtime.block_on(async {
            loop {
                let deadline = tokio::time::Instant::from_std(self.deadline());
                match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(Some(entry)) => self.write(entry)?,
                    Ok(None) => return Ok::<(), LogError>(()),
                    Err(_) => self.tick()?,
                }
            }
        })?;
        self.close_segment()?;
        Ok(Summary {
            segments: self.segments,
            frames: self.next_seq,
            last_seq: self.next_seq.saturating_sub(1),
        })
    }

    fn write(&mut self, entry: Entry) -> Result<(), LogError> {
        if self.segment_bytes >= self.config.max_segment_bytes || now_ns() >= self.rotate_at_ns {
            self.rotate()?;
        }
        self.put(
            entry.kind,
            entry.source_ts_ns,
            entry.capture_ts_ns,
            entry.payload,
        )?;
        self.sync_if_due()
    }

    fn tick(&mut self) -> Result<(), LogError> {
        if now_ns() >= self.rotate_at_ns {
            self.rotate()?;
        }
        self.sync_if_due()
    }

    /// The next moment something is due: a sync, or the end of the segment's hour.
    fn deadline(&self) -> Instant {
        let now = Instant::now();
        let to_rotation = u64::try_from(self.rotate_at_ns.saturating_sub(now_ns())).unwrap_or(0);
        let rotation = now + Duration::from_nanos(to_rotation);
        match self.unsynced_since {
            Some(since) => rotation.min(since + self.config.sync_every),
            None => rotation,
        }
    }

    fn sync_if_due(&mut self) -> Result<(), LogError> {
        if let Some(since) = self.unsynced_since
            && since.elapsed() >= self.config.sync_every
        {
            self.file.sync_data().map_err(LogError::io(&self.path))?;
            self.unsynced_since = None;
        }
        Ok(())
    }

    /// Ends the segment: marker, then `sync_all`.
    fn close_segment(&mut self) -> Result<(), LogError> {
        let marker = SegmentEnd {
            last_seq: self.next_seq.checked_sub(1),
            frames: self.segment_frames,
            clean: true,
        };
        self.put(Kind::SegmentEnd, 0, now_ns(), to_json(&marker))?;
        self.file.sync_all().map_err(LogError::io(&self.path))?;
        self.unsynced_since = None;
        Ok(())
    }

    /// Marker and sync on the old segment, then the new segment named after the next seq,
    /// starting with its Meta. A log file is never renamed.
    fn rotate(&mut self) -> Result<(), LogError> {
        self.close_segment()?;
        let (path, file) = create_segment(&self.log_dir, self.next_seq)?;
        self.path = path;
        self.file = file;
        self.segment_frames = 0;
        self.segment_bytes = 0;
        self.segments += 1;
        self.rotate_at_ns = next_boundary(now_ns(), self.config.rotate_every);
        self.put_meta()
    }

    fn put_meta(&mut self) -> Result<(), LogError> {
        self.meta.first_seq = self.next_seq;
        let payload = to_json(&self.meta);
        self.put(Kind::Meta, 0, now_ns(), payload)
    }

    fn put(
        &mut self,
        kind: Kind,
        source_ts_ns: i64,
        capture_ts_ns: i64,
        payload: String,
    ) -> Result<(), LogError> {
        let frame = Frame {
            seq: self.next_seq,
            kind,
            source_ts_ns,
            capture_ts_ns,
            payload,
        };
        let bytes = frame.encode()?;
        let io = LogError::io(&self.path);
        match self.config.split_frames {
            None => self.file.write_all(&bytes).map_err(io)?,
            Some(pause) => {
                let (first, second) = bytes.split_at(bytes.len() / 2);
                self.file
                    .write_all(first)
                    .map_err(LogError::io(&self.path))?;
                thread::sleep(pause);
                self.file.write_all(second).map_err(io)?;
            }
        }
        self.next_seq += 1;
        self.segment_frames += 1;
        self.segment_bytes += bytes.len() as u64;
        self.unsynced_since.get_or_insert_with(Instant::now);
        Ok(())
    }
}

fn create_segment(log_dir: &Path, first_seq: u64) -> Result<(PathBuf, File), LogError> {
    let path = log_dir.join(segment::file_name(first_seq));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(LogError::io(&path))?;
    Ok((path, file))
}

/// The first multiple of `every` since the epoch that is later than `now_ns`.
fn next_boundary(now_ns: i64, every: Duration) -> i64 {
    let every = i64::try_from(every.as_nanos()).unwrap_or(i64::MAX).max(1);
    now_ns
        .div_euclid(every)
        .saturating_add(1)
        .saturating_mul(every)
}

/// The payload of a Meta or `SegmentEnd` frame. Plain structs of strings and numbers always
/// serialize.
fn to_json(value: &impl serde::Serialize) -> String {
    serde_json::to_string(value).expect("a Meta or SegmentEnd payload always serializes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_end_on_whole_multiples_of_the_rotation_period() {
        let hour = Duration::from_hours(1);
        let nanos_per_hour = 3_600_000_000_000;
        assert_eq!(next_boundary(0, hour), nanos_per_hour);
        assert_eq!(next_boundary(nanos_per_hour - 1, hour), nanos_per_hour);
        assert_eq!(next_boundary(nanos_per_hour, hour), 2 * nanos_per_hour);
        assert_eq!(next_boundary(5, Duration::ZERO), 6);
    }
}
