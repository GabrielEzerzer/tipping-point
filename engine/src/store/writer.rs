//! The log writer. One thread owns the open segment file. It writes each frame with a single
//! `write_all` (no `BufWriter`, whose buffer would die with the process) and calls `sync_data`
//! at most a second after any write. The live loop hands it frames through a bounded queue and
//! waits when the queue is full, so a stalled disk slows the loop instead of growing memory.
//!
//! The live loop's handle numbers the frames: [`LogWriter::append`] returns the seq it gave the
//! entry, so the loop knows the seq of every input it applies. The handle also decides when a
//! segment ends, at every UTC hour or once a segment reaches 1 GiB, because it sees the clock and
//! every frame's size; it then queues the segment's end marker and the next segment's Meta with
//! their own seqs. The thread only writes, and refuses any frame whose seq is not the previous
//! one plus one. The live loop is the only sender and logs each input right before applying it,
//! so seq order is the order the feed changed in.

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

/// What the handle queues for the writer thread.
#[derive(Debug)]
enum Job {
    /// Write one frame to the open segment.
    Write(Frame),
    /// Close the open segment with its marker `end`, sync it, then open the segment named after
    /// `meta`'s seq and write `meta` first. One job, so a segment is never left half rotated.
    Rotate { end: Frame, meta: Frame },
}

/// The handle the live loop holds: it numbers frames, decides rotation, and queues both for the
/// writer thread.
#[derive(Debug)]
pub struct LogWriter {
    tx: mpsc::Sender<Job>,
    thread: JoinHandle<Result<Summary, LogError>>,
    meta: Meta,
    max_segment_bytes: u64,
    rotate_every: Duration,
    /// The seq the next frame gets.
    next_seq: u64,
    /// Frames and bytes queued for the open segment, its Meta included.
    segment_frames: u64,
    segment_bytes: u64,
    /// When the open segment's hour ends.
    rotate_at_ns: i64,
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
        let now = now_ns();
        let first = meta_frame(&meta, 0, now);
        let first_len = first.encoded_len() as u64;
        let queue = config.queue.max(1);
        let (max_segment_bytes, rotate_every) = (config.max_segment_bytes, config.rotate_every);
        let writer = Writer::open(log_dir, config, lock, &first)?;
        let (tx, rx) = mpsc::channel(queue);
        let thread = thread::Builder::new()
            .name("tplog-writer".to_owned())
            .spawn(move || writer.run(rx))
            .map_err(LogError::io(run_dir))?;
        Ok(Self {
            tx,
            thread,
            meta,
            max_segment_bytes,
            rotate_every,
            next_seq: 1,
            segment_frames: 1,
            segment_bytes: first_len,
            rotate_at_ns: next_boundary(now, rotate_every),
        })
    }

    /// Numbers one entry and queues it, waiting while the queue is full; first ends the segment
    /// when its hour is over or it has reached its size. Returns the entry's seq.
    ///
    /// # Errors
    /// Returns [`LogError::WriterGone`] once the writer has stopped (call [`Self::finish`] for
    /// its error), or [`FrameError::TooLong`] for a payload over the limit.
    pub async fn append(&mut self, entry: Entry) -> Result<u64, LogError> {
        check(&entry)?;
        let now = now_ns();
        if self.segment_bytes >= self.max_segment_bytes || now >= self.rotate_at_ns {
            let job = self.rotation(now);
            self.tx.send(job).await.map_err(|_| LogError::WriterGone)?;
            self.rotated(now);
        }
        let frame = self.number(entry);
        let len = frame.encoded_len();
        self.tx
            .send(Job::Write(frame))
            .await
            .map_err(|_| LogError::WriterGone)?;
        Ok(self.queued_frame(len))
    }

    /// [`Self::append`] for callers outside an async runtime.
    ///
    /// # Errors
    /// As [`Self::append`].
    pub fn append_blocking(&mut self, entry: Entry) -> Result<u64, LogError> {
        check(&entry)?;
        let now = now_ns();
        if self.segment_bytes >= self.max_segment_bytes || now >= self.rotate_at_ns {
            let job = self.rotation(now);
            self.tx
                .blocking_send(job)
                .map_err(|_| LogError::WriterGone)?;
            self.rotated(now);
        }
        let frame = self.number(entry);
        let len = frame.encoded_len();
        self.tx
            .blocking_send(Job::Write(frame))
            .map_err(|_| LogError::WriterGone)?;
        Ok(self.queued_frame(len))
    }

    /// Ends the segment if its hour is over, even with nothing to append, so an idle hour is
    /// still a segment of its own. The live loop calls this on its timer. True when it rotated.
    ///
    /// # Errors
    /// Returns [`LogError::WriterGone`] once the writer has stopped.
    pub async fn tick(&mut self) -> Result<bool, LogError> {
        let now = now_ns();
        if now < self.rotate_at_ns {
            return Ok(false);
        }
        let job = self.rotation(now);
        self.tx.send(job).await.map_err(|_| LogError::WriterGone)?;
        self.rotated(now);
        Ok(true)
    }

    /// [`Self::tick`] for callers outside an async runtime.
    ///
    /// # Errors
    /// As [`Self::tick`].
    pub fn tick_blocking(&mut self) -> Result<bool, LogError> {
        let now = now_ns();
        if now < self.rotate_at_ns {
            return Ok(false);
        }
        let job = self.rotation(now);
        self.tx
            .blocking_send(job)
            .map_err(|_| LogError::WriterGone)?;
        self.rotated(now);
        Ok(true)
    }

    /// Frames queued and not yet written: how far the disk is behind the loop.
    #[must_use]
    pub fn queued(&self) -> usize {
        self.tx.max_capacity() - self.tx.capacity()
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

    /// The entry as the next frame.
    fn number(&self, entry: Entry) -> Frame {
        Frame {
            seq: self.next_seq,
            kind: entry.kind,
            source_ts_ns: entry.source_ts_ns,
            capture_ts_ns: entry.capture_ts_ns,
            payload: entry.payload,
        }
    }

    /// Counts a queued frame of `len` bytes and returns its seq.
    fn queued_frame(&mut self, len: usize) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.segment_frames += 1;
        self.segment_bytes += len as u64;
        seq
    }

    /// The open segment's marker and the next segment's Meta, with the next two seqs.
    fn rotation(&self, now: i64) -> Job {
        let marker = SegmentEnd {
            last_seq: self.next_seq.checked_sub(1),
            frames: self.segment_frames,
            clean: true,
        };
        let end = Frame {
            seq: self.next_seq,
            kind: Kind::SegmentEnd,
            source_ts_ns: 0,
            capture_ts_ns: now,
            payload: to_json(&marker),
        };
        let meta = meta_frame(&self.meta, self.next_seq + 1, now);
        Job::Rotate { end, meta }
    }

    /// Counts a queued rotation: two frames, and a new segment holding only its Meta.
    fn rotated(&mut self, now: i64) {
        let meta_len = meta_frame(&self.meta, self.next_seq + 1, now).encoded_len();
        self.next_seq += 2;
        self.segment_frames = 1;
        self.segment_bytes = meta_len as u64;
        self.rotate_at_ns = next_boundary(now, self.rotate_every);
    }
}

fn check(entry: &Entry) -> Result<(), LogError> {
    if entry.payload.len() > MAX_PAYLOAD as usize {
        return Err(FrameError::TooLong(entry.payload.len() as u64).into());
    }
    Ok(())
}

/// The Meta frame that opens the segment starting at `seq`.
fn meta_frame(meta: &Meta, seq: u64, now: i64) -> Frame {
    let meta = Meta {
        first_seq: seq,
        ..meta.clone()
    };
    Frame {
        seq,
        kind: Kind::Meta,
        source_ts_ns: 0,
        capture_ts_ns: now,
        payload: to_json(&meta),
    }
}

/// The writer thread's state: the open segment and what it has written.
#[derive(Debug)]
struct Writer {
    log_dir: PathBuf,
    config: LogConfig,
    path: PathBuf,
    file: File,
    /// The seq the next frame must carry.
    next_seq: u64,
    segment_frames: u64,
    segments: u64,
    unsynced_since: Option<Instant>,
    _lock: File,
}

impl Writer {
    /// Creates the first segment and writes its Meta frame.
    fn open(
        log_dir: PathBuf,
        config: LogConfig,
        lock: File,
        meta: &Frame,
    ) -> Result<Self, LogError> {
        let (path, file) = create_segment(&log_dir, 0)?;
        let mut writer = Self {
            log_dir,
            config,
            path,
            file,
            next_seq: 0,
            segment_frames: 0,
            segments: 1,
            unsynced_since: None,
            _lock: lock,
        };
        writer.put(meta)?;
        Ok(writer)
    }

    /// Writes jobs as they come until the handle is gone, syncing on time in between, then
    /// closes the last segment. It runs a single-threaded tokio runtime of its own only so that
    /// it can wait for whichever comes first, the next job or the next sync.
    fn run(mut self, mut rx: mpsc::Receiver<Job>) -> Result<Summary, LogError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .map_err(LogError::io(&self.log_dir))?;
        runtime.block_on(async {
            loop {
                let job = match self.unsynced_since {
                    Some(since) => {
                        let due = tokio::time::Instant::from_std(since + self.config.sync_every);
                        let Ok(job) = tokio::time::timeout_at(due, rx.recv()).await else {
                            self.sync_if_due()?;
                            continue;
                        };
                        job
                    }
                    None => rx.recv().await,
                };
                match job {
                    Some(Job::Write(frame)) => self.put(&frame)?,
                    Some(Job::Rotate { end, meta }) => self.rotate(&end, &meta)?,
                    None => return Ok::<(), LogError>(()),
                }
                self.sync_if_due()?;
            }
        })?;
        self.close_segment()?;
        Ok(Summary {
            segments: self.segments,
            frames: self.next_seq,
            last_seq: self.next_seq.saturating_sub(1),
        })
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

    /// Ends the last segment: its marker, numbered after the last frame, then `sync_all`.
    fn close_segment(&mut self) -> Result<(), LogError> {
        let marker = SegmentEnd {
            last_seq: self.next_seq.checked_sub(1),
            frames: self.segment_frames,
            clean: true,
        };
        let end = Frame {
            seq: self.next_seq,
            kind: Kind::SegmentEnd,
            source_ts_ns: 0,
            capture_ts_ns: now_ns(),
            payload: to_json(&marker),
        };
        self.put(&end)?;
        self.sync_all()
    }

    fn sync_all(&mut self) -> Result<(), LogError> {
        self.file.sync_all().map_err(LogError::io(&self.path))?;
        self.unsynced_since = None;
        Ok(())
    }

    /// The marker and a sync on the old segment, then the new segment named after the Meta's
    /// seq, starting with that Meta. A log file is never renamed.
    fn rotate(&mut self, end: &Frame, meta: &Frame) -> Result<(), LogError> {
        self.put(end)?;
        self.sync_all()?;
        let (path, file) = create_segment(&self.log_dir, meta.seq)?;
        self.path = path;
        self.file = file;
        self.segment_frames = 0;
        self.segments += 1;
        self.put(meta)
    }

    /// Writes one frame, which must carry the next seq.
    fn put(&mut self, frame: &Frame) -> Result<(), LogError> {
        if frame.seq != self.next_seq {
            return Err(LogError::OutOfOrder {
                expected: self.next_seq,
                got: frame.seq,
            });
        }
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
    use std::collections::BTreeMap;

    use super::*;
    use crate::store::run::Producer;

    #[test]
    fn segments_end_on_whole_multiples_of_the_rotation_period() {
        let hour = Duration::from_hours(1);
        let nanos_per_hour = 3_600_000_000_000;
        assert_eq!(next_boundary(0, hour), nanos_per_hour);
        assert_eq!(next_boundary(nanos_per_hour - 1, hour), nanos_per_hour);
        assert_eq!(next_boundary(nanos_per_hour, hour), 2 * nanos_per_hour);
        assert_eq!(next_boundary(5, Duration::ZERO), 6);
    }

    /// The thread refuses a frame that is not numbered right after the last one it wrote, and
    /// stops with that error rather than write a log a reader would refuse.
    #[test]
    fn the_writer_thread_refuses_a_frame_out_of_order() -> Result<(), LogError> {
        let dir = std::env::temp_dir().join(format!("tp-writer-order-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).map_err(LogError::io(&dir))?;
        let meta = Meta {
            run_id: "order".to_owned(),
            first_seq: 0,
            producer: Producer {
                name: "writer-test".to_owned(),
                version: crate::version().to_owned(),
                git_sha: None,
                dirty: None,
            },
            venue: crate::venue::Venue::BinanceUsdm,
            scales: BTreeMap::new(),
        };
        let writer = LogWriter::start(&dir, meta, LogConfig::default())?;
        let skipped = Frame {
            seq: 2,
            kind: Kind::Trade,
            source_ts_ns: 0,
            capture_ts_ns: 0,
            payload: "x".to_owned(),
        };
        writer
            .tx
            .blocking_send(Job::Write(skipped))
            .map_err(|_| LogError::WriterGone)?;
        assert!(matches!(
            writer.finish(),
            Err(LogError::OutOfOrder {
                expected: 1,
                got: 2
            })
        ));
        fs::remove_dir_all(&dir).map_err(LogError::io(&dir))
    }
}
