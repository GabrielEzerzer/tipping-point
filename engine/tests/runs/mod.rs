//! Helpers for the storage tests: scratch folders, logs built byte by byte, and runs built from
//! a live-loop script the way the live loop would have logged it.

#![allow(dead_code)] // each test binary uses a different subset

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use engine::binance::pipeline::entries;
use engine::binance::recording::{Record, read_records};
use engine::store::frame::{Frame, Kind, Meta};
use engine::store::run::{self, BookSample, MarketParams, Producer};
use engine::store::segment::{self, LOG_DIR, SegmentReader};
use engine::store::writer::{LogConfig, LogWriter, Summary};

/// An empty folder of its own under cargo's scratch dir for integration tests.
pub fn fresh_dir(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch dir");
    dir
}

/// A committed synthetic fixture.
pub fn synthetic(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/synthetic")
        .join(name)
}

/// The Meta every test log starts with.
pub fn meta() -> Meta {
    Meta {
        run_id: "test".to_owned(),
        first_seq: 0,
        producer: Producer {
            name: "engine-tests".to_owned(),
            version: engine::version().to_owned(),
            git_sha: None,
            dirty: None,
        },
        scales: BTreeMap::new(),
    }
}

/// A frame as a test writes it by hand.
pub fn frame(seq: u64, kind: Kind, payload: &str) -> Frame {
    Frame {
        seq,
        kind,
        source_ts_ns: 1_700_000_000_000_000_000 + i64::try_from(seq).expect("a small seq"),
        capture_ts_ns: 1_700_000_000_000_500_000 + i64::try_from(seq).expect("a small seq"),
        payload: payload.to_owned(),
    }
}

/// The frames encoded back to back, with the byte offset where each one ends.
pub fn encode_all(frames: &[Frame]) -> (Vec<u8>, Vec<usize>) {
    let mut bytes = Vec::new();
    let mut ends = Vec::new();
    for frame in frames {
        bytes.extend(frame.encode().expect("a frame under the size limit"));
        ends.push(bytes.len());
    }
    (bytes, ends)
}

/// Writes `bytes` as the first segment of a log folder and returns its path.
pub fn segment_file(dir: &Path, bytes: &[u8]) -> PathBuf {
    let path = dir.join(segment::file_name(0));
    fs::write(&path, bytes).expect("a segment file");
    path
}

/// Every good frame of a segment, and how it ended (`None` when the reader refused it).
pub fn read_segment(path: &Path) -> (Vec<Frame>, Option<segment::Ending>) {
    let mut reader = SegmentReader::open(path).expect("a segment");
    let mut frames = Vec::new();
    loop {
        match reader.next_frame() {
            Ok(Some(frame)) => frames.push(frame),
            Ok(None) => return (frames, reader.ending().cloned()),
            Err(_) => return (frames, None),
        }
    }
}

/// Every frame of a run's log, in order.
pub fn log_frames(run_dir: &Path) -> Vec<Frame> {
    let mut frames = Vec::new();
    for (_, path) in segment::list(&run_dir.join(LOG_DIR)).expect("segments") {
        let mut reader = SegmentReader::open(&path).expect("a segment");
        while let Some(frame) = reader.next_frame().expect("a good frame") {
            frames.push(frame);
        }
    }
    frames
}

/// A run built from a live-loop script: every message turned into log entries, in order, as
/// the live loop logs them. Returns the run folder and what the log holds.
pub fn write_run(script: &Path, dir: &Path, log: LogConfig) -> (PathBuf, Summary) {
    let file = File::open(script).expect("a script");
    let mut records = read_records(BufReader::new(file))
        .expect("records")
        .into_iter();
    let Some(Record::Scales { scales }) = records.next() else {
        panic!("a script starts with its scales");
    };
    let run_dir = dir.join(run::MARKET).join("20231114T221320Z-synthetic");
    let params = MarketParams {
        symbols: scales.keys().cloned().collect(),
        scales,
        streams: vec!["depth@100ms".to_owned(), "aggTrade".to_owned()],
        book_sample: BookSample::default(),
    };
    let manifest = run::create(&run_dir, params).expect("a run folder");
    let writer = LogWriter::start(&run_dir, Meta::for_run(&manifest), log).expect("a writer");
    for record in records {
        let message = record
            .into_message()
            .expect("only the first line holds scales");
        for entry in entries(message) {
            writer.append_blocking(entry).expect("an append");
        }
    }
    let summary = writer.finish().expect("a clean finish");
    run::complete(&run_dir, &summary).expect("complete.json");
    (run_dir, summary)
}
