//! Recovery after a crash: a torn tail is cut back to the last good frame and the segment is
//! closed with a marker; damage in the middle is refused and the file left exactly as it was.

mod runs;

use std::error::Error;
use std::fs;

use engine::store::LogError;
use engine::store::frame::{Frame, Kind, SegmentEnd};
use engine::store::segment::{Ending, recover, recover_run};
use engine::store::writer::{Entry, LogConfig, LogWriter};
use runs::{encode_all, frame, fresh_dir, meta, read_segment, segment_file, synthetic};

type Outcome = Result<(), Box<dyn Error>>;

/// A segment's frames as a crash leaves them: no end marker.
fn open_segment() -> Vec<Frame> {
    vec![
        frame(0, Kind::Meta, r#"{"run_id":"r"}"#),
        frame(1, Kind::Depth, r#"{"stream":"btcusdt@depth@100ms"}"#),
        frame(2, Kind::Trade, &"t".repeat(64)),
        frame(3, Kind::Snapshot, r#"{"symbol":"BTCUSDT","body":"{}"}"#),
        frame(
            4,
            Kind::Control,
            r#"{"type":"connected","stream":"depth","attempt":1}"#,
        ),
        frame(5, Kind::Depth, ""),
    ]
}

fn closed_by_recovery(kept: u64) -> Ending {
    Ending::Closed(SegmentEnd {
        last_seq: kept.checked_sub(1),
        frames: kept,
        clean: false,
    })
}

/// Cut the file at every byte offset: recovery keeps exactly the frames that end at or before
/// the cut, and its marker takes the next seq, so a writer would carry on at last seq + 1.
#[test]
fn a_torn_tail_at_every_offset_recovers_to_the_frames_before_it() -> Outcome {
    let frames = open_segment();
    let (bytes, ends) = encode_all(&frames);
    let dir = fresh_dir("torn");
    for cut in 0..=bytes.len() {
        let path = segment_file(&dir, &bytes[..cut]);
        let kept = ends.iter().filter(|&&end| end <= cut).count();
        let good_end = if kept == 0 { 0 } else { ends[kept - 1] };

        let recovery = recover(&path)?.expect("an unclosed segment to close");
        assert_eq!(recovery.frames, kept as u64, "cut {cut}");
        assert_eq!(recovery.marker_seq, kept as u64, "cut {cut}");
        assert_eq!(recovery.cut, (cut - good_end) as u64, "cut {cut}");

        let (read, ending) = read_segment(&path);
        assert_eq!(read.len(), kept + 1, "cut {cut}");
        assert_eq!(&read[..kept], &frames[..kept], "cut {cut}");
        assert_eq!(read[kept].seq, kept as u64);
        assert_eq!(ending, Some(closed_by_recovery(kept as u64)), "cut {cut}");
        assert_eq!(recover(&path)?, None, "a recovered segment is closed");
    }
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// What an OS crash can leave after the last good frame: zeros, or junk.
#[test]
fn junk_after_the_last_good_frame_is_cut_off() -> Outcome {
    let frames = open_segment();
    let (bytes, _) = encode_all(&frames);
    let dir = fresh_dir("junk");
    let mut noise = 0x2545_F491_4F6C_DD1D_u64;
    let mut junk = || {
        noise ^= noise << 13;
        noise ^= noise >> 7;
        noise ^= noise << 17;
        noise.to_le_bytes()[0]
    };
    for len in [1, 7, 35, 36, 39, 40, 41, 300, 4096] {
        for tail in [vec![0; len], (0..len).map(|_| junk()).collect::<Vec<u8>>()] {
            let mut torn = bytes.clone();
            torn.extend_from_slice(&tail);
            let path = segment_file(&dir, &torn);
            let recovery = recover(&path)?.expect("a junk tail to cut");
            assert_eq!(recovery.cut, len as u64);
            assert_eq!(recovery.frames, frames.len() as u64);
            let (read, ending) = read_segment(&path);
            assert_eq!(&read[..frames.len()], &frames[..]);
            assert_eq!(ending, Some(closed_by_recovery(frames.len() as u64)));
        }
    }
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A byte flipped in a frame that has good frames after it is not a torn write: recovery names
/// the frame and its offset and changes nothing.
#[test]
fn damage_in_the_middle_is_refused_and_changes_nothing() -> Outcome {
    let frames = open_segment();
    let (mut bytes, ends) = encode_all(&frames);
    bytes[ends[1] + 40] ^= 0x20; // inside frame 2's payload
    let dir = fresh_dir("damage");
    let path = segment_file(&dir, &bytes);
    match recover(&path) {
        Err(LogError::Damaged { index, offset, .. }) => {
            assert_eq!((index, offset), (2, ends[1] as u64));
        }
        other => panic!("expected damage, got {other:?}"),
    }
    assert_eq!(fs::read(&path)?, bytes, "the file is untouched");
    let (read, ending) = read_segment(&path);
    assert_eq!(&read[..], &frames[..2]);
    assert_eq!(ending, None, "the reader refuses it too");
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn a_closed_segment_is_left_alone_and_bytes_after_its_marker_are_damage() -> Outcome {
    let mut frames = open_segment();
    let marker = SegmentEnd {
        last_seq: Some(5),
        frames: 6,
        clean: true,
    };
    frames.push(frame(6, Kind::SegmentEnd, &serde_json::to_string(&marker)?));
    let (mut bytes, _) = encode_all(&frames);
    let dir = fresh_dir("closed");
    let path = segment_file(&dir, &bytes);
    assert_eq!(recover(&path)?, None);
    assert_eq!(fs::read(&path)?, bytes);
    assert_eq!(
        read_segment(&path),
        (frames.clone(), Some(Ending::Closed(marker)))
    );

    bytes.extend_from_slice(b"stray");
    let path = segment_file(&dir, &bytes);
    assert!(matches!(
        recover(&path),
        Err(LogError::Damaged { index: 7, .. })
    ));
    assert_eq!(fs::read(&path)?, bytes);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A run being written cannot be recovered under the writer; once the writer has closed it
/// there is nothing to recover.
#[test]
fn recovery_waits_for_the_writer_to_let_go() -> Outcome {
    let dir = fresh_dir("in-use");
    let writer = LogWriter::start(&dir, meta(), LogConfig::default())?;
    writer.append_blocking(Entry {
        kind: Kind::Depth,
        source_ts_ns: 0,
        capture_ts_ns: 1,
        payload: "x".to_owned(),
    })?;
    assert!(matches!(recover_run(&dir), Err(LogError::InUse(_))));
    assert!(matches!(
        LogWriter::start(&dir, meta(), LogConfig::default()),
        Err(LogError::InUse(_))
    ));
    writer.finish()?;
    assert_eq!(recover_run(&dir)?, None);
    assert!(
        matches!(
            LogWriter::start(&dir, meta(), LogConfig::default()),
            Err(LogError::NotEmpty(_))
        ),
        "a new run never appends to an old log"
    );
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// The frames of the committed golden segment, written by an implementation independent of
/// this crate. Any change to the format must bump the version and the golden file with it.
fn golden_frames() -> Vec<Frame> {
    let at = |source_ts_ns: i64, capture_ts_ns: i64, seq: u64, kind: Kind, payload: &str| Frame {
        seq,
        kind,
        source_ts_ns,
        capture_ts_ns,
        payload: payload.to_owned(),
    };
    vec![
        at(
            0,
            1_700_000_000_000_000_000,
            0,
            Kind::Meta,
            r#"{"run_id":"golden","first_seq":0,"producer":{"name":"engine","version":"0.1.0","git_sha":null,"dirty":null},"scales":{"BTCUSDT":{"price_decimals":1,"qty_decimals":3}}}"#,
        ),
        at(
            1_700_000_000_100_000_000,
            1_700_000_000_101_000_000,
            1,
            Kind::Depth,
            r#"{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1700000000100,"T":1700000000098,"s":"BTCUSDT","U":100,"u":102,"pu":99,"b":[["50000.0","1.000"]],"a":[]}}"#,
        ),
        at(
            1_700_000_000_250_000_000,
            1_700_000_000_252_000_000,
            2,
            Kind::Snapshot,
            r#"{"symbol":"BTCUSDT","body":"{\"lastUpdateId\":101,\"E\":1700000000250,\"bids\":[],\"asks\":[]}"}"#,
        ),
        at(
            0,
            1_700_000_000_300_000_000,
            3,
            Kind::Control,
            r#"{"type":"disconnected","stream":"depth","reason":"stream ended","dropped_before":0}"#,
        ),
        at(
            1_700_000_000_350_000_000,
            1_700_000_000_351_000_000,
            4,
            Kind::Trade,
            r#"{"stream":"btcusdt@aggTrade","data":{"e":"aggTrade","E":1700000000350,"a":7,"s":"BTCUSDT","p":"50000.5","q":"0.010","f":70,"l":71,"T":1700000000349,"m":true}}"#,
        ),
        at(
            0,
            1_700_000_000_400_000_000,
            5,
            Kind::SegmentEnd,
            r#"{"last_seq":4,"frames":5,"clean":true}"#,
        ),
    ]
}

#[test]
fn the_golden_segment_decodes_to_its_known_frames_and_back() -> Outcome {
    let path = synthetic("golden/00000000000000000000.tplog");
    let (read, ending) = read_segment(&path);
    assert_eq!(read, golden_frames());
    let marker = SegmentEnd {
        last_seq: Some(4),
        frames: 5,
        clean: true,
    };
    assert_eq!(ending, Some(Ending::Closed(marker)));
    assert_eq!(encode_all(&golden_frames()).0, fs::read(&path)?);
    Ok(())
}
