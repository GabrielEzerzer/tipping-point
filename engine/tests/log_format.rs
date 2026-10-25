//! The log format and the writer, property-tested: whatever goes in comes back out frame for
//! frame, and a single flipped bit anywhere is caught, never read back as data.

mod runs;

use std::fs;

use engine::store::LogError;
use engine::store::frame::{
    Frame, FrameError, HEADER_LEN, Header, Kind, MAGIC, MAX_PAYLOAD, SegmentEnd,
};
use engine::store::segment::{self, Ending, LOG_DIR, SegmentReader, recover};
use engine::store::writer::{Entry, LogConfig, LogWriter};
use proptest::prelude::*;
use runs::{encode_all, fresh_dir, meta, read_segment, segment_file};

fn kind() -> impl Strategy<Value = Kind> {
    prop_oneof![
        Just(Kind::Meta),
        Just(Kind::Depth),
        Just(Kind::Trade),
        Just(Kind::Snapshot),
        Just(Kind::Control),
        Just(Kind::SegmentEnd),
    ]
}

/// The kinds the live loop appends; the writer adds Meta and `SegmentEnd` itself.
fn input_kind() -> impl Strategy<Value = Kind> {
    prop_oneof![
        Just(Kind::Depth),
        Just(Kind::Trade),
        Just(Kind::Snapshot),
        Just(Kind::Control),
    ]
}

fn any_frame() -> impl Strategy<Value = Frame> {
    (any::<u64>(), kind(), any::<i64>(), any::<i64>(), ".{0,64}").prop_map(
        |(seq, kind, source_ts_ns, capture_ts_ns, payload)| Frame {
            seq,
            kind,
            source_ts_ns,
            capture_ts_ns,
            payload,
        },
    )
}

/// A segment's worth of input frames, seqs from 0.
fn segment_frames() -> impl Strategy<Value = Vec<Frame>> {
    prop::collection::vec((input_kind(), any::<i64>(), any::<i64>(), ".{0,48}"), 2..8).prop_map(
        |raw| {
            (0u64..)
                .zip(raw)
                .map(
                    |(seq, (kind, source_ts_ns, capture_ts_ns, payload))| Frame {
                        seq,
                        kind,
                        source_ts_ns,
                        capture_ts_ns,
                        payload,
                    },
                )
                .collect()
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn a_frame_decodes_to_what_was_encoded(frame in any_frame()) {
        let bytes = frame.encode()?;
        prop_assert_eq!(bytes.len(), frame.encoded_len());
        let (head, body) = bytes.split_first_chunk::<HEADER_LEN>().expect("a whole header");
        let header = Header::parse(head)?;
        prop_assert_eq!(HEADER_LEN + header.body_len(), bytes.len());
        prop_assert_eq!(header.finish(head, body.to_vec())?, frame);
    }

    /// Any length over the limit is refused from the header alone, before anything is read or
    /// allocated for the payload.
    #[test]
    fn an_oversized_length_is_refused_up_front(len in (MAX_PAYLOAD + 1)..=u32::MAX) {
        let mut head = [0; HEADER_LEN];
        head[..4].copy_from_slice(&MAGIC);
        head[4] = 1;
        head[32..36].copy_from_slice(&len.to_le_bytes());
        prop_assert_eq!(Header::parse(&head), Err(FrameError::TooLong(u64::from(len))));
    }

    /// Flip any one bit of a segment: every frame before the hit one still reads back exactly,
    /// the hit frame never comes back as data, nor does anything after it. A hit in the last
    /// frame is a torn tail that recovery cuts off; a hit anywhere else is damage that recovery
    /// refuses, leaving the file as it was.
    #[test]
    fn a_single_flipped_bit_is_caught(
        frames in segment_frames(),
        at in any::<prop::sample::Index>(),
        bit in 0u8..8,
    ) {
        let (mut bytes, ends) = encode_all(&frames);
        let flip = at.index(bytes.len());
        bytes[flip] ^= 1 << bit;
        let hit = ends.iter().position(|&end| flip < end).expect("the flip is inside a frame");
        let start = if hit == 0 { 0 } else { ends[hit - 1] };

        let dir = fresh_dir("bit-flip");
        let path = segment_file(&dir, &bytes);
        let (read, ending) = read_segment(&path);
        prop_assert_eq!(&read[..], &frames[..hit]);

        if hit + 1 == frames.len() {
            prop_assert!(
                matches!(&ending, Some(Ending::Torn { at, .. }) if *at == start as u64),
                "{ending:?}"
            );
            let recovery = recover(&path)?.expect("a torn tail to cut");
            prop_assert_eq!(recovery.frames, hit as u64);
            let (kept, closed) = read_segment(&path);
            prop_assert_eq!(&kept[..hit], &frames[..hit]);
            prop_assert!(
                matches!(&closed, Some(Ending::Closed(SegmentEnd { clean: false, .. }))),
                "{closed:?}"
            );
        } else {
            let refused = recover(&path);
            prop_assert!(
                matches!(&refused, Err(LogError::Damaged { index, offset, .. })
                    if *index == hit as u64 && *offset == start as u64),
                "{refused:?}"
            );
            prop_assert_eq!(fs::read(&path)?, bytes);
        }
        fs::remove_dir_all(&dir)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Entries through the writer come back from the reader frame for frame, in order, however
    /// small the segments: every segment opens with its Meta, closes with a clean marker that
    /// counts it, and the seq never skips across them.
    #[test]
    fn a_log_reads_back_exactly_what_was_written(
        entries in prop::collection::vec((input_kind(), any::<i64>(), any::<i64>(), ".{0,200}"), 0..40),
        max_segment_bytes in 64u64..4096,
    ) {
        let dir = fresh_dir("round-trip");
        let config = LogConfig { max_segment_bytes, ..LogConfig::default() };
        let mut writer = LogWriter::start(&dir, meta(), config)?;
        let runtime = runs::runtime();
        let mut given = Vec::new();
        for (kind, source_ts_ns, capture_ts_ns, payload) in &entries {
            given.push(runtime.block_on(writer.append(Entry {
                kind: *kind,
                source_ts_ns: *source_ts_ns,
                capture_ts_ns: *capture_ts_ns,
                payload: payload.clone(),
            }))?);
        }
        let summary = writer.finish()?;

        let mut seq = 0;
        let mut data = Vec::new();
        let mut stored = Vec::new();
        let segments = segment::list(&dir.join(LOG_DIR))?;
        prop_assert_eq!(segments.len() as u64, summary.segments);
        for (first_seq, path) in &segments {
            let mut reader = SegmentReader::open(path)?;
            let mut in_segment = 0;
            while let Some(frame) = reader.next_frame()? {
                prop_assert_eq!(frame.seq, seq);
                if in_segment == 0 {
                    prop_assert_eq!(frame.kind, Kind::Meta);
                    prop_assert_eq!(frame.seq, *first_seq);
                }
                match frame.kind {
                    Kind::Meta | Kind::SegmentEnd => {}
                    kind => {
                        stored.push(frame.seq);
                        data.push((kind, frame.source_ts_ns, frame.capture_ts_ns, frame.payload));
                    }
                }
                seq += 1;
                in_segment += 1;
            }
            let closed = SegmentEnd { last_seq: Some(seq - 2), frames: in_segment - 1, clean: true };
            prop_assert_eq!(reader.ending(), Some(&Ending::Closed(closed)));
        }
        prop_assert_eq!(seq, summary.frames);
        prop_assert_eq!(data, entries);
        prop_assert_eq!(given, stored, "append returns the seq the entry is stored under");
        fs::remove_dir_all(&dir)?;
    }
}

/// Segments end on the clock too, even with nothing to write: the live loop ticks the writer on
/// its timer, and each piece of an idle hour becomes a segment of its own, Meta and marker. The
/// seqs the handle hands out run on across those markers. Driven through the live loop's own
/// `append` and `tick` (#74).
#[test]
fn segments_also_end_on_the_clock() -> Result<(), Box<dyn std::error::Error>> {
    let dir = fresh_dir("clock");
    let config = LogConfig {
        rotate_every: std::time::Duration::from_millis(40),
        ..LogConfig::default()
    };
    let entry = |payload: &str| Entry {
        kind: Kind::Trade,
        source_ts_ns: 0,
        capture_ts_ns: 0,
        payload: payload.to_owned(),
    };
    let mut writer = LogWriter::start(&dir, meta(), config)?;
    let runtime = runs::runtime();
    let before = runtime.block_on(writer.append(entry("before")))?;
    let mut ticks = 0;
    for _ in 0..15 {
        std::thread::sleep(std::time::Duration::from_millis(10));
        ticks += u64::from(runtime.block_on(writer.tick())?);
    }
    let after = runtime.block_on(writer.append(entry("after")))?;
    let summary = writer.finish()?;
    assert!(summary.segments >= 3, "{summary:?}");
    assert!(ticks >= 2, "{ticks} idle rotations");
    // Seq 0 is the first Meta and every rotation takes two seqs, its marker and the next Meta.
    // An hour can end before the first entry too, so only the total is fixed.
    assert_eq!(before % 2, 1, "{before}: Meta, then rotations in pairs");
    assert_eq!(
        after,
        2 * summary.segments,
        "both entries and every rotation's two frames: {summary:?}"
    );

    let mut holder = Vec::new();
    for (first_seq, path) in segment::list(&dir.join(LOG_DIR))? {
        let mut reader = SegmentReader::open(&path)?;
        while let Some(frame) = reader.next_frame()? {
            if frame.kind == Kind::Trade {
                holder.push((frame.payload, first_seq));
            }
        }
        assert!(matches!(reader.ending(), Some(Ending::Closed(_))));
    }
    assert_eq!(holder.len(), 2);
    assert_ne!(
        holder[0].1, holder[1].1,
        "an hour boundary passed between them"
    );
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn a_header_claiming_more_than_the_file_holds_is_a_torn_tail()
-> Result<(), Box<dyn std::error::Error>> {
    let frames = vec![
        runs::frame(0, Kind::Meta, "{}"),
        runs::frame(1, Kind::Depth, "abc"),
    ];
    let (mut bytes, ends) = encode_all(&frames);
    // A whole header for frame 2 that promises a megabyte nobody wrote.
    let mut promised = runs::frame(2, Kind::Depth, "").encode()?;
    promised.truncate(HEADER_LEN);
    promised[32..36].copy_from_slice(&(1u32 << 20).to_le_bytes());
    bytes.extend_from_slice(&promised);
    let dir = fresh_dir("promised");
    let (read, ending) = read_segment(&segment_file(&dir, &bytes));
    assert_eq!(read, frames);
    assert!(
        matches!(&ending, Some(Ending::Torn { at, cause: FrameError::Truncated { .. } }) if *at == ends[1] as u64),
        "{ending:?}"
    );
    fs::remove_dir_all(&dir)?;
    Ok(())
}
