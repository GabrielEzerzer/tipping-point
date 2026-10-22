//! Replay: a run's frames, from its log or from its Parquet through `DuckDB`, driven through the
//! same [`Pipeline`] the live loop runs, so a replay rebuilds the books the live loop had and
//! raises the events it raised. Frames go in seq order from seq 0; the first is the run's Meta,
//! whose scales build the feed. A time-window replay still starts at seq 0, because a book
//! needs its snapshot and bridge, and only reports events for inputs captured in the window.

use std::ops::Range;
use std::path::Path;

use super::frame::{Frame, Kind, Meta};
use super::segment::{self, Ending, LOG_DIR, SegmentReader};
use super::{StoreError, query};
use crate::binance::feed::Feed;
use crate::binance::live::LiveEvent;
use crate::binance::pipeline::{Applied, Pipeline, Replayed, Report};

/// A replay in progress: push frames in seq order, starting with the run's Meta at seq 0.
#[derive(Debug)]
pub struct Replay {
    pipeline: Option<Pipeline>,
    report: Report,
    previous: Option<u64>,
    window: Option<Range<i64>>,
}

impl Replay {
    /// A replay reporting events for every input, or only for those captured in `window`.
    #[must_use]
    pub fn new(window: Option<Range<i64>>) -> Self {
        Self {
            pipeline: None,
            report: Report::default(),
            previous: None,
            window,
        }
    }

    /// The pipeline, once the Meta frame has built it.
    #[must_use]
    pub fn pipeline(&self) -> Option<&Pipeline> {
        self.pipeline.as_ref()
    }

    /// Applies the next frame.
    ///
    /// # Errors
    /// Returns an error when the frames do not start with a Meta at seq 0, skip or repeat a
    /// seq, or the Meta does not parse. Inputs the feed cannot use are not errors: they are
    /// counted in the report, as the live loop shrugs them off.
    pub fn push(
        &mut self,
        frame: &Frame,
        on_event: &mut impl FnMut(LiveEvent<'_>),
    ) -> Result<(), StoreError> {
        match self.previous {
            None if frame.seq != 0 || frame.kind != Kind::Meta => {
                return Err(StoreError::MissingMeta {
                    kind: frame.kind,
                    seq: frame.seq,
                });
            }
            Some(previous) if frame.seq != previous + 1 => {
                return Err(StoreError::SeqGap {
                    previous,
                    got: frame.seq,
                });
            }
            _ => self.previous = Some(frame.seq),
        }
        let Some(pipeline) = self.pipeline.as_mut() else {
            let meta: Meta =
                serde_json::from_str(&frame.payload).map_err(|error| StoreError::Payload {
                    what: "Meta",
                    seq: frame.seq,
                    error,
                })?;
            self.pipeline = Some(Pipeline::new(Feed::new(meta.scales)));
            self.report.note(frame.seq, &Applied::NotAnInput);
            return Ok(());
        };
        let inside = self
            .window
            .as_ref()
            .is_none_or(|window| window.contains(&frame.capture_ts_ns));
        let applied = if inside {
            pipeline.apply(frame.kind, &frame.payload, on_event)
        } else {
            pipeline.apply(frame.kind, &frame.payload, &mut |_| {})
        };
        self.report.note(frame.seq, &applied);
        Ok(())
    }

    /// The pipeline in its final state, and the report.
    ///
    /// # Errors
    /// Returns [`StoreError::NoFrames`] when no frame was ever pushed.
    pub fn finish(self) -> Result<Replayed, StoreError> {
        let pipeline = self.pipeline.ok_or(StoreError::NoFrames)?;
        Ok(Replayed {
            pipeline,
            report: self.report,
        })
    }
}

/// Replays a run from its log, segment by segment. The newest segment may be open or torn (a
/// run still being written, or one that crashed and has not been recovered): its good frames
/// are replayed and the report says how it ended. Every other segment must be closed.
///
/// # Errors
/// Returns an error for mid-file damage, an unclosed segment that is not the newest, a seq
/// that skips, or a run without a log.
pub fn replay_log(
    run_dir: &Path,
    window: Option<Range<i64>>,
    on_event: &mut impl FnMut(LiveEvent<'_>),
) -> Result<Replayed, StoreError> {
    let segments = segment::list(&run_dir.join(LOG_DIR))?;
    let mut replay = Replay::new(window);
    let mut unclosed = None;
    for (index, (_, path)) in segments.iter().enumerate() {
        let mut reader = SegmentReader::open(path)?;
        while let Some(frame) = reader.next_frame()? {
            replay.push(&frame, on_event)?;
        }
        match reader.ending() {
            Some(Ending::Closed(_)) => {}
            Some(Ending::Torn { at, cause }) if index + 1 == segments.len() => {
                unclosed = Some(format!("{} is torn at byte {at}: {cause}", path.display()));
            }
            Some(Ending::Open) if index + 1 == segments.len() => {
                unclosed = Some(format!("{} has no end marker yet", path.display()));
            }
            _ => return Err(StoreError::Unclosed(path.clone())),
        }
    }
    let mut replayed = replay.finish()?;
    replayed.report.unclosed = unclosed;
    Ok(replayed)
}

/// Replays a run from its `frames` Parquet through `DuckDB`, one segment's file at a time in seq
/// order, each read `ORDER BY seq`. Covers the segments rotate has written.
///
/// # Errors
/// Returns `DuckDB` errors, a seq that skips, or a run without `frames` files.
pub fn replay_parquet(
    run_dir: &Path,
    window: Option<Range<i64>>,
    on_event: &mut impl FnMut(LiveEvent<'_>),
) -> Result<Replayed, StoreError> {
    let conn = query::connect()?;
    let mut replay = Replay::new(window);
    for path in query::table_files(run_dir, "frames")? {
        query::for_each_frame(&conn, &path, |frame| replay.push(&frame, on_event))?;
    }
    replay.finish()
}
