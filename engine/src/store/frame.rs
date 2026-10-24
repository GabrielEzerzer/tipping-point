//! The log's frame format: a 36-byte little-endian header, the payload, and a CRC-32C trailer.
//!
//! | offset | bytes | field |
//! |---|---|---|
//! | 0 | 4 | magic `FF 54 50 4C` (`\xFF` + `TPL`): 0xFF never occurs in UTF-8, so no payload text can look like a frame start |
//! | 4 | 1 | version, 1 |
//! | 5 | 1 | kind ([`Kind`]) |
//! | 6 | 2 | reserved, 0 |
//! | 8 | 8 | seq: counts every frame of a run from 0, across segments |
//! | 16 | 8 | `source_ts_ns`: exchange event time, 0 if the input has none |
//! | 24 | 8 | `capture_ts_ns`: wall clock when the input arrived |
//! | 32 | 4 | payload length, at most [`MAX_PAYLOAD`] |
//! | 36 | n | payload, UTF-8 |
//! | 36 + n | 4 | CRC-32C over bytes `[0, 36 + n)` |

use std::collections::BTreeMap;

use crc::{CRC_32_ISCSI, Crc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::run::{Manifest, Producer};
use crate::market::{Scales, Symbol};
use crate::venue::Venue;

/// The four bytes every frame starts with.
pub const MAGIC: [u8; 4] = [0xFF, b'T', b'P', b'L'];
/// The format version this code writes and reads; any change to the layout bumps it.
pub const VERSION: u8 = 1;
/// Bytes before the payload.
pub const HEADER_LEN: usize = 36;
/// Bytes after the payload: the CRC.
pub const TRAILER_LEN: usize = 4;
/// A larger length is corruption, refused before anything is allocated for it.
pub const MAX_PAYLOAD: u32 = 16 * 1024 * 1024;

const CASTAGNOLI: Crc<u32> = Crc::<u32>::new(&CRC_32_ISCSI);

/// CRC-32C (Castagnoli), the checksum each frame carries.
#[must_use]
pub fn crc32c(bytes: &[u8]) -> u32 {
    CASTAGNOLI.checksum(bytes)
}

/// What a frame holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// First in every segment: the run, the segment's first seq, the producer and the scales.
    Meta,
    /// Raw text from the depth socket.
    Depth,
    /// Raw text from the trade socket. Separate from depth because the live loop reacts
    /// differently to unreadable text on each socket.
    Trade,
    /// A REST depth snapshot: `{"symbol", "body"}`, the body as received.
    Snapshot,
    /// Something the sockets or REST did that the feed reacts to: a connect, a disconnect,
    /// dropped frames, a failed snapshot.
    Control,
    /// Last in every closed segment: `{"last_seq", "frames", "clean"}`.
    SegmentEnd,
}

impl Kind {
    /// The byte this kind is stored as.
    #[must_use]
    pub fn code(self) -> u8 {
        match self {
            Self::Meta => 0,
            Self::Depth => 1,
            Self::Trade => 2,
            Self::Snapshot => 3,
            Self::Control => 4,
            Self::SegmentEnd => 5,
        }
    }

    /// The kind stored as `code`, if any.
    #[must_use]
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => Self::Meta,
            1 => Self::Depth,
            2 => Self::Trade,
            3 => Self::Snapshot,
            4 => Self::Control,
            5 => Self::SegmentEnd,
            _ => return None,
        })
    }
}

/// One frame of the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub seq: u64,
    pub kind: Kind,
    pub source_ts_ns: i64,
    pub capture_ts_ns: i64,
    pub payload: String,
}

/// Why bytes are not a valid frame.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrameError {
    #[error("needs {needed} more bytes, only {available} left")]
    Truncated { needed: u64, available: u64 },
    #[error("bad magic {0:02x?}")]
    BadMagic([u8; 4]),
    #[error("unknown version {0}")]
    BadVersion(u8),
    #[error("unknown kind {0}")]
    BadKind(u8),
    #[error("reserved bytes are {0}, not 0")]
    BadReserved(u16),
    #[error("payload length {0} is over the {MAX_PAYLOAD}-byte limit")]
    TooLong(u64),
    #[error("crc mismatch: stored {stored:08x}, computed {computed:08x}")]
    BadCrc { stored: u32, computed: u32 },
    #[error("payload is not UTF-8")]
    NotUtf8,
    #[error("seq {got} where {expected} was due")]
    SeqGap { expected: u64, got: u64 },
    #[error("segment-end marker does not parse: {0}")]
    BadMarker(String),
    #[error("bytes after the segment-end marker")]
    AfterEnd,
}

/// The fixed-size part of a frame, checked field by field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub kind: Kind,
    pub seq: u64,
    pub source_ts_ns: i64,
    pub capture_ts_ns: i64,
    pub payload_len: u32,
}

impl Header {
    /// Parses and checks a header: magic, version, kind, reserved bytes and length limit.
    ///
    /// # Errors
    /// Returns the first field that is wrong.
    pub fn parse(bytes: &[u8; HEADER_LEN]) -> Result<Self, FrameError> {
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if magic != MAGIC {
            return Err(FrameError::BadMagic(magic));
        }
        if bytes[4] != VERSION {
            return Err(FrameError::BadVersion(bytes[4]));
        }
        let kind = Kind::from_code(bytes[5]).ok_or(FrameError::BadKind(bytes[5]))?;
        let reserved = u16::from_le_bytes([bytes[6], bytes[7]]);
        if reserved != 0 {
            return Err(FrameError::BadReserved(reserved));
        }
        let payload_len = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        if payload_len > MAX_PAYLOAD {
            return Err(FrameError::TooLong(u64::from(payload_len)));
        }
        Ok(Self {
            kind,
            seq: u64::from_le_bytes(eight(bytes, 8)),
            source_ts_ns: i64::from_le_bytes(eight(bytes, 16)),
            capture_ts_ns: i64::from_le_bytes(eight(bytes, 24)),
            payload_len,
        })
    }

    /// Payload plus trailer: what follows the header on disk.
    #[must_use]
    pub fn body_len(&self) -> usize {
        // MAX_PAYLOAD fits in usize on every target with a 32-bit or wider pointer.
        usize::try_from(self.payload_len).unwrap_or(usize::MAX) + TRAILER_LEN
    }

    /// Checks the CRC trailer at the end of `body` (payload then CRC, [`Self::body_len`] bytes)
    /// against `header_bytes` and the payload.
    ///
    /// # Errors
    /// Returns [`FrameError::BadCrc`] on a mismatch, or [`FrameError::Truncated`] when `body`
    /// is shorter than the header says.
    pub fn verify(&self, header_bytes: &[u8; HEADER_LEN], body: &[u8]) -> Result<(), FrameError> {
        let needed = self.body_len();
        if body.len() < needed {
            return Err(FrameError::Truncated {
                needed: needed as u64,
                available: body.len() as u64,
            });
        }
        let payload_len = needed - TRAILER_LEN;
        let mut trailer = [0; TRAILER_LEN];
        trailer.copy_from_slice(&body[payload_len..needed]);
        let stored = u32::from_le_bytes(trailer);
        let mut digest = CASTAGNOLI.digest();
        digest.update(header_bytes);
        digest.update(&body[..payload_len]);
        let computed = digest.finalize();
        if stored == computed {
            Ok(())
        } else {
            Err(FrameError::BadCrc { stored, computed })
        }
    }

    /// Verifies `body` (payload then CRC) and builds the frame from it.
    ///
    /// # Errors
    /// Returns an error for a CRC mismatch or a payload that is not UTF-8.
    pub fn finish(
        self,
        header_bytes: &[u8; HEADER_LEN],
        mut body: Vec<u8>,
    ) -> Result<Frame, FrameError> {
        self.verify(header_bytes, &body)?;
        body.truncate(self.body_len() - TRAILER_LEN);
        let payload = String::from_utf8(body).map_err(|_| FrameError::NotUtf8)?;
        Ok(Frame {
            seq: self.seq,
            kind: self.kind,
            source_ts_ns: self.source_ts_ns,
            capture_ts_ns: self.capture_ts_ns,
            payload,
        })
    }
}

fn eight(bytes: &[u8; HEADER_LEN], at: usize) -> [u8; 8] {
    let mut out = [0; 8];
    out.copy_from_slice(&bytes[at..at + 8]);
    out
}

impl Frame {
    /// Bytes this frame takes on disk.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        HEADER_LEN + self.payload.len() + TRAILER_LEN
    }

    /// The frame as it goes to disk, in one buffer so it can be written with one `write_all`.
    ///
    /// # Errors
    /// Returns [`FrameError::TooLong`] when the payload is over [`MAX_PAYLOAD`].
    pub fn encode(&self) -> Result<Vec<u8>, FrameError> {
        let payload_len = u32::try_from(self.payload.len())
            .ok()
            .filter(|&len| len <= MAX_PAYLOAD)
            .ok_or(FrameError::TooLong(self.payload.len() as u64))?;
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(self.kind.code());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&self.source_ts_ns.to_le_bytes());
        out.extend_from_slice(&self.capture_ts_ns.to_le_bytes());
        out.extend_from_slice(&payload_len.to_le_bytes());
        out.extend_from_slice(self.payload.as_bytes());
        let crc = crc32c(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        Ok(out)
    }

    /// Decodes the frame at the start of `bytes` and returns it with the number of bytes it
    /// took. Nothing is allocated for a length over the limit or past the end of `bytes`.
    ///
    /// # Errors
    /// Returns why the bytes are not a valid frame.
    pub fn decode(bytes: &[u8]) -> Result<(Self, usize), FrameError> {
        let available = bytes.len() as u64;
        let Some(head) = bytes.first_chunk::<HEADER_LEN>() else {
            return Err(FrameError::Truncated {
                needed: HEADER_LEN as u64,
                available,
            });
        };
        let header = Header::parse(head)?;
        let total = HEADER_LEN + header.body_len();
        if bytes.len() < total {
            return Err(FrameError::Truncated {
                needed: total as u64,
                available,
            });
        }
        let frame = header.finish(head, bytes[HEADER_LEN..total].to_vec())?;
        Ok((frame, total))
    }
}

/// A Meta frame's payload. Every segment starts with one, so any segment can be read alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Meta {
    pub run_id: String,
    /// The seq of this Meta frame, which the segment file is named after.
    pub first_seq: u64,
    pub producer: Producer,
    /// The venue whose streams the run holds. Logs written before there were venues are
    /// Binance's.
    #[serde(default = "binance")]
    pub venue: Venue,
    /// The scales every price and quantity in the run was parsed with.
    pub scales: BTreeMap<Symbol, Scales>,
}

/// The venue of every run recorded before runs said which venue they were.
fn binance() -> Venue {
    Venue::BinanceUsdm
}

impl Meta {
    /// The Meta every segment of a run starts with; the writer fills in `first_seq`.
    #[must_use]
    pub fn for_run(manifest: &Manifest) -> Self {
        Self {
            run_id: manifest.run_id.clone(),
            first_seq: 0,
            producer: manifest.producer.clone(),
            venue: manifest.params.venue,
            scales: manifest.params.scales.clone(),
        }
    }
}

/// A `SegmentEnd` frame's payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentEnd {
    /// The seq of the frame before the marker, `None` for a segment with no frames.
    pub last_seq: Option<u64>,
    /// Frames in the segment before the marker.
    pub frames: u64,
    /// True when the writer closed the segment, false when recovery closed it after a crash.
    pub clean: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(kind: Kind, payload: &str) -> Frame {
        Frame {
            seq: 7,
            kind,
            source_ts_ns: 1_700_000_000_123_000_000,
            capture_ts_ns: 1_700_000_000_125_000_000,
            payload: payload.to_owned(),
        }
    }

    #[test]
    fn crc32c_matches_the_published_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn a_frame_encodes_to_the_documented_layout() -> Result<(), FrameError> {
        let bytes = frame(Kind::Control, "{}").encode()?;
        assert_eq!(bytes.len(), HEADER_LEN + 2 + TRAILER_LEN);
        assert_eq!(&bytes[..4], &[0xFF, 0x54, 0x50, 0x4C]);
        assert_eq!((bytes[4], bytes[5]), (1, 4));
        assert_eq!(&bytes[6..8], &[0, 0]);
        assert_eq!(&bytes[8..16], &7u64.to_le_bytes());
        assert_eq!(&bytes[16..24], &1_700_000_000_123_000_000i64.to_le_bytes());
        assert_eq!(&bytes[24..32], &1_700_000_000_125_000_000i64.to_le_bytes());
        assert_eq!(&bytes[32..36], &2u32.to_le_bytes());
        assert_eq!(&bytes[36..38], b"{}");
        assert_eq!(&bytes[38..], &crc32c(&bytes[..38]).to_le_bytes());
        Ok(())
    }

    #[test]
    fn decode_reverses_encode_and_reports_the_length() -> Result<(), FrameError> {
        let original = frame(Kind::Depth, "{\"e\":\"depthUpdate\"}");
        let mut bytes = original.encode()?;
        let len = bytes.len();
        bytes.extend_from_slice(b"next frame");
        assert_eq!(Frame::decode(&bytes)?, (original, len));
        Ok(())
    }

    #[test]
    fn every_kind_has_a_distinct_code() {
        for code in 0..=u8::MAX {
            if let Some(kind) = Kind::from_code(code) {
                assert_eq!(kind.code(), code);
            }
        }
        assert_eq!(Kind::from_code(6), None);
    }

    #[test]
    fn bad_headers_are_named() -> Result<(), FrameError> {
        let good = frame(Kind::Meta, "{}").encode()?;
        let with = |at: usize, value: u8| {
            let mut bytes = good.clone();
            bytes[at] = value;
            Frame::decode(&bytes).err()
        };
        assert_eq!(
            with(0, 0xFE),
            Some(FrameError::BadMagic([0xFE, b'T', b'P', b'L']))
        );
        assert_eq!(with(4, 2), Some(FrameError::BadVersion(2)));
        assert_eq!(with(5, 9), Some(FrameError::BadKind(9)));
        assert_eq!(with(6, 1), Some(FrameError::BadReserved(1)));
        assert!(matches!(with(36, b'['), Some(FrameError::BadCrc { .. })));
        Ok(())
    }

    #[test]
    fn an_oversized_length_is_refused_before_reading_the_payload() -> Result<(), FrameError> {
        let mut bytes = frame(Kind::Meta, "{}").encode()?;
        bytes[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            Frame::decode(&bytes).err(),
            Some(FrameError::TooLong(u64::from(u32::MAX)))
        );
        Ok(())
    }

    #[test]
    fn a_short_buffer_is_truncated_not_a_panic() -> Result<(), FrameError> {
        let bytes = frame(Kind::Meta, "{\"a\":1}").encode()?;
        for cut in 0..bytes.len() {
            assert!(
                matches!(
                    Frame::decode(&bytes[..cut]),
                    Err(FrameError::Truncated { .. })
                ),
                "cut at {cut}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_payload_over_the_limit_does_not_encode() {
        let big = frame(Kind::Depth, &"x".repeat(MAX_PAYLOAD as usize + 1));
        assert!(matches!(big.encode(), Err(FrameError::TooLong(_))));
    }
}
