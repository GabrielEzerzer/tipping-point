//! The one `DuckDB` library archive the build accepts (#66). libduckdb-sys downloads the
//! prebuilt library from the `DuckDB` GitHub release and checks no hash, and a release's assets can
//! be replaced under the same tag (v1.5.6 is not immutable), so the build script refuses any
//! archive but this one before it copies the library anywhere. A test checks the check.
//!
//! The pin is the SHA-256 of the archive libduckdb-sys downloaded into
//! `target/duckdb-download/x86_64-pc-windows-msvc/1.5.6/` (`sha256sum`, or `certutil -hashfile
//! <zip> SHA256`, on 2026-10-24), which equals the digest GitHub recorded for the asset at upload
//! (`gh api repos/duckdb/duckdb/releases/tags/v1.5.6`, the asset's `digest`; 13 408 787 bytes,
//! uploaded 2026-09-28). Bump it in the same commit as the duckdb and libduckdb-sys pins, from
//! the new release asset's GitHub digest.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

/// The archive libduckdb-sys downloads for Windows on x86-64.
pub const NAME: &str = "libduckdb-windows-amd64.zip";

/// Its SHA-256, lower-case hex.
pub const SHA256: &str = "44cf59583f9951d2cb09b1bf115a63ecb2d8901e363903029d86c7d8683fe96a";

/// The SHA-256 of the file at `path`, lower-case hex.
///
/// # Errors
/// Returns the I/O error from reading it.
pub fn sha256(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        }))
}

/// Whether the archive at `path` is the pinned one.
///
/// # Errors
/// Says why not: it cannot be read, or it hashes to something else.
pub fn check(path: &Path) -> Result<(), String> {
    let actual =
        sha256(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if actual == SHA256 {
        Ok(())
    } else {
        Err(format!(
            "{} has SHA-256 {actual}, not the pinned {SHA256}",
            path.display()
        ))
    }
}
