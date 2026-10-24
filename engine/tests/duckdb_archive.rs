//! The build accepts only the pinned `DuckDB` library archive (#66): the build script's own
//! check, run against the archive this build downloaded and checked, and against a copy of it
//! with one byte changed, as a replaced release asset would be.

#[path = "../build/duckdb_archive.rs"]
mod duckdb_archive;

use std::error::Error;
use std::fs;
use std::path::Path;

#[test]
fn only_the_pinned_archive_passes() -> Result<(), Box<dyn Error>> {
    let Some(archive) = option_env!("TP_DUCKDB_ARCHIVE") else {
        println!("skipped: this build downloaded no DuckDB archive to check");
        return Ok(());
    };
    let archive = Path::new(archive);
    assert_eq!(archive.file_name(), Some(duckdb_archive::NAME.as_ref()));
    assert_eq!(duckdb_archive::check(archive), Ok(()));

    let mut bytes = fs::read(archive)?;
    let middle = bytes.len() / 2;
    bytes[middle] ^= 1;
    let replaced = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "replaced-{}-{}",
        std::process::id(),
        duckdb_archive::NAME
    ));
    fs::write(&replaced, &bytes)?;
    let refused = duckdb_archive::check(&replaced);
    fs::remove_file(&replaced)?;
    let error = refused.err().ok_or("a replaced archive passed the check")?;
    assert!(error.contains(duckdb_archive::SHA256), "{error}");
    Ok(())
}
