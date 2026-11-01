//! The build accepts only the pinned `DuckDB` library archive (#66): the build script's own
//! check, run against the archive this build downloaded and checked, and against a copy of it
//! with one byte changed, as a replaced release asset would be. A build with no archive to check
//! says so on a line the gate counts (#105).

#[path = "../build/duckdb_archive.rs"]
mod duckdb_archive;

use std::error::Error;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Says a check did not run, on a line of its own past libtest's capture of a passing test, which
/// the gate counts as a skip: a copy of `engine_harness::skipped`, which this crate's tests cannot
/// import, since engine-harness depends on engine.
fn skipped(why: &str) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "\nSKIPPED: {why}");
    let _ = out.flush();
}

/// The archive this build has to check, or `None` once it has said why there is none. On Windows
/// x64 every build downloads the library unless `DUCKDB_LIB_DIR` names one, so an archive the build
/// script did not check means the check is no longer wired in.
fn archive_to_check(
    windows_x64: bool,
    lib_dir: Option<&str>,
    archive: Option<&'static str>,
) -> Option<&'static str> {
    if windows_x64 && lib_dir.is_some() {
        skipped("DuckDB comes from DUCKDB_LIB_DIR, so nothing was downloaded");
        return None;
    }
    if archive.is_none() {
        assert!(!windows_x64, "the build script checked no DuckDB archive");
        skipped("this build downloaded no DuckDB archive to check");
    }
    archive
}

/// Each build with nothing to check, played by this binary re-run as the gate runs it: libtest
/// keeps a passing test's `println!` to itself, so only a line past its capture reaches the gate.
#[test]
fn a_build_with_nothing_to_check_says_so_where_the_gate_reads_it() -> Result<(), Box<dyn Error>> {
    const PLAY: &str = "TP_DUCKDB_ARCHIVE_TEST_PLAYS";
    match std::env::var(PLAY).as_deref() {
        Ok("lib-dir") => assert_eq!(archive_to_check(true, Some("lib"), None), None),
        Ok(_) => assert_eq!(archive_to_check(false, None, None), None),
        Err(_) => {
            for build in ["lib-dir", "no-download"] {
                let run = Command::new(std::env::current_exe()?)
                    .args([
                        "--exact",
                        "a_build_with_nothing_to_check_says_so_where_the_gate_reads_it",
                    ])
                    .env(PLAY, build)
                    .output()?;
                let out = String::from_utf8_lossy(&run.stdout);
                assert!(run.status.success(), "{build}: {out}");
                assert!(
                    out.lines().any(|line| line.starts_with("SKIPPED: ")),
                    "{build}: {out}"
                );
            }
        }
    }
    Ok(())
}

/// A build that had to download but checked no archive fails, before `skipped` reaches the gate.
#[test]
#[should_panic(expected = "checked no DuckDB archive")]
fn a_windows_x64_build_with_no_checked_archive_fails() {
    archive_to_check(true, None, None);
}

#[test]
fn only_the_pinned_archive_passes() -> Result<(), Box<dyn Error>> {
    let Some(archive) = archive_to_check(
        cfg!(all(windows, target_arch = "x86_64")),
        option_env!("DUCKDB_LIB_DIR"),
        option_env!("TP_DUCKDB_ARCHIVE"),
    ) else {
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
