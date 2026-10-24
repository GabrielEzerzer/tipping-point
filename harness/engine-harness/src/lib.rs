//! Shared harness code: transcripts for the replay-equals-live checks, the payloads the
//! crash-injection writer writes, the in-process server and clients of the API checks, a guard
//! that never lets a test's child process outlive the test, and the line that says a check was
//! skipped.

pub mod api;
pub mod load;
pub mod transcript;

use std::ops::{Deref, DerefMut};
use std::process::Child;

/// A child process that is killed and reaped when this drops, so a test that fails or returns
/// early never leaves it running. A test runner killed outright runs no `Drop`: a child that
/// would run forever (the crash writer, told `--end-with-stdin`) also ends when its piped stdin
/// closes, which happens however its parent ends.
#[derive(Debug)]
pub struct KillOnDrop(pub Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl Deref for KillOnDrop {
    type Target = Child;

    fn deref(&self) -> &Child {
        &self.0
    }
}

impl DerefMut for KillOnDrop {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}

/// Says a check did not run, on a line of its own that the gate counts as a skip (#60). Rust's
/// test harness has no runtime skip: a test that returns early passes, and libtest keeps a
/// passing test's `println!` output to itself. So this writes to stdout's handle, which libtest
/// does not capture, and a check that could not run never reads as one that passed.
pub fn skipped(why: &str) {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "\nSKIPPED: {why}");
    let _ = out.flush();
}

/// The payload of the `i`-th entry the crash writer appends: a known function of `i`, of varying
/// length, so a recovered log can be checked frame by frame.
#[must_use]
pub fn crash_payload(i: u64) -> String {
    let pad = usize::try_from(i % 97).unwrap_or(0);
    format!("{{\"i\":{i},\"pad\":\"{}\"}}", "x".repeat(pad))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crash_payloads_differ_and_vary_in_length() {
        assert_ne!(crash_payload(1), crash_payload(2));
        assert_ne!(crash_payload(1).len(), crash_payload(2).len());
        assert!(crash_payload(5).starts_with("{\"i\":5,"));
    }
}
