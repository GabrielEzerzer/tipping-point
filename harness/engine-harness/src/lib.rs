//! Shared harness code: transcripts for the replay-equals-live checks, and the payloads the
//! crash-injection writer writes.

pub mod transcript;

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
