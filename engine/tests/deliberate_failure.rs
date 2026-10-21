//! Deliberately failing test: proves the gate goes red. Removed in the next commit.

#[test]
fn deliberately_fails() {
    assert_eq!(engine::version(), "not-the-version", "deliberate failure");
}
