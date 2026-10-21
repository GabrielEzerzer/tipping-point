//! Integration smoke test: the engine links and answers from outside its own crate.

#[test]
fn version_is_reachable_from_outside_the_crate() {
    assert!(engine::version().starts_with("0."));
}
