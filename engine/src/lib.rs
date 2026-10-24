//! Tipping Point engine: streaming critical-transition detection.

pub mod api;
pub mod market;
pub mod store;
pub mod venue;

/// Crate version, exposed so the scaffold has something to test.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_set() {
        assert!(version().starts_with("0."));
    }
}
