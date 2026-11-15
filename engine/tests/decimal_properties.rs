//! Property tests on the decimal conversions: units rendered and read back are unchanged, reading
//! them at more decimals scales them by ten per decimal, and one unit's step has its own decimals.

use engine::market::decimal::{format_units, parse_units, step_decimals};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

fn config() -> Config {
    Config {
        cases: 512,
        rng_seed: RngSeed::Fixed(26),
        failure_persistence: None,
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn units_rendered_then_read_back_are_unchanged(units in 0..=i64::MAX, decimals in 0u32..=18) {
        prop_assert_eq!(parse_units(&format_units(units, decimals), decimals), Ok(units));
    }

    #[test]
    fn trailing_zeros_change_nothing_and_extra_decimals_scale_by_ten(
        units in 0i64..=1_000_000_000_000,
        decimals in 0u32..=6,
        extra in 0u32..=6,
        zeros in 0usize..=4,
    ) {
        let text = format_units(units, decimals);
        let point = if decimals == 0 { "." } else { "" };
        let padded = format!("{text}{point}{}", "0".repeat(zeros));
        prop_assert_eq!(parse_units(&padded, decimals), Ok(units));
        prop_assert_eq!(parse_units(&text, decimals + extra), Ok(units * 10_i64.pow(extra)));
    }

    #[test]
    fn the_step_of_one_unit_has_as_many_decimals_as_its_scale(
        decimals in 0u32..=18,
        zeros in 0usize..=4,
    ) {
        let step = format_units(1, decimals);
        let point = if decimals == 0 { "." } else { "" };
        prop_assert_eq!(step_decimals(&step), decimals);
        prop_assert_eq!(step_decimals(&format!("{step}{point}{}", "0".repeat(zeros))), decimals);
    }
}
