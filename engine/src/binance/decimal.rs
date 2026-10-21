//! Exact conversion of Binance's decimal strings to integer units, with no float round-trip.

use thiserror::Error;

/// Why a decimal string could not be converted.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DecimalError {
    #[error("empty decimal string")]
    Empty,
    #[error("invalid character {0:?} in decimal string")]
    InvalidChar(char),
    #[error("{0:?} has more precision than {1} decimals")]
    TooPrecise(String, u32),
    #[error("{0:?} overflows i64 at {1} decimals")]
    Overflow(String, u32),
}

/// Parses a non-negative decimal string like `"12345.60"` into integer units of `10^-decimals`.
///
/// `"12345.60"` with `decimals = 1` is `123456`; `"0.001"` with `decimals = 3` is `1`.
/// Trailing zeros beyond the scale are fine; any other extra precision is an error, because
/// silently rounding a price or quantity would corrupt the book.
///
/// # Errors
/// Returns an error for anything that is not plain `digits[.digits]`, for excess precision, or
/// for a value that does not fit in an `i64`.
pub fn parse_units(text: &str, decimals: u32) -> Result<i64, DecimalError> {
    if text.is_empty() {
        return Err(DecimalError::Empty);
    }
    let (int_part, frac_part) = text.split_once('.').unwrap_or((text, ""));
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(DecimalError::Empty);
    }

    let mut value: i64 = 0;
    let mut push_digit = |c: char| -> Result<(), DecimalError> {
        let digit = c.to_digit(10).ok_or(DecimalError::InvalidChar(c))?;
        value = value
            .checked_mul(10)
            .and_then(|v| v.checked_add(i64::from(digit)))
            .ok_or_else(|| DecimalError::Overflow(text.to_owned(), decimals))?;
        Ok(())
    };

    for c in int_part.chars() {
        push_digit(c)?;
    }
    let wanted = decimals as usize;
    let mut frac = frac_part.chars();
    for _ in 0..wanted {
        push_digit(frac.next().unwrap_or('0'))?;
    }
    for c in frac {
        if !c.is_ascii_digit() {
            return Err(DecimalError::InvalidChar(c));
        }
        if c != '0' {
            return Err(DecimalError::TooPrecise(text.to_owned(), decimals));
        }
    }
    Ok(value)
}

/// Number of significant decimals in a step like `"0.10"` (1) or `"0.001"` (3) or `"1"` (0).
///
/// Binance filters express tick and lot sizes as decimal strings; the scale is how many digits
/// after the point survive once trailing zeros are stripped.
#[must_use]
pub fn step_decimals(step: &str) -> u32 {
    let frac = step.split_once('.').map_or("", |(_, f)| f);
    u32::try_from(frac.trim_end_matches('0').len()).unwrap_or(u32::MAX)
}

/// Renders integer units back to a decimal string, e.g. `123456` at 1 decimal is `"12345.6"`.
#[must_use]
pub fn format_units(units: i64, decimals: u32) -> String {
    if decimals == 0 {
        return units.to_string();
    }
    let width = decimals as usize;
    let sign = if units < 0 { "-" } else { "" };
    let digits = units.unsigned_abs().to_string();
    let padded = format!("{digits:0>w$}", w = width + 1);
    let (int_part, frac_part) = padded.split_at(padded.len() - width);
    format!("{sign}{int_part}.{frac_part}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prices_and_quantities_exactly() {
        assert_eq!(parse_units("12345.60", 1), Ok(123_456));
        assert_eq!(parse_units("12345.6", 1), Ok(123_456));
        assert_eq!(parse_units("12345", 1), Ok(123_450));
        assert_eq!(parse_units("0.001", 3), Ok(1));
        assert_eq!(parse_units("0.000", 3), Ok(0));
        assert_eq!(parse_units("7", 0), Ok(7));
        assert_eq!(parse_units(".5", 1), Ok(5));
        assert_eq!(parse_units("5.", 1), Ok(50));
    }

    #[test]
    fn rejects_excess_precision_and_junk() {
        assert_eq!(
            parse_units("0.0015", 3),
            Err(DecimalError::TooPrecise("0.0015".into(), 3))
        );
        assert_eq!(parse_units("0.0010", 3), Ok(1));
        assert_eq!(parse_units("", 1), Err(DecimalError::Empty));
        assert_eq!(parse_units(".", 1), Err(DecimalError::Empty));
        assert_eq!(parse_units("-1", 1), Err(DecimalError::InvalidChar('-')));
        assert_eq!(parse_units("1e5", 1), Err(DecimalError::InvalidChar('e')));
        assert_eq!(parse_units("1.2x", 3), Err(DecimalError::InvalidChar('x')));
        assert!(matches!(
            parse_units("99999999999999999999", 0),
            Err(DecimalError::Overflow(..))
        ));
    }

    #[test]
    fn step_decimals_strips_trailing_zeros() {
        assert_eq!(step_decimals("0.10"), 1);
        assert_eq!(step_decimals("0.01"), 2);
        assert_eq!(step_decimals("0.001"), 3);
        assert_eq!(step_decimals("1"), 0);
        assert_eq!(step_decimals("1.000"), 0);
    }

    #[test]
    fn formats_back() {
        assert_eq!(format_units(123_456, 1), "12345.6");
        assert_eq!(format_units(1, 3), "0.001");
        assert_eq!(format_units(0, 2), "0.00");
        assert_eq!(format_units(7, 0), "7");
        assert_eq!(format_units(-15, 1), "-1.5");
    }
}
