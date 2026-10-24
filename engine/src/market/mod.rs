//! Market data, whichever venue it comes from: integer-unit order books ([`book`]) kept right by
//! each venue's sync rule ([`sync`]), a multi-symbol [`feed`] that turns a venue's raw frames into
//! book updates, and [`pipeline`], the one copy of how the engine reacts to every input. The
//! network ([`ws`], [`live`]) is a thin layer around it, so the same pipeline runs on a live
//! socket, a replayed log, a script, or a synthetic test stream. What differs between venues
//! (wire formats, sync rules, endpoints, scales) lives in [`crate::venue`].

pub mod book;
pub mod decimal;
pub mod feed;
pub mod live;
pub mod pipeline;
pub mod recording;
pub mod sync;
pub mod ws;

use decimal::{DecimalError, parse_units};
use sync::Level;

/// An exchange symbol in its venue's own form, e.g. `BTCUSDT`. Two venues can share one, so a
/// symbol names a book only together with its venue.
pub type Symbol = String;

/// How a symbol's decimal strings map to integer units: a price of `"12345.6"` with one price
/// decimal is `123456` ticks, a quantity of `"0.001"` with three quantity decimals is `1` lot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Scales {
    pub price_decimals: u32,
    pub qty_decimals: u32,
}

impl Scales {
    /// `[price, qty]` decimal pairs as levels in integer units.
    ///
    /// # Errors
    /// Returns an error when a price or quantity is not representable at this scale.
    pub fn levels<S: AsRef<str>>(self, pairs: &[[S; 2]]) -> Result<Vec<Level>, DecimalError> {
        pairs
            .iter()
            .map(|[price, qty]| {
                Ok(Level {
                    price: parse_units(price.as_ref(), self.price_decimals)?,
                    qty: parse_units(qty.as_ref(), self.qty_decimals)?,
                })
            })
            .collect()
    }
}
