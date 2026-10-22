//! Binance USD-M futures: the wire shapes of its depth and trade streams and REST replies
//! ([`messages`]), its order-book sync rule ([`sync`]: a REST snapshot bridged onto the buffered
//! diff stream, then the `pu` chain), its REST client for scales and snapshots ([`rest`]), and
//! its live loop ([`live`]: two combined-stream sockets, depth and aggregated trades).

pub mod live;
pub mod messages;
pub mod rest;
pub mod sync;
