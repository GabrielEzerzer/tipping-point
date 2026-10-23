//! Bybit v5 linear perpetuals: the wire shape of its order-book stream, which is also the line
//! format of its public order-book archive ([`messages`]), and its sync rule ([`sync`]: books come
//! in band, a snapshot then deltas whose update id runs on by one). It has no live loop yet; the
//! venue is read from its archive, or from a script.

pub mod messages;
pub mod sync;
