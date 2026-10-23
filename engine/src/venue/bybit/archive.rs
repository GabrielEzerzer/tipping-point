//! Bybit's public order-book archive: one day of one linear perpetual's book, `ob200` (200
//! levels a side, from 2025-08-21) or `ob500` (500, from 2023-01-18), every message of its
//! stream every 100 ms, one per CRLF-ended line, in the same shape as the live stream's
//! (`orderbook.{depth}.{symbol}`): a snapshot, then deltas whose `u` runs on by one, and a
//! snapshot again whenever Bybit sent one. A record's frame is its line, unchanged: the stream's
//! own parser reads it.
//!
//! Published at `https://quote-saver.bycsi.com/orderbook/linear/<SYMBOL>/<date>_<SYMBOL>_ob<N>.data.zip`.

use serde::Deserialize;

use crate::store::frame::Kind;
use crate::venue::archive::{ArchiveFormat, ArchiveFrame, ArchiveName, Day, Scan};

/// Reads `<yyyy-mm-dd>_<SYMBOL>_ob<N>.data.zip`.
pub(crate) fn name(file_name: &str) -> Option<ArchiveName> {
    let stem = file_name.strip_suffix(".data.zip")?;
    let mut parts = stem.split('_');
    let (date, symbol, depth) = (parts.next()?, parts.next()?, parts.next()?);
    let symbol_ok = !symbol.is_empty()
        && symbol
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit());
    let depth: u32 = depth.strip_prefix("ob")?.parse().ok()?;
    if parts.next().is_some() || !symbol_ok {
        return None;
    }
    Some(ArchiveName {
        format: ArchiveFormat::BybitOrderbook,
        symbol: symbol.to_owned(),
        day: Day::parse(date)?,
        depth: Some(depth),
        file_name: file_name.to_owned(),
    })
}

pub(crate) fn url(name: &ArchiveName) -> String {
    format!(
        "https://quote-saver.bycsi.com/orderbook/linear/{}/{}",
        name.symbol, name.file_name
    )
}

/// What the pre-scan needs from a line, borrowed from it.
#[derive(Deserialize)]
struct Line<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    ts: u64,
    #[serde(borrow)]
    data: Data<'a>,
}

#[derive(Deserialize)]
struct Data<'a> {
    s: &'a str,
    #[serde(borrow)]
    b: Vec<[&'a str; 2]>,
    #[serde(borrow)]
    a: Vec<[&'a str; 2]>,
    u: u64,
}

/// Reads one line in the pre-scan: a message of this symbol's book, every price and quantity a
/// plain decimal, its `u` and time noted.
pub(crate) fn scan(name: &ArchiveName, scan: &mut Scan, line: &str) -> Result<(), String> {
    let message: Line<'_> = serde_json::from_str(line).map_err(|error| error.to_string())?;
    if message.data.s != name.symbol {
        return Err(format!(
            "a message of {}, not {}",
            message.data.s, name.symbol
        ));
    }
    let whole = match message.kind {
        "snapshot" => true,
        "delta" => message.data.u == 1,
        other => return Err(format!("a message of type {other:?}")),
    };
    for [price, qty] in message.data.b.iter().chain(&message.data.a) {
        scan.price(price)?;
        scan.qty(qty)?;
    }
    scan.record(name.day, message.ts, message.data.u, whole);
    Ok(())
}

/// The frame a line becomes: the line itself, a depth frame at its message time.
pub(crate) fn frame(line: &str) -> Result<ArchiveFrame, String> {
    let time_ns = super::messages::frame_event_time_ns(line);
    if time_ns == 0 {
        return Err("a message without its time".to_owned());
    }
    Ok(ArchiveFrame {
        kind: Kind::Depth,
        time_ns,
        payload: line.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::Scales;

    const SNAPSHOT: &str = r#"{"topic":"orderbook.200.BTCUSDT","type":"snapshot","ts":1792454400495,"data":{"s":"BTCUSDT","b":[["81900.70","3.914"]],"a":[["81900.80","1.206"]],"u":5788995,"seq":828958074456},"cts":1792454400393}"#;

    fn btc() -> ArchiveName {
        name("2026-10-20_BTCUSDT_ob200.data.zip").expect("a Bybit archive name")
    }

    #[test]
    fn a_line_is_its_own_frame_at_its_message_time() -> Result<(), String> {
        let frame = frame(SNAPSHOT)?;
        assert_eq!(frame.kind, Kind::Depth);
        assert_eq!(frame.time_ns, 1_792_454_400_495_000_000);
        assert_eq!(frame.payload, SNAPSHOT);
        Ok(())
    }

    #[test]
    fn the_scan_reads_scales_ids_and_times() -> Result<(), String> {
        let mut found = Scan::new(ArchiveFormat::BybitOrderbook);
        scan(&btc(), &mut found, SNAPSHOT)?;
        let delta = SNAPSHOT
            .replace(r#""type":"snapshot""#, r#""type":"delta""#)
            .replace("5788995", "5788997");
        scan(&btc(), &mut found, &delta)?;
        assert_eq!(
            found.scales(),
            Scales {
                price_decimals: 1,
                qty_decimals: 3
            }
        );
        let integrity = found.integrity();
        assert_eq!(
            (integrity.records, integrity.snapshots, integrity.id_gaps),
            (2, Some(1), 1)
        );
        assert_eq!(
            (integrity.first_id, integrity.last_id),
            (Some(5_788_995), Some(5_788_997))
        );
        Ok(())
    }

    #[test]
    fn another_symbol_or_type_is_refused() {
        let mut found = Scan::new(ArchiveFormat::BybitOrderbook);
        let eth = SNAPSHOT.replace(r#""s":"BTCUSDT""#, r#""s":"ETHUSDT""#);
        assert!(scan(&btc(), &mut found, &eth).is_err());
        let odd = SNAPSHOT.replace(r#""type":"snapshot""#, r#""type":"trade""#);
        assert!(scan(&btc(), &mut found, &odd).is_err());
        assert!(scan(&btc(), &mut found, "not json").is_err());
        assert!(frame(r#"{"topic":"x","data":{}}"#).is_err());
    }
}
