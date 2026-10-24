//! Binance's daily aggregated-trades archive for USD-M futures: one CSV row per aggregated trade,
//! `agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker`, with
//! a header line in files from 2022 on and none before. A record's frame is the `aggTrade` event
//! the live stream sends for that trade, its numbers copied as written, so the stream's own parser
//! reads it; the archive has no event time, so the trade time stands for it (`E` = `T`).
//!
//! Published at `https://data.binance.vision/data/futures/um/daily/aggTrades/<SYMBOL>/<SYMBOL>-aggTrades-<date>.zip`,
//! with a `.CHECKSUM` (SHA-256) beside each file.

use crate::store::frame::Kind;
use crate::store::ms_to_ns;
use crate::venue::archive::{ArchiveFormat, ArchiveFrame, ArchiveName, Day, Scan};

/// The header line of files from 2022 on.
const HEADER: &str =
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker";

/// Reads `<SYMBOL>-aggTrades-<yyyy-mm-dd>.zip`.
pub(crate) fn name(file_name: &str) -> Option<ArchiveName> {
    let stem = file_name.strip_suffix(".zip")?;
    let (symbol, date) = stem.split_once("-aggTrades-")?;
    let symbol_ok = !symbol.is_empty()
        && symbol
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit());
    if !symbol_ok {
        return None;
    }
    Some(ArchiveName {
        format: ArchiveFormat::BinanceAggTrades,
        symbol: symbol.to_owned(),
        day: Day::parse(date)?,
        depth: None,
        file_name: file_name.to_owned(),
    })
}

pub(crate) fn url(name: &ArchiveName) -> String {
    format!(
        "https://data.binance.vision/data/futures/um/daily/aggTrades/{}/{}",
        name.symbol, name.file_name
    )
}

/// One CSV row, its numbers still as written.
struct Row<'a> {
    agg_trade_id: u64,
    price: &'a str,
    qty: &'a str,
    first_trade_id: u64,
    last_trade_id: u64,
    time_ms: u64,
    buyer_is_maker: bool,
}

/// Reads a row; `None` for the header on the first line.
fn row(line: &str, first: bool) -> Result<Option<Row<'_>>, String> {
    if first && line == HEADER {
        return Ok(None);
    }
    let fields: Vec<&str> = line.split(',').collect();
    let [id, price, qty, first_id, last_id, time, maker] = fields[..] else {
        return Err(format!("{} fields, not 7", fields.len()));
    };
    let number = |text: &str, what: &str| -> Result<u64, String> {
        text.parse()
            .map_err(|_| format!("{what} {text:?} is not a whole number"))
    };
    let buyer_is_maker = match maker {
        "true" | "True" => true,
        "false" | "False" => false,
        other => return Err(format!("is_buyer_maker {other:?} is not true or false")),
    };
    Ok(Some(Row {
        agg_trade_id: number(id, "agg_trade_id")?,
        price,
        qty,
        first_trade_id: number(first_id, "first_trade_id")?,
        last_trade_id: number(last_id, "last_trade_id")?,
        time_ms: number(time, "transact_time")?,
        buyer_is_maker,
    }))
}

/// Reads one line in the pre-scan: every price and quantity a plain decimal, the aggregate trade
/// ids and the trade ids each chained on the last row's, the time noted.
pub(crate) fn scan(
    name: &ArchiveName,
    scan: &mut Scan,
    line: &str,
    first: bool,
) -> Result<(), String> {
    let Some(row) = row(line, first)? else {
        return Ok(());
    };
    scan.price(row.price)?;
    scan.qty(row.qty)?;
    scan.record(name.day, row.time_ms, row.agg_trade_id, false);
    scan.trade_ids(row.first_trade_id, row.last_trade_id);
    Ok(())
}

/// The frame a row becomes: the live stream's `aggTrade` event for it, a trade frame at its
/// trade time; `None` for the header.
pub(crate) fn frame(name: &ArchiveName, line: &str) -> Result<Option<ArchiveFrame>, String> {
    // Every field is checked here (the numbers are plain decimals and whole numbers, so they
    // need no escaping in JSON); the pre-scan has checked the decimals.
    let Some(row) = row(line, true)? else {
        return Ok(None);
    };
    if row.price.contains(['"', '\\']) || row.qty.contains(['"', '\\']) {
        return Err("a price or quantity that is not a plain decimal".to_owned());
    }
    let payload = format!(
        r#"{{"stream":"{}@aggTrade","data":{{"e":"aggTrade","E":{t},"a":{},"s":"{}","p":"{}","q":"{}","f":{},"l":{},"T":{t},"m":{}}}}}"#,
        name.symbol.to_lowercase(),
        row.agg_trade_id,
        name.symbol,
        row.price,
        row.qty,
        row.first_trade_id,
        row.last_trade_id,
        row.buyer_is_maker,
        t = row.time_ms,
    );
    Ok(Some(ArchiveFrame {
        kind: Kind::Trade,
        time_ns: ms_to_ns(row.time_ms),
        payload,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::Scales;
    use crate::venue::{Parsed, RawTrade, Venue};

    fn btc() -> ArchiveName {
        name("BTCUSDT-aggTrades-2026-10-20.zip").expect("a Binance archive name")
    }

    const ROW: &str = "3490482571,81903.3,0.003,8187372393,8187372394,1792454400007,true";

    #[test]
    fn a_row_becomes_the_live_streams_agg_trade_event() -> Result<(), String> {
        let frame = frame(&btc(), ROW)?.ok_or("a record")?;
        assert_eq!(frame.kind, Kind::Trade);
        assert_eq!(frame.time_ns, 1_792_454_400_007_000_000);
        let Ok(Parsed::Trade(RawTrade::Binance(raw))) = Venue::BinanceUsdm.parse(&frame.payload)
        else {
            return Err(format!("not an aggTrade event: {}", frame.payload));
        };
        assert_eq!(
            (raw.agg_trade_id, raw.price.as_str(), raw.qty.as_str()),
            (3_490_482_571, "81903.3", "0.003")
        );
        assert_eq!(
            (raw.first_trade_id, raw.last_trade_id),
            (Some(8_187_372_393), Some(8_187_372_394))
        );
        assert_eq!(
            (raw.trade_time_ms, raw.event_time_ms),
            (1_792_454_400_007, 1_792_454_400_007)
        );
        assert!(raw.buyer_is_maker);
        assert_eq!(raw.symbol, "BTCUSDT");
        Ok(())
    }

    #[test]
    fn the_header_is_no_record_and_only_first() -> Result<(), String> {
        assert_eq!(frame(&btc(), HEADER)?, None);
        let mut found = Scan::new(ArchiveFormat::BinanceAggTrades);
        scan(&btc(), &mut found, HEADER, true)?;
        assert!(
            scan(&btc(), &mut found, HEADER, false).is_err(),
            "a header mid-file"
        );
        scan(&btc(), &mut found, ROW, false)?;
        let next = "3490482572,81903.4,0.002,8187372396,8187372396,1792454400150,false";
        scan(&btc(), &mut found, next, false)?;
        assert_eq!(
            found.scales(),
            Scales {
                price_decimals: 1,
                qty_decimals: 3
            }
        );
        let integrity = found.integrity();
        assert_eq!((integrity.records, integrity.id_gaps), (2, 0));
        assert_eq!(integrity.trade_id_gaps, Some(1), "8187372395 is missing");
        Ok(())
    }

    #[test]
    fn a_short_or_odd_row_is_refused() {
        for bad in [
            "1,2,3",
            "x,81903.3,0.003,1,1,1,true",
            "1,81903.3,0.003,1,1,1,yes",
            "1,81903.3,0.003,1,1,1,true,extra",
        ] {
            assert!(frame(&btc(), bad).is_err(), "{bad}");
        }
    }
}
