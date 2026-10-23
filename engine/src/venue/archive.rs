//! The archives venues publish, read record by record: what an archive file is, from its name;
//! what a pre-scan of its records finds (the scales its prices and quantities need, and its
//! integrity: how many records, whether the venue's ids run on, whether the times sit inside the
//! day); and the frame each record becomes. A record's frame is what the venue's live stream
//! would have carried for it, so the pipeline reads archives with the very parser it reads the
//! live streams with.
//!
//! | venue | archive | file | a record | its frame |
//! |---|---|---|---|---|
//! | `bybit-linear` | order book | `2026-10-20_BTCUSDT_ob200.data.zip` | a stream message, one per line | the line, a depth frame |
//! | `binance-usdm` | aggregated trades | `BTCUSDT-aggTrades-2026-10-20.zip` | a CSV row | the `aggTrade` event, a trade frame |

use serde::{Deserialize, Serialize};

use super::{Venue, binance, bybit};
use crate::market::decimal::step_decimals;
use crate::market::{Scales, Symbol};
use crate::store::frame::Kind;
use crate::store::ms_to_ns;
use crate::store::run::{DAY_NS, Integrity, civil_from_days, days_from_civil};

/// An archive format the engine imports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchiveFormat {
    /// Bybit's order-book archive: its stream's messages, a snapshot then deltas, one per line.
    #[serde(rename = "bybit-orderbook")]
    BybitOrderbook,
    /// Binance's daily aggregated trades: CSV, a header line in files from 2022 on.
    #[serde(rename = "binance-aggtrades")]
    BinanceAggTrades,
}

impl ArchiveFormat {
    /// The format's name in manifests.
    #[must_use]
    pub fn id(self) -> &'static str {
        match self {
            Self::BybitOrderbook => "bybit-orderbook",
            Self::BinanceAggTrades => "binance-aggtrades",
        }
    }

    /// The venue that publishes it.
    #[must_use]
    pub fn venue(self) -> Venue {
        match self {
            Self::BybitOrderbook => Venue::BybitLinear,
            Self::BinanceAggTrades => Venue::BinanceUsdm,
        }
    }

    /// The tables an import writes unless told otherwise: what the archive is for. The rest are
    /// opt-in, because the archive itself is the lossless copy.
    #[must_use]
    pub fn default_tables(self) -> &'static [&'static str] {
        match self {
            Self::BybitOrderbook => &["book_samples"],
            Self::BinanceAggTrades => &["trades"],
        }
    }
}

/// A UTC calendar day.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Day {
    pub year: i64,
    pub month: u32,
    pub day: u32,
}

impl Day {
    /// Reads `yyyy-mm-dd`.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let mut parts = text.split('-');
        let (year, month, day) = (parts.next()?, parts.next()?, parts.next()?);
        let digits =
            |part: &str, len: usize| part.len() == len && part.bytes().all(|b| b.is_ascii_digit());
        if parts.next().is_some() || !digits(year, 4) || !digits(month, 2) || !digits(day, 2) {
            return None;
        }
        let day = Self {
            year: year.parse().ok()?,
            month: month.parse().ok()?,
            day: day.parse().ok()?,
        };
        // A real date comes back from its day count unchanged; 2026-02-30 would come back as
        // 2026-03-02.
        let days = days_from_civil(day.year, day.month, day.day);
        let back = civil_from_days(days);
        let same = back == (day.year, i64::from(day.month), i64::from(day.day));
        ((1..=12).contains(&day.month) && same).then_some(day)
    }

    /// Midnight UTC that starts the day, in nanoseconds since the epoch.
    #[must_use]
    pub fn start_ns(self) -> i64 {
        days_from_civil(self.year, self.month, self.day).saturating_mul(DAY_NS)
    }

    /// `yyyymmdd`.
    #[must_use]
    pub fn compact(self) -> String {
        format!("{:04}{:02}{:02}", self.year, self.month, self.day)
    }

    /// True when `ns` falls inside the day.
    #[must_use]
    pub fn holds(self, ns: i64) -> bool {
        let start = self.start_ns();
        (start..start.saturating_add(DAY_NS)).contains(&ns)
    }
}

/// What a downloaded archive file is, read from its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveName {
    pub format: ArchiveFormat,
    pub symbol: Symbol,
    pub day: Day,
    /// Bybit's book depth (`ob200`, `ob500`); none for trades.
    pub depth: Option<u32>,
    pub file_name: String,
}

impl ArchiveName {
    /// Recognises an archive by its file name, as the venue publishes it.
    #[must_use]
    pub fn parse(file_name: &str) -> Option<Self> {
        bybit::archive::name(file_name).or_else(|| binance::archive::name(file_name))
    }

    #[must_use]
    pub fn venue(&self) -> Venue {
        self.format.venue()
    }

    /// The one file the archive holds: its own name without `.zip` (Bybit's `.data`), or with
    /// `.csv` for it (Binance's).
    #[must_use]
    pub fn member(&self) -> String {
        let stem = self
            .file_name
            .strip_suffix(".zip")
            .unwrap_or(&self.file_name);
        match self.format {
            ArchiveFormat::BybitOrderbook => stem.to_owned(),
            ArchiveFormat::BinanceAggTrades => format!("{stem}.csv"),
        }
    }

    /// Where the venue publishes it.
    #[must_use]
    pub fn url(&self) -> String {
        match self.format {
            ArchiveFormat::BybitOrderbook => bybit::archive::url(self),
            ArchiveFormat::BinanceAggTrades => binance::archive::url(self),
        }
    }

    /// The streams its records are, as a run's manifest lists them.
    #[must_use]
    pub fn streams(&self) -> Vec<String> {
        match self.format {
            ArchiveFormat::BybitOrderbook => {
                vec![format!("orderbook.{}", self.depth.unwrap_or(200))]
            }
            ArchiveFormat::BinanceAggTrades => vec![binance::live::TRADE_STREAM.to_owned()],
        }
    }

    /// Reads one line of the archive in the pre-scan. `first` says it is the file's first line,
    /// the only place a CSV header may be.
    ///
    /// # Errors
    /// Returns why the line is not a record of this archive.
    pub fn scan(&self, scan: &mut Scan, line: &str, first: bool) -> Result<(), String> {
        match self.format {
            ArchiveFormat::BybitOrderbook => bybit::archive::scan(self, scan, line),
            ArchiveFormat::BinanceAggTrades => binance::archive::scan(self, scan, line, first),
        }
    }

    /// The frame one line becomes; `None` for a CSV header.
    ///
    /// # Errors
    /// Returns why the line is not a record of this archive.
    pub fn frame(&self, line: &str) -> Result<Option<ArchiveFrame>, String> {
        match self.format {
            ArchiveFormat::BybitOrderbook => bybit::archive::frame(line).map(Some),
            ArchiveFormat::BinanceAggTrades => binance::archive::frame(self, line),
        }
    }
}

/// The frame an archive record becomes: its kind, its exchange time (which is also its capture
/// time: archives run on the exchange clock), and its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveFrame {
    pub kind: Kind,
    pub time_ns: i64,
    pub payload: String,
}

/// What a pre-scan of an archive has found so far.
#[derive(Debug, Clone)]
pub struct Scan {
    price_decimals: u32,
    qty_decimals: u32,
    integrity: Integrity,
    prev_id: Option<u64>,
    prev_last_trade: Option<u64>,
}

impl Scan {
    /// Before the first record of an archive of `format`.
    #[must_use]
    pub fn new(format: ArchiveFormat) -> Self {
        let integrity = Integrity {
            snapshots: (format == ArchiveFormat::BybitOrderbook).then_some(0),
            trade_id_gaps: (format == ArchiveFormat::BinanceAggTrades).then_some(0),
            ..Integrity::default()
        };
        Self {
            price_decimals: 0,
            qty_decimals: 0,
            integrity,
            prev_id: None,
            prev_last_trade: None,
        }
    }

    /// The scales every price and quantity seen needs: the most significant decimals any of
    /// them has, so each converts exactly. Read from the file itself, never from today's
    /// exchange info (Binance's BTCUSDT tick was 0.01 in 2021-22 and 0.1 since).
    #[must_use]
    pub fn scales(&self) -> Scales {
        Scales {
            price_decimals: self.price_decimals,
            qty_decimals: self.qty_decimals,
        }
    }

    /// The integrity found so far.
    #[must_use]
    pub fn integrity(&self) -> &Integrity {
        &self.integrity
    }

    /// Takes a price as written.
    ///
    /// # Errors
    /// Returns why the text is not a plain non-negative decimal.
    pub(super) fn price(&mut self, text: &str) -> Result<(), String> {
        self.price_decimals = self.price_decimals.max(decimals(text)?);
        Ok(())
    }

    /// Takes a quantity as written.
    ///
    /// # Errors
    /// Returns why the text is not a plain non-negative decimal.
    pub(super) fn qty(&mut self, text: &str) -> Result<(), String> {
        self.qty_decimals = self.qty_decimals.max(decimals(text)?);
        Ok(())
    }

    /// Takes one record: its exchange time, the venue's continuity id, and whether it is a
    /// whole book (where the id may jump: Bybit's snapshots).
    pub(super) fn record(&mut self, day: Day, time_ms: u64, id: u64, whole: bool) {
        let integrity = &mut self.integrity;
        integrity.records += 1;
        let ns = ms_to_ns(time_ms);
        if !day.holds(ns) {
            integrity.outside_day += 1;
        }
        integrity.first_ts_ns.get_or_insert(ns);
        integrity.last_ts_ns = Some(ns);
        integrity.first_id.get_or_insert(id);
        integrity.last_id = Some(id);
        if whole {
            if let Some(snapshots) = integrity.snapshots.as_mut() {
                *snapshots += 1;
            }
        } else if self
            .prev_id
            .is_some_and(|prev| prev.checked_add(1) != Some(id))
        {
            integrity.id_gaps += 1;
        }
        self.prev_id = Some(id);
    }

    /// Takes the trade ids an aggregated trade covers: the first must follow the last one's.
    pub(super) fn trade_ids(&mut self, first: u64, last: u64) {
        if self
            .prev_last_trade
            .is_some_and(|prev| prev.checked_add(1) != Some(first))
            && let Some(gaps) = self.integrity.trade_id_gaps.as_mut()
        {
            *gaps += 1;
        }
        self.prev_last_trade = Some(last);
    }
}

/// The significant decimals of a plain non-negative decimal, `digits[.digits]`.
fn decimals(text: &str) -> Result<u32, String> {
    let (int, frac) = text.split_once('.').unwrap_or((text, ""));
    let plain = !(int.is_empty() && frac.is_empty())
        && int.bytes().chain(frac.bytes()).all(|b| b.is_ascii_digit());
    if plain {
        Ok(step_decimals(text))
    } else {
        Err(format!("{text:?} is not a plain decimal"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archives_are_recognised_by_their_published_names() {
        let bybit = ArchiveName::parse("2026-10-20_BTCUSDT_ob200.data.zip").expect("Bybit's");
        assert_eq!(bybit.format, ArchiveFormat::BybitOrderbook);
        assert_eq!((bybit.symbol.as_str(), bybit.depth), ("BTCUSDT", Some(200)));
        assert_eq!(bybit.member(), "2026-10-20_BTCUSDT_ob200.data");
        assert_eq!(bybit.streams(), vec!["orderbook.200".to_owned()]);
        assert_eq!(
            bybit.url(),
            "https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/2026-10-20_BTCUSDT_ob200.data.zip"
        );
        let binance = ArchiveName::parse("ETHUSDT-aggTrades-2023-06-15.zip").expect("Binance's");
        assert_eq!(binance.format, ArchiveFormat::BinanceAggTrades);
        assert_eq!(
            binance.day,
            Day {
                year: 2023,
                month: 6,
                day: 15
            }
        );
        assert_eq!(binance.member(), "ETHUSDT-aggTrades-2023-06-15.csv");
        assert_eq!(
            binance.url(),
            "https://data.binance.vision/data/futures/um/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-2023-06-15.zip"
        );
        for other in [
            "2026-10-20_BTCUSDT_ob200.data",
            "2026-13-20_BTCUSDT_ob200.data.zip",
            "2026-02-30_BTCUSDT_ob200.data.zip",
            "2026-10-20_btc/usdt_ob200.data.zip",
            "BTCUSDT-trades-2026-10-20.zip",
            "BTCUSDT-aggTrades-2026-10-20.csv",
            "BTCUSDT-aggTrades-2026-10.zip",
        ] {
            assert_eq!(ArchiveName::parse(other), None, "{other}");
        }
    }

    #[test]
    fn a_day_knows_its_bounds() {
        let day = Day::parse("2026-10-20").expect("a day");
        assert_eq!(day.start_ns(), 1_792_454_400_000_000_000);
        assert!(day.holds(day.start_ns()) && !day.holds(day.start_ns() + DAY_NS));
        assert_eq!(day.compact(), "20261020");
    }

    #[test]
    fn scales_are_the_most_decimals_any_value_needs() -> Result<(), String> {
        let mut scan = Scan::new(ArchiveFormat::BybitOrderbook);
        for price in ["81900.70", "81900.65", "81901"] {
            scan.price(price)?;
        }
        for qty in ["0", "3.914", "1.000"] {
            scan.qty(qty)?;
        }
        assert_eq!(
            scan.scales(),
            Scales {
                price_decimals: 2,
                qty_decimals: 3
            }
        );
        assert!(scan.price("1e5").is_err() && scan.qty("-1").is_err() && scan.qty(".").is_err());
        Ok(())
    }

    #[test]
    fn ids_that_do_not_run_on_are_gaps_except_at_a_whole_book() {
        let day = Day::parse("2026-10-20").expect("a day");
        let t = 1_792_454_400_000;
        let mut scan = Scan::new(ArchiveFormat::BybitOrderbook);
        scan.record(day, t, 10, true);
        scan.record(day, t + 100, 11, false);
        scan.record(day, t + 200, 13, false);
        scan.record(day, t + 300, 1, true);
        scan.record(day, t + 400, 2, false);
        scan.record(day, t - 1, 3, false);
        let integrity = scan.integrity();
        assert_eq!(
            (
                integrity.records,
                integrity.id_gaps,
                integrity.snapshots,
                integrity.outside_day
            ),
            (6, 1, Some(2), 1)
        );
        let mut trades = Scan::new(ArchiveFormat::BinanceAggTrades);
        trades.trade_ids(1, 3);
        trades.trade_ids(4, 4);
        trades.trade_ids(6, 7);
        assert_eq!(trades.integrity().trade_id_gaps, Some(1));
    }
}
