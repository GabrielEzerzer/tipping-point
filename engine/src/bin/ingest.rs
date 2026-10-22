//! Connects to Binance, keeps the BTC and ETH books, and logs top-of-book plus every resync.
//!
//! `cargo run -p engine --bin ingest -- [--seconds N]`

use std::process::ExitCode;
use std::time::Duration;

use engine::binance::decimal::format_units;
use engine::binance::feed::Feed;
use engine::binance::live::{LiveConfig, LiveEvent, run_live, utc_clock};
use engine::binance::sync::BookSync;

fn parse_seconds(args: &[String]) -> Result<Option<u64>, String> {
    match args {
        [] => Ok(None),
        [flag, value] if flag == "--seconds" => value
            .parse()
            .map(Some)
            .map_err(|_| format!("--seconds wants a whole number, got {value:?}")),
        _ => Err("usage: ingest [--seconds N]".to_owned()),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seconds = match parse_seconds(&args) {
        Ok(seconds) => seconds,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    let config = LiveConfig {
        duration: seconds.map(Duration::from_secs),
        ..LiveConfig::default()
    };
    let result = run_live(config, |event| match event {
        LiveEvent::Report { feed, stats, .. } => print_tops(feed, stats.snapshot_fetches),
        other => println!("{} {other:?}", utc_clock()),
    })
    .await;
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ingest failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn print_tops(feed: &Feed, snapshot_fetches: u64) {
    let counters = feed.stats();
    for symbol in feed.symbols() {
        let Some(scales) = feed.scales(symbol) else {
            continue;
        };
        let state = match feed.sync(symbol) {
            Some(sync) if sync.is_synced() => "synced",
            Some(sync) if sync.book().is_some() => "bridging",
            _ => "buffering",
        };
        let top = feed.book(symbol).map_or_else(
            || "no book".to_owned(),
            |book| {
                let side = |level: Option<engine::binance::sync::Level>| {
                    level.map_or_else(
                        || "-".to_owned(),
                        |l| {
                            format!(
                                "{} x {}",
                                format_units(l.price, scales.price_decimals),
                                format_units(l.qty, scales.qty_decimals)
                            )
                        },
                    )
                };
                let (bids, asks) = book.depth();
                format!(
                    "bid {} | ask {} | levels {bids}/{asks}{}",
                    side(book.best_bid()),
                    side(book.best_ask()),
                    if book.is_crossed() { " CROSSED" } else { "" }
                )
            },
        );
        println!("{} {symbol:<8} {state:<9} {top}", utc_clock());
    }
    println!(
        "{} applied={} trades={} resyncs={} gaps={} crossed={} snapshots={} synced={}",
        utc_clock(),
        counters.applied,
        counters.trades,
        counters.resyncs,
        counters.sequence_gaps,
        counters.crossed,
        snapshot_fetches,
        feed.symbols()
            .filter(|s| feed.sync(s).is_some_and(BookSync::is_synced))
            .count()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &[&str]) -> Result<Option<u64>, String> {
        parse_seconds(&raw.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn runs_until_stopped_or_for_the_given_seconds() {
        assert_eq!(parse(&[]), Ok(None));
        assert_eq!(parse(&["--seconds", "60"]), Ok(Some(60)));
    }

    #[test]
    fn refuses_anything_else() {
        assert_eq!(
            parse(&["--seconds", "1.5"]),
            Err("--seconds wants a whole number, got \"1.5\"".to_owned())
        );
        assert!(parse(&["--seconds"]).is_err());
        assert!(parse(&["--minutes", "1"]).is_err());
        assert!(parse(&["--seconds", "1", "--seconds", "2"]).is_err());
    }
}
