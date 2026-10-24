//! Connects to Binance, keeps the BTC and ETH books, and logs top-of-book plus every resync.
//! With `--record` it also records every input to a new run under the data dir: `TP_DATA_DIR`,
//! or else a per-user folder outside the repository. Ctrl-C ends a run cleanly.
//!
//! `cargo run --release -p engine --bin ingest -- [--seconds N] [--record]`

use std::process::ExitCode;
use std::time::Duration;

use engine::market::decimal::format_units;
use engine::market::feed::Feed;
use engine::market::live::{LiveEvent, utc_clock};
use engine::market::sync::BookSync;
use engine::store::run;
use engine::store::writer::LogConfig;
use engine::venue::binance::live::{LiveConfig, Recording, run_live};

const USAGE: &str = "usage: ingest [--seconds N] [--record]";

#[derive(Debug, PartialEq, Eq)]
struct Args {
    seconds: Option<u64>,
    record: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        seconds: None,
        record: false,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--seconds" if parsed.seconds.is_none() => {
                let value = it.next().ok_or_else(|| USAGE.to_owned())?;
                let seconds = value
                    .parse()
                    .map_err(|_| format!("--seconds wants a whole number, got {value:?}"))?;
                parsed.seconds = Some(seconds);
            }
            "--record" if !parsed.record => parsed.record = true,
            _ => return Err(USAGE.to_owned()),
        }
    }
    Ok(parsed)
}

/// `btceth` for BTCUSDT and ETHUSDT.
fn run_tag(symbols: &[String]) -> String {
    symbols
        .iter()
        .map(|s| s.trim_end_matches("USDT").to_lowercase())
        .collect()
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    let mut config = LiveConfig {
        duration: args.seconds.map(Duration::from_secs),
        ..LiveConfig::default()
    };
    if args.record {
        let run_dir = match run::market_run_dir(&run::data_dir(), &run_tag(&config.symbols)) {
            Ok(run_dir) => run_dir,
            Err(message) => {
                eprintln!("{message}");
                return ExitCode::FAILURE;
            }
        };
        println!("{} recording to {}", utc_clock(), run_dir.display());
        config.record = Some(Recording {
            run_dir,
            log: LogConfig::default(),
        });
    }
    let result = run_live(config, |event: LiveEvent<'_>| match event {
        LiveEvent::Report { feed, stats, .. } => print_tops(feed, stats.snapshot_fetches),
        LiveEvent::Applied { .. } => {}
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
                let side = |level: Option<engine::market::sync::Level>| {
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

    fn parse(raw: &[&str]) -> Result<Args, String> {
        parse_args(&raw.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn runs_until_stopped_or_for_the_given_seconds_recording_on_request() {
        assert_eq!(
            parse(&[]),
            Ok(Args {
                seconds: None,
                record: false
            })
        );
        assert_eq!(parse(&["--seconds", "60"]).map(|a| a.seconds), Ok(Some(60)));
        assert_eq!(
            parse(&["--record", "--seconds", "5"]),
            Ok(Args {
                seconds: Some(5),
                record: true
            })
        );
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
        assert!(parse(&["--record", "--record"]).is_err());
    }

    #[test]
    fn the_run_tag_names_the_coins() {
        let symbols = ["BTCUSDT".to_owned(), "ETHUSDT".to_owned()];
        assert_eq!(run_tag(&symbols), "btceth");
    }
}
