//! Soak instrument: runs the live pipeline and logs gaps, resyncs, reconnects and frames
//! dropped under backpressure, once per period, to stdout and to a line-per-period file
//! under `harness/out/`.
//!
//! `cargo run --release -p engine-harness --bin soak -- [--hours H | --seconds N] [--report-secs S] [--out PATH]`

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use engine::market::feed::{Feed, Stats};
use engine::market::live::{LiveEvent, LiveStats, utc_clock};
use engine::market::recording::now_ms;
use engine::market::sync::BookSync;
use engine::venue::binance::live::{LiveConfig, run_live};

struct Args {
    duration: Duration,
    report_every: Duration,
    out: PathBuf,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    // From this crate's manifest, so the log lands in the ignored `harness/out/` from any cwd.
    let mut parsed = Args {
        duration: Duration::from_hours(24),
        report_every: Duration::from_hours(1),
        out: Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../out")
            .join(format!("soak-{}.log", now_ms())),
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} wants a value"))?;
        let number = || {
            value
                .parse::<u64>()
                .map_err(|_| format!("{flag} wants a whole number, got {value:?}"))
        };
        match flag.as_str() {
            "--hours" => parsed.duration = Duration::from_hours(number()?),
            "--seconds" => parsed.duration = Duration::from_secs(number()?),
            "--report-secs" => parsed.report_every = Duration::from_secs(number()?),
            "--out" => parsed.out = PathBuf::from(value),
            _ => {
                return Err(
                    "usage: soak [--hours H | --seconds N] [--report-secs S] [--out PATH]"
                        .to_owned(),
                );
            }
        }
    }
    Ok(parsed)
}

#[tokio::main]
async fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&raw) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    match soak(&args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("soak failed: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn soak(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(dir) = args.out.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut file: File = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.out)?;
    println!(
        "{} soak for {:?}, reporting every {:?} to {}",
        utc_clock(),
        args.duration,
        args.report_every,
        args.out.display()
    );

    let config = LiveConfig {
        duration: Some(args.duration),
        report_every: args.report_every,
        ..LiveConfig::default()
    };
    let mut previous = (Stats::default(), LiveStats::default());
    let mut period = 0u64;
    run_live(config, |event: LiveEvent<'_>| match event {
        LiveEvent::Report {
            elapsed,
            feed,
            stats,
        } => {
            period += 1;
            let line = format!(
                "{} {}",
                utc_clock(),
                report_line(period, elapsed, feed, stats, &previous)
            );
            println!("{line}");
            if let Err(error) = writeln!(file, "{line}").and_then(|()| file.flush()) {
                eprintln!("could not write {}: {error}", args.out.display());
            }
            previous = (*feed.stats(), *stats);
        }
        LiveEvent::Applied { .. } => {}
        other => println!("{} {other:?}", utc_clock()),
    })
    .await?;
    Ok(())
}

/// One period's line: the counters, most with their change since the `previous` line, and how
/// many symbols are in sync. Frames that could not be read or converted show as `parse_errors`,
/// so a symbol that stopped syncing for good (every frame unreadable, or a finer tick size) says
/// why rather than only `synced=1/2` with `applied` flat.
fn report_line(
    period: u64,
    elapsed: Duration,
    feed: &Feed,
    stats: &LiveStats,
    previous: &(Stats, LiveStats),
) -> String {
    let feed_stats = feed.stats();
    let synced = feed
        .symbols()
        .filter(|s| feed.sync(s).is_some_and(BookSync::is_synced))
        .count();
    format!(
        "period={period} elapsed={}s gaps={} (+{}) resyncs={} (+{}) reconnects={} (+{}) \
         dropped={} (+{}) snapshots={} (+{}) parse_errors={} (+{}) crossed={} applied={} \
         trades={} synced={}/{}",
        elapsed.as_secs(),
        feed_stats.sequence_gaps,
        feed_stats.sequence_gaps - previous.0.sequence_gaps,
        feed_stats.resyncs,
        feed_stats.resyncs - previous.0.resyncs,
        stats.depth_reconnects + stats.trade_reconnects,
        stats.depth_reconnects + stats.trade_reconnects
            - previous.1.depth_reconnects
            - previous.1.trade_reconnects,
        stats.depth_frames_dropped + stats.trade_frames_dropped,
        stats.depth_frames_dropped + stats.trade_frames_dropped
            - previous.1.depth_frames_dropped
            - previous.1.trade_frames_dropped,
        stats.snapshot_fetches,
        stats.snapshot_fetches - previous.1.snapshot_fetches,
        feed_stats.parse_errors,
        feed_stats.parse_errors - previous.0.parse_errors,
        feed_stats.crossed,
        feed_stats.applied,
        feed_stats.trades,
        synced,
        feed.symbols().count(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &[&str]) -> Result<Args, String> {
        parse_args(&raw.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn soaks_a_day_reporting_hourly_into_harness_out_by_default() {
        let args = parse(&[]).expect("no flags is valid");
        assert_eq!(args.duration, Duration::from_hours(24));
        assert_eq!(args.report_every, Duration::from_hours(1));
        let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../out");
        assert_eq!(args.out.parent(), Some(out.as_path()));
    }

    #[test]
    fn takes_its_flags_and_refuses_anything_else() {
        let args = parse(&["--seconds", "90", "--report-secs", "30", "--out", "s.log"])
            .expect("valid flags");
        assert_eq!(args.duration, Duration::from_secs(90));
        assert_eq!(args.report_every, Duration::from_secs(30));
        assert_eq!(args.out, PathBuf::from("s.log"));
        let hours = parse(&["--hours", "2"]).expect("valid flags");
        assert_eq!(hours.duration, Duration::from_hours(2));
        assert_eq!(
            parse(&["--hours", "-1"]).err().as_deref(),
            Some("--hours wants a whole number, got \"-1\"")
        );
        assert_eq!(
            parse(&["--out"]).err().as_deref(),
            Some("--out wants a value")
        );
        assert!(parse(&["--days", "1"]).is_err());
    }

    /// A feed that stopped syncing because its frames no longer read says so on the periodic
    /// line, with how many failed since the last one (#43).
    #[test]
    fn the_periodic_line_counts_unreadable_frames() {
        let scales = engine::market::Scales {
            price_decimals: 1,
            qty_decimals: 3,
        };
        let mut feed = Feed::new(
            engine::venue::Venue::BinanceUsdm,
            std::collections::BTreeMap::from([("BTCUSDT".to_owned(), scales)]),
        );
        for _ in 0..3 {
            assert!(feed.on_text("not json").is_err());
        }
        let stats = LiveStats::default();
        let start = (Stats::default(), stats);
        let first = report_line(1, Duration::from_secs(60), &feed, &stats, &start);
        assert!(first.contains(" parse_errors=3 (+3) "), "{first}");
        assert!(first.ends_with(" synced=0/1"), "{first}");

        let since = (*feed.stats(), stats);
        assert!(feed.on_text("not json either").is_err());
        let second = report_line(2, Duration::from_secs(120), &feed, &stats, &since);
        assert!(second.contains(" parse_errors=4 (+1) "), "{second}");
    }
}
