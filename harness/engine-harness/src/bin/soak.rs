//! Soak instrument: runs the live pipeline and logs gaps, resyncs, reconnects and frames
//! dropped under backpressure, once per period, to stdout and to a line-per-period file
//! under `harness/out/`.
//!
//! `cargo run -p engine-harness --bin soak -- [--hours H | --seconds N] [--report-secs S] [--out PATH]`

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use engine::binance::feed::Stats;
use engine::binance::live::{LiveConfig, LiveEvent, LiveStats, run_live, utc_clock};
use engine::binance::recording::now_ms;
use engine::binance::sync::BookSync;

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
    run_live(config, |event| match event {
        LiveEvent::Report {
            elapsed,
            feed,
            stats,
        } => {
            period += 1;
            let feed_stats = *feed.stats();
            let synced = feed
                .symbols()
                .filter(|s| feed.sync(s).is_some_and(BookSync::is_synced))
                .count();
            let line = format!(
                "{} period={period} elapsed={}s gaps={} (+{}) resyncs={} (+{}) reconnects={} (+{}) \
                 dropped={} (+{}) snapshots={} (+{}) crossed={} applied={} trades={} synced={}/{}",
                utc_clock(),
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
                feed_stats.crossed,
                feed_stats.applied,
                feed_stats.trades,
                synced,
                feed.symbols().count(),
            );
            println!("{line}");
            if let Err(error) = writeln!(file, "{line}").and_then(|()| file.flush()) {
                eprintln!("could not write {}: {error}", args.out.display());
            }
            previous = (feed_stats, *stats);
        }
        other => println!("{} {other:?}", utc_clock()),
    })
    .await?;
    Ok(())
}
