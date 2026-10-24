//! Records a live session the way production records one: the live loop with its log on, into
//! a run under the gitignored `harness/fixtures/recorded/` (`market/<run_id>/`), plus the live
//! loop's own transcript (`live-transcript.json`) for the replay-equals-live check. It then
//! replays the log and says whether the replay matched. Recordings are never committed.
//!
//! `cargo run --release -p engine-harness --bin record -- [--seconds N] [--out DIR]`

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use engine::market::live::{LiveEvent, utc_clock};
use engine::store::replay::replay_log;
use engine::store::run;
use engine::store::writer::LogConfig;
use engine::venue::binance::live::{LiveConfig, Recording, run_live};
use engine_harness::transcript::Transcript;

/// Where the live loop's transcript goes, inside the run folder.
const TRANSCRIPT: &str = "live-transcript.json";

struct Args {
    seconds: u64,
    out: PathBuf,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    // From this crate's manifest, not the cwd: `harness/fixtures/recorded/` is the one place
    // `.gitignore` keeps recordings out of the repo and the replay tests look.
    let mut parsed = Args {
        seconds: 60,
        out: Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/recorded"),
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} wants a value"))?;
        match flag.as_str() {
            "--seconds" => {
                parsed.seconds = value
                    .parse()
                    .map_err(|_| format!("--seconds wants a whole number, got {value:?}"))?;
            }
            "--out" => parsed.out = PathBuf::from(value),
            _ => return Err("usage: record [--seconds N] [--out DIR]".to_owned()),
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
    match record(&args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("record failed: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn record(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let run_dir = run::market_run_dir(&args.out, "btceth")?;
    eprintln!("{} recording to {}", utc_clock(), run_dir.display());
    let config = LiveConfig {
        duration: Some(Duration::from_secs(args.seconds)),
        report_every: Duration::from_secs(10),
        record: Some(Recording {
            run_dir: run_dir.clone(),
            log: LogConfig::default(),
        }),
        ..LiveConfig::default()
    };
    let mut live = Transcript::default();
    run_live(config, |event: LiveEvent<'_>| {
        live.observe(&event);
        match event {
            LiveEvent::Applied { .. } => {}
            LiveEvent::Report { feed, .. } => {
                let stats = feed.stats();
                eprintln!(
                    "{} applied={} trades={} resyncs={}",
                    utc_clock(),
                    stats.applied,
                    stats.trades,
                    stats.resyncs
                );
            }
            other => eprintln!("{} {other:?}", utc_clock()),
        }
    })
    .await?;
    live.save(&run_dir.join(TRANSCRIPT))?;

    let mut replayed = Transcript::default();
    let replay = replay_log(&run_dir, None, &mut |event| replayed.observe(&event))?;
    eprintln!(
        "wrote {} ({} frames, {} inputs, {} the feed could not use)",
        run_dir.display(),
        replay.report.frames,
        replay.report.inputs,
        replay.report.errors
    );
    match live.first_difference(&replayed) {
        None => {
            eprintln!(
                "replay check: the log replays exactly as it ran live ({} inputs)",
                live.steps.len()
            );
            Ok(())
        }
        Some(difference) => Err(format!("replay check failed at {difference}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &[&str]) -> Result<Args, String> {
        parse_args(&raw.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn records_a_minute_into_the_gitignored_dir_by_default() {
        let args = parse(&[]).expect("no flags is valid");
        assert_eq!(args.seconds, 60);
        let recorded = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/recorded");
        assert_eq!(args.out, recorded);
    }

    #[test]
    fn takes_seconds_and_out_and_refuses_anything_else() {
        let args = parse(&["--seconds", "5", "--out", "data"]).expect("valid flags");
        assert_eq!(args.seconds, 5);
        assert_eq!(args.out, PathBuf::from("data"));
        assert_eq!(
            parse(&["--seconds"]).err().as_deref(),
            Some("--seconds wants a value")
        );
        assert_eq!(
            parse(&["--seconds", "five"]).err().as_deref(),
            Some("--seconds wants a whole number, got \"five\"")
        );
        assert!(parse(&["--minutes", "5"]).is_err());
    }
}
