//! Upkeep for recorded runs.
//!
//! ```text
//! store recover <run dir>              close a crashed run's newest segment, cutting a torn tail
//! store rotate <run dir>               write Parquet for every closed segment (safe to re-run)
//! store replay <run dir> [--parquet]   replay the log (or its Parquet) and show what it rebuilt
//! ```
//!
//! `cargo run -p engine --bin store -- rotate <run dir>`

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use engine::market::pipeline::Replayed;
use engine::market::sync::BookSync;
use engine::store::replay::{replay_log, replay_parquet};
use engine::store::rotate::rotate;
use engine::store::segment::recover_run;

const USAGE: &str =
    "usage: store recover <run dir> | store rotate <run dir> | store replay <run dir> [--parquet]";

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Recover(PathBuf),
    Rotate(PathBuf),
    Replay { run_dir: PathBuf, parquet: bool },
}

fn parse_args(args: &[String]) -> Result<Command, String> {
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["recover", run_dir] => Ok(Command::Recover(PathBuf::from(run_dir))),
        ["rotate", run_dir] => Ok(Command::Rotate(PathBuf::from(run_dir))),
        ["replay", run_dir] => Ok(Command::Replay {
            run_dir: PathBuf::from(run_dir),
            parquet: false,
        }),
        ["replay", run_dir, "--parquet"] => Ok(Command::Replay {
            run_dir: PathBuf::from(run_dir),
            parquet: true,
        }),
        _ => Err(USAGE.to_owned()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = match parse_args(&args) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    match run(&command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("store failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(command: &Command) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::Recover(run_dir) => match recover_run(run_dir)? {
            None => println!("nothing to recover: the newest segment is closed"),
            Some(recovery) => println!(
                "closed {}: kept {} frames, cut {} bytes{}",
                recovery.path.display(),
                recovery.frames,
                recovery.cut,
                recovery
                    .cause
                    .map(|cause| format!(" ({cause})"))
                    .unwrap_or_default()
            ),
        },
        Command::Rotate(run_dir) => {
            let rotation = rotate(run_dir)?;
            for path in &rotation.written {
                println!("wrote {}", path.display());
            }
            println!(
                "{} closed segments, {} files written, {} already there, {} temp files cleaned, {} frames gave no rows",
                rotation.segments,
                rotation.written.len(),
                rotation.skipped,
                rotation.cleaned,
                rotation.unparsed
            );
            if let Some(open) = rotation.open {
                println!("left for later: {} has no end marker yet", open.display());
            }
        }
        Command::Replay { run_dir, parquet } => {
            let replayed = if *parquet {
                replay_parquet(run_dir, None, &mut |_| {})?
            } else {
                replay_log(run_dir, None, &mut |_| {})?
            };
            print_replay(run_dir, &replayed);
        }
    }
    Ok(())
}

fn print_replay(run_dir: &Path, replayed: &Replayed) {
    let report = &replayed.report;
    let feed = replayed.pipeline.feed();
    let stats = feed.stats();
    println!(
        "{}: {} frames, {} inputs, {} the feed could not use",
        run_dir.display(),
        report.frames,
        report.inputs,
        report.errors
    );
    for (seq, error) in &report.first_errors {
        println!("  seq {seq}: {error}");
    }
    if let Some(unclosed) = &report.unclosed {
        println!("  {unclosed}");
    }
    println!(
        "applied={} buffered={} stale={} snapshots={} resyncs={} gaps={} crossed={} trades={}",
        stats.applied,
        stats.buffered,
        stats.dropped_stale,
        stats.snapshots,
        stats.resyncs,
        stats.sequence_gaps,
        stats.crossed,
        stats.trades
    );
    for symbol in feed.symbols() {
        let synced = feed.sync(symbol).is_some_and(BookSync::is_synced);
        let depth = feed.book(symbol).map(engine::market::book::Book::depth);
        println!("{symbol}: synced={synced} levels={depth:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &[&str]) -> Result<Command, String> {
        parse_args(&raw.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn takes_one_command_and_a_run_dir() {
        assert_eq!(
            parse(&["rotate", "r"]),
            Ok(Command::Rotate(PathBuf::from("r")))
        );
        assert_eq!(
            parse(&["recover", "r"]),
            Ok(Command::Recover(PathBuf::from("r")))
        );
        assert_eq!(
            parse(&["replay", "r", "--parquet"]),
            Ok(Command::Replay {
                run_dir: PathBuf::from("r"),
                parquet: true
            })
        );
    }

    #[test]
    fn refuses_anything_else() {
        for raw in [
            &[][..],
            &["rotate"],
            &["rotate", "a", "b"],
            &["delete", "r"],
            &["replay", "r", "--duckdb"],
        ] {
            assert!(parse(raw).is_err(), "{raw:?}");
        }
    }
}
