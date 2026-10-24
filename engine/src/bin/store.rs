//! Upkeep for recorded runs, and archive imports.
//!
//! ```text
//! store recover <run dir>              close a crashed run's newest segment, cutting a torn tail
//! store rotate <run dir>               write Parquet for every closed segment (safe to re-run)
//! store replay <run dir> [--parquet]   replay the log (or its Parquet) and show what it rebuilt
//! store import <archive> [--data-dir DIR] [--url URL] [--tables T,...]
//!                                      a downloaded archive day into its run (safe to re-run)
//! ```
//!
//! An import reads Bybit's order-book archive (`<date>_<SYMBOL>_ob<N>.data.zip`) or Binance's
//! aggregated trades (`<SYMBOL>-aggTrades-<date>.zip`) into `<data dir>/market/<run_id>/`, the
//! data dir being `--data-dir`, `TP_DATA_DIR` or the per-user default. By default it writes book
//! samples for an order book and trades for trades; `--tables` names others (`frames`,
//! `depth_updates`, `snapshots`).
//!
//! `cargo run -p engine --bin store -- rotate <run dir>`

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use engine::market::pipeline::Replayed;
use engine::market::sync::BookSync;
use engine::store::import::{ImportConfig, import};
use engine::store::replay::{replay_log, replay_parquet};
use engine::store::rotate::rotate;
use engine::store::run;
use engine::store::segment::recover_run;

const USAGE: &str = "usage: store recover <run dir> | store rotate <run dir> | store replay <run dir> [--parquet] \
                     | store import <archive> [--data-dir DIR] [--url URL] [--tables T,...]";

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Recover(PathBuf),
    Rotate(PathBuf),
    Replay {
        run_dir: PathBuf,
        parquet: bool,
    },
    Import {
        archive: PathBuf,
        data_dir: Option<PathBuf>,
        url: Option<String>,
        tables: Option<Vec<String>>,
    },
}

fn parse_args(args: &[String]) -> Result<Command, String> {
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    if let ["import", archive, options @ ..] = args.as_slice() {
        return parse_import(archive, options);
    }
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

fn parse_import(archive: &str, options: &[&str]) -> Result<Command, String> {
    let (mut data_dir, mut url, mut tables) = (None, None, None);
    let mut it = options.iter();
    while let Some(&flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} wants a value"))?;
        match flag {
            "--data-dir" if data_dir.is_none() => data_dir = Some(PathBuf::from(value)),
            "--url" if url.is_none() => url = Some((*value).to_owned()),
            "--tables" if tables.is_none() => {
                tables = Some(value.split(',').map(str::to_owned).collect());
            }
            _ => return Err(USAGE.to_owned()),
        }
    }
    Ok(Command::Import {
        archive: PathBuf::from(archive),
        data_dir,
        url,
        tables,
    })
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
        Command::Import {
            archive,
            data_dir,
            url,
            tables,
        } => {
            let config = ImportConfig {
                data_dir: data_dir.clone().unwrap_or_else(run::data_dir),
                url: url.clone(),
                tables: tables.clone(),
            };
            let imported = import(archive, &config, &mut |_| {})?;
            if imported.done_before {
                println!(
                    "{}: imported already, nothing to do",
                    imported.run_dir.display()
                );
            }
            for path in &imported.written {
                println!("wrote {}", path.display());
            }
            let rows: Vec<String> = imported
                .rows
                .iter()
                .map(|(table, rows)| format!("{table}={rows}"))
                .collect();
            println!("{}: {}", imported.run_dir.display(), rows.join(" "));
            let check = &imported.integrity;
            println!(
                "records={} outside_day={} id_gaps={} trade_id_gaps={} snapshots={} errors={} resyncs={} crossed={} unsynced_samples={}",
                check.records,
                check.outside_day,
                check.id_gaps,
                check
                    .trade_id_gaps
                    .map_or_else(|| "-".to_owned(), |n| n.to_string()),
                check
                    .snapshots
                    .map_or_else(|| "-".to_owned(), |n| n.to_string()),
                check.errors,
                check.resyncs,
                check.crossed,
                check.unsynced_samples
            );
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
        assert_eq!(
            parse(&[
                "import",
                "a.zip",
                "--tables",
                "frames,book_samples",
                "--data-dir",
                "d"
            ]),
            Ok(Command::Import {
                archive: PathBuf::from("a.zip"),
                data_dir: Some(PathBuf::from("d")),
                url: None,
                tables: Some(vec!["frames".to_owned(), "book_samples".to_owned()]),
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
            &["import"],
            &["import", "a.zip", "--url"],
            &["import", "a.zip", "--speed", "1"],
            &["import", "a.zip", "--url", "x", "--url", "y"],
        ] {
            assert!(parse(raw).is_err(), "{raw:?}");
        }
    }
}
