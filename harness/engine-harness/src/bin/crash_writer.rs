//! Writes a log until it is killed: the production writer, with every frame split in two and a
//! pause between the halves so that a kill lands mid-frame, and small segments so that kills
//! land mid-rotation too. Entry `i` carries [`crash_payload`]`(i)`. The crash-recovery test
//! kills it at a random moment and checks what recovery keeps.
//!
//! With `--end-with-stdin` it also ends when its stdin closes. The test passes it with a pipe
//! whose other end closes when the test's process ends, however that ends, so a killed test
//! runner never leaves it writing forever. Without it stdin is never read, so a writer started
//! with stdin closed or null writes until it is killed, rather than ending at once with nothing
//! written and exit 0, as if it had run (#93). Run by hand, end it with Ctrl-C.
//!
//! `crash-writer [--end-with-stdin] <run dir>`

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use engine::store::frame::{Kind, Meta};
use engine::store::now_ns;
use engine::store::run::Producer;
use engine::store::writer::{Entry, LogConfig, LogWriter};
use engine::venue::Venue;
use engine_harness::crash_payload;

/// Where to write, and whether to end when stdin closes.
#[derive(Debug, PartialEq, Eq)]
struct Args<'a> {
    run_dir: &'a Path,
    end_with_stdin: bool,
}

fn parse_args(args: &[String]) -> Result<Args<'_>, String> {
    let (run_dir, end_with_stdin) = match args {
        [run_dir] => (run_dir, false),
        [flag, run_dir] if flag == "--end-with-stdin" => (run_dir, true),
        _ => return Err("usage: crash-writer [--end-with-stdin] <run dir>".to_owned()),
    };
    Ok(Args {
        run_dir: Path::new(run_dir),
        end_with_stdin,
    })
}

fn config() -> LogConfig {
    LogConfig {
        max_segment_bytes: 16 * 1024,
        split_frames: Some(Duration::from_millis(1)),
        ..LogConfig::default()
    }
}

/// On a runtime of its own so that it appends through the live loop's own `append`.
#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Args {
        run_dir,
        end_with_stdin,
    } = match parse_args(&args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    let meta = Meta {
        run_id: "crash".to_owned(),
        first_seq: 0,
        producer: Producer {
            name: "crash-writer".to_owned(),
            version: engine::version().to_owned(),
            git_sha: None,
            dirty: None,
        },
        venue: Venue::BinanceUsdm,
        scales: BTreeMap::new(),
    };
    if end_with_stdin {
        std::thread::spawn(|| {
            let _ = std::io::copy(&mut std::io::stdin(), &mut std::io::sink());
            std::process::exit(0);
        });
    }
    let mut log = match LogWriter::start(run_dir, meta, config()) {
        Ok(log) => log,
        Err(error) => {
            eprintln!("crash-writer: {error}");
            return ExitCode::FAILURE;
        }
    };
    for i in 0.. {
        let entry = Entry {
            kind: Kind::Trade,
            source_ts_ns: 0,
            capture_ts_ns: now_ns(),
            payload: crash_payload(i),
        };
        if let Err(error) = log.append(entry).await {
            eprintln!("crash-writer: {error}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takes_a_run_dir_and_the_stdin_flag_before_it() {
        let parse = |raw: &[&str]| {
            let raw: Vec<String> = raw.iter().map(|s| (*s).to_owned()).collect();
            parse_args(&raw).map(|args| (args.run_dir.to_owned(), args.end_with_stdin))
        };
        assert_eq!(parse(&["r"]), Ok((Path::new("r").to_owned(), false)));
        assert_eq!(
            parse(&["--end-with-stdin", "r"]),
            Ok((Path::new("r").to_owned(), true))
        );
        assert!(parse(&[]).is_err());
        assert!(parse(&["a", "b"]).is_err());
        assert!(parse(&["r", "--end-with-stdin"]).is_err());
    }

    #[test]
    fn frames_are_split_and_segments_small() {
        let config = config();
        assert!(config.split_frames.is_some());
        assert!(config.max_segment_bytes <= 64 * 1024);
    }
}
