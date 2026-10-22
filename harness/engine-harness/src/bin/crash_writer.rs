//! Writes a log until it is killed: the production writer, with every frame split in two and a
//! pause between the halves so that a kill lands mid-frame, and small segments so that kills
//! land mid-rotation too. Entry `i` carries [`crash_payload`]`(i)`. The crash-recovery test
//! kills it at a random moment and checks what recovery keeps.
//!
//! `crash-writer <run dir>`

use std::collections::BTreeMap;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use engine::store::frame::{Kind, Meta};
use engine::store::now_ns;
use engine::store::run::Producer;
use engine::store::writer::{Entry, LogConfig, LogWriter};
use engine_harness::crash_payload;

fn parse_args(args: &[String]) -> Result<&Path, String> {
    match args {
        [run_dir] => Ok(Path::new(run_dir)),
        _ => Err("usage: crash-writer <run dir>".to_owned()),
    }
}

fn config() -> LogConfig {
    LogConfig {
        max_segment_bytes: 16 * 1024,
        split_frames: Some(Duration::from_millis(1)),
        ..LogConfig::default()
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let run_dir = match parse_args(&args) {
        Ok(run_dir) => run_dir,
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
        scales: BTreeMap::new(),
    };
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
        if let Err(error) = log.append_blocking(entry) {
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
    fn takes_exactly_a_run_dir() {
        assert_eq!(parse_args(&["r".to_owned()]), Ok(Path::new("r")));
        assert!(parse_args(&[]).is_err());
        assert!(parse_args(&["a".to_owned(), "b".to_owned()]).is_err());
    }

    #[test]
    fn frames_are_split_and_segments_small() {
        let config = config();
        assert!(config.split_frames.is_some());
        assert!(config.max_segment_bytes <= 64 * 1024);
    }
}
