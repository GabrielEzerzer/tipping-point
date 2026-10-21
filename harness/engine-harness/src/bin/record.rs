//! Records raw depth frames and the snapshots they pair with, so a replay can drive the same
//! pipeline offline. Output goes to the gitignored `harness/fixtures/recorded/` and is never
//! committed.
//!
//! `cargo run -p engine-harness --bin record -- [--seconds N] [--out PATH]`

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use engine::binance::live::install_crypto_provider;
use engine::binance::recording::{Record, now_ms, read_records, replay, write_record};
use engine::binance::rest::Rest;
use engine::binance::ws::{self, Inbound};
use tokio::sync::mpsc;
use tokio::time::sleep;

const SYMBOLS: [&str; 2] = ["BTCUSDT", "ETHUSDT"];
/// Frames queue here while the loop fetches snapshots, so nothing is dropped during that.
const CHANNEL_CAPACITY: usize = 8192;
/// The stream must be open before the snapshot is asked for, or the snapshot cannot bridge.
const SNAPSHOT_AFTER: Duration = Duration::from_secs(2);

struct Args {
    seconds: u64,
    out: PathBuf,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    // From this crate's manifest, not the cwd: `harness/fixtures/recorded/` is the one place
    // `.gitignore` keeps recordings out of the repo and the replay test looks.
    let mut parsed = Args {
        seconds: 60,
        out: Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../fixtures/recorded")
            .join(format!("depth-{}.jsonl", now_ms())),
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
            _ => return Err("usage: record [--seconds N] [--out PATH]".to_owned()),
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
    install_crypto_provider();
    let symbols: Vec<String> = SYMBOLS.iter().map(|s| (*s).to_owned()).collect();
    let rest = Rest::new()?;
    let scales = rest.fetch_scales(&symbols).await?;

    if let Some(dir) = args.out.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut out = BufWriter::new(File::create(&args.out)?);
    write_record(&mut out, &Record::Scales { scales })?;

    let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
    let url = ws::stream_url(ws::DEPTH_HOST, &symbols, "depth@100ms");
    let reader = tokio::spawn(ws::run_reader(url, ws::DEFAULT_MAX_CONNECTION_AGE, tx));

    let stop = sleep(Duration::from_secs(args.seconds));
    tokio::pin!(stop);
    let snapshot_due = sleep(SNAPSHOT_AFTER);
    tokio::pin!(snapshot_due);
    let mut snapshots_taken = false;
    let (mut frames, mut dropped, mut disconnects) = (0u64, 0u64, 0u64);

    loop {
        tokio::select! {
            Some(inbound) = rx.recv() => match inbound {
                Inbound::Text { text, dropped_before } => {
                    dropped += dropped_before;
                    write_record(&mut out, &Record::Frame { t_ms: now_ms(), text })?;
                    frames += 1;
                }
                Inbound::Connected { attempt } => eprintln!("connected (attempt {attempt})"),
                Inbound::ConnectFailed { error } => eprintln!("connect failed: {error}"),
                Inbound::Disconnected { reason, dropped_before } => {
                    dropped += dropped_before;
                    disconnects += 1;
                    eprintln!("disconnected: {reason} (the recording now has a gap)");
                }
            },
            () = &mut snapshot_due, if !snapshots_taken => {
                snapshots_taken = true;
                for symbol in &symbols {
                    let body = rest.fetch_snapshot(symbol).await?;
                    eprintln!("snapshot {symbol}: {} bytes", body.len());
                    write_record(&mut out, &Record::Snapshot { t_ms: now_ms(), symbol: symbol.clone(), body })?;
                }
            }
            () = &mut stop => break,
        }
    }
    reader.abort();
    out.flush()?;
    drop(out);

    eprintln!(
        "wrote {} ({frames} frames, {dropped} dropped, {disconnects} disconnects)",
        args.out.display()
    );
    let feed = replay(read_records(BufReader::new(File::open(&args.out)?))?)?;
    let stats = feed.stats();
    eprintln!(
        "replay check: applied={} buffered={} stale={} resyncs={} crossed={} synced={}",
        stats.applied,
        stats.buffered,
        stats.dropped_stale,
        stats.resyncs,
        stats.crossed,
        feed.symbols()
            .filter(|s| feed
                .sync(s)
                .is_some_and(engine::binance::sync::BookSync::is_synced))
            .count()
    );
    Ok(())
}
