//! The engine as its own process: the live feed and the local API on 127.0.0.1 (see
//! `engine::api`). The app is a WebSocket client and the Python worker a third process. It
//! prints `serve listening on http://127.0.0.1:<port>` once it accepts connections, so a parent
//! that passed `--port 0` can read the port it got.
//!
//! The feed is Binance's live streams, or with `--script` a recorded script (the JSON-lines
//! format of the synthetic fixtures) played back at `--speed` times its own clock (0 for as fast
//! as possible), after which the API keeps serving. With `--record` the feed is recorded to a
//! new run in the data dir, and closed segments are rotated into Parquet in the background, so
//! a replay reaches the rotated hours through `DuckDB` and the open hour from memory. Ctrl-C or
//! `--seconds` ends it.
//!
//! `cargo run -p engine --bin serve -- [--port N] [--record] [--data-dir DIR]
//!   [--allow-origin URL]... [--script FILE [--speed X]] [--seconds N]`

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, Write};
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use engine::api::hub::{Hub, HubConfig, LiveTap};
use engine::api::{ApiConfig, DEFAULT_PORT, serve};
use engine::market::feed::Feed;
use engine::market::live::{Log, Observer, handle, utc_clock};
use engine::market::pipeline::Pipeline;
use engine::market::recording::{Record, read_records};
use engine::store::frame::Meta;
use engine::store::rotate::rotate;
use engine::store::run::{self, BookSample, MarketParams};
use engine::store::segment::{self, LOG_DIR};
use engine::store::writer::{LogConfig, LogWriter};
use engine::venue::Venue;
use engine::venue::binance::live::{LiveConfig, Recording, run_live};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::time::{Instant, sleep, sleep_until};

const USAGE: &str = "usage: serve [--port N] [--record] [--data-dir DIR] [--allow-origin URL]... \
                     [--script FILE [--speed X]] [--seconds N]";

/// How often a recording run looks for closed segments to rotate.
const ROTATE_EVERY: Duration = Duration::from_secs(10);

#[derive(Debug, PartialEq)]
struct Args {
    port: u16,
    record: bool,
    data_dir: Option<PathBuf>,
    origins: Vec<String>,
    script: Option<PathBuf>,
    speed: f64,
    seconds: Option<u64>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut parsed = Args {
        port: DEFAULT_PORT,
        record: false,
        data_dir: None,
        origins: Vec::new(),
        script: None,
        speed: 1.0,
        seconds: None,
    };
    let mut speed_given = false;
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        if flag == "--record" {
            parsed.record = true;
            continue;
        }
        let value = it.next().ok_or_else(|| format!("{flag} wants a value"))?;
        match flag.as_str() {
            "--port" => {
                parsed.port = value
                    .parse()
                    .map_err(|_| format!("--port wants a port number, got {value:?}"))?;
            }
            "--data-dir" => parsed.data_dir = Some(PathBuf::from(value)),
            "--allow-origin" => parsed.origins.push(value.clone()),
            "--script" => parsed.script = Some(PathBuf::from(value)),
            "--speed" => {
                parsed.speed = value
                    .parse::<f64>()
                    .ok()
                    .filter(|speed| speed.is_finite() && *speed >= 0.0)
                    .ok_or_else(|| format!("--speed wants a number, 0 or more, got {value:?}"))?;
                speed_given = true;
            }
            "--seconds" => {
                parsed.seconds = Some(
                    value
                        .parse()
                        .map_err(|_| format!("--seconds wants a whole number, got {value:?}"))?,
                );
            }
            _ => return Err(USAGE.to_owned()),
        }
    }
    if speed_given && parsed.script.is_none() {
        return Err("--speed paces a --script".to_owned());
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
    match run(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("serve failed: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let data_dir = args.data_dir.clone().unwrap_or_else(run::data_dir);
    // A script is read before anything starts: its venue is the one the hub serves.
    let script = match &args.script {
        Some(path) => Some(read_records(BufReader::new(File::open(path)?))?),
        None => None,
    };
    let venue = match script.as_deref() {
        Some([Record::Scales { venue, .. }, ..]) => *venue,
        Some(_) => return Err("a script starts with its scales".into()),
        None => Venue::BinanceUsdm,
    };
    let hub = Hub::new(HubConfig {
        venues: vec![venue],
        ..HubConfig::default()
    });
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, args.port)).await?;
    let port = listener.local_addr()?.port();
    let mut config = ApiConfig::new(data_dir.clone());
    config.origins.extend(args.origins.iter().cloned());
    let (stop, stopped) = watch::channel(false);
    let server = tokio::spawn(serve(listener, Arc::clone(&hub), config, async move {
        let mut stopped = stopped;
        let _ = stopped.wait_for(|stop| *stop).await;
    }));
    println!("serve listening on http://127.0.0.1:{port}");
    std::io::stdout().flush()?;

    let deadline = args
        .seconds
        .map(|seconds| Instant::now() + Duration::from_secs(seconds));
    let tag = if args.script.is_some() {
        "script"
    } else {
        "btceth"
    };
    let recording = if args.record {
        let run_dir = run::market_run_dir(&data_dir, tag)?;
        println!("{} recording to {}", utc_clock(), run_dir.display());
        Some(run_dir)
    } else {
        None
    };
    let (stop_rotating, rotating_stopped) = watch::channel(false);
    let rotating = recording
        .clone()
        .map(|run_dir| tokio::spawn(rotate_closed_segments(run_dir, rotating_stopped)));

    let fed = if let Some(script) = script {
        play_script(script, args.speed, recording.as_deref(), &hub).await
    } else {
        let config = LiveConfig {
            duration: args.seconds.map(Duration::from_secs),
            record: recording.clone().map(|run_dir| Recording {
                run_dir,
                log: LogConfig::default(),
            }),
            ..LiveConfig::default()
        };
        run_live(config, LiveTap::new(Arc::clone(&hub)))
            .await
            .map_err(Into::into)
    };
    hub.end();
    // Let a rotate already running finish, so the last one below is not refused its lock.
    let _ = stop_rotating.send(true);
    if let Some(rotating) = rotating {
        let _ = rotating.await;
    }
    if let Some(run_dir) = recording {
        // The run is over and its last segment closed: rotate everything left.
        match tokio::task::spawn_blocking(move || rotate(&run_dir)).await? {
            Ok(rotation) => println!(
                "{} rotated {} segments into {} files",
                utc_clock(),
                rotation.segments,
                rotation.written.len()
            ),
            Err(error) => eprintln!("{} rotate failed: {error}", utc_clock()),
        }
    }
    if let Err(error) = &fed {
        eprintln!("{} the feed stopped: {error}", utc_clock());
    }
    if args.script.is_some() {
        // A played-back feed is over, but its tables and replays are still served.
        let ctrl_c = tokio::signal::ctrl_c();
        match deadline {
            Some(deadline) => tokio::select! {
                () = sleep_until(deadline) => {}
                _ = ctrl_c => {}
            },
            None => {
                let _ = ctrl_c.await;
            }
        }
    }
    let _ = stop.send(true);
    server.await??;
    fed
}

/// Plays a recorded script through the live path, paced at `speed` times its own clock.
async fn play_script(
    records: Vec<Record>,
    speed: f64,
    recording: Option<&Path>,
    hub: &Arc<Hub>,
) -> Result<(), Box<dyn Error>> {
    let mut records = records.into_iter();
    let Some(Record::Scales { venue, scales }) = records.next() else {
        return Err("a script starts with its scales".into());
    };
    let mut log = match recording {
        Some(run_dir) => {
            let params = MarketParams {
                venue,
                symbols: scales.keys().cloned().collect(),
                scales: scales.clone(),
                streams: venue.streams().iter().map(|&s| s.to_owned()).collect(),
                book_sample: BookSample::default(),
            };
            let manifest = run::create(run_dir, params)?;
            let writer = LogWriter::start(run_dir, Meta::for_run(&manifest), LogConfig::default())?;
            Log::recorded(writer)
        }
        None => Log::unrecorded(),
    };
    let mut pipeline = Pipeline::new(Feed::new(venue, scales));
    let mut tap = LiveTap::new(Arc::clone(hub));
    let run_id = recording
        .and_then(Path::file_name)
        .and_then(|name| name.to_str());
    tap.started(pipeline.feed(), run_id);
    let start = Instant::now();
    let mut first_ms = None;
    for record in records {
        if speed > 0.0
            && let Some(t_ms) = record.t_ms()
        {
            let first = *first_ms.get_or_insert(t_ms);
            #[allow(clippy::cast_precision_loss)]
            let after = t_ms.saturating_sub(first) as f64 / 1000.0 / speed;
            sleep_until(start + Duration::from_secs_f64(after)).await;
        }
        let message = record
            .into_message()
            .ok_or("only the first line of a script holds scales")?;
        handle(message, &mut pipeline, &mut log, &mut tap).await?;
    }
    let summary = tokio::task::spawn_blocking(move || log.finish()).await??;
    if let (Some(summary), Some(run_dir)) = (summary, recording) {
        run::complete(run_dir, &summary)?;
    }
    Ok(())
}

/// Rotates each segment of a run being recorded once it is closed, one rotate at a time.
/// Rotate reads the run from seq 0 every time, so its cost grows with the run (#52).
async fn rotate_closed_segments(run_dir: PathBuf, mut stop: watch::Receiver<bool>) {
    loop {
        tokio::select! {
            () = sleep(ROTATE_EVERY) => {}
            _ = stop.wait_for(|stop| *stop) => return,
        }
        if !has_unrotated_segment(&run_dir) {
            continue;
        }
        let dir = run_dir.clone();
        match tokio::task::spawn_blocking(move || rotate(&dir)).await {
            Ok(Ok(rotation)) if !rotation.written.is_empty() => {
                println!("{} rotated {} files", utc_clock(), rotation.written.len());
            }
            Ok(Ok(_)) => {}
            Ok(Err(error)) => eprintln!("{} rotate failed: {error}", utc_clock()),
            Err(error) => eprintln!("{} rotate panicked: {error}", utc_clock()),
        }
    }
}

/// True when a closed segment (any but the newest, which the writer has open) has no `frames`
/// Parquet yet.
fn has_unrotated_segment(run_dir: &Path) -> bool {
    let Ok(segments) = segment::list(&run_dir.join(LOG_DIR)) else {
        return false;
    };
    segments.iter().rev().skip(1).any(|(first_seq, _)| {
        !run_dir
            .join("frames")
            .join(format!("{first_seq:020}.parquet"))
            .exists()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(raw: &[&str]) -> Result<Args, String> {
        parse_args(&raw.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
    }

    #[test]
    fn serves_live_on_the_default_port_unless_told_otherwise() {
        let args = parse(&[]).expect("no flags is valid");
        assert_eq!(args.port, DEFAULT_PORT);
        assert!(!args.record && args.script.is_none() && args.seconds.is_none());
        let args = parse(&[
            "--port",
            "0",
            "--record",
            "--script",
            "s.jsonl",
            "--speed",
            "0",
            "--seconds",
            "5",
            "--allow-origin",
            "http://localhost:3000",
        ])
        .expect("valid flags");
        assert_eq!(args.port, 0);
        assert!(args.record);
        assert_eq!(args.script, Some(PathBuf::from("s.jsonl")));
        assert_eq!(args.speed, 0.0);
        assert_eq!(args.seconds, Some(5));
        assert_eq!(args.origins, vec!["http://localhost:3000".to_owned()]);
    }

    #[test]
    fn refuses_anything_else() {
        for raw in [
            &["--port"][..],
            &["--port", "99999"],
            &["--speed", "2"],
            &["--script", "s", "--speed", "-1"],
            &["--minutes", "1"],
        ] {
            assert!(parse(raw).is_err(), "{raw:?}");
        }
    }

    #[test]
    fn only_closed_segments_wait_for_rotation() -> Result<(), Box<dyn Error>> {
        let dir = std::env::temp_dir().join(format!("tp-serve-rotation-{}", std::process::id()));
        let log = dir.join(LOG_DIR);
        std::fs::create_dir_all(&log)?;
        std::fs::write(log.join(segment::file_name(0)), b"")?;
        assert!(!has_unrotated_segment(&dir), "the newest segment is open");
        std::fs::write(log.join(segment::file_name(9)), b"")?;
        assert!(has_unrotated_segment(&dir));
        std::fs::create_dir_all(dir.join("frames"))?;
        std::fs::write(dir.join("frames").join(format!("{:020}.parquet", 0)), b"")?;
        assert!(!has_unrotated_segment(&dir));
        std::fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
