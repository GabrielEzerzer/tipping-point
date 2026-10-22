//! The Python round trip: the engine serves in-process on an ephemeral port and the Python client
//! (`tipping_point.engine`, pyarrow only) reads every Arrow stream it offers. First the live book
//! samples, joined halfway through the session with a backfill, until the feed ends; then every
//! table of the market run and of the Ising run, compared with direct `pyarrow.dataset` reads of
//! the same Parquet, the run replayed as if live, and the live rows against the rotated Parquet.
//! Every batch must reach torch through `torch.from_dlpack` without a copy. Python publishes a
//! meter back and the app's WebSocket must receive it. If Python cannot start, this is red.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::api::ApiConfig;
use engine::api::hub::{Hub, HubConfig};
use engine::store::rotate::rotate;
use engine::store::writer::LogConfig;
use engine_harness::api::{
    ISING_RUN, LivePath, Outcome, Server, copy_ising_fixture, fixture, hello, next_json, python,
    script, send_json, ws_connect,
};
use serde_json::json;

const RUN: &str = "20231114T221320Z-session";

fn client() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/api_client.py")
}

/// Starts the client with `args`.
fn spawn(args: &[&str]) -> std::io::Result<std::process::Child> {
    Command::new(python())
        .arg(client())
        .args(args)
        .env("PYTHONUTF8", "1")
        .spawn()
}

/// Waits for a child without blocking the runtime, at most `limit`.
async fn wait(child: &mut std::process::Child, limit: Duration) -> Outcome<ExitStatus> {
    let deadline = Instant::now() + limit;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            return Err(format!("the Python client ran past {limit:?}").into());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_python_client_reads_every_stream_and_publishes_back() -> Outcome {
    let dir =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("api-python-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir)?;
    copy_ising_fixture(&dir)?;
    let (scales, messages) = script(&fixture("synthetic/session-faults.jsonl"))?;
    let hub = Hub::new(HubConfig::default());
    let small = LogConfig {
        max_segment_bytes: 2048,
        ..LogConfig::default()
    };
    let mut live = LivePath::new(scales, Arc::clone(&hub), &dir, Some(RUN), small)?;
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.clone())).await?;
    let url = server.url("");

    // Half the session first, so the client joins a feed already running and backfills it.
    let (first, rest) = messages.split_at(messages.len() / 2);
    for message in first {
        live.push(message.clone()).await?;
    }
    let saved = dir.join("live.arrows");
    let saved_arg = saved.to_string_lossy().to_string();
    let mut reader = spawn(&[
        "live",
        "--url",
        &url,
        "--run",
        RUN,
        "--backfill-s",
        "3600",
        "--out",
        &saved_arg,
    ])?;
    let joined = Instant::now();
    while hub.arrow_subscribers() == 0 {
        if let Some(status) = reader.try_wait()? {
            return Err(format!("the Python client exited before subscribing: {status}").into());
        }
        if joined.elapsed() > Duration::from_secs(120) {
            let _ = reader.kill();
            return Err("the Python client never subscribed".into());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for message in rest {
        live.push(message.clone()).await?;
    }
    hub.end();
    let status = wait(&mut reader, Duration::from_secs(120)).await?;
    assert!(status.success(), "python live: {status}");
    live.finish().await?;
    rotate(&dir.join("market").join(RUN))?;

    // The app listens for what Python publishes.
    let mut app = ws_connect(server.addr).await?;
    hello(&mut app).await?;
    send_json(&mut app, &json!({"t": "sub", "topics": ["meters"]})).await?;
    let _ack = next_json(&mut app, Duration::from_secs(10)).await?;

    let root = dir.to_string_lossy().to_string();
    let mut tables = spawn(&[
        "tables",
        "--url",
        &url,
        "--root",
        &root,
        "--run",
        RUN,
        "--ising-run",
        ISING_RUN,
        "--live",
        &saved_arg,
    ])?;
    let status = wait(&mut tables, Duration::from_secs(180)).await?;
    assert!(status.success(), "python tables: {status}");

    let meter = loop {
        let message = next_json(&mut app, Duration::from_secs(10)).await?;
        if message["t"] == "meters" {
            break message;
        }
    };
    assert_eq!(meter["data"]["from"], "python");
    assert_eq!(meter["data"]["symbol"], "BTCUSDT");
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}
