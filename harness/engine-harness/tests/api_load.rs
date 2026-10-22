//! The API at ten times the live load: 200 depth frames and 2,000 trades a second, with the
//! capture clock running ten times faster than the wall so the book is sampled 100 times a
//! second, through the live path with its log recording. Nine app clients and two Arrow clients
//! read normally; one app client and one Arrow client stop reading at once.
//!
//! The feed must not wait: the generator keeps its rate, the log writer's queue never fills, and
//! the stalled clients change nothing for the others. Each client is cut or coalesced as its kind
//! says: the stalled ones are cut, the normal app clients skip book samples but get every trade,
//! and the normal Arrow clients get every batch. Capture-to-receive latency, delivered against
//! produced, gaps, the writer's queue, the live path's own time per input and peak memory go to
//! `harness/out/api-load-<ms>.json`. Five seconds by default; `TP_LOAD_SECONDS` runs longer.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{Int64Array, RecordBatch};
use arrow::buffer::Buffer;
use arrow::ipc::reader::StreamDecoder;
use engine::api::ApiConfig;
use engine::api::hub::{Hub, HubConfig};
use engine::store::now_ns;
use engine::store::writer::LogConfig;
use engine_harness::api::{
    Frame, LivePath, Outcome, Server, hello, next_frame, next_json, send_json, ws_connect,
};
use engine_harness::load::{Generator, peak_memory_bytes, percentile};
use futures_util::SinkExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpSocket, TcpStream};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{WebSocketStream, client_async};

const DEPTH_PER_S: f64 = 200.0;
const TRADES_PER_S: f64 = 2000.0;
/// The capture clock runs this many times faster than the wall: ten times the book samples.
const CLOCK: u32 = 10;
const APP_CLIENTS: usize = 9;
const ARROW_CLIENTS: usize = 2;
/// How long a client may stall before the engine cuts it, here.
const STALL: Duration = Duration::from_secs(2);
/// Latency the load must stay under (a bound with room for a loaded machine; the target is 50 ms).
const LATENCY_BOUND: Duration = Duration::from_millis(250);

/// What one app client got: book messages and trades with when they arrived, and gaps.
#[derive(Debug, Default)]
struct AppReport {
    /// Symbol, sample time, seq, arrival.
    books: Vec<(String, i64, i64, Instant)>,
    /// Seq, arrival.
    trades: Vec<(u64, Instant)>,
    lost: u64,
}

async fn app_client(
    addr: std::net::SocketAddr,
    subscribed: Arc<AtomicUsize>,
) -> Outcome<AppReport> {
    let mut socket = ws_connect(addr).await?;
    hello(&mut socket).await?;
    send_json(
        &mut socket,
        &json!({"t": "sub", "topics": ["book", "trades", "status"]}),
    )
    .await?;
    let ack = next_json(&mut socket, Duration::from_secs(30)).await?;
    if ack != json!({"t": "source", "mode": "live"}) {
        return Err(format!("wanted the subscription answered, got {ack}").into());
    }
    subscribed.fetch_add(1, Ordering::Relaxed);
    let mut report = AppReport::default();
    loop {
        let Frame::Text(text) = next_frame(&mut socket, Duration::from_secs(30)).await else {
            return Err("an app client lost its socket".into());
        };
        let now = Instant::now();
        let message: Value = serde_json::from_str(&text)?;
        match message["t"].as_str() {
            Some("book") => report.books.push((
                message["symbol"].as_str().unwrap_or_default().to_owned(),
                message["ts_ms"].as_i64().unwrap_or_default(),
                message["seq"].as_i64().unwrap_or_default(),
                now,
            )),
            Some("trades") => {
                for trade in message["trades"].as_array().into_iter().flatten() {
                    report
                        .trades
                        .push((trade["seq"].as_u64().unwrap_or_default(), now));
                }
            }
            Some("gap") => report.lost += message["lost"].as_u64().unwrap_or_default(),
            Some("source") if message["mode"] == "ended" => return Ok(report),
            _ => {}
        }
    }
}

/// What one Arrow client got: each batch's seq and arrival, and whether the stream ended
/// cleanly.
#[derive(Debug, Default)]
struct ArrowReport {
    batches: Vec<(i64, Instant)>,
    rows: usize,
    complete: bool,
}

async fn arrow_client(url: String) -> Outcome<ArrowReport> {
    let mut response = reqwest::get(url).await?;
    let mut decoder = StreamDecoder::new();
    let mut report = ArrowReport::default();
    while let Some(chunk) = response.chunk().await? {
        let now = Instant::now();
        let mut buffer = Buffer::from(chunk.to_vec());
        while !buffer.is_empty() {
            match decoder.decode(&mut buffer)? {
                Some(batch) => {
                    report.rows += batch.num_rows();
                    report.batches.push((first_seq(&batch), now));
                }
                None => break,
            }
        }
    }
    report.complete = decoder.finish().is_ok();
    Ok(report)
}

fn first_seq(batch: &RecordBatch) -> i64 {
    batch
        .column_by_name("seq")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .map_or(-1, |seqs| seqs.value(0))
}

/// A TCP connection that takes almost nothing: a 1 KiB receive buffer, and it is never read.
async fn stalled_socket(addr: std::net::SocketAddr) -> std::io::Result<TcpStream> {
    let socket = TcpSocket::new_v4()?;
    socket.set_recv_buffer_size(1024)?;
    socket.connect(addr).await
}

/// An app client that subscribes to everything and then never reads again.
async fn stalled_app(addr: std::net::SocketAddr) -> Outcome<WebSocketStream<TcpStream>> {
    let stream = stalled_socket(addr).await?;
    let (mut socket, _) = client_async(format!("ws://{addr}/v1/ws"), stream).await?;
    socket
        .send(WsMessage::text(json!({"t": "hello", "v": 1}).to_string()))
        .await?;
    socket
        .send(WsMessage::text(
            json!({"t": "sub", "topics": ["book", "trades", "status"]}).to_string(),
        ))
        .await?;
    Ok(socket)
}

/// An Arrow client that asks for the live stream and never reads the answer.
async fn stalled_arrow(addr: std::net::SocketAddr) -> Outcome<TcpStream> {
    let mut stream = stalled_socket(addr).await?;
    let request = format!("GET /v1/live/book_samples HTTP/1.1\r\nHost: {addr}\r\n\r\n");
    stream.write_all(request.as_bytes()).await?;
    Ok(stream)
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Waits up to a minute for `ready`.
async fn wait_until(what: &str, ready: impl Fn() -> bool) -> Outcome {
    let start = Instant::now();
    while !ready() {
        if start.elapsed() > Duration::from_secs(60) {
            return Err(format!("timed out waiting for {what}").into());
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    Ok(())
}

#[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ten_times_the_live_load_never_waits_on_a_client() -> Outcome {
    let seconds: u64 = std::env::var("TP_LOAD_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let dir =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("api-load-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir)?;
    let mut generator = Generator::default();
    let hub = Hub::new(HubConfig::default());
    let config = ApiConfig {
        ws_stall: STALL,
        arrow_stall: STALL,
        ..ApiConfig::new(dir.clone())
    };
    let mut live = LivePath::new(
        generator.scales(),
        Arc::clone(&hub),
        &dir,
        Some("20261022T000000Z-load"),
        LogConfig::default(),
    )?;
    let server = Server::start(Arc::clone(&hub), config).await?;
    let wall0 = Instant::now();
    let capture0 = now_ns();
    let capture = |at: Instant| {
        capture0 + i64::try_from((at - wall0).as_nanos() * u128::from(CLOCK)).unwrap_or(i64::MAX)
    };
    for message in generator.opening(capture(Instant::now())) {
        live.push(message).await?;
    }

    // The normal clients first, each subscribed (an app client counts once its subscription is
    // answered), then the stalled ones, which the engine cuts a stall timeout after they stop
    // taking data, so the load starts the moment they are in.
    let subscribed = Arc::new(AtomicUsize::new(0));
    let apps: Vec<_> = (0..APP_CLIENTS)
        .map(|_| tokio::spawn(app_client(server.addr, Arc::clone(&subscribed))))
        .collect();
    let arrows: Vec<_> = (0..ARROW_CLIENTS)
        .map(|_| tokio::spawn(arrow_client(server.url("/v1/live/book_samples"))))
        .collect();
    wait_until("the normal clients subscribe", || {
        subscribed.load(Ordering::Relaxed) == APP_CLIENTS
            && hub.arrow_subscribers() == ARROW_CLIENTS
    })
    .await?;
    let apps_now = || hub.counters.ws_clients.load(Ordering::Relaxed);
    let mut stalled_app = stalled_app(server.addr).await?;
    wait_until("the stalled app client to connect", || {
        apps_now() > APP_CLIENTS
    })
    .await
    .map_err(|error| format!("{error}: {} app connections", apps_now()))?;
    let mut stalled_arrow = stalled_arrow(server.addr).await?;
    wait_until("the stalled Arrow client to subscribe", || {
        hub.arrow_subscribers() > ARROW_CLIENTS
    })
    .await
    .map_err(|error| {
        format!(
            "{error}: {} subscribers, {} app connections",
            hub.arrow_subscribers(),
            apps_now()
        )
    })?;
    // Both are in, and stay in until the load stalls them: nothing refused them up front.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let at_start = (apps_now(), hub.arrow_subscribers());
    assert_eq!(
        at_start,
        (APP_CLIENTS + 1, ARROW_CLIENTS + 1),
        "every client is connected when the load starts"
    );

    let mut per_input = Vec::new();
    let mut max_queued = 0;
    let (mut depth, mut trades) = (0_u64, 0_u64);
    let start = Instant::now();
    let length = Duration::from_secs(seconds);
    let mut tick = tokio::time::interval(Duration::from_millis(1));
    while start.elapsed() < length {
        tick.tick().await;
        let due = start.elapsed().as_secs_f64();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (want_depth, want_trades) = ((due * DEPTH_PER_S) as u64, (due * TRADES_PER_S) as u64);
        while depth < want_depth || trades < want_trades {
            let depth_next = depth < want_depth
                && (trades >= want_trades
                    || depth as f64 / DEPTH_PER_S <= trades as f64 / TRADES_PER_S);
            let at = capture(Instant::now());
            let message = if depth_next {
                depth += 1;
                generator.depth(at)
            } else {
                trades += 1;
                generator.trade(at)
            };
            let began = Instant::now();
            live.push(message).await?;
            per_input.push(began.elapsed());
        }
        max_queued = max_queued.max(
            live.log
                .writer()
                .map_or(0, engine::store::writer::LogWriter::queued),
        );
    }
    let elapsed = start.elapsed();
    let apps_at_end = hub.counters.ws_clients.load(Ordering::Relaxed);
    let arrows_at_end = hub.arrow_subscribers();
    hub.end();

    let mut app_reports = Vec::new();
    for app in apps {
        app_reports.push(app.await??);
    }
    let mut arrow_reports = Vec::new();
    for arrow in arrows {
        arrow_reports.push(arrow.await??);
    }
    let produced_samples = hub.counters.samples.load(Ordering::Relaxed);
    let produced_trades = hub.counters.trades.load(Ordering::Relaxed);
    let (summary, watched, pipeline) = live.finish().await?;
    let summary = summary.ok_or("recorded")?;
    let (ring_samples, ring_trades) = hub.ring_len();

    // The stalled clients: cut by the engine during the run, never handed a clean end.
    let stalled_app_seen = tokio::time::timeout(Duration::from_secs(5), async {
        use futures_util::StreamExt;
        loop {
            match stalled_app.next().await {
                Some(Ok(WsMessage::Close(frame))) => return Some(frame.map(|f| u16::from(f.code))),
                Some(Ok(_)) => {}
                Some(Err(_)) | None => return None,
            }
        }
    })
    .await
    .unwrap_or(None);
    let mut leftover = Vec::new();
    let _ = tokio::time::timeout(
        Duration::from_secs(5),
        stalled_arrow.read_to_end(&mut leftover),
    )
    .await;
    let eos = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];

    // Latency: from the moment the live path reached the input that made the message.
    let reached: HashMap<u64, Instant> = watched
        .inputs
        .iter()
        .map(|(seq, _, at)| (*seq, *at))
        .collect();
    let since = |seq: u64, at: Instant| {
        reached
            .get(&seq)
            .map(|reached| at.saturating_duration_since(*reached))
    };
    let mut book_latency: Vec<Duration> = Vec::new();
    let mut trade_latency: Vec<Duration> = Vec::new();
    for report in &app_reports {
        book_latency.extend(
            report
                .books
                .iter()
                .filter_map(|(_, _, seq, at)| since(u64::try_from(*seq).ok()? + 1, *at)),
        );
        trade_latency.extend(
            report
                .trades
                .iter()
                .filter_map(|(seq, at)| since(*seq, *at)),
        );
    }
    let mut arrow_latency: Vec<Duration> = arrow_reports
        .iter()
        .flat_map(|report| report.batches.iter())
        .filter_map(|(seq, at)| since(u64::try_from(*seq).ok()? + 1, *at))
        .collect();
    let book_messages: usize = app_reports.iter().map(|r| r.books.len()).sum();
    let report = json!({
        "seconds": elapsed.as_secs_f64(),
        "target": {"depth_per_s": DEPTH_PER_S, "trades_per_s": TRADES_PER_S, "samples_per_s": 10 * CLOCK},
        "achieved": {
            "depth_per_s": depth as f64 / elapsed.as_secs_f64(),
            "trades_per_s": trades as f64 / elapsed.as_secs_f64(),
        },
        "produced": {"samples": produced_samples, "trades": produced_trades, "log_frames": summary.frames},
        "app_clients": {
            "normal": APP_CLIENTS,
            "book_messages_per_client": book_messages as f64 / APP_CLIENTS as f64,
            "trades_per_client": app_reports.iter().map(|r| r.trades.len()).collect::<Vec<_>>(),
            "lost": app_reports.iter().map(|r| r.lost).collect::<Vec<_>>(),
            "connected_at_end": apps_at_end,
            "stalled_close_code": stalled_app_seen,
        },
        "arrow_clients": {
            "normal": ARROW_CLIENTS,
            "batches": arrow_reports.iter().map(|r| r.batches.len()).collect::<Vec<_>>(),
            "complete": arrow_reports.iter().map(|r| r.complete).collect::<Vec<_>>(),
            "subscribed_at_end": arrows_at_end,
            "stalled_bytes": leftover.len(),
        },
        "latency_ms": {
            "book": {"p50": millis(percentile(&mut book_latency, 50.0)), "p99": millis(percentile(&mut book_latency, 99.0)), "n": book_latency.len()},
            "trades": {"p50": millis(percentile(&mut trade_latency, 50.0)), "p99": millis(percentile(&mut trade_latency, 99.0)), "n": trade_latency.len()},
            "arrow": {"p50": millis(percentile(&mut arrow_latency, 50.0)), "p99": millis(percentile(&mut arrow_latency, 99.0)), "n": arrow_latency.len()},
            "target_p99": 50.0,
        },
        "live_path_ms_per_input": {"p50": millis(percentile(&mut per_input, 50.0)), "p99": millis(percentile(&mut per_input, 99.0)), "max": millis(per_input.iter().copied().max().unwrap_or_default())},
        "writer_queue": {"max": max_queued, "capacity": LogConfig::default().queue},
        "memory": {"peak_bytes": peak_memory_bytes(), "ring_samples": ring_samples, "ring_trades": ring_trades},
        "books_synced": pipeline.feed().symbols().all(|s| pipeline.feed().sync(s).is_some_and(engine::binance::sync::BookSync::is_synced)),
    });
    let out = report_path();
    fs::create_dir_all(out.parent().ok_or("a report dir")?)?;
    fs::write(&out, serde_json::to_string_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);

    // The feed did not wait.
    let (depth_rate, trade_rate) = (
        depth as f64 / elapsed.as_secs_f64(),
        trades as f64 / elapsed.as_secs_f64(),
    );
    assert!(
        depth_rate >= 0.9 * DEPTH_PER_S && trade_rate >= 0.9 * TRADES_PER_S,
        "the feed fell behind its rate: {report}"
    );
    assert!(
        max_queued < LogConfig::default().queue,
        "the log writer's queue filled: {report}"
    );
    assert!(
        report["books_synced"] == true,
        "the generated books lost sync: {report}"
    );
    // The stalled clients were cut during the run.
    assert_eq!(
        apps_at_end, APP_CLIENTS,
        "the stalled app client still connected: {report}"
    );
    assert!(
        matches!(stalled_app_seen, None | Some(None | Some(4002))),
        "the stalled app client was closed for something else than stalling: {report}"
    );
    assert_eq!(
        arrows_at_end, ARROW_CLIENTS,
        "the stalled Arrow client still subscribed: {report}"
    );
    assert!(
        !leftover.ends_with(&eos),
        "the cut Arrow stream ended cleanly"
    );
    // Normal app clients: every trade, no gap; book samples coalesced, in order per symbol.
    for report in &app_reports {
        assert_eq!(report.lost, 0, "an app client lost trades");
        assert_eq!(
            report.trades.len() as u64,
            produced_trades,
            "an app client missed trades"
        );
        let mut last: BTreeMap<&str, i64> = BTreeMap::new();
        for (symbol, ts, _, _) in &report.books {
            let previous = last.insert(symbol, *ts);
            assert!(
                previous.is_none_or(|p| p < *ts),
                "{symbol} went back in time"
            );
        }
        assert!(
            (report.books.len() as u64) < 2 * produced_samples,
            "books are coalesced, never repeated"
        );
    }
    // Normal Arrow clients: every batch, in order, and a clean end.
    for report in &arrow_reports {
        assert!(
            report.complete,
            "an Arrow client's stream did not end cleanly"
        );
        assert_eq!(
            report.batches.len() as u64,
            produced_samples,
            "an Arrow client missed batches"
        );
        assert!(
            report.batches.windows(2).all(|w| w[0].0 <= w[1].0),
            "Arrow batches out of order"
        );
    }
    for (name, values) in [
        ("book", &book_latency),
        ("trades", &trade_latency),
        ("arrow", &arrow_latency),
    ] {
        assert!(!values.is_empty(), "no {name} latency measured");
        let mut values = values.clone();
        let p99 = percentile(&mut values, 99.0);
        assert!(
            p99 < LATENCY_BOUND,
            "{name} p99 {p99:?} over {LATENCY_BOUND:?}: {report}"
        );
    }
    server.stop().await;
    let _ = fs::remove_dir_all(&dir);
    Ok(())
}

fn report_path() -> PathBuf {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../out")
        .join(format!("api-load-{ms}.json"))
}
