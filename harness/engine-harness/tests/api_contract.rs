//! The API's contract, pinned by golden files under `engine/tests/fixtures/api/`: every WebSocket
//! message the engine sends or accepts, the lattice frame, every Arrow schema it serves, and a
//! small Arrow stream. A protocol change fails here until the goldens are regenerated in the same
//! commit (`TP_UPDATE_GOLDEN=1 cargo test -p engine-harness`). The app's mock replays the same
//! files, so its fixtures come from the engine.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use engine::api::ApiConfig;
use engine::api::hub::{Hub, HubConfig, Published, Source, VenueSource};
use engine::api::ipc::{Encoder, SCHEMA_KEY};
use engine::api::messages::{
    BookMessage, ClientMessage, LatticeHeader, PROTOCOL, ServerMessage, SourceMessage, Status,
    StreamStatus, Streams, SymbolStatus, Topic, TradeMessage, VenueStatus, Welcome, lattice_frame,
};
use engine::market::Scales;
use engine::market::sync::Level;
use engine::store::rotate::rotate;
use engine::store::samples::{BookRow, Sample};
use engine::store::writer::LogConfig;
use engine::venue::Venue;
use engine_harness::api::{
    Frame, ISING_RUN, LivePath, Outcome, Server, Socket, copy_ising_fixture, decode_stream,
    fixture, get_arrow, golden, hello, merged, next_frame, next_json, one_batch, script, send_json,
    ws_connect,
};
use engine_harness::load::Generator;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

const RUN: &str = "20231114T221320Z-session";
/// The Bybit run recorded beside it.
const BYBIT_RUN: &str = "20231114T221320Z-bybit";
const WAIT: Duration = Duration::from_secs(20);

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch dir");
    dir
}

fn api_fixture(name: &str) -> PathBuf {
    fixture("api").join(name)
}

/// The results the session publishes for the app, one per relayed topic.
fn results() -> [(&'static str, Value); 4] {
    [
        (
            "meters",
            json!({"symbol": "BTCUSDT", "ts_ms": 1_700_000_002_000_i64, "variance": 0.25, "ac1": 0.5, "latent": 1.5}),
        ),
        (
            "jev",
            json!({"symbol": "BTCUSDT", "ts_ms": 1_700_000_002_000_i64, "label": "regime_shift", "p": 0.8}),
        ),
        (
            "pnl",
            json!({"symbol": "BTCUSDT", "ts_ms": 1_700_000_002_000_i64, "position": 1, "pnl": -2.5}),
        ),
        (
            "ising",
            json!({"run": ISING_RUN, "step": 3, "m": 0.25, "E": -1.5, "T": 2.3}),
        ),
    ]
}

/// The committed sessions, Binance's and Bybit's at once, each through its own live path into one
/// hub, with the API attached and a lossless client and an Arrow subscriber connected from the
/// start; then results published for the app. The WebSocket frames, as text, in the order they
/// arrived, the live Arrow stream's bytes, and the Binance run's folder.
async fn record_session(dir: &Path) -> Outcome<(Vec<String>, Vec<u8>, PathBuf)> {
    let binance = script(&fixture("synthetic/session-faults.jsonl"))?;
    let bybit = script(&fixture("synthetic/bybit-session.jsonl"))?;
    let hub = Hub::new(HubConfig {
        lossless: true,
        venues: vec![binance.venue, bybit.venue],
        ..HubConfig::default()
    });
    let config = LogConfig {
        max_segment_bytes: 2048,
        ..LogConfig::default()
    };
    let mut loops = [
        LivePath::new(
            binance.venue,
            binance.scales,
            Arc::clone(&hub),
            dir,
            Some(RUN),
            config.clone(),
        )?,
        LivePath::new(
            bybit.venue,
            bybit.scales,
            Arc::clone(&hub),
            dir,
            Some(BYBIT_RUN),
            config,
        )?,
    ];
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.to_owned())).await?;

    let mut ws = ws_connect(server.addr).await?;
    send_json(&mut ws, &json!({"t": "hello", "v": PROTOCOL})).await?;
    let mut frames = Vec::new();
    let Frame::Text(welcome) = next_frame(&mut ws, WAIT).await else {
        return Err("no welcome".into());
    };
    frames.push(welcome);
    send_json(&mut ws, &json!({"t": "sub", "topics": Topic::ALL})).await?;
    let Frame::Text(ack) = next_frame(&mut ws, WAIT).await else {
        return Err("no answer to sub".into());
    };
    frames.push(ack);

    let url = server.url("/v1/live/book_samples");
    let arrow = tokio::spawn(async move { reqwest::get(url).await?.bytes().await });
    while hub.arrow_subscribers() == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for (which, message) in merged(binance.messages, bybit.messages) {
        loops[which].push(message).await?;
    }
    let client = reqwest::Client::new();
    for (topic, body) in results() {
        let response = client
            .post(server.url(&format!("/v1/publish/{topic}")))
            .body(body.to_string())
            .send()
            .await?;
        assert_eq!(response.status(), 204, "{topic}");
    }
    hub.end();
    loop {
        match next_frame(&mut ws, WAIT).await {
            Frame::Text(text) => {
                let ended = serde_json::from_str::<Value>(&text)?["mode"] == "ended";
                frames.push(text);
                if ended {
                    break;
                }
            }
            other => return Err(format!("the session stopped early: {other:?}").into()),
        }
    }
    let arrow = arrow.await??.to_vec();
    for live in loops {
        live.finish().await?;
    }
    server.stop().await;
    Ok((frames, arrow, dir.join("market").join(RUN)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_session_is_sent_as_its_golden_files_say() -> Outcome {
    let dir = scratch("api-golden");
    let (frames, arrow, _) = record_session(&dir).await?;

    // Every kind of input the session holds shows up, and the order is the publish order.
    let kinds: Vec<String> = frames
        .iter()
        .map(|text| {
            serde_json::from_str::<Value>(text)
                .map(|v| v["t"].as_str().unwrap_or_default().to_owned())
        })
        .collect::<Result<_, _>>()?;
    for kind in [
        "welcome", "source", "book", "trades", "status", "meters", "jev", "pnl", "ising",
    ] {
        assert!(
            kinds.iter().any(|k| k == kind),
            "the session sent no {kind}"
        );
    }
    assert_eq!(kinds[0], "welcome");
    let statuses: Vec<Value> = frames
        .iter()
        .filter_map(|text| serde_json::from_str::<Value>(text).ok())
        .filter(|v| v["t"] == "status")
        .collect();
    let status = statuses.last().ok_or("no status")?;
    assert_eq!(status["venues"]["bybit-linear"]["resyncs"], 5, "{status}");
    let venues: std::collections::BTreeSet<String> = frames
        .iter()
        .filter_map(|text| serde_json::from_str::<Value>(text).ok())
        .filter(|v| v["t"] == "book")
        .filter_map(|v| v["venue"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(venues.len(), 2, "both venues' books were sent: {venues:?}");
    let last = &status["venues"]["binance-usdm"];
    assert!(
        last["streams"]["depth"]["reconnects"].as_u64() >= Some(1),
        "{status}"
    );
    assert!(
        last["streams"]["depth"]["frames_dropped"].as_u64() >= Some(1),
        "{status}"
    );
    assert!(last["resyncs"].as_u64() >= Some(1), "{status}");

    let mut text = frames.join("\n");
    text.push('\n');
    golden(&api_fixture("ws-session.jsonl"), text.as_bytes())?;

    // The live Arrow stream: compared as decoded batches, so an encoder detail cannot fail it.
    let (schema, batches) = decode_stream(&arrow)?;
    let path = api_fixture("book_samples.arrows");
    if std::env::var_os("TP_UPDATE_GOLDEN").is_some() {
        golden(&path, &arrow)?;
    }
    let (golden_schema, golden_batches) = decode_stream(&fs::read(&path)?)?;
    assert_eq!(schema, golden_schema, "{}", path.display());
    assert_eq!(
        one_batch(&schema, &batches)?,
        one_batch(&golden_schema, &golden_batches)?,
        "{} holds other rows than the live stream sends now",
        path.display()
    );
    let rows: usize = batches
        .iter()
        .map(arrow::array::RecordBatch::num_rows)
        .sum();
    assert!(rows >= 30, "{rows} rows");
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// The book of the catalog: one synced symbol with levels, one unsynced.
fn catalog_sample() -> Sample {
    let level = |price, qty| Level { price, qty };
    Sample {
        sample_ts_ns: 1_700_000_000_300_000_000,
        seq: 12,
        venue: Venue::BinanceUsdm,
        rows: vec![
            BookRow {
                symbol: "BTCUSDT".to_owned(),
                synced: true,
                last_update_id: Some(106),
                bids: vec![level(499_990, 500), level(499_980, 3000)],
                asks: vec![level(500_010, 1500)],
            },
            BookRow {
                symbol: "ETHUSDT".to_owned(),
                synced: false,
                last_update_id: None,
                bids: Vec::new(),
                asks: Vec::new(),
            },
        ],
    }
}

/// The status of the catalog: every counter set.
fn catalog_status() -> Status {
    let stream = |connected, reconnects, frames_dropped| StreamStatus {
        connected,
        reconnects,
        frames_dropped,
    };
    let binance = VenueStatus {
        ts_ms: 1_700_000_000_300,
        streams: Streams {
            depth: stream(true, 1, 2),
            trades: stream(false, 0, 3),
        },
        symbols: BTreeMap::from([
            ("BTCUSDT".to_owned(), SymbolStatus { synced: true }),
            ("ETHUSDT".to_owned(), SymbolStatus { synced: false }),
        ]),
        resyncs: 4,
        sequence_gaps: 1,
        snapshot_failures: 1,
        parse_errors: 2,
    };
    let bybit = VenueStatus {
        ts_ms: 1_700_000_000_250,
        streams: Streams {
            depth: stream(true, 0, 0),
            trades: stream(false, 0, 0),
        },
        symbols: BTreeMap::from([("BTCUSDT".to_owned(), SymbolStatus { synced: true })]),
        resyncs: 1,
        sequence_gaps: 1,
        snapshot_failures: 0,
        parse_errors: 0,
    };
    Status {
        ts_ms: 1_700_000_000_300,
        venues: BTreeMap::from([(Venue::BinanceUsdm, binance), (Venue::BybitLinear, bybit)]),
    }
}

/// One of every message the engine sends, made by the code that sends them. The app's types and
/// mock are written against these files.
#[test]
fn every_server_message_matches_the_catalog() -> Outcome {
    let scale = |price_decimals, qty_decimals| Scales {
        price_decimals,
        qty_decimals,
    };
    let scales = BTreeMap::from([
        ("BTCUSDT".to_owned(), scale(1, 3)),
        ("ETHUSDT".to_owned(), scale(2, 3)),
    ]);
    let source = Source {
        venues: BTreeMap::from([
            (
                Venue::BinanceUsdm,
                VenueSource {
                    scales,
                    run: Some(RUN.to_owned()),
                },
            ),
            (
                Venue::BybitLinear,
                VenueSource {
                    scales: BTreeMap::from([("BTCUSDT".to_owned(), scale(1, 3))]),
                    run: Some(BYBIT_RUN.to_owned()),
                },
            ),
        ]),
    };
    let welcome = Welcome::new(&source, 20, 100_000_000);
    let sample = catalog_sample();
    let books: Vec<BookMessage<'_>> = sample
        .rows
        .iter()
        .map(|row| BookMessage::new(&sample, row, 20))
        .collect();
    let trades = [TradeMessage {
        venue: Venue::BinanceUsdm,
        symbol: "BTCUSDT".to_owned(),
        seq: 5,
        ts_ms: 1_700_000_000_149,
        id: "1".to_owned(),
        px: 500_005,
        qty: 10,
        buyer_is_maker: false,
    }];
    let status = catalog_status();
    let source = |mode, run: Option<&str>, from_ms, speed| SourceMessage {
        mode,
        run: run.map(str::to_owned),
        from_ms,
        speed,
        error: None,
    };
    let sources = [
        source("live", None, None, None),
        source("replay", Some(RUN), Some(1_700_000_000_500), Some(10.0)),
        source("paused", Some(RUN), Some(1_700_000_000_500), Some(10.0)),
        source("ended", Some(RUN), None, None),
        SourceMessage {
            error: Some("parquet: no magic bytes at the end of the file".to_owned()),
            ..source("failed", Some(RUN), None, None)
        },
    ];
    let results = results();
    let mut lines = vec![ServerMessage::Welcome(&welcome).to_text()];
    lines.extend(books.iter().map(|book| ServerMessage::Book(book).to_text()));
    lines.push(ServerMessage::Trades { trades: &trades }.to_text());
    lines.push(ServerMessage::Status(&status).to_text());
    lines.push(
        ServerMessage::Gap {
            topic: Topic::Trades,
            lost: 12,
        }
        .to_text(),
    );
    lines.push(
        ServerMessage::Gap {
            topic: Topic::Ising,
            lost: 3,
        }
        .to_text(),
    );
    lines.push(
        ServerMessage::Gap {
            topic: Topic::Book,
            lost: 4,
        }
        .to_text(),
    );
    lines.extend(sources.iter().map(|s| ServerMessage::Source(s).to_text()));
    for (topic, data) in &results {
        let topic = Topic::relayed(topic).ok_or("a relayed topic")?;
        lines.push(ServerMessage::relay(topic, data).to_text());
    }
    golden(
        &api_fixture("ws-server.jsonl"),
        (lines.join("\n") + "\n").as_bytes(),
    )
}

/// One of every message the engine accepts.
#[test]
fn every_client_message_matches_the_catalog() -> Outcome {
    let topics = Topic::ALL.to_vec();
    let client = [
        ClientMessage::Hello { v: PROTOCOL },
        ClientMessage::Sub {
            topics: topics.clone(),
            venues: None,
            symbols: None,
            levels: None,
            backfill_s: None,
        },
        ClientMessage::Sub {
            topics: vec![Topic::Book, Topic::Status],
            venues: Some(vec![Venue::BinanceUsdm]),
            symbols: Some(vec!["BTCUSDT".to_owned()]),
            levels: Some(5),
            backfill_s: Some(30.0),
        },
        ClientMessage::Replay {
            run: RUN.to_owned(),
            from_ms: Some(1_700_000_000_500),
            speed: 10.0,
        },
        ClientMessage::Seek {
            to_ms: 1_700_000_001_000,
        },
        ClientMessage::Pause,
        ClientMessage::Play,
        ClientMessage::Live,
    ];
    let mut text = String::new();
    for message in &client {
        let line = serde_json::to_string(message)?;
        assert_eq!(
            &serde_json::from_str::<ClientMessage>(&line)?,
            message,
            "{line}"
        );
        text.push_str(&line);
        text.push('\n');
    }
    golden(&api_fixture("ws-client.jsonl"), text.as_bytes())
}

/// The one binary frame: a lattice with its header, the spins as `configs` stores them.
#[test]
fn the_lattice_frame_matches_its_golden_file() -> Outcome {
    let header = LatticeHeader {
        t: "lattice".to_owned(),
        run: ISING_RUN.to_owned(),
        replica: 1,
        step: 3,
        l: 4,
        temperature: 2.3,
    };
    golden(
        &api_fixture("ws-lattice.bin"),
        &lattice_frame(&header, &[0b1000_0001, 0b0110_0000]),
    )
}

/// The schema of every table served, market and Ising, and of the live stream: names, types,
/// nullability, and the engine's metadata.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_arrow_schema_matches_its_golden_file() -> Outcome {
    let dir = scratch("api-schemas");
    let (_, arrow, run_dir) = record_session(&dir).await?;
    rotate(&run_dir)?;
    copy_ising_fixture(&dir)?;
    let hub = Hub::new(HubConfig::default());
    let server = Server::start(hub, ApiConfig::new(dir.clone())).await?;

    let describe = |schema: &arrow::datatypes::Schema| {
        let fields: Vec<Value> = schema
            .fields()
            .iter()
            .map(|f| json!({"name": f.name(), "type": f.data_type().to_string(), "nullable": f.is_nullable()}))
            .collect();
        let metadata: BTreeMap<&String, &String> = schema.metadata().iter().collect();
        json!({"metadata": metadata, "fields": fields})
    };
    let mut schemas = serde_json::Map::new();
    let (live_schema, _) = decode_stream(&arrow)?;
    schemas.insert("live/book_samples".to_owned(), describe(&live_schema));
    let tables = [
        ("market", RUN, "frames"),
        ("market", RUN, "depth_updates"),
        ("market", RUN, "trades"),
        ("market", RUN, "snapshots"),
        ("market", RUN, "book_samples"),
        ("ising", ISING_RUN, "stats"),
        ("ising", ISING_RUN, "configs"),
    ];
    for (domain, run, table) in tables {
        let (schema, batches) =
            get_arrow(&server.url(&format!("/v1/table/{domain}/{run}/{table}"))).await?;
        assert!(!batches.is_empty(), "{domain}/{table} has rows");
        // The market tables keyed by venue are at 2; frames and the Ising tables at 1.
        let major = if domain == "market" && table != "frames" {
            2
        } else {
            1
        };
        let id = format!("{table}/{major}");
        assert_eq!(schema.metadata()[SCHEMA_KEY], id);
        schemas.insert(format!("{domain}/{table}"), describe(&schema));
    }
    let text = serde_json::to_string_pretty(&Value::Object(schemas))? + "\n";
    golden(&api_fixture("arrow-schemas.json"), text.as_bytes())?;

    // The runs the data dir holds, by domain, each saying whether it ended cleanly.
    let runs: Value =
        serde_json::from_str(&reqwest::get(server.url("/v1/runs")).await?.text().await?)?;
    let find = |domain: &str, run: &str| {
        runs[domain]
            .as_array()
            .and_then(|runs| runs.iter().find(|r| r["run"] == run).cloned())
    };
    let market = find("market", RUN).ok_or("the market run is listed")?;
    assert_eq!(market["complete"], true);
    assert_eq!(market["key"], "seq");
    assert_eq!(market["venue"], "binance-usdm");
    let ising = find("ising", ISING_RUN).ok_or("the Ising run is listed")?;
    assert_eq!(ising["key"], "step");
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A steady feed leaves the status unchanged for as long as it stays steady, yet the status's
/// `ts_ms` keeps up with it: unchanged, it goes out again once a second of the feed has passed,
/// so a client comparing `ts_ms` with its own clock never takes a healthy feed for a stale one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_status_time_keeps_up_with_a_steady_feed() -> Outcome {
    const T0: i64 = 1_700_000_000_000_000_000;
    let dir = scratch("api-status-time");
    let mut generator = Generator::default();
    let hub = Hub::new(HubConfig::default());
    let mut live = LivePath::new(
        Venue::BinanceUsdm,
        generator.scales(),
        Arc::clone(&hub),
        &dir,
        None,
        LogConfig::default(),
    )?;
    for message in generator.opening(T0) {
        live.push(message).await?;
    }
    let status = hub.watch_status();
    // Ten seconds of depth updates 50 ms apart: each book syncs on its first and stays synced.
    for i in 1..=200 {
        let captured_ms = (T0 + i * 50_000_000) / 1_000_000;
        live.push(generator.depth(captured_ms * 1_000_000)).await?;
        let times = status.borrow().as_ref().and_then(|status| {
            let venue = status.venues.get(&Venue::BinanceUsdm)?;
            Some((status.ts_ms, venue.ts_ms))
        });
        let (ts_ms, venue_ts_ms) = times.ok_or("a status")?;
        for behind in [captured_ms - ts_ms, captured_ms - venue_ts_ms] {
            assert!(behind < 1000, "input {i}: the status is {behind} ms behind");
        }
    }
    live.finish().await?;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Wrong version, a first message that is not hello, and an unreadable message each close the
/// connection with their code.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn protocol_errors_close_with_their_codes() -> Outcome {
    let dir = scratch("api-close");
    let hub = Hub::new(HubConfig::default());
    let feed = engine::market::feed::Feed::new(Venue::BinanceUsdm, BTreeMap::new());
    let mut tap = engine::api::hub::LiveTap::new(Arc::clone(&hub));
    engine::market::live::Observer::started(&mut tap, &feed, None);
    let server = Server::start(hub, ApiConfig::new(dir.clone())).await?;

    // A client of the protocol before venues.
    let mut ws = ws_connect(server.addr).await?;
    send_json(&mut ws, &json!({"t": "hello", "v": 1})).await?;
    assert_eq!(next_frame(&mut ws, WAIT).await, Frame::Closed(Some(4001)));

    let mut ws = ws_connect(server.addr).await?;
    send_json(&mut ws, &json!({"t": "sub", "topics": ["book"]})).await?;
    assert_eq!(next_frame(&mut ws, WAIT).await, Frame::Closed(Some(4000)));

    let mut ws = ws_connect(server.addr).await?;
    send_json(&mut ws, &json!({"t": "hello", "v": PROTOCOL})).await?;
    assert_eq!(next_json(&mut ws, WAIT).await?["t"], "welcome");
    send_json(&mut ws, &json!({"t": "subscribe"})).await?;
    assert_eq!(next_frame(&mut ws, WAIT).await, Frame::Closed(Some(4000)));

    let mut ws = ws_connect(server.addr).await?;
    send_json(&mut ws, &json!({"t": "hello", "v": PROTOCOL})).await?;
    assert_eq!(next_json(&mut ws, WAIT).await?["t"], "welcome");
    send_json(&mut ws, &json!({"t": "seek", "to_ms": 1})).await?;
    assert_eq!(
        next_frame(&mut ws, WAIT).await,
        Frame::Closed(Some(4000)),
        "seek needs a replay"
    );
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A client may send its subscription right behind its hello, before the welcome, even while the
/// live feed has not started: it is answered after the welcome, never refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_subscription_sent_before_the_welcome_is_answered_after_it() -> Outcome {
    let dir = scratch("api-early");
    let hub = Hub::new(HubConfig::default());
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.clone())).await?;
    let sub = json!({"t": "sub", "topics": ["status"]});

    // The feed not started yet: the welcome waits for it.
    let mut waiting = ws_connect(server.addr).await?;
    send_json(&mut waiting, &json!({"t": "hello", "v": PROTOCOL})).await?;
    send_json(&mut waiting, &sub).await?;
    assert_eq!(
        next_frame(&mut waiting, Duration::from_millis(300)).await,
        Frame::Silent
    );
    let feed = engine::market::feed::Feed::new(Venue::BinanceUsdm, BTreeMap::new());
    let mut tap = engine::api::hub::LiveTap::new(Arc::clone(&hub));
    engine::market::live::Observer::started(&mut tap, &feed, None);
    assert_eq!(next_json(&mut waiting, WAIT).await?["t"], "welcome");
    assert_eq!(
        next_json(&mut waiting, WAIT).await?,
        json!({"t": "source", "mode": "live"})
    );

    // The feed running: the same, many times over, never a refusal.
    for _ in 0..20 {
        let mut eager = ws_connect(server.addr).await?;
        send_json(&mut eager, &json!({"t": "hello", "v": PROTOCOL})).await?;
        send_json(&mut eager, &sub).await?;
        assert_eq!(next_json(&mut eager, WAIT).await?["t"], "welcome");
        assert_eq!(
            next_json(&mut eager, WAIT).await?,
            json!({"t": "source", "mode": "live"})
        );
    }
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A client that connects after the live feed ended is told so, and its subscription is answered
/// the same, where it was answered `live` though nothing was (#102): the connection stays ended
/// until `replay` or `live`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_subscription_after_the_feed_ended_is_answered_ended() -> Outcome {
    let dir = scratch("api-after-end");
    let server = started_server(&dir, HubConfig::default()).await?;
    server
        .hub
        .publish_status(Venue::BinanceUsdm, VenueStatus::default());
    server.hub.end();
    let ended = json!({"t": "source", "mode": "ended"});
    let mut late = ws_connect(server.addr).await?;
    hello(&mut late).await?;
    assert_eq!(next_json(&mut late, WAIT).await?, ended, "on connecting");
    let sub = json!({"t": "sub", "topics": ["book", "trades", "status"]});
    send_json(&mut late, &sub).await?;
    let answer = next_json(&mut late, WAIT).await?;
    assert_eq!(answer, ended, "the subscription's answer");
    assert_eq!(next_json(&mut late, WAIT).await?["t"], "status");
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A subscription sent in the same breath as the hello, to a feed that already ended, hears
/// `ended` first: it was answered `live` for one send before `ended` came (#102).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_subscription_sent_with_the_hello_to_an_ended_feed_hears_ended_first() -> Outcome {
    let dir = scratch("api-ended-at-once");
    let server = started_server(&dir, HubConfig::default()).await?;
    server.hub.end();
    let ended = json!({"t": "source", "mode": "ended"});
    let mut eager = ws_connect(server.addr).await?;
    send_json(&mut eager, &json!({"t": "hello", "v": PROTOCOL})).await?;
    send_json(
        &mut eager,
        &json!({"t": "sub", "topics": ["book", "trades"]}),
    )
    .await?;
    assert_eq!(next_json(&mut eager, WAIT).await?["t"], "welcome");
    assert_eq!(next_json(&mut eager, WAIT).await?, ended, "first");
    let answer = next_json(&mut eager, WAIT).await?;
    assert_eq!(answer, ended, "the subscription's answer");
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A lossless client that connects after the feed ended is told so, and so are its subscription
/// and a pause. Its feed's `Ordered::End` went out before it subscribed, so it heard nothing of
/// the end: its `sub` was answered `live` and a pause closed it with 4000 (#102).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lossless_client_after_the_feed_ended_is_told_it_ended() -> Outcome {
    let dir = scratch("api-lossless-after-end");
    let hub = Hub::new(HubConfig {
        lossless: true,
        ..HubConfig::default()
    });
    let feed = VenueSource {
        scales: BTreeMap::new(),
        run: None,
    };
    assert!(hub.start(Venue::BinanceUsdm, feed));
    hub.end();
    let server = Server::start(hub, ApiConfig::new(dir.clone())).await?;
    let ended = json!({"t": "source", "mode": "ended"});
    let mut late = ws_connect(server.addr).await?;
    hello(&mut late).await?;
    send_json(
        &mut late,
        &json!({"t": "sub", "topics": ["book", "status"]}),
    )
    .await?;
    assert_eq!(next_json(&mut late, WAIT).await?, ended, "on connecting");
    let answer = next_json(&mut late, WAIT).await?;
    assert_eq!(answer, ended, "the subscription's answer");
    send_json(&mut late, &json!({"t": "pause"})).await?;
    let answer = next_json(&mut late, WAIT).await?;
    assert_eq!(answer, ended, "a pause, not a 4000 close");
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A foreign `Origin` is refused, the WebSocket upgrade included, and so is a `Host` that names
/// another server (DNS rebinding) and a page's plain GET from another site, which carries no
/// `Origin`; the allowed origins and a client without `Origin` get in.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreign_origins_and_hosts_are_refused() -> Outcome {
    let dir = scratch("api-origin");
    let server = Server::start(Hub::new(HubConfig::default()), ApiConfig::new(dir.clone())).await?;

    let mut request = format!("ws://{}/v1/ws", server.addr).into_client_request()?;
    request
        .headers_mut()
        .insert("Origin", "https://evil.example".parse()?);
    let refused = tokio_tungstenite::connect_async(request).await;
    assert!(
        matches!(&refused, Err(tokio_tungstenite::tungstenite::Error::Http(response)) if response.status() == 403),
        "{refused:?}"
    );
    let mut request = format!("ws://{}/v1/ws", server.addr).into_client_request()?;
    request
        .headers_mut()
        .insert("Origin", "http://tauri.localhost".parse()?);
    assert!(tokio_tungstenite::connect_async(request).await.is_ok());

    let client = reqwest::Client::new();
    let health = client.get(server.url("/v1/health")).send().await?;
    assert_eq!(health.status(), 200);
    let foreign = client
        .get(server.url("/v1/health"))
        .header("Origin", "https://evil.example")
        .send()
        .await?;
    assert_eq!(foreign.status(), 403);
    // What a browser sends for `fetch(url, {mode: 'no-cors'})` or an image on another site's page.
    let no_cors = client
        .get(server.url("/v1/health"))
        .header("Sec-Fetch-Site", "cross-site")
        .header("Sec-Fetch-Mode", "no-cors")
        .send()
        .await?;
    assert_eq!(no_cors.status(), 403, "a page's no-cors GET");
    let typed = client
        .get(server.url("/v1/health"))
        .header("Sec-Fetch-Site", "none")
        .send()
        .await?;
    assert_eq!(typed.status(), 200, "a URL typed into the address bar");
    let publish = client
        .post(server.url("/v1/publish/meters"))
        .header("Origin", "https://evil.example")
        .body(json!({"v": 1}).to_string())
        .send()
        .await?;
    assert_eq!(publish.status(), 403, "a page cannot post results either");

    // A rebound name reaches 127.0.0.1 with its own Host.
    let mut socket = tokio::net::TcpStream::connect(server.addr).await?;
    socket
        .write_all(b"GET /v1/health HTTP/1.1\r\nHost: evil.example:80\r\nConnection: close\r\n\r\n")
        .await?;
    let mut answer = String::new();
    socket.read_to_string(&mut answer).await?;
    assert!(answer.starts_with("HTTP/1.1 403"), "{answer}");
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A replay opening its run on a slow disk holds up no other connection, even with the whole
/// server on one thread: the opening runs on a blocking thread, so another client asking meanwhile
/// is answered at once. Then the replay of a run that is not there is refused with 4000, as ever.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replay_opening_on_a_slow_disk_holds_up_no_other_connection() -> Outcome {
    const SLOW: Duration = Duration::from_secs(3);
    let dir = scratch("api-slow-open");
    let hub = Hub::new(HubConfig::default());
    let feed = engine::market::feed::Feed::new(Venue::BinanceUsdm, BTreeMap::new());
    let mut tap = engine::api::hub::LiveTap::new(Arc::clone(&hub));
    engine::market::live::Observer::started(&mut tap, &feed, None);
    let config = ApiConfig {
        open_delay: Some(SLOW),
        ..ApiConfig::new(dir.clone())
    };
    let (addr_tx, addr_rx) = tokio::sync::oneshot::channel();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let server = std::thread::spawn(move || -> std::io::Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async move {
            let server = Server::start(hub, config).await?;
            let _ = addr_tx.send(server.addr);
            let _ = stop_rx.await;
            server.stop().await;
            Ok(())
        })
    });
    let addr = addr_rx.await?;
    let mut clients = Vec::new();
    for _ in 0..2 {
        let mut ws = ws_connect(addr).await?;
        send_json(&mut ws, &json!({"t": "hello", "v": PROTOCOL})).await?;
        assert_eq!(next_json(&mut ws, WAIT).await?["t"], "welcome");
        clients.push(ws);
    }
    let missing = json!({"t": "replay", "run": "20990101T000000Z-none", "speed": 0});
    send_json(&mut clients[0], &missing).await?;
    // The server has the replay well before the other client asks.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let asked = std::time::Instant::now();
    send_json(&mut clients[1], &json!({"t": "sub", "topics": ["status"]})).await?;
    let answer = next_json(&mut clients[1], WAIT).await?;
    let waited = asked.elapsed();
    assert_eq!(answer, json!({"t": "source", "mode": "live"}));
    assert!(
        waited < SLOW / 3,
        "answered after {waited:?}, behind the slow disk"
    );
    let refused = next_frame(&mut clients[0], WAIT).await;
    assert_eq!(
        refused,
        Frame::Closed(Some(4000)),
        "a run that is not there"
    );
    let _ = stop_tx.send(());
    server.join().map_err(|_| "the server thread panicked")??;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// The messages a client reads until nothing more comes for `quiet`.
async fn until_quiet(ws: &mut Socket, quiet: Duration) -> Outcome<Vec<Value>> {
    let mut got = Vec::new();
    loop {
        match next_frame(ws, quiet).await {
            Frame::Text(text) => got.push(serde_json::from_str(&text)?),
            Frame::Silent => return Ok(got),
            other => return Err(format!("the socket stopped: {other:?}").into()),
        }
    }
}

/// The relayed messages a client reads until nothing more comes for `quiet`: each as its topic and
/// data.
async fn relayed_until_quiet(ws: &mut Socket, quiet: Duration) -> Outcome<Vec<(String, Value)>> {
    let got = until_quiet(ws, quiet).await?;
    Ok(got
        .into_iter()
        .map(|message| {
            let topic = message["t"].as_str().unwrap_or_default().to_owned();
            (topic, message["data"].clone())
        })
        .collect())
}

/// Publishing takes JSON objects on the relayed topics only, and an app client of a production
/// (coalescing) hub gets each result on its own topic, once (#72): the harness's lossless path
/// is the only other way these reach a client, and the golden session takes that one. A second
/// P&L for a symbol replaces the first: it goes out, and a client subscribing later gets only it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn only_objects_on_relayed_topics_are_published() -> Outcome {
    let dir = scratch("api-publish");
    let hub = Hub::new(HubConfig::default());
    let feed = VenueSource {
        scales: BTreeMap::new(),
        run: None,
    };
    assert!(hub.start(Venue::BinanceUsdm, feed));
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.clone())).await?;
    let client = reqwest::Client::new();
    let post = |topic: &str, body: &str| {
        client
            .post(server.url(&format!("/v1/publish/{topic}")))
            .body(body.to_owned())
            .send()
    };
    let relayed = json!({"t": "sub", "topics": ["meters", "jev", "pnl", "ising"]});
    let mut app = ws_connect(server.addr).await?;
    hello(&mut app).await?;
    send_json(&mut app, &relayed).await?;
    assert_eq!(
        next_json(&mut app, WAIT).await?,
        json!({"t": "source", "mode": "live"})
    );
    let quiet = Duration::from_millis(500);
    for (topic, data) in results() {
        assert_eq!(
            post(topic, &data.to_string()).await?.status(),
            204,
            "{topic}"
        );
    }
    let mut got = relayed_until_quiet(&mut app, quiet).await?;
    got.sort_by(|a, b| a.0.cmp(&b.0));
    let mut want: Vec<(String, Value)> = results()
        .into_iter()
        .map(|(topic, data)| (topic.to_owned(), data))
        .collect();
    want.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(got, want, "each result once, on its own topic");
    let mut pnl = results()[2].1.clone();
    assert_eq!(results()[2].0, "pnl");
    pnl["pnl"] = json!(7.5);
    assert_eq!(post("pnl", &pnl.to_string()).await?.status(), 204);
    let got = relayed_until_quiet(&mut app, quiet).await?;
    assert_eq!(got, vec![("pnl".to_owned(), pnl.clone())], "the newer P&L");
    // Latest wins: a later subscriber gets the meters and only the second P&L, and no append
    // topic's past.
    let mut later = ws_connect(server.addr).await?;
    hello(&mut later).await?;
    send_json(&mut later, &relayed).await?;
    let _ack = next_json(&mut later, WAIT).await?;
    let mut got = relayed_until_quiet(&mut later, quiet).await?;
    got.sort_by(|a, b| a.0.cmp(&b.0));
    let meters = results()[0].1.clone();
    assert_eq!(
        got,
        vec![("meters".to_owned(), meters), ("pnl".to_owned(), pnl)],
        "the latest per topic"
    );

    assert_eq!(post("meters", r#"{"v":1}"#).await?.status(), 204);
    assert_eq!(post("book", r#"{"v":1}"#).await?.status(), 404);
    assert_eq!(post("meters", "[1,2]").await?.status(), 400);
    assert_eq!(post("meters", "not json").await?.status(), 400);
    assert_eq!(post("jev", &"x".repeat(100 * 1024)).await?.status(), 413);
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A hub whose one venue has started, serving `dir`.
async fn started_server(dir: &Path, config: HubConfig) -> std::io::Result<Server> {
    let hub = Hub::new(config);
    let feed = VenueSource {
        scales: BTreeMap::new(),
        run: None,
    };
    assert!(hub.start(Venue::BinanceUsdm, feed));
    Server::start(hub, ApiConfig::new(dir.to_owned())).await
}

/// Reads past text frames to the close, and returns its code.
async fn close_code(ws: &mut Socket) -> Option<u16> {
    loop {
        match next_frame(ws, WAIT).await {
            Frame::Text(_) | Frame::Binary(_) => {}
            Frame::Closed(code) => return code,
            Frame::Silent => return None,
        }
    }
}

/// `backfill_s` is one rule on both routes (#87): a negative, NaN or infinite one is refused, a
/// 400 on the Arrow route and a 4000 close on the WebSocket, and one too big for a `Duration` is
/// capped, never a panic. Each route had its own copy of the rule, which nothing tested.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backfill_is_refused_or_capped_alike_on_both_routes() -> Outcome {
    let dir = scratch("api-backfill-rule");
    let server = started_server(&dir, HubConfig::default()).await?;
    let mut ws = ws_connect(server.addr).await?;
    hello(&mut ws).await?;
    let huge = json!({"t": "sub", "topics": ["book"], "backfill_s": 1e300});
    send_json(&mut ws, &huge).await?;
    assert_eq!(
        next_json(&mut ws, WAIT).await?,
        json!({"t": "source", "mode": "live"})
    );
    let negative = json!({"t": "sub", "topics": ["book"], "backfill_s": -1});
    send_json(&mut ws, &negative).await?;
    assert_eq!(close_code(&mut ws).await, Some(4000));

    // Ended, so an Arrow stream ends as soon as its backfill is sent.
    server.hub.end();
    for refused in ["-1", "NaN", "inf"] {
        let url = server.url(&format!("/v1/live/book_samples?backfill_s={refused}"));
        let response = reqwest::get(url).await?;
        assert_eq!(response.status(), 400, "backfill_s={refused}");
        let body = response.text().await?;
        assert!(body.contains("backfill_s must be"), "{body}");
    }
    let (_, batches) = get_arrow(&server.url("/v1/live/book_samples?backfill_s=1e300")).await?;
    assert!(batches.is_empty(), "nothing was published");
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A client of `addr` subscribed to `topic`, past the subscription's answer.
async fn subscribed(addr: std::net::SocketAddr, topic: &str) -> Outcome<Socket> {
    let mut ws = ws_connect(addr).await?;
    hello(&mut ws).await?;
    send_json(&mut ws, &json!({"t": "sub", "topics": [topic]})).await?;
    let ack = next_json(&mut ws, WAIT).await?;
    if ack != json!({"t": "source", "mode": "live"}) {
        return Err(format!("wanted the subscription answered, got {ack}").into());
    }
    Ok(ws)
}

/// What a client of an append topic got: the `id_field` of each item it received, in order, and
/// the losses its gaps on `topic` counted. Items come one per message, or, for trades, several.
fn received(messages: &[Value], topic: &str, id_field: &str) -> Outcome<(Vec<u64>, u64)> {
    let (mut ids, mut lost) = (Vec::new(), 0);
    for message in messages {
        let items = match message["t"].as_str() {
            Some("gap") if message["topic"] == topic => {
                lost += message["lost"].as_u64().ok_or("a count")?;
                continue;
            }
            Some("trades") => message["trades"].as_array().cloned().unwrap_or_default(),
            Some(t) if t == topic => vec![message["data"].clone()],
            _ => return Err(format!("not {topic}: {message}").into()),
        };
        for item in items {
            ids.push(item[id_field].as_u64().ok_or("an id")?);
        }
    }
    Ok((ids, lost))
}

/// A client that falls behind on an append topic is told so (#87, where the three topics' drain
/// loops became one): on ising and trades, a gap with the count lost and the newest after it,
/// every message either delivered or counted; on jev, which is never dropped, a 4002 close. The
/// hub holds two of each for a client, and a hundred are published at once, well inside the
/// 16 ms a connection waits between sends.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn falling_behind_on_an_append_topic_is_a_gap_or_a_close() -> Outcome {
    const BURST: u64 = 100;
    let dir = scratch("api-behind");
    let config = HubConfig {
        trades_capacity: 2,
        relay_capacity: 2,
        ..HubConfig::default()
    };
    let server = started_server(&dir, config).await?;
    let hub = Arc::clone(&server.hub);
    let quiet = Duration::from_millis(500);
    let check = |what: &str, (ids, lost): (Vec<u64>, u64)| {
        assert!(lost > 0, "{what}: nothing lost: {ids:?}");
        assert_eq!(
            lost + ids.len() as u64,
            BURST,
            "{what}: delivered or counted, each once"
        );
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "{what}: {ids:?}");
        assert_eq!(
            ids.last(),
            Some(&(BURST - 1)),
            "{what}: the newest is never lost"
        );
    };

    let mut ising = subscribed(server.addr, "ising").await?;
    for step in 0..BURST {
        hub.relay(Topic::Ising, json!({"run": ISING_RUN, "step": step}));
    }
    check(
        "ising",
        received(&until_quiet(&mut ising, quiet).await?, "ising", "step")?,
    );

    let mut trades = subscribed(server.addr, "trades").await?;
    for seq in 0..BURST {
        let trade = TradeMessage {
            venue: Venue::BinanceUsdm,
            symbol: "BTCUSDT".to_owned(),
            seq,
            ts_ms: 1_700_000_000_000 + seq,
            id: seq.to_string(),
            px: 500_000,
            qty: 1,
            buyer_is_maker: false,
        };
        hub.publish_trade(1_700_000_000_000_000_000, trade);
    }
    check(
        "trades",
        received(&until_quiet(&mut trades, quiet).await?, "trades", "seq")?,
    );

    let mut jev = subscribed(server.addr, "jev").await?;
    for n in 0..BURST {
        hub.relay(
            Topic::Jev,
            json!({"symbol": "BTCUSDT", "label": "regime_shift", "n": n}),
        );
    }
    assert_eq!(
        close_code(&mut jev).await,
        Some(4002),
        "jev is never dropped"
    );
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// When a client got its next `count` status messages, other messages skipped; an error unless
/// they all come within `within`.
async fn status_times(
    ws: &mut Socket,
    count: usize,
    within: Duration,
) -> Outcome<Vec<tokio::time::Instant>> {
    let deadline = tokio::time::Instant::now() + within;
    let mut times = Vec::new();
    while times.len() < count {
        let left = deadline.saturating_duration_since(tokio::time::Instant::now());
        match next_frame(ws, left).await {
            Frame::Text(text) => {
                if serde_json::from_str::<Value>(&text)?["t"] == "status" {
                    times.push(tokio::time::Instant::now());
                }
            }
            other => {
                let got = times.len();
                return Err(format!("{got} status messages in {within:?}, then {other:?}").into());
            }
        }
    }
    Ok(times)
}

/// Status goes out every second even unchanged, as the protocol promises (#85), and only a
/// production (coalescing) connection runs that heartbeat. With nothing published, a status-only
/// client hears it again a second apart; and so does a client taking books while samples are
/// published faster than it is sent to, where the heartbeat came last among what a connection
/// waits on and lost every turn to the hub's wake-up, which a busy hub keeps ready.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_status_goes_out_every_second_unchanged() -> Outcome {
    const WITHIN: Duration = Duration::from_secs(10);
    let dir = scratch("api-heartbeat");
    let server = started_server(&dir, HubConfig::default()).await?;
    let hub = Arc::clone(&server.hub);
    hub.publish_status(Venue::BinanceUsdm, VenueStatus::default());
    // The status as subscribed, then one a second after the last: n of them span n - 2 seconds at
    // least, so none came in a burst on a change.
    let periodic = |times: &[tokio::time::Instant]| {
        let seconds = u64::try_from(times.len().saturating_sub(2)).unwrap_or(0);
        times[times.len() - 1] - times[0] >= Duration::from_secs(seconds)
    };

    // Subscribed in the same breath as its hello, so the subscription is answered before the
    // heartbeat first ticks, and that tick sent the same status again 16 ms later (#102): now none
    // comes within half a second of another.
    let mut quiet = ws_connect(server.addr).await?;
    send_json(&mut quiet, &json!({"t": "hello", "v": PROTOCOL})).await?;
    send_json(&mut quiet, &json!({"t": "sub", "topics": ["status"]})).await?;
    assert_eq!(next_json(&mut quiet, WAIT).await?["t"], "welcome");
    let answer = next_json(&mut quiet, WAIT).await?;
    assert_eq!(answer, json!({"t": "source", "mode": "live"}));
    let times = status_times(&mut quiet, 4, WITHIN).await?;
    assert!(periodic(&times), "quiet: {times:?}");
    let apart = |w: &[tokio::time::Instant]| w[1] - w[0] >= Duration::from_millis(500);
    assert!(
        times.windows(2).all(apart),
        "quiet, back to back: {times:?}"
    );

    let publishing = Arc::new(AtomicBool::new(true));
    let publisher = std::thread::spawn({
        let (hub, publishing) = (Arc::clone(&hub), Arc::clone(&publishing));
        move || {
            let mut i = 0;
            while publishing.load(Ordering::Relaxed) {
                let mut sample = catalog_sample();
                sample.sample_ts_ns += i * 100_000_000;
                let captured_ns = sample.sample_ts_ns;
                let sample = Arc::new(sample);
                hub.publish_sample(Published {
                    sample,
                    ipc: Vec::new(),
                    captured_ns,
                });
                i += 1;
                std::thread::sleep(Duration::from_millis(1));
            }
            i
        }
    });
    let mut busy = ws_connect(server.addr).await?;
    hello(&mut busy).await?;
    send_json(
        &mut busy,
        &json!({"t": "sub", "topics": ["book", "status"]}),
    )
    .await?;
    // Five: starved, the heartbeat still slipped through now and then, when the publisher
    // missed a whole send interval, but never four times in ten seconds.
    let times = status_times(&mut busy, 5, WITHIN).await;
    publishing.store(false, Ordering::Relaxed);
    let samples = publisher.join().map_err(|_| "the publisher panicked")?;
    let times = times.map_err(|error| format!("busy, {samples} samples published: {error}"))?;
    assert!(periodic(&times), "busy: {times:?}");
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// The encoder's output is a stream a reader takes back whole; used by the goldens above.
#[test]
fn an_encoded_stream_round_trips() -> Outcome {
    let schema = engine::store::samples::SampleRows::schema(2);
    let (mut encoder, mut bytes) = Encoder::new(Arc::clone(&schema))?;
    let mut rows = engine::store::samples::SampleRows::new(2);
    rows.push(&Sample {
        sample_ts_ns: 1,
        seq: 0,
        venue: Venue::BinanceUsdm,
        rows: Vec::new(),
    });
    bytes.extend(encoder.batch(&rows.take()?)?);
    bytes.extend(encoder.finish()?);
    let (decoded, batches) = decode_stream(&bytes)?;
    assert_eq!(decoded, schema);
    assert_eq!(batches.len(), 1);
    Ok(())
}
