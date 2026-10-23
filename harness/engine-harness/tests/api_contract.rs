//! The API's contract, pinned by golden files under `engine/tests/fixtures/api/`: every WebSocket
//! message the engine sends or accepts, the lattice frame, every Arrow schema it serves, and a
//! small Arrow stream. A protocol change fails here until the goldens are regenerated in the same
//! commit (`TP_UPDATE_GOLDEN=1 cargo test -p engine-harness`). The app's mock replays the same
//! files, so its fixtures come from the engine.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use engine::api::ApiConfig;
use engine::api::hub::{Hub, HubConfig, Source, VenueSource};
use engine::api::ipc::{Encoder, SCHEMA_KEY, major};
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
    Frame, ISING_RUN, LivePath, Outcome, Script, Server, copy_ising_fixture, decode_stream,
    fixture, get_arrow, golden, next_frame, next_json, one_batch, script, send_json, ws_connect,
};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

const RUN: &str = "20231114T221320Z-session";
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

/// The committed session through the live path, with the API attached and a lossless client
/// and an Arrow subscriber connected from the start; then results published for the app. The
/// WebSocket frames, as text, in the order they arrived, and the live Arrow stream's bytes.
async fn record_session(dir: &Path) -> Outcome<(Vec<String>, Vec<u8>, PathBuf)> {
    let Script {
        venue,
        scales,
        messages,
    } = script(&fixture("synthetic/session-faults.jsonl"))?;
    let hub = Hub::new(HubConfig {
        lossless: true,
        ..HubConfig::default()
    });
    let config = LogConfig {
        max_segment_bytes: 2048,
        ..LogConfig::default()
    };
    let mut live = LivePath::new(venue, scales, Arc::clone(&hub), dir, Some(RUN), config)?;
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.to_owned())).await?;

    let mut ws = ws_connect(server.addr).await?;
    send_json(&mut ws, &json!({"t": "hello", "v": PROTOCOL})).await?;
    let mut frames = Vec::new();
    let Frame::Text(welcome) = next_frame(&mut ws, WAIT).await else {
        return Err("no welcome".into());
    };
    frames.push(welcome);
    let topics: Vec<&str> = Topic::ALL.iter().map(|topic| topic.name()).collect();
    send_json(&mut ws, &json!({"t": "sub", "topics": topics})).await?;
    let Frame::Text(ack) = next_frame(&mut ws, WAIT).await else {
        return Err("no answer to sub".into());
    };
    frames.push(ack);

    let url = server.url("/v1/live/book_samples");
    let arrow = tokio::spawn(async move { reqwest::get(url).await?.bytes().await });
    while hub.arrow_subscribers() == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for message in messages {
        live.push(message).await?;
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
    let (_summary, _, _) = live.finish().await?;
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
    Status {
        ts_ms: 1_700_000_000_300,
        venues: BTreeMap::from([(Venue::BinanceUsdm, binance)]),
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
        venues: BTreeMap::from([(
            Venue::BinanceUsdm,
            VenueSource {
                scales,
                run: Some(RUN.to_owned()),
            },
        )]),
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
        let id = format!("{table}/{}", major(domain, table));
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

/// A foreign `Origin` is refused, the WebSocket upgrade included, and so is a `Host` that names
/// another server (DNS rebinding); the allowed origins and a client without `Origin` get in.
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

/// Publishing takes JSON objects on the relayed topics only.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn only_objects_on_relayed_topics_are_published() -> Outcome {
    let dir = scratch("api-publish");
    let server = Server::start(Hub::new(HubConfig::default()), ApiConfig::new(dir.clone())).await?;
    let client = reqwest::Client::new();
    let post = |topic: &str, body: &str| {
        client
            .post(server.url(&format!("/v1/publish/{topic}")))
            .body(body.to_owned())
            .send()
    };
    assert_eq!(post("meters", r#"{"v":1}"#).await?.status(), 204);
    assert_eq!(post("book", r#"{"v":1}"#).await?.status(), 404);
    assert_eq!(post("meters", "[1,2]").await?.status(), 400);
    assert_eq!(post("meters", "not json").await?.status(), 400);
    assert_eq!(post("jev", &"x".repeat(100 * 1024)).await?.status(), 413);
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
