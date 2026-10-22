//! The API serves exactly what is stored. The committed session runs through the live path
//! with the server attached; then rotate. The live Arrow rows must equal the rotated Parquet
//! field for field, seq included, and each row's seq must be the input right before the one that
//! moved the capture clock past it. A read through the API equals both, for every table. Live
//! and replayed `book` messages are each an in-order subsequence of those rows (coalescing skips,
//! never alters), ending on the same last row per symbol; a replay through the API equals the
//! replay through the log that rotate made the rows from. A stream that fails partway fails at
//! the client; it never passes for a short table.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, BooleanArray, Int64Array, RecordBatch, StringArray};
use engine::api::ApiConfig;
use engine::api::hub::{Hub, HubConfig};
use engine::store::rotate::rotate;
use engine::store::writer::LogConfig;
use engine_harness::api::{
    Frame, LivePath, Outcome, Server, Socket, fixture, get_arrow, hello, next_frame, next_json,
    one_batch, read_parquet, script, send_json, ws_connect,
};
use serde_json::{Value, json};

const RUN: &str = "20231114T221320Z-session";
const WAIT: Duration = Duration::from_secs(20);

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch dir");
    dir
}

/// One `book_samples` row as a book message carries it.
type BookRow = (
    String,
    i64,
    i64,
    bool,
    Option<i64>,
    Vec<[i64; 2]>,
    Vec<[i64; 2]>,
);

fn int64<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int64Array {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .unwrap_or_else(|| panic!("an Int64 column {name}"))
}

/// The rows of `book_samples` batches, as book messages would carry them.
fn book_rows(batch: &RecordBatch) -> Vec<BookRow> {
    let symbols = batch
        .column_by_name("symbol")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .expect("a symbol column");
    let synced = batch
        .column_by_name("synced")
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        .expect("a synced column");
    let (times, seqs, ids) = (
        int64(batch, "sample_ts_ns"),
        int64(batch, "seq"),
        int64(batch, "last_update_id"),
    );
    let levels = |row: usize, side: &str| -> Vec<[i64; 2]> {
        (0..20)
            .map_while(|i| {
                let px = int64(batch, &format!("{side}_px_{i}"));
                let qty = int64(batch, &format!("{side}_qty_{i}"));
                px.is_valid(row).then(|| [px.value(row), qty.value(row)])
            })
            .collect()
    };
    (0..batch.num_rows())
        .map(|row| {
            (
                symbols.value(row).to_owned(),
                times.value(row) / 1_000_000,
                seqs.value(row),
                synced.value(row),
                ids.is_valid(row).then(|| ids.value(row)),
                levels(row, "bid"),
                levels(row, "ask"),
            )
        })
        .collect()
}

fn book_of(message: &Value) -> BookRow {
    let pairs = |side: &Value| -> Vec<[i64; 2]> {
        side.as_array()
            .expect("levels")
            .iter()
            .map(|pair| {
                [
                    pair[0].as_i64().expect("px"),
                    pair[1].as_i64().expect("qty"),
                ]
            })
            .collect()
    };
    (
        message["symbol"].as_str().expect("symbol").to_owned(),
        message["ts_ms"].as_i64().expect("ts_ms"),
        message["seq"].as_i64().expect("seq"),
        message["synced"].as_bool().expect("synced"),
        message["last_update_id"].as_i64(),
        pairs(&message["bids"]),
        pairs(&message["asks"]),
    )
}

/// Every message is one of `rows`, later ones later, and each symbol's last message is its last
/// row.
fn assert_subsequence(messages: &[BookRow], rows: &[BookRow], what: &str) {
    assert!(!messages.is_empty(), "{what}: no book messages");
    let mut at = 0;
    for message in messages {
        let found = rows[at..].iter().position(|row| row == message);
        let Some(offset) = found else {
            panic!("{what}: {message:?} is not a stored row after position {at}");
        };
        at += offset + 1;
    }
    let mut last_rows: BTreeMap<&str, &BookRow> = BTreeMap::new();
    for row in rows {
        last_rows.insert(&row.0, row);
    }
    let mut last_messages: BTreeMap<&str, &BookRow> = BTreeMap::new();
    for message in messages {
        last_messages.insert(&message.0, message);
    }
    assert_eq!(last_messages, last_rows, "{what}: the last book per symbol");
}

/// Skips what the live feed sent before the replay began (the newest live book goes to every
/// new subscriber) up to the source switching to the replay.
async fn until_replay(socket: &mut Socket) -> Outcome {
    loop {
        let message = next_json(socket, WAIT).await?;
        if message["t"] == "source" && message["mode"] == "replay" {
            return Ok(());
        }
    }
}

/// Reads book messages (and trades) until the source says it ended.
async fn read_until_ended(socket: &mut Socket) -> Outcome<(Vec<BookRow>, Vec<Value>, Vec<Value>)> {
    let (mut books, mut trades, mut others) = (Vec::new(), Vec::new(), Vec::new());
    loop {
        let Frame::Text(text) = next_frame(socket, WAIT).await else {
            return Err("the socket stopped before the source ended".into());
        };
        let message: Value = serde_json::from_str(&text)?;
        match message["t"].as_str() {
            Some("book") => books.push(book_of(&message)),
            Some("trades") => {
                trades.extend(message["trades"].as_array().cloned().unwrap_or_default());
            }
            Some("source") if message["mode"] == "ended" => return Ok((books, trades, others)),
            _ => others.push(message),
        }
    }
}

/// What the live phase left: the server still up, the rotated run, and what the live
/// subscribers received.
struct Played {
    dir: PathBuf,
    server: Server,
    run_dir: PathBuf,
    live: RecordBatch,
    live_books: Vec<BookRow>,
    live_trades: Vec<Value>,
    /// The app's replay of the run while part of it was rotated and the rest only in memory.
    mixed_books: Vec<BookRow>,
    mixed_trades: Vec<Value>,
    /// Every input's seq, as the loop numbered it.
    inputs: BTreeSet<u64>,
}

/// The session through the live path with an Arrow subscriber and a coalescing app client
/// connected from the start, then the log closed and rotated.
async fn play(dir: PathBuf) -> Outcome<Played> {
    let (scales, messages) = script(&fixture("synthetic/session-faults.jsonl"))?;
    let hub = Hub::new(HubConfig::default());
    let small = LogConfig {
        max_segment_bytes: 2048,
        ..LogConfig::default()
    };
    let mut live = LivePath::new(scales, Arc::clone(&hub), &dir, Some(RUN), small)?;
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.clone())).await?;
    let url = server.url("/v1/live/book_samples");
    let arrow = tokio::spawn(async move { get_arrow(&url).await });
    let mut app = ws_connect(server.addr).await?;
    hello(&mut app).await?;
    send_json(&mut app, &json!({"t": "sub", "topics": ["book", "trades"]})).await?;
    let ack = next_json(&mut app, WAIT).await?;
    assert_eq!(ack, json!({"t": "source", "mode": "live"}));
    while hub.arrow_subscribers() == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for message in messages {
        live.push(message).await?;
        // Real time between inputs, so the app client coalesces some samples and sends others.
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    // Rotate while the log is still open: the closed segments become Parquet, the newest stays
    // in the log, so a replay now reads the rotated hours through DuckDB and the rest from memory.
    let run_dir = dir.join("market").join(RUN);
    let early = rotate(&run_dir)?;
    assert!(early.open.is_some() && early.segments >= 2, "{early:?}");
    hub.end();
    let (live_schema, live_batches) = arrow.await??;
    let (live_books, live_trades, _) = read_until_ended(&mut app).await?;
    let mut viewer = ws_connect(server.addr).await?;
    hello(&mut viewer).await?;
    send_json(
        &mut viewer,
        &json!({"t": "sub", "topics": ["book", "trades"]}),
    )
    .await?;
    let _ack = next_json(&mut viewer, WAIT).await?;
    send_json(&mut viewer, &json!({"t": "replay", "run": RUN, "speed": 0})).await?;
    until_replay(&mut viewer).await?;
    let (mixed_books, mixed_trades, _) = read_until_ended(&mut viewer).await?;

    let (summary, watched, _) = live.finish().await?;
    assert!(summary.ok_or("recorded")?.segments >= 3);
    rotate(&run_dir)?;
    Ok(Played {
        dir,
        server,
        run_dir,
        live: one_batch(&live_schema, &live_batches)?,
        live_books,
        live_trades,
        mixed_books,
        mixed_trades,
        inputs: watched.inputs.iter().map(|(seq, _, _)| *seq).collect(),
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_api_serves_exactly_what_is_stored() -> Outcome {
    let played = play(scratch("api-parity")).await?;

    // Live rows equal the rotated Parquet, seq included.
    let (stored_schema, stored_batches) = read_parquet(&played.run_dir, "book_samples")?;
    let stored = one_batch(&stored_schema, &stored_batches)?;
    assert_eq!(played.live, stored, "the live stream is the rotated table");
    assert!(stored.num_rows() >= 30);

    // Each row's seq is the input right before the one that moved the clock past its instant.
    let seqs = int64(&stored, "seq");
    for row in 0..stored.num_rows() {
        let seq = u64::try_from(seqs.value(row))?;
        assert!(
            played.inputs.contains(&(seq + 1)),
            "row {row}: seq {seq} + 1 is not an input"
        );
    }

    check_table_reads(&played).await?;

    // Live book messages: an in-order subsequence of the rows, and every trade arrived.
    let rows = book_rows(&stored);
    assert_subsequence(&played.live_books, &rows, "live");
    let (_, trades) = read_parquet(&played.run_dir, "agg_trades")?;
    let trades: usize = trades.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(
        played.live_trades.len(),
        trades,
        "every live trade reached the app"
    );

    check_replays(&played, &stored, &rows).await?;
    assert_subsequence(
        &played.mixed_books,
        &rows,
        "replay of rotated hours, then memory",
    );
    let seqs = |trades: &[Value]| trades.iter().map(|t| t["seq"].clone()).collect::<Vec<_>>();
    assert_eq!(
        seqs(&played.mixed_trades),
        seqs(&played.live_trades),
        "trades replayed from Parquet then memory are the live ones"
    );
    played.server.stop().await;
    fs::remove_dir_all(&played.dir)?;
    Ok(())
}

/// A read through the API (`DuckDB`, cast back to the stored schema) equals the Parquet, every
/// table; a key range is a range of seq.
async fn check_table_reads(played: &Played) -> Outcome {
    let tables = [
        "frames",
        "depth_updates",
        "agg_trades",
        "snapshots",
        "book_samples",
    ];
    for table in tables {
        let url = played
            .server
            .url(&format!("/v1/table/market/{RUN}/{table}"));
        let (schema, batches) = get_arrow(&url).await?;
        let (parquet_schema, parquet) = read_parquet(&played.run_dir, table)?;
        assert_eq!(
            schema.fields(),
            parquet_schema.fields(),
            "{table}: stored schema"
        );
        assert_eq!(
            one_batch(&schema, &batches)?,
            one_batch(&parquet_schema, &parquet)?,
            "{table}: API read"
        );
    }
    let url = played
        .server
        .url(&format!("/v1/table/market/{RUN}/frames?start=10&end=20"));
    let (ranged_schema, ranged) = get_arrow(&url).await?;
    let ranged = one_batch(&ranged_schema, &ranged)?;
    assert_eq!(
        int64(&ranged, "seq").values().to_vec(),
        (10..20).collect::<Vec<i64>>()
    );
    Ok(())
}

/// Replays through the API: the Arrow replay-as-live, paced and unpaced, is the stored table,
/// which rotate made by replaying the log; the app's replay sends those rows as book messages
/// and every trade, and a seek replays from there.
async fn check_replays(played: &Played, stored: &RecordBatch, rows: &[BookRow]) -> Outcome {
    for speed in ["0", "50"] {
        let url = played
            .server
            .url(&format!("/v1/live/book_samples?run={RUN}&speed={speed}"));
        let (schema, batches) = get_arrow(&url).await?;
        assert_eq!(
            &one_batch(&schema, &batches)?,
            stored,
            "replay at speed {speed}"
        );
        if speed != "0" {
            assert!(
                batches.len() >= stored.num_rows() / 2,
                "one batch per instant"
            );
        }
    }
    let mut viewer = ws_connect(played.server.addr).await?;
    hello(&mut viewer).await?;
    send_json(
        &mut viewer,
        &json!({"t": "sub", "topics": ["book", "trades"]}),
    )
    .await?;
    let _ack = next_json(&mut viewer, WAIT).await?;
    send_json(&mut viewer, &json!({"t": "replay", "run": RUN, "speed": 0})).await?;
    until_replay(&mut viewer).await?;
    let (books, trades, _) = read_until_ended(&mut viewer).await?;
    assert_subsequence(&books, rows, "replay");
    let seqs = |trades: &[Value]| trades.iter().map(|t| t["seq"].clone()).collect::<Vec<_>>();
    assert_eq!(
        seqs(&trades),
        seqs(&played.live_trades),
        "replayed trades are the live ones"
    );

    let from_ms = rows[rows.len() / 2].1;
    let seek = json!({"t": "replay", "run": RUN, "from_ms": from_ms, "speed": 0});
    send_json(&mut viewer, &seek).await?;
    until_replay(&mut viewer).await?;
    let (sought, _, _) = read_until_ended(&mut viewer).await?;
    assert!(
        sought.iter().all(|book| book.1 >= from_ms),
        "nothing before the seek"
    );
    let later: Vec<BookRow> = rows
        .iter()
        .filter(|row| row.1 >= from_ms)
        .cloned()
        .collect();
    assert_subsequence(&sought, &later, "replay after a seek");
    Ok(())
}

/// The app's replay controls on one connection: a replay at real speed, paused (nothing comes),
/// played on, sought forward (only later samples come), then back to the live feed, each change
/// answered by a `source` message.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_replay_controls_pause_play_seek_and_go_live() -> Outcome {
    let played = play(scratch("api-controls")).await?;
    let (stored_schema, stored_batches) = read_parquet(&played.run_dir, "book_samples")?;
    let rows = book_rows(&one_batch(&stored_schema, &stored_batches)?);
    let (first_ms, last_ms) = (rows[0].1, rows[rows.len() - 1].1);
    let mut viewer = ws_connect(played.server.addr).await?;
    hello(&mut viewer).await?;
    send_json(&mut viewer, &json!({"t": "sub", "topics": ["book"]})).await?;
    let _ack = next_json(&mut viewer, WAIT).await?;
    send_json(&mut viewer, &json!({"t": "replay", "run": RUN, "speed": 1})).await?;
    until_replay(&mut viewer).await?;

    send_json(&mut viewer, &json!({"t": "pause"})).await?;
    let paused = until_source(&mut viewer, "paused").await?;
    assert_eq!(paused["run"], RUN);
    let quiet = next_frame(&mut viewer, Duration::from_millis(400)).await;
    assert_eq!(quiet, Frame::Silent, "nothing comes while paused");

    send_json(&mut viewer, &json!({"t": "play"})).await?;
    until_source(&mut viewer, "replay").await?;
    let resumed = next_json(&mut viewer, WAIT).await?;
    assert_eq!(resumed["t"], "book", "{resumed}");

    let to_ms = first_ms + (last_ms - first_ms) * 3 / 4;
    send_json(&mut viewer, &json!({"t": "seek", "to_ms": to_ms})).await?;
    let sought = until_source(&mut viewer, "replay").await?;
    assert_eq!(sought["from_ms"], to_ms);
    let (books, _, _) = read_until_ended(&mut viewer).await?;
    assert!(
        !books.is_empty() && books.iter().all(|book| book.1 >= to_ms),
        "{books:?}"
    );

    send_json(&mut viewer, &json!({"t": "live"})).await?;
    until_source(&mut viewer, "live").await?;
    played.server.stop().await;
    fs::remove_dir_all(&played.dir)?;
    Ok(())
}

/// Skips to the next `source` message in `mode`, and returns it.
async fn until_source(socket: &mut Socket, mode: &str) -> Outcome<Value> {
    loop {
        let message = next_json(socket, WAIT).await?;
        if message["t"] == "source" && message["mode"] == mode {
            return Ok(message);
        }
    }
}

/// A failure partway through a stream reaches the client as a failure: the response is aborted
/// and the end-of-stream marker never comes. Injected here after k batches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stream_that_fails_partway_fails_at_the_client() -> Outcome {
    let dir = scratch("api-fail");
    let (scales, messages) = script(&fixture("synthetic/session-faults.jsonl"))?;
    let hub = Hub::new(HubConfig::default());
    let mut live = LivePath::new(
        scales,
        Arc::clone(&hub),
        &dir,
        Some(RUN),
        LogConfig::default(),
    )?;
    let config = ApiConfig {
        fail_after: Some(3),
        ..ApiConfig::new(dir.clone())
    };
    let server = Server::start(Arc::clone(&hub), config).await?;
    let url = server.url("/v1/live/book_samples");
    let arrow = tokio::spawn(async move { get_arrow(&url).await });
    while hub.arrow_subscribers() == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for message in messages {
        live.push(message).await?;
    }
    hub.end();
    let cut = arrow.await?;
    assert!(
        cut.is_err(),
        "a cut live stream must not read as complete: {cut:?}"
    );
    live.finish().await?;
    rotate(&dir.join("market").join(RUN))?;
    let table = get_arrow(&server.url(&format!("/v1/table/market/{RUN}/frames"))).await;
    assert!(
        table.is_ok(),
        "fewer batches than the failure point pass: {table:?}"
    );
    let replay = get_arrow(&server.url(&format!("/v1/live/book_samples?run={RUN}&speed=50"))).await;
    assert!(
        replay.is_err(),
        "a cut replay must not read as complete: {replay:?}"
    );
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}
