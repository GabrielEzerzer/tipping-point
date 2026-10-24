//! The API serves exactly what is stored. The committed session runs through the live path
//! with the server attached; then rotate. The live Arrow rows must equal the rotated Parquet
//! field for field, seq included, and each row's seq must be the input right before the one that
//! moved the capture clock past it. A read through the API equals both, for every table. Live
//! and replayed `book` messages are each an in-order subsequence of those rows (coalescing skips,
//! never alters), ending on the same last row per symbol; a replay through the API equals the
//! replay through the log that rotate made the rows from. A stream that fails partway fails at
//! the client; it never passes for a short table. Two venues fed at once, each by its own loop,
//! serve each venue's rows exactly as that venue's own run stores them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, BooleanArray, Int64Array, RecordBatch, StringArray};
use engine::api::ApiConfig;
use engine::api::hub::{Hub, HubConfig};
use engine::api::stream::Fault;
use engine::store::rotate::rotate;
use engine::store::writer::{LogConfig, LogWriter};
use engine::venue::Venue;
use engine_harness::api::{
    Frame, LivePath, Outcome, Script, Server, Socket, fixture, get_arrow, hello, merged,
    next_frame, next_json, one_batch, read_parquet, script, send_json, ws_connect,
};
use engine_harness::load::Generator;
use serde_json::{Value, json};

const RUN: &str = "20231114T221320Z-session";
const WAIT: Duration = Duration::from_secs(20);

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch dir");
    dir
}

/// One `book_samples` row as a book message carries it: venue, symbol, instant (ms), seq,
/// synced, last update id, bids, asks.
type BookRow = (
    String,
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
    let strings = |name: &str| {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            .unwrap_or_else(|| panic!("a {name} column"))
    };
    let (venues, symbols) = (strings("venue"), strings("symbol"));
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
                venues.value(row).to_owned(),
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
        message["venue"].as_str().expect("venue").to_owned(),
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
    let mut last_rows: BTreeMap<(&str, &str), &BookRow> = BTreeMap::new();
    for row in rows {
        last_rows.insert((&row.0, &row.1), row);
    }
    let mut last_messages: BTreeMap<(&str, &str), &BookRow> = BTreeMap::new();
    for message in messages {
        last_messages.insert((&message.0, &message.1), message);
    }
    assert_eq!(
        last_messages, last_rows,
        "{what}: the last book per venue and symbol"
    );
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
    let Script {
        venue,
        scales,
        messages,
    } = script(&fixture("synthetic/session-faults.jsonl"))?;
    let hub = Hub::new(HubConfig::default());
    let small = LogConfig {
        max_segment_bytes: 2048,
        ..LogConfig::default()
    };
    let mut live = LivePath::new(venue, scales, Arc::clone(&hub), &dir, Some(RUN), small)?;
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
    // The writer thread can lag the loop on a loaded machine; let it catch up first, so the
    // segments the loop has closed are on disk when rotate looks; a writer that stalls fails here.
    let deadline = tokio::time::Instant::now() + WAIT;
    while let Some(queued) = live.log.writer().map(LogWriter::queued).filter(|&q| q > 0) {
        if tokio::time::Instant::now() > deadline {
            return Err(
                format!("the log writer still had {queued} frames queued after {WAIT:?}").into(),
            );
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
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
    let (_, trades) = read_parquet(&played.run_dir, "trades")?;
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
        "trades",
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

    let from_ms = rows[rows.len() / 2].2;
    let seek = json!({"t": "replay", "run": RUN, "from_ms": from_ms, "speed": 0});
    send_json(&mut viewer, &seek).await?;
    until_replay(&mut viewer).await?;
    let (sought, _, _) = read_until_ended(&mut viewer).await?;
    assert!(
        sought.iter().all(|book| book.2 >= from_ms),
        "nothing before the seek"
    );
    let later: Vec<BookRow> = rows
        .iter()
        .filter(|row| row.2 >= from_ms)
        .cloned()
        .collect();
    assert_subsequence(&sought, &later, "replay after a seek");
    Ok(())
}

/// The app's replay controls on one connection: a replay at real speed, paused (nothing comes),
/// played on, sought forward (only later samples come), controls racing its end, then back to the
/// live feed, each change answered by a `source` message.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_replay_controls_pause_play_seek_and_go_live() -> Outcome {
    let played = play(scratch("api-controls")).await?;
    let (stored_schema, stored_batches) = read_parquet(&played.run_dir, "book_samples")?;
    let rows = book_rows(&one_batch(&stored_schema, &stored_batches)?);
    let (first_ms, last_ms) = (rows[0].2, rows[rows.len() - 1].2);
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
        !books.is_empty() && books.iter().all(|book| book.2 >= to_ms),
        "{books:?}"
    );

    // Controls sent as the replay ran out, which the engine has before the client hears `ended`:
    // pause and play are told it ended, and a seek starts it again.
    for control in ["pause", "play"] {
        send_json(&mut viewer, &json!({"t": control})).await?;
        let told = next_json(&mut viewer, WAIT).await?;
        let ended = json!({"t": "source", "mode": "ended", "run": RUN});
        assert_eq!(told, ended, "{control} after the end");
    }
    let near_end = last_ms - 500;
    send_json(&mut viewer, &json!({"t": "seek", "to_ms": near_end})).await?;
    let again = until_source(&mut viewer, "replay").await?;
    assert_eq!(again["from_ms"], near_end);
    let (books, _, _) = read_until_ended(&mut viewer).await?;
    assert!(books.iter().all(|book| book.2 >= near_end), "{books:?}");

    send_json(&mut viewer, &json!({"t": "live"})).await?;
    until_source(&mut viewer, "live").await?;
    played.server.stop().await;
    fs::remove_dir_all(&played.dir)?;
    Ok(())
}

/// Going back to live after a replay brings the trades back (#84). Starting a replay drops the
/// connection's live trades; only going back to live subscribes them again. Here the feed is
/// still running: a viewer subscribed to book and trades replays the live run to its end
/// halfway through, goes back to live, and gets the rest of the session, every trade and newer
/// books, none of them from before.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn going_back_to_live_after_a_replay_brings_the_trades_back() -> Outcome {
    const LIVE_RUN: &str = "20261024T120000Z-golive";
    const T0: i64 = 1_700_000_000_000_000_000;
    /// Each half: a depth update and a trade every 50 ms of the capture clock.
    const HALF: usize = 100;
    let dir = scratch("api-go-live");
    let mut generator = Generator::default();
    let hub = Hub::new(HubConfig::default());
    let mut live = LivePath::new(
        Venue::BinanceUsdm,
        generator.scales(),
        Arc::clone(&hub),
        &dir,
        Some(LIVE_RUN),
        LogConfig::default(),
    )?;
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.clone())).await?;
    let mut at = T0;
    for message in generator.opening(at) {
        live.push(message).await?;
    }
    for _ in 0..HALF {
        at += 50_000_000;
        live.push(generator.depth(at)).await?;
        live.push(generator.trade(at)).await?;
    }

    let mut viewer = ws_connect(server.addr).await?;
    hello(&mut viewer).await?;
    let sub = json!({"t": "sub", "topics": ["book", "trades"]});
    send_json(&mut viewer, &sub).await?;
    until_source(&mut viewer, "live").await?;
    let replay = json!({"t": "replay", "run": LIVE_RUN, "speed": 0});
    send_json(&mut viewer, &replay).await?;
    until_replay(&mut viewer).await?;
    let (books, trades, _) = read_until_ended(&mut viewer).await?;
    let seqs = |trades: &[Value]| -> Vec<u64> {
        trades.iter().filter_map(|t| t["seq"].as_u64()).collect()
    };
    let replayed = seqs(&trades);
    assert_eq!(
        replayed.len(),
        HALF,
        "the replay sent the first half's trades"
    );
    let last_trade = replayed.iter().copied().max().ok_or("replayed trades")?;
    let last_book = books.iter().map(|b| b.2).max().ok_or("replayed books")?;

    send_json(&mut viewer, &json!({"t": "live"})).await?;
    until_source(&mut viewer, "live").await?;
    for _ in 0..HALF {
        at += 50_000_000;
        live.push(generator.depth(at)).await?;
        live.push(generator.trade(at)).await?;
    }
    hub.end();
    let (books, trades, _) = read_until_ended(&mut viewer).await?;
    let after = seqs(&trades);
    assert_eq!(
        after.len(),
        HALF,
        "every trade of the rest of the session, live: {after:?}"
    );
    assert!(
        after.windows(2).all(|w| w[0] < w[1]) && after.first() > Some(&last_trade),
        "the live trades, in order and after the replay's last ({last_trade}): {after:?}"
    );
    assert!(
        books.iter().all(|b| b.2 >= last_book) && books.iter().any(|b| b.2 > last_book),
        "the live books, from the newest on: {books:?}"
    );
    live.finish().await?;
    server.stop().await;
    fs::remove_dir_all(&dir)?;
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
    let Script {
        venue,
        scales,
        messages,
    } = script(&fixture("synthetic/session-faults.jsonl"))?;
    let hub = Hub::new(HubConfig::default());
    let mut live = LivePath::new(
        venue,
        scales,
        Arc::clone(&hub),
        &dir,
        Some(RUN),
        LogConfig::default(),
    )?;
    let config = ApiConfig {
        fault: Some(Fault::Abort { after: 3 }),
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

/// A producer that panics partway, as a bug would, aborts its response as an error does: the
/// client never reads the well-formed prefix as a complete table, on the live stream (an async
/// producer) or on a replay and a table read (producers on blocking threads). The server serves
/// on. A replay speed so slow its waits would overflow the clock is refused, not a panic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_producer_that_panics_fails_at_the_client() -> Outcome {
    let dir = scratch("api-panic");
    let Script {
        venue,
        scales,
        messages,
    } = script(&fixture("synthetic/session-faults.jsonl"))?;
    let hub = Hub::new(HubConfig::default());
    let mut live = LivePath::new(
        venue,
        scales,
        Arc::clone(&hub),
        &dir,
        Some(RUN),
        LogConfig::default(),
    )?;
    let config = ApiConfig {
        fault: Some(Fault::Panic { after: 1 }),
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
    // The HTTP body itself fails: a body that ended cleanly without the end marker would read as
    // complete to pyarrow, so a missing marker alone is not enough.
    let body_failed =
        |read: &Result<_, String>| read.as_ref().is_err_and(|e| e.starts_with("body"));
    assert!(body_failed(&cut), "the live stream ended cleanly: {cut:?}");
    live.finish().await?;
    rotate(&dir.join("market").join(RUN))?;
    for path in [
        format!("/v1/live/book_samples?run={RUN}&speed=0"),
        format!("/v1/table/market/{RUN}/frames"),
    ] {
        let read = get_arrow(&server.url(&path)).await;
        assert!(body_failed(&read), "{path} ended cleanly: {read:?}");
    }
    let health = reqwest::get(server.url("/v1/health")).await?;
    assert_eq!(health.status(), 200, "the server serves on");
    let slow = server.url(&format!("/v1/live/book_samples?run={RUN}&speed=1e-300"));
    assert_eq!(reqwest::get(slow).await?.status(), 400);
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Replaying the live run when memory no longer reaches back to the requested start, or to the
/// run's first sample when the replay names no start (nothing rotated, a ring of 400 ms): the
/// client is told how many instants are missing with a book gap, before the samples memory still
/// holds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replay_says_when_memory_does_not_reach_back() -> Outcome {
    let dir = scratch("api-gap");
    let Script {
        venue,
        scales,
        messages,
    } = script(&fixture("synthetic/session-faults.jsonl"))?;
    let hub = Hub::new(HubConfig {
        ring: Duration::from_millis(400),
        ..HubConfig::default()
    });
    let mut live = LivePath::new(
        venue,
        scales,
        Arc::clone(&hub),
        &dir,
        Some(RUN),
        LogConfig::default(),
    )?;
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.clone())).await?;
    let url = server.url("/v1/live/book_samples");
    let arrow = tokio::spawn(async move { get_arrow(&url).await });
    while hub.arrow_subscribers() == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for message in messages {
        live.push(message).await?;
    }
    hub.end();
    let (schema, batches) = arrow.await??;
    let all = book_rows(&one_batch(&schema, &batches)?);
    let first_ms = all.first().ok_or("the session has samples")?.2;
    let held = hub.recent(Venue::BinanceUsdm, i64::MIN).samples;
    let first_held_ms = held.first().ok_or("memory holds samples")?.sample_ts_ns / 1_000_000;
    assert!(
        first_held_ms > first_ms + 100,
        "the ring must have dropped samples"
    );

    let mut viewer = ws_connect(server.addr).await?;
    hello(&mut viewer).await?;
    send_json(&mut viewer, &json!({"t": "sub", "topics": ["book"]})).await?;
    let _ack = next_json(&mut viewer, WAIT).await?;
    let replay = json!({"t": "replay", "run": RUN, "from_ms": first_ms, "speed": 0});
    send_json(&mut viewer, &replay).await?;
    until_replay(&mut viewer).await?;
    let gap = next_json(&mut viewer, WAIT).await?;
    let lost = (first_held_ms - first_ms) / 100;
    assert_eq!(gap, json!({"t": "gap", "topic": "book", "lost": lost}));
    // Unpaced, the books coalesce to the newest; whatever comes is from memory, after the gap.
    let (books, _, _) = read_until_ended(&mut viewer).await?;
    assert!(
        books.iter().all(|book| book.2 >= first_held_ms),
        "{books:?}"
    );
    let held_rows: Vec<BookRow> = all
        .iter()
        .filter(|row| row.2 >= first_held_ms)
        .cloned()
        .collect();
    assert_subsequence(&books, &held_rows, "replay from memory after a gap");
    // From before the run's first sample, as from the run's start: the same instants are lost.
    let early = json!({"t": "replay", "run": RUN, "from_ms": first_ms - 2_000, "speed": 0});
    send_json(&mut viewer, &early).await?;
    until_replay(&mut viewer).await?;
    let gap = next_json(&mut viewer, WAIT).await?;
    assert_eq!(gap, json!({"t": "gap", "topic": "book", "lost": lost}));
    read_until_ended(&mut viewer).await?;
    // With no start at all, the replay begins at the run's first sample: the same are lost.
    let whole = json!({"t": "replay", "run": RUN, "speed": 0});
    send_json(&mut viewer, &whole).await?;
    until_replay(&mut viewer).await?;
    let gap = next_json(&mut viewer, WAIT).await?;
    assert_eq!(gap, json!({"t": "gap", "topic": "book", "lost": lost}));
    read_until_ended(&mut viewer).await?;
    live.finish().await?;
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A replay that cannot go on (a damaged Parquet file it reaches) tells the client it failed and
/// why, never that the run ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replay_that_fails_says_so_and_never_ended() -> Outcome {
    let played = play(scratch("api-replay-fail")).await?;
    let mut files: Vec<PathBuf> = fs::read_dir(played.run_dir.join("book_samples"))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    files.sort();
    let last = files.last().ok_or("book_samples files")?;
    assert!(files.len() >= 2, "the replay starts on a good file");
    let bytes = fs::read(last)?;
    fs::write(last, &bytes[..bytes.len() / 2])?;

    let mut viewer = ws_connect(played.server.addr).await?;
    hello(&mut viewer).await?;
    send_json(&mut viewer, &json!({"t": "sub", "topics": ["book"]})).await?;
    let _ack = next_json(&mut viewer, WAIT).await?;
    send_json(&mut viewer, &json!({"t": "replay", "run": RUN, "speed": 0})).await?;
    until_replay(&mut viewer).await?;
    let outcome = loop {
        let message = next_json(&mut viewer, WAIT).await?;
        if message["t"] == "source" {
            break message;
        }
    };
    assert_eq!(outcome["mode"], "failed", "{outcome}");
    assert_eq!(outcome["run"], RUN);
    assert!(
        outcome["error"].as_str().is_some_and(|e| !e.is_empty()),
        "{outcome}"
    );
    played.server.stop().await;
    fs::remove_dir_all(&played.dir)?;
    Ok(())
}

/// What a two-venue live phase left: the rotated runs' folders, the live Arrow rows, and what the
/// app received.
struct TwoVenues {
    dir: PathBuf,
    live: Vec<BookRow>,
    books: Vec<BookRow>,
    trades: Vec<Value>,
    others: Vec<Value>,
}

/// The Bybit run recorded beside the Binance one.
const BYBIT_RUN: &str = "20231114T221320Z-bybit";

/// Binance's and Bybit's sessions at once, each through its own live path into one hub, with an
/// Arrow subscriber and a coalescing app client connected from the start; then both runs rotated.
async fn play_two_venues(dir: PathBuf) -> Outcome<TwoVenues> {
    let binance = script(&fixture("synthetic/session-faults.jsonl"))?;
    let bybit = script(&fixture("synthetic/bybit-session.jsonl"))?;
    let hub = Hub::new(HubConfig {
        venues: vec![binance.venue, bybit.venue],
        ..HubConfig::default()
    });
    let small = LogConfig {
        max_segment_bytes: 2048,
        ..LogConfig::default()
    };
    let mut loops = [
        LivePath::new(
            binance.venue,
            binance.scales,
            Arc::clone(&hub),
            &dir,
            Some(RUN),
            small.clone(),
        )?,
        LivePath::new(
            bybit.venue,
            bybit.scales,
            Arc::clone(&hub),
            &dir,
            Some(BYBIT_RUN),
            small,
        )?,
    ];
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.clone())).await?;
    let url = server.url("/v1/live/book_samples");
    let arrow = tokio::spawn(async move { get_arrow(&url).await });
    let mut app = ws_connect(server.addr).await?;
    let welcome = hello(&mut app).await?;
    assert_eq!(
        welcome["runs"],
        json!({"binance-usdm": RUN, "bybit-linear": BYBIT_RUN})
    );
    let venues: BTreeSet<&str> = welcome["symbols"]
        .as_array()
        .ok_or("welcome symbols")?
        .iter()
        .filter_map(|s| s["venue"].as_str())
        .collect();
    assert_eq!(venues, BTreeSet::from(["binance-usdm", "bybit-linear"]));
    send_json(
        &mut app,
        &json!({"t": "sub", "topics": ["book", "trades", "status"]}),
    )
    .await?;
    while hub.arrow_subscribers() == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for (which, message) in merged(binance.messages, bybit.messages) {
        loops[which].push(message).await?;
        // Real time between inputs, so the app client coalesces some samples and sends others.
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    hub.end();
    let (live_schema, live_batches) = arrow.await??;
    let (books, trades, others) = read_until_ended(&mut app).await?;
    for live in loops {
        live.finish().await?;
    }
    for run in [RUN, BYBIT_RUN] {
        rotate(&dir.join("market").join(run))?;
    }
    server.stop().await;
    Ok(TwoVenues {
        dir,
        live: book_rows(&one_batch(&live_schema, &live_batches)?),
        books,
        trades,
        others,
    })
}

/// Two venues at once, as `serve` will run Binance and Bybit: each venue's live loop, through
/// its own tap, feeds one hub. The welcome lists both venues' symbols and runs; every live Arrow
/// row, book message and trade names its venue; each venue's live rows equal its own run's
/// rotated Parquet field for field, on one absolute grid, so the venues join exactly on
/// `sample_ts_ns`; the app's book messages for each venue are an in-order subsequence of that
/// venue's rows; the status holds each venue's own sockets and counters.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_venues_serve_exactly_what_each_stores() -> Outcome {
    let played = play_two_venues(scratch("api-two-venues")).await?;
    let mut grids = Vec::new();
    for (venue, run) in [("binance-usdm", RUN), ("bybit-linear", BYBIT_RUN)] {
        let (schema, batches) = read_parquet(&played.dir.join("market").join(run), "book_samples")?;
        let stored = book_rows(&one_batch(&schema, &batches)?);
        assert!(
            stored.iter().all(|row| row.0 == venue),
            "{run} holds only {venue}"
        );
        let live: Vec<BookRow> = played
            .live
            .iter()
            .filter(|row| row.0 == venue)
            .cloned()
            .collect();
        assert_eq!(live, stored, "{venue}: the live rows are its rotated table");
        assert!(stored.len() >= 30, "{venue}: {} rows", stored.len());
        let books: Vec<BookRow> = played
            .books
            .iter()
            .filter(|b| b.0 == venue)
            .cloned()
            .collect();
        assert_subsequence(&books, &stored, venue);
        let instants: BTreeSet<i64> = stored.iter().map(|row| row.2).collect();
        assert!(
            instants.iter().all(|ms| ms % 100 == 0),
            "{venue}: an absolute grid"
        );
        grids.push(instants);
    }
    let lo = grids
        .iter()
        .filter_map(|g| g.first())
        .copied()
        .max()
        .ok_or("instants")?;
    let hi = grids
        .iter()
        .filter_map(|g| g.last())
        .copied()
        .min()
        .ok_or("instants")?;
    let overlap = |grid: &BTreeSet<i64>| grid.range(lo..=hi).copied().collect::<Vec<i64>>();
    assert_eq!(
        overlap(&grids[0]),
        overlap(&grids[1]),
        "both venues are sampled at the same instants where they overlap"
    );

    // Only Binance's script trades; every trade says its venue and carries a string id.
    let (_, stored) = read_parquet(&played.dir.join("market").join(RUN), "trades")?;
    let stored: usize = stored.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(played.trades.len(), stored, "every trade reached the app");
    for trade in &played.trades {
        assert_eq!(trade["venue"], "binance-usdm", "{trade}");
        assert!(trade["id"].is_string(), "{trade}");
    }
    let status = played
        .others
        .iter()
        .rev()
        .find(|m| m["t"] == "status")
        .ok_or("a status")?;
    let bybit = &status["venues"]["bybit-linear"];
    assert_eq!(bybit["streams"]["depth"]["reconnects"], 1, "{status}");
    assert_eq!(bybit["resyncs"], 5, "{status}");
    assert!(
        status["venues"]["binance-usdm"]["resyncs"].as_u64() >= Some(1),
        "{status}"
    );
    fs::remove_dir_all(&played.dir)?;
    Ok(())
}

/// With three symbols an instant's rows straddle `DuckDB`'s chunks (2,048 rows end mid-instant).
/// A replay still sends whole instants (one per batch when paced, several when not) and every row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replays_send_whole_instants_across_duckdb_chunks() -> Outcome {
    const THREE: &str = "20261022T000000Z-three";
    let dir = scratch("api-instants");
    write_three_symbol_run(&dir.join("market").join(THREE))?;
    let server = Server::start(Hub::new(HubConfig::default()), ApiConfig::new(dir.clone())).await?;

    // A table read hands `DuckDB`'s chunks on as they come: some end mid-instant.
    let url = server.url(&format!("/v1/table/market/{THREE}/book_samples"));
    let (schema, raw) = get_arrow(&url).await?;
    let sizes: Vec<usize> = raw.iter().map(RecordBatch::num_rows).collect();
    assert!(
        sizes.iter().any(|rows| rows % 3 != 0),
        "no chunk straddles an instant: {sizes:?}"
    );
    let stored = one_batch(&schema, &raw)?;

    for speed in ["0", "100000"] {
        let url = server.url(&format!("/v1/live/book_samples?run={THREE}&speed={speed}"));
        let (schema, batches) = get_arrow(&url).await?;
        assert_eq!(
            one_batch(&schema, &batches)?,
            stored,
            "speed {speed}: every row"
        );
        let times = |batch: &RecordBatch| int64(batch, "sample_ts_ns").values().to_vec();
        for batch in &batches {
            let ts = times(batch);
            assert_eq!(
                ts.len() % 3,
                0,
                "speed {speed}: a batch with part of an instant"
            );
            if speed != "0" {
                assert!(
                    ts.iter().all(|t| *t == ts[0]),
                    "speed {speed}: one instant per batch"
                );
            }
        }
        for pair in batches.windows(2) {
            assert_ne!(
                times(&pair[0]).last(),
                times(&pair[1]).first(),
                "an instant split"
            );
        }
    }
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A run rotated before there were venues (a format-1 manifest without a venue, `book_samples`
/// without its `venue` column, trades in `agg_trades` with numeric ids) is served as it was stored:
/// its tables at major 1, never labelled with a contract they do not keep, and the app's replay of
/// it puts the run's venue (Binance's, for such runs) on every book and trade and runs to its end.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_rotated_before_venues_is_served_as_it_was_stored() -> Outcome {
    const OLD: &str = "20261021T000000Z-before-venues";
    let dir = scratch("api-before-venues");
    write_pre_venue_run(&dir.join("market").join(OLD))?;
    let hub = Hub::new(HubConfig::default());
    let feed = engine::market::feed::Feed::new(Venue::BinanceUsdm, BTreeMap::new());
    let mut tap = engine::api::hub::LiveTap::new(Arc::clone(&hub));
    engine::market::live::Observer::started(&mut tap, &feed, None);
    let server = Server::start(Arc::clone(&hub), ApiConfig::new(dir.clone())).await?;

    let url = server.url(&format!("/v1/table/market/{OLD}/book_samples"));
    let (schema, batches) = get_arrow(&url).await?;
    assert_eq!(schema.metadata()["tp.schema"], "book_samples/1");
    assert!(schema.field_with_name("venue").is_err(), "{schema:?}");
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 30);
    let url = server.url(&format!("/v1/table/market/{OLD}/agg_trades"));
    let (schema, _) = get_arrow(&url).await?;
    assert_eq!(schema.metadata()["tp.schema"], "agg_trades/1");

    let mut viewer = ws_connect(server.addr).await?;
    hello(&mut viewer).await?;
    let sub = json!({"t": "sub", "topics": ["book", "trades"]});
    send_json(&mut viewer, &sub).await?;
    let _ack = next_json(&mut viewer, WAIT).await?;
    send_json(&mut viewer, &json!({"t": "replay", "run": OLD, "speed": 0})).await?;
    until_replay(&mut viewer).await?;
    let (mut books, mut trades) = (0, Vec::new());
    let outcome = loop {
        let message = next_json(&mut viewer, WAIT).await?;
        match message["t"].as_str() {
            Some("book") => {
                assert_eq!(message["venue"], "binance-usdm", "{message}");
                books += 1;
            }
            Some("trades") => {
                trades.extend(message["trades"].as_array().cloned().unwrap_or_default());
            }
            Some("source") => break message,
            _ => {}
        }
    };
    assert_eq!(outcome["mode"], "ended", "{outcome}");
    assert!(books > 0, "the replay sent books");
    let ids: Vec<&str> = trades.iter().filter_map(|t| t["id"].as_str()).collect();
    assert_eq!(ids, ["1", "2", "3"], "numeric ids as strings");
    assert!(
        trades.iter().all(|t| t["venue"] == "binance-usdm"),
        "{trades:?}"
    );
    server.stop().await;
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A market run as rotate wrote it before there were venues: a format-1 manifest without a venue,
/// 30 instants of `book_samples` without the `venue` column, three trades in `agg_trades`.
fn write_pre_venue_run(run_dir: &Path) -> Outcome {
    use arrow::array::{ArrayRef, BooleanArray, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use engine::market::sync::Level;
    use engine::store::samples::{BookRow as Row, Sample, SampleRows};

    const T0: i64 = 1_792_540_800_000_000_000;
    fs::create_dir_all(run_dir.join("book_samples"))?;
    fs::create_dir_all(run_dir.join("agg_trades"))?;
    let manifest = json!({
        "format": 1, "domain": "market", "run_id": run_dir.file_name().and_then(|n| n.to_str()),
        "producer": {"name": "engine", "version": "0.1.0", "git_sha": null, "dirty": null},
        "started_ns": T0, "key": "seq",
        "tables": ["frames", "depth_updates", "agg_trades", "snapshots", "book_samples"],
        "params": {
            "symbols": ["BTCUSDT"],
            "scales": {"BTCUSDT": {"price_decimals": 1, "qty_decimals": 3}},
            "streams": ["depth@100ms", "aggTrade"],
            "book_sample": {"levels": 2, "every_ns": 100_000_000},
        },
    });
    fs::write(run_dir.join("manifest.json"), manifest.to_string())?;

    let mut rows = SampleRows::new(2);
    for i in 0..30_i64 {
        rows.push(&Sample {
            sample_ts_ns: T0 + i * 100_000_000,
            seq: i,
            venue: Venue::BinanceUsdm,
            rows: vec![Row {
                symbol: "BTCUSDT".to_owned(),
                synced: true,
                last_update_id: Some(i),
                bids: vec![Level {
                    price: 1000,
                    qty: 1 + i,
                }],
                asks: vec![Level {
                    price: 1001,
                    qty: 2,
                }],
            }],
        });
    }
    let samples = rows.take()?;
    // Every column but the venue, as rotate wrote them then.
    let kept: Vec<usize> = (0..samples.num_columns())
        .filter(|&i| samples.schema().field(i).name() != "venue")
        .collect();
    write_parquet(
        &run_dir
            .join("book_samples")
            .join(format!("{:020}.parquet", 0)),
        &samples.project(&kept)?,
    )?;

    let int64 = |name: &str, nullable: bool| Field::new(name, DataType::Int64, nullable);
    let schema = Arc::new(Schema::new(vec![
        int64("seq", false),
        int64("source_ts_ns", false),
        int64("trade_ts_ns", false),
        int64("capture_ts_ns", false),
        Field::new("symbol", DataType::Utf8, false),
        int64("agg_trade_id", false),
        int64("first_trade_id", true),
        int64("last_trade_id", true),
        int64("price", false),
        int64("qty", false),
        Field::new("buyer_is_maker", DataType::Boolean, false),
    ]));
    let times: Vec<i64> = (1..=3).map(|k| T0 + k * 700_000_000).collect();
    let column = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
    let trades = RecordBatch::try_new(
        schema,
        vec![
            column(vec![40, 41, 42]),
            column(times.clone()),
            column(times.clone()),
            column(times),
            Arc::new(StringArray::from(vec!["BTCUSDT"; 3])),
            column(vec![1, 2, 3]),
            column(vec![10, 11, 13]),
            column(vec![10, 12, 13]),
            column(vec![1000, 1001, 1000]),
            column(vec![5, 6, 7]),
            Arc::new(BooleanArray::from(vec![false, true, false])),
        ],
    )?;
    write_parquet(
        &run_dir
            .join("agg_trades")
            .join(format!("{:020}.parquet", 0)),
        &trades,
    )
}

fn write_parquet(path: &Path, batch: &RecordBatch) -> Outcome {
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(fs::File::create(path)?, batch.schema(), None)?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

/// A market run with three symbols and 1,500 instants of book samples (4,500 rows, one row group).
fn write_three_symbol_run(run_dir: &Path) -> Outcome {
    use engine::market::sync::Level;
    use engine::store::samples::{BookRow as Row, Sample, SampleRows};

    let symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"];
    let scales: serde_json::Map<String, Value> = symbols
        .iter()
        .map(|s| {
            (
                (*s).to_owned(),
                json!({"price_decimals": 1, "qty_decimals": 3}),
            )
        })
        .collect();
    fs::create_dir_all(run_dir.join("book_samples"))?;
    let manifest = json!({
        "format": 1, "domain": "market", "run_id": run_dir.file_name().and_then(|n| n.to_str()),
        "key": "seq", "tables": ["book_samples"], "params": {"scales": scales},
    });
    fs::write(run_dir.join("manifest.json"), manifest.to_string())?;
    let mut rows = SampleRows::new(2);
    for i in 0..1500_i64 {
        let sample = Sample {
            sample_ts_ns: 1_700_000_000_000_000_000 + i * 100_000_000,
            seq: i,
            venue: Venue::BinanceUsdm,
            rows: symbols
                .iter()
                .map(|symbol| Row {
                    symbol: (*symbol).to_owned(),
                    synced: true,
                    last_update_id: Some(i),
                    bids: vec![Level {
                        price: 1000 - i % 7,
                        qty: 1 + i % 5,
                    }],
                    asks: vec![Level {
                        price: 1001 + i % 3,
                        qty: 2,
                    }],
                })
                .collect(),
        };
        rows.push(&sample);
    }
    let batch = rows.take()?;
    let file = fs::File::create(
        run_dir
            .join("book_samples")
            .join(format!("{:020}.parquet", 0)),
    )?;
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}
