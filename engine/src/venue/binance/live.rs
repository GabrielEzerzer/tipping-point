//! Binance's live loop: two combined-stream sockets (depth on `/public`, aggregated trades on
//! `/market`) and a REST client around one [`Pipeline`], recording every input it applies to a
//! run's log when asked to. Every input gets the seq its log gives it, recorded or not, and an
//! [`Observer`] sees each one, numbered, right before it is applied.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::{Instant, interval, sleep, sleep_until};

use super::rest::{Rest, RestError};
use crate::market::feed::Feed;
use crate::market::live::{LiveEvent, Log, Observer, Stream, handle, install_crypto_provider};
use crate::market::pipeline::{Message, Pipeline};
use crate::market::ws;
use crate::market::{Scales, Symbol};
use crate::store::frame::Meta;
use crate::store::run::{self, BookSample, Clock, DataSource, MarketParams};
use crate::store::writer::{LogConfig, LogWriter};
use crate::store::{LogError, now_ns};
use crate::venue::Venue;

/// The depth stream the loop subscribes to.
pub const DEPTH_STREAM: &str = "depth@100ms";
/// The trade stream the loop subscribes to.
pub const TRADE_STREAM: &str = "aggTrade";

/// Routed path that serves `@depth@100ms`.
pub const DEPTH_HOST: &str = "wss://fstream.binance.com/public/stream";
/// Routed path that serves `@aggTrade` (it is not delivered on `/public`).
pub const TRADE_HOST: &str = "wss://fstream.binance.com/market/stream";

/// Binance closes a connection after 24 h; leaving before that keeps the reconnect on our terms.
pub const DEFAULT_MAX_CONNECTION_AGE: Duration = Duration::from_hours(23);

/// Depth updates in a row a symbol must fail to convert before the loop asks for its scales
/// again: one bad frame is no changed tick size, but a finer one fails every update (#45).
const RECHECK_SCALES_AFTER: u32 = 10;

/// The least time between two such asks for one symbol, which stays down meanwhile: exchange
/// info costs REST weight, and Binance announces tick changes days ahead.
const RECHECK_SCALES_EVERY: Duration = Duration::from_mins(10);

/// Combined-stream URL for the given symbols and stream suffix, e.g. `depth@100ms`.
#[must_use]
pub fn stream_url(host: &str, symbols: &[Symbol], suffix: &str) -> String {
    let streams: Vec<String> = symbols
        .iter()
        .map(|s| format!("{}@{suffix}", s.to_lowercase()))
        .collect();
    format!("{host}?streams={}", streams.join("/"))
}

/// Knobs for a live run.
#[derive(Debug, Clone)]
pub struct LiveConfig {
    pub symbols: Vec<Symbol>,
    /// Stop after this long; `None` runs until Ctrl-C or until the task is dropped.
    pub duration: Option<Duration>,
    /// How often a [`LiveEvent::Report`] fires.
    pub report_every: Duration,
    /// Reader-to-loop channel depth per connection; full means frames drop and books resync.
    pub channel_capacity: usize,
    /// Floor between two snapshot fetches for one symbol, to respect REST weight on a flapping
    /// stream. The first fetch after a gap is still immediate.
    pub min_snapshot_interval: Duration,
    pub max_connection_age: Duration,
    /// Record every input to a new run, or `None` to record nothing.
    pub record: Option<Recording>,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            symbols: vec!["BTCUSDT".to_owned(), "ETHUSDT".to_owned()],
            duration: None,
            report_every: Duration::from_secs(1),
            channel_capacity: 1024,
            min_snapshot_interval: Duration::from_secs(2),
            max_connection_age: DEFAULT_MAX_CONNECTION_AGE,
            record: None,
        }
    }
}

/// Where and how a live run records.
#[derive(Debug, Clone)]
pub struct Recording {
    /// The run folder to create, normally `<data dir>/market/<run_id>/`; it must not exist.
    pub run_dir: PathBuf,
    pub log: LogConfig,
}

/// A finished snapshot fetch: the symbol, when its reply arrived, and the body or the error.
type Fetched = (Symbol, i64, Result<String, RestError>);

/// A finished fetch of the scales of the symbols due a check, all from one exchange-info reply.
type FetchedScales = (Vec<Symbol>, Result<BTreeMap<Symbol, Scales>, RestError>);

/// Why the loop could not start or record.
#[derive(Debug, Error)]
pub enum LiveError {
    #[error(transparent)]
    Rest(#[from] RestError),
    #[error("recording: {0}")]
    Log(#[from] LogError),
    #[error("recording to {}: {source}", .path.display())]
    Run { path: PathBuf, source: io::Error },
}

/// Runs the pipeline live until `config.duration` elapses or Ctrl-C, telling `observer` as it
/// goes. When recording, the run folder gets its manifest at the start and `complete.json`
/// after a clean end.
///
/// # Errors
/// Returns an error when the REST client cannot be built, the scales cannot be fetched or the
/// venue does not list a symbol, or the recording cannot be created or written.
pub async fn run_live(config: LiveConfig, mut observer: impl Observer) -> Result<(), LiveError> {
    install_crypto_provider();
    let rest = Rest::new()?;
    let scales = starting_scales(&rest, &config.symbols).await?;
    let venue = Venue::BinanceUsdm;
    let mut log = match &config.record {
        Some(recording) => {
            Log::recorded(start_recording(recording, venue, &config.symbols, &scales)?)
        }
        None => Log::unrecorded(),
    };
    let mut pipeline = Pipeline::new(Feed::new(venue, scales));
    let run_id = config
        .record
        .as_ref()
        .and_then(|recording| recording.run_dir.file_name())
        .and_then(|name| name.to_str());
    observer.started(pipeline.feed(), run_id);

    let (depth_tx, mut depth_rx) = mpsc::channel(config.channel_capacity);
    let (trade_tx, mut trade_rx) = mpsc::channel(config.channel_capacity);
    let depth_url = stream_url(DEPTH_HOST, &config.symbols, DEPTH_STREAM);
    let trade_url = stream_url(TRADE_HOST, &config.symbols, TRADE_STREAM);
    let depth_reader = tokio::spawn(ws::run_reader(
        depth_url,
        config.max_connection_age,
        depth_tx,
    ));
    let trade_reader = tokio::spawn(ws::run_reader(
        trade_url,
        config.max_connection_age,
        trade_tx,
    ));

    let (snapshot_tx, mut snapshot_rx) = mpsc::channel::<Fetched>(16);
    let mut next_fetch_allowed: BTreeMap<Symbol, Instant> = BTreeMap::new();
    let (scales_tx, mut scales_rx) = mpsc::channel::<FetchedScales>(4);
    let mut scales_checked: BTreeMap<Symbol, Instant> = BTreeMap::new();

    let start = Instant::now();
    // tokio caps a very long sleep rather than overflowing, so "forever" is just a long sleep.
    let stop = sleep(
        config
            .duration
            .unwrap_or(Duration::from_secs(u64::from(u32::MAX))),
    );
    tokio::pin!(stop);
    // A ctrl_c future that fails to register resolves at once; it is then disarmed, never
    // polled again, and only the duration stops the loop.
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let mut ctrl_c_armed = true;
    let mut report = interval(config.report_every);
    report.tick().await; // the first tick fires immediately; skip it

    let ended: Result<(), LiveError> = loop {
        let message = tokio::select! {
            Some(inbound) = depth_rx.recv() => Message::Socket { stream: Stream::Depth, inbound },
            Some(inbound) = trade_rx.recv() => Message::Socket { stream: Stream::Trades, inbound },
            Some((symbol, capture_ns, result)) = snapshot_rx.recv() => Message::Snapshot {
                symbol,
                capture_ns,
                result: result.map_err(|error| error.to_string()),
            },
            Some((symbols, fetched)) = scales_rx.recv() => {
                say_what_scales_show(pipeline.feed(), &symbols, &fetched, &mut observer);
                continue;
            }
            _ = report.tick() => {
                observer.event(LiveEvent::Report { elapsed: start.elapsed(), feed: pipeline.feed(), stats: pipeline.stats() });
                if let Err(error) = log.tick().await {
                    break Err(error.into());
                }
                continue;
            }
            () = &mut stop => break Ok(()),
            pressed = &mut ctrl_c, if ctrl_c_armed => {
                ctrl_c_armed = false;
                if pressed.is_ok() {
                    break Ok(());
                }
                continue;
            }
        };
        if let Err(error) = handle(message, &mut pipeline, &mut log, &mut observer).await {
            break Err(error.into());
        }
        fetch_snapshots(
            &mut pipeline,
            &rest,
            &snapshot_tx,
            &mut next_fetch_allowed,
            &config,
        );
        fetch_scales(pipeline.feed(), &rest, &scales_tx, &mut scales_checked);
    };

    depth_reader.abort();
    trade_reader.abort();
    let recorded = match (log, &config.record) {
        (Log::Recorded(log), Some(recording)) => {
            finish_recording(*log, &recording.run_dir, ended.is_ok()).await
        }
        _ => Ok(()),
    };
    // The writer's own error explains a loop that stopped because the writer did.
    recorded.and(ended)
}

/// Starts a fetch of every snapshot the feed now needs, each symbol's at least the config's
/// `min_snapshot_interval` after its last; the replies come back on `tx`.
fn fetch_snapshots(
    pipeline: &mut Pipeline,
    rest: &Rest,
    tx: &mpsc::Sender<Fetched>,
    next_allowed: &mut BTreeMap<Symbol, Instant>,
    config: &LiveConfig,
) {
    for symbol in pipeline.feed_mut().snapshots_due() {
        let at = fetch_at(
            next_allowed,
            &symbol,
            Instant::now(),
            config.min_snapshot_interval,
        );
        pipeline.stats_mut().snapshot_fetches += 1;
        let rest = rest.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            sleep_until(at).await;
            let result = rest.fetch_snapshot(&symbol).await;
            let _ = tx.send((symbol, now_ns(), result)).await;
        });
    }
}

/// When the snapshot fetch for `symbol` asked for at `now` starts: at once, or `floor` after the
/// start of the symbol's last fetch, whichever is later. The floor alone spaces a symbol's
/// refetches out: a failed fetch leaves the symbol due again in the same loop pass, and a
/// 1000-level snapshot costs 20 of Binance's 2,400 REST weight a minute, past which come a 429
/// and then an IP ban (#73). After a quiet spell the next fetch is immediate, never a burst owed
/// for the gap.
fn fetch_at(
    next: &mut BTreeMap<Symbol, Instant>,
    symbol: &str,
    now: Instant,
    floor: Duration,
) -> Instant {
    let at = next.get(symbol).copied().unwrap_or(now).max(now);
    next.insert(symbol.to_owned(), at + floor);
    at
}

/// The scales of every one of `symbols`, to start a feed at: refused when the venue does not list
/// one, as a feed starts with every symbol it was asked for or not at all (#104).
async fn starting_scales(
    rest: &Rest,
    symbols: &[Symbol],
) -> Result<BTreeMap<Symbol, Scales>, RestError> {
    let scales = rest.fetch_scales(symbols).await?;
    match symbols.iter().find(|symbol| !scales.contains_key(*symbol)) {
        Some(missing) => Err(RestError::MissingSymbol(missing.clone())),
        None => Ok(scales),
    }
}

/// Raises, for each of `symbols`, [`LiveEvent::ScalesChanged`] when the fetch found other scales
/// than its feed reads it at, or [`LiveEvent::ScalesCheckFailed`] when the fetch failed or did not
/// list the symbol, so a check that could not be made never reads as unchanged scales (#100), and
/// a symbol the venue dropped fails its own check alone (#104). Said, not applied: a run keeps one
/// set of scales (#45).
fn say_what_scales_show(
    feed: &Feed,
    symbols: &[Symbol],
    fetched: &Result<BTreeMap<Symbol, Scales>, RestError>,
    observer: &mut impl Observer,
) {
    for symbol in symbols {
        let found = match fetched {
            Ok(scales) => scales
                .get(symbol)
                .copied()
                .ok_or_else(|| RestError::MissingSymbol(symbol.clone()).to_string()),
            Err(error) => Err(error.to_string()),
        };
        match (found, feed.scales(symbol)) {
            (Err(error), _) => {
                observer.event(LiveEvent::ScalesCheckFailed {
                    symbol,
                    error: &error,
                });
            }
            (Ok(now), Some(was)) if now != was => {
                observer.event(LiveEvent::ScalesChanged { symbol, was, now });
            }
            (Ok(_), _) => {}
        }
    }
}

/// Starts one fetch of the scales of every symbol [`scales_due`] names, however many: exchange
/// info is one multi-MB download that holds every symbol, so the symbols due together share it
/// (#100). The reply comes back on `tx`.
fn fetch_scales(
    feed: &Feed,
    rest: &Rest,
    tx: &mpsc::Sender<FetchedScales>,
    checked: &mut BTreeMap<Symbol, Instant>,
) {
    let due = scales_due(feed, checked, Instant::now());
    if due.is_empty() {
        return;
    }
    let rest = rest.clone();
    let tx = tx.clone();
    tokio::spawn(async move {
        let fetched = rest.fetch_scales(&due).await;
        let _ = tx.send((due, fetched)).await;
    });
}

/// The symbols whose scales to ask the venue for again now (#45): those whose last
/// [`RECHECK_SCALES_AFTER`] depth updates did not convert, each at most once every
/// [`RECHECK_SCALES_EVERY`]. Marks them checked at `now`.
fn scales_due(feed: &Feed, checked: &mut BTreeMap<Symbol, Instant>, now: Instant) -> Vec<Symbol> {
    let due: Vec<Symbol> = feed
        .symbols()
        .filter(|symbol| feed.unconverted_in_a_row(symbol) >= RECHECK_SCALES_AFTER)
        .filter(|symbol| {
            checked
                .get(*symbol)
                .is_none_or(|at| now >= *at + RECHECK_SCALES_EVERY)
        })
        .cloned()
        .collect();
    for symbol in &due {
        checked.insert(symbol.clone(), now);
    }
    due
}

/// Creates the run folder and its manifest for a feed of `venue`'s `symbols` at `scales`, then
/// starts the log writer. Every recorded feed starts here, this loop's and `serve`'s scripted
/// one alike, so a scripted run is recorded exactly as a live one (#75).
///
/// # Errors
/// Returns an error when the run folder or its manifest cannot be created (the folder must not
/// exist yet), or the log writer cannot start.
pub fn start_recording(
    recording: &Recording,
    venue: Venue,
    symbols: &[Symbol],
    scales: &BTreeMap<Symbol, Scales>,
) -> Result<LogWriter, LiveError> {
    let params = MarketParams {
        venue,
        symbols: symbols.to_vec(),
        scales: scales.clone(),
        streams: venue.streams().iter().map(|&s| s.to_owned()).collect(),
        book_sample: BookSample::default(),
        clock: Clock::Capture,
        source: DataSource::Live,
    };
    let manifest = run::create(&recording.run_dir, params).map_err(|source| LiveError::Run {
        path: recording.run_dir.clone(),
        source,
    })?;
    let log = LogWriter::start(
        &recording.run_dir,
        Meta::for_run(&manifest),
        recording.log.clone(),
    )?;
    Ok(log)
}

/// Closes the log, then writes `complete.json` if the feed itself ended cleanly: a run cut short
/// by an error stays incomplete even though its log closed properly. Every recorded feed ends
/// here, as it starts in [`start_recording`].
///
/// # Errors
/// Returns the error that stopped the writer, or the I/O error from writing `complete.json`.
pub async fn finish_recording(
    log: LogWriter,
    run_dir: &Path,
    clean: bool,
) -> Result<(), LiveError> {
    let summary = tokio::task::spawn_blocking(move || log.finish())
        .await
        .map_err(|_| LogError::WriterPanicked)??;
    if !clean {
        return Ok(());
    }
    run::complete(run_dir, &summary).map_err(|source| LiveError::Run {
        path: run_dir.to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A tick size made finer fails every depth update of its symbol: after ten in a row the loop
    /// asks for the symbol's scales again, then not before ten minutes have passed (#45).
    #[tokio::test]
    async fn scales_are_asked_for_again_once_a_symbols_updates_stop_converting() {
        let scales = Scales {
            price_decimals: 1,
            qty_decimals: 3,
        };
        let mut feed = Feed::new(
            Venue::BinanceUsdm,
            BTreeMap::from([("BTCUSDT".to_owned(), scales)]),
        );
        let mut finer_than_the_tick = (1..).map(|u: u64| {
            format!(
                r#"{{"stream":"btcusdt@depth@100ms","data":{{"e":"depthUpdate","E":1,"T":1,"s":"BTCUSDT","U":{u},"u":{u},"pu":{},"b":[["100.05","1.000"]],"a":[]}}}}"#,
                u - 1
            )
        });
        let (mut checked, start) = (BTreeMap::new(), Instant::now());
        let btc = vec!["BTCUSDT".to_owned()];
        for text in finer_than_the_tick.by_ref().take(9) {
            assert!(feed.on_text(&text).is_err());
        }
        assert_eq!(scales_due(&feed, &mut checked, start), Vec::<Symbol>::new());
        let tenth = finer_than_the_tick.next().unwrap_or_default();
        assert!(feed.on_text(&tenth).is_err());
        assert_eq!(scales_due(&feed, &mut checked, start), btc);
        let later = start + Duration::from_mins(1);
        assert_eq!(scales_due(&feed, &mut checked, later), Vec::<Symbol>::new());
        assert_eq!(
            scales_due(&feed, &mut checked, start + RECHECK_SCALES_EVERY),
            btc
        );
    }

    /// Exchange info for a stand-in venue: BTCUSDT's tick made finer (1 decimal to 2), ETHUSDT's
    /// as it was.
    const EXCHANGE_INFO: &str = r#"{"symbols":[
        {"symbol":"BTCUSDT","filters":[{"filterType":"PRICE_FILTER","tickSize":"0.01"},
            {"filterType":"LOT_SIZE","stepSize":"0.001"}]},
        {"symbol":"ETHUSDT","filters":[{"filterType":"PRICE_FILTER","tickSize":"0.01"},
            {"filterType":"LOT_SIZE","stepSize":"0.001"}]}]}"#;

    /// Serves [`EXCHANGE_INFO`] on 127.0.0.1, counting the downloads, after a 429 to the first.
    /// Its base URL.
    async fn exchange_info_stand_in(downloads: Arc<AtomicUsize>) -> std::io::Result<String> {
        use axum::http::StatusCode;

        let app = axum::Router::new().route(
            "/fapi/v1/exchangeInfo",
            axum::routing::get(move || {
                let first = downloads.fetch_add(1, Ordering::SeqCst) == 0;
                async move {
                    if first {
                        (StatusCode::TOO_MANY_REQUESTS, "Too many requests")
                    } else {
                        (StatusCode::OK, EXCHANGE_INFO)
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let base = format!("http://{}", listener.local_addr()?);
        tokio::spawn(async move { axum::serve(listener, app).await });
        Ok(base)
    }

    /// Two symbols due a check of their scales at once share one exchange-info download, where
    /// each had its own; and a check the venue fails is said for each, where it was dropped and
    /// read as "the scales did not change" (#100).
    #[tokio::test]
    async fn due_symbols_share_one_scales_download_and_a_failed_check_is_said()
    -> Result<(), Box<dyn std::error::Error>> {
        install_crypto_provider();
        let downloads = Arc::new(AtomicUsize::new(0));
        let rest = Rest::at(&exchange_info_stand_in(Arc::clone(&downloads)).await?)?;
        let scale = |price_decimals| Scales {
            price_decimals,
            qty_decimals: 3,
        };
        let mut feed = Feed::new(
            Venue::BinanceUsdm,
            BTreeMap::from([
                ("BTCUSDT".to_owned(), scale(1)),
                ("ETHUSDT".to_owned(), scale(2)),
            ]),
        );
        // Each symbol's updates stop converting: prices finer than its tick.
        for u in 1..=u64::from(RECHECK_SCALES_AFTER) {
            for (symbol, price) in [("BTCUSDT", "100.05"), ("ETHUSDT", "100.005")] {
                let text = format!(
                    r#"{{"stream":"{}@depth@100ms","data":{{"e":"depthUpdate","E":1,"T":1,"s":"{symbol}","U":{u},"u":{u},"pu":{},"b":[["{price}","1.000"]],"a":[]}}}}"#,
                    symbol.to_lowercase(),
                    u - 1
                );
                assert!(feed.on_text(&text).is_err());
            }
        }
        let (tx, mut rx) = mpsc::channel(4);
        let (mut checked, mut said) = (BTreeMap::new(), Vec::new());
        for round in 1..=2 {
            fetch_scales(&feed, &rest, &tx, &mut checked);
            let (symbols, fetched) = rx.recv().await.ok_or("a reply")?;
            assert_eq!(symbols, ["BTCUSDT", "ETHUSDT"], "round {round}");
            assert_eq!(
                downloads.load(Ordering::SeqCst),
                round,
                "one download a round"
            );
            let mut observer = |event: LiveEvent<'_>| said.push(format!("{event:?}"));
            say_what_scales_show(&feed, &symbols, &fetched, &mut observer);
            checked.clear();
        }
        assert_eq!(said.len(), 3, "{said:#?}");
        for (event, symbol) in said.iter().zip(["BTCUSDT", "ETHUSDT"]) {
            assert!(
                event.starts_with(&format!("ScalesCheckFailed {{ symbol: \"{symbol}\""))
                    && event.contains("429"),
                "{event}"
            );
        }
        assert!(
            said[2].starts_with("ScalesChanged { symbol: \"BTCUSDT\""),
            "only BTCUSDT changed: {}",
            said[2]
        );
        Ok(())
    }

    /// A symbol the venue no longer lists fails only its own scales check: another symbol due a
    /// check with it still hears that its tick changed, where both were told the exchange info had
    /// no such symbol, every ten minutes (#104). A feed asked for that symbol is still refused at
    /// the start, and one asked for listed symbols is not.
    #[tokio::test]
    async fn a_symbol_the_venue_no_longer_lists_fails_only_its_own_scales_check()
    -> Result<(), Box<dyn std::error::Error>> {
        install_crypto_provider();
        // Counted from one, so the stand-in refuses no download.
        let rest = Rest::at(&exchange_info_stand_in(Arc::new(AtomicUsize::new(1))).await?)?;
        let scale = |price_decimals| Scales {
            price_decimals,
            qty_decimals: 3,
        };
        let feed = Feed::new(
            Venue::BinanceUsdm,
            BTreeMap::from([
                ("BTCUSDT".to_owned(), scale(1)),
                ("SOLUSDT".to_owned(), scale(2)),
            ]),
        );
        let symbols = ["BTCUSDT".to_owned(), "SOLUSDT".to_owned()];
        let fetched = rest.fetch_scales(&symbols).await;
        let mut said = Vec::new();
        let mut observer = |event: LiveEvent<'_>| said.push(format!("{event:?}"));
        say_what_scales_show(&feed, &symbols, &fetched, &mut observer);
        assert_eq!(said.len(), 2, "{said:#?}");
        assert!(
            said[0].starts_with("ScalesChanged { symbol: \"BTCUSDT\""),
            "{said:#?}"
        );
        assert!(
            said[1].starts_with("ScalesCheckFailed { symbol: \"SOLUSDT\"")
                && said[1].contains("has no symbol"),
            "{said:#?}"
        );
        let found: Option<Vec<&Symbol>> = fetched.as_ref().ok().map(|s| s.keys().collect());
        assert_eq!(found, Some(vec![&symbols[0]]), "the symbols it found");

        let refused = starting_scales(&rest, &symbols).await;
        assert!(
            matches!(&refused, Err(RestError::MissingSymbol(s)) if s == "SOLUSDT"),
            "{refused:?}"
        );
        let listed = ["BTCUSDT".to_owned(), "ETHUSDT".to_owned()];
        assert_eq!(starting_scales(&rest, &listed).await?.len(), 2);
        Ok(())
    }

    /// A symbol whose snapshots keep failing is refetched a floor apart, never at round-trip
    /// speed, and another symbol is not held up by it (#73).
    #[test]
    fn a_symbols_snapshot_fetches_are_a_floor_apart() {
        let floor = LiveConfig::default().min_snapshot_interval;
        assert_eq!(floor, Duration::from_secs(2), "the default floor");
        let (mut next, now) = (BTreeMap::new(), Instant::now());
        let mut fetch = |symbol: &str, now: Instant| fetch_at(&mut next, symbol, now, floor);
        assert_eq!(fetch("BTCUSDT", now), now, "the first fetch is immediate");
        assert_eq!(fetch("BTCUSDT", now), now + floor);
        assert_eq!(fetch("BTCUSDT", now), now + 2 * floor);
        assert_eq!(fetch("ETHUSDT", now), now, "another symbol is not held up");
        // A quiet spell owes nothing: the next fetch is immediate and the floor counts from it.
        let later = now + 10 * floor;
        assert_eq!(fetch("BTCUSDT", later), later);
        assert_eq!(fetch("BTCUSDT", later), later + floor);
    }

    /// A feed that ended cleanly marks its run complete; one cut short by an error leaves it
    /// incomplete, though its log closes properly either way (#73).
    #[tokio::test]
    async fn only_a_feed_that_ended_cleanly_completes_its_run() -> Result<(), LiveError> {
        let dir = std::env::temp_dir().join(format!("tp-finish-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let symbols = vec!["BTCUSDT".to_owned()];
        let scales = BTreeMap::from([(
            symbols[0].clone(),
            Scales {
                price_decimals: 1,
                qty_decimals: 3,
            },
        )]);
        for (clean, name) in [(true, "clean"), (false, "cut")] {
            let recording = Recording {
                run_dir: dir.join(name),
                log: LogConfig::default(),
            };
            let log = start_recording(&recording, Venue::BinanceUsdm, &symbols, &scales)?;
            finish_recording(log, &recording.run_dir, clean).await?;
            let complete = recording.run_dir.join(run::COMPLETE).is_file();
            assert_eq!(complete, clean, "the {name} run");
            assert!(recording.run_dir.join(run::MANIFEST).is_file());
        }
        let _ = std::fs::remove_dir_all(&dir);
        Ok(())
    }

    #[test]
    fn builds_combined_stream_urls() {
        let symbols = vec!["BTCUSDT".to_owned(), "ETHUSDT".to_owned()];
        assert_eq!(
            stream_url(DEPTH_HOST, &symbols, "depth@100ms"),
            "wss://fstream.binance.com/public/stream?streams=btcusdt@depth@100ms/ethusdt@depth@100ms"
        );
        assert_eq!(
            stream_url(TRADE_HOST, &symbols[..1], "aggTrade"),
            "wss://fstream.binance.com/market/stream?streams=btcusdt@aggTrade"
        );
    }
}
