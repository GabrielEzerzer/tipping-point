//! Archive import (`store import`): the committed synthetic archives, one in each venue's format
//! (`engine/tests/fixtures/archives/`, made by `make_archive_fixtures.py`), through the import into
//! Parquet. Every count is read back with `DuckDB` and checked against what this test reads from
//! the archive itself, with its own zip reader: a frame per record, a `depth_updates` row per
//! delta level, a `snapshots` row per snapshot level, a trade per CSV row, a book sample per
//! 100 ms instant of the exchange clock. The manifest names the venue, the symbol, the clock and
//! the archive with its SHA-256; `complete.json` holds the integrity checks. An import of a done
//! run changes nothing; another archive or other tables for the same run are refused. The same
//! records fed as a live stream through the venue's adapter raise the same events with the same
//! books, input for input. Real archive days, kept locally in the gitignored
//! `harness/fixtures/archives/`, are imported too when present.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use engine::market::feed::Feed;
use engine::market::live::{LiveEvent, Log, Stream, handle};
use engine::market::pipeline::{Message, Pipeline};
use engine::market::ws::Inbound;
use engine::store::import::{ImportConfig, ImportError, Imported, import};
use engine::store::query::connect;
use engine::store::rotate::TABLES;
use engine::store::run::{self, Clock, Complete, DataSource, Manifest};
use engine::venue::Venue;
use engine::venue::archive::ArchiveName;
use engine_harness::api::repo;
use engine_harness::transcript::Transcript;
use serde_json::Value;
use sha2::{Digest, Sha256};

type Outcome = Result<(), Box<dyn Error>>;

const BYBIT: &str = "2026-10-20_BTCUSDT_ob200.data.zip";
const BINANCE: &str = "BTCUSDT-aggTrades-2026-10-20.zip";
const GRID_NS: i64 = 100_000_000;

fn archive(name: &str) -> PathBuf {
    repo().join("engine/tests/fixtures/archives").join(name)
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch dir");
    dir
}

fn config(data_dir: &Path, tables: Option<&[&str]>) -> ImportConfig {
    ImportConfig {
        data_dir: data_dir.to_owned(),
        url: None,
        tables: tables.map(|tables| tables.iter().map(|&t| t.to_owned()).collect()),
    }
}

/// Every line of an archive's one file, as this test reads it: its own zip reader, line endings
/// off, blank lines skipped.
fn lines(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let mut zip = zip::ZipArchive::new(File::open(path)?)?;
    let member = zip.by_index(0)?;
    let mut lines = Vec::new();
    for line in BufReader::new(member).lines() {
        let line = line?;
        let line = line.trim_end_matches('\r');
        if !line.is_empty() {
            lines.push(line.to_owned());
        }
    }
    Ok(lines)
}

fn sha256_hex(path: &Path) -> Result<String, Box<dyn Error>> {
    use std::fmt::Write as _;
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .fold(String::new(), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        }))
}

/// A `trades` row as the check reads it: trade id, first and last trade ids, price, quantity,
/// trade, capture and source times, buyer is maker, venue, symbol.
type TradeRow = (
    String,
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    bool,
    String,
    String,
);

/// `count(*)` of a run's table through `DuckDB`.
fn count(run_dir: &Path, table: &str) -> Result<u64, Box<dyn Error>> {
    let glob = run_dir.join(table).to_string_lossy().replace('\\', "/");
    let conn = connect()?;
    let rows: i64 = conn.query_row(
        &format!("SELECT count(*) FROM read_parquet('{glob}/*.parquet')"),
        [],
        |row| row.get(0),
    )?;
    Ok(u64::try_from(rows)?)
}

/// The grid instants `[ceil(first), last)` on the 100 ms grid: what the sampler samples between
/// a run's first input and its last.
fn instants(first_ns: i64, last_ns: i64) -> u64 {
    let start =
        first_ns.div_euclid(GRID_NS) * GRID_NS + if first_ns % GRID_NS == 0 { 0 } else { GRID_NS };
    let count = (start..last_ns)
        .step_by(usize::try_from(GRID_NS).unwrap_or(1))
        .count();
    u64::try_from(count).unwrap_or(0)
}

/// Every file under `dir` with its bytes.
fn files(dir: &Path) -> Result<BTreeMap<PathBuf, Vec<u8>>, Box<dyn Error>> {
    let mut found = BTreeMap::new();
    let mut folders = vec![dir.to_owned()];
    while let Some(folder) = folders.pop() {
        for entry in fs::read_dir(&folder)? {
            let path = entry?.path();
            if path.is_dir() {
                folders.push(path);
            } else if path.file_name().is_some_and(|n| n != "rotate.lock") {
                found.insert(path.clone(), fs::read(&path)?);
            }
        }
    }
    Ok(found)
}

/// A one-file zip, stored, as an archive of another content would be.
fn write_zip(path: &Path, member: &str, text: &str) -> Outcome {
    let mut zip = zip::ZipWriter::new(File::create(path)?);
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file(member, stored)?;
    zip.write_all(text.as_bytes())?;
    zip.finish()?;
    Ok(())
}

#[test]
fn a_bybit_book_day_round_trips_into_parquet() -> Outcome {
    let path = archive(BYBIT);
    let data = scratch("import-bybit");
    let imported = import(&path, &config(&data, Some(&TABLES)), &mut |_| {})?;
    let run_dir = &imported.run_dir;
    assert!(
        run_dir.ends_with("market/20261020T000000Z-bybit-linear-btcusdt"),
        "{}",
        run_dir.display()
    );

    // What the archive holds, read here.
    let lines = lines(&path)?;
    let (mut snapshots, mut snapshot_levels, mut delta_levels, mut times) = (0, 0, 0, Vec::new());
    for line in &lines {
        let message: Value = serde_json::from_str(line)?;
        let levels = message["data"]["b"].as_array().map_or(0, Vec::len)
            + message["data"]["a"].as_array().map_or(0, Vec::len);
        if message["type"] == "snapshot" {
            snapshots += 1;
            snapshot_levels += levels;
        } else {
            delta_levels += levels;
        }
        times.push(message["ts"].as_i64().ok_or("a ts")? * 1_000_000);
    }
    let records = u64::try_from(lines.len())?;
    assert_eq!((records, snapshots), (35, 2), "the fixture");

    assert_eq!(
        count(run_dir, "frames")?,
        records + 1,
        "the run's Meta, then a frame per record"
    );
    assert_eq!(
        count(run_dir, "depth_updates")?,
        u64::try_from(delta_levels)?
    );
    assert_eq!(
        count(run_dir, "snapshots")?,
        u64::try_from(snapshot_levels)?
    );
    assert_eq!(count(run_dir, "trades")?, 0);
    let (first, last) = (times[0], *times.last().ok_or("times")?);
    assert_eq!(count(run_dir, "book_samples")?, instants(first, last));

    // DuckDB again: frames keyed by seq, on the exchange clock; samples on its grid, all after
    // the snapshot, unsynced only between the gap and Bybit's next snapshot.
    let conn = connect()?;
    let glob = |table: &str| {
        format!(
            "read_parquet('{}/*.parquet')",
            run_dir.join(table).to_string_lossy().replace('\\', "/")
        )
    };
    let (distinct, clock_ok): (i64, bool) = conn.query_row(
        &format!("SELECT count(DISTINCT seq), bool_and(kind = 0 OR capture_ts_ns = source_ts_ns) FROM {}", glob("frames")),
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    assert_eq!(u64::try_from(distinct)?, records + 1);
    assert!(
        clock_ok,
        "an archive frame's capture time is its exchange time"
    );
    let (on_grid, unsynced, venues): (bool, i64, i64) = conn.query_row(
        &format!(
            "SELECT bool_and(sample_ts_ns % 100000000 = 0), count(*) FILTER (WHERE NOT synced), count(DISTINCT venue) FROM {}",
            glob("book_samples")
        ),
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    assert!(on_grid);
    assert!(
        unsynced > 0,
        "the gap leaves the book unsynced until the snapshot"
    );
    assert_eq!(venues, 1);
    let venue: String = conn.query_row(
        &format!("SELECT any_value(venue) FROM {}", glob("depth_updates")),
        [],
        |row| row.get(0),
    )?;
    assert_eq!(venue, "bybit-linear");
    let ids_only_bybit_sends: i64 = conn.query_row(
        &format!("SELECT count(*) FROM {} WHERE first_update_id IS NOT NULL OR prev_last_update_id IS NOT NULL", glob("depth_updates")),
        [],
        |row| row.get(0),
    )?;
    assert_eq!(ids_only_bybit_sends, 0, "Bybit sends only u");

    // The manifest names the venue, the clock and the archive.
    let manifest: Manifest = run::read_json(&run_dir.join(run::MANIFEST))?;
    assert_eq!(manifest.format, run::FORMAT);
    assert_eq!(manifest.params.venue, Venue::BybitLinear);
    assert_eq!(manifest.params.symbols, vec!["BTCUSDT".to_owned()]);
    assert_eq!(manifest.params.streams, vec!["orderbook.200".to_owned()]);
    assert_eq!(manifest.params.clock, Clock::Exchange);
    assert_eq!(manifest.started_ns, 1_792_454_400_000_000_000);
    assert_eq!(manifest.tables, TABLES.map(str::to_owned).to_vec());
    let DataSource::Archive {
        format,
        files: archived,
    } = &manifest.params.source
    else {
        return Err("an archive run".into());
    };
    assert_eq!(format, "bybit-orderbook");
    assert_eq!(archived.len(), 1);
    assert_eq!(archived[0].name, BYBIT);
    assert_eq!(archived[0].sha256, sha256_hex(&path)?);
    assert_eq!(archived[0].bytes, fs::metadata(&path)?.len());
    assert_eq!(
        archived[0].url,
        format!("https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/{BYBIT}")
    );

    // complete.json: rows as DuckDB counts them, and the integrity checks.
    let complete: Complete = run::read_json(&run_dir.join(run::COMPLETE))?;
    assert_eq!(complete.rows, imported.rows);
    for (table, rows) in &complete.rows {
        assert_eq!(*rows, count(run_dir, table)?, "{table}");
    }
    let check = complete.integrity.ok_or("integrity")?;
    assert_eq!(
        (
            check.records,
            check.snapshots,
            check.id_gaps,
            check.outside_day
        ),
        (35, Some(2), 1, 0)
    );
    assert_eq!((check.resyncs, check.errors, check.crossed), (1, 0, 0));
    assert_eq!(
        (check.first_ts_ns, check.last_ts_ns),
        (Some(first), Some(last))
    );
    assert_eq!(check, imported.integrity);

    // Again: nothing to do, nothing touched.
    let before = files(run_dir)?;
    let again = import(&path, &config(&data, Some(&TABLES)), &mut |_| {})?;
    assert!(again.done_before && again.written.is_empty(), "{again:?}");
    assert_eq!(files(run_dir)?, before, "a done run is left alone");

    // Other tables, or another archive of the same day, are refused.
    let fewer = import(&path, &config(&data, None), &mut |_| {});
    assert!(
        matches!(fewer, Err(ImportError::OtherTables { .. })),
        "{fewer:?}"
    );
    let other = scratch("import-bybit-other").join(BYBIT);
    let text = lines[..lines.len() - 1].join("\r\n") + "\r\n";
    write_zip(&other, "2026-10-20_BTCUSDT_ob200.data", &text)?;
    let refused = import(&other, &config(&data, Some(&TABLES)), &mut |_| {});
    assert!(
        matches!(refused, Err(ImportError::OtherArchive { .. })),
        "{refused:?}"
    );
    assert_eq!(files(run_dir)?, before, "a refused import touches nothing");
    fs::remove_dir_all(&data)?;
    fs::remove_dir_all(other.parent().ok_or("a folder")?)?;
    Ok(())
}

#[test]
fn a_binance_trades_day_round_trips_into_parquet() -> Outcome {
    let path = archive(BINANCE);
    let data = scratch("import-binance");
    let imported = import(&path, &config(&data, None), &mut |_| {})?;
    let run_dir = &imported.run_dir;
    assert!(run_dir.ends_with("market/20261020T000000Z-binance-usdm-btcusdt"));
    assert_eq!(
        imported.rows.keys().collect::<Vec<_>>(),
        vec!["trades"],
        "trades by default"
    );

    let lines = lines(&path)?;
    let rows: Vec<Vec<&str>> = lines[1..]
        .iter()
        .map(|line| line.split(',').collect())
        .collect();
    assert!(lines[0].starts_with("agg_trade_id,"), "the header");
    assert_eq!(count(run_dir, "trades")?, u64::try_from(rows.len())?);

    // Row for row: ids as strings, prices and quantities at the scanned scales, the trade time
    // as every time column, the venue.
    let conn = connect()?;
    let glob = run_dir.join("trades").to_string_lossy().replace('\\', "/");
    let mut statement = conn.prepare(&format!(
        "SELECT trade_id, first_trade_id, last_trade_id, price, qty, trade_ts_ns, capture_ts_ns, source_ts_ns, buyer_is_maker, venue, symbol FROM read_parquet('{glob}/*.parquet') ORDER BY seq"
    ))?;
    let stored: Vec<TradeRow> = statement
        .query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
                row.get(8)?,
                row.get(9)?,
                row.get(10)?,
            ))
        })?
        .collect::<Result<_, _>>()?;
    let units = |text: &str, decimals: usize| -> Result<i64, Box<dyn Error>> {
        let (int, frac) = text.split_once('.').unwrap_or((text, ""));
        Ok(format!("{int}{frac:0<decimals$}").parse()?)
    };
    for (row, line) in stored.iter().zip(&rows) {
        let t: i64 = line[5].parse::<i64>()? * 1_000_000;
        assert_eq!(row.0, line[0]);
        assert_eq!((row.1, row.2), (line[3].parse()?, line[4].parse()?));
        assert_eq!((row.3, row.4), (units(line[1], 1)?, units(line[2], 3)?));
        assert_eq!((row.5, row.6, row.7), (t, t, t));
        assert_eq!(row.8, line[6] == "true");
        assert_eq!(
            (row.9.as_str(), row.10.as_str()),
            ("binance-usdm", "BTCUSDT")
        );
    }
    let check = &imported.integrity;
    assert_eq!(
        (
            check.records,
            check.id_gaps,
            check.trade_id_gaps,
            check.snapshots
        ),
        (24, 1, Some(1), None)
    );
    assert_eq!((check.errors, check.resyncs, check.outside_day), (0, 0, 0));
    let manifest: Manifest = run::read_json(&run_dir.join(run::MANIFEST))?;
    assert_eq!(manifest.params.venue, Venue::BinanceUsdm);
    assert_eq!(manifest.params.streams, vec!["aggTrade".to_owned()]);
    let scales = manifest.params.scales["BTCUSDT"];
    assert_eq!((scales.price_decimals, scales.qty_decimals), (1, 3));

    // A .CHECKSUM that does not match refuses the archive before anything is written.
    let elsewhere = scratch("import-binance-checksum");
    let copy = elsewhere.join(BINANCE);
    fs::copy(&path, &copy)?;
    fs::write(
        elsewhere.join(format!("{BINANCE}.CHECKSUM")),
        format!("{}  {BINANCE}\n", "0".repeat(64)),
    )?;
    let data2 = scratch("import-binance-checksum-data");
    let refused = import(&copy, &config(&data2, None), &mut |_| {});
    assert!(
        matches!(refused, Err(ImportError::Checksum { .. })),
        "{refused:?}"
    );
    assert!(!data2.join("market").exists(), "nothing written");
    for dir in [data, elsewhere, data2] {
        fs::remove_dir_all(dir)?;
    }
    Ok(())
}

/// A broken archive is refused before anything is written: a record that is not one, a zip whose
/// CRC does not match, a file that is not the one its name says.
#[test]
fn a_broken_archive_leaves_no_run_behind() -> Outcome {
    let lines = lines(&archive(BYBIT))?;
    let folder = scratch("import-broken");
    let data = folder.join("data");
    let member = |day: &str| format!("{day}_BTCUSDT_ob200.data");
    let mut cases = Vec::new();

    // A line that is no message of the book.
    let mut text = lines.join("\r\n") + "\r\n";
    text.push_str("{\"topic\":\"orderbook.200.BTCUSDT\",\"type\":\"trade\"}\r\n");
    cases.push(("2026-10-21", member("2026-10-21"), text));
    // The stored bytes damaged after the CRC was taken.
    cases.push((
        "2026-10-22",
        member("2026-10-22"),
        lines.join("\r\n") + "\r\n",
    ));
    // A file inside named for another day.
    cases.push((
        "2026-10-23",
        member("2026-10-24"),
        lines.join("\r\n") + "\r\n",
    ));
    for (day, inside, text) in cases {
        let path = folder.join(format!("{day}_BTCUSDT_ob200.data.zip"));
        write_zip(&path, &inside, &text)?;
        if day == "2026-10-22" {
            let mut bytes = fs::read(&path)?;
            let at = bytes
                .windows(4)
                .position(|w| w == b"\"u\":")
                .ok_or("a u field")?;
            bytes[at + 5] = if bytes[at + 5] == b'1' { b'2' } else { b'1' };
            fs::write(&path, bytes)?;
        }
        let refused = import(&path, &config(&data, None), &mut |_| {});
        assert!(
            matches!(refused, Err(ImportError::BadArchive { .. })),
            "{day}: {refused:?}"
        );
        assert!(
            !data.join("market").exists(),
            "{day}: no run is left behind"
        );
    }
    fs::remove_dir_all(&folder)?;
    Ok(())
}

/// What the live path raises for each input, as a transcript.
async fn live_transcript(
    venue: Venue,
    imported: &Imported,
    messages: Vec<Message>,
) -> Result<Transcript, Box<dyn Error>> {
    let manifest: Manifest = run::read_json(&imported.run_dir.join(run::MANIFEST))?;
    let mut pipeline = Pipeline::new(Feed::new(venue, manifest.params.scales));
    let mut log = Log::unrecorded();
    let mut transcript = Transcript::default();
    let mut observer = |event: LiveEvent<'_>| transcript.observe(&event);
    for message in messages {
        handle(message, &mut pipeline, &mut log, &mut observer).await?;
    }
    Ok(transcript)
}

/// Replay equals import: each archive's records, fed as the venue's live stream (each record's
/// frame as socket text, captured at its exchange time) through the live loop's own path, raise
/// the same events with the same books after every input as the import does.
#[tokio::test]
async fn an_archive_imports_exactly_as_its_records_fed_live() -> Outcome {
    for (file, stream) in [(BYBIT, Stream::Depth), (BINANCE, Stream::Trades)] {
        let path = archive(file);
        let name = ArchiveName::parse(file).ok_or("an archive")?;
        let data = scratch("import-equals-live");
        let mut imported_transcript = Transcript::default();
        let imported = import(&path, &config(&data, None), &mut |event| {
            imported_transcript.observe(&event);
        })?;
        let mut messages = Vec::new();
        for line in lines(&path)? {
            let Some(frame) = name.frame(&line)? else {
                continue;
            };
            let inbound = Inbound::Text {
                text: frame.payload,
                dropped_before: 0,
                capture_ns: frame.time_ns,
            };
            messages.push(Message::Socket { stream, inbound });
        }
        let records = messages.len();
        let live = live_transcript(name.venue(), &imported, messages).await?;
        assert_eq!(live.steps.len(), records, "{file}: one step per record");
        assert_eq!(live.first_difference(&imported_transcript), None, "{file}");
        if stream == Stream::Depth {
            assert!(live.raised("SequenceGap"), "{file}: the gap is in both");
        }
        fs::remove_dir_all(&data)?;
    }
    Ok(())
}

/// Real archive days kept locally (gitignored): each imports with its rows matching its records,
/// and passes its integrity checks. Skips, and says so, when there are none.
#[test]
fn local_archive_days_import_cleanly() -> Outcome {
    let dir = repo().join("harness/fixtures/archives");
    let mut found: Vec<PathBuf> = match fs::read_dir(&dir) {
        Ok(entries) => entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| ArchiveName::parse(name).is_some())
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    found.sort();
    if found.is_empty() {
        println!("skipped: no archive days in {}", dir.display());
        return Ok(());
    }
    let data = scratch("import-local");
    for path in found {
        let imported = import(&path, &config(&data, None), &mut |_| {})?;
        let check = &imported.integrity;
        println!("{}: {:?} {check:?}", path.display(), imported.rows);
        assert_eq!(
            (check.errors, check.outside_day, check.crossed),
            (0, 0, 0),
            "{}",
            path.display()
        );
        if let Some(rows) = imported.rows.get("trades") {
            assert_eq!(
                *rows,
                check.records,
                "{}: a trade per record",
                path.display()
            );
        }
        if let (Some(rows), Some(first), Some(last)) = (
            imported.rows.get("book_samples"),
            check.first_ts_ns,
            check.last_ts_ns,
        ) {
            assert_eq!(
                *rows,
                instants(first, last),
                "{}: a sample per instant",
                path.display()
            );
        }
    }
    fs::remove_dir_all(&data)?;
    Ok(())
}
