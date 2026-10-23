//! Shared code for the API checks: the server in-process on an ephemeral port, the live path
//! driven from a script or a generator with the API attached, and small clients for the Arrow
//! streams and the WebSocket.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, Cursor};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{RecordBatch, RecordBatchReader};
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use arrow::ipc::reader::StreamReader;
use engine::api::hub::{Hub, LiveTap};
use engine::api::{ApiConfig, serve};
use engine::market::feed::Feed;
use engine::market::live::{LiveEvent, Log, Observer, handle};
use engine::market::pipeline::{Message, Pipeline};
use engine::market::recording::{Record, read_records};
use engine::market::{Scales, Symbol};
use engine::store::frame::Meta;
use engine::store::run::{self, BookSample, MarketParams};
use engine::store::writer::{Entry, LogConfig, LogWriter, Summary};
use engine::venue::Venue;
use futures_util::{SinkExt, StreamExt};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

pub type Outcome<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

/// The repository root.
///
/// # Panics
/// When the crate is not inside the repository it was built from.
#[must_use]
pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the repository root exists")
}

/// A committed engine fixture, `engine/tests/fixtures/<rel>`.
#[must_use]
pub fn fixture(rel: &str) -> PathBuf {
    repo().join("engine/tests/fixtures").join(rel)
}

/// The run id of the committed Ising fixture.
pub const ISING_RUN: &str = "20261022T000000Z-fixture";

/// Copies the committed Ising fixture run into `data_dir/ising/`.
///
/// # Errors
/// Returns the I/O error.
pub fn copy_ising_fixture(data_dir: &Path) -> std::io::Result<PathBuf> {
    let from = fixture("ising").join(ISING_RUN);
    let to = data_dir.join("ising").join(ISING_RUN);
    let mut folders = vec![PathBuf::new()];
    while let Some(rel) = folders.pop() {
        fs::create_dir_all(to.join(&rel))?;
        for entry in fs::read_dir(from.join(&rel))? {
            let entry = entry?;
            let name = rel.join(entry.file_name());
            if entry.file_type()?.is_dir() {
                folders.push(name);
            } else {
                fs::copy(from.join(&name), to.join(&name))?;
            }
        }
    }
    Ok(to)
}

/// The API served in-process on 127.0.0.1 at an ephemeral port.
#[derive(Debug)]
pub struct Server {
    pub addr: SocketAddr,
    pub hub: Arc<Hub>,
    stop: watch::Sender<bool>,
    task: JoinHandle<std::io::Result<()>>,
}

impl Server {
    /// Starts serving `hub`.
    ///
    /// # Errors
    /// Returns the I/O error from binding.
    pub async fn start(hub: Arc<Hub>, config: ApiConfig) -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let (stop, stopped) = watch::channel(false);
        let task = tokio::spawn(serve(listener, Arc::clone(&hub), config, async move {
            let mut stopped = stopped;
            let _ = stopped.wait_for(|stop| *stop).await;
        }));
        Ok(Self {
            addr,
            hub,
            stop,
            task,
        })
    }

    #[must_use]
    pub fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    /// Stops accepting connections.
    pub async fn stop(self) {
        let _ = self.stop.send(true);
        let _ = self.task.await;
    }
}

/// The live path's observer in the checks: the API's tap, plus every input's seq, capture time
/// and the wall-clock moment the loop reached it.
#[derive(Debug)]
pub struct Watched {
    pub tap: LiveTap,
    pub inputs: Vec<(u64, i64, Instant)>,
}

impl Observer for Watched {
    fn started(&mut self, feed: &Feed, run_id: Option<&str>) {
        self.tap.started(feed, run_id);
    }

    fn input(&mut self, seq: u64, entry: &Entry, feed: &Feed) {
        self.inputs.push((seq, entry.capture_ts_ns, Instant::now()));
        self.tap.input(seq, entry, feed);
    }

    fn event(&mut self, event: LiveEvent<'_>) {
        self.tap.event(event);
    }
}

/// The live loop's own path (`handle`), network aside, with the API's tap attached and, when
/// recording, the run's log.
#[derive(Debug)]
pub struct LivePath {
    pub pipeline: Pipeline,
    pub log: Log,
    pub observer: Watched,
    pub run_dir: Option<PathBuf>,
}

impl LivePath {
    /// A path for `venue`'s streams recording to `data_dir/market/<run_id>` (small segments by
    /// `log`), or recording nothing when `run_id` is `None`.
    ///
    /// # Errors
    /// Returns the error from creating the run or starting its log.
    pub fn new(
        venue: Venue,
        scales: BTreeMap<Symbol, Scales>,
        hub: Arc<Hub>,
        data_dir: &Path,
        run_id: Option<&str>,
        log: LogConfig,
    ) -> Outcome<Self> {
        let (log, run_dir) = match run_id {
            Some(run_id) => {
                let run_dir = data_dir.join(run::MARKET).join(run_id);
                let params = MarketParams {
                    venue,
                    symbols: scales.keys().cloned().collect(),
                    scales: scales.clone(),
                    streams: vec!["depth@100ms".to_owned(), "aggTrade".to_owned()],
                    book_sample: BookSample::default(),
                };
                let manifest = run::create(&run_dir, params)?;
                let writer = LogWriter::start(&run_dir, Meta::for_run(&manifest), log)?;
                (Log::recorded(writer), Some(run_dir))
            }
            None => (Log::unrecorded(), None),
        };
        let pipeline = Pipeline::new(Feed::new(venue, scales));
        let mut observer = Watched {
            tap: LiveTap::new(hub),
            inputs: Vec::new(),
        };
        observer.started(pipeline.feed(), run_id);
        Ok(Self {
            pipeline,
            log,
            observer,
            run_dir,
        })
    }

    /// Feeds one message through the live path.
    ///
    /// # Errors
    /// Returns the log's error.
    pub async fn push(&mut self, message: Message) -> Outcome {
        handle(
            message,
            &mut self.pipeline,
            &mut self.log,
            &mut self.observer,
        )
        .await?;
        Ok(())
    }

    /// Closes the log and marks the run complete.
    ///
    /// # Errors
    /// Returns the writer's error.
    pub async fn finish(self) -> Outcome<(Option<Summary>, Watched, Pipeline)> {
        let log = self.log;
        let summary = tokio::task::spawn_blocking(move || log.finish()).await??;
        if let (Some(summary), Some(run_dir)) = (&summary, &self.run_dir) {
            run::complete(run_dir, summary)?;
        }
        Ok((summary, self.observer, self.pipeline))
    }
}

/// A script: its venue, its scales and its messages.
#[derive(Debug)]
pub struct Script {
    pub venue: Venue,
    pub scales: BTreeMap<Symbol, Scales>,
    pub messages: Vec<Message>,
}

/// Reads a script.
///
/// # Errors
/// Returns the error from reading the script.
pub fn script(path: &Path) -> Outcome<Script> {
    let mut records = read_records(BufReader::new(File::open(path)?))?.into_iter();
    let Some(Record::Scales { venue, scales }) = records.next() else {
        return Err("a script starts with its scales".into());
    };
    let messages = records
        .map(|record| record.into_message().ok_or("scales only on the first line"))
        .collect::<Result<_, _>>()?;
    Ok(Script {
        venue,
        scales,
        messages,
    })
}

/// Reads a whole Arrow IPC stream from the engine: the schema and every batch. An aborted
/// response, a stream without its end marker or a status other than 200 is an error.
///
/// # Errors
/// Returns what went wrong, as text.
pub async fn get_arrow(url: &str) -> Result<(SchemaRef, Vec<RecordBatch>), String> {
    let response = reqwest::get(url).await.map_err(|e| e.to_string())?;
    let status = response.status();
    let kind = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let bytes = response.bytes().await.map_err(|e| format!("body: {e}"))?;
    if !status.is_success() {
        return Err(format!("{status}: {}", String::from_utf8_lossy(&bytes)));
    }
    if kind != engine::api::ipc::STREAM_TYPE {
        return Err(format!("content type {kind:?}"));
    }
    decode_stream(&bytes)
}

/// Decodes an Arrow IPC stream, requiring its end-of-stream marker.
///
/// # Errors
/// Returns what went wrong, as text.
pub fn decode_stream(bytes: &[u8]) -> Result<(SchemaRef, Vec<RecordBatch>), String> {
    const EOS: [u8; 8] = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];
    if !bytes.ends_with(&EOS) {
        return Err("the stream has no end-of-stream marker".to_owned());
    }
    let reader = StreamReader::try_new(Cursor::new(bytes), None).map_err(|e| e.to_string())?;
    let schema = reader.schema();
    let batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    Ok((schema, batches))
}

/// Every Parquet file of a run's table, in name order, read with parquet-rs.
///
/// # Errors
/// Returns the I/O or Parquet error.
pub fn read_parquet(run_dir: &Path, table: &str) -> Outcome<(SchemaRef, Vec<RecordBatch>)> {
    let mut paths: Vec<PathBuf> = fs::read_dir(run_dir.join(table))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    paths.retain(|path| path.extension().is_some_and(|ext| ext == "parquet"));
    paths.sort();
    let mut schema = None;
    let mut batches = Vec::new();
    for path in paths {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?.build()?;
        schema.get_or_insert_with(|| reader.schema());
        for batch in reader {
            batches.push(batch?);
        }
    }
    Ok((schema.ok_or("no Parquet files")?, batches))
}

/// The batches as one, in the fields of `schema` (metadata dropped), so two reads compare
/// field for field.
///
/// # Errors
/// Returns the Arrow error when the batches do not fit the schema.
pub fn one_batch(schema: &SchemaRef, batches: &[RecordBatch]) -> Outcome<RecordBatch> {
    let bare = Arc::new(arrow::datatypes::Schema::new(schema.fields().clone()));
    let batches: Vec<RecordBatch> = batches
        .iter()
        .map(|batch| RecordBatch::try_new(Arc::clone(&bare), batch.columns().to_vec()))
        .collect::<Result<_, _>>()?;
    Ok(concat_batches(&bare, &batches)?)
}

/// A WebSocket client of the engine.
pub type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Connects to `/v1/ws` (no `Origin`: a non-browser client).
///
/// # Errors
/// Returns the handshake error.
pub async fn ws_connect(addr: SocketAddr) -> Outcome<Socket> {
    let (socket, _) = connect_async(format!("ws://{addr}/v1/ws")).await?;
    Ok(socket)
}

/// Sends a JSON text frame.
///
/// # Errors
/// Returns the socket error.
pub async fn send_json(socket: &mut Socket, value: &Value) -> Outcome {
    socket.send(WsMessage::text(value.to_string())).await?;
    Ok(())
}

/// What the next frame was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    Text(String),
    Binary(Vec<u8>),
    /// The engine closed the connection, with this code.
    Closed(Option<u16>),
    /// Nothing came within the wait.
    Silent,
}

/// The next data frame, waiting at most `wait`.
pub async fn next_frame(socket: &mut Socket, wait: Duration) -> Frame {
    let deadline = tokio::time::Instant::now() + wait;
    loop {
        match tokio::time::timeout_at(deadline, socket.next()).await {
            Err(_) => return Frame::Silent,
            Ok(Some(Ok(WsMessage::Text(text)))) => return Frame::Text(text.to_string()),
            Ok(Some(Ok(WsMessage::Binary(bytes)))) => return Frame::Binary(bytes.to_vec()),
            Ok(Some(Ok(WsMessage::Close(frame)))) => {
                return Frame::Closed(frame.map(|f| u16::from(f.code)));
            }
            Ok(Some(Ok(_))) => {}
            Ok(Some(Err(_)) | None) => return Frame::Closed(None),
        }
    }
}

/// The next text frame as JSON.
///
/// # Errors
/// Returns what came instead.
pub async fn next_json(socket: &mut Socket, wait: Duration) -> Outcome<Value> {
    match next_frame(socket, wait).await {
        Frame::Text(text) => Ok(serde_json::from_str(&text)?),
        other => Err(format!("wanted a text frame, got {other:?}").into()),
    }
}

/// Says hello and returns the welcome.
///
/// # Errors
/// Returns what came instead.
pub async fn hello(socket: &mut Socket) -> Outcome<Value> {
    let v = engine::api::messages::PROTOCOL;
    send_json(socket, &serde_json::json!({"t": "hello", "v": v})).await?;
    let welcome = next_json(socket, Duration::from_secs(10)).await?;
    if welcome["t"] != "welcome" {
        return Err(format!("wanted welcome, got {welcome}").into());
    }
    Ok(welcome)
}

/// Writes `actual` as the golden file at `path` when `TP_UPDATE_GOLDEN` is set; otherwise
/// requires them equal, naming the file and how to regenerate it.
///
/// # Errors
/// Returns the difference, or the I/O error.
pub fn golden(path: &Path, actual: &[u8]) -> Outcome {
    if std::env::var_os("TP_UPDATE_GOLDEN").is_some() {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, actual)?;
        return Ok(());
    }
    let expected = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if expected == actual {
        return Ok(());
    }
    let first = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a != b)
        .unwrap_or(expected.len().min(actual.len()));
    Err(format!(
        "{} differs from what the engine produces now (first at byte {first}, {} vs {} bytes). \
         If the protocol change is intended, regenerate it in the same commit: \
         TP_UPDATE_GOLDEN=1 cargo test -p engine-harness",
        path.display(),
        expected.len(),
        actual.len()
    )
    .into())
}

/// The Python that runs the client checks: `TP_PYTHON`, the active virtual environment, or the
/// repository's `.venv`.
///
/// # Panics
/// When there is none: a check that cannot run is red, not skipped.
#[must_use]
pub fn python() -> PathBuf {
    if let Some(python) = std::env::var_os("TP_PYTHON") {
        return PathBuf::from(python);
    }
    let in_venv = |venv: &Path| {
        [venv.join("Scripts/python.exe"), venv.join("bin/python")]
            .into_iter()
            .find(|path| path.is_file())
    };
    std::env::var_os("VIRTUAL_ENV")
        .and_then(|venv| in_venv(Path::new(&venv)))
        .or_else(|| in_venv(&repo().join(".venv")))
        .expect("no Python: run `uv sync --extra cpu` or set TP_PYTHON; the client checks cannot be skipped")
}
