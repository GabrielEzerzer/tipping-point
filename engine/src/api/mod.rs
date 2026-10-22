//! The engine's local API: one HTTP server on 127.0.0.1, run by the `serve` binary as its own
//! process. Arrow IPC streams go to the Python side, JSON over a WebSocket to the app.
//!
//! | route | what |
//! |---|---|
//! | `GET /v1/health` | JSON: alive, protocol version, whether the live feed has started |
//! | `GET /v1/runs` | JSON: every recorded run in the data dir, by domain |
//! | `GET /v1/live/book_samples` | Arrow stream: live book samples, one batch per 100 ms instant; `?backfill_s=` starts with that much of the recent past, `?run=&speed=` replays a recorded run instead |
//! | `GET /v1/table/{domain}/{run}/{table}` | Arrow stream: any table of a market or Ising run, `?start=&end=` a range of its key, through `DuckDB`, in the stored schema |
//! | `POST /v1/publish/{topic}` | JSON object: a result (`meters`, `jev`, `pnl`, `ising`) relayed to the app |
//! | `GET /v1/ws` | the app's WebSocket ([`messages`] has the protocol) |
//!
//! Training reads Parquet directly (Kaggle cannot reach a local engine); the engine's Arrow
//! streams serve live inference and replay tests of it. An Arrow stream is lossless or dead: a
//! subscriber that falls behind, an error, or a stalled client aborts the response and drops the
//! connection, so a reader never takes a cut stream for a short one.
//!
//! No token, but two checks on every request, the WebSocket upgrade included: `Host` must name
//! this server by loopback address or `localhost` (which stops DNS rebinding), and a browser's
//! `Origin` must be on the allowlist (browsers apply no same-origin rule to a WebSocket). A request
//! without `Origin` comes from a non-browser client and is let through.

pub mod hub;
pub mod ipc;
pub mod messages;
mod replay;
pub mod stream;
pub mod table;
mod ws;

use std::future::Future;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::broadcast::error::RecvError;
use tower::ServiceExt;

use hub::{Hub, LiveItem, live_schema};
use ipc::{Encoder, STREAM_TYPE};
use messages::{PROTOCOL, Topic};
use stream::{Chunks, Fault, Kill};
use table::{DOMAINS, TableError, locate};

use crate::store::run::{self, COMPLETE, MANIFEST, MARKET};
use crate::store::samples::{Sample, SampleRows};

/// The port `serve` listens on unless told otherwise.
pub const DEFAULT_PORT: u16 = 7731;
/// The largest result a client can publish.
const MAX_PUBLISH: usize = 64 * 1024;
/// Instants of a backfill sent per Arrow batch.
const BACKFILL_BATCH: usize = 512;

/// The origins a browser may connect from by default: the Tauri app's webview on Windows
/// (`http(s)://tauri.localhost`) and elsewhere (`tauri://localhost`), and the Vite dev server
/// (Tauri's template port 1420, Vite's own 5173).
pub const DEFAULT_ORIGINS: [&str; 7] = [
    "http://tauri.localhost",
    "https://tauri.localhost",
    "tauri://localhost",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
];

/// How the API serves.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Where recorded runs live: `<data dir>/<domain>/<run_id>/`.
    pub data_dir: PathBuf,
    /// Origins a browser may connect from.
    pub origins: Vec<String>,
    /// A WebSocket client whose socket takes nothing for this long is closed.
    pub ws_stall: Duration,
    /// An Arrow client whose connection takes nothing for this long is cut.
    pub arrow_stall: Duration,
    /// Harness only: a failure every Arrow stream meets partway. `None` in production.
    pub fault: Option<Fault>,
}

impl ApiConfig {
    /// The defaults, serving runs from `data_dir`.
    #[must_use]
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            origins: DEFAULT_ORIGINS.iter().map(|&o| o.to_owned()).collect(),
            ws_stall: Duration::from_secs(10),
            arrow_stall: Duration::from_secs(10),
            fault: None,
        }
    }
}

/// What every handler shares.
#[derive(Debug)]
pub struct Api {
    pub hub: Arc<Hub>,
    pub config: ApiConfig,
    /// The `Host` values that name this server.
    hosts: Vec<String>,
}

impl Api {
    /// Why a request is refused, if it is: a `Host` naming another server, or a browser
    /// `Origin` off the allowlist.
    fn refuse(&self, headers: &HeaderMap) -> Option<&'static str> {
        let host = headers.get(header::HOST).and_then(|v| v.to_str().ok());
        if !host.is_some_and(|host| self.hosts.iter().any(|ours| ours == host)) {
            return Some("Host is not this server");
        }
        if let Some(origin) = headers.get(header::ORIGIN) {
            let allowed = origin
                .to_str()
                .is_ok_and(|origin| self.config.origins.iter().any(|o| o == origin));
            if !allowed {
                return Some("Origin not allowed");
            }
        }
        None
    }
}

/// An error answer: a status and a JSON `{"error": ...}`.
#[derive(Debug)]
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<TableError> for ApiError {
    fn from(error: TableError) -> Self {
        match error {
            TableError::BadName(why) => Self(StatusCode::BAD_REQUEST, why),
            TableError::NotFound(why) => Self(StatusCode::NOT_FOUND, why),
            TableError::Store(error) => Self(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
        }
    }
}

fn bad_request(why: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, why.into())
}

/// The routes, behind the Host and Origin checks.
pub fn router(api: Arc<Api>) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/runs", get(runs))
        .route("/v1/live/book_samples", get(live_samples))
        .route("/v1/table/{domain}/{run}/{table}", get(table_stream))
        .route(
            "/v1/publish/{topic}",
            post(publish).layer(DefaultBodyLimit::max(MAX_PUBLISH)),
        )
        .route("/v1/ws", get(ws::upgrade))
        .layer(middleware::from_fn_with_state(Arc::clone(&api), guard))
        .with_state(api)
}

async fn guard(State(api): State<Arc<Api>>, request: Request, next: Next) -> Response {
    if let Some(why) = api.refuse(request.headers()) {
        return ApiError(StatusCode::FORBIDDEN, why.to_owned()).into_response();
    }
    next.run(request).await
}

/// Serves `hub` on `listener` until `shutdown` resolves. Each connection runs on its own task;
/// a streaming response that has to be cut drops its connection through a [`Kill`] put in the
/// request's extensions.
///
/// # Errors
/// Returns the error from reading the listener's address.
pub async fn serve(
    listener: TcpListener,
    hub: Arc<Hub>,
    config: ApiConfig,
    shutdown: impl Future<Output = ()> + Send,
) -> io::Result<()> {
    let port = listener.local_addr()?.port();
    let api = Arc::new(Api {
        hub,
        config,
        hosts: vec![format!("127.0.0.1:{port}"), format!("localhost:{port}")],
    });
    let app = router(api);
    tokio::pin!(shutdown);
    loop {
        let accepted = tokio::select! {
            accepted = listener.accept() => accepted,
            () = &mut shutdown => return Ok(()),
        };
        // Out of sockets, or a client gone mid-accept: the server carries on.
        let Ok((stream, _)) = accepted else {
            tokio::time::sleep(Duration::from_millis(50)).await;
            continue;
        };
        let _ = stream.set_nodelay(true);
        let kill = Kill::default();
        let service = app.clone().map_request({
            let kill = kill.clone();
            move |mut request: Request<Incoming>| {
                request.extensions_mut().insert(kill.clone());
                request
            }
        });
        tokio::spawn(async move {
            let connection = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), TowerToHyperService::new(service))
                .with_upgrades();
            tokio::select! {
                _ = connection => {}
                () = kill.killed() => {}
            }
        });
    }
}

async fn health(State(api): State<Arc<Api>>) -> Json<Value> {
    let source = api.hub.started();
    let state = if api.hub.has_ended() {
        "ended"
    } else if source.is_some() {
        "live"
    } else {
        "starting"
    };
    Json(json!({
        "status": "ok",
        "v": PROTOCOL,
        "engine": crate::version(),
        "source": state,
        "run": source.and_then(|source| source.run.clone()),
    }))
}

/// Every run with a manifest, by domain: id, start, key, tables, and whether it ended cleanly.
async fn runs(State(api): State<Arc<Api>>) -> Result<Json<Value>, ApiError> {
    let data_dir = api.config.data_dir.clone();
    let listed = tokio::task::spawn_blocking(move || {
        let mut out = serde_json::Map::new();
        for domain in DOMAINS {
            let mut runs = Vec::new();
            for run_dir in run::runs(&data_dir.join(domain))? {
                let Ok(manifest) = run::read_json::<Value>(&run_dir.join(MANIFEST)) else {
                    continue;
                };
                runs.push(json!({
                    "run": manifest.get("run_id"),
                    "started_ns": manifest.get("started_ns"),
                    "key": manifest.get("key"),
                    "tables": manifest.get("tables"),
                    "complete": run_dir.join(COMPLETE).is_file(),
                }));
            }
            out.insert(domain.to_owned(), Value::Array(runs));
        }
        Ok::<_, io::Error>(Value::Object(out))
    })
    .await
    .map_err(|error| ApiError(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?
    .map_err(|error| ApiError(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    Ok(Json(listed))
}

fn arrow_response(body: Body) -> Response {
    let mut response = Response::new(body);
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(STREAM_TYPE));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

fn body_for(api: &Api, kill: Option<Extension<Kill>>) -> (Chunks, Body) {
    stream::channel(
        kill.map(|Extension(kill)| kill),
        api.config.arrow_stall,
        api.config.fault,
    )
}

/// A non-negative, finite number of seconds.
fn seconds(value: Option<f64>, name: &str) -> Result<Option<f64>, ApiError> {
    match value {
        Some(s) if !s.is_finite() || s < 0.0 => Err(bad_request(format!(
            "{name} must be a number of seconds, 0 or more"
        ))),
        other => Ok(other),
    }
}

#[derive(Debug, Deserialize)]
struct LiveQuery {
    backfill_s: Option<f64>,
    run: Option<String>,
    speed: Option<f64>,
}

async fn live_samples(
    State(api): State<Arc<Api>>,
    Query(query): Query<LiveQuery>,
    kill: Option<Extension<Kill>>,
) -> Result<Response, ApiError> {
    let backfill = seconds(query.backfill_s, "backfill_s")?;
    let speed = query.speed.unwrap_or(1.0);
    if !replay::valid_speed(speed) {
        return Err(bad_request(format!(
            "speed is 0 (as fast as possible) or at least {}",
            replay::MIN_SPEED
        )));
    }
    if let Some(run) = query.run {
        let table = locate(&api.config.data_dir, MARKET, &run, "book_samples")?;
        let schema = blocking_schema(table.clone()).await?;
        let (chunks, body) = body_for(&api, kill);
        let runtime = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            let _ = table.replay_samples(&schema, speed, chunks, &runtime);
        });
        return Ok(arrow_response(body));
    }
    let hub = Arc::clone(&api.hub);
    let source = hub.source().await;
    let levels = hub.config().book_sample.levels;
    let schema = live_schema(&source, levels);
    let backfill = Duration::from_secs_f64(backfill.unwrap_or(0.0).min(1e9));
    let (past, rx, ended) = hub.subscribe_samples(backfill);
    let (chunks, body) = body_for(&api, kill);
    tokio::spawn(live_stream(schema, levels, past, rx, ended, chunks));
    Ok(arrow_response(body))
}

async fn blocking_schema(table: table::Table) -> Result<arrow::datatypes::SchemaRef, ApiError> {
    tokio::task::spawn_blocking(move || table.schema())
        .await
        .map_err(|error| ApiError(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?
        .map_err(|error| ApiError(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))
}

/// The live stream: the schema, the backfill, then each instant as the hub publishes it, until
/// the feed ends (a clean end) or the subscriber falls behind (an abort).
async fn live_stream(
    schema: arrow::datatypes::SchemaRef,
    levels: usize,
    past: Vec<Arc<Sample>>,
    mut rx: tokio::sync::broadcast::Receiver<LiveItem>,
    ended: bool,
    mut chunks: Chunks,
) {
    let Ok((mut encoder, header)) = Encoder::new(schema) else {
        chunks.abort("cannot encode the schema");
        return;
    };
    if chunks.send(header).await.is_err() {
        return;
    }
    let mut rows = SampleRows::new(levels);
    for group in past.chunks(BACKFILL_BATCH) {
        for sample in group {
            rows.push(sample);
        }
        let chunk = rows
            .take()
            .ok()
            .and_then(|batch| encoder.batch(&batch).ok());
        let Some(chunk) = chunk else {
            chunks.abort("cannot encode the backfill");
            return;
        };
        if chunks.send(chunk).await.is_err() {
            return;
        }
    }
    if !ended {
        loop {
            match rx.recv().await {
                Ok(LiveItem::Sample(published)) => {
                    if chunks.send(published.ipc.clone()).await.is_err() {
                        return;
                    }
                }
                Ok(LiveItem::End) => break,
                Err(RecvError::Lagged(_)) => {
                    chunks.abort("fell behind the live feed");
                    return;
                }
                Err(RecvError::Closed) => {
                    chunks.abort("the live feed is gone");
                    return;
                }
            }
        }
    }
    match encoder.finish() {
        Ok(end) => {
            let _ = chunks.finish(end).await;
        }
        Err(_) => chunks.abort("cannot end the stream"),
    }
}

#[derive(Debug, Deserialize)]
struct KeyRange {
    start: Option<i64>,
    end: Option<i64>,
}

async fn table_stream(
    State(api): State<Arc<Api>>,
    Path((domain, run, name)): Path<(String, String, String)>,
    Query(range): Query<KeyRange>,
    kill: Option<Extension<Kill>>,
) -> Result<Response, ApiError> {
    let table = locate(&api.config.data_dir, &domain, &run, &name)?;
    let schema = blocking_schema(table.clone()).await?;
    let (chunks, body) = body_for(&api, kill);
    let runtime = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let _ = table.stream(&schema, (range.start, range.end), chunks, &runtime);
    });
    Ok(arrow_response(body))
}

async fn publish(
    State(api): State<Arc<Api>>,
    Path(topic): Path<String>,
    body: Bytes,
) -> Result<StatusCode, ApiError> {
    let topic = Topic::relayed(&topic).ok_or_else(|| {
        ApiError(
            StatusCode::NOT_FOUND,
            format!("{topic:?} is not a topic results are published to (meters, jev, pnl, ising)"),
        )
    })?;
    let data: Value =
        serde_json::from_slice(&body).map_err(|error| bad_request(format!("not JSON: {error}")))?;
    if !data.is_object() {
        return Err(bad_request("a result is a JSON object"));
    }
    api.hub.relay(topic, data);
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api() -> Api {
        Api {
            hub: Hub::new(hub::HubConfig::default()),
            config: ApiConfig::new(PathBuf::from(".")),
            hosts: vec!["127.0.0.1:9".to_owned(), "localhost:9".to_owned()],
        }
    }

    fn headers(pairs: &[(header::HeaderName, &'static str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(name.clone(), HeaderValue::from_static(value));
        }
        headers
    }

    #[test]
    fn only_this_server_and_allowed_origins_get_in() {
        let api = api();
        assert_eq!(api.refuse(&headers(&[(header::HOST, "127.0.0.1:9")])), None);
        assert_eq!(api.refuse(&headers(&[(header::HOST, "localhost:9")])), None);
        assert!(api.refuse(&headers(&[])).is_some(), "no Host");
        assert!(
            api.refuse(&headers(&[(header::HOST, "evil.example:9")]))
                .is_some(),
            "DNS rebinding"
        );
        assert!(
            api.refuse(&headers(&[(header::HOST, "127.0.0.1:10")]))
                .is_some()
        );
        let tauri = [
            (header::HOST, "127.0.0.1:9"),
            (header::ORIGIN, "http://tauri.localhost"),
        ];
        assert_eq!(api.refuse(&headers(&tauri)), None);
        let foreign = [
            (header::HOST, "127.0.0.1:9"),
            (header::ORIGIN, "https://evil.example"),
        ];
        assert_eq!(api.refuse(&headers(&foreign)), Some("Origin not allowed"));
    }
}
