//! The engine's binaries start outside cargo. `cargo run` and `cargo test` put
//! `target/<profile>/deps`, where `DuckDB`'s library lands, on `PATH`; a shell, a Tauri sidecar or
//! a copied folder does not. Each binary runs here with nothing of cargo's in its environment and
//! a `PATH` of system folders only, so it can load `DuckDB` only from beside itself (#52).

use std::path::Path;
use std::process::{Child, Command, Output};

/// A child killed and reaped when the test ends, however it ends short of the runner itself
/// being killed (`serve` is also started with `--seconds`, so it never runs on for long).
struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Runs `exe` with `args` and an environment holding only what any process needs.
fn run_bare(exe: &str, args: &[&str]) -> Output {
    let mut command = Command::new(exe);
    command.args(args).env_clear();
    for name in ["SystemRoot", "windir"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    if let Some(root) = std::env::var_os("SystemRoot") {
        command.env("PATH", Path::new(&root).join("System32"));
    }
    command.output().expect("the binary can be spawned")
}

/// Arrow's end-of-stream marker: a continuation token and a zero length.
const EOS: [u8; 8] = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];

/// The body of a whole chunked HTTP/1.1 response. hyper writes the terminating zero-length chunk
/// only for a body that ran to its end, never for one the engine aborted, so its absence, or a
/// chunk cut short, is an error.
fn whole_body(response: &[u8]) -> Result<Vec<u8>, String> {
    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("no end of the headers")?;
    let head = String::from_utf8_lossy(&response[..split]).to_ascii_lowercase();
    if !head.contains("transfer-encoding: chunked") {
        return Err(format!("not a chunked body: {head}"));
    }
    let mut rest = &response[split + 4..];
    let mut body = Vec::new();
    loop {
        let line = rest
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or("the body stops before its terminating chunk")?;
        let size = std::str::from_utf8(&rest[..line])
            .ok()
            .and_then(|text| usize::from_str_radix(text.trim(), 16).ok())
            .ok_or("an unreadable chunk size")?;
        rest = &rest[line + 2..];
        if size == 0 {
            return if rest == b"\r\n" {
                Ok(body)
            } else {
                Err("bytes after the terminating chunk".to_owned())
            };
        }
        if rest.len() < size + 2 || &rest[size..size + 2] != b"\r\n" {
            return Err("a chunk cut short".to_owned());
        }
        body.extend_from_slice(&rest[..size]);
        rest = &rest[size + 2..];
    }
}

/// The rows of a whole Arrow IPC stream in an HTTP response: the body is whole, every message
/// decodes, and the end marker is the last of them and the body's last 8 bytes.
fn whole_table(response: &[u8]) -> Result<usize, Box<dyn std::error::Error>> {
    use arrow::buffer::Buffer;
    use arrow::ipc::reader::StreamDecoder;

    let body = whole_body(response)?;
    if !body.ends_with(&EOS) {
        return Err("the stream does not end with Arrow's end marker".into());
    }
    let mut decoder = StreamDecoder::new();
    let mut buffer = Buffer::from(body);
    let mut rows = 0;
    // Past the end marker the decoder refuses any byte, so nothing follows it.
    while let Some(batch) = decoder.decode(&mut buffer)? {
        rows += batch.num_rows();
    }
    decoder.finish()?;
    Ok(rows)
}

/// The rows of a table's Parquet files.
fn stored_rows(table_dir: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    use parquet::file::reader::{FileReader, SerializedFileReader};

    let mut rows = 0;
    for entry in std::fs::read_dir(table_dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "parquet") {
            let reader = SerializedFileReader::new(std::fs::File::open(&path)?)?;
            rows += usize::try_from(reader.metadata().file_metadata().num_rows())?;
        }
    }
    Ok(rows)
}

/// A binary that cannot find its DLL never reaches `main`: Windows ends it with
/// `STATUS_DLL_NOT_FOUND`. Reaching the usage error proves it loaded.
#[test]
fn store_starts_without_cargo_on_path() {
    let output = run_bare(env!("CARGO_BIN_EXE_store"), &[]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("usage: store"), "{stderr}");
}

#[test]
fn serve_starts_without_cargo_on_path() {
    let output = run_bare(env!("CARGO_BIN_EXE_serve"), &["--minutes", "1"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("usage: serve"), "{stderr}");
}

/// The real `serve` process, as the app's sidecar or a shell would start it: a recorded script
/// played back and recorded on an ephemeral port, then asked for its health and a table that
/// only exists once it has rotated the run through `DuckDB`. The run's manifest names the commit
/// the binary was built from, though no git is on its `PATH` (#79). Killed at the end.
#[test]
fn serve_plays_a_script_records_it_and_serves_its_tables() -> Result<(), Box<dyn std::error::Error>>
{
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    use std::process::Stdio;

    let data = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("serve-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic/session-faults.jsonl");
    let mut command = Command::new(env!("CARGO_BIN_EXE_serve"));
    command
        .args(["--port", "0", "--record", "--speed", "0", "--seconds", "60"])
        .arg("--script")
        .arg(&script)
        .arg("--data-dir")
        .arg(&data)
        .env_clear()
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for name in ["SystemRoot", "windir"] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    let mut child = KillOnDrop(command.spawn()?);
    let mut lines = BufReader::new(child.0.stdout.take().ok_or("stdout")?).lines();
    let ready = lines.next().ok_or("serve printed nothing")??;
    let port: u16 = ready
        .strip_prefix("serve listening on http://127.0.0.1:")
        .ok_or_else(|| format!("not a ready line: {ready}"))?
        .parse()?;
    // The second line names the run; then rotation is reported once the script has played.
    let recording = lines.next().ok_or("no recording line")??;
    let run_id = recording
        .rsplit(['/', '\\'])
        .next()
        .ok_or("a run dir")?
        .to_owned();
    let rotated = lines.next().ok_or("no rotation line")??;
    assert!(rotated.contains("rotated"), "{rotated}");

    let get = |path: &str| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut socket = TcpStream::connect(("127.0.0.1", port))?;
        write!(
            socket,
            "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
        )?;
        let mut answer = Vec::new();
        socket.read_to_end(&mut answer)?;
        Ok(answer)
    };
    let health = String::from_utf8(get("/v1/health")?)?;
    assert!(health.starts_with("HTTP/1.1 200"), "{health}");
    assert!(health.contains("\"source\":\"ended\""), "{health}");
    let table = get(&format!("/v1/table/market/{run_id}/book_samples"))?;
    let head = String::from_utf8_lossy(&table[..table.len().min(300)]).to_string();
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(
        head.contains("application/vnd.apache.arrow.stream"),
        "{head}"
    );
    // The whole table, read as a client must: the end marker's 8 bytes also occur inside data
    // batches, so finding them anywhere passed a stream the engine cut short (#86).
    let rows = whole_table(&table)?;
    let stored = stored_rows(&data.join("market").join(&run_id).join("book_samples"))?;
    assert!(stored > 0, "rotate stored no book samples");
    assert_eq!(
        rows, stored,
        "the table served holds every row rotate stored"
    );
    let manifest = data.join("market").join(&run_id).join("manifest.json");
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let producer = &manifest["producer"];
    let built = option_env!("TP_GIT_SHA").ok_or("the build found no commit")?;
    assert_eq!(producer["git_sha"], built, "{producer}");
    let dirty = option_env!("TP_GIT_DIRTY") == Some("true");
    assert_eq!(producer["dirty"], dirty, "{producer}");
    child.0.kill()?;
    child.0.wait()?;
    let _ = std::fs::remove_dir_all(&data);
    Ok(())
}
