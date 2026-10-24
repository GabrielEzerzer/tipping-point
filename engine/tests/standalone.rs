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
/// only exists once it has rotated the run through `DuckDB`. Killed at the end.
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
    assert!(
        table
            .windows(8)
            .any(|w| w == [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]),
        "the stream ends with Arrow's end marker"
    );
    child.0.kill()?;
    child.0.wait()?;
    let _ = std::fs::remove_dir_all(&data);
    Ok(())
}
