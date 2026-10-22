//! The engine's binaries start outside cargo. `cargo run` and `cargo test` put
//! `target/<profile>/deps`, where `DuckDB`'s library lands, on `PATH`; a shell, a Tauri sidecar or
//! a copied folder does not. Each binary runs here with nothing of cargo's in its environment and
//! a `PATH` of system folders only, so it can load `DuckDB` only from beside itself (#52).

use std::path::Path;
use std::process::{Command, Output};

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
