//! Puts `DuckDB`'s shared library next to the engine's binaries (`store`, `serve`), so that they
//! start from a plain shell, a Tauri sidecar or a copied folder, not only under cargo (#52).
//!
//! The build links `DuckDB`'s prebuilt library, which libduckdb-sys downloads into
//! `target/duckdb-download/` and copies only into `target/<profile>/deps/`, the folder cargo
//! puts on `PATH` for `cargo run` and `cargo test`. Windows has no rpath: a binary run any other
//! way looks for `duckdb.dll` beside itself, then on `PATH`, and fails to start without it.
//! Linux and macOS binaries carry an rpath to the download folder, so only Windows needs this.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const LIBRARY: &str = "duckdb.dll";

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if env::var_os("CARGO_CFG_WINDOWS").is_none() {
        return;
    }
    // libduckdb-sys (`links = "duckdb"`) tells its direct dependents where the library it
    // downloaded lives; depending on it directly is also what makes its build script run first.
    println!("cargo::rerun-if-env-changed=DEP_DUCKDB_INCLUDE");
    let Some(source) = env::var_os("DEP_DUCKDB_INCLUDE")
        .map(|dir| PathBuf::from(dir).join(LIBRARY))
        .filter(|path| path.is_file())
    else {
        println!(
            "cargo::warning=no downloaded {LIBRARY} to copy beside the binaries; they will need it on PATH"
        );
        return;
    };
    let Some(profile) = env::var_os("OUT_DIR").and_then(|out| profile_dir(Path::new(&out))) else {
        println!("cargo::warning=cannot tell the profile folder from OUT_DIR");
        return;
    };
    let target = profile.join(LIBRARY);
    println!("cargo::rerun-if-changed={}", source.display());
    // A missing copy reruns this script, so deleting it brings it back on the next build.
    println!("cargo::rerun-if-changed={}", target.display());
    // A binary still running from the profile folder holds the old copy open; the build goes on
    // and says so rather than failing.
    if let Err(error) = fs::copy(&source, &target) {
        println!(
            "cargo::warning=could not copy {} to {}: {error}",
            source.display(),
            target.display()
        );
    }
}

/// `target/<profile>` (or `target/<triple>/<profile>`) from `.../<profile>/build/<pkg>/out`.
fn profile_dir(out_dir: &Path) -> Option<PathBuf> {
    out_dir.ancestors().nth(3).map(Path::to_owned)
}
