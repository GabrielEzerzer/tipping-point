//! Bakes in the commit the engine is built from, for every run's manifest (#79), and puts
//! `DuckDB`'s shared library next to the engine's binaries (`store`, `serve`), so that they start
//! from a plain shell, a Tauri sidecar or a copied folder, not only under cargo (#52).
//!
//! The build links `DuckDB`'s prebuilt library, which libduckdb-sys downloads into
//! `target/duckdb-download/` and copies only into `target/<profile>/deps/`, the folder cargo
//! puts on `PATH` for `cargo run` and `cargo test`. Windows has no rpath: a binary run any other
//! way looks for `duckdb.dll` beside itself, then on `PATH`, and fails to start without it.
//! Linux and macOS binaries carry an rpath to the download folder, so only Windows needs this.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const LIBRARY: &str = "duckdb.dll";

/// What the engine is built from, besides its dependencies' own sources: whether any of it
/// differs from the commit is the manifests' `dirty`, and a change to it reruns this script.
const SOURCES: [&str; 5] = [
    "src",
    "build.rs",
    "Cargo.toml",
    "../Cargo.toml",
    "../Cargo.lock",
];

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    commit();
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

/// The commit the build is made from, as `TP_GIT_SHA`, and whether the engine's sources differ
/// from it, as `TP_GIT_DIRTY`. Taken at build time: a binary started outside cargo can run long
/// after the checkout has moved on. Neither is set without git or outside a checkout.
///
/// The script reruns when the sources change and when HEAD moves: HEAD names another commit or
/// branch, its branch gets a commit (in the loose ref, or in packed-refs once packed), and HEAD's
/// log gains a line on every move, the first commit after a pack included, whose loose ref was no
/// file to watch before. Only paths that exist are watched: cargo reruns a script on every build
/// for one that does not.
fn commit() {
    let git = |args: &[&str]| -> Option<String> {
        let output = Command::new("git").args(args).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let mut watched: Vec<String> = SOURCES.iter().map(|&path| path.to_owned()).collect();
    let in_git = ["HEAD", "logs/HEAD", "packed-refs"]
        .into_iter()
        .chain(git(&["symbolic-ref", "-q", "HEAD"]).as_deref())
        .filter_map(|name| git(&["rev-parse", "--git-path", name]))
        .collect::<Vec<_>>();
    watched.extend(in_git);
    for path in watched.iter().filter(|path| Path::new(path).exists()) {
        println!("cargo::rerun-if-changed={path}");
    }
    let Some(sha) = git(&["rev-parse", "HEAD"]) else {
        return;
    };
    let Some(status) = git(&[&["status", "--porcelain", "--"][..], &SOURCES].concat()) else {
        return;
    };
    println!("cargo::rustc-env=TP_GIT_SHA={sha}");
    println!("cargo::rustc-env=TP_GIT_DIRTY={}", !status.is_empty());
}

/// `target/<profile>` (or `target/<triple>/<profile>`) from `.../<profile>/build/<pkg>/out`.
fn profile_dir(out_dir: &Path) -> Option<PathBuf> {
    out_dir.ancestors().nth(3).map(Path::to_owned)
}
