//! `DuckDB` over a run's Parquet. Connections are in-memory and only read Parquet (there is no
//! `.duckdb` file), so this library and a `duckdb` CLI used for exploring never have to agree on
//! a database format. Extensions are never downloaded: the Parquet reader is built into the
//! library this crate links.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use duckdb::{Config, Connection};

use super::StoreError;
use super::frame::{Frame, Kind};

/// An in-memory connection that will not fetch extensions from the network.
///
/// # Errors
/// Returns the `DuckDB` error when the database cannot be opened.
pub fn connect() -> Result<Connection, StoreError> {
    let config = Config::default()
        .with("autoinstall_known_extensions", "false")?
        .with("autoload_known_extensions", "false")?;
    Ok(Connection::open_in_memory_with_flags(config)?)
}

/// A table's Parquet files in a run, in seq order (their names are zero-padded first seqs).
/// Temp files from an unfinished rotate are not Parquet files and never match.
///
/// # Errors
/// Returns the I/O error from listing the table's folder; a missing folder is no files.
pub fn table_files(run_dir: &Path, table: &str) -> Result<Vec<PathBuf>, StoreError> {
    let dir = run_dir.join(table);
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(StoreError::io(&dir)(error)),
    };
    let mut files = Vec::new();
    for entry in entries {
        let path = entry.map_err(StoreError::io(&dir))?.path();
        if path.extension().is_some_and(|ext| ext == "parquet") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// A path as a `DuckDB` string literal: forward slashes, single quotes doubled. Every path that
/// reaches SQL goes through here, the API's tables included.
pub(crate) fn literal(path: &Path) -> String {
    let text = path
        .to_string_lossy()
        .replace('\\', "/")
        .replace('\'', "''");
    format!("'{text}'")
}

/// Calls `each` with every frame in one segment's `frames` file, in seq order.
///
/// # Errors
/// Returns `DuckDB` errors, a row that is not a frame, or the first error `each` returns.
pub fn for_each_frame(
    conn: &Connection,
    path: &Path,
    mut each: impl FnMut(Frame) -> Result<(), StoreError>,
) -> Result<(), StoreError> {
    let sql = format!(
        "SELECT seq, kind, source_ts_ns, capture_ts_ns, payload FROM read_parquet({}) ORDER BY seq",
        literal(path)
    );
    let mut statement = conn.prepare(&sql)?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let seq: i64 = row.get(0)?;
        let code: i8 = row.get(1)?;
        let kind = u8::try_from(code)
            .ok()
            .and_then(Kind::from_code)
            .ok_or(StoreError::BadKind(i64::from(code)))?;
        each(Frame {
            seq: u64::try_from(seq).map_err(|_| StoreError::BadSeq(seq))?,
            kind,
            source_ts_ns: row.get(2)?,
            capture_ts_ns: row.get(3)?,
            payload: row.get(4)?,
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_become_safe_literals() {
        assert_eq!(
            literal(Path::new(r"C:\data\market\it's\frames\x.parquet")),
            "'C:/data/market/it''s/frames/x.parquet'"
        );
    }

    #[test]
    fn the_parquet_reader_is_built_in() -> Result<(), StoreError> {
        let conn = connect()?;
        let mode: String = conn.query_row(
            "SELECT install_mode FROM duckdb_extensions() WHERE extension_name = 'parquet'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(mode, "STATICALLY_LINKED");
        Ok(())
    }
}
