//! Arrow IPC streams as HTTP bodies: the schema message, one message per record batch, then the
//! 8-byte end-of-stream marker. No footer and no seeking, so a stream suits a live feed; no
//! compression and no dictionaries on loopback.
//!
//! A stream's schema is the stored one with the engine's metadata: `tp.schema` is
//! `"<table>/<major>"` (adding a column keeps the major, a rename, a type change or a change of
//! what keys a row bumps it) and `tp.domain`. An Ising stream adds `tp.run`; a market stream adds
//! `tp.runs` and `tp.scales`, JSON objects keyed by venue (a market stream can carry several
//! venues live, each recording its own run): `{"<venue>": "<run>"}` and
//! `{"<venue>": {"<symbol>": {"price_decimals", "qty_decimals"}}}`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::{CastOptions, cast_with_options};
use arrow::datatypes::{Schema, SchemaRef};
use arrow::error::ArrowError;
use arrow::ipc::MetadataVersion;
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};

use crate::market::{Scales, Symbol};
use crate::store::run::MARKET;
use crate::venue::Venue;

/// The media type of an Arrow IPC stream.
pub const STREAM_TYPE: &str = "application/vnd.apache.arrow.stream";
/// The schema-metadata key naming a stream's table and major version.
pub const SCHEMA_KEY: &str = "tp.schema";

/// The column the market tables gained with venues.
pub const VENUE_COLUMN: &str = "venue";

/// A table's major version, read from the schema it was stored with, never from its name: a
/// market table whose rows name their `venue` is at 2 (its rows are keyed by venue as well as
/// symbol; `trades`, once `agg_trades`, took string ids; `depth_updates` made the ids only some
/// venues send nullable). Everything else is at 1: frames, the Ising tables, and market tables
/// rotated before there were venues, which are served as they were stored.
#[must_use]
pub fn major(domain: &str, schema: &Schema) -> u32 {
    if domain == MARKET && schema.field_with_name(VENUE_COLUMN).is_ok() {
        2
    } else {
        1
    }
}

/// What a stream's schema says about where its rows come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance<'a> {
    pub table: &'a str,
    pub domain: &'a str,
    pub origin: Origin<'a>,
}

/// The runs a stream's rows come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin<'a> {
    /// One run of a domain without venues (Ising).
    Run(&'a str),
    /// Market rows: the run each venue records (a venue recording nothing has none) and each
    /// venue's scales.
    Market {
        runs: BTreeMap<Venue, &'a str>,
        scales: BTreeMap<Venue, &'a BTreeMap<Symbol, Scales>>,
    },
}

/// `schema` with its metadata replaced by the engine's.
///
/// # Panics
/// Never: runs and scales are plain strings and numbers and always serialize.
#[must_use]
pub fn tagged(schema: &Schema, provenance: &Provenance<'_>) -> SchemaRef {
    let major = major(provenance.domain, schema);
    let mut metadata = HashMap::from([
        (
            SCHEMA_KEY.to_owned(),
            format!("{}/{major}", provenance.table),
        ),
        ("tp.domain".to_owned(), provenance.domain.to_owned()),
    ]);
    match &provenance.origin {
        Origin::Run(run) => {
            metadata.insert("tp.run".to_owned(), (*run).to_owned());
        }
        Origin::Market { runs, scales } => {
            let runs = serde_json::to_string(runs).expect("runs always serialize");
            let scales = serde_json::to_string(scales).expect("scales always serialize");
            metadata.insert("tp.runs".to_owned(), runs);
            metadata.insert("tp.scales".to_owned(), scales);
        }
    }
    Arc::new(Schema::new_with_metadata(schema.fields().clone(), metadata))
}

/// 8-byte alignment: the format's minimum, which keeps a two-row live batch small.
fn options() -> IpcWriteOptions {
    IpcWriteOptions::try_new(8, false, MetadataVersion::V5)
        .expect("8-byte alignment with metadata V5 is a valid option set")
}

/// Turns batches into the messages of one stream, a chunk per call.
pub struct Encoder {
    writer: StreamWriter<Vec<u8>>,
    schema: SchemaRef,
}

impl std::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl Encoder {
    /// An encoder for `schema`, and the stream's first chunk: its schema message.
    ///
    /// # Errors
    /// Returns the Arrow error for a schema IPC cannot carry.
    pub fn new(schema: SchemaRef) -> Result<(Self, Vec<u8>), ArrowError> {
        let mut writer = StreamWriter::try_new_with_options(Vec::new(), &schema, options())?;
        let header = std::mem::take(writer.get_mut());
        Ok((Self { writer, schema }, header))
    }

    /// One batch's message. The batch takes the stream's schema (its fields must match).
    ///
    /// # Errors
    /// Returns the Arrow error when the batch's fields are not the stream's.
    pub fn batch(&mut self, batch: &RecordBatch) -> Result<Vec<u8>, ArrowError> {
        let batch = batch.clone().with_schema(self.schema.clone())?;
        self.writer.write(&batch)?;
        Ok(std::mem::take(self.writer.get_mut()))
    }

    /// The end-of-stream marker. Only a stream that really ended gets it: an error aborts the
    /// response instead, so that a reader never takes a cut stream for a short table.
    ///
    /// # Errors
    /// Returns the Arrow error, which a stream writer at its end never gives.
    pub fn finish(mut self) -> Result<Vec<u8>, ArrowError> {
        self.writer.finish()?;
        Ok(std::mem::take(self.writer.get_mut()))
    }
}

/// A batch read back through `DuckDB` in the schema the table was stored with. `DuckDB` marks
/// every column nullable and returns fixed-size binary (the Ising spins) as plain binary, so each
/// column is cast back to its stored type, and a stored non-nullable column holding a null is an
/// error rather than a silent change of contract.
///
/// # Errors
/// Returns the Arrow error for a missing column, a cast that would lose data, or a null in a
/// non-nullable column.
pub fn conform(batch: &RecordBatch, schema: &SchemaRef) -> Result<RecordBatch, ArrowError> {
    let strict = CastOptions {
        safe: false,
        ..CastOptions::default()
    };
    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            let column = batch.column_by_name(field.name()).ok_or_else(|| {
                ArrowError::SchemaError(format!("the result has no column {}", field.name()))
            })?;
            if column.data_type() == field.data_type() {
                Ok(Arc::clone(column))
            } else {
                cast_with_options(column, field.data_type(), &strict)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(Arc::clone(schema), columns)
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, BinaryArray, FixedSizeBinaryArray, Int64Array};
    use arrow::datatypes::{DataType, Field};
    use arrow::ipc::reader::StreamReader;

    use super::*;

    fn stored() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("step", DataType::Int64, false),
            Field::new("spins", DataType::FixedSizeBinary(2), true),
        ]))
    }

    /// What `DuckDB` hands back for that table: every column nullable, the spins as binary.
    fn from_duckdb(steps: Vec<Option<i64>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("step", DataType::Int64, true),
            Field::new("spins", DataType::Binary, true),
        ]));
        let spins: Vec<Option<&[u8]>> = steps.iter().map(|_| Some(&[0x80_u8, 0x01][..])).collect();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(steps)),
            Arc::new(BinaryArray::from(spins)),
        ];
        RecordBatch::try_new(schema, columns).expect("a valid batch")
    }

    #[test]
    fn a_duckdb_batch_is_cast_back_to_the_stored_schema() -> Result<(), ArrowError> {
        let batch = conform(&from_duckdb(vec![Some(1), Some(2)]), &stored())?;
        assert_eq!(batch.schema(), stored());
        let spins = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .expect("fixed-size binary");
        assert_eq!(spins.value(1), &[0x80, 0x01]);
        Ok(())
    }

    #[test]
    fn the_major_comes_from_what_was_stored() {
        let with = |names: &[&str]| {
            Schema::new(
                names
                    .iter()
                    .map(|name| Field::new(*name, DataType::Int64, false))
                    .collect::<Vec<_>>(),
            )
        };
        let keyed = with(&["sample_ts_ns", VENUE_COLUMN, "seq"]);
        let before_venues = with(&["sample_ts_ns", "seq"]);
        assert_eq!(major(MARKET, &keyed), 2);
        assert_eq!(major(MARKET, &before_venues), 1, "served as it was stored");
        assert_eq!(major("ising", &keyed), 1);
    }

    #[test]
    fn a_null_in_a_stored_non_nullable_column_is_refused() {
        assert!(conform(&from_duckdb(vec![Some(1), None]), &stored()).is_err());
    }

    #[test]
    fn spins_of_the_wrong_width_are_refused() {
        let narrow = Arc::new(Schema::new(vec![
            Field::new("step", DataType::Int64, false),
            Field::new("spins", DataType::FixedSizeBinary(3), true),
        ]));
        assert!(conform(&from_duckdb(vec![Some(1)]), &narrow).is_err());
    }

    #[test]
    fn chunks_concatenate_into_a_stream_that_reads_back() -> Result<(), ArrowError> {
        let provenance = Provenance {
            table: "stats",
            domain: "ising",
            origin: Origin::Run("r"),
        };
        let schema = tagged(&stored(), &provenance);
        let (mut encoder, mut bytes) = Encoder::new(schema.clone())?;
        let batch = conform(&from_duckdb(vec![Some(7)]), &stored())?;
        bytes.extend(encoder.batch(&batch)?);
        bytes.extend(encoder.finish()?);
        let reader = StreamReader::try_new(bytes.as_slice(), None)?;
        assert_eq!(reader.schema(), schema);
        assert_eq!(reader.schema().metadata()[SCHEMA_KEY], "stats/1");
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        Ok(())
    }
}
