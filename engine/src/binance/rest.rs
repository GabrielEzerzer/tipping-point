//! The two REST calls the ingest needs: exchange info for scales, and depth snapshots.

use std::collections::BTreeMap;

use serde::Deserialize;
use thiserror::Error;

use super::decimal::step_decimals;
use super::{Scales, Symbol};

/// Base URL of the USD-M futures REST API.
pub const BASE_URL: &str = "https://fapi.binance.com";

/// Snapshot depth to ask for; 1000 is the deepest the endpoint serves (weight 20).
const SNAPSHOT_LIMIT: u32 = 1000;

/// Why a REST call failed.
#[derive(Debug, Error)]
pub enum RestError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("{0} answered {1}: {2}")]
    Status(String, u16, String),
    #[error("exchange info has no symbol {0:?}")]
    MissingSymbol(String),
    #[error("exchange info has no {1} filter for {0:?}")]
    MissingFilter(String, &'static str),
}

/// A thin client over the two endpoints. Cheap to clone; the connection pool is shared.
#[derive(Debug, Clone)]
pub struct Rest {
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct ExchangeInfo {
    symbols: Vec<SymbolInfo>,
}

#[derive(Debug, Deserialize)]
struct SymbolInfo {
    symbol: String,
    filters: Vec<Filter>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Filter {
    #[serde(rename = "filterType")]
    kind: String,
    tick_size: Option<String>,
    step_size: Option<String>,
}

impl Rest {
    /// A client against [`BASE_URL`].
    ///
    /// # Errors
    /// Returns an error when the TLS-backed HTTP client cannot be built.
    pub fn new() -> Result<Self, RestError> {
        let client = reqwest::Client::builder()
            .user_agent(concat!("tipping-point-engine/", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self { client })
    }

    /// Reads tick and lot sizes for the given symbols from `/fapi/v1/exchangeInfo`.
    ///
    /// # Errors
    /// Returns an error when the call fails or a symbol or its filters are missing.
    pub async fn fetch_scales(
        &self,
        symbols: &[Symbol],
    ) -> Result<BTreeMap<Symbol, Scales>, RestError> {
        let body = self.get("/fapi/v1/exchangeInfo", &[]).await?;
        let info: ExchangeInfo = serde_json::from_str(&body).map_err(|e| {
            RestError::Status("/fapi/v1/exchangeInfo".to_owned(), 200, e.to_string())
        })?;
        let mut scales = BTreeMap::new();
        for symbol in symbols {
            let entry = info
                .symbols
                .iter()
                .find(|s| &s.symbol == symbol)
                .ok_or_else(|| RestError::MissingSymbol(symbol.clone()))?;
            scales.insert(symbol.clone(), scales_of(entry)?);
        }
        Ok(scales)
    }

    /// Fetches the raw body of `/fapi/v1/depth` for a symbol, at full depth.
    ///
    /// # Errors
    /// Returns an error when the call fails or answers a non-2xx status.
    pub async fn fetch_snapshot(&self, symbol: &str) -> Result<String, RestError> {
        let limit = SNAPSHOT_LIMIT.to_string();
        self.get("/fapi/v1/depth", &[("symbol", symbol), ("limit", &limit)])
            .await
    }

    async fn get(&self, path: &str, query: &[(&str, &str)]) -> Result<String, RestError> {
        let url = format!("{BASE_URL}{path}");
        let response = self.client.get(&url).query(query).send().await?;
        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            let detail: String = body.chars().take(200).collect();
            return Err(RestError::Status(path.to_owned(), status.as_u16(), detail));
        }
        Ok(body)
    }
}

fn scales_of(info: &SymbolInfo) -> Result<Scales, RestError> {
    let find = |kind: &'static str| {
        info.filters
            .iter()
            .find(|f| f.kind == kind)
            .ok_or_else(|| RestError::MissingFilter(info.symbol.clone(), kind))
    };
    let tick = find("PRICE_FILTER")?
        .tick_size
        .as_deref()
        .ok_or_else(|| RestError::MissingFilter(info.symbol.clone(), "PRICE_FILTER.tickSize"))?;
    let step = find("LOT_SIZE")?
        .step_size
        .as_deref()
        .ok_or_else(|| RestError::MissingFilter(info.symbol.clone(), "LOT_SIZE.stepSize"))?;
    Ok(Scales {
        price_decimals: step_decimals(tick),
        qty_decimals: step_decimals(step),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_come_from_the_price_and_lot_filters() {
        let info: ExchangeInfo = serde_json::from_str(
            r#"{"symbols":[{"symbol":"BTCUSDT","filters":[
                {"filterType":"PRICE_FILTER","tickSize":"0.10","minPrice":"1"},
                {"filterType":"LOT_SIZE","stepSize":"0.001"},
                {"filterType":"MARKET_LOT_SIZE","stepSize":"0.001"}]}]}"#,
        )
        .expect("fixture parses");
        let scales = scales_of(&info.symbols[0]).expect("both filters present");
        assert_eq!(
            scales,
            Scales {
                price_decimals: 1,
                qty_decimals: 3
            }
        );
    }

    #[test]
    fn a_missing_filter_is_named() {
        let info = SymbolInfo {
            symbol: "X".to_owned(),
            filters: vec![],
        };
        assert!(matches!(
            scales_of(&info),
            Err(RestError::MissingFilter(_, "PRICE_FILTER"))
        ));
    }
}
