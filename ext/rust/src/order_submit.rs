// order_submit.rs — Rust WS order gateway for Binance Futures
//
// Builds signed JSON-RPC messages for Binance WS-API order placement.
// HMAC-SHA256 signing done in Rust (no Python round-trip).
// Sends via RustWsClient.send() for ~4ms latency (vs ~30-200ms REST).

use hmac::{Hmac, Mac};
use pyo3::prelude::*;
use sha2::Sha256;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn hmac_sha256_hex(secret: &str, payload: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC key");
    mac.update(payload.as_bytes());
    let result = mac.finalize();
    let bytes = result.into_bytes();
    let mut hex = String::with_capacity(bytes.len() * 2);
    for b in bytes.iter() {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

fn url_encode_val(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

/// Rust WS order gateway for Binance Futures.
///
/// Builds signed JSON-RPC order messages and sends via a RustWsClient.
/// All signing and serialization done in Rust — no Python overhead.
///
/// Usage:
///   gateway = RustWsOrderGateway(api_key, api_secret, recv_window=5000)
///   gateway.connect(ws_client, "wss://ws-fapi.binance.com/ws-fapi/v1")
///   request_id = gateway.place_order(symbol="BTCUSDT", side="BUY", ...)
///   response = ws_client.recv()  # JSON-RPC response
#[pyclass(name = "RustWsOrderGateway")]
pub struct RustWsOrderGateway {
    api_key: String,
    api_secret: String,
    recv_window: i64,
}

#[pymethods]
impl RustWsOrderGateway {
    #[new]
    #[pyo3(signature = (api_key, api_secret, recv_window=5000))]
    fn new(api_key: &str, api_secret: &str, recv_window: i64) -> Self {
        Self {
            api_key: api_key.to_string(),
            api_secret: api_secret.to_string(),
            recv_window,
        }
    }

    /// Build and sign a Binance WS-API order.place message.
    ///
    /// Returns the JSON string ready to send via ws_client.send().
    /// Also returns the request ID for correlating with response.
    #[pyo3(signature = (
        symbol, side, order_type,
        quantity=None, price=None, time_in_force=None,
        reduce_only=None, new_client_order_id=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build_order_message(
        &self,
        symbol: &str,
        side: &str,
        order_type: &str,
        quantity: Option<&str>,
        price: Option<&str>,
        time_in_force: Option<&str>,
        reduce_only: Option<bool>,
        new_client_order_id: Option<&str>,
    ) -> PyResult<(String, String)> {
        let req_id = format!("ord_{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed));
        let ts = now_ms();

        // Build params for signing (sorted alphabetically)
        let mut params: Vec<(&str, String)> = Vec::with_capacity(12);
        params.push(("apiKey", self.api_key.clone()));
        if let Some(cid) = new_client_order_id {
            params.push(("newClientOrderId", cid.to_string()));
        }
        if let Some(p) = price {
            params.push(("price", p.to_string()));
        }
        if let Some(qty) = quantity {
            params.push(("quantity", qty.to_string()));
        }
        params.push(("recvWindow", self.recv_window.to_string()));
        if let Some(ro) = reduce_only {
            params.push(("reduceOnly", if ro { "true" } else { "false" }.to_string()));
        }
        params.push(("side", side.to_uppercase()));
        params.push(("symbol", symbol.to_string()));
        if let Some(tif) = time_in_force {
            params.push(("timeInForce", tif.to_uppercase()));
        }
        params.push(("timestamp", ts.to_string()));
        params.push(("type", order_type.to_uppercase()));

        // Sort by key for signing
        params.sort_by(|a, b| a.0.cmp(&b.0));

        // Build query string for signing
        let query = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, url_encode_val(v)))
            .collect::<Vec<_>>()
            .join("&");

        let signature = hmac_sha256_hex(&self.api_secret, &query);

        // Build JSON-RPC message
        // Build params object as JSON manually for speed
        let mut json = String::with_capacity(512);
        json.push_str("{\"id\":\"");
        json.push_str(&req_id);
        json.push_str("\",\"method\":\"order.place\",\"params\":{");

        let mut first = true;
        for (k, v) in &params {
            if !first {
                json.push(',');
            }
            first = false;
            json.push('"');
            json.push_str(k);
            json.push_str("\":\"");
            json.push_str(v);
            json.push('"');
        }
        // Add signature
        json.push_str(",\"signature\":\"");
        json.push_str(&signature);
        json.push_str("\"}}");

        Ok((json, req_id))
    }

    /// Build and sign a cancel order message.
    #[pyo3(signature = (symbol, order_id=None, orig_client_order_id=None))]
    fn build_cancel_message(
        &self,
        symbol: &str,
        order_id: Option<i64>,
        orig_client_order_id: Option<&str>,
    ) -> PyResult<(String, String)> {
        let req_id = format!("cxl_{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed));
        let ts = now_ms();

        let mut params: Vec<(&str, String)> = Vec::with_capacity(8);
        params.push(("apiKey", self.api_key.clone()));
        if let Some(oid) = order_id {
            params.push(("orderId", oid.to_string()));
        }
        if let Some(cid) = orig_client_order_id {
            params.push(("origClientOrderId", cid.to_string()));
        }
        params.push(("recvWindow", self.recv_window.to_string()));
        params.push(("symbol", symbol.to_string()));
        params.push(("timestamp", ts.to_string()));

        params.sort_by(|a, b| a.0.cmp(&b.0));

        let query = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, url_encode_val(v)))
            .collect::<Vec<_>>()
            .join("&");

        let signature = hmac_sha256_hex(&self.api_secret, &query);

        let mut json = String::with_capacity(384);
        json.push_str("{\"id\":\"");
        json.push_str(&req_id);
        json.push_str("\",\"method\":\"order.cancel\",\"params\":{");

        let mut first = true;
        for (k, v) in &params {
            if !first {
                json.push(',');
            }
            first = false;
            json.push('"');
            json.push_str(k);
            json.push_str("\":\"");
            json.push_str(v);
            json.push('"');
        }
        json.push_str(",\"signature\":\"");
        json.push_str(&signature);
        json.push_str("\"}}");

        Ok((json, req_id))
    }

    /// Build a query order status message.
    #[pyo3(signature = (symbol, order_id=None, orig_client_order_id=None))]
    fn build_query_message(
        &self,
        symbol: &str,
        order_id: Option<i64>,
        orig_client_order_id: Option<&str>,
    ) -> PyResult<(String, String)> {
        let req_id = format!("qry_{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed));
        let ts = now_ms();

        let mut params: Vec<(&str, String)> = Vec::with_capacity(8);
        params.push(("apiKey", self.api_key.clone()));
        if let Some(oid) = order_id {
            params.push(("orderId", oid.to_string()));
        }
        if let Some(cid) = orig_client_order_id {
            params.push(("origClientOrderId", cid.to_string()));
        }
        params.push(("recvWindow", self.recv_window.to_string()));
        params.push(("symbol", symbol.to_string()));
        params.push(("timestamp", ts.to_string()));

        params.sort_by(|a, b| a.0.cmp(&b.0));

        let query = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, url_encode_val(v)))
            .collect::<Vec<_>>()
            .join("&");

        let signature = hmac_sha256_hex(&self.api_secret, &query);

        let mut json = String::with_capacity(384);
        json.push_str("{\"id\":\"");
        json.push_str(&req_id);
        json.push_str("\",\"method\":\"order.status\",\"params\":{");

        let mut first = true;
        for (k, v) in &params {
            if !first {
                json.push(',');
            }
            first = false;
            json.push('"');
            json.push_str(k);
            json.push_str("\":\"");
            json.push_str(v);
            json.push('"');
        }
        json.push_str(",\"signature\":\"");
        json.push_str(&signature);
        json.push_str("\"}}");

        Ok((json, req_id))
    }
}
