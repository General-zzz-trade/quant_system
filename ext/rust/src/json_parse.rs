use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

/// Parse Binance kline WebSocket JSON → Python dict.
///
/// Returns None if not a kline event or if only_closed and kline is not closed.
/// Prices are kept as strings (Decimal construction happens in Python).
///
/// Mirrors kline_processor.py process_raw().
#[pyfunction]
#[pyo3(signature = (raw, only_closed=true))]
pub fn rust_parse_kline<'py>(
    py: Python<'py>,
    raw: &str,
    only_closed: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let v: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    // Combined stream format: {"data": {...}} or direct
    let data = if v.get("data").is_some() {
        &v["data"]
    } else {
        &v
    };

    if data.get("e").and_then(|e| e.as_str()) != Some("kline") {
        return Ok(None);
    }

    let k = match data.get("k") {
        Some(k) if k.is_object() => k,
        _ => return Ok(None),
    };

    if only_closed {
        if k.get("x").and_then(|x| x.as_bool()) != Some(true) {
            return Ok(None);
        }
    }

    let ts_ms = match k.get("t").and_then(|t| t.as_i64()) {
        Some(t) => t,
        None => return Ok(None),
    };

    let symbol = data
        .get("s")
        .and_then(|s| s.as_str())
        .unwrap_or("");

    let dict = PyDict::new(py);
    dict.set_item("symbol", symbol.to_uppercase())?;
    dict.set_item("ts_ms", ts_ms)?;
    // Prices as strings for Decimal construction
    dict.set_item("open", json_str(k, "o"))?;
    dict.set_item("high", json_str(k, "h"))?;
    dict.set_item("low", json_str(k, "l"))?;
    dict.set_item("close", json_str(k, "c"))?;
    dict.set_item("volume", json_str(k, "v"))?;
    Ok(Some(dict))
}

/// Parse Binance depthUpdate WebSocket JSON → Python dict.
///
/// Returns dict with symbol, ts_ms, last_update_id, bids, asks.
/// Bids/asks are lists of [price_str, qty_str].
///
/// Mirrors depth_processor.py process_raw().
#[pyfunction]
#[pyo3(signature = (raw, max_levels=20))]
pub fn rust_parse_depth<'py>(
    py: Python<'py>,
    raw: &str,
    max_levels: usize,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let v: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let data = if v.get("data").is_some() {
        &v["data"]
    } else {
        &v
    };

    if data.get("e").and_then(|e| e.as_str()) != Some("depthUpdate") {
        return Ok(None);
    }

    let symbol = data.get("s").and_then(|s| s.as_str()).unwrap_or("");
    let ts_ms = data.get("E").and_then(|e| e.as_i64()).unwrap_or(0);
    let last_update_id = data.get("u").and_then(|u| u.as_i64()).unwrap_or(0);

    let bids = extract_levels(data.get("b"), max_levels);
    let asks = extract_levels(data.get("a"), max_levels);

    let dict = PyDict::new(py);
    dict.set_item("symbol", symbol)?;
    dict.set_item("ts_ms", ts_ms)?;
    dict.set_item("last_update_id", last_update_id)?;
    dict.set_item("bids", bids)?;
    dict.set_item("asks", asks)?;
    Ok(Some(dict))
}

/// Parse user data stream event → Python dict with event type + raw fields.
///
/// Simply extracts "e" (event type) and returns the full parsed dict
/// so Python-side mapper dispatch can route it.
#[pyfunction]
pub fn rust_demux_user_stream<'py>(
    py: Python<'py>,
    raw: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let v: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let data = if v.get("data").is_some() {
        &v["data"]
    } else {
        &v
    };

    let event_type = match data.get("e").and_then(|e| e.as_str()) {
        Some(e) => e,
        None => return Ok(None),
    };

    let dict = PyDict::new(py);
    dict.set_item("event_type", event_type)?;

    // Extract common fields
    if let Some(obj) = data.as_object() {
        for (k, val) in obj.iter() {
            if k == "e" {
                continue;
            }
            match val {
                Value::String(s) => { dict.set_item(k.as_str(), s.as_str())?; }
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        dict.set_item(k.as_str(), i)?;
                    } else if let Some(f) = n.as_f64() {
                        dict.set_item(k.as_str(), f)?;
                    }
                }
                Value::Bool(b) => { dict.set_item(k.as_str(), *b)?; }
                _ => {
                    // Nested objects/arrays: serialize back to JSON string
                    dict.set_item(k.as_str(), val.to_string())?;
                }
            }
        }
    }

    Ok(Some(dict))
}

/// Parse Binance aggTrade WebSocket JSON → Python dict.
///
/// Returns dict with symbol, price, qty, side, trade_id, ts_ms.
/// Prices/qty as strings for Decimal construction.
#[pyfunction]
pub fn rust_parse_agg_trade<'py>(
    py: Python<'py>,
    raw: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let v: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let data = if v.get("data").is_some() {
        &v["data"]
    } else {
        &v
    };

    if data.get("e").and_then(|e| e.as_str()) != Some("aggTrade") {
        return Ok(None);
    }

    let symbol = data.get("s").and_then(|s| s.as_str()).unwrap_or("");
    let price = data.get("p").and_then(|p| p.as_str()).unwrap_or("0");
    let qty = data.get("q").and_then(|q| q.as_str()).unwrap_or("0");
    let buyer_is_maker = data.get("m").and_then(|m| m.as_bool()).unwrap_or(false);
    let side = if buyer_is_maker { "sell" } else { "buy" };
    let trade_id = data.get("a").and_then(|a| a.as_i64()).unwrap_or(0);
    let ts_ms = data.get("T").and_then(|t| t.as_i64()).unwrap_or(0);

    let dict = PyDict::new(py);
    dict.set_item("symbol", symbol)?;
    dict.set_item("price", price)?;
    dict.set_item("qty", qty)?;
    dict.set_item("side", side)?;
    dict.set_item("trade_id", trade_id)?;
    dict.set_item("ts_ms", ts_ms)?;
    Ok(Some(dict))
}

// ── helpers ──

fn json_str(obj: &Value, key: &str) -> String {
    obj.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("0")
        .to_owned()
}

fn extract_levels(arr: Option<&Value>, max_levels: usize) -> Vec<(String, String)> {
    match arr.and_then(|a| a.as_array()) {
        Some(levels) => levels
            .iter()
            .take(max_levels)
            .filter_map(|level| {
                let a = level.as_array()?;
                let price = a.first()?.as_str()?.to_owned();
                let qty = a.get(1)?.as_str()?.to_owned();
                Some((price, qty))
            })
            .collect(),
        None => Vec::new(),
    }
}
