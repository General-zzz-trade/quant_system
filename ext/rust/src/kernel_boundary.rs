use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Deserialize, Serialize)]
struct KernelHeader {
    event_id: String,
    event_type: String,
    version: i64,
    ts_ns: i64,
    source: String,
    parent_event_id: Option<String>,
    root_event_id: Option<String>,
    run_id: Option<String>,
    seq: Option<i64>,
    correlation_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct KernelEventEnvelope {
    event_type: String,
    #[serde(default)]
    header: Option<KernelHeader>,
    #[serde(default)]
    attrs: Value,
}

fn detect_kind_str(raw: &str) -> String {
    let u = raw.to_ascii_uppercase();
    if u.contains("MARKET") {
        return "MARKET".to_string();
    }
    if u.contains("FILL") {
        return "FILL".to_string();
    }
    if u.contains("FUNDING") {
        return "FUNDING".to_string();
    }
    if u.contains("ORDER") {
        return "ORDER".to_string();
    }
    if u.contains("INTENT") {
        return "INTENT".to_string();
    }
    if u.contains("SIGNAL") {
        return "SIGNAL".to_string();
    }
    if u.contains("RISK") {
        return "RISK".to_string();
    }
    if u.contains("CONTROL") {
        return "CONTROL".to_string();
    }
    "UNKNOWN".to_string()
}

fn detect_kind_env(env: &KernelEventEnvelope) -> String {
    if !env.event_type.is_empty() {
        return detect_kind_str(&env.event_type);
    }
    if let Some(header) = &env.header {
        if !header.event_type.is_empty() {
            return detect_kind_str(&header.event_type);
        }
    }
    "UNKNOWN".to_string()
}

fn attr<'a>(env: &'a KernelEventEnvelope, key: &str) -> Option<&'a Value> {
    env.attrs.get(key)
}

fn attr_str(env: &KernelEventEnvelope, key: &str) -> Option<String> {
    attr(env, key).and_then(|v| {
        if v.is_null() {
            None
        } else if let Some(s) = v.as_str() {
            Some(s.to_string())
        } else {
            Some(v.to_string())
        }
    })
}

fn attr_f64(env: &KernelEventEnvelope, key: &str, default: f64) -> f64 {
    match attr(env, key) {
        Some(v) if v.is_number() => v.as_f64().unwrap_or(default),
        Some(v) if v.is_string() => v.as_str().unwrap_or("").parse::<f64>().unwrap_or(default),
        _ => default,
    }
}

fn header_value(env: &KernelEventEnvelope) -> Value {
    match &env.header {
        Some(h) => serde_json::to_value(h).unwrap_or(Value::Null),
        None => Value::Null,
    }
}

fn parse_env(payload_json: &str) -> PyResult<KernelEventEnvelope> {
    serde_json::from_str(payload_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid kernel event payload: {}", e))
    })
}

#[pyfunction]
#[pyo3(signature = (payload_json))]
pub fn rust_detect_kernel_event_kind(payload_json: &str) -> PyResult<String> {
    let env = parse_env(payload_json)?;
    Ok(detect_kind_env(&env))
}

#[pyfunction]
#[pyo3(signature = (payload_json))]
pub fn rust_normalize_kernel_event_to_facts(payload_json: &str) -> PyResult<String> {
    let env = parse_env(payload_json)?;
    let kind = detect_kind_env(&env);
    let header = header_value(&env);

    let facts = match kind.as_str() {
        "MARKET" => vec![json!({
            "event_type": "market",
            "header": header,
            "symbol": attr_str(&env, "symbol"),
            "open": attr(&env, "open").cloned().unwrap_or(Value::Null),
            "high": attr(&env, "high").cloned().unwrap_or(Value::Null),
            "low": attr(&env, "low").cloned().unwrap_or(Value::Null),
            "close": attr(&env, "close").cloned().unwrap_or(Value::Null),
            "volume": attr(&env, "volume").cloned().unwrap_or(Value::Null),
            "ts": attr(&env, "ts").cloned().unwrap_or(Value::Null),
        })],
        "FUNDING" => vec![json!({
            "event_type": "funding",
            "header": header,
            "symbol": attr_str(&env, "symbol"),
            "funding_rate": attr(&env, "funding_rate").cloned().unwrap_or(Value::Null),
            "mark_price": attr(&env, "mark_price").cloned().unwrap_or(Value::Null),
            "position_qty": attr(&env, "position_qty").cloned().unwrap_or(Value::Null),
            "ts": attr(&env, "ts").cloned().unwrap_or(Value::Null),
        })],
        "ORDER" => vec![json!({
            "event_type": "ORDER_UPDATE",
            "header": header,
            "symbol": attr_str(&env, "symbol"),
            "venue": attr(&env, "venue").cloned().unwrap_or(Value::Null),
            "order_id": attr(&env, "order_id").cloned().unwrap_or(Value::Null),
            "client_order_id": attr(&env, "client_order_id").cloned().unwrap_or(Value::Null),
            "status": attr(&env, "status").cloned().unwrap_or(Value::Null),
            "side": attr(&env, "side").cloned().unwrap_or(Value::Null),
            "order_type": attr(&env, "order_type").cloned().unwrap_or(Value::Null),
            "tif": attr(&env, "tif").cloned().unwrap_or(Value::Null),
            "qty": attr(&env, "qty").cloned().unwrap_or(Value::Null),
            "price": attr(&env, "price").cloned().unwrap_or(Value::Null),
            "filled_qty": attr(&env, "filled_qty").cloned().unwrap_or(Value::Null),
            "avg_price": attr(&env, "avg_price").cloned().unwrap_or(Value::Null),
            "order_key": attr(&env, "order_key").cloned().unwrap_or(Value::Null),
            "payload_digest": attr(&env, "payload_digest").cloned().unwrap_or(Value::Null),
        })],
        "FILL" => {
            let side_raw = attr_str(&env, "side").ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("FILL fact missing side")
            })?;
            let side_norm = match side_raw.trim().to_ascii_lowercase().as_str() {
                "buy" | "long" => "buy",
                "sell" | "short" => "sell",
                other => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "unsupported fill side: {}",
                        other
                    )))
                }
            };
            let qty = match attr(&env, "qty").or_else(|| attr(&env, "quantity")) {
                Some(v) => {
                    if v.is_number() {
                        v.as_f64().unwrap_or(0.0).abs()
                    } else if v.is_string() {
                        v.as_str().unwrap_or("").parse::<f64>().unwrap_or(0.0).abs()
                    } else {
                        0.0
                    }
                }
                None => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "FILL fact missing qty/quantity",
                    ))
                }
            };
            vec![json!({
                "event_type": "fill",
                "header": header,
                "symbol": attr_str(&env, "symbol"),
                "side": side_norm,
                "qty": qty,
                "quantity": qty,
                "price": attr_f64(&env, "price", 0.0),
                "fee": attr_f64(&env, "fee", 0.0),
                "realized_pnl": attr_f64(&env, "realized_pnl", 0.0),
                "margin_change": attr_f64(&env, "margin_change", 0.0),
                "cash_delta": attr_f64(&env, "cash_delta", 0.0),
                "ts": attr(&env, "ts").cloned().unwrap_or(Value::Null),
            })]
        }
        _ => Vec::new(),
    };

    serde_json::to_string(&facts).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("failed to encode kernel facts: {}", e))
    })
}
