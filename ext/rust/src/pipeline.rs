use pyo3::prelude::*;
use pyo3::types::PyString;

// ---------------------------------------------------------------------------
// Helper: read an attribute from a Python object, returning None on failure
// ---------------------------------------------------------------------------
fn getattr_opt<'py>(obj: &Bound<'py, PyAny>, name: &str) -> Option<Bound<'py, PyAny>> {
    obj.getattr(name).ok().and_then(|v| {
        if v.is_none() {
            None
        } else {
            Some(v)
        }
    })
}

fn getattr_or_none<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>, name: &str) -> PyObject {
    match obj.getattr(name) {
        Ok(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        Err(_) => py.None(),
    }
}

fn to_float(py: Python<'_>, obj: &Bound<'_, PyAny>, attr: &str, default: f64) -> f64 {
    match obj.getattr(attr) {
        Ok(v) => {
            if v.is_none() {
                return default;
            }
            match v.extract::<f64>() {
                Ok(f) => f,
                Err(_) => {
                    // Try str -> float
                    match v.str().and_then(|s| s.to_str().map(|s| s.to_string())) {
                        Ok(s) => s.parse::<f64>().unwrap_or(default),
                        Err(_) => default,
                    }
                }
            }
        }
        Err(_) => default,
    }
}

// ---------------------------------------------------------------------------
// _detect_kind: classify an event by its event_type / EVENT_TYPE attrs
// ---------------------------------------------------------------------------
fn detect_kind_inner(event: &Bound<'_, PyAny>) -> String {
    // Style A: event.event_type (may be Enum with .value or plain string)
    if let Ok(et) = event.getattr("event_type") {
        if !et.is_none() {
            // Try .value first (Enum), then str()
            let et_val = if let Ok(v) = et.getattr("value") {
                v
            } else {
                et
            };
            if let Ok(s) = et_val.str() {
                if let Ok(st) = s.to_str() {
                    let u = st.to_ascii_uppercase();
                    if u.contains("MARKET") { return "MARKET".to_string(); }
                    if u.contains("FILL") { return "FILL".to_string(); }
                    if u.contains("FUNDING") { return "FUNDING".to_string(); }
                    if u.contains("ORDER") { return "ORDER".to_string(); }
                    if u.contains("INTENT") { return "INTENT".to_string(); }
                    if u.contains("SIGNAL") { return "SIGNAL".to_string(); }
                    if u.contains("RISK") { return "RISK".to_string(); }
                    if u.contains("CONTROL") { return "CONTROL".to_string(); }
                }
            }
        }
    }

    // Style B: EVENT_TYPE class attribute (string)
    if let Ok(name) = event.getattr("EVENT_TYPE") {
        if let Ok(s) = name.extract::<String>() {
            if !s.is_empty() {
                let n = s.to_ascii_lowercase();
                if n.contains("market") { return "MARKET".to_string(); }
                if n.contains("fill") { return "FILL".to_string(); }
                if n.contains("funding") { return "FUNDING".to_string(); }
                if n.contains("order") { return "ORDER".to_string(); }
                if n.contains("intent") { return "INTENT".to_string(); }
                if n.contains("signal") { return "SIGNAL".to_string(); }
                if n.contains("risk") { return "RISK".to_string(); }
                if n.contains("control") { return "CONTROL".to_string(); }
            }
        }
    }

    "UNKNOWN".to_string()
}

/// Detect the kind of an event from its event_type / EVENT_TYPE attributes.
///
/// Returns one of: "MARKET", "FILL", "ORDER", "SIGNAL", "INTENT",
///                 "RISK", "CONTROL", "FUNDING", "UNKNOWN"
#[pyfunction]
#[pyo3(signature = (event))]
pub fn rust_detect_event_kind(event: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(detect_kind_inner(event))
}

// ---------------------------------------------------------------------------
// Helper: build a SimpleNamespace from Python
// ---------------------------------------------------------------------------
fn make_namespace(py: Python<'_>, attrs: Vec<(&str, PyObject)>) -> PyResult<PyObject> {
    let types = py.import("types")?;
    let ns_cls = types.getattr("SimpleNamespace")?;
    let kwargs = pyo3::types::PyDict::new(py);
    for (k, v) in attrs {
        kwargs.set_item(k, v)?;
    }
    let ns = ns_cls.call((), Some(&kwargs))?;
    Ok(ns.unbind())
}

// ---------------------------------------------------------------------------
// normalize_to_facts: event -> list of SimpleNamespace fact events
// ---------------------------------------------------------------------------

/// Normalize a raw event into a list of SimpleNamespace fact events that
/// reducers can consume. Returns an empty list for non-fact event kinds.
///
/// This is the hot path called on every incoming event.
#[pyfunction]
#[pyo3(signature = (event))]
pub fn rust_normalize_to_facts(py: Python<'_>, event: &Bound<'_, PyAny>) -> PyResult<Vec<PyObject>> {
    let kind = detect_kind_inner(event);
    let header = getattr_or_none(py, event, "header");

    match kind.as_str() {
        "MARKET" => {
            let ns = make_namespace(py, vec![
                ("event_type", PyString::new(py, "market").into_any().unbind()),
                ("header", header),
                ("symbol", getattr_or_none(py, event, "symbol")),
                ("open", getattr_or_none(py, event, "open")),
                ("high", getattr_or_none(py, event, "high")),
                ("low", getattr_or_none(py, event, "low")),
                ("close", getattr_or_none(py, event, "close")),
                ("volume", getattr_or_none(py, event, "volume")),
                ("ts", getattr_or_none(py, event, "ts")),
            ])?;
            Ok(vec![ns])
        }

        "FUNDING" => {
            let ns = make_namespace(py, vec![
                ("event_type", PyString::new(py, "funding").into_any().unbind()),
                ("header", header),
                ("symbol", getattr_or_none(py, event, "symbol")),
                ("funding_rate", getattr_or_none(py, event, "funding_rate")),
                ("mark_price", getattr_or_none(py, event, "mark_price")),
                ("ts", getattr_or_none(py, event, "ts")),
            ])?;
            Ok(vec![ns])
        }

        "ORDER" => {
            let ns = make_namespace(py, vec![
                ("event_type", PyString::new(py, "ORDER_UPDATE").into_any().unbind()),
                ("header", header),
                ("symbol", getattr_or_none(py, event, "symbol")),
                ("venue", getattr_or_none(py, event, "venue")),
                ("order_id", getattr_or_none(py, event, "order_id")),
                ("client_order_id", getattr_or_none(py, event, "client_order_id")),
                ("status", getattr_or_none(py, event, "status")),
                ("side", getattr_or_none(py, event, "side")),
                ("order_type", getattr_or_none(py, event, "order_type")),
                ("tif", getattr_or_none(py, event, "tif")),
                ("qty", getattr_or_none(py, event, "qty")),
                ("price", getattr_or_none(py, event, "price")),
                ("filled_qty", getattr_or_none(py, event, "filled_qty")),
                ("avg_price", getattr_or_none(py, event, "avg_price")),
                ("order_key", getattr_or_none(py, event, "order_key")),
                ("payload_digest", getattr_or_none(py, event, "payload_digest")),
            ])?;
            Ok(vec![ns])
        }

        "FILL" => {
            // Get qty: try .qty first, then .quantity
            let raw_qty_obj = match getattr_opt(event, "qty") {
                Some(v) => v,
                None => match getattr_opt(event, "quantity") {
                    Some(v) => v,
                    None => return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "FILL event missing qty/quantity"
                    )),
                },
            };

            // Get side (required)
            let side_obj = match getattr_opt(event, "side") {
                Some(v) => v,
                None => return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "FILL 事实事件缺少 side"
                )),
            };

            // Normalize side: check .value (Enum), then str
            let side_raw = if let Ok(v) = side_obj.getattr("value") {
                v
            } else {
                side_obj
            };
            let side_str = side_raw.str()?.to_string().trim().to_lowercase();
            let side_norm = match side_str.as_str() {
                "buy" | "long" => "buy",
                "sell" | "short" => "sell",
                _ => return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("不支持的 fill side: {:?}", side_str)
                )),
            };

            // Parse qty as float, take abs
            let qty_f: f64 = raw_qty_obj.extract::<f64>()
                .or_else(|_| {
                    raw_qty_obj.str()
                        .and_then(|s| s.to_str().map(|s| s.to_string()))
                        .map(|s| s.parse::<f64>().unwrap_or(0.0))
                })
                .unwrap_or(0.0)
                .abs();

            let price_f = to_float(py, event, "price", 0.0);
            let fee_f = to_float(py, event, "fee", 0.0);
            let realized_pnl_f = to_float(py, event, "realized_pnl", 0.0);
            let margin_change_f = to_float(py, event, "margin_change", 0.0);

            let ns = make_namespace(py, vec![
                ("event_type", PyString::new(py, "FILL").into_any().unbind()),
                ("header", header),
                ("symbol", getattr_or_none(py, event, "symbol")),
                ("side", PyString::new(py, side_norm).into_any().unbind()),
                ("qty", qty_f.into_pyobject(py)?.into_any().unbind()),
                ("quantity", qty_f.into_pyobject(py)?.into_any().unbind()),
                ("price", price_f.into_pyobject(py)?.into_any().unbind()),
                ("fee", fee_f.into_pyobject(py)?.into_any().unbind()),
                ("realized_pnl", realized_pnl_f.into_pyobject(py)?.into_any().unbind()),
                ("margin_change", margin_change_f.into_pyobject(py)?.into_any().unbind()),
            ])?;
            Ok(vec![ns])
        }

        _ => Ok(vec![]),
    }
}
