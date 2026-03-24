use pyo3::prelude::*;

use crate::state::fixed_decimal::Fd8;

/// Pure Rust result type — no PyObject allocation.
pub struct InnerReducerResult<S> {
    pub state: S,
    pub changed: bool,
    pub note: Option<String>,
}

// ===========================================================================
// Helper functions
// ===========================================================================

pub fn get_event_type(event: &Bound<'_, PyAny>) -> PyResult<String> {
    let raw = if let Ok(header) = event.getattr("header") {
        if !header.is_none() {
            if let Ok(et) = header.getattr("event_type") {
                if !et.is_none() {
                    Some(et.into())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let raw: Bound<'_, PyAny> = match raw {
        Some(r) => r,
        None => match event.getattr("event_type") {
            Ok(et) if !et.is_none() => et,
            _ => return Ok(String::new()),
        },
    };

    let val = if let Ok(v) = raw.getattr("value") {
        v
    } else if let Ok(n) = raw.getattr("name") {
        n
    } else {
        raw
    };

    let s: String = val.str()?.to_string();
    Ok(s.trim().to_lowercase())
}

pub fn get_symbol(event: &Bound<'_, PyAny>, default: &str) -> PyResult<String> {
    if let Ok(sym) = event.getattr("symbol") {
        if !sym.is_none() {
            return Ok(sym.str()?.to_string());
        }
    }
    if let Ok(bar) = event.getattr("bar") {
        if !bar.is_none() {
            if let Ok(sym) = bar.getattr("symbol") {
                if !sym.is_none() {
                    return Ok(sym.str()?.to_string());
                }
            }
        }
    }
    Ok(default.to_string())
}

pub fn get_event_ts(event: &Bound<'_, PyAny>) -> PyResult<Option<PyObject>> {
    if let Ok(header) = event.getattr("header") {
        if !header.is_none() {
            if let Ok(ts) = header.getattr("ts") {
                if !ts.is_none() {
                    return Ok(Some(ts.unbind()));
                }
            }
        }
    }
    if let Ok(ts) = event.getattr("ts") {
        if !ts.is_none() {
            return Ok(Some(ts.unbind()));
        }
    }
    Ok(None)
}

pub fn ts_to_opt_string(py: Python<'_>, ts: &Option<PyObject>) -> PyResult<Option<String>> {
    match ts {
        Some(t) => {
            let bound = t.bind(py);
            if bound.is_none() {
                Ok(None)
            } else {
                Ok(Some(bound.str()?.to_string()))
            }
        }
        None => Ok(None),
    }
}

/// Apply side to qty: buy/long -> +abs, sell/short -> -abs.
pub fn signed_qty(qty: Fd8, side: &str) -> Fd8 {
    let s = side.trim().to_lowercase();
    match s.as_str() {
        "buy" | "long" => qty.abs(),
        "sell" | "short" => -qty.abs(),
        _ => qty,
    }
}
