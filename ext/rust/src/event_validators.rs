use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

// ============================================================
// rust_validate_monotonic_time
// ============================================================
// Takes a list of ISO timestamp strings (or None), returns indices
// where the timestamp goes backwards (non-monotonic).
// Uses simple lexicographic comparison on ISO strings for speed;
// this is correct for ISO 8601 with consistent formatting.

#[pyfunction]
pub fn rust_validate_monotonic_time(events_ts_list: Vec<Option<String>>) -> Vec<usize> {
    let mut violations = Vec::new();
    let mut last_ts: Option<&str> = None;
    // We need owned strings for lifetime, so collect refs differently
    let mut last_owned: Option<String> = None;

    for (i, ts_opt) in events_ts_list.iter().enumerate() {
        match ts_opt {
            None => {
                // Skip None entries — they don't participate in monotonicity
            }
            Some(ts) => {
                if let Some(ref prev) = last_owned {
                    if ts.as_str() < prev.as_str() {
                        violations.push(i);
                    }
                }
                last_owned = Some(ts.clone());
            }
        }
    }

    violations
}

// ============================================================
// rust_validate_required_fields
// ============================================================
// Takes an event as a dict and a list of required field names.
// Returns list of missing field names.

#[pyfunction]
pub fn rust_validate_required_fields(
    event_dict: &Bound<'_, PyDict>,
    required: Vec<String>,
) -> PyResult<Vec<String>> {
    let mut missing = Vec::new();
    for field in &required {
        match event_dict.get_item(field)? {
            None => missing.push(field.clone()),
            Some(val) => {
                if val.is_none() {
                    missing.push(field.clone());
                }
            }
        }
    }
    Ok(missing)
}

// ============================================================
// rust_validate_numeric_range
// ============================================================
// Check if value is within [min_val, max_val].

#[pyfunction]
pub fn rust_validate_numeric_range(value: f64, min_val: f64, max_val: f64) -> bool {
    value >= min_val && value <= max_val
}

// ============================================================
// rust_validate_enum_value
// ============================================================
// Check if value is in the allowed list.

#[pyfunction]
pub fn rust_validate_enum_value(value: &str, allowed: Vec<String>) -> bool {
    allowed.iter().any(|a| a == value)
}

// ============================================================
// rust_validate_event_batch
// ============================================================
// Validates a batch of event dicts: checks required fields,
// monotonic timestamps, and enum constraints in one pass.
// Returns a dict of { index: [error_strings] }.

#[pyfunction]
#[pyo3(signature = (events, required_fields, ts_field="ts".to_string(), enum_checks=None))]
pub fn rust_validate_event_batch<'py>(
    py: Python<'py>,
    events: &Bound<'py, PyList>,
    required_fields: Vec<String>,
    ts_field: String,
    enum_checks: Option<HashMap<String, Vec<String>>>,
) -> PyResult<Bound<'py, PyDict>> {
    let errors = PyDict::new(py);
    let mut last_ts: Option<String> = None;

    for i in 0..events.len() {
        let item = events.get_item(i)?;
        let dict: &Bound<'_, PyDict> = item.downcast()?;
        let mut errs: Vec<String> = Vec::new();

        // Check required fields
        for field in &required_fields {
            match dict.get_item(field)? {
                None => errs.push(format!("missing field: {}", field)),
                Some(val) => {
                    if val.is_none() {
                        errs.push(format!("null field: {}", field));
                    }
                }
            }
        }

        // Check monotonic timestamp
        if let Some(ts_val) = dict.get_item(&ts_field)? {
            if !ts_val.is_none() {
                let ts_str: String = ts_val.extract().unwrap_or_default();
                if !ts_str.is_empty() {
                    if let Some(ref prev) = last_ts {
                        if ts_str.as_str() < prev.as_str() {
                            errs.push(format!(
                                "non-monotonic ts: {} < {}",
                                ts_str, prev
                            ));
                        }
                    }
                    last_ts = Some(ts_str);
                }
            }
        }

        // Check enum constraints
        if let Some(ref checks) = enum_checks {
            for (field, allowed) in checks {
                if let Some(val) = dict.get_item(field)? {
                    if !val.is_none() {
                        if let Ok(s) = val.extract::<String>() {
                            if !allowed.contains(&s) {
                                errs.push(format!(
                                    "invalid {} '{}', allowed: {:?}",
                                    field, s, allowed
                                ));
                            }
                        }
                    }
                }
            }
        }

        if !errs.is_empty() {
            errors.set_item(i, errs)?;
        }
    }

    Ok(errors)
}

// ============================================================
// rust_validate_side
// ============================================================
// Convenience: check if side is "buy" or "sell".

#[pyfunction]
pub fn rust_validate_side(side: &str) -> bool {
    side == "buy" || side == "sell"
}

// ============================================================
// rust_validate_signal_side
// ============================================================
// Convenience: check if side is "buy", "sell", or "flat".

#[pyfunction]
pub fn rust_validate_signal_side(side: &str) -> bool {
    side == "buy" || side == "sell" || side == "flat"
}

// ============================================================
// rust_validate_venue
// ============================================================

#[pyfunction]
pub fn rust_validate_venue(venue: &str) -> bool {
    venue == "BINANCE" || venue == "BYBIT" || venue == "SIM"
}

// ============================================================
// rust_validate_order_type
// ============================================================

#[pyfunction]
pub fn rust_validate_order_type(order_type: &str) -> bool {
    order_type == "market"
        || order_type == "limit"
        || order_type == "stop"
        || order_type == "stop_limit"
}

// ============================================================
// rust_validate_tif
// ============================================================

#[pyfunction]
pub fn rust_validate_tif(tif: &str) -> bool {
    tif == "GTC" || tif == "IOC" || tif == "FOK" || tif == "GTX"
}
