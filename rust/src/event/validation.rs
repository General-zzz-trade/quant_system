use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::collections::VecDeque;

// ============================================================
// BoundedDedup — LRU-like bounded dedup set
// ============================================================
// Uses HashMap + VecDeque for O(1) lookup with bounded memory.
// Prevents the unbounded memory leak identified in P1-15.

struct BoundedDedup {
    map: HashMap<String, ()>,
    order: VecDeque<String>,
    capacity: usize,
}

impl BoundedDedup {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity.min(65536)),
            order: VecDeque::with_capacity(capacity.min(65536)),
            capacity,
        }
    }

    /// Returns true if the ID was already seen (duplicate).
    fn check_and_insert(&mut self, id: &str) -> bool {
        if self.map.contains_key(id) {
            return true; // duplicate
        }
        if self.order.len() >= self.capacity {
            // Evict oldest entry
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.map.insert(id.to_string(), ());
        self.order.push_back(id.to_string());
        false // new
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

// ============================================================
// RustEventValidator — stateful event validator (PyO3 class)
// ============================================================
// Single validate() call performs:
//   1. Header extraction (event_id, timestamp, stream)
//   2. Timestamp positivity + finiteness
//   3. Bounded LRU dedup (prevents memory leak)
//   4. Per-stream monotonic time check
//   5. Type-specific field validation

#[pyclass(name = "RustEventValidator")]
pub struct RustEventValidator {
    stream_timestamps: HashMap<String, f64>,
    seen_ids: BoundedDedup,
}

#[pymethods]
impl RustEventValidator {
    #[new]
    #[pyo3(signature = (max_seen=100000))]
    fn new(max_seen: usize) -> Self {
        Self {
            stream_timestamps: HashMap::new(),
            seen_ids: BoundedDedup::new(max_seen),
        }
    }

    /// Validate an event dict. Raises ValueError on failure.
    /// Checks: required fields, dedup, monotonic time, type-specific validation.
    fn validate(&mut self, event_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        // 1. Extract header — try as dict first, fallback to object with attributes
        let (event_id, timestamp, stream) = self.extract_header(event_dict)?;

        // 2. Validate timestamp is positive and finite
        if !timestamp.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "timestamp must be finite, got {}",
                timestamp
            )));
        }
        if timestamp <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "timestamp must be positive, got {}",
                timestamp
            )));
        }

        // 3. Dedup check using bounded LRU
        if self.seen_ids.check_and_insert(&event_id) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "duplicate event: {}",
                event_id
            )));
        }

        // 4. Stream monotonic time check
        if !stream.is_empty() {
            if let Some(&last_ts) = self.stream_timestamps.get(&stream) {
                if timestamp < last_ts {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "non-monotonic timestamp: {} < {} for stream '{}'",
                        timestamp, last_ts, stream
                    )));
                }
            }
            self.stream_timestamps.insert(stream, timestamp);
        }

        // 5. Type-specific validation
        if let Some(event_type_val) = event_dict.get_item("event_type")? {
            if let Ok(event_type_str) = event_type_val.extract::<String>() {
                self.validate_by_type(&event_type_str, event_dict)?;
            }
        }

        Ok(())
    }

    /// Type-specific validation dispatched by event_type string.
    fn validate_by_type(&self, event_type: &str, data: &Bound<'_, PyDict>) -> PyResult<()> {
        match event_type.to_lowercase().as_str() {
            "market" => {
                // Validate close, volume are finite if present
                self.validate_finite_field(data, "close")?;
                self.validate_finite_field(data, "volume")?;
            }
            "fill" => {
                // Validate qty > 0, price > 0, both finite
                self.validate_positive_field(data, "qty")?;
                self.validate_positive_field(data, "price")?;
            }
            "order" => {
                // Validate symbol non-empty, qty > 0
                self.validate_non_empty_str(data, "symbol")?;
                self.validate_positive_field(data, "qty")?;
            }
            "risk" => {
                // Soft check: verify "decision" key exists if present
                // (no-op for now — risk events pass through)
            }
            _ => {
                // Unknown types pass through without type-specific checks
            }
        }
        Ok(())
    }

    /// Number of event IDs currently tracked in the dedup set.
    fn seen_count(&self) -> usize {
        self.seen_ids.len()
    }

    /// Number of distinct streams being tracked for monotonic time.
    fn stream_count(&self) -> usize {
        self.stream_timestamps.len()
    }

    /// Clear the dedup seen-set (does not affect stream timestamps).
    fn clear_seen(&mut self) {
        self.seen_ids.clear();
    }

    /// Clear stream timestamp tracking (does not affect dedup).
    fn clear_streams(&mut self) {
        self.stream_timestamps.clear();
    }

    /// Clear all internal state (dedup + streams).
    fn clear_all(&mut self) {
        self.seen_ids.clear();
        self.stream_timestamps.clear();
    }
}

// ============================================================
// Private helpers (not exposed to Python)
// ============================================================

impl RustEventValidator {
    /// Extract (event_id, timestamp, stream) from event dict.
    /// Tries "header" key as a dict first, then as object with attributes.
    /// Falls back to top-level keys if no header sub-object.
    fn extract_header(&self, event_dict: &Bound<'_, PyDict>) -> PyResult<(String, f64, String)> {
        // Try to get "header" from the dict
        if let Some(header_val) = event_dict.get_item("header")? {
            // Try header as a dict
            if let Ok(header_dict) = header_val.downcast::<PyDict>() {
                let event_id = self.extract_string_from_dict(header_dict, "event_id")?;
                let timestamp = self.extract_f64_from_dict(header_dict, "timestamp")?;
                let stream = self.extract_optional_string_from_dict(header_dict, "stream")?;
                return Ok((event_id, timestamp, stream));
            }

            // Try header as object with attributes
            if let Ok(eid) = header_val.getattr("event_id") {
                let event_id: String = eid.extract().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "header.event_id must be a string",
                    )
                })?;
                let timestamp: f64 = header_val
                    .getattr("timestamp")
                    .and_then(|v| v.extract())
                    .map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(
                            "header.timestamp must be a float",
                        )
                    })?;
                let stream = header_val
                    .getattr("stream")
                    .and_then(|v| v.extract::<String>())
                    .unwrap_or_default();
                return Ok((event_id, timestamp, stream));
            }
        }

        // Fallback: try top-level keys
        let event_id = self.extract_string_from_dict(
            event_dict.downcast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("event must be a dict")
            })?,
            "event_id",
        )?;
        let timestamp = self.extract_f64_from_dict(
            event_dict.downcast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("event must be a dict")
            })?,
            "timestamp",
        )?;
        let stream = self.extract_optional_string_from_dict(
            event_dict.downcast::<PyDict>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("event must be a dict")
            })?,
            "stream",
        )?;
        Ok((event_id, timestamp, stream))
    }

    /// Extract a required string field from a PyDict.
    fn extract_string_from_dict(
        &self,
        d: &Bound<'_, PyDict>,
        key: &str,
    ) -> PyResult<String> {
        match d.get_item(key)? {
            Some(val) => val.extract::<String>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "'{}' must be a string",
                    key
                ))
            }),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "missing required field: '{}'",
                key
            ))),
        }
    }

    /// Extract a required f64 field from a PyDict.
    fn extract_f64_from_dict(
        &self,
        d: &Bound<'_, PyDict>,
        key: &str,
    ) -> PyResult<f64> {
        match d.get_item(key)? {
            Some(val) => val.extract::<f64>().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "'{}' must be a number",
                    key
                ))
            }),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "missing required field: '{}'",
                key
            ))),
        }
    }

    /// Extract an optional string field from a PyDict, defaulting to "".
    fn extract_optional_string_from_dict(
        &self,
        d: &Bound<'_, PyDict>,
        key: &str,
    ) -> PyResult<String> {
        match d.get_item(key)? {
            Some(val) => {
                if val.is_none() {
                    Ok(String::new())
                } else {
                    Ok(val.extract::<String>().unwrap_or_default())
                }
            }
            None => Ok(String::new()),
        }
    }

    /// Validate that a numeric field (if present) is finite.
    fn validate_finite_field(&self, d: &Bound<'_, PyDict>, key: &str) -> PyResult<()> {
        if let Some(val) = d.get_item(key)? {
            if !val.is_none() {
                let v: f64 = val.extract().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{} must be numeric",
                        key
                    ))
                })?;
                if !v.is_finite() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{} must be finite, got {}",
                        key, v
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate that a numeric field (if present) is positive and finite.
    fn validate_positive_field(&self, d: &Bound<'_, PyDict>, key: &str) -> PyResult<()> {
        if let Some(val) = d.get_item(key)? {
            if !val.is_none() {
                let v: f64 = val.extract().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{} must be numeric",
                        key
                    ))
                })?;
                if !v.is_finite() || v <= 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{} must be positive finite, got {}",
                        key, v
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate that a string field (if present) is non-empty.
    fn validate_non_empty_str(&self, d: &Bound<'_, PyDict>, key: &str) -> PyResult<()> {
        if let Some(val) = d.get_item(key)? {
            if !val.is_none() {
                let s: String = val.extract().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "{} must be a string",
                        key
                    ))
                })?;
                if s.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{} must not be empty",
                        key
                    )));
                }
            }
        }
        Ok(())
    }
}
