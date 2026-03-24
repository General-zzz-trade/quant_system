use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// Generate a UUID v4 string. Faster than Python's uuid.uuid4() + str().
#[pyfunction]
pub fn rust_event_id() -> String {
    Uuid::new_v4().to_string()
}

/// Current time as nanoseconds since epoch. Matches Python's int(time.time() * 1e9).
#[pyfunction]
pub fn rust_now_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64
}
