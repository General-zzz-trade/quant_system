// core_types.rs — Sprint C: Core types ported to Rust PyO3
//
// Contains: RustInterceptorChain, RustSystemClock, RustSimulatedClock, RustTradingGate
//
// Python sources:
//   core/interceptors.py, core/clock.py, policy/gating.py

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::time::Instant;

// ============================================================
// RustInterceptorChain
// ============================================================

#[pyclass]
pub struct RustInterceptorChain {
    interceptors: Vec<PyObject>,
}

#[pymethods]
impl RustInterceptorChain {
    #[new]
    fn new() -> Self {
        Self {
            interceptors: Vec::new(),
        }
    }

    /// Add a Python callable interceptor.
    /// The callable should accept (event, context) and return ("continue"|"reject"|"kill", reason_str).
    fn add(&mut self, interceptor: PyObject) {
        self.interceptors.push(interceptor);
    }

    /// Run all interceptors. Stops on first "reject" or "kill".
    /// Returns (action, reason) where action is "continue", "reject", or "kill".
    fn run(&self, py: Python<'_>, event: PyObject, context: PyObject) -> PyResult<(String, String)> {
        for interceptor in &self.interceptors {
            let result = interceptor.call1(py, (event.clone_ref(py), context.clone_ref(py)))?;

            // Result should be a tuple (action_str, reason_str)
            let tuple = result.downcast_bound::<PyTuple>(py)?;
            let action: String = tuple.get_item(0)?.extract()?;
            let reason: String = tuple.get_item(1)?.extract()?;

            if action == "reject" || action == "kill" {
                return Ok((action, reason));
            }
        }
        Ok(("continue".to_string(), String::new()))
    }

    /// Number of interceptors in the chain.
    fn __len__(&self) -> usize {
        self.interceptors.len()
    }

    /// Clear all interceptors.
    fn clear(&mut self) {
        self.interceptors.clear();
    }
}

// ============================================================
// RustSystemClock
// ============================================================

#[pyclass]
pub struct RustSystemClock {
    start: Instant,
}

#[pymethods]
impl RustSystemClock {
    #[new]
    fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Current UTC time as ISO 8601 string.
    fn now(&self) -> String {
        // Use system time for wall clock
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let secs = now.as_secs();
        let nanos = now.subsec_nanos();

        // Format as ISO 8601: YYYY-MM-DDTHH:MM:SS.ffffffZ
        let days_since_epoch = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;
        let micros = nanos / 1000;

        // Civil date from days since epoch (1970-01-01)
        let (year, month, day) = days_to_civil(days_since_epoch as i64);

        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
            year, month, day, hours, minutes, seconds, micros
        )
    }

    /// Monotonic seconds since clock creation.
    fn monotonic(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Epoch seconds (float).
    fn epoch_seconds(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }
}

/// Convert days since Unix epoch to (year, month, day).
/// Algorithm from Howard Hinnant's date library.
fn days_to_civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}

// ============================================================
// RustSimulatedClock
// ============================================================

#[pyclass]
pub struct RustSimulatedClock {
    epoch_seconds: f64,
    mono: f64,
}

#[pymethods]
impl RustSimulatedClock {
    #[new]
    #[pyo3(signature = (start_epoch_seconds=1704067200.0))]
    fn new(start_epoch_seconds: f64) -> Self {
        // Default: 2024-01-01T00:00:00Z
        Self {
            epoch_seconds: start_epoch_seconds,
            mono: 0.0,
        }
    }

    /// Current time as ISO 8601 string.
    fn now(&self) -> String {
        let secs = self.epoch_seconds as u64;
        let frac = self.epoch_seconds - secs as f64;
        let micros = (frac * 1_000_000.0) as u64;

        let days_since_epoch = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        let (year, month, day) = days_to_civil(days_since_epoch as i64);

        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
            year, month, day, hours, minutes, seconds, micros
        )
    }

    /// Monotonic seconds since clock start.
    fn monotonic(&self) -> f64 {
        self.mono
    }

    /// Raw epoch seconds.
    fn epoch_seconds(&self) -> f64 {
        self.epoch_seconds
    }

    /// Advance time by the given number of seconds.
    fn advance(&mut self, seconds: f64) {
        self.epoch_seconds += seconds;
        self.mono += seconds;
    }

    /// Set absolute epoch seconds. Monotonic advances if new time is later.
    fn set(&mut self, epoch_seconds: f64) {
        let delta = epoch_seconds - self.epoch_seconds;
        if delta > 0.0 {
            self.mono += delta;
        }
        self.epoch_seconds = epoch_seconds;
    }
}

// ============================================================
// RustTradingGate
// ============================================================

#[pyclass]
pub struct RustTradingGate {
    halted: bool,
    reasons: Vec<String>,
}

#[pymethods]
impl RustTradingGate {
    #[new]
    fn new() -> Self {
        Self {
            halted: false,
            reasons: Vec::new(),
        }
    }

    /// Halt trading with an optional reason.
    #[pyo3(signature = (reason="manual halt".to_string()))]
    fn halt(&mut self, reason: String) {
        self.halted = true;
        self.reasons.push(reason);
    }

    /// Resume trading, clearing all halt reasons.
    fn resume(&mut self) {
        self.halted = false;
        self.reasons.clear();
    }

    /// Check gate status. Returns (result, reasons) where result is "allow" or "block".
    fn check(&self) -> (String, Vec<String>) {
        if self.halted {
            ("block".to_string(), self.reasons.clone())
        } else {
            ("allow".to_string(), Vec::new())
        }
    }

    /// Whether trading is currently halted.
    #[getter]
    fn is_halted(&self) -> bool {
        self.halted
    }
}
