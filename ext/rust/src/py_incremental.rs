// py_incremental.rs — PyO3 wrappers for incremental indicator trackers.
//
// Exposes EmaTracker, RsiTracker, AtrTracker, AdxTracker to Python as
// PyEmaTracker, PyRsiTracker, PyAtrTracker, PyAdxTracker.

use pyo3::prelude::*;

use crate::incremental_trackers::{AdxTracker, AtrTracker, EmaTracker, RsiTracker};

// ============================================================
// PyEmaTracker
// ============================================================

#[pyclass]
pub struct PyEmaTracker {
    inner: EmaTracker,
}

#[pymethods]
impl PyEmaTracker {
    #[new]
    fn new(period: usize) -> Self {
        Self {
            inner: EmaTracker::new(period),
        }
    }

    /// Push a new value. NaN/infinite inputs are ignored.
    fn push(&mut self, value: f64) {
        self.inner.push(value);
    }

    /// Current EMA value, or None if no data pushed yet.
    fn value(&self) -> Option<f64> {
        self.inner.value()
    }

    /// True when at least `period` bars have been pushed.
    fn ready(&self) -> bool {
        self.inner.ready()
    }

    /// Number of values pushed so far.
    fn count(&self) -> usize {
        self.inner.count()
    }
}

// ============================================================
// PyRsiTracker
// ============================================================

#[pyclass]
pub struct PyRsiTracker {
    inner: RsiTracker,
}

#[pymethods]
impl PyRsiTracker {
    #[new]
    fn new(period: usize) -> Self {
        Self {
            inner: RsiTracker::new(period),
        }
    }

    /// Push a new close price. NaN/infinite inputs are ignored.
    fn push(&mut self, close: f64) {
        self.inner.push(close);
    }

    /// Current RSI value, or None if fewer than `period` changes processed.
    fn value(&self) -> Option<f64> {
        self.inner.value()
    }

    /// True when at least `period` changes have been processed.
    fn ready(&self) -> bool {
        self.inner.ready()
    }
}

// ============================================================
// PyAtrTracker
// ============================================================

#[pyclass]
pub struct PyAtrTracker {
    inner: AtrTracker,
}

#[pymethods]
impl PyAtrTracker {
    #[new]
    fn new(period: usize) -> Self {
        Self {
            inner: AtrTracker::new(period),
        }
    }

    /// Push a new bar (high, low, close). NaN/infinite inputs are ignored.
    fn push(&mut self, high: f64, low: f64, close: f64) {
        self.inner.push(high, low, close);
    }

    /// Current ATR value, or None if fewer than `period` bars processed.
    fn value(&self) -> Option<f64> {
        self.inner.value()
    }

    /// ATR normalized by close price, or None if not ready or close is zero.
    fn normalized(&self, close: f64) -> Option<f64> {
        self.inner.normalized(close)
    }

    /// True when at least `period` bars have been processed.
    fn ready(&self) -> bool {
        self.inner.ready()
    }
}

// ============================================================
// PyAdxTracker
// ============================================================

#[pyclass]
pub struct PyAdxTracker {
    inner: AdxTracker,
}

#[pymethods]
impl PyAdxTracker {
    #[new]
    fn new(period: usize) -> Self {
        Self {
            inner: AdxTracker::new(period),
        }
    }

    /// Push a new bar (high, low, close). NaN/infinite inputs are ignored.
    fn push(&mut self, high: f64, low: f64, close: f64) {
        self.inner.push(high, low, close);
    }

    /// Current ADX value, or None if not enough data (requires 2*period warmup).
    fn value(&self) -> Option<f64> {
        self.inner.value()
    }

    /// True when ADX has been initialized (2*period bars of warmup).
    fn ready(&self) -> bool {
        self.inner.ready()
    }
}
