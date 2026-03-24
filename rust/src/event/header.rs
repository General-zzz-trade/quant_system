//! RustEventHeader — causation tracing header for all framework events.
//!
//! Provides parent/root event ID propagation so every event in a causal chain
//! (market → signal → intent → order → fill) can be traced back to its root.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::event::id::{rust_event_id, rust_now_ns};

#[pyclass(name = "RustEventHeader", frozen)]
#[derive(Clone, Debug)]
pub struct RustEventHeader {
    #[pyo3(get)]
    pub event_id: String,
    #[pyo3(get)]
    pub event_type: String, // "market", "signal", "intent", "order", "fill", "risk", "control", "funding"
    #[pyo3(get)]
    pub version: i32,
    #[pyo3(get)]
    pub ts_ns: i64,
    #[pyo3(get)]
    pub source: String,
    #[pyo3(get)]
    pub parent_event_id: Option<String>,
    #[pyo3(get)]
    pub root_event_id: Option<String>,
    #[pyo3(get)]
    pub run_id: Option<String>,
    #[pyo3(get)]
    pub seq: Option<i64>,
    #[pyo3(get)]
    pub correlation_id: Option<String>,
}

impl RustEventHeader {
    /// Rust-accessible serialization (non-PyO3).
    pub fn serialize_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("event_id", &self.event_id)?;
        dict.set_item("event_type", &self.event_type)?;
        dict.set_item("version", self.version)?;
        dict.set_item("ts_ns", self.ts_ns)?;
        dict.set_item("source", &self.source)?;
        dict.set_item("parent_event_id", &self.parent_event_id)?;
        dict.set_item("root_event_id", &self.root_event_id)?;
        dict.set_item("run_id", &self.run_id)?;
        dict.set_item("seq", self.seq)?;
        dict.set_item("correlation_id", &self.correlation_id)?;
        Ok(dict)
    }
}

#[pymethods]
impl RustEventHeader {
    #[new]
    #[pyo3(signature = (event_id, event_type, version, ts_ns, source, parent_event_id=None, root_event_id=None, run_id=None, seq=None, correlation_id=None))]
    fn new(
        event_id: String,
        event_type: String,
        version: i32,
        ts_ns: i64,
        source: String,
        parent_event_id: Option<String>,
        root_event_id: Option<String>,
        run_id: Option<String>,
        seq: Option<i64>,
        correlation_id: Option<String>,
    ) -> Self {
        Self {
            event_id,
            event_type,
            version,
            ts_ns,
            source,
            parent_event_id,
            root_event_id,
            run_id,
            seq,
            correlation_id,
        }
    }

    /// Create a root header (no parent). The event_id becomes its own root_event_id.
    #[staticmethod]
    #[pyo3(signature = (event_type, version, source, run_id=None, correlation_id=None))]
    fn new_root(
        event_type: String,
        version: i32,
        source: String,
        run_id: Option<String>,
        correlation_id: Option<String>,
    ) -> Self {
        let eid = rust_event_id();
        let ts = rust_now_ns();
        Self {
            event_id: eid.clone(),
            event_type,
            version,
            ts_ns: ts,
            source,
            parent_event_id: None,
            root_event_id: Some(eid),
            run_id,
            seq: None,
            correlation_id,
        }
    }

    /// Create a child header that inherits root_event_id, run_id, and correlation_id from parent.
    #[staticmethod]
    fn from_parent(
        parent: &RustEventHeader,
        event_type: String,
        version: i32,
        source: String,
    ) -> Self {
        let eid = rust_event_id();
        let ts = rust_now_ns();
        Self {
            event_id: eid,
            event_type,
            version,
            ts_ns: ts,
            source,
            parent_event_id: Some(parent.event_id.clone()),
            root_event_id: parent.root_event_id.clone(),
            run_id: parent.run_id.clone(),
            seq: None,
            correlation_id: parent.correlation_id.clone(),
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.serialize_to_dict(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustEventHeader(id={}, type={}, source={})",
            self.event_id, self.event_type, self.source
        )
    }
}
