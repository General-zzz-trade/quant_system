// tick_result.rs — RustTickResult: output type for PyO3 tick processing
//
// Extracted from tick_processor.rs to keep each file under 500 lines.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::features::engine::{N_FEATURES, FEATURE_NAMES};

/// Result of processing a single market tick through the full pipeline.
#[pyclass(name = "RustTickResult", frozen)]
pub struct RustTickResult {
    #[pyo3(get)]
    pub advanced: bool,
    #[pyo3(get)]
    pub changed: bool,
    #[pyo3(get)]
    pub event_index: i64,
    #[pyo3(get)]
    pub ml_score: f64,
    #[pyo3(get)]
    pub ml_short_score: f64,
    #[pyo3(get)]
    pub raw_score: f64,
    #[pyo3(get)]
    pub last_event_id: Option<String>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
    // State exports (PyObject, built once at construction)
    #[pyo3(get)]
    pub markets: PyObject,
    #[pyo3(get)]
    pub positions: PyObject,
    #[pyo3(get)]
    pub account: PyObject,
    #[pyo3(get)]
    pub portfolio: PyObject,
    #[pyo3(get)]
    pub risk: PyObject,
    // Features buffer (not exposed directly)
    pub features_buf: [f64; N_FEATURES],
    // Pre-built features dict (set by process_tick_full)
    #[pyo3(get)]
    pub features_dict: Option<PyObject>,
}

#[pymethods]
impl RustTickResult {
    /// Export features as PyDict, skipping NaN values.
    fn get_features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (i, &name) in FEATURE_NAMES.iter().enumerate() {
            let v = self.features_buf[i];
            if !v.is_nan() {
                dict.set_item(name, v)?;
            }
        }
        Ok(dict)
    }
}
