use pyo3::prelude::*;

use crate::state::type_helpers::opt_str_repr;

// ===========================================================================
// ReducerResult
// ===========================================================================
#[pyclass(name = "RustReducerResult", frozen)]
pub struct RustReducerResult {
    #[pyo3(get)]
    pub state: PyObject,
    #[pyo3(get)]
    pub changed: bool,
    #[pyo3(get)]
    pub note: Option<String>,
}

impl Clone for RustReducerResult {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            state: self.state.clone_ref(py),
            changed: self.changed,
            note: self.note.clone(),
        })
    }
}

#[pymethods]
impl RustReducerResult {
    #[new]
    #[pyo3(signature = (state, changed, note=None))]
    fn new(state: PyObject, changed: bool, note: Option<String>) -> Self {
        Self { state, changed, note }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let state_repr = self
            .state
            .bind(py)
            .repr()
            .map(|r| r.to_string())
            .unwrap_or_else(|_| "<?>".to_string());
        format!(
            "RustReducerResult(state={}, changed={}, note={})",
            state_repr,
            self.changed,
            opt_str_repr(&self.note),
        )
    }

    fn __eq__(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        if self.changed != other.changed {
            return Ok(false);
        }
        if self.note != other.note {
            return Ok(false);
        }
        let eq = self.state.bind(py).eq(other.state.bind(py))?;
        Ok(eq)
    }
}
