use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(name = "RustPayloadDedupGuard")]
pub struct RustPayloadDedupGuard {
    seen: HashMap<String, String>,
}

#[pymethods]
impl RustPayloadDedupGuard {
    #[new]
    fn new() -> Self {
        Self {
            seen: HashMap::new(),
        }
    }

    fn check_and_insert(&mut self, key: &str, digest: &str) -> PyResult<bool> {
        match self.seen.get(key) {
            Some(prev) if prev == digest => Ok(false),
            Some(_) => Err(PyValueError::new_err(format!(
                "payload mismatch for duplicate key: {}",
                key
            ))),
            None => {
                self.seen.insert(key.to_owned(), digest.to_owned());
                Ok(true)
            }
        }
    }

    fn __len__(&self) -> usize {
        self.seen.len()
    }

    fn clear(&mut self) {
        self.seen.clear();
    }
}
