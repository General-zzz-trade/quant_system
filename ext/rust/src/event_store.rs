use pyo3::prelude::*;

#[pyclass(name = "RustInMemoryEventStore")]
pub struct RustInMemoryEventStore {
    events: Vec<Py<PyAny>>,
}

#[pymethods]
impl RustInMemoryEventStore {
    #[new]
    fn new() -> Self {
        Self { events: Vec::new() }
    }

    fn append(&mut self, event: Py<PyAny>) {
        self.events.push(event);
    }

    fn iter_events(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.events.iter().map(|event| event.clone_ref(py)).collect()
    }

    fn size(&self) -> usize {
        self.events.len()
    }
}
