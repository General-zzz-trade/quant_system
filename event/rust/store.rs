use pyo3::prelude::*;
use std::collections::VecDeque;

/// Bounded in-memory event store (ring buffer).
///
/// Earlier this was an unbounded `Vec<Py<PyAny>>`. coordinator.emit()
/// appends every event but nothing in production ever reads back —
/// we observed okx-alpha RSS growing ~8.5 MB/h over 21h (227 → 405 MB)
/// which would have hit the 1024 MB systemd cap in another ~3 days.
///
/// Cap = 1000 events (debug context for crashes / introspection).
/// At ~120 events/hour that's ~8 hours of recent state, which is plenty.
const MAX_EVENTS: usize = 1000;

#[pyclass(name = "RustInMemoryEventStore")]
pub struct RustInMemoryEventStore {
    events: VecDeque<Py<PyAny>>,
}

#[pymethods]
impl RustInMemoryEventStore {
    #[new]
    fn new() -> Self {
        Self { events: VecDeque::with_capacity(MAX_EVENTS) }
    }

    fn append(&mut self, event: Py<PyAny>) {
        if self.events.len() >= MAX_EVENTS {
            self.events.pop_front();
        }
        self.events.push_back(event);
    }

    fn iter_events(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.events.iter().map(|event| event.clone_ref(py)).collect()
    }

    fn size(&self) -> usize {
        self.events.len()
    }
}
