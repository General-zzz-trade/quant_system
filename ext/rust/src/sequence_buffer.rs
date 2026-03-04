use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};

/// Sequence-based reordering buffer.
///
/// Mirrors both execution/ingress/sequence_buffer.py and
/// execution/safety/out_of_order_guard.py in a single unified implementation.
///
/// Note: payloads are stored as opaque PyObject since we don't inspect them.
#[pyclass]
pub struct RustSequenceBuffer {
    max_buffer: usize,
    expected_seq: HashMap<String, i64>,
    buffer: HashMap<String, BTreeMap<i64, PyObject>>,
}

#[pymethods]
impl RustSequenceBuffer {
    #[new]
    #[pyo3(signature = (max_buffer_size=1000))]
    fn new(max_buffer_size: usize) -> Self {
        Self {
            max_buffer: max_buffer_size,
            expected_seq: HashMap::new(),
            buffer: HashMap::new(),
        }
    }

    /// Push a sequenced message. Returns list of payloads released in order.
    fn push(&mut self, key: &str, seq: i64, payload: PyObject) -> Vec<PyObject> {
        let expected = *self.expected_seq.get(key).unwrap_or(&0);

        if seq < expected {
            return vec![]; // duplicate/expired
        }

        if seq > expected {
            let buf = self.buffer.entry(key.to_owned()).or_default();
            if buf.len() < self.max_buffer {
                buf.insert(seq, payload);
            }
            return vec![];
        }

        // seq == expected: release consecutive
        let mut released = vec![payload];
        let mut next = seq + 1;

        let buf = self.buffer.entry(key.to_owned()).or_default();
        while let Some(p) = buf.remove(&next) {
            released.push(p);
            next += 1;
        }
        self.expected_seq.insert(key.to_owned(), next);

        // Clean up empty buffer
        if buf.is_empty() {
            self.buffer.remove(key);
        }

        released
    }

    fn expected_seq(&self, key: &str) -> i64 {
        *self.expected_seq.get(key).unwrap_or(&0)
    }

    fn buffered_count(&self, key: &str) -> usize {
        self.buffer.get(key).map_or(0, |b| b.len())
    }

    /// Flush all buffered payloads for a key, in sequence order.
    fn flush(&mut self, key: &str) -> Vec<PyObject> {
        self.expected_seq.remove(key);
        match self.buffer.remove(key) {
            Some(buf) => buf.into_values().collect(),
            None => vec![],
        }
    }

    #[pyo3(signature = (key=None))]
    fn pending_count(&self, key: Option<&str>) -> usize {
        match key {
            Some(k) => self.buffer.get(k).map_or(0, |b| b.len()),
            None => self.buffer.values().map(|b| b.len()).sum(),
        }
    }

    fn reset_key(&mut self, key: &str) {
        self.expected_seq.remove(key);
        self.buffer.remove(key);
    }

    fn clear(&mut self) {
        self.expected_seq.clear();
        self.buffer.clear();
    }
}
