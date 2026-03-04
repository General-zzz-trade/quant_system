use pyo3::prelude::*;
use std::collections::HashMap;

/// Event dedup guard with TTL + capacity pruning.
///
/// Mirrors engine/dispatcher.py lines 115-127.
#[pyclass]
pub struct DuplicateGuard {
    seen: HashMap<String, f64>,
    ttl_sec: f64,
    max_size: usize,
    last_prune: f64,
}

#[pymethods]
impl DuplicateGuard {
    #[new]
    #[pyo3(signature = (ttl_sec=86400.0, max_size=500_000))]
    fn new(ttl_sec: f64, max_size: usize) -> Self {
        Self {
            seen: HashMap::new(),
            ttl_sec,
            max_size,
            last_prune: 0.0,
        }
    }

    /// Check if event_id is new. Returns true=new, false=duplicate.
    /// `now` is monotonic timestamp from Python side.
    fn check_and_insert(&mut self, event_id: &str, now: f64) -> bool {
        // Prune expired entries first (matches Python dispatcher behavior
        // where prune runs on the previous dispatch cycle)
        if (now - self.last_prune > 60.0) || self.seen.len() > self.max_size {
            let cutoff = now - self.ttl_sec;
            self.seen.retain(|_, ts| *ts > cutoff);
            self.last_prune = now;
        }

        if self.seen.contains_key(event_id) {
            return false;
        }
        self.seen.insert(event_id.to_owned(), now);
        true
    }

    fn __len__(&self) -> usize {
        self.seen.len()
    }

    fn clear(&mut self) {
        self.seen.clear();
        self.last_prune = 0.0;
    }
}
