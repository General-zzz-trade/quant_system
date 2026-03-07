use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

struct AckEntry {
    payload_json: String,
    ts: f64,
}

struct DedupEntry {
    digest: String,
    ts: f64,
}

fn current_time_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn is_expired(ts: f64, ttl_sec: Option<f64>, now_ts: f64) -> bool {
    ttl_sec
        .map(|ttl| (now_ts - ts) > ttl)
        .unwrap_or(false)
}

#[pyclass(name = "RustAckStore")]
pub struct RustAckStore {
    items: HashMap<String, AckEntry>,
    ttl_sec: Option<f64>,
}

#[pymethods]
impl RustAckStore {
    #[new]
    #[pyo3(signature = (ttl_sec=None))]
    fn new(ttl_sec: Option<f64>) -> Self {
        Self {
            items: HashMap::new(),
            ttl_sec,
        }
    }

    #[pyo3(signature = (key, now_ts=None))]
    fn get_json(&mut self, key: &str, now_ts: Option<f64>) -> Option<String> {
        let now = now_ts.unwrap_or_else(current_time_ts);
        let state = self.items.get(key).map(|entry| {
            (
                is_expired(entry.ts, self.ttl_sec, now),
                entry.payload_json.clone(),
            )
        });
        match state {
            Some((true, _)) => {
                self.items.remove(key);
                None
            }
            Some((false, payload_json)) => Some(payload_json),
            None => None,
        }
    }

    #[pyo3(signature = (key, payload_json, now_ts=None))]
    fn put_json(&mut self, key: &str, payload_json: &str, now_ts: Option<f64>) {
        self.items.insert(
            key.to_owned(),
            AckEntry {
                payload_json: payload_json.to_owned(),
                ts: now_ts.unwrap_or_else(current_time_ts),
            },
        );
    }

    #[pyo3(signature = (now_ts=None))]
    fn prune(&mut self, now_ts: Option<f64>) -> usize {
        let Some(ttl_sec) = self.ttl_sec else {
            return 0;
        };
        let now = now_ts.unwrap_or_else(current_time_ts);
        let before = self.items.len();
        self.items.retain(|_, entry| (now - entry.ts) <= ttl_sec);
        before - self.items.len()
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
    }
}

#[pyclass(name = "RustDedupStore")]
pub struct RustDedupStore {
    items: HashMap<String, DedupEntry>,
    ttl_sec: Option<f64>,
}

#[pymethods]
impl RustDedupStore {
    #[new]
    #[pyo3(signature = (ttl_sec=None))]
    fn new(ttl_sec: Option<f64>) -> Self {
        Self {
            items: HashMap::new(),
            ttl_sec,
        }
    }

    #[pyo3(signature = (key, now_ts=None))]
    fn get(&mut self, key: &str, now_ts: Option<f64>) -> Option<String> {
        let now = now_ts.unwrap_or_else(current_time_ts);
        let state = self.items.get(key).map(|entry| {
            (
                is_expired(entry.ts, self.ttl_sec, now),
                entry.digest.clone(),
            )
        });
        match state {
            Some((true, _)) => {
                self.items.remove(key);
                None
            }
            Some((false, digest)) => Some(digest),
            None => None,
        }
    }

    #[pyo3(signature = (key, digest, now_ts=None))]
    fn put(&mut self, key: &str, digest: &str, now_ts: Option<f64>) {
        self.items.insert(
            key.to_owned(),
            DedupEntry {
                digest: digest.to_owned(),
                ts: now_ts.unwrap_or_else(current_time_ts),
            },
        );
    }

    #[pyo3(signature = (now_ts=None))]
    fn prune(&mut self, now_ts: Option<f64>) -> usize {
        let Some(ttl_sec) = self.ttl_sec else {
            return 0;
        };
        let now = now_ts.unwrap_or_else(current_time_ts);
        let before = self.items.len();
        self.items.retain(|_, entry| (now - entry.ts) <= ttl_sec);
        before - self.items.len()
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn clear(&mut self) {
        self.items.clear();
    }
}
