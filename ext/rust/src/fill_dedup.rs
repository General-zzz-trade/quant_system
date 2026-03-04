use pyo3::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Fill/Order dedup guard with payload digest corruption detection.
///
/// Mirrors execution/safety/duplicate_guard.py DuplicateGuard:
///   1. New key → store digest, return "new"
///   2. Same key + same digest → return "duplicate"
///   3. Same key + different digest → return "corrupted"
#[pyclass]
pub struct RustFillDedupGuard {
    seen: HashMap<String, Entry>,
    ttl_sec: f64,
    max_size: usize,
    last_prune: f64,
}

struct Entry {
    digest: String,
    ts: f64,
}

#[pymethods]
impl RustFillDedupGuard {
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

    /// Check a key+digest pair.
    /// Returns: "new", "duplicate", or "corrupted:<old_digest>:<new_digest>"
    fn check(&mut self, key: &str, digest: &str, now: f64) -> String {
        // Prune first
        if (now - self.last_prune > 60.0) || self.seen.len() > self.max_size {
            let cutoff = now - self.ttl_sec;
            self.seen.retain(|_, e| e.ts > cutoff);
            self.last_prune = now;
        }

        if let Some(existing) = self.seen.get(key) {
            if existing.digest == digest {
                return "duplicate".to_owned();
            }
            return format!("corrupted:{}:{}", existing.digest, digest);
        }

        self.seen.insert(
            key.to_owned(),
            Entry {
                digest: digest.to_owned(),
                ts: now,
            },
        );
        "new".to_owned()
    }

    /// Compute digest from sorted fields and check in one call.
    /// Returns: "new", "duplicate", or "corrupted:..."
    fn check_with_fields(&mut self, key: &str, fields: Vec<(String, String)>, now: f64) -> String {
        let digest = compute_digest_from_fields(&fields);
        self.check(key, &digest, now)
    }

    fn contains(&self, key: &str) -> bool {
        self.seen.contains_key(key)
    }

    fn __len__(&self) -> usize {
        self.seen.len()
    }

    fn clear(&mut self) {
        self.seen.clear();
        self.last_prune = 0.0;
    }
}

fn compute_digest_from_fields(fields: &[(String, String)]) -> String {
    let mut sorted: Vec<_> = fields.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut json = String::with_capacity(sorted.len() * 32);
    json.push('{');
    for (i, (k, v)) in sorted.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        json.push('"');
        json.push_str(k);
        json.push_str("\":\"");
        json.push_str(v);
        json.push('"');
    }
    json.push('}');

    let hash = Sha256::digest(json.as_bytes());
    format!("{:x}", hash)[..16].to_owned()
}
