use pyo3::prelude::*;
use sha2::{Digest, Sha256};

/// Sanitize a string: replace non-alphanumeric/non-dash/non-underscore with "-",
/// strip leading/trailing dashes. Mirrors request_ids.py _sanitize().
#[pyfunction]
pub fn rust_sanitize(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return "x".to_owned();
    }
    let mut out = String::with_capacity(trimmed.len());
    for ch in trimmed.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('-');
        }
    }
    let result = out.trim_matches('-').to_owned();
    if result.is_empty() {
        "x".to_owned()
    } else {
        result
    }
}

/// SHA-256 hex digest truncated to n chars. Mirrors request_ids.py _short_hash().
#[pyfunction]
#[pyo3(signature = (text, n=10))]
pub fn rust_short_hash(text: &str, n: usize) -> String {
    let hash = Sha256::digest(text.as_bytes());
    let hex = format!("{:x}", hash);
    hex[..n.min(hex.len())].to_owned()
}

/// Stable idempotency key. Mirrors request_ids.py make_idempotency_key().
#[pyfunction]
pub fn rust_make_idempotency_key(venue: &str, action: &str, key: &str) -> String {
    let v = rust_sanitize(venue).to_lowercase();
    let a = rust_sanitize(action).to_lowercase();
    let k = rust_sanitize(key);
    let base = format!("{}|{}|{}", v, a, k);
    let hash = Sha256::digest(base.as_bytes());
    format!("{:x}", hash)
}

/// Generate client_order_id. Mirrors RequestIdFactory.client_order_id().
#[pyfunction]
#[pyo3(signature = (namespace, run_id, strategy, symbol, logical_id=None, nonce=None, deterministic=true, max_len=36))]
pub fn rust_client_order_id(
    namespace: &str,
    run_id: &str,
    strategy: &str,
    symbol: &str,
    logical_id: Option<&str>,
    nonce: Option<i64>,
    deterministic: bool,
    max_len: usize,
) -> String {
    let ns = rust_sanitize(namespace).to_lowercase();
    let run = rust_sanitize(run_id).to_lowercase();
    let strat = rust_sanitize(strategy).to_lowercase();
    let sym = rust_sanitize(symbol).to_uppercase();

    let suffix = if deterministic && logical_id.is_some() {
        let lid = logical_id.unwrap();
        let text = format!("{}|{}|{}|{}|{}", ns, run, strat, sym, lid);
        rust_short_hash(&text, 10)
    } else {
        let n = nonce.unwrap_or(0);
        format!("n{:x}", n)
    };

    let raw = format!("{}-{}-{}-{}-{}", ns, run, strat, sym, suffix);
    let raw = rust_sanitize(&raw);

    if raw.len() <= max_len {
        return raw;
    }

    // Truncation: preserve suffix
    let tail = format!("-{}", suffix);
    let budget = max_len.saturating_sub(tail.len());
    if budget < 8 {
        let compact_hash_len = max_len.saturating_sub(ns.len() + 1).max(8);
        let compact = format!("{}-{}", ns, rust_short_hash(&raw, compact_hash_len));
        return compact[..compact.len().min(max_len)].to_owned();
    }

    let head = format!("{}-{}-{}-{}", ns, run, strat, sym);
    let head = rust_sanitize(&head);
    let head_trimmed = head[..head.len().min(budget)].trim_end_matches('-');
    let result = format!("{}{}", head_trimmed, tail);
    result[..result.len().min(max_len)].to_owned()
}
