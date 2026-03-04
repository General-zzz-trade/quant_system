use pyo3::prelude::*;
use sha2::{Digest, Sha256};

/// Compute SHA-256 hex digest of sorted key=value pairs.
///
/// Mirrors the pattern used across 6 Python files:
///   json.dumps(dict(payload), sort_keys=True, default=str) -> sha256 -> hexdigest[:length]
///
/// Accepts a list of (key, value_str) tuples — caller must pre-convert
/// Decimal/int/etc to strings on the Python side (trivial).
#[pyfunction]
#[pyo3(signature = (fields, length=16))]
pub fn rust_payload_digest(fields: Vec<(String, String)>, length: usize) -> String {
    let mut sorted = fields;
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    // Build compact JSON: {"k1": "v1", "k2": "v2", ...} with sorted keys
    // Matches json.dumps(sort_keys=True, separators=(",", ":"), default=str)
    let mut json = String::with_capacity(sorted.len() * 32);
    json.push('{');
    for (i, (k, v)) in sorted.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        // JSON-escape key and value
        json.push('"');
        json_escape_into(&mut json, k);
        json.push_str("\":\"");
        json_escape_into(&mut json, v);
        json.push('"');
    }
    json.push('}');

    let hash = Sha256::digest(json.as_bytes());
    let hex = hex_encode(&hash);
    if length >= hex.len() {
        hex
    } else {
        hex[..length].to_owned()
    }
}

/// SHA-256 hex digest of a plain string, truncated to `length`.
/// Mirrors execution/adapters/common/hashing.py stable_hash().
#[pyfunction]
#[pyo3(signature = (text, length=16))]
pub fn rust_stable_hash(text: &str, length: usize) -> String {
    let hash = Sha256::digest(text.as_bytes());
    let hex = hex_encode(&hash);
    if length >= hex.len() {
        hex
    } else {
        hex[..length].to_owned()
    }
}

fn json_escape_into(buf: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if c < '\x20' => {
                buf.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => buf.push(c),
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}
