use hmac::{Hmac, Mac};
use pyo3::prelude::*;
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// HMAC-SHA256 sign params. Mirrors signer.py HmacSha256Signer.sign().
///
/// Takes secret and a list of (key, value) string pairs.
/// Auto-adds timestamp if missing. Returns (query_string, signature_hex).
#[pyfunction]
pub fn rust_hmac_sign(secret: &str, params: Vec<(String, String)>) -> (String, String) {
    let mut items = params;

    // Add timestamp if missing
    let has_ts = items.iter().any(|(k, _)| k == "timestamp");
    if !has_ts {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        items.push(("timestamp".to_owned(), ts.to_string()));
    }

    // Sort by key
    items.sort_by(|a, b| a.0.cmp(&b.0));

    // URL-encode (simple: key=value&key=value — Binance params are ASCII)
    let query = items
        .iter()
        .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
        .collect::<Vec<_>>()
        .join("&");

    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC key");
    mac.update(query.as_bytes());
    let result = mac.finalize();
    let sig = hex_encode(&result.into_bytes());

    (query, sig)
}

/// Verify HMAC-SHA256 signature. Mirrors HmacSha256Signer.verify().
#[pyfunction]
pub fn rust_hmac_verify(secret: &str, params: Vec<(String, String)>, signature: &str) -> bool {
    let mut items: Vec<(String, String)> = params
        .into_iter()
        .filter(|(k, _)| k != "signature")
        .collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));

    let query = items
        .iter()
        .map(|(k, v)| format!("{}={}", url_encode(k), url_encode(v)))
        .collect::<Vec<_>>()
        .join("&");

    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC key");
    mac.update(query.as_bytes());
    let result = mac.finalize();
    let expected = hex_encode(&result.into_bytes());

    constant_time_eq(expected.as_bytes(), signature.as_bytes())
}

fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}
