use pyo3::prelude::*;

mod dedup_guard;
mod digest;
mod event_id;
mod fill_dedup;
mod json_parse;
mod ml_decision;
mod rate_limiter;
mod request_id;
mod route_match;
mod sequence_buffer;
mod signer;

#[pymodule]
fn _quant_hotpath(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 1 classes
    m.add_class::<dedup_guard::DuplicateGuard>()?;
    m.add_class::<rate_limiter::RustRateLimitPolicy>()?;
    m.add_class::<ml_decision::RustMLDecision>()?;
    m.add_function(wrap_pyfunction!(json_parse::rust_parse_kline, m)?)?;
    m.add_function(wrap_pyfunction!(json_parse::rust_parse_depth, m)?)?;
    m.add_function(wrap_pyfunction!(json_parse::rust_demux_user_stream, m)?)?;

    // Phase 2 classes + functions
    m.add_function(wrap_pyfunction!(digest::rust_payload_digest, m)?)?;
    m.add_function(wrap_pyfunction!(digest::rust_stable_hash, m)?)?;
    m.add_class::<fill_dedup::RustFillDedupGuard>()?;
    m.add_class::<sequence_buffer::RustSequenceBuffer>()?;
    m.add_function(wrap_pyfunction!(event_id::rust_event_id, m)?)?;
    m.add_function(wrap_pyfunction!(event_id::rust_now_ns, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_sanitize, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_short_hash, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_make_idempotency_key, m)?)?;
    m.add_function(wrap_pyfunction!(request_id::rust_client_order_id, m)?)?;
    m.add_function(wrap_pyfunction!(signer::rust_hmac_sign, m)?)?;
    m.add_function(wrap_pyfunction!(signer::rust_hmac_verify, m)?)?;
    m.add_function(wrap_pyfunction!(route_match::rust_route_event_type, m)?)?;
    Ok(())
}
