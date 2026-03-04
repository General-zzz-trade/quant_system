use pyo3::prelude::*;

/// Fast event type routing. Mirrors dispatcher.py _route_from_type() +
/// pipeline.py _detect_kind().
///
/// Returns: "pipeline", "decision", "execution", or "drop".
#[pyfunction]
pub fn rust_route_event_type(event_type: &str) -> &'static str {
    let et = event_type.to_ascii_uppercase();

    // Order reports/updates are PIPELINE (must check before generic ORDER)
    if et.contains("ORDER_UPDATE") || et.contains("ORDER_REPORT") || et.contains("ORDER_STATUS") {
        return "pipeline";
    }

    if et.contains("MARKET") || et.contains("FILL") || et.contains("FUNDING") {
        return "pipeline";
    }

    if et.contains("SIGNAL") || et.contains("INTENT") || et.contains("RISK") {
        return "decision";
    }

    // Generic ORDER (submit/cancel/replace) → execution
    if et.contains("ORDER") {
        return "execution";
    }

    "drop"
}
