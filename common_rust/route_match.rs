use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// Route a full event object in a single Rust call (eliminates Python getattr chain).
///
/// Extracts event_type from event.event_type / event.header.event_type / event.EVENT_TYPE
/// then routes to "pipeline", "decision", "execution", or "drop".
#[pyfunction]
pub fn rust_route_event(event: &Bound<'_, pyo3::PyAny>) -> &'static str {
    // Try event.event_type
    if let Ok(et) = event.getattr("event_type") {
        // Could be Enum with .value
        let et_str = if let Ok(val) = et.getattr("value") {
            val.to_string()
        } else {
            et.to_string()
        };
        let routed = route_label(&et_str);
        if routed != "drop" {
            return routed;
        }
    }

    // Try event.header.event_type
    if let Ok(header) = event.getattr("header") {
        if let Ok(et) = header.getattr("event_type") {
            let et_str = if let Ok(val) = et.getattr("value") {
                val.to_string()
            } else {
                et.to_string()
            };
            let routed = route_label(&et_str);
            if routed != "drop" {
                return routed;
            }
        }
    }

    // Try event.EVENT_TYPE
    if let Ok(name) = event.getattr("EVENT_TYPE") {
        if let Ok(s) = name.extract::<String>() {
            return route_label(&s);
        }
    }

    "drop"
}

fn route_label(event_type: &str) -> &'static str {
    rust_route_event_type_inner(event_type)
}

/// Fast event type routing. Mirrors dispatcher.py _route_from_type() +
/// pipeline.py _detect_kind().
///
/// Returns: "pipeline", "decision", "execution", or "drop".
#[pyfunction]
pub fn rust_route_event_type(event_type: &str) -> &'static str {
    rust_route_event_type_inner(event_type)
}

fn rust_route_event_type_inner(event_type: &str) -> &'static str {
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
