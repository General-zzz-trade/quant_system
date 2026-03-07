use pyo3::prelude::*;
use std::collections::HashMap;

// ===========================================================================
// RustGuardConfig
// ===========================================================================

#[pyclass(name = "RustGuardConfig", frozen)]
#[derive(Clone)]
pub struct RustGuardConfig {
    #[pyo3(get)]
    pub stop_on_fatal: bool,
    #[pyo3(get)]
    pub max_consecutive_errors: i32,
    #[pyo3(get)]
    pub max_consecutive_domain_errors: i32,
    #[pyo3(get)]
    pub max_consecutive_execution_errors: i32,
    #[pyo3(get)]
    pub stop_on_unknown_exception: bool,
    #[pyo3(get)]
    pub default_retry_after_s: f64,
}

#[pymethods]
impl RustGuardConfig {
    #[new]
    #[pyo3(signature = (
        stop_on_fatal = true,
        max_consecutive_errors = 5,
        max_consecutive_domain_errors = 5,
        max_consecutive_execution_errors = 2,
        stop_on_unknown_exception = true,
        default_retry_after_s = 0.2,
    ))]
    fn new(
        stop_on_fatal: bool,
        max_consecutive_errors: i32,
        max_consecutive_domain_errors: i32,
        max_consecutive_execution_errors: i32,
        stop_on_unknown_exception: bool,
        default_retry_after_s: f64,
    ) -> Self {
        Self {
            stop_on_fatal,
            max_consecutive_errors,
            max_consecutive_domain_errors,
            max_consecutive_execution_errors,
            stop_on_unknown_exception,
            default_retry_after_s,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustGuardConfig(stop_on_fatal={}, max_consecutive_errors={}, max_consecutive_domain_errors={}, max_consecutive_execution_errors={}, stop_on_unknown_exception={}, default_retry_after_s={})",
            self.stop_on_fatal,
            self.max_consecutive_errors,
            self.max_consecutive_domain_errors,
            self.max_consecutive_execution_errors,
            self.stop_on_unknown_exception,
            self.default_retry_after_s,
        )
    }
}

// ===========================================================================
// RustBasicGuard
// ===========================================================================

/// Mirrors engine/guards.py BasicGuard — a state machine tracking
/// consecutive errors per domain, returning (action, reason, retry_after).
///
/// The Python side passes severity/domain/code as strings rather than
/// the full ClassifiedError, keeping the Rust boundary simple.
#[pyclass(name = "RustBasicGuard")]
pub struct RustBasicGuard {
    cfg: RustGuardConfig,
    consecutive_errors: i32,
    domain_consecutive: HashMap<String, i32>,
    execution_consecutive: i32,
}

#[pymethods]
impl RustBasicGuard {
    #[new]
    #[pyo3(signature = (cfg = None))]
    fn new(cfg: Option<RustGuardConfig>) -> Self {
        Self {
            cfg: cfg.unwrap_or(RustGuardConfig::new(true, 5, 5, 2, true, 0.2)),
            consecutive_errors: 0,
            domain_consecutive: HashMap::new(),
            execution_consecutive: 0,
        }
    }

    /// Called before processing an event. Currently always allows.
    /// Returns (action, reason).
    #[pyo3(signature = ())]
    fn before_event(&self) -> (&'static str, &'static str) {
        ("allow", "ok")
    }

    /// Called after successfully processing an event. Resets all counters.
    /// Returns (action, reason).
    #[pyo3(signature = ())]
    fn after_event(&mut self) -> (&'static str, &'static str) {
        self.consecutive_errors = 0;
        self.domain_consecutive.clear();
        self.execution_consecutive = 0;
        ("allow", "ok")
    }

    /// Called when an error occurs during event processing.
    ///
    /// Args:
    ///     severity: "fatal", "error", "warning"
    ///     domain: "state", "risk", "execution", "invariant", etc.
    ///     code: error code string, e.g. "RETRYABLE", "TIMEOUT", etc.
    ///
    /// Returns (action, reason, retry_after_s) where:
    ///     action: "allow" | "drop" | "retry" | "stop"
    ///     reason: human-readable explanation
    ///     retry_after_s: Optional delay in seconds (only for "retry")
    #[pyo3(signature = (severity, domain, code))]
    fn on_error(&mut self, severity: &str, domain: &str, code: &str) -> (String, String, Option<f64>) {
        let sev = severity.to_ascii_lowercase();
        let dom = domain.to_ascii_lowercase();
        let code_u = code.to_ascii_uppercase();

        // Update counters
        self.consecutive_errors += 1;
        let dom_count = self.domain_consecutive.entry(dom.clone()).or_insert(0);
        *dom_count += 1;
        let dom_n = *dom_count;

        if dom == "execution" {
            self.execution_consecutive += 1;
        }

        // 1) FATAL severity -> STOP (if configured)
        if sev == "fatal" && self.cfg.stop_on_fatal {
            return (
                "stop".to_string(),
                format!("fatal: {}/{}", dom, code),
                None,
            );
        }

        // 2) INVARIANT domain -> always STOP
        if dom == "invariant" {
            return (
                "stop".to_string(),
                format!("invariant: {}", code),
                None,
            );
        }

        // 3) Execution errors: stricter threshold
        if dom == "execution" && self.execution_consecutive >= self.cfg.max_consecutive_execution_errors {
            return (
                "stop".to_string(),
                format!("execution errors >= {}", self.cfg.max_consecutive_execution_errors),
                None,
            );
        }

        // 4) Per-domain consecutive error threshold
        if dom_n >= self.cfg.max_consecutive_domain_errors {
            return (
                "stop".to_string(),
                format!("{} errors >= {}", dom, self.cfg.max_consecutive_domain_errors),
                None,
            );
        }

        // 5) Global consecutive error threshold
        if self.consecutive_errors >= self.cfg.max_consecutive_errors {
            return (
                "stop".to_string(),
                format!("consecutive errors >= {}", self.cfg.max_consecutive_errors),
                None,
            );
        }

        // 6) Retryable / timeout -> RETRY with delay
        if code_u == "RETRYABLE" || code_u == "TIMEOUT" {
            return (
                "retry".to_string(),
                format!("retry: {}/{}", dom, code),
                Some(self.cfg.default_retry_after_s),
            );
        }

        // 7) Default: DROP current event
        (
            "drop".to_string(),
            format!("drop on error: {}/{}", dom, code),
            None,
        )
    }

    /// Reset all internal counters (useful for testing).
    #[pyo3(signature = ())]
    fn reset(&mut self) {
        self.consecutive_errors = 0;
        self.domain_consecutive.clear();
        self.execution_consecutive = 0;
    }

    fn __repr__(&self) -> String {
        format!(
            "RustBasicGuard(consecutive_errors={}, execution_consecutive={}, domains={:?})",
            self.consecutive_errors,
            self.execution_consecutive,
            self.domain_consecutive.keys().collect::<Vec<_>>(),
        )
    }
}
