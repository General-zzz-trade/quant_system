// risk_engine.rs — Sprint C: Risk layer ported to Rust PyO3
//
// Contains: RustKillSwitch, RustCircuitBreaker, RustOrderLimiter, RustRiskGate
//
// Python sources:
//   risk/kill_switch.py, execution/safety/circuit_breaker.py,
//   execution/safety/limits.py, execution/safety/risk_gate.py

use pyo3::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

// ============================================================
// RustKillSwitch
// ============================================================

#[derive(Clone)]
struct KillRecord {
    mode: String,
    reason: String,
    armed_at: f64,
    ttl_seconds: Option<f64>,
    source: String,
}

impl KillRecord {
    fn is_expired(&self, now: f64) -> bool {
        match self.ttl_seconds {
            Some(ttl) => now >= self.armed_at + ttl,
            None => false,
        }
    }
}

#[pyclass]
pub struct RustKillSwitch {
    kills: HashMap<(String, String), KillRecord>,
}

#[pymethods]
impl RustKillSwitch {
    #[new]
    fn new() -> Self {
        Self {
            kills: HashMap::new(),
        }
    }

    /// Arm a kill switch at the given scope/key.
    #[pyo3(signature = (scope, key, mode, reason, *, ttl_seconds=None, source="risk".to_string(), now_ts=None))]
    fn arm(
        &mut self,
        scope: String,
        key: String,
        mode: String,
        reason: String,
        ttl_seconds: Option<f64>,
        source: String,
        now_ts: Option<f64>,
    ) {
        let real_key = if scope == "global" {
            "*".to_string()
        } else {
            key
        };
        let now = now_ts.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
        });
        self.kills.insert(
            (scope, real_key),
            KillRecord {
                mode,
                reason,
                armed_at: now,
                ttl_seconds,
                source,
            },
        );
    }

    /// Disarm a kill switch at the given scope/key.
    fn disarm(&mut self, scope: String, key: String) -> bool {
        let real_key = if scope == "global" {
            "*".to_string()
        } else {
            key
        };
        self.kills.remove(&(scope, real_key)).is_some()
    }

    /// Clear all kill switches.
    fn clear_all(&mut self) {
        self.kills.clear();
    }

    /// Check if any/specific kill is armed (prunes expired first).
    #[pyo3(signature = (*, scope=None, key=None, now_ts=None))]
    fn is_armed(
        &mut self,
        scope: Option<String>,
        key: Option<String>,
        now_ts: Option<f64>,
    ) -> bool {
        self.prune_expired(now_ts);
        match (scope, key) {
            (Some(s), Some(k)) => {
                let real_key = if s == "global" { "*".to_string() } else { k };
                self.kills.contains_key(&(s, real_key))
            }
            (Some(s), None) => self.kills.keys().any(|(ks, _)| ks == &s),
            (None, Some(k)) => self.kills.keys().any(|(_, kk)| kk == &k),
            (None, None) => !self.kills.is_empty(),
        }
    }

    /// Check if an order is allowed given current kill switches.
    /// Returns (allowed, reason_or_none).
    #[pyo3(signature = (*, symbol, strategy=None, reduce_only=false, now_ts=None))]
    fn allow_order(
        &mut self,
        symbol: String,
        strategy: Option<String>,
        reduce_only: bool,
        now_ts: Option<f64>,
    ) -> (bool, Option<String>) {
        self.prune_expired(now_ts);

        // Priority: GLOBAL > STRATEGY > SYMBOL
        let checks: Vec<(String, String)> = {
            let mut v = vec![("global".to_string(), "*".to_string())];
            if let Some(ref s) = strategy {
                v.push(("strategy".to_string(), s.clone()));
            }
            v.push(("symbol".to_string(), symbol));
            v
        };

        for (scope, key) in &checks {
            if let Some(rec) = self.kills.get(&(scope.clone(), key.clone())) {
                if rec.mode == "hard_kill" {
                    let reason = format!(
                        "kill_switch: scope={} key={} mode={} reason={}",
                        scope, key, rec.mode, rec.reason
                    );
                    return (false, Some(reason));
                }
                if rec.mode == "reduce_only" {
                    if reduce_only {
                        return (true, None);
                    }
                    let reason = format!(
                        "kill_switch: scope={} key={} mode=reduce_only (order not reduce_only)",
                        scope, key
                    );
                    return (false, Some(reason));
                }
                // Unknown mode — conservative block
                let reason = format!(
                    "kill_switch: scope={} key={} mode={} reason={}",
                    scope, key, rec.mode, rec.reason
                );
                return (false, Some(reason));
            }
        }

        (true, None)
    }

    /// Number of active kill records (after pruning expired).
    #[pyo3(signature = (*, now_ts=None))]
    fn active_count(&mut self, now_ts: Option<f64>) -> usize {
        self.prune_expired(now_ts);
        self.kills.len()
    }
}

impl RustKillSwitch {
    fn prune_expired(&mut self, now_ts: Option<f64>) {
        let now = now_ts.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
        });
        self.kills.retain(|_, rec| !rec.is_expired(now));
    }
}

// ============================================================
// RustCircuitBreaker
// ============================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum CBState {
    Closed,
    Open,
    HalfOpen,
}

impl CBState {
    fn as_str(&self) -> &'static str {
        match self {
            CBState::Closed => "closed",
            CBState::Open => "open",
            CBState::HalfOpen => "half_open",
        }
    }
}

#[pyclass]
pub struct RustCircuitBreaker {
    failure_threshold: usize,
    window_s: f64,
    recovery_timeout_s: f64,
    half_open_max: usize,
    state: CBState,
    failures: Vec<f64>,       // monotonic timestamps
    open_since: f64,
    half_open_count: usize,
    start: Instant,
}

impl RustCircuitBreaker {
    fn mono_now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn maybe_transition(&mut self) {
        if self.state == CBState::Open {
            let elapsed = self.mono_now() - self.open_since;
            if elapsed >= self.recovery_timeout_s {
                self.state = CBState::HalfOpen;
                self.half_open_count = 0;
            }
        }
    }
}

#[pymethods]
impl RustCircuitBreaker {
    #[new]
    #[pyo3(signature = (failure_threshold=5, window_s=60.0, recovery_timeout_s=30.0, half_open_max=1))]
    fn new(
        failure_threshold: usize,
        window_s: f64,
        recovery_timeout_s: f64,
        half_open_max: usize,
    ) -> Self {
        Self {
            failure_threshold,
            window_s,
            recovery_timeout_s,
            half_open_max,
            state: CBState::Closed,
            failures: Vec::new(),
            open_since: 0.0,
            half_open_count: 0,
            start: Instant::now(),
        }
    }

    fn record_success(&mut self) {
        if self.state == CBState::HalfOpen {
            self.state = CBState::Closed;
        }
        self.failures.clear();
        self.half_open_count = 0;
    }

    fn record_failure(&mut self) {
        let now = self.mono_now();
        self.failures.push(now);
        let cutoff = now - self.window_s;
        self.failures.retain(|&t| t > cutoff);

        match self.state {
            CBState::HalfOpen => {
                self.state = CBState::Open;
                self.open_since = now;
                self.half_open_count = 0;
            }
            CBState::Closed => {
                if self.failures.len() >= self.failure_threshold {
                    self.state = CBState::Open;
                    self.open_since = now;
                }
            }
            CBState::Open => {}
        }
    }

    fn allow_request(&mut self) -> bool {
        self.maybe_transition();
        match self.state {
            CBState::Closed => true,
            CBState::HalfOpen => {
                if self.half_open_count < self.half_open_max {
                    self.half_open_count += 1;
                    true
                } else {
                    false
                }
            }
            CBState::Open => false,
        }
    }

    #[getter]
    fn state(&mut self) -> String {
        self.maybe_transition();
        self.state.as_str().to_string()
    }

    fn reset(&mut self) {
        self.state = CBState::Closed;
        self.failures.clear();
        self.open_since = 0.0;
        self.half_open_count = 0;
    }

    /// Returns (state_str, failure_count, open_since).
    fn snapshot(&mut self) -> (String, usize, f64) {
        self.maybe_transition();
        (
            self.state.as_str().to_string(),
            self.failures.len(),
            self.open_since,
        )
    }
}

// ============================================================
// RustOrderLimiter
// ============================================================

#[pyclass]
pub struct RustOrderLimiter {
    max_order_qty: Option<f64>,
    max_order_notional: Option<f64>,
    max_position_notional: Option<f64>,
    max_orders_per_sec: Option<f64>,
    max_daily_orders: Option<u64>,
    max_daily_notional: Option<f64>,

    order_timestamps: Vec<f64>,    // monotonic timestamps for rate limiting
    daily_order_count: u64,
    daily_notional: f64,
    day_start: f64,
    start: Instant,
}

impl RustOrderLimiter {
    fn mono_now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn maybe_reset_daily(&mut self, now: f64) {
        if now - self.day_start > 86400.0 {
            self.daily_order_count = 0;
            self.daily_notional = 0.0;
            self.day_start = now;
        }
    }
}

#[pymethods]
impl RustOrderLimiter {
    #[new]
    #[pyo3(signature = (
        max_order_qty=None,
        max_order_notional=None,
        max_position_notional=None,
        max_orders_per_sec=None,
        max_daily_orders=None,
        max_daily_notional=None,
    ))]
    fn new(
        max_order_qty: Option<f64>,
        max_order_notional: Option<f64>,
        max_position_notional: Option<f64>,
        max_orders_per_sec: Option<f64>,
        max_daily_orders: Option<u64>,
        max_daily_notional: Option<f64>,
    ) -> Self {
        let start = Instant::now();
        Self {
            max_order_qty,
            max_order_notional,
            max_position_notional,
            max_orders_per_sec,
            max_daily_orders,
            max_daily_notional,
            order_timestamps: Vec::new(),
            daily_order_count: 0,
            daily_notional: 0.0,
            day_start: start.elapsed().as_secs_f64(),
            start,
        }
    }

    /// Pre-flight check. Returns (allowed, reason_or_none).
    #[pyo3(signature = (*, qty, price, current_position_notional=0.0))]
    fn check(
        &mut self,
        qty: f64,
        price: f64,
        current_position_notional: f64,
    ) -> (bool, Option<String>) {
        let now = self.mono_now();
        let notional = qty * price;

        self.maybe_reset_daily(now);

        // max_order_qty
        if let Some(max_q) = self.max_order_qty {
            if qty > max_q {
                return (
                    false,
                    Some(format!("max_order_qty: qty={} > max={}", qty, max_q)),
                );
            }
        }

        // max_order_notional
        if let Some(max_n) = self.max_order_notional {
            if notional > max_n {
                return (
                    false,
                    Some(format!(
                        "max_order_notional: notional={:.2} > max={:.2}",
                        notional, max_n
                    )),
                );
            }
        }

        // max_position_notional
        if let Some(max_p) = self.max_position_notional {
            let projected = current_position_notional + notional;
            if projected > max_p {
                return (
                    false,
                    Some(format!(
                        "max_position_notional: projected={:.2} > max={:.2}",
                        projected, max_p
                    )),
                );
            }
        }

        // max_orders_per_second (rate limit)
        if let Some(max_rate) = self.max_orders_per_sec {
            let cutoff = now - 1.0;
            self.order_timestamps.retain(|&t| t > cutoff);
            if self.order_timestamps.len() as f64 >= max_rate {
                return (
                    false,
                    Some(format!(
                        "max_orders_per_second: rate={}/s",
                        self.order_timestamps.len()
                    )),
                );
            }
        }

        // max_daily_orders
        if let Some(max_d) = self.max_daily_orders {
            if self.daily_order_count >= max_d {
                return (
                    false,
                    Some(format!(
                        "max_daily_orders: count={}",
                        self.daily_order_count
                    )),
                );
            }
        }

        // max_daily_notional
        if let Some(max_dn) = self.max_daily_notional {
            if self.daily_notional + notional > max_dn {
                return (
                    false,
                    Some(format!(
                        "max_daily_notional: total={:.2}",
                        self.daily_notional + notional
                    )),
                );
            }
        }

        (true, None)
    }

    /// Record an order (after it passes check and is sent).
    fn record_order(&mut self, notional: f64) {
        let now = self.mono_now();
        self.order_timestamps.push(now);
        self.daily_order_count += 1;
        self.daily_notional += notional;
    }

    /// Reset daily counters.
    fn reset_daily(&mut self) {
        self.daily_order_count = 0;
        self.daily_notional = 0.0;
        self.day_start = self.mono_now();
    }
}

// ============================================================
// RustRiskGate
// ============================================================

#[pyclass]
pub struct RustRiskGate {
    max_open_orders: u32,
    max_order_notional: Option<f64>,
    max_position_notional: Option<f64>,
    max_portfolio_notional: Option<f64>,
}

#[pymethods]
impl RustRiskGate {
    #[new]
    #[pyo3(signature = (
        max_open_orders=10,
        max_order_notional=None,
        max_position_notional=None,
        max_portfolio_notional=None,
    ))]
    fn new(
        max_open_orders: u32,
        max_order_notional: Option<f64>,
        max_position_notional: Option<f64>,
        max_portfolio_notional: Option<f64>,
    ) -> Self {
        Self {
            max_open_orders,
            max_order_notional,
            max_position_notional,
            max_portfolio_notional,
        }
    }

    /// Pre-execution risk gate check.
    /// Returns (allowed, reason_or_none).
    #[pyo3(signature = (
        *,
        symbol,
        side,
        qty,
        price,
        open_order_count=0,
        position_notional=0.0,
        portfolio_notional=0.0,
        kill_switch_armed=false,
    ))]
    #[allow(unused_variables)]
    fn check(
        &self,
        symbol: &str,
        side: &str,
        qty: f64,
        price: f64,
        open_order_count: u32,
        position_notional: f64,
        portfolio_notional: f64,
        kill_switch_armed: bool,
    ) -> (bool, Option<String>) {
        // 1. Kill switch
        if kill_switch_armed {
            return (false, Some("kill_switch_active".to_string()));
        }

        // 2. Open order limit
        if open_order_count >= self.max_open_orders {
            return (
                false,
                Some(format!(
                    "max_open_orders: {}>={}",
                    open_order_count, self.max_open_orders
                )),
            );
        }

        let notional = (qty * price).abs();

        // 3. Order notional
        if let Some(max_on) = self.max_order_notional {
            if notional > max_on {
                return (
                    false,
                    Some(format!(
                        "order_notional: {:.2}>{:.2}",
                        notional, max_on
                    )),
                );
            }
        }

        // 4. Position notional
        if let Some(max_pn) = self.max_position_notional {
            let projected = position_notional + notional;
            if projected > max_pn {
                return (
                    false,
                    Some(format!(
                        "position_notional: {:.2}>{:.2}",
                        projected, max_pn
                    )),
                );
            }
        }

        // 5. Portfolio notional
        if let Some(max_port) = self.max_portfolio_notional {
            let projected = portfolio_notional + notional;
            if projected > max_port {
                return (
                    false,
                    Some(format!(
                        "portfolio_notional: {:.2}>{:.2}",
                        projected, max_port
                    )),
                );
            }
        }

        (true, None)
    }
}
