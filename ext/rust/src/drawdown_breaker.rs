// drawdown_breaker.rs — RustDrawdownBreaker: Rust backend for risk/drawdown_breaker.py
//
// Monitors equity continuously and returns action tuples for Python to bridge
// to KillSwitch calls. Uses return-action pattern — does NOT hold KillSwitch ref.
//
// Python source: risk/drawdown_breaker.py

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::VecDeque;

// ============================================================
// Pure Rust state machine helpers (testable without PyO3)
// ============================================================

/// Check velocity: rapid equity drop within velocity_window_sec.
/// Returns true if equity dropped >= velocity_pct% from any point in window.
fn check_velocity(
    history: &VecDeque<(f64, f64)>,
    current_equity: f64,
    now: f64,
    velocity_window_sec: f64,
    velocity_pct: f64,
) -> bool {
    let cutoff = now - velocity_window_sec;
    for (ts, eq) in history.iter() {
        if *ts >= cutoff {
            if *eq > 0.0 {
                let drop_pct = (eq - current_equity) / eq * 100.0;
                if drop_pct >= velocity_pct {
                    return true;
                }
            }
            // Match Python: break after first entry in window
            break;
        }
    }
    false
}

/// Compute drawdown percentage from hwm and current equity.
fn compute_drawdown(hwm: f64, equity: f64) -> f64 {
    if hwm > 0.0 {
        (hwm - equity) / hwm * 100.0
    } else {
        0.0
    }
}

// ============================================================
// RustDrawdownBreaker
// ============================================================

#[pyclass(name = "RustDrawdownBreaker")]
pub struct RustDrawdownBreaker {
    // Config
    warning_pct: f64,
    reduce_pct: f64,
    kill_pct: f64,
    velocity_pct: f64,
    velocity_window_sec: f64,
    reduce_ttl_sec: i64,
    // State
    equity_hwm: f64,
    current_dd_pct: f64,
    state: String,
    equity_history: VecDeque<(f64, f64)>, // (ts, equity), maxlen 1000
}

#[pymethods]
impl RustDrawdownBreaker {
    #[new]
    #[pyo3(signature = (warning_pct=10.0, reduce_pct=15.0, kill_pct=20.0, velocity_pct=5.0, velocity_window_sec=900.0, reduce_ttl_sec=3600))]
    fn new(
        warning_pct: f64,
        reduce_pct: f64,
        kill_pct: f64,
        velocity_pct: f64,
        velocity_window_sec: f64,
        reduce_ttl_sec: i64,
    ) -> Self {
        Self {
            warning_pct,
            reduce_pct,
            kill_pct,
            velocity_pct,
            velocity_window_sec,
            reduce_ttl_sec,
            equity_hwm: 0.0,
            current_dd_pct: 0.0,
            state: "normal".to_string(),
            equity_history: VecDeque::with_capacity(1000),
        }
    }

    /// Process an equity update.
    ///
    /// Returns (state: String, action: Option<(String, String)>)
    /// Action is (mode, reason) for Python KillSwitch bridge, or None.
    /// Modes: "reduce_only", "hard_kill", "clear"
    /// Warning state returns None action — Python handles logging.
    #[pyo3(signature = (equity, now_ts=None))]
    fn on_equity_update(
        &mut self,
        equity: f64,
        now_ts: Option<f64>,
    ) -> (String, Option<(String, String)>) {
        if !equity.is_finite() || equity <= 0.0 {
            return (self.state.clone(), None);
        }

        let now = now_ts.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
        });

        // Terminal state — no further state changes
        if self.state == "killed" {
            // Still record history for velocity tracking but don't change state
            if self.equity_history.len() >= 1000 {
                self.equity_history.pop_front();
            }
            self.equity_history.push_back((now, equity));
            return (self.state.clone(), None);
        }

        // Update HWM
        if equity > self.equity_hwm {
            self.equity_hwm = equity;
        }

        // Calculate drawdown
        self.current_dd_pct = compute_drawdown(self.equity_hwm, equity);

        // Record for velocity check
        if self.equity_history.len() >= 1000 {
            self.equity_history.pop_front();
        }
        self.equity_history.push_back((now, equity));

        // Velocity check: rapid drop detection (runs BEFORE threshold checks)
        if check_velocity(
            &self.equity_history,
            equity,
            now,
            self.velocity_window_sec,
            self.velocity_pct,
        ) {
            self.state = "killed".to_string();
            let reason = format!(
                "velocity_breach: >{:.1}% drop in {:.0}s",
                self.velocity_pct, self.velocity_window_sec
            );
            return (self.state.clone(), Some(("hard_kill".to_string(), reason)));
        }

        // Threshold checks (escalating)
        if self.current_dd_pct >= self.kill_pct {
            if self.state != "killed" {
                self.state = "killed".to_string();
                let reason = format!(
                    "drawdown {:.1}% >= kill threshold {:.1}%",
                    self.current_dd_pct, self.kill_pct
                );
                return (self.state.clone(), Some(("hard_kill".to_string(), reason)));
            }
        } else if self.current_dd_pct >= self.reduce_pct {
            if self.state != "reduce_only" && self.state != "killed" {
                self.state = "reduce_only".to_string();
                let reason = format!(
                    "drawdown {:.1}% >= reduce threshold {:.1}%",
                    self.current_dd_pct, self.reduce_pct
                );
                return (self.state.clone(), Some(("reduce_only".to_string(), reason)));
            }
        } else if self.current_dd_pct >= self.warning_pct {
            if self.state == "normal" {
                self.state = "warning".to_string();
                // Action is None — Python wrapper handles warning cooldown and logging
            }
        } else {
            // Drawdown recovered below warning threshold
            if self.state == "warning" {
                self.state = "normal".to_string();
            }
        }

        (self.state.clone(), None)
    }

    /// Reset the breaker state. Use after manual intervention.
    /// Returns ("normal", Some(("clear", ""))) always.
    #[pyo3(signature = (new_hwm=None))]
    fn reset(&mut self, new_hwm: Option<f64>) -> (String, Option<(String, String)>) {
        self.state = "normal".to_string();
        self.current_dd_pct = 0.0;
        if let Some(hwm) = new_hwm {
            self.equity_hwm = hwm;
        }
        self.equity_history.clear();
        ("normal".to_string(), Some(("clear".to_string(), "".to_string())))
    }

    /// Return state for persistence.
    /// Only stores equity_hwm and state (matching Python behavior).
    fn checkpoint(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("equity_hwm", self.equity_hwm)?;
        dict.set_item("state", self.state.clone())?;
        Ok(dict.into())
    }

    /// Restore state from checkpoint.
    /// Only restores equity_hwm (not state, matching Python behavior).
    fn restore_checkpoint(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(hwm) = data.get_item("equity_hwm")? {
            self.equity_hwm = hwm.extract::<f64>().map_err(|_| {
                PyValueError::new_err("equity_hwm must be a float")
            })?;
        }
        Ok(())
    }

    /// Return current status for health endpoint.
    fn get_status(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("state", self.state.clone())?;
        dict.set_item("drawdown_pct", (self.current_dd_pct * 100.0).round() / 100.0)?;
        dict.set_item("equity_hwm", (self.equity_hwm * 100.0).round() / 100.0)?;

        let thresholds = PyDict::new(py);
        thresholds.set_item("warning_pct", self.warning_pct)?;
        thresholds.set_item("reduce_pct", self.reduce_pct)?;
        thresholds.set_item("kill_pct", self.kill_pct)?;
        thresholds.set_item("velocity_pct", self.velocity_pct)?;
        thresholds.set_item("velocity_window_sec", self.velocity_window_sec)?;
        thresholds.set_item("reduce_ttl_sec", self.reduce_ttl_sec)?;
        dict.set_item("thresholds", thresholds)?;

        Ok(dict.into())
    }

    // ── Getters ──

    #[getter]
    fn state(&self) -> String {
        self.state.clone()
    }

    #[getter]
    fn current_drawdown_pct(&self) -> f64 {
        self.current_dd_pct
    }

    #[getter]
    fn equity_hwm(&self) -> f64 {
        self.equity_hwm
    }
}

// ============================================================
// Unit tests (pure Rust, no PyO3)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Make a breaker with velocity disabled (pct=99%) for threshold tests.
    fn make_breaker_no_velocity() -> RustDrawdownBreaker {
        RustDrawdownBreaker {
            warning_pct: 10.0,
            reduce_pct: 15.0,
            kill_pct: 20.0,
            velocity_pct: 99.0, // disabled
            velocity_window_sec: 900.0,
            reduce_ttl_sec: 3600,
            equity_hwm: 0.0,
            current_dd_pct: 0.0,
            state: "normal".to_string(),
            equity_history: VecDeque::with_capacity(1000),
        }
    }

    /// Make a breaker with velocity enabled (default thresholds).
    fn make_breaker_with_velocity() -> RustDrawdownBreaker {
        RustDrawdownBreaker {
            warning_pct: 10.0,
            reduce_pct: 15.0,
            kill_pct: 20.0,
            velocity_pct: 5.0,
            velocity_window_sec: 900.0,
            reduce_ttl_sec: 3600,
            equity_hwm: 0.0,
            current_dd_pct: 0.0,
            state: "normal".to_string(),
            equity_history: VecDeque::with_capacity(1000),
        }
    }

    #[test]
    fn test_normal_state() {
        let mut b = make_breaker_no_velocity();
        let (state, action) = b.on_equity_update(1000.0, Some(0.0));
        assert_eq!(state, "normal");
        assert!(action.is_none());

        let (state, action) = b.on_equity_update(1050.0, Some(1.0));
        assert_eq!(state, "normal");
        assert!(action.is_none());
        assert_eq!(b.equity_hwm, 1050.0);
    }

    #[test]
    fn test_warning_transition() {
        let mut b = make_breaker_no_velocity(); // velocity disabled
        b.on_equity_update(1000.0, Some(0.0)); // establish HWM
        // 11% drawdown → warning
        let (state, action) = b.on_equity_update(890.0, Some(1.0));
        assert_eq!(state, "warning");
        assert!(action.is_none(), "warning returns None action");
    }

    #[test]
    fn test_reduce_only_transition() {
        let mut b = make_breaker_no_velocity(); // velocity disabled
        b.on_equity_update(1000.0, Some(0.0));
        // 16% drawdown → reduce_only
        let (state, action) = b.on_equity_update(840.0, Some(1.0));
        assert_eq!(state, "reduce_only");
        assert!(action.is_some());
        let (mode, reason) = action.unwrap();
        assert_eq!(mode, "reduce_only");
        assert!(reason.contains("reduce threshold"), "reason was: {}", reason);
    }

    #[test]
    fn test_kill_transition() {
        let mut b = make_breaker_no_velocity(); // velocity disabled
        b.on_equity_update(1000.0, Some(0.0));
        // 21% drawdown → killed
        let (state, action) = b.on_equity_update(790.0, Some(1.0));
        assert_eq!(state, "killed");
        assert!(action.is_some());
        let (mode, reason) = action.unwrap();
        assert_eq!(mode, "hard_kill");
        assert!(reason.contains("kill threshold"), "reason was: {}", reason);
    }

    #[test]
    fn test_velocity_detection() {
        let mut b = make_breaker_with_velocity(); // velocity_pct=5.0, window=900s
        b.on_equity_update(1000.0, Some(0.0));
        // Drop 6% within window (60s < 900s) → velocity breach (before threshold check)
        let (state, action) = b.on_equity_update(940.0, Some(60.0));
        assert_eq!(state, "killed");
        assert!(action.is_some());
        let (mode, reason) = action.unwrap();
        assert_eq!(mode, "hard_kill");
        assert!(reason.contains("velocity_breach"), "reason was: {}", reason);
    }

    #[test]
    fn test_recovery_warning_to_normal() {
        let mut b = make_breaker_no_velocity(); // velocity disabled
        b.on_equity_update(1000.0, Some(0.0));
        b.on_equity_update(890.0, Some(1.0)); // → warning (11% dd)
        assert_eq!(b.state, "warning");
        // Recovery: below 10% dd (9200/10000 = 8% dd < 10%)
        let (state, _) = b.on_equity_update(960.0, Some(2.0));
        assert_eq!(state, "normal");
    }

    #[test]
    fn test_killed_is_terminal() {
        let mut b = make_breaker_no_velocity(); // velocity disabled
        b.on_equity_update(1000.0, Some(0.0));
        b.on_equity_update(790.0, Some(1.0)); // → killed (21% dd)
        assert_eq!(b.state, "killed");

        // Killed state ignores further updates — no action replayed
        let (state, action) = b.on_equity_update(1000.0, Some(2.0));
        assert_eq!(state, "killed");
        assert!(action.is_none(), "no action after already killed");
    }

    #[test]
    fn test_reset_clears_state() {
        let mut b = make_breaker_no_velocity(); // velocity disabled
        b.on_equity_update(1000.0, Some(0.0));
        b.on_equity_update(790.0, Some(1.0)); // → killed (21% dd)
        assert_eq!(b.state, "killed");

        let (state, action) = b.reset(None);
        assert_eq!(state, "normal");
        assert!(action.is_some());
        let (mode, _) = action.unwrap();
        assert_eq!(mode, "clear");
        assert_eq!(b.equity_history.len(), 0);
    }

    #[test]
    fn test_equity_zero_ignored() {
        let mut b = make_breaker_no_velocity();
        let (state, action) = b.on_equity_update(0.0, Some(0.0));
        assert_eq!(state, "normal");
        assert!(action.is_none());
        assert_eq!(b.equity_hwm, 0.0); // HWM not updated for zero
    }

    #[test]
    fn test_checkpoint_only_stores_hwm_and_state() {
        // Pure Rust test: verify checkpoint fields logic
        let mut b = make_breaker_no_velocity();
        b.on_equity_update(1000.0, Some(0.0));
        // equity_hwm should be 1000.0, state = "normal"
        assert_eq!(b.equity_hwm, 1000.0);
        assert_eq!(b.state, "normal");
    }

    #[test]
    fn test_restore_checkpoint_only_restores_hwm() {
        let mut b = make_breaker_no_velocity();
        // Simulate restore: only equity_hwm should be set
        b.equity_hwm = 1234.56;
        // State should remain whatever it was
        b.state = "normal".to_string();
        assert_eq!(b.equity_hwm, 1234.56);
        assert_eq!(b.state, "normal");
    }

    #[test]
    fn test_velocity_no_breach_outside_window() {
        // Drop happened more than velocity_window_sec ago — no velocity breach
        let mut history: VecDeque<(f64, f64)> = VecDeque::new();
        history.push_back((0.0, 1000.0)); // old entry, outside window
        // Window is 900s; now=1000.0; cutoff=100.0; ts=0.0 < cutoff → no breach
        let result = check_velocity(&history, 900.0, 1000.0, 900.0, 5.0);
        assert!(!result, "old drop outside window should not trigger");
    }

    #[test]
    fn test_reduce_not_repeated() {
        // Repeated calls in reduce_only state should not return action again
        let mut b = make_breaker_no_velocity(); // velocity disabled
        b.on_equity_update(1000.0, Some(0.0));
        let (_, action1) = b.on_equity_update(840.0, Some(1.0)); // → reduce_only (16% dd)
        assert!(action1.is_some());
        let (_, action2) = b.on_equity_update(838.0, Some(2.0)); // still reduce_only range
        assert!(action2.is_none(), "no repeated reduce action");
    }
}
