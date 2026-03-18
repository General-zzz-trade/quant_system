// risk_aggregator.rs — PyO3-exposed RustRiskAggregator
//
// Aggregates multiple RiskRule evaluations with worst-verdict-wins semantics.
// Thread-safe stats tracking via Mutex.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::risk_rules::*;

// ---------------------------------------------------------------------------
// EvalContext::from_pydict
// ---------------------------------------------------------------------------

fn get_f64(d: &Bound<'_, PyDict>, key: &str, default: f64) -> f64 {
    match d.get_item(key) {
        Ok(Some(v)) => v.extract::<f64>().unwrap_or(default),
        _ => default,
    }
}

fn get_u32(d: &Bound<'_, PyDict>, key: &str, default: u32) -> u32 {
    match d.get_item(key) {
        Ok(Some(v)) => v.extract::<u32>().unwrap_or(default),
        _ => default,
    }
}

fn get_bool(d: &Bound<'_, PyDict>, key: &str, default: bool) -> bool {
    match d.get_item(key) {
        Ok(Some(v)) => v.extract::<bool>().unwrap_or(default),
        _ => default,
    }
}

fn get_string(d: &Bound<'_, PyDict>, key: &str, default: &str) -> String {
    match d.get_item(key) {
        Ok(Some(v)) => v.extract::<String>().unwrap_or_else(|_| default.to_string()),
        _ => default.to_string(),
    }
}

impl EvalContext {
    fn from_pydict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let symbol = get_string(d, "symbol", "");
        let side = get_string(d, "side", "buy").to_lowercase();
        let qty = get_f64(d, "qty", 0.0);
        let price = get_f64(d, "price", 0.0);
        let notional_raw = get_f64(d, "notional", f64::NAN);
        let notional = if notional_raw.is_finite() {
            notional_raw
        } else {
            qty * price
        };

        Ok(EvalContext {
            symbol,
            side,
            qty,
            price,
            notional,
            current_position_qty: get_f64(d, "current_position_qty", 0.0),
            current_position_notional: get_f64(d, "current_position_notional", 0.0),
            account_equity: get_f64(d, "account_equity", 0.0),
            gross_exposure: get_f64(d, "gross_exposure", 0.0),
            net_exposure: get_f64(d, "net_exposure", 0.0),
            max_symbol_concentration: get_f64(d, "max_symbol_concentration", 0.0),
            drawdown_pct: get_f64(d, "drawdown_pct", 0.0),
            recent_order_count: get_u32(d, "recent_order_count", 0),
            recent_order_window_secs: get_f64(d, "recent_order_window_secs", 0.0),
            avg_correlation: get_f64(d, "avg_correlation", f64::NAN),
            portfolio_var_pct: get_f64(d, "portfolio_var_pct", f64::NAN),
            is_reduce_only: get_bool(d, "is_reduce_only", false),
        })
    }
}

// ---------------------------------------------------------------------------
// RuleStats
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct RuleStats {
    calls: u64,
    allows: u64,
    rejects: u64,
    reduces: u64,
}

// ---------------------------------------------------------------------------
// RustRiskAggregator
// ---------------------------------------------------------------------------

#[pyclass(name = "RustRiskAggregator")]
pub struct RustRiskAggregator {
    rules: Vec<Box<dyn RiskRule>>,
    rule_names: Vec<String>,
    enabled: Vec<bool>,
    stats: Mutex<HashMap<String, RuleStats>>,
}

#[pymethods]
impl RustRiskAggregator {
    #[new]
    fn new() -> Self {
        RustRiskAggregator {
            rules: Vec::new(),
            rule_names: Vec::new(),
            enabled: Vec::new(),
            stats: Mutex::new(HashMap::new()),
        }
    }

    /// Add a rule by type name and config dict.
    ///
    /// rule_type: "max_position", "leverage_cap", "max_drawdown",
    ///            "portfolio_limits", "order_frequency", "correlation_limit", "var_limit"
    fn add_rule(
        &mut self,
        name: &str,
        rule_type: &str,
        config: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let rule: Box<dyn RiskRule> = match rule_type {
            "max_position" => Box::new(MaxPositionRule {
                name: name.to_string(),
                max_qty: get_f64(config, "max_qty", f64::MAX),
                max_notional: get_f64(config, "max_notional", f64::MAX),
            }),
            "leverage_cap" => Box::new(LeverageCapRule {
                name: name.to_string(),
                max_gross_leverage: get_f64(config, "max_gross_leverage", 3.0),
                max_net_leverage: get_f64(config, "max_net_leverage", 1.0),
            }),
            "max_drawdown" => Box::new(MaxDrawdownRule {
                name: name.to_string(),
                warning_pct: get_f64(config, "warning_pct", 0.15),
                kill_pct: get_f64(config, "kill_pct", 0.20),
            }),
            "portfolio_limits" => Box::new(PortfolioLimitsRule {
                name: name.to_string(),
                max_concentration: get_f64(config, "max_concentration", 0.4),
                max_gross_exposure: get_f64(config, "max_gross_exposure", 3.0),
            }),
            "order_frequency" => Box::new(OrderFrequencyRule::new(
                name.to_string(),
                get_f64(config, "max_per_minute", 30.0) as u32,
                get_f64(config, "window_secs", 300.0),
            )),
            "correlation_limit" => Box::new(CorrelationLimitRule {
                name: name.to_string(),
                max_avg_correlation: get_f64(config, "max_avg_correlation", 0.7),
            }),
            "var_limit" => Box::new(VaRLimitRule {
                name: name.to_string(),
                max_var_pct: get_f64(config, "max_var_pct", 5.0),
            }),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown rule_type: '{}'. Expected one of: max_position, leverage_cap, \
                     max_drawdown, portfolio_limits, order_frequency, correlation_limit, var_limit",
                    rule_type
                )));
            }
        };

        self.rule_names.push(name.to_string());
        self.rules.push(rule);
        self.enabled.push(true);

        // Initialize stats
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.entry(name.to_string()).or_default();

        Ok(())
    }

    /// Enable a rule by name
    fn enable(&mut self, name: &str) {
        for (i, n) in self.rule_names.iter().enumerate() {
            if n == name {
                self.enabled[i] = true;
            }
        }
    }

    /// Disable a rule by name
    fn disable(&mut self, name: &str) {
        for (i, n) in self.rule_names.iter().enumerate() {
            if n == name {
                self.enabled[i] = false;
            }
        }
    }

    /// Evaluate all enabled rules against context dict.
    ///
    /// Returns (verdict_str, list_of_reason_dicts)
    /// verdict_str: "allow", "reject", or "reduce"
    /// Each reason dict: {"rule": str, "reason": str, "factor": float_or_None}
    fn evaluate(&self, py: Python<'_>, ctx: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let eval_ctx = EvalContext::from_pydict(ctx)?;

        // Validate critical inputs: NaN price → immediate reject
        if !eval_ctx.price.is_finite() || !eval_ctx.qty.is_finite() {
            if !eval_ctx.is_reduce_only {
                let reasons = PyList::empty(py);
                let d = PyDict::new(py);
                d.set_item("rule", "input_validation")?;
                d.set_item(
                    "reason",
                    format!(
                        "Non-finite critical inputs: price={}, qty={}",
                        eval_ctx.price, eval_ctx.qty
                    ),
                )?;
                d.set_item("factor", py.None())?;
                reasons.append(d)?;

                let result = PyTuple::new(py, &[
                    PyString::new(py, "reject").into_any(),
                    reasons.into_any(),
                ])?;
                return Ok(result.into());
            }
        }

        // Collect verdicts from all enabled rules
        // Priority: reject(2) > reduce(1) > allow(0)
        let mut worst_priority: u8 = 0; // 0=allow
        let mut all_reasons: Vec<(String, String, Option<f64>)> = Vec::new();
        let mut min_factor: f64 = 1.0;

        for (i, rule) in self.rules.iter().enumerate() {
            if !self.enabled[i] {
                continue;
            }

            let verdict = rule.evaluate(&eval_ctx);
            let rule_name = rule.name().to_string();

            // Update stats
            {
                let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
                let entry = stats.entry(rule_name.clone()).or_default();
                entry.calls += 1;

                match &verdict {
                    RiskVerdict::Allow => {
                        entry.allows += 1;
                    }
                    RiskVerdict::Reject { .. } => {
                        entry.rejects += 1;
                    }
                    RiskVerdict::Reduce { .. } => {
                        entry.reduces += 1;
                    }
                }
            }

            match verdict {
                RiskVerdict::Allow => {
                    // No action needed
                }
                RiskVerdict::Reject { rule, reason } => {
                    if worst_priority < 2 {
                        worst_priority = 2;
                    }
                    all_reasons.push((rule, reason, None));
                }
                RiskVerdict::Reduce {
                    rule,
                    factor,
                    reason,
                } => {
                    if worst_priority < 1 {
                        worst_priority = 1;
                    }
                    if factor < min_factor {
                        min_factor = factor;
                    }
                    all_reasons.push((rule, reason, Some(factor)));
                }
            }
        }

        // Build result
        let verdict_str = match worst_priority {
            2 => "reject",
            1 => "reduce",
            _ => "allow",
        };

        let reasons_list = PyList::empty(py);
        for (rule, reason, factor) in &all_reasons {
            let d = PyDict::new(py);
            d.set_item("rule", rule)?;
            d.set_item("reason", reason)?;
            match factor {
                Some(f) => d.set_item("factor", *f)?,
                None => d.set_item("factor", py.None())?,
            }
            reasons_list.append(d)?;
        }

        let result = PyTuple::new(py, &[
            PyString::new(py, verdict_str).into_any(),
            reasons_list.into_any(),
        ])?;
        Ok(result.into())
    }

    /// Return stats snapshot as dict: {rule_name: {"calls": int, "allows": int, ...}}
    fn snapshot(&self, py: Python<'_>) -> PyResult<PyObject> {
        let outer = PyDict::new(py);
        let stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        for (name, s) in stats.iter() {
            let inner = PyDict::new(py);
            inner.set_item("calls", s.calls)?;
            inner.set_item("allows", s.allows)?;
            inner.set_item("rejects", s.rejects)?;
            inner.set_item("reduces", s.reduces)?;
            outer.set_item(name, inner)?;
        }
        Ok(outer.into())
    }

    /// Reset all stats counters
    fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        for s in stats.values_mut() {
            *s = RuleStats::default();
        }
    }

    /// Total number of rules
    fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Number of enabled rules
    fn enabled_count(&self) -> usize {
        self.enabled.iter().filter(|&&e| e).count()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregator_basic() {
        // Just test internal logic without PyO3
        let mut agg = RustRiskAggregator::new();
        assert_eq!(agg.rule_count(), 0);
        assert_eq!(agg.enabled_count(), 0);

        // Manually add a rule
        agg.rules.push(Box::new(MaxPositionRule {
            name: "test_pos".to_string(),
            max_qty: 100.0,
            max_notional: 1_000_000.0,
        }));
        agg.rule_names.push("test_pos".to_string());
        agg.enabled.push(true);

        assert_eq!(agg.rule_count(), 1);
        assert_eq!(agg.enabled_count(), 1);

        agg.disable("test_pos");
        assert_eq!(agg.enabled_count(), 0);

        agg.enable("test_pos");
        assert_eq!(agg.enabled_count(), 1);
    }

    #[test]
    fn test_eval_context_reducing_exposure() {
        let ctx = EvalContext {
            symbol: "ETHUSDT".to_string(),
            side: "sell".to_string(),
            qty: 1.0,
            price: 3000.0,
            notional: 3000.0,
            current_position_qty: 5.0,
            current_position_notional: 15000.0,
            account_equity: 10000.0,
            gross_exposure: 15000.0,
            net_exposure: 15000.0,
            max_symbol_concentration: 1.0,
            drawdown_pct: 0.0,
            recent_order_count: 0,
            recent_order_window_secs: 0.0,
            avg_correlation: 0.0,
            portfolio_var_pct: 0.0,
            is_reduce_only: false,
        };
        // Selling when long → reducing
        assert!(ctx.is_reducing_exposure());
    }

    #[test]
    fn test_eval_context_not_reducing() {
        let ctx = EvalContext {
            symbol: "ETHUSDT".to_string(),
            side: "buy".to_string(),
            qty: 1.0,
            price: 3000.0,
            notional: 3000.0,
            current_position_qty: 5.0,
            current_position_notional: 15000.0,
            account_equity: 10000.0,
            gross_exposure: 15000.0,
            net_exposure: 15000.0,
            max_symbol_concentration: 1.0,
            drawdown_pct: 0.0,
            recent_order_count: 0,
            recent_order_window_secs: 0.0,
            avg_correlation: 0.0,
            portfolio_var_pct: 0.0,
            is_reduce_only: false,
        };
        // Buying when already long → NOT reducing
        assert!(!ctx.is_reducing_exposure());
    }
}
