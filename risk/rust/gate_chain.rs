// gate_chain.rs — RustGateChain: configurable gate chain orchestrator
//
// Python source: runner/gate_chain.py + runner/gates/*.py
//
// Design: Instead of reimplementing all 13 gates in Rust (many depend on
// external Python objects), we create a configurable gate chain that handles
// COMMON gate patterns (threshold checks, scaling) in pure Rust, and
// delegates complex gates back to Python via callbacks.
//
// Pure-Rust gates:
//   equity_leverage  — Kelly bracket leverage + z-score scaling
//   consensus        — Cross-symbol signal agreement scaling
//   drawdown         — Equity drawdown rejection
//   correlation      — Average correlation rejection
//   alpha_health     — Pass-through alpha health scale
//   regime_sizer     — Pass-through regime scale
//   staged_risk      — Pass-through staged risk scale
//
// Python-delegated gates use the `python` gate type with a PyObject callback.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

// ── Helpers for extracting typed values from PyDict ─────────────────────────

fn get_f64(d: &Bound<'_, PyDict>, key: &str, default: f64) -> f64 {
    match d.get_item(key) {
        Ok(Some(v)) => v.extract::<f64>().unwrap_or(default),
        _ => default,
    }
}

fn get_i32(d: &Bound<'_, PyDict>, key: &str, default: i32) -> i32 {
    match d.get_item(key) {
        Ok(Some(v)) => v.extract::<i32>().unwrap_or(default),
        _ => default,
    }
}

fn get_string(d: &Bound<'_, PyDict>, key: &str, default: &str) -> String {
    match d.get_item(key) {
        Ok(Some(v)) => v.extract::<String>().unwrap_or_else(|_| default.to_string()),
        _ => default.to_string(),
    }
}

fn get_bool(d: &Bound<'_, PyDict>, key: &str, default: bool) -> bool {
    match d.get_item(key) {
        Ok(Some(v)) => v.extract::<bool>().unwrap_or(default),
        _ => default,
    }
}

// ── Result from a single gate check (exported to Python) ────────────────────

#[pyclass(name = "RustGateResult", frozen)]
#[derive(Clone, Debug)]
pub struct RustGateResult {
    #[pyo3(get)]
    pub allowed: bool,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub gate_name: String,
}

#[pymethods]
impl RustGateResult {
    #[new]
    #[pyo3(signature = (allowed, scale=1.0, reason=String::new(), gate_name=String::new()))]
    fn new(allowed: bool, scale: f64, reason: String, gate_name: String) -> Self {
        Self { allowed, scale, reason, gate_name }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustGateResult(allowed={}, scale={:.4}, gate={}, reason={})",
            self.allowed, self.scale, self.gate_name, self.reason,
        )
    }
}

// ── Internal gate context ───────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]  // Fields extracted from PyDict for future gates
struct GateContext {
    symbol: String,
    side: String,
    signal: i32,
    qty: f64,
    price: f64,
    equity: f64,
    peak_equity: f64,
    drawdown_pct: f64,
    z_score: f64,
    avg_correlation: f64,
    alpha_health_scale: f64,
    staged_risk_scale: f64,
    regime_scale: f64,
    consensus: HashMap<String, i32>,
}

impl GateContext {
    fn from_pydict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let symbol = get_string(d, "symbol", "");
        let side = get_string(d, "side", "");
        let signal = get_i32(d, "signal", 0);
        let qty = get_f64(d, "qty", 0.0);
        let price = get_f64(d, "price", 0.0);
        let equity = get_f64(d, "equity", 0.0);
        let peak_equity = get_f64(d, "peak_equity", 0.0);
        let drawdown_pct = get_f64(d, "drawdown_pct", 0.0);
        let z_score = get_f64(d, "z_score", 0.0);
        let avg_correlation = get_f64(d, "avg_correlation", 0.0);
        let alpha_health_scale = get_f64(d, "alpha_health_scale", 1.0);
        let staged_risk_scale = get_f64(d, "staged_risk_scale", 1.0);
        let regime_scale = get_f64(d, "regime_scale", 1.0);

        // Extract consensus dict: {symbol_str: signal_int}
        let mut consensus = HashMap::new();
        if let Ok(Some(cons_val)) = d.get_item("consensus") {
            if let Ok(cons_dict) = cons_val.downcast::<PyDict>() {
                for (k, v) in cons_dict.iter() {
                    if let (Ok(key), Ok(val)) = (k.extract::<String>(), v.extract::<i32>()) {
                        consensus.insert(key, val);
                    }
                }
            }
        }

        Ok(Self {
            symbol,
            side,
            signal,
            qty,
            price,
            equity,
            peak_equity,
            drawdown_pct,
            z_score,
            avg_correlation,
            alpha_health_scale,
            staged_risk_scale,
            regime_scale,
            consensus,
        })
    }
}

// ── Internal gate result ────────────────────────────────────────────────────

struct GateResult {
    allowed: bool,
    scale: f64,
    reason: String,
}

impl GateResult {
    fn allow(scale: f64) -> Self {
        Self { allowed: true, scale, reason: String::new() }
    }

    fn allow_with_reason(scale: f64, reason: String) -> Self {
        Self { allowed: true, scale, reason }
    }

    fn reject(reason: String) -> Self {
        Self { allowed: false, scale: 0.0, reason }
    }
}

// ── Pure-Rust gate trait ────────────────────────────────────────────────────

trait Gate: Send + Sync {
    fn name(&self) -> &str;
    fn check(&self, ctx: &GateContext) -> GateResult;
}

// ── Equity-leverage gate ────────────────────────────────────────────────────
// Python: runner/gates/equity_leverage_gate.py
// Kelly brackets + z-score scaling. Always allows; sizing only.

struct EquityLeverageGate {
    brackets: Vec<(f64, f64, f64)>, // (min_equity, max_equity, leverage)
}

impl EquityLeverageGate {
    fn new(brackets: Vec<(f64, f64, f64)>) -> Self {
        Self { brackets }
    }

    fn default_brackets() -> Vec<(f64, f64, f64)> {
        vec![
            (0.0, 5_000.0, 1.5),
            (5_000.0, 20_000.0, 1.5),
            (20_000.0, 50_000.0, 1.0),
            (50_000.0, f64::INFINITY, 1.0),
        ]
    }
}

fn bracket_leverage(equity: f64, brackets: &[(f64, f64, f64)]) -> f64 {
    for &(min_eq, max_eq, lev) in brackets {
        if equity >= min_eq && equity < max_eq {
            return lev;
        }
    }
    1.0
}

fn z_scale_factor(z: f64) -> f64 {
    let az = z.abs();
    if az > 2.0 {
        1.5
    } else if az > 1.0 {
        1.0
    } else if az > 0.5 {
        0.7
    } else {
        0.5
    }
}

impl Gate for EquityLeverageGate {
    fn name(&self) -> &str {
        "equity_leverage"
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        let lev = bracket_leverage(ctx.equity, &self.brackets);
        let z_scale = z_scale_factor(ctx.z_score);
        let scale = lev * z_scale;
        GateResult::allow_with_reason(
            scale,
            format!("lev={:.2} z_scale={:.2}", lev, z_scale),
        )
    }
}

// ── Consensus scaling gate ──────────────────────────────────────────────────
// Python: runner/gates/consensus_scaling_gate.py
// Contrarian boost when all others disagree; reduction on weak consensus.

struct ConsensusScalingGate;

impl Gate for ConsensusScalingGate {
    fn name(&self) -> &str {
        "consensus_scaling"
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        if ctx.signal == 0 {
            return GateResult::allow_with_reason(1.0, "flat_signal".into());
        }

        // Filter to other symbols with active (non-zero) signals
        let others: Vec<(&String, &i32)> = ctx
            .consensus
            .iter()
            .filter(|(s, sig)| s.as_str() != ctx.symbol && **sig != 0)
            .collect();

        if others.is_empty() {
            return GateResult::allow_with_reason(1.0, "no_other_signals".into());
        }

        let total = others.len();
        let agree_count = others.iter().filter(|(_, sig)| **sig == ctx.signal).count();

        let scale = if agree_count == 0 {
            // Every active other disagrees -> contrarian boost
            1.3
        } else {
            let ratio = agree_count as f64 / total as f64;
            if ratio >= 0.75 {
                1.0 // strong consensus
            } else if ratio >= 0.25 {
                0.7 // mixed
            } else {
                0.5 // near-contrarian
            }
        };

        let agree_pct = (agree_count as f64 / total as f64) * 100.0;
        GateResult::allow_with_reason(scale, format!("agree={:.0}%", agree_pct))
    }
}

// ── Drawdown gate ───────────────────────────────────────────────────────────
// Rejects orders when drawdown exceeds threshold.

struct DrawdownGate {
    max_drawdown_pct: f64, // e.g. 0.20 for 20%
}

impl Gate for DrawdownGate {
    fn name(&self) -> &str {
        "drawdown"
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        if ctx.drawdown_pct >= self.max_drawdown_pct {
            GateResult::reject(format!(
                "drawdown {:.2}% >= limit {:.2}%",
                ctx.drawdown_pct * 100.0,
                self.max_drawdown_pct * 100.0,
            ))
        } else {
            GateResult::allow(1.0)
        }
    }
}

// ── Correlation gate ────────────────────────────────────────────────────────
// Rejects when avg portfolio correlation is too high.

struct CorrelationGate {
    max_avg_correlation: f64,
}

impl Gate for CorrelationGate {
    fn name(&self) -> &str {
        "correlation"
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        if ctx.avg_correlation > self.max_avg_correlation {
            GateResult::reject(format!(
                "correlation {:.3} > limit {:.3}",
                ctx.avg_correlation, self.max_avg_correlation,
            ))
        } else {
            GateResult::allow(1.0)
        }
    }
}

// ── Alpha health gate ───────────────────────────────────────────────────────
// Pass-through scale from context. Rejects when scale <= 0.

struct AlphaHealthGate;

impl Gate for AlphaHealthGate {
    fn name(&self) -> &str {
        "alpha_health"
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        if ctx.alpha_health_scale <= 0.0 {
            GateResult::reject("alpha_unhealthy".into())
        } else {
            GateResult::allow(ctx.alpha_health_scale)
        }
    }
}

// ── Regime sizer gate ───────────────────────────────────────────────────────
// Pass-through regime scale from context. Never rejects.

struct RegimeSizerGate;

impl Gate for RegimeSizerGate {
    fn name(&self) -> &str {
        "regime_sizer"
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        GateResult::allow(ctx.regime_scale.max(0.0))
    }
}

// ── Staged risk gate ────────────────────────────────────────────────────────
// Pass-through staged risk scale. Rejects when scale <= 0 (halted).

struct StagedRiskGate;

impl Gate for StagedRiskGate {
    fn name(&self) -> &str {
        "staged_risk"
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        if ctx.staged_risk_scale <= 0.0 {
            GateResult::reject("staged_risk_blocked".into())
        } else {
            GateResult::allow(ctx.staged_risk_scale)
        }
    }
}

// ── Notional limit gate ─────────────────────────────────────────────────────
// Rejects when qty * price exceeds max_notional. Matches MAX_ORDER_NOTIONAL.

struct NotionalLimitGate {
    max_notional: f64,
}

impl Gate for NotionalLimitGate {
    fn name(&self) -> &str {
        "notional_limit"
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        let notional = ctx.qty.abs() * ctx.price;
        if notional > self.max_notional && ctx.price > 0.0 {
            // Clamp rather than block (matches AlphaRunner behavior)
            let clamped_qty = self.max_notional / ctx.price;
            let scale = if ctx.qty.abs() > 0.0 {
                clamped_qty / ctx.qty.abs()
            } else {
                1.0
            };
            GateResult::allow_with_reason(
                scale,
                format!("notional {:.2} > {:.2}, clamped", notional, self.max_notional),
            )
        } else {
            GateResult::allow(1.0)
        }
    }
}

// ── Min-qty gate ────────────────────────────────────────────────────────────
// Rejects when scaled qty falls below min_qty (venue minimum).

struct MinQtyGate {
    min_qty: f64,
}

impl Gate for MinQtyGate {
    fn name(&self) -> &str {
        "min_qty"
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        if ctx.qty.abs() < self.min_qty {
            GateResult::reject(format!(
                "qty {:.6} < min {:.6}",
                ctx.qty.abs(),
                self.min_qty,
            ))
        } else {
            GateResult::allow(1.0)
        }
    }
}

// ── Python callback gate ────────────────────────────────────────────────────
// Delegates to a Python callable for complex gates that depend on Python objects.
// The callable should accept (symbol: str, context: dict) and return
// a dict with keys: allowed (bool), scale (float), reason (str).

struct PythonCallbackGate {
    gate_name: String,
    callback: PyObject,
}

impl Gate for PythonCallbackGate {
    fn name(&self) -> &str {
        &self.gate_name
    }

    fn check(&self, ctx: &GateContext) -> GateResult {
        Python::with_gil(|py| {
            let result = self
                .callback
                .call1(py, (&ctx.symbol,))
                .and_then(|obj| {
                    let dict = obj.downcast_bound::<PyDict>(py)?;
                    let allowed = get_bool(dict, "allowed", true);
                    let scale = get_f64(dict, "scale", 1.0);
                    let reason = get_string(dict, "reason", "");
                    Ok(GateResult {
                        allowed,
                        scale,
                        reason,
                    })
                });

            match result {
                Ok(r) => r,
                Err(e) => {
                    // On Python error, reject conservatively
                    GateResult::reject(format!("python_gate_error: {}", e))
                }
            }
        })
    }
}

// SAFETY: PyObject is Send+Sync when accessed only via Python::with_gil
unsafe impl Send for PythonCallbackGate {}
unsafe impl Sync for PythonCallbackGate {}

// ── PyO3 exposed orchestrator ───────────────────────────────────────────────

#[pyclass(name = "RustGateChain")]
pub struct RustGateChain {
    gates: Vec<Box<dyn Gate>>,
    /// When true, scaling is cumulative (multiply). When false, use min scale.
    cumulative_scaling: bool,
}

#[pymethods]
impl RustGateChain {
    /// Create a new RustGateChain.
    ///
    /// Args:
    ///     cumulative_scaling: if True (default), gate scales multiply together.
    ///                        if False, final scale = minimum of all gate scales.
    #[new]
    #[pyo3(signature = (cumulative_scaling=true))]
    fn new(cumulative_scaling: bool) -> Self {
        Self {
            gates: Vec::new(),
            cumulative_scaling,
        }
    }

    /// Add a gate by type name and optional config dict.
    ///
    /// Supported gate_type values:
    ///   "equity_leverage" — config: {brackets: [[min, max, lev], ...]}
    ///   "consensus_scaling" — no config needed
    ///   "drawdown" — config: {max_drawdown_pct: float}
    ///   "correlation" — config: {max_avg_correlation: float}
    ///   "alpha_health" — no config needed
    ///   "regime_sizer" — no config needed
    ///   "staged_risk" — no config needed
    ///   "notional_limit" — config: {max_notional: float}
    ///   "min_qty" — config: {min_qty: float}
    ///   "python" — config: {name: str, callback: callable}
    fn add_gate(&mut self, gate_type: &str, config: &Bound<'_, PyDict>) -> PyResult<()> {
        let gate: Box<dyn Gate> = match gate_type {
            "equity_leverage" => {
                let brackets = self.parse_brackets(config)?;
                Box::new(EquityLeverageGate::new(brackets))
            }
            "consensus_scaling" => Box::new(ConsensusScalingGate),
            "drawdown" => {
                let max_dd = get_f64(config, "max_drawdown_pct", 0.20);
                Box::new(DrawdownGate {
                    max_drawdown_pct: max_dd,
                })
            }
            "correlation" => {
                let max_corr = get_f64(config, "max_avg_correlation", 0.70);
                Box::new(CorrelationGate {
                    max_avg_correlation: max_corr,
                })
            }
            "alpha_health" => Box::new(AlphaHealthGate),
            "regime_sizer" => Box::new(RegimeSizerGate),
            "staged_risk" => Box::new(StagedRiskGate),
            "notional_limit" => {
                let max_notional = get_f64(config, "max_notional", 5_000.0);
                Box::new(NotionalLimitGate { max_notional })
            }
            "min_qty" => {
                let min_qty = get_f64(config, "min_qty", 0.001);
                Box::new(MinQtyGate { min_qty })
            }
            "python" => {
                let name = get_string(config, "name", "python_gate");
                let callback = match config.get_item("callback")? {
                    Some(cb) => cb.unbind(),
                    None => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "python gate requires 'callback' in config",
                        ));
                    }
                };
                Box::new(PythonCallbackGate {
                    gate_name: name,
                    callback,
                })
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown gate type: '{}'. Supported: equity_leverage, consensus_scaling, \
                     drawdown, correlation, alpha_health, regime_sizer, staged_risk, \
                     notional_limit, min_qty, python",
                    gate_type
                )));
            }
        };
        self.gates.push(gate);
        Ok(())
    }

    /// Process context through ALL gates in one Rust call.
    ///
    /// Args:
    ///     context: dict with keys: symbol, side, signal, qty, price, equity,
    ///              peak_equity, drawdown_pct, z_score, avg_correlation,
    ///              alpha_health_scale, staged_risk_scale, regime_scale,
    ///              consensus (dict of {symbol: signal_int})
    ///
    /// Returns:
    ///     dict with keys:
    ///       allowed (bool) — True if all gates passed
    ///       scale (float) — cumulative scale factor (0.0 if rejected)
    ///       audit (list[dict]) — per-gate audit trail
    ///       drop_gate (str, optional) — name of gate that rejected
    fn process(&self, py: Python<'_>, context: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let ctx = GateContext::from_pydict(context)?;
        let mut cumulative_scale = 1.0_f64;
        let audit_list = PyList::empty(py);

        for gate in &self.gates {
            let result = gate.check(&ctx);

            let entry = PyDict::new(py);
            entry.set_item("gate", gate.name())?;
            entry.set_item("allowed", result.allowed)?;
            entry.set_item("scale", result.scale)?;
            entry.set_item("reason", &result.reason)?;
            audit_list.append(&entry)?;

            if !result.allowed {
                let ret = PyDict::new(py);
                ret.set_item("allowed", false)?;
                ret.set_item("scale", 0.0)?;
                ret.set_item("audit", &audit_list)?;
                ret.set_item("drop_gate", gate.name())?;
                return Ok(ret.into_any().unbind());
            }

            if self.cumulative_scaling {
                cumulative_scale *= result.scale;
            } else {
                cumulative_scale = cumulative_scale.min(result.scale);
            }
        }

        let ret = PyDict::new(py);
        ret.set_item("allowed", true)?;
        ret.set_item("scale", cumulative_scale)?;
        ret.set_item("audit", &audit_list)?;
        Ok(ret.into_any().unbind())
    }

    /// Process context and return list of RustGateResult objects.
    ///
    /// Unlike process(), this returns structured RustGateResult objects
    /// instead of dicts. Useful for detailed inspection.
    fn process_detailed(
        &self,
        context: &Bound<'_, PyDict>,
    ) -> PyResult<(bool, f64, Vec<RustGateResult>)> {
        let ctx = GateContext::from_pydict(context)?;
        let mut cumulative_scale = 1.0_f64;
        let mut trail: Vec<RustGateResult> = Vec::with_capacity(self.gates.len());

        for gate in &self.gates {
            let result = gate.check(&ctx);
            let gate_result = RustGateResult {
                allowed: result.allowed,
                scale: result.scale,
                reason: result.reason.clone(),
                gate_name: gate.name().to_string(),
            };
            trail.push(gate_result);

            if !result.allowed {
                return Ok((false, 0.0, trail));
            }

            if self.cumulative_scaling {
                cumulative_scale *= result.scale;
            } else {
                cumulative_scale = cumulative_scale.min(result.scale);
            }
        }

        Ok((true, cumulative_scale, trail))
    }

    /// Number of gates in the chain.
    fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// List of gate names in order.
    fn gate_names(&self) -> Vec<String> {
        self.gates.iter().map(|g| g.name().to_string()).collect()
    }

    /// Remove all gates.
    fn clear(&mut self) {
        self.gates.clear();
    }

    /// Standalone z-score scale factor (for testing / external use).
    #[staticmethod]
    fn z_scale(z: f64) -> f64 {
        z_scale_factor(z)
    }

    /// Standalone bracket leverage lookup (for testing / external use).
    #[staticmethod]
    #[pyo3(signature = (equity, brackets=None))]
    fn bracket_lev(equity: f64, brackets: Option<Vec<(f64, f64, f64)>>) -> f64 {
        match brackets {
            Some(b) => bracket_leverage(equity, &b),
            None => bracket_leverage(equity, &EquityLeverageGate::default_brackets()),
        }
    }

    /// Standalone consensus scale (for testing / external use).
    #[staticmethod]
    fn consensus_scale(symbol: &str, signal: i32, consensus: HashMap<String, i32>) -> f64 {
        // Filter to other symbols with active signals
        let others: Vec<(&String, &i32)> = consensus
            .iter()
            .filter(|(s, sig)| s.as_str() != symbol && **sig != 0)
            .collect();

        if others.is_empty() {
            return 1.0;
        }

        let total = others.len();
        let agree_count = others.iter().filter(|(_, sig)| **sig == signal).count();

        if agree_count == 0 {
            1.3
        } else {
            let ratio = agree_count as f64 / total as f64;
            if ratio >= 0.75 {
                1.0
            } else if ratio >= 0.25 {
                0.7
            } else {
                0.5
            }
        }
    }
}

// ── Private helpers ─────────────────────────────────────────────────────────

impl RustGateChain {
    /// Parse brackets from config dict.
    /// Expects config["brackets"] = [[min, max, lev], ...] or uses defaults.
    fn parse_brackets(&self, config: &Bound<'_, PyDict>) -> PyResult<Vec<(f64, f64, f64)>> {
        match config.get_item("brackets")? {
            Some(brackets_val) => {
                let brackets_list = brackets_val.downcast::<PyList>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "brackets must be a list of [min, max, leverage] triples",
                    )
                })?;
                let mut brackets = Vec::with_capacity(brackets_list.len());
                for item in brackets_list.iter() {
                    let triple = item.downcast::<PyList>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "each bracket must be a list [min, max, leverage]",
                        )
                    })?;
                    if triple.len() != 3 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "each bracket must have exactly 3 elements",
                        ));
                    }
                    let min_eq = triple.get_item(0)?.extract::<f64>()?;
                    let max_eq = triple.get_item(1)?.extract::<f64>()?;
                    let lev = triple.get_item(2)?.extract::<f64>()?;
                    brackets.push((min_eq, max_eq, lev));
                }
                Ok(brackets)
            }
            None => Ok(EquityLeverageGate::default_brackets()),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> GateContext {
        GateContext {
            symbol: "ETHUSDT".to_string(),
            side: "buy".to_string(),
            signal: 1,
            qty: 10.0,
            price: 2000.0,
            equity: 1000.0,
            peak_equity: 1200.0,
            drawdown_pct: 0.0,
            z_score: 1.5,
            avg_correlation: 0.3,
            alpha_health_scale: 1.0,
            staged_risk_scale: 1.0,
            regime_scale: 1.0,
            consensus: HashMap::new(),
        }
    }

    #[test]
    fn test_bracket_leverage_default() {
        let brackets = EquityLeverageGate::default_brackets();
        assert_eq!(bracket_leverage(500.0, &brackets), 1.5);
        assert_eq!(bracket_leverage(10_000.0, &brackets), 1.5);
        assert_eq!(bracket_leverage(30_000.0, &brackets), 1.0);
        assert_eq!(bracket_leverage(100_000.0, &brackets), 1.0);
    }

    #[test]
    fn test_z_scale_factor() {
        assert_eq!(z_scale_factor(2.5), 1.5);
        assert_eq!(z_scale_factor(-2.5), 1.5);
        assert_eq!(z_scale_factor(1.5), 1.0);
        assert_eq!(z_scale_factor(0.7), 0.7);
        assert_eq!(z_scale_factor(0.3), 0.5);
        assert_eq!(z_scale_factor(0.0), 0.5);
    }

    #[test]
    fn test_equity_leverage_gate() {
        let gate = EquityLeverageGate::new(EquityLeverageGate::default_brackets());
        let mut ctx = make_ctx();
        ctx.equity = 1000.0;
        ctx.z_score = 1.5;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        // equity=1000 -> bracket 1.5, z=1.5 -> z_scale=1.0, total=1.5
        assert!((r.scale - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_gate_contrarian_boost() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        ctx.consensus.insert("BTCUSDT".into(), -1);
        ctx.consensus.insert("SOLUSDT".into(), -1);
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 1.3).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_gate_strong_agreement() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        ctx.consensus.insert("BTCUSDT".into(), 1);
        ctx.consensus.insert("SOLUSDT".into(), 1);
        ctx.consensus.insert("SUIUSDT".into(), 1);
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_gate_flat_signal() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 0;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_drawdown_gate_allows() {
        let gate = DrawdownGate {
            max_drawdown_pct: 0.20,
        };
        let mut ctx = make_ctx();
        ctx.drawdown_pct = 0.10;
        let r = gate.check(&ctx);
        assert!(r.allowed);
    }

    #[test]
    fn test_drawdown_gate_rejects() {
        let gate = DrawdownGate {
            max_drawdown_pct: 0.20,
        };
        let mut ctx = make_ctx();
        ctx.drawdown_pct = 0.25;
        let r = gate.check(&ctx);
        assert!(!r.allowed);
        assert!(r.reason.contains("drawdown"));
    }

    #[test]
    fn test_correlation_gate() {
        let gate = CorrelationGate {
            max_avg_correlation: 0.70,
        };
        let mut ctx = make_ctx();

        ctx.avg_correlation = 0.5;
        assert!(gate.check(&ctx).allowed);

        ctx.avg_correlation = 0.8;
        assert!(!gate.check(&ctx).allowed);
    }

    #[test]
    fn test_alpha_health_gate() {
        let gate = AlphaHealthGate;
        let mut ctx = make_ctx();

        ctx.alpha_health_scale = 0.5;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.5).abs() < 1e-6);

        ctx.alpha_health_scale = 0.0;
        assert!(!gate.check(&ctx).allowed);

        ctx.alpha_health_scale = -0.1;
        assert!(!gate.check(&ctx).allowed);
    }

    #[test]
    fn test_staged_risk_gate() {
        let gate = StagedRiskGate;
        let mut ctx = make_ctx();

        ctx.staged_risk_scale = 0.8;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.8).abs() < 1e-6);

        ctx.staged_risk_scale = 0.0;
        assert!(!gate.check(&ctx).allowed);
    }

    #[test]
    fn test_regime_sizer_gate() {
        let gate = RegimeSizerGate;
        let mut ctx = make_ctx();

        ctx.regime_scale = 0.6;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.6).abs() < 1e-6);

        // Negative clamped to 0
        ctx.regime_scale = -0.5;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_notional_limit_gate() {
        let gate = NotionalLimitGate {
            max_notional: 5_000.0,
        };
        let mut ctx = make_ctx();

        // Within limit: qty=1.0, price=2000 -> notional=2000
        ctx.qty = 1.0;
        ctx.price = 2000.0;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 1.0).abs() < 1e-6);

        // Over limit: qty=5.0, price=2000 -> notional=10000
        ctx.qty = 5.0;
        ctx.price = 2000.0;
        let r = gate.check(&ctx);
        assert!(r.allowed);
        // Should clamp: 5000/2000 = 2.5, scale = 2.5/5.0 = 0.5
        assert!((r.scale - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_min_qty_gate() {
        let gate = MinQtyGate { min_qty: 0.01 };
        let mut ctx = make_ctx();

        ctx.qty = 0.1;
        assert!(gate.check(&ctx).allowed);

        ctx.qty = 0.005;
        assert!(!gate.check(&ctx).allowed);
    }

    #[test]
    fn test_chain_short_circuits_on_rejection() {
        // Build a chain: drawdown (will reject) -> alpha_health (should not run)
        let gates: Vec<Box<dyn Gate>> = vec![
            Box::new(DrawdownGate {
                max_drawdown_pct: 0.10,
            }),
            Box::new(AlphaHealthGate),
        ];

        let mut ctx = make_ctx();
        ctx.drawdown_pct = 0.15; // above 10% limit

        // Simulate chain logic
        let mut rejected = false;
        let mut reject_gate = String::new();
        for gate in &gates {
            let r = gate.check(&ctx);
            if !r.allowed {
                rejected = true;
                reject_gate = gate.name().to_string();
                break;
            }
        }
        assert!(rejected);
        assert_eq!(reject_gate, "drawdown");
    }

    #[test]
    fn test_cumulative_scaling() {
        let gates: Vec<Box<dyn Gate>> = vec![
            Box::new(AlphaHealthGate),
            Box::new(RegimeSizerGate),
            Box::new(StagedRiskGate),
        ];

        let mut ctx = make_ctx();
        ctx.alpha_health_scale = 0.5;
        ctx.regime_scale = 0.8;
        ctx.staged_risk_scale = 0.9;

        let mut cumulative = 1.0_f64;
        for gate in &gates {
            let r = gate.check(&ctx);
            assert!(r.allowed);
            cumulative *= r.scale;
        }
        // 0.5 * 0.8 * 0.9 = 0.36
        assert!((cumulative - 0.36).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_mixed_signals() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        // 2 agree, 2 disagree -> 50% agree -> mixed -> 0.7
        ctx.consensus.insert("BTCUSDT".into(), 1);
        ctx.consensus.insert("SOLUSDT".into(), 1);
        ctx.consensus.insert("SUIUSDT".into(), -1);
        ctx.consensus.insert("AXSUSDT".into(), -1);
        let r = gate.check(&ctx);
        assert!(r.allowed);
        assert!((r.scale - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_self_excluded() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        ctx.symbol = "ETHUSDT".to_string();
        // Only self in consensus -> no other signals -> 1.0
        ctx.consensus.insert("ETHUSDT".into(), 1);
        let r = gate.check(&ctx);
        assert!((r.scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_consensus_ignores_flat_others() {
        let gate = ConsensusScalingGate;
        let mut ctx = make_ctx();
        ctx.signal = 1;
        // Others are flat (signal=0) -> no active others -> 1.0
        ctx.consensus.insert("BTCUSDT".into(), 0);
        ctx.consensus.insert("SOLUSDT".into(), 0);
        let r = gate.check(&ctx);
        assert!((r.scale - 1.0).abs() < 1e-6);
    }
}
