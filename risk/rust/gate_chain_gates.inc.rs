// gate_chain_gates.inc.rs — Helpers, Gate trait, and individual gate implementations.
// Included by gate_chain.rs via include!() macro.

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

