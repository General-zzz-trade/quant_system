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

// ── Helpers, Gate trait, individual gate implementations ──
include!("gate_chain_gates.inc.rs");

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
include!("gate_chain_tests.inc.rs");
