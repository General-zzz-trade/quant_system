// risk_evaluator.inc.rs — RustRiskEvaluator + RustRiskResult.
// Included by engine.rs via include!() macro.

/// Result of a single rule evaluation.
#[pyclass(name = "RustRiskResult", frozen)]
#[derive(Clone, Debug)]
pub struct RustRiskResult {
    #[pyo3(get)]
    pub rule: String,
    /// "allow", "reduce", "reject", "kill"
    #[pyo3(get)]
    pub action: String,
    /// Risk code: "max_position", "max_leverage", "max_drawdown", etc.
    #[pyo3(get)]
    pub code: String,
    /// Human-readable message
    #[pyo3(get)]
    pub message: String,
    /// For REDUCE: max allowed qty
    #[pyo3(get)]
    pub max_qty: Option<f64>,
}

#[pyclass(name = "RustRiskEvaluator")]
pub struct RustRiskEvaluator {
    max_position_qty: Option<f64>,
    max_leverage: f64,
    max_drawdown_pct: f64,
    max_gross_leverage: f64,
    max_net_leverage: f64,
    max_concentration: f64,
    allow_auto_reduce: bool,
    dd_action: String,  // "kill" or "reject"
}

#[pymethods]
impl RustRiskEvaluator {
    #[new]
    #[pyo3(signature = (
        max_position_qty=None,
        max_leverage=3.0,
        max_drawdown_pct=0.20,
        max_gross_leverage=3.0,
        max_net_leverage=1.0,
        max_concentration=0.4,
        allow_auto_reduce=true,
        dd_action="kill".to_string(),
    ))]
    fn new(
        max_position_qty: Option<f64>,
        max_leverage: f64,
        max_drawdown_pct: f64,
        max_gross_leverage: f64,
        max_net_leverage: f64,
        max_concentration: f64,
        allow_auto_reduce: bool,
        dd_action: String,
    ) -> Self {
        Self {
            max_position_qty,
            max_leverage,
            max_drawdown_pct,
            max_gross_leverage,
            max_net_leverage,
            max_concentration,
            allow_auto_reduce,
            dd_action,
        }
    }

    /// Evaluate all rules for an order. Returns list of violations (empty = ALLOW).
    ///
    /// Arguments (all flat f64, extracted from meta by Python wrapper):
    ///   cur_qty: current position qty (signed)
    ///   delta_qty: order qty delta (signed: +buy, -sell)
    ///   is_reducing: whether the order reduces exposure
    ///   equity: account equity
    ///   gross_notional: current gross notional
    ///   net_notional: current net notional
    ///   peak_equity: peak equity (for drawdown)
    ///   price: order price
    ///   sym_notional: current symbol notional (for concentration)
    ///   multiplier: contract multiplier (default 1.0)
    #[pyo3(signature = (
        cur_qty,
        delta_qty,
        is_reducing,
        equity,
        gross_notional,
        net_notional,
        peak_equity,
        price,
        sym_notional,
        multiplier=1.0,
    ))]
    fn evaluate_order(
        &self,
        cur_qty: f64,
        delta_qty: f64,
        is_reducing: bool,
        equity: f64,
        gross_notional: f64,
        net_notional: f64,
        peak_equity: f64,
        price: f64,
        sym_notional: f64,
        multiplier: f64,
    ) -> Vec<RustRiskResult> {
        let mut violations = Vec::new();

        // 1) Max position
        if let Some(max_qty) = self.max_position_qty {
            if !is_reducing {
                let projected = (cur_qty + delta_qty).abs();
                if projected > max_qty {
                    let headroom = max_qty - cur_qty.abs();
                    if headroom > 0.0 && self.allow_auto_reduce {
                        violations.push(RustRiskResult {
                            rule: "max_position".into(),
                            action: "reduce".into(),
                            code: "max_position".into(),
                            message: format!(
                                "Position {:.4} exceeds limit {:.4}",
                                projected, max_qty,
                            ),
                            max_qty: Some(headroom),
                        });
                    } else {
                        violations.push(RustRiskResult {
                            rule: "max_position".into(),
                            action: "reject".into(),
                            code: "max_position".into(),
                            message: format!(
                                "Position {:.4} exceeds limit {:.4}",
                                projected, max_qty,
                            ),
                            max_qty: None,
                        });
                    }
                }
            }
        }

        // 2) Leverage cap
        if equity > 0.0 && !is_reducing {
            let delta_notional = delta_qty.abs() * price * multiplier;
            let projected_gross = gross_notional + delta_notional;
            let projected_lev = projected_gross / equity;

            if projected_lev > self.max_leverage {
                let max_gross = self.max_leverage * equity;
                let headroom = max_gross - gross_notional;
                let denom = price * multiplier;

                if headroom > 0.0 && denom > 0.0 && self.allow_auto_reduce {
                    violations.push(RustRiskResult {
                        rule: "leverage_cap".into(),
                        action: "reduce".into(),
                        code: "max_leverage".into(),
                        message: format!(
                            "Leverage {:.2}x exceeds cap {:.2}x",
                            projected_lev, self.max_leverage,
                        ),
                        max_qty: Some(headroom / denom),
                    });
                } else {
                    violations.push(RustRiskResult {
                        rule: "leverage_cap".into(),
                        action: "reject".into(),
                        code: "max_leverage".into(),
                        message: format!(
                            "Leverage {:.2}x exceeds cap {:.2}x",
                            projected_lev, self.max_leverage,
                        ),
                        max_qty: None,
                    });
                }
            }
        }

        // 3) Max drawdown
        if peak_equity > 0.0 {
            let dd = if equity >= peak_equity {
                0.0
            } else {
                (peak_equity - equity) / peak_equity
            };

            if dd > self.max_drawdown_pct && !is_reducing {
                violations.push(RustRiskResult {
                    rule: "max_drawdown".into(),
                    action: self.dd_action.clone(),
                    code: "max_drawdown".into(),
                    message: format!(
                        "Drawdown {:.1}% exceeds limit {:.1}%",
                        dd * 100.0, self.max_drawdown_pct * 100.0,
                    ),
                    max_qty: None,
                });
            }
        }

        // 4) Gross exposure
        if equity > 0.0 && !is_reducing {
            let delta_notional = delta_qty.abs() * price * multiplier;
            let projected_gross = gross_notional + delta_notional;
            let projected_lev = projected_gross / equity;

            if projected_lev > self.max_gross_leverage {
                let max_gross = self.max_gross_leverage * equity;
                let headroom = max_gross - gross_notional;
                let denom = price * multiplier;

                if headroom > 0.0 && denom > 0.0 && self.allow_auto_reduce {
                    violations.push(RustRiskResult {
                        rule: "gross_exposure".into(),
                        action: "reduce".into(),
                        code: "max_gross".into(),
                        message: format!(
                            "Gross exposure {:.2}x exceeds limit {:.2}x",
                            projected_lev, self.max_gross_leverage,
                        ),
                        max_qty: Some(headroom / denom),
                    });
                } else {
                    violations.push(RustRiskResult {
                        rule: "gross_exposure".into(),
                        action: "reject".into(),
                        code: "max_gross".into(),
                        message: format!(
                            "Gross exposure {:.2}x exceeds limit {:.2}x",
                            projected_lev, self.max_gross_leverage,
                        ),
                        max_qty: None,
                    });
                }
            }
        }

        // 5) Net exposure
        if equity > 0.0 {
            let delta_notional = delta_qty * price * multiplier;  // signed
            let projected_net = net_notional + delta_notional;
            let projected_lev = projected_net.abs() / equity;

            if projected_lev > self.max_net_leverage {
                violations.push(RustRiskResult {
                    rule: "net_exposure".into(),
                    action: "reject".into(),
                    code: "max_net".into(),
                    message: format!(
                        "Net exposure {:.2}x exceeds limit {:.2}x",
                        projected_lev, self.max_net_leverage,
                    ),
                    max_qty: None,
                });
            }
        }

        // 6) Concentration
        if gross_notional > 0.0 {
            let delta_notional = delta_qty.abs() * price * multiplier;
            let projected_sym = sym_notional.abs() + delta_notional;
            let projected_gross = gross_notional + delta_notional;

            if projected_gross > 0.0 {
                let weight = projected_sym / projected_gross;
                if weight > self.max_concentration {
                    violations.push(RustRiskResult {
                        rule: "concentration".into(),
                        action: "reject".into(),
                        code: "max_position".into(),
                        message: format!(
                            "Concentration {:.1}% exceeds limit {:.1}%",
                            weight * 100.0, self.max_concentration * 100.0,
                        ),
                        max_qty: None,
                    });
                }
            }
        }

        violations
    }

    /// Quick check: is drawdown breached?
    #[pyo3(signature = (equity, peak_equity))]
    fn check_drawdown(&self, equity: f64, peak_equity: f64) -> bool {
        if peak_equity <= 0.0 {
            return false;
        }
        let dd = if equity >= peak_equity { 0.0 } else { (peak_equity - equity) / peak_equity };
        dd > self.max_drawdown_pct
    }

    /// Quick check: is leverage breached?
    #[pyo3(signature = (gross_notional, equity))]
    fn check_leverage(&self, gross_notional: f64, equity: f64) -> bool {
        if equity <= 0.0 {
            return false;
        }
        gross_notional / equity > self.max_leverage
    }
}
