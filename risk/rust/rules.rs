// risk_rules.rs — 7 risk rules with unified trait (Rust port of Python risk/rules/*.py)

use std::collections::VecDeque;
use std::sync::Mutex;

// ── EvalContext — all data needed to evaluate any risk rule ──

pub struct EvalContext {
    pub symbol: String,
    pub side: String, // "buy" / "sell"
    pub qty: f64,
    pub price: f64,   // mark_price
    pub notional: f64, // qty * price

    // Position state
    pub current_position_qty: f64,
    pub current_position_notional: f64,

    // Account state
    pub account_equity: f64,
    pub gross_exposure: f64,
    pub net_exposure: f64,
    pub max_symbol_concentration: f64,

    // Risk state
    pub drawdown_pct: f64,
    pub recent_order_count: u32,
    pub recent_order_window_secs: f64,

    // Optional
    pub avg_correlation: f64,
    pub portfolio_var_pct: f64,
    pub is_reduce_only: bool,
}

impl EvalContext {
    /// Compute signed delta: BUY adds, SELL subtracts
    fn signed_delta(&self) -> f64 {
        if self.side == "buy" {
            self.qty
        } else {
            -self.qty
        }
    }

    /// Whether this order reduces absolute position (moves toward zero)
    pub fn is_reducing_exposure(&self) -> bool {
        if self.is_reduce_only {
            return true;
        }
        let cur = self.current_position_qty;
        if cur == 0.0 {
            return false;
        }
        let delta = self.signed_delta();
        (cur > 0.0 && delta < 0.0) || (cur < 0.0 && delta > 0.0)
    }
}

// ---------------------------------------------------------------------------
// RiskVerdict
// ---------------------------------------------------------------------------

pub enum RiskVerdict {
    Allow,
    Reject {
        rule: String,
        reason: String,
    },
    Reduce {
        rule: String,
        factor: f64,
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// RiskRule trait
// ---------------------------------------------------------------------------

pub trait RiskRule: Send + Sync {
    fn name(&self) -> &str;
    fn evaluate(&self, ctx: &EvalContext) -> RiskVerdict;
}

// ---------------------------------------------------------------------------
// Helper: validate basic numeric inputs
// ---------------------------------------------------------------------------

fn basic_input_check(rule_name: &str, ctx: &EvalContext) -> Option<RiskVerdict> {
    // If price or qty is not finite, reject (unless reduce-only)
    if !ctx.price.is_finite() || !ctx.qty.is_finite() {
        if ctx.is_reduce_only {
            return Some(RiskVerdict::Allow);
        }
        return Some(RiskVerdict::Reject {
            rule: rule_name.to_string(),
            reason: format!(
                "Non-finite inputs: price={}, qty={}",
                ctx.price, ctx.qty
            ),
        });
    }
    None
}

// ===========================================================================
// 1. MaxPositionRule
// ===========================================================================

pub struct MaxPositionRule {
    pub name: String,
    pub max_qty: f64,
    pub max_notional: f64,
}

impl RiskRule for MaxPositionRule {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, ctx: &EvalContext) -> RiskVerdict {
        if let Some(v) = basic_input_check(&self.name, ctx) {
            return v;
        }

        // If reducing exposure, always allow
        if ctx.is_reducing_exposure() {
            return RiskVerdict::Allow;
        }

        let delta = ctx.signed_delta();
        let projected_qty = ctx.current_position_qty + delta;

        // Check qty limit
        if self.max_qty > 0.0 && projected_qty.abs() > self.max_qty {
            let headroom = self.max_qty - ctx.current_position_qty.abs();
            if headroom <= 0.0 {
                return RiskVerdict::Reject {
                    rule: self.name.clone(),
                    reason: format!(
                        "Position at limit: current={:.6}, max_qty={:.6}, no headroom",
                        ctx.current_position_qty, self.max_qty
                    ),
                };
            }
            // Reduce: factor = headroom / requested qty
            let factor = headroom / ctx.qty;
            let factor = factor.clamp(0.0, 1.0);
            if factor <= 0.0 {
                return RiskVerdict::Reject {
                    rule: self.name.clone(),
                    reason: format!(
                        "Projected qty {:.6} exceeds max {:.6}",
                        projected_qty.abs(),
                        self.max_qty
                    ),
                };
            }
            return RiskVerdict::Reduce {
                rule: self.name.clone(),
                factor,
                reason: format!(
                    "Projected qty {:.6} exceeds max {:.6}, reduce factor={:.4}",
                    projected_qty.abs(),
                    self.max_qty,
                    factor
                ),
            };
        }

        // Check notional limit
        if self.max_notional > 0.0 {
            let projected_notional = projected_qty.abs() * ctx.price;
            if projected_notional > self.max_notional {
                let current_notional = ctx.current_position_qty.abs() * ctx.price;
                let headroom_notional = self.max_notional - current_notional;
                if headroom_notional <= 0.0 {
                    return RiskVerdict::Reject {
                        rule: self.name.clone(),
                        reason: format!(
                            "Notional at limit: current={:.2}, max={:.2}",
                            current_notional, self.max_notional
                        ),
                    };
                }
                let order_notional = ctx.qty * ctx.price;
                if order_notional <= 0.0 {
                    return RiskVerdict::Reject {
                        rule: self.name.clone(),
                        reason: "Order notional is zero or negative".to_string(),
                    };
                }
                let factor = (headroom_notional / order_notional).clamp(0.0, 1.0);
                if factor <= 0.0 {
                    return RiskVerdict::Reject {
                        rule: self.name.clone(),
                        reason: format!(
                            "Projected notional {:.2} exceeds max {:.2}",
                            projected_notional, self.max_notional
                        ),
                    };
                }
                return RiskVerdict::Reduce {
                    rule: self.name.clone(),
                    factor,
                    reason: format!(
                        "Projected notional {:.2} exceeds max {:.2}, reduce factor={:.4}",
                        projected_notional, self.max_notional, factor
                    ),
                };
            }
        }

        RiskVerdict::Allow
    }
}

// ===========================================================================
// 2. LeverageCapRule
// ===========================================================================

pub struct LeverageCapRule {
    pub name: String,
    pub max_gross_leverage: f64,
    pub max_net_leverage: f64,
}

impl RiskRule for LeverageCapRule {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, ctx: &EvalContext) -> RiskVerdict {
        if let Some(v) = basic_input_check(&self.name, ctx) {
            return v;
        }

        // If reducing exposure, always allow
        if ctx.is_reducing_exposure() {
            return RiskVerdict::Allow;
        }

        let equity = ctx.account_equity;
        if !equity.is_finite() || equity <= 0.0 {
            if ctx.is_reduce_only {
                return RiskVerdict::Allow;
            }
            return RiskVerdict::Reject {
                rule: self.name.clone(),
                reason: format!("Invalid equity: {}", equity),
            };
        }

        // Gross leverage check
        let order_notional = ctx.qty * ctx.price;
        let projected_gross = ctx.gross_exposure + order_notional;
        let projected_gross_lev = projected_gross / equity;

        if self.max_gross_leverage > 0.0 && projected_gross_lev > self.max_gross_leverage {
            // Compute scaling factor
            let max_gross = self.max_gross_leverage * equity;
            let headroom = max_gross - ctx.gross_exposure;
            if headroom <= 0.0 {
                return RiskVerdict::Reject {
                    rule: self.name.clone(),
                    reason: format!(
                        "Gross leverage at limit: current={:.2}x, cap={:.2}x",
                        ctx.gross_exposure / equity,
                        self.max_gross_leverage
                    ),
                };
            }
            if order_notional > 0.0 {
                let factor = (headroom / order_notional).clamp(0.0, 1.0);
                return RiskVerdict::Reduce {
                    rule: self.name.clone(),
                    factor,
                    reason: format!(
                        "Gross leverage {:.2}x exceeds cap {:.2}x, reduce factor={:.4}",
                        projected_gross_lev, self.max_gross_leverage, factor
                    ),
                };
            }
            return RiskVerdict::Reject {
                rule: self.name.clone(),
                reason: format!(
                    "Gross leverage {:.2}x exceeds cap {:.2}x",
                    projected_gross_lev, self.max_gross_leverage
                ),
            };
        }

        // Net leverage check
        if self.max_net_leverage > 0.0 {
            let delta = ctx.signed_delta();
            let delta_notional = delta * ctx.price;
            let projected_net = ctx.net_exposure + delta_notional;
            let projected_net_lev = projected_net.abs() / equity;

            if projected_net_lev > self.max_net_leverage {
                return RiskVerdict::Reject {
                    rule: self.name.clone(),
                    reason: format!(
                        "Net leverage {:.2}x exceeds cap {:.2}x",
                        projected_net_lev, self.max_net_leverage
                    ),
                };
            }
        }

        RiskVerdict::Allow
    }
}

// ===========================================================================
// 3. MaxDrawdownRule
// ===========================================================================

pub struct MaxDrawdownRule {
    pub name: String,
    pub warning_pct: f64,
    pub kill_pct: f64,
}

impl RiskRule for MaxDrawdownRule {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, ctx: &EvalContext) -> RiskVerdict {
        if let Some(v) = basic_input_check(&self.name, ctx) {
            return v;
        }

        let dd = ctx.drawdown_pct;

        // If drawdown data is NaN/not-finite, allow reduce-only; otherwise reject
        if !dd.is_finite() {
            if ctx.is_reduce_only || ctx.is_reducing_exposure() {
                return RiskVerdict::Allow;
            }
            return RiskVerdict::Reject {
                rule: self.name.clone(),
                reason: format!("Non-finite drawdown_pct: {}", dd),
            };
        }

        // Kill threshold: reject (maps to Python KILL → REJECT in Rust)
        if dd >= self.kill_pct {
            // Still allow reduce-only orders past kill threshold
            if ctx.is_reducing_exposure() {
                return RiskVerdict::Allow;
            }
            return RiskVerdict::Reject {
                rule: self.name.clone(),
                reason: format!(
                    "Drawdown {:.2}% >= kill threshold {:.2}%",
                    dd * 100.0,
                    self.kill_pct * 100.0
                ),
            };
        }

        // Warning threshold: reduce with factor 0.5
        if dd >= self.warning_pct {
            if ctx.is_reducing_exposure() {
                return RiskVerdict::Allow;
            }
            return RiskVerdict::Reduce {
                rule: self.name.clone(),
                factor: 0.5,
                reason: format!(
                    "Drawdown {:.2}% >= warning threshold {:.2}%, reduce by 50%",
                    dd * 100.0,
                    self.warning_pct * 100.0
                ),
            };
        }

        RiskVerdict::Allow
    }
}

// ===========================================================================
// 4. PortfolioLimitsRule
// ===========================================================================

pub struct PortfolioLimitsRule {
    pub name: String,
    pub max_concentration: f64,   // max single symbol weight
    pub max_gross_exposure: f64,  // max total gross notional / equity
}

impl RiskRule for PortfolioLimitsRule {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, ctx: &EvalContext) -> RiskVerdict {
        if let Some(v) = basic_input_check(&self.name, ctx) {
            return v;
        }

        if ctx.is_reducing_exposure() {
            return RiskVerdict::Allow;
        }

        let equity = ctx.account_equity;
        if !equity.is_finite() || equity <= 0.0 {
            if ctx.is_reduce_only {
                return RiskVerdict::Allow;
            }
            return RiskVerdict::Reject {
                rule: self.name.clone(),
                reason: format!("Invalid equity: {}", equity),
            };
        }

        // Concentration check: symbol_notional / gross_exposure
        if self.max_concentration > 0.0 && ctx.gross_exposure > 0.0 {
            let concentration = ctx.max_symbol_concentration;
            if concentration.is_finite() && concentration > self.max_concentration {
                return RiskVerdict::Reject {
                    rule: self.name.clone(),
                    reason: format!(
                        "Symbol concentration {:.1}% exceeds limit {:.1}%",
                        concentration * 100.0,
                        self.max_concentration * 100.0
                    ),
                };
            }

            // Also check projected concentration for the order's symbol
            let order_notional = ctx.qty * ctx.price;
            let projected_sym_notional = ctx.current_position_notional.abs() + order_notional;
            let projected_gross = ctx.gross_exposure + order_notional;
            if projected_gross > 0.0 {
                let projected_weight = projected_sym_notional / projected_gross;
                if projected_weight > self.max_concentration {
                    return RiskVerdict::Reject {
                        rule: self.name.clone(),
                        reason: format!(
                            "Projected concentration {:.1}% exceeds limit {:.1}%",
                            projected_weight * 100.0,
                            self.max_concentration * 100.0
                        ),
                    };
                }
            }
        }

        // Gross exposure check: gross_exposure / equity
        if self.max_gross_exposure > 0.0 {
            let order_notional = ctx.qty * ctx.price;
            let projected_gross = ctx.gross_exposure + order_notional;
            let projected_lev = projected_gross / equity;
            if projected_lev > self.max_gross_exposure {
                let max_gross = self.max_gross_exposure * equity;
                let headroom = max_gross - ctx.gross_exposure;
                if headroom <= 0.0 {
                    return RiskVerdict::Reject {
                        rule: self.name.clone(),
                        reason: format!(
                            "Gross exposure at limit: {:.2}x, cap={:.2}x",
                            ctx.gross_exposure / equity,
                            self.max_gross_exposure
                        ),
                    };
                }
                if order_notional > 0.0 {
                    let factor = (headroom / order_notional).clamp(0.0, 1.0);
                    return RiskVerdict::Reduce {
                        rule: self.name.clone(),
                        factor,
                        reason: format!(
                            "Gross exposure {:.2}x exceeds cap {:.2}x, reduce factor={:.4}",
                            projected_lev, self.max_gross_exposure, factor
                        ),
                    };
                }
            }
        }

        RiskVerdict::Allow
    }
}

// ── Rules 5-7 (OrderFrequency, Correlation, VaR) — see rules_extra.inc.rs ──

include!("rules_extra.inc.rs");

// ── Tests — see rules_tests.inc.rs ──

include!("rules_tests.inc.rs");
