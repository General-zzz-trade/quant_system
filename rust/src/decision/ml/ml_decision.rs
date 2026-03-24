use pyo3::prelude::*;

/// Order intent returned by RustMLDecision.decide().
/// Python side wraps this into SimpleNamespace with EventHeader.
#[pyclass]
#[derive(Clone)]
pub struct OrderIntent {
    #[pyo3(get)]
    pub side: String,
    #[pyo3(get)]
    pub qty: f64,
    #[pyo3(get)]
    pub reason: String,
}

#[pymethods]
impl OrderIntent {
    fn __repr__(&self) -> String {
        format!(
            "OrderIntent(side={}, qty={}, reason={})",
            self.side, self.qty, self.reason
        )
    }
}

/// Quantize: floor to 3 decimal places (matches Decimal quantize ROUND_DOWN).
fn q3(x: f64) -> f64 {
    (x * 1000.0).floor() / 1000.0
}

/// Full 5-phase ML decision state machine.
/// Mirrors decision/ml_decision.py MLDecisionModule.
#[pyclass]
pub struct RustMLDecision {
    symbol: String,
    risk_pct: f64,
    threshold: f64,
    threshold_short: f64,
    atr_stop: f64,
    trailing_atr: f64,
    min_hold_bars: i32,
    vol_target: f64,
    dd_limit: f64,
    dd_cooldown: i32,

    // Position tracking state
    entry_price: Option<f64>,
    peak_price: Option<f64>,
    bars_held: i32,
    entry_atr: Option<f64>,

    // Drawdown breaker state
    hwm: f64,
    dd_cooldown_remaining: i32,
}

#[pymethods]
impl RustMLDecision {
    #[new]
    #[pyo3(signature = (
        symbol,
        risk_pct = 0.5,
        threshold = 0.001,
        threshold_short = None,
        atr_stop = 0.0,
        trailing_atr = 0.0,
        min_hold_bars = 0,
        vol_target = 0.0,
        dd_limit = 0.0,
        dd_cooldown = 48,
    ))]
    fn new(
        symbol: &str,
        risk_pct: f64,
        threshold: f64,
        threshold_short: Option<f64>,
        atr_stop: f64,
        trailing_atr: f64,
        min_hold_bars: i32,
        vol_target: f64,
        dd_limit: f64,
        dd_cooldown: i32,
    ) -> Self {
        Self {
            symbol: symbol.to_uppercase(),
            risk_pct,
            threshold,
            threshold_short: threshold_short.unwrap_or(threshold),
            atr_stop,
            trailing_atr,
            min_hold_bars,
            vol_target,
            dd_limit: dd_limit.abs(),
            dd_cooldown,
            entry_price: None,
            peak_price: None,
            bars_held: 0,
            entry_atr: None,
            hwm: 0.0,
            dd_cooldown_remaining: 0,
        }
    }

    /// Main decision entry point.
    ///
    /// Args:
    ///   close: current close price
    ///   ml_score: ML model prediction score
    ///   current_qty: current position qty (signed, + = long, - = short)
    ///   balance: account equity balance
    ///   atr_norm: normalized ATR (atr_norm_14), or None
    ///
    /// Returns: list of OrderIntent
    #[pyo3(signature = (close, ml_score, current_qty, balance, atr_norm=None))]
    fn decide(
        &mut self,
        close: f64,
        ml_score: f64,
        current_qty: f64,
        balance: f64,
        atr_norm: Option<f64>,
    ) -> Vec<OrderIntent> {
        if close <= 0.0 {
            return vec![];
        }

        let current_side = if current_qty > 0.0 {
            "long"
        } else if current_qty < 0.0 {
            "short"
        } else {
            "flat"
        };

        // Track HWM for DD breaker (must run every tick)
        if self.dd_limit > 0.0 {
            self.hwm = self.hwm.max(balance);
        }

        // Update bars held counter
        if current_side != "flat" {
            self.bars_held += 1;
        } else {
            self.bars_held = 0;
        }

        // Update peak price for trailing stop
        if let Some(ref mut peak) = self.peak_price {
            if current_side == "long" {
                *peak = peak.max(close);
            } else if current_side == "short" {
                *peak = peak.min(close);
            }
        }

        // ── Phase 1: Stop-loss checks ──
        if let Some(reason) = self.check_stops(current_side, close, atr_norm) {
            let orders = self.flatten(current_qty, &reason);
            self.clear_entry_state();
            return orders;
        }

        // ── Phase 2: Signal with asymmetric thresholds ──
        let desired = if ml_score > self.threshold {
            "long"
        } else if ml_score < -self.threshold_short {
            "short"
        } else {
            "flat"
        };

        // No change needed — both flat
        if desired == "flat" && current_side == "flat" {
            return vec![];
        }

        // ── Phase 2.5: Gradual rebalance (same direction, qty changed) ──
        if desired == current_side && desired != "flat" {
            let target_qty = self.compute_qty(balance, close, atr_norm, ml_score);
            let abs_current = current_qty.abs();
            let delta = target_qty - abs_current;
            if delta.abs() > abs_current * 0.01 {
                if delta > 0.0 {
                    let side = if current_side == "long" { "BUY" } else { "SELL" };
                    return vec![OrderIntent {
                        side: side.to_owned(),
                        qty: q3(delta),
                        reason: "rebalance_up".to_owned(),
                    }];
                } else {
                    let side = if current_side == "long" { "SELL" } else { "BUY" };
                    return vec![OrderIntent {
                        side: side.to_owned(),
                        qty: q3(delta.abs()),
                        reason: "rebalance_down".to_owned(),
                    }];
                }
            }
            return vec![];
        }

        // ── Phase 3: Min hold suppression ──
        if self.min_hold_bars > 0 && current_side != "flat" {
            if self.bars_held < self.min_hold_bars {
                return vec![];
            }
        }

        // ── Phase 3.5: Drawdown circuit breaker ──
        if self.dd_limit > 0.0 {
            let dd = if self.hwm > 0.0 {
                1.0 - balance / self.hwm
            } else {
                0.0
            };
            if dd >= self.dd_limit {
                self.dd_cooldown_remaining = self.dd_cooldown;
            }
            if self.dd_cooldown_remaining > 0 {
                self.dd_cooldown_remaining -= 1;
                if current_side != "flat" {
                    return self.flatten(current_qty, "dd_breaker");
                }
                return vec![];
            }
        }

        // ── Phase 4: Compute target qty ──
        let target_qty = if desired == "long" || desired == "short" {
            self.compute_qty(balance, close, atr_norm, ml_score)
        } else {
            0.0
        };

        if desired != "flat" && target_qty <= 0.0 {
            return vec![];
        }

        // ── Phase 5: Generate orders ──
        let mut orders = Vec::new();

        if desired == "long" {
            if current_qty < 0.0 {
                orders.push(OrderIntent {
                    side: "BUY".to_owned(),
                    qty: q3(current_qty.abs()),
                    reason: "close_short".to_owned(),
                });
            }
            orders.push(OrderIntent {
                side: "BUY".to_owned(),
                qty: q3(target_qty),
                reason: "open_long".to_owned(),
            });
            self.record_entry(close, atr_norm);
        } else if desired == "short" {
            if current_qty > 0.0 {
                orders.push(OrderIntent {
                    side: "SELL".to_owned(),
                    qty: q3(current_qty),
                    reason: "close_long".to_owned(),
                });
            }
            orders.push(OrderIntent {
                side: "SELL".to_owned(),
                qty: q3(target_qty),
                reason: "open_short".to_owned(),
            });
            self.record_entry(close, atr_norm);
        } else if desired == "flat" && current_qty != 0.0 {
            let mut flat_orders = self.flatten(current_qty, "flatten");
            orders.append(&mut flat_orders);
            self.clear_entry_state();
        }

        orders
    }

    /// Reset all internal state.
    fn reset(&mut self) {
        self.entry_price = None;
        self.peak_price = None;
        self.bars_held = 0;
        self.entry_atr = None;
        self.hwm = 0.0;
        self.dd_cooldown_remaining = 0;
    }
}

// ── private methods ──

impl RustMLDecision {
    fn check_stops(&self, current_side: &str, close: f64, atr_norm: Option<f64>) -> Option<String> {
        if current_side == "flat" {
            return None;
        }
        let atr = atr_norm?;
        let entry = self.entry_price?;
        let is_long = current_side == "long";

        // Hard stop-loss
        if self.atr_stop > 0.0 {
            if let Some(entry_atr) = self.entry_atr {
                let stop_dist = entry_atr * entry * self.atr_stop;
                if is_long && close <= entry - stop_dist {
                    return Some("stop_loss".to_owned());
                }
                if !is_long && close >= entry + stop_dist {
                    return Some("stop_loss".to_owned());
                }
            }
        }

        // Trailing stop
        if self.trailing_atr > 0.0 {
            if let Some(peak) = self.peak_price {
                let trail_dist = atr * close * self.trailing_atr;
                if is_long && close <= peak - trail_dist {
                    return Some("trailing_stop".to_owned());
                }
                if !is_long && close >= peak + trail_dist {
                    return Some("trailing_stop".to_owned());
                }
            }
        }

        None
    }

    fn compute_qty(&self, balance: f64, close: f64, atr_norm: Option<f64>, ml_score: f64) -> f64 {
        let weight = ml_score.abs().min(1.0);

        if self.vol_target > 0.0 && self.atr_stop > 0.0 {
            if let Some(atr) = atr_norm {
                if atr > 0.0 {
                    let risk_budget = balance * self.risk_pct;
                    let stop_dist = atr * self.atr_stop;
                    let raw_qty = risk_budget / (stop_dist * close);
                    if raw_qty <= 0.0 {
                        return 0.0;
                    }
                    return q3(raw_qty * weight);
                }
            }
        }

        // Default: fixed % of equity
        let target_notional = balance * self.risk_pct * weight;
        q3(target_notional / close)
    }

    fn record_entry(&mut self, close: f64, atr_norm: Option<f64>) {
        self.entry_price = Some(close);
        self.peak_price = Some(close);
        self.bars_held = 0;
        self.entry_atr = atr_norm;
    }

    fn clear_entry_state(&mut self) {
        self.entry_price = None;
        self.peak_price = None;
        self.bars_held = 0;
        self.entry_atr = None;
    }

    fn flatten(&self, current_qty: f64, reason: &str) -> Vec<OrderIntent> {
        let side = if current_qty > 0.0 { "SELL" } else { "BUY" };
        vec![OrderIntent {
            side: side.to_owned(),
            qty: q3(current_qty.abs()),
            reason: reason.to_owned(),
        }]
    }
}
