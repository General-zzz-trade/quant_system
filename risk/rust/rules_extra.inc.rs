// rules_extra.inc.rs — Rules 5-7: OrderFrequencyRule, CorrelationLimitRule, VaRLimitRule.
// Included by rules.rs via include!() macro.

// ===========================================================================
// 5. OrderFrequencyRule
// ===========================================================================

pub struct OrderFrequencyRule {
    pub name: String,
    pub max_per_minute: u32,
    pub window_secs: f64,
    pub timestamps: Mutex<VecDeque<f64>>,
}

impl OrderFrequencyRule {
    pub fn new(name: String, max_per_minute: u32, window_secs: f64) -> Self {
        Self {
            name,
            max_per_minute,
            window_secs,
            timestamps: Mutex::new(VecDeque::new()),
        }
    }

    /// Record an order timestamp and prune expired entries
    fn record_and_count(&self, now: f64) -> u32 {
        let mut ts = self.timestamps.lock().unwrap();
        let cutoff = now - self.window_secs;
        while let Some(&front) = ts.front() {
            if front < cutoff {
                ts.pop_front();
            } else {
                break;
            }
        }
        ts.push_back(now);
        ts.len() as u32
    }
}

impl RiskRule for OrderFrequencyRule {
    fn name(&self) -> &str {
        &self.name
    }

    fn evaluate(&self, ctx: &EvalContext) -> RiskVerdict {
        // For order frequency we use the context's recent_order_count and window
        // Also maintain internal timestamps for self-tracking

        // Check the context-provided count first (from meta)
        if ctx.recent_order_count > 0 {
            // Scale to per-minute rate
            let window_minutes = if ctx.recent_order_window_secs > 0.0 {
                ctx.recent_order_window_secs / 60.0
            } else {
                self.window_secs / 60.0
            };
            let rate_per_min = if window_minutes > 0.0 {
                ctx.recent_order_count as f64 / window_minutes
            } else {
                ctx.recent_order_count as f64
            };

            if rate_per_min > self.max_per_minute as f64 {
                return RiskVerdict::Reject {
                    rule: self.name.clone(),
                    reason: format!(
                        "Order rate {:.1}/min exceeds limit {}/min",
                        rate_per_min, self.max_per_minute
                    ),
                };
            }
        }

        // Internal sliding window: use current time approximation
        // In production the caller records timestamps externally; this is a fallback
        // We use recent_order_window_secs as "now" if positive, else skip
        if ctx.recent_order_window_secs > 0.0 {
            let count = self.record_and_count(ctx.recent_order_window_secs);
            let window_minutes = self.window_secs / 60.0;
            if window_minutes > 0.0 {
                let rate = count as f64 / window_minutes;
                if rate > self.max_per_minute as f64 {
                    return RiskVerdict::Reject {
                        rule: self.name.clone(),
                        reason: format!(
                            "Internal order rate {:.1}/min exceeds limit {}/min ({} orders in {:.0}s window)",
                            rate, self.max_per_minute, count, self.window_secs
                        ),
                    };
                }
            }
        }

        RiskVerdict::Allow
    }
}

// ===========================================================================
// 6. CorrelationLimitRule
// ===========================================================================

pub struct CorrelationLimitRule {
    pub name: String,
    pub max_avg_correlation: f64,
}

impl RiskRule for CorrelationLimitRule {
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

        let corr = ctx.avg_correlation;
        if !corr.is_finite() {
            // No correlation data — allow (similar to Python: inactive when meta missing)
            return RiskVerdict::Allow;
        }

        if corr > self.max_avg_correlation {
            return RiskVerdict::Reject {
                rule: self.name.clone(),
                reason: format!(
                    "Avg portfolio correlation {:.3} exceeds limit {:.3}",
                    corr, self.max_avg_correlation
                ),
            };
        }

        RiskVerdict::Allow
    }
}

// ===========================================================================
// 7. VaRLimitRule
// ===========================================================================

pub struct VaRLimitRule {
    pub name: String,
    pub max_var_pct: f64,
}

impl RiskRule for VaRLimitRule {
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

        let var = ctx.portfolio_var_pct;
        if !var.is_finite() {
            // No VaR data — allow (similar to Python: inactive when meta missing)
            return RiskVerdict::Allow;
        }

        if var > self.max_var_pct {
            return RiskVerdict::Reject {
                rule: self.name.clone(),
                reason: format!(
                    "Portfolio VaR {:.2}% exceeds limit {:.2}%",
                    var, self.max_var_pct
                ),
            };
        }

        RiskVerdict::Allow
    }
}
