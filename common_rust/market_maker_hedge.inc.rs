// market_maker_hedge.inc.rs — Ping-pong hedge, timeout, regime, and utility methods.
// Included by market_maker.rs via include!() at module level.

#[pymethods]
impl RustMarketMaker {
    /// Ping-pong: after a fill, post the counter-order as maker (at BBO).
    /// Called by Python after recording a fill.
    /// Returns (side_is_buy, qty, price) for the hedge limit order.
    fn pingpong_after_fill(&mut self, filled_is_buy: bool, filled_qty: f64, now_s: f64) -> Option<(bool, f64, f64)> {
        let step = self.cfg.qty_step;
        let qty = ((filled_qty / step).floor() * step).max(step);

        if filled_is_buy {
            // Bought → sell at current best ask (join BBO)
            let price = self.best_ask;
            if price <= 0.0 { return None; }
            self.pending_hedge_side_is_buy = Some(false);
            self.pending_hedge_qty = qty;
            self.pending_hedge_price = price;
            self.pending_hedge_origin_price = price; // record origin for chase limit
            self.pending_hedge_ts = now_s;
            Some((false, qty, price))
        } else {
            // Sold → buy at current best bid (join BBO)
            let price = self.best_bid;
            if price <= 0.0 { return None; }
            self.pending_hedge_side_is_buy = Some(true);
            self.pending_hedge_qty = qty;
            self.pending_hedge_price = price;
            self.pending_hedge_origin_price = price;
            self.pending_hedge_ts = now_s;
            Some((true, qty, price))
        }
    }

    /// v9: Update hedge price to track BBO.
    /// Chase limit only applies to ADVERSE direction (losing money).
    /// Always allows moving to FAVORABLE direction (closer to fill).
    fn update_hedge_price(&mut self) -> Option<(bool, f64, f64)> {
        let is_buy = match self.pending_hedge_side_is_buy {
            Some(b) => b,
            None => return None,
        };
        let new_price = if is_buy { self.best_bid } else { self.best_ask };
        if new_price <= 0.0 { return None; }

        let tick = self.cfg.tick_size;
        let max_chase = self.cfg.max_chase_ticks as f64 * tick;
        let origin = self.pending_hedge_origin_price;

        // Determine if moving to favorable or adverse direction
        // For sell hedge: lower price = adverse (selling cheaper), higher = favorable
        // For buy hedge: higher price = adverse (buying more expensive), lower = favorable
        let is_adverse = if is_buy {
            new_price > origin  // buying higher than origin = adverse
        } else {
            new_price < origin  // selling lower than origin = adverse
        };

        if is_adverse {
            // Limit adverse chase to max_chase_ticks
            let adverse_dist = (new_price - origin).abs();
            if adverse_dist > max_chase {
                return None;  // don't chase further into loss
            }
        }
        // Favorable direction: always allow (helps close the position)

        if (new_price - self.pending_hedge_price).abs() >= tick {
            self.pending_hedge_price = new_price;
            Some((is_buy, self.pending_hedge_qty, new_price))
        } else {
            None
        }
    }

    /// Check if pending hedge has timed out.
    /// v8: timeout → don't taker, just clear pending and let next fill cycle handle it.
    fn check_hedge_timeout(&mut self, now_s: f64) -> bool {
        if self.pending_hedge_side_is_buy.is_none() {
            return false;
        }
        if now_s - self.pending_hedge_ts > self.hedge_timeout_s {
            self.pending_hedge_side_is_buy = None;
            self.hedge_timeouts += 1;
            return true;
        }
        false
    }

    /// Called when the hedge order fills. Clears pending state + tracks risk.
    fn hedge_filled(&mut self, rpnl: f64) {
        self.pending_hedge_side_is_buy = None;
        self.round_trips += 1;

        // R2: Track consecutive negative RTs
        if rpnl < -0.01 {
            self.consecutive_neg_rts += 1;
            if self.consecutive_neg_rts > self.max_consecutive_neg {
                self.max_consecutive_neg = self.consecutive_neg_rts;
            }
        } else {
            self.consecutive_neg_rts = 0;
        }

        // Track drawdown
        self.session_peak_pnl = self.session_peak_pnl.max(self.daily_pnl);
        let dd = self.daily_pnl - self.session_peak_pnl;
        if dd < self.session_max_dd {
            self.session_max_dd = dd;
        }
    }

    /// Returns true if we're waiting for a hedge fill (don't place new quotes).
    fn has_pending_hedge(&self) -> bool {
        self.pending_hedge_side_is_buy.is_some()
    }

    /// Legacy compute_hedge for forced inventory reduction.
    fn compute_hedge(&self) -> Option<(bool, f64, f64)> {
        if self.net_qty.abs() < self.cfg.qty_step {
            return None;
        }
        let is_buy = self.net_qty < 0.0;
        let qty = ((self.net_qty.abs() / self.cfg.qty_step).floor() * self.cfg.qty_step)
            .max(self.cfg.qty_step);
        let price = if is_buy { self.best_ask } else { self.best_bid };
        if price <= 0.0 { return None; }
        Some((is_buy, qty, price))
    }

    /// Regime filter: returns true if market is ranging (good for MM).
    /// Uses vol percentile from price history variance.
    fn is_ranging_regime(&self) -> bool {
        if self.price_history.len() < 120 {
            return true; // default to active during warmup
        }
        // Compute recent volatility from price history
        let n = self.price_history.len();
        let recent = &self.price_history;

        // 1-min returns volatility
        let mut sum_sq = 0.0;
        let mut count = 0;
        let step = 10; // sample every 10 ticks
        for i in (step..n).step_by(step) {
            let ret = (recent[i] / recent[i - step] - 1.0).abs();
            sum_sq += ret * ret;
            count += 1;
        }
        if count == 0 { return true; }
        let vol = (sum_sq / count as f64).sqrt();

        // Ranging = vol < 0.05% per sample (~low volatility)
        vol < 0.0005
    }

    fn reset_daily(&mut self) {
        self.daily_pnl = 0.0;
        self.consecutive_losses = 0;
        self.killed = false;
        self.kill_reason.clear();
    }

    /// H12 fix: reset consecutive neg RTs (called from Python after pause timeout)
    fn reset_consecutive_neg(&mut self) {
        self.consecutive_neg_rts = 0;
    }

    fn force_kill(&mut self, reason: &str) {
        self.killed = true;
        self.kill_reason = reason.to_string();
    }
}
