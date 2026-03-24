// micro_alpha_types.inc.rs — TickPosition + DepthSnapshot types.
// Included by micro_alpha.rs via include!() macro.

/// Depth snapshot from @depth5 stream
#[derive(Debug, Clone)]
pub struct DepthSnapshot {
    pub ts_ms: i64,
    pub best_bid: f64,
    pub best_ask: f64,
    /// Bid total quantity at top 5 levels
    pub bid_qty: f64,
    /// Ask total quantity at top 5 levels
    pub ask_qty: f64,
}

// ── Virtual position for tick-level P&L ──────────────────────

/// Virtual position for dry-run P&L tracking at tick level.
#[derive(Debug, Clone)]
pub struct TickPosition {
    pub qty: f64,           // positive = long, negative = short
    pub entry_price: f64,
    pub entry_ts: i64,
    pub realized_pnl: f64,  // cumulative realized P&L (in price units)
    pub trade_count: u32,
    pub win_count: u32,
}

impl TickPosition {
    pub fn new() -> Self {
        Self { qty: 0.0, entry_price: 0.0, entry_ts: 0, realized_pnl: 0.0, trade_count: 0, win_count: 0 }
    }

    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.qty == 0.0 { return 0.0; }
        self.qty * (current_price - self.entry_price)
    }

    /// Open or flip position. Returns realized P&L if closing.
    pub fn enter(&mut self, ts_ms: i64, price: f64, qty: f64, cost_bps: f64) -> f64 {
        let cost = price * qty.abs() * cost_bps;
        if self.qty == 0.0 {
            // Open new
            self.qty = qty;
            self.entry_price = price;
            self.entry_ts = ts_ms;
            -cost
        } else if (self.qty > 0.0 && qty > 0.0) || (self.qty < 0.0 && qty < 0.0) {
            // Add to position — average entry
            let total_qty = self.qty + qty;
            self.entry_price = (self.entry_price * self.qty.abs() + price * qty.abs()) / total_qty.abs();
            self.qty = total_qty;
            -cost
        } else {
            // Close/flip
            let close_qty = qty.abs().min(self.qty.abs());
            let pnl = if self.qty > 0.0 {
                close_qty * (price - self.entry_price)
            } else {
                close_qty * (self.entry_price - price)
            };
            let net_pnl = pnl - cost;
            self.realized_pnl += net_pnl;
            self.trade_count += 1;
            if net_pnl > 0.0 { self.win_count += 1; }

            let remaining = self.qty.abs() - close_qty;
            if remaining < 1e-12 && qty.abs() > close_qty {
                // Flip
                let flip_qty = qty.abs() - close_qty;
                self.qty = if qty > 0.0 { flip_qty } else { -flip_qty };
                self.entry_price = price;
                self.entry_ts = ts_ms;
            } else if remaining < 1e-12 {
                // Flat
                self.qty = 0.0;
                self.entry_price = 0.0;
            } else {
                // Partially closed, keep direction
                self.qty = if self.qty > 0.0 { remaining } else { -remaining };
            }
            net_pnl
        }
    }

    pub fn win_rate(&self) -> f64 {
        if self.trade_count == 0 { 0.0 } else { self.win_count as f64 / self.trade_count as f64 }
    }
}
