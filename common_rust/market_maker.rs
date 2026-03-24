// market_maker.rs — High-performance Rust market maker engine
//
// Core quote cycle runs in pure Rust: ~1μs per tick (vs Python ~100μs).
// Components: A-S quoter, inventory tracker, VPIN gate, alpha direction.
// Exposed to Python via PyO3 for integration with Bybit/Binance adapters.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::VecDeque;

// ── Configuration ───────────────────────────────────────────────────

#[pyclass(name = "RustMMConfig")]
#[derive(Clone)]
pub struct RustMMConfig {
    #[pyo3(get, set)]
    pub tick_size: f64,
    #[pyo3(get, set)]
    pub qty_step: f64,
    #[pyo3(get, set)]
    pub order_size: f64,
    #[pyo3(get, set)]
    pub max_inventory_notional: f64,
    #[pyo3(get, set)]
    pub gamma: f64,
    #[pyo3(get, set)]
    pub kappa: f64,
    #[pyo3(get, set)]
    pub time_horizon_s: f64,
    #[pyo3(get, set)]
    pub min_spread_bps: f64,
    #[pyo3(get, set)]
    pub max_spread_bps: f64,
    #[pyo3(get, set)]
    pub vpin_pause_threshold: f64,
    #[pyo3(get, set)]
    pub vpin_widen_threshold: f64,
    #[pyo3(get, set)]
    pub vpin_spread_mult: f64,
    #[pyo3(get, set)]
    pub trend_pause_pct: f64,
    #[pyo3(get, set)]
    pub inv_timeout_s: f64,
    #[pyo3(get, set)]
    pub daily_loss_limit: f64,
    #[pyo3(get, set)]
    pub funding_bias_mult: f64,
    #[pyo3(get, set)]
    pub max_chase_ticks: i32,   // max ticks to chase hedge (0=no chase)
}

#[pymethods]
impl RustMMConfig {
    #[new]
    #[pyo3(signature = (
        tick_size=0.01, qty_step=0.01, order_size=2.0,
        max_inventory_notional=6500.0, gamma=0.3, kappa=1.5,
        time_horizon_s=300.0, min_spread_bps=0.5, max_spread_bps=15.0,
        vpin_pause_threshold=0.8, vpin_widen_threshold=0.6,
        vpin_spread_mult=3.0, trend_pause_pct=0.005,
        inv_timeout_s=120.0, daily_loss_limit=1000.0,
        funding_bias_mult=1.0, max_chase_ticks=3,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        tick_size: f64, qty_step: f64, order_size: f64,
        max_inventory_notional: f64, gamma: f64, kappa: f64,
        time_horizon_s: f64, min_spread_bps: f64, max_spread_bps: f64,
        vpin_pause_threshold: f64, vpin_widen_threshold: f64,
        vpin_spread_mult: f64, trend_pause_pct: f64,
        inv_timeout_s: f64, daily_loss_limit: f64,
        funding_bias_mult: f64, max_chase_ticks: i32,
    ) -> Self {
        Self {
            tick_size, qty_step, order_size, max_inventory_notional,
            gamma, kappa, time_horizon_s, min_spread_bps, max_spread_bps,
            vpin_pause_threshold, vpin_widen_threshold, vpin_spread_mult,
            trend_pause_pct, inv_timeout_s, daily_loss_limit,
            funding_bias_mult, max_chase_ticks,
        }
    }
}

// ── Quote Result ────────────────────────────────────────────────────

#[pyclass(name = "RustMMQuote")]
pub struct RustMMQuote {
    #[pyo3(get)]
    pub bid_price: f64,
    #[pyo3(get)]
    pub bid_size: f64,
    #[pyo3(get)]
    pub ask_price: f64,
    #[pyo3(get)]
    pub ask_size: f64,
    #[pyo3(get)]
    pub spread_bps: f64,
    #[pyo3(get)]
    pub alpha_direction: i32,
    #[pyo3(get)]
    pub mode: String,
    #[pyo3(get)]
    pub paused: bool,
    #[pyo3(get)]
    pub pause_reason: String,
}

// ── Main Engine ─────────────────────────────────────────────────────

#[pyclass(name = "RustMarketMaker")]
pub struct RustMarketMaker {
    cfg: RustMMConfig,
    // Market state
    best_bid: f64,
    best_ask: f64,
    mid: f64,
    vpin: f64,
    ob_imbalance: f64,
    funding_rate: f64,
    // Inventory
    net_qty: f64,
    avg_entry: f64,
    daily_pnl: f64,
    total_fills: u64,
    consecutive_losses: u32,
    // Vol estimator (EMA)
    vol_ema: f64,
    vol_ready: bool,
    last_price: f64,
    trade_count: u32,
    // Price history for momentum
    price_history: VecDeque<f64>,
    // Alpha direction
    alpha_direction: i32,
    momentum_1m: i32,
    momentum_5m: i32,
    funding_signal: i32,
    ml_signal: i32,
    // State flags
    killed: bool,
    kill_reason: String,
    tick_count: u64,
    // Ping-pong state
    pending_hedge_side_is_buy: Option<bool>,
    pending_hedge_qty: f64,
    pending_hedge_price: f64,
    pending_hedge_origin_price: f64,  // original fill price (for chase limit)
    pending_hedge_ts: f64,
    hedge_timeout_s: f64,
    round_trips: u64,
    hedge_timeouts: u64,
    // Risk tracking (R1-R4 fixes)
    max_inventory_seen: f64,       // peak |inv| ever
    inv_breach_count: u32,         // times |inv| > 70% of limit
    consecutive_neg_rts: u32,      // R2: consecutive negative RT rpnl
    max_consecutive_neg: u32,      // worst streak
    hourly_pnl_window: VecDeque<f64>,  // rolling 1h PnL for drawdown
    session_peak_pnl: f64,         // peak PnL this session
    session_max_dd: f64,           // max drawdown this session
}

#[pymethods]
impl RustMarketMaker {
    #[new]
    fn new(cfg: RustMMConfig) -> Self {
        Self {
            cfg,
            best_bid: 0.0, best_ask: 0.0, mid: 0.0,
            vpin: 0.0, ob_imbalance: 0.0, funding_rate: 0.0,
            net_qty: 0.0, avg_entry: 0.0, daily_pnl: 0.0,
            total_fills: 0, consecutive_losses: 0,
            vol_ema: 0.0, vol_ready: false, last_price: 0.0, trade_count: 0,
            price_history: VecDeque::with_capacity(600),
            alpha_direction: 0, momentum_1m: 0, momentum_5m: 0,
            funding_signal: 0, ml_signal: 0,
            killed: false, kill_reason: String::new(),
            tick_count: 0,
            pending_hedge_side_is_buy: None,
            pending_hedge_qty: 0.0,
            pending_hedge_price: 0.0,
            pending_hedge_origin_price: 0.0,
            pending_hedge_ts: 0.0,
            hedge_timeout_s: 300.0, // 5min timeout (was 50min — too long, blocks quoting)
            round_trips: 0,
            hedge_timeouts: 0,
            // Risk tracking
            max_inventory_seen: 0.0,
            inv_breach_count: 0,
            consecutive_neg_rts: 0,
            max_consecutive_neg: 0,
            hourly_pnl_window: VecDeque::with_capacity(3600),
            session_peak_pnl: 0.0,
            session_max_dd: 0.0,
        }
    }

    /// Feed a trade tick for vol estimation + momentum
    fn on_trade(&mut self, price: f64, qty: f64, is_buy: bool) {
        if price <= 0.0 { return; }
        // EMA vol from log returns
        if self.last_price > 0.0 {
            let log_ret = (price / self.last_price).ln();
            let sq = log_ret * log_ret;
            let alpha = 0.01;
            self.vol_ema = (1.0 - alpha) * self.vol_ema + alpha * sq;
            self.trade_count += 1;
            if self.trade_count >= 5 {  // was 20 — ORDI/TIA have fewer trades
                self.vol_ready = true;
            }
        }
        self.last_price = price;
    }

    /// Feed depth update
    fn on_depth(&mut self, best_bid: f64, best_ask: f64, ob_imbalance: f64, vpin: f64) {
        if best_bid <= 0.0 || best_ask <= 0.0 { return; }
        self.best_bid = best_bid;
        self.best_ask = best_ask;
        self.mid = (best_bid + best_ask) / 2.0;
        self.ob_imbalance = ob_imbalance;
        self.vpin = vpin;
        self.tick_count += 1;

        // Price history for momentum
        self.price_history.push_back(self.mid);
        if self.price_history.len() > 600 {
            self.price_history.pop_front();
        }

        // Momentum signals
        self.momentum_1m = 0;
        if self.price_history.len() >= 60 {
            let p60 = self.price_history[self.price_history.len() - 60];
            let move_1m = (self.mid - p60) / p60;
            if move_1m > 0.001 { self.momentum_1m = 1; }
            else if move_1m < -0.001 { self.momentum_1m = -1; }
        }
        self.momentum_5m = 0;
        if self.price_history.len() >= 300 {
            let p300 = self.price_history[self.price_history.len() - 300];
            let move_5m = (self.mid - p300) / p300;
            if move_5m > 0.002 { self.momentum_5m = 1; }
            else if move_5m < -0.002 { self.momentum_5m = -1; }
        }

        // Funding signal
        self.funding_signal = if self.funding_rate > 5e-5 { -1 }
                              else if self.funding_rate < -5e-5 { 1 }
                              else { 0 };

        // Combined alpha (ML counts 2x)
        let votes = self.momentum_1m + self.momentum_5m
                  + self.funding_signal + self.ml_signal * 2
                  + if self.ob_imbalance > 0.3 { 1 }
                    else if self.ob_imbalance < -0.3 { -1 }
                    else { 0 };
        self.alpha_direction = if votes >= 1 { 1 } else if votes <= -1 { -1 } else { 0 };
    }

    /// Set external signals
    fn set_funding_rate(&mut self, rate: f64) { self.funding_rate = rate; }
    fn set_ml_signal(&mut self, signal: i32) { self.ml_signal = signal; }

    /// Record a fill
    fn on_fill(&mut self, is_buy: bool, qty: f64, price: f64) -> f64 {
        self.total_fills += 1;
        let signed_qty = if is_buy { qty } else { -qty };
        let mut rpnl = 0.0;

        // Compute realised PnL if reducing
        if self.net_qty != 0.0 {
            let reducing = (self.net_qty > 0.0 && signed_qty < 0.0)
                        || (self.net_qty < 0.0 && signed_qty > 0.0);
            if reducing {
                let close_qty = qty.min(self.net_qty.abs());
                rpnl = if self.net_qty > 0.0 {
                    close_qty * (price - self.avg_entry)
                } else {
                    close_qty * (self.avg_entry - price)
                };
                self.daily_pnl += rpnl;
                if rpnl < 0.0 { self.consecutive_losses += 1; }
                else { self.consecutive_losses = 0; }
            }
        }

        // Update position
        let old = self.net_qty;
        let new = old + signed_qty;
        if new.abs() < 1e-10 {
            self.avg_entry = 0.0;
        } else if (old >= 0.0 && new > 0.0 && signed_qty > 0.0) {
            self.avg_entry = (self.avg_entry * old + price * qty) / new;
        } else if (old <= 0.0 && new < 0.0 && signed_qty < 0.0) {
            self.avg_entry = (self.avg_entry * old.abs() + price * qty) / new.abs();
        } else if (old > 0.0 && new < 0.0) || (old < 0.0 && new > 0.0) {
            self.avg_entry = price;
        }
        self.net_qty = new;

        // R1: Track max inventory
        let abs_new = new.abs();
        if abs_new > self.max_inventory_seen {
            self.max_inventory_seen = abs_new;
        }
        let max_inv_qty = self.cfg.max_inventory_notional / self.mid.max(1.0);
        if abs_new > max_inv_qty * 0.7 {
            self.inv_breach_count += 1;
        }

        // M7 fix: rebate handled externally (Python knows if maker or taker)

        rpnl
    }

    /// Core: compute quotes for current market state. ~1μs.
    fn compute_quote(&self, py: Python<'_>, time_frac: f64) -> PyResult<RustMMQuote> {
        let cfg = &self.cfg;
        let mid = self.mid;
        let inv = self.net_qty;
        let abs_inv = inv.abs();
        let inv_notional = abs_inv * mid;

        // ── Kill check ──────────────────────────────────────
        if self.killed || self.daily_pnl <= -cfg.daily_loss_limit {
            return Ok(RustMMQuote {
                bid_price: 0.0, bid_size: 0.0, ask_price: 0.0, ask_size: 0.0,
                spread_bps: 0.0, alpha_direction: self.alpha_direction,
                mode: "killed".into(), paused: true,
                pause_reason: "daily_loss_limit".into(),
            });
        }

        // ── R2: Consecutive loss pause (5+ neg RTs) ──
        // H12 fix: pause is time-limited — reset_consecutive_neg() called from Python
        if self.consecutive_neg_rts >= 5 {
            return Ok(RustMMQuote {
                bid_price: 0.0, bid_size: 0.0, ask_price: 0.0, ask_size: 0.0,
                spread_bps: 0.0, alpha_direction: self.alpha_direction,
                mode: "paused".into(), paused: true,
                pause_reason: format!("consec_loss={}", self.consecutive_neg_rts),
            });
        }

        // ── R1: Hard inventory cap (|inv| > limit → pause) ──
        if abs_inv * mid > cfg.max_inventory_notional * 1.2 {
            return Ok(RustMMQuote {
                bid_price: 0.0, bid_size: 0.0, ask_price: 0.0, ask_size: 0.0,
                spread_bps: 0.0, alpha_direction: self.alpha_direction,
                mode: "paused".into(), paused: true,
                pause_reason: format!("inv_breach={:.1}", abs_inv),
            });
        }

        // ── VPIN pause ──────────────────────────────────────
        if self.vpin > cfg.vpin_pause_threshold {
            return Ok(RustMMQuote {
                bid_price: 0.0, bid_size: 0.0, ask_price: 0.0, ask_size: 0.0,
                spread_bps: 0.0, alpha_direction: self.alpha_direction,
                mode: "paused".into(), paused: true,
                pause_reason: format!("vpin={:.3}", self.vpin),
            });
        }

        // ── Trend pause (extreme) ───────────────────────────
        if self.price_history.len() >= 60 {
            let p60 = self.price_history[self.price_history.len() - 60];
            let mv = ((mid - p60) / p60).abs();
            if mv > cfg.trend_pause_pct {
                return Ok(RustMMQuote {
                    bid_price: 0.0, bid_size: 0.0, ask_price: 0.0, ask_size: 0.0,
                    spread_bps: 0.0, alpha_direction: self.alpha_direction,
                    mode: "paused".into(), paused: true,
                    pause_reason: format!("trend={:.2}%", mv * 100.0),
                });
            }
        }

        if mid <= 0.0 {
            return Ok(RustMMQuote {
                bid_price: 0.0, bid_size: 0.0, ask_price: 0.0, ask_size: 0.0,
                spread_bps: 0.0, alpha_direction: 0,
                mode: "warmup".into(), paused: true,
                pause_reason: "no_mid".into(),
            });
        }
        // No vol_ready gate — altcoins have sparse trades

        let tick = cfg.tick_size;

        // ── v8: BBO Join mode — quote at best bid/ask directly ──
        // This maximizes fill rate (join the queue at BBO)
        let bid = self.best_bid;  // join best bid
        let ask = self.best_ask;  // join best ask

        if bid <= 0.0 || ask <= bid {
            return Ok(RustMMQuote {
                bid_price: 0.0, bid_size: 0.0, ask_price: 0.0, ask_size: 0.0,
                spread_bps: 0.0, alpha_direction: self.alpha_direction,
                mode: "invalid".into(), paused: true,
                pause_reason: "bad_bbo".into(),
            });
        }

        // ── Ping-pong: symmetric sizing (no alpha skew) ────
        let base = cfg.order_size;
        let mut bid_size = base;
        let mut ask_size = base;

        // Only inventory-based skew: reduce adding side when inv high
        if abs_inv > 1.0 {
            let reduce = (1.0 - abs_inv / 5.0).max(0.1);
            if inv > 0.0 { bid_size *= reduce; }  // long → less buying
            else { ask_size *= reduce; }           // short → less selling
        };

        // Round to step
        let step = cfg.qty_step;
        bid_size = ((bid_size / step).floor() * step).max(step);
        ask_size = ((ask_size / step).floor() * step).max(step);

        // Side blocking: inventory limit only (no alpha blocking in ping-pong)
        let can_buy = inv < 0.0 || (inv * mid) < cfg.max_inventory_notional;
        let can_sell = inv > 0.0 || (inv.abs() * mid) < cfg.max_inventory_notional;

        let final_bid = if can_buy { bid } else { 0.0 };
        let final_ask = if can_sell { ask } else { 0.0 };

        if final_bid <= 0.0 { bid_size = 0.0; }
        if final_ask <= 0.0 { ask_size = 0.0; }

        let spread_bps = if mid > 0.0 { (final_ask - final_bid) / mid * 10000.0 } else { 0.0 };

        Ok(RustMMQuote {
            bid_price: final_bid,
            bid_size,
            ask_price: final_ask,
            ask_size,
            spread_bps,
            alpha_direction: self.alpha_direction,
            mode: "active".into(),
            paused: false,
            pause_reason: String::new(),
        })
    }

    // ── Getters for Python ──────────────────────────────────

    #[getter] fn mid(&self) -> f64 { self.mid }
    #[getter] fn vpin(&self) -> f64 { self.vpin }
    #[getter] fn net_qty(&self) -> f64 { self.net_qty }
    #[getter] fn daily_pnl(&self) -> f64 { self.daily_pnl }
    #[getter] fn total_fills(&self) -> u64 { self.total_fills }
    #[getter] fn alpha_direction(&self) -> i32 { self.alpha_direction }
    #[getter] fn momentum_1m(&self) -> i32 { self.momentum_1m }
    #[getter] fn funding_signal(&self) -> i32 { self.funding_signal }
    #[getter] fn vol(&self) -> f64 { self.vol_ema.sqrt() }
    #[getter] fn tick_count(&self) -> u64 { self.tick_count }
    #[getter] fn killed(&self) -> bool { self.killed }
    #[getter] fn round_trips(&self) -> u64 { self.round_trips }
    #[getter] fn hedge_timeouts(&self) -> u64 { self.hedge_timeouts }
    #[getter] fn pending_hedge(&self) -> bool { self.pending_hedge_side_is_buy.is_some() }
    #[getter] fn max_inventory_seen(&self) -> f64 { self.max_inventory_seen }
    #[getter] fn inv_breach_count(&self) -> u32 { self.inv_breach_count }
    #[getter] fn consecutive_neg_rts(&self) -> u32 { self.consecutive_neg_rts }
    #[getter] fn max_consecutive_neg(&self) -> u32 { self.max_consecutive_neg }
    #[getter] fn session_max_dd(&self) -> f64 { self.session_max_dd }
    #[getter] fn session_peak_pnl(&self) -> f64 { self.session_peak_pnl }

}

// ── Ping-pong hedge, timeout, regime, and utility methods ──
include!("market_maker_hedge.inc.rs");
