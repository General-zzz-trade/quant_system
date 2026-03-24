// micro_alpha.rs — Enhanced tick-level microstructure alpha engine
//
// V2: Multi-timeframe signals, large trade clustering, depth integration,
// tick-level entry/exit with virtual P&L tracking.
//
// Signals computed from aggTrade stream (no kline/ML dependency):
//   1. Trade flow imbalance (3s/10s/30s multi-timeframe)
//   2. Volume spike (relative to rolling baseline)
//   3. Large trade clustering (consecutive same-direction whale trades)
//   4. Price momentum (EMA-based, not coarse 3-point derivative)
//   5. Depth imbalance (from @depth5 stream, optional)
//
// Combined via multi-timeframe ensemble:
//   score = w_fast * sig_3s + w_mid * sig_10s + w_slow * sig_30s
// Amplified by volume spike + depth confirmation.

use std::collections::VecDeque;

// ── Configuration ────────────────────────────────────────────

/// Configuration for micro alpha computation.
pub struct MicroAlphaConfig {
    /// Rolling window in milliseconds (default: 10_000 = 10s)
    pub window_ms: i64,
    /// Large trade = qty > mean_qty * this multiplier (default: 5.0)
    pub large_trade_mult: f64,
    /// Decay half-life for large trade signal in ms (default: 3000)
    pub large_trade_decay_ms: f64,
    /// Minimum trades before signals are valid (default: 20)
    pub min_trades: usize,
}

impl Default for MicroAlphaConfig {
    fn default() -> Self {
        Self {
            window_ms: 10_000,
            large_trade_mult: 5.0,
            large_trade_decay_ms: 3000.0,
            min_trades: 20,
        }
    }
}

// ── Signal output ────────────────────────────────────────────

/// Result of micro alpha computation.
#[derive(Debug, Clone)]
pub struct MicroAlphaSignal {
    /// Buy volume - sell volume / total volume, in [-1, 1]
    pub trade_flow_imbalance: f64,
    /// Current volume rate / average volume rate
    pub volume_spike: f64,
    /// Large trade momentum signal, decaying, in [-1, 1]
    pub large_trade_signal: f64,
    /// Price acceleration (2nd derivative), normalized
    pub price_acceleration: f64,
    /// Combined micro score: weighted sum of above
    pub micro_score: f64,
    /// Number of trades in window
    pub trade_count: usize,
    /// Whether enough trades to produce valid signals
    pub valid: bool,
}

/// Enhanced signal with multi-timeframe + depth data.
#[derive(Debug, Clone)]
pub struct TickSignal {
    // Multi-timeframe flow imbalance
    pub flow_3s: f64,
    pub flow_10s: f64,
    pub flow_30s: f64,
    // Multi-timeframe ensemble score
    pub ensemble_score: f64,
    // Volume spike (window rate / baseline)
    pub volume_spike: f64,
    // Large trade cluster signal
    pub large_cluster: f64,
    // Price momentum (EMA-based)
    pub price_momentum: f64,
    // Depth imbalance (if available, else 0)
    pub depth_imbalance: f64,
    // Final combined tick score
    pub tick_score: f64,
    // Signal confidence (0-1)
    pub confidence: f64,
    // Trade count in primary window
    pub trade_count: usize,
    pub valid: bool,
}

// ── Internal types ───────────────────────────────────────────

struct TradeRecord {
    ts_ms: i64,
    qty: f64,
    is_buy: bool,
    notional: f64,
}

/// Large trade cluster tracker
struct ClusterState {
    /// Recent large trades: (ts_ms, is_buy, notional)
    events: VecDeque<(i64, bool, f64)>,
    /// Cluster detection window (ms)
    cluster_window_ms: i64,
    /// Min consecutive same-direction large trades
    min_cluster_size: usize,
}

impl ClusterState {
    fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(64),
            cluster_window_ms: 2000, // 2s window for clustering
            min_cluster_size: 2,     // 2+ same-direction = cluster
        }
    }

    fn push(&mut self, ts_ms: i64, is_buy: bool, notional: f64) {
        self.events.push_back((ts_ms, is_buy, notional));
        // Evict old
        let cutoff = ts_ms - self.cluster_window_ms * 2;
        while self.events.front().map(|(ts, _, _)| *ts < cutoff).unwrap_or(false) {
            self.events.pop_front();
        }
    }

    /// Compute cluster signal: consecutive same-direction large trades in window.
    /// Returns [-1, 1]: positive = buy cluster, negative = sell cluster.
    fn compute(&self, now_ms: i64) -> f64 {
        let cutoff = now_ms - self.cluster_window_ms;
        let mut buy_count = 0u32;
        let mut sell_count = 0u32;
        let mut buy_notional = 0.0;
        let mut sell_notional = 0.0;

        for (ts, is_buy, notional) in self.events.iter().rev() {
            if *ts < cutoff { break; }
            if *is_buy {
                buy_count += 1;
                buy_notional += notional;
            } else {
                sell_count += 1;
                sell_notional += notional;
            }
        }

        let total = buy_count + sell_count;
        if total < self.min_cluster_size as u32 {
            return 0.0;
        }

        // Cluster strength: direction dominance * notional weight
        let total_not = buy_notional + sell_notional;
        if total_not <= 0.0 { return 0.0; }

        let direction = (buy_notional - sell_notional) / total_not;
        let dominance = (buy_count.max(sell_count) as f64) / (total as f64);

        // Only signal if clear dominance (>= 70% same direction)
        if dominance < 0.7 {
            return 0.0;
        }

        // Scale: cluster of 2 = 0.5x, 3 = 0.75x, 4+ = 1.0x
        let size_scale = ((total as f64 - 1.0) / 3.0).min(1.0);
        (direction * size_scale).clamp(-1.0, 1.0)
    }
}

/// EMA price momentum tracker
struct PriceMomentum {
    /// Fast EMA (decay ~1s at BTC trade rate)
    ema_fast: f64,
    /// Slow EMA (decay ~5s)
    ema_slow: f64,
    /// Very slow EMA (decay ~30s) for baseline
    ema_baseline: f64,
    /// Last update timestamp
    last_ts: i64,
    /// Whether EMAs are initialized
    initialized: bool,
    /// Trade count for warmup
    count: u64,
}

impl PriceMomentum {
    fn new() -> Self {
        Self {
            ema_fast: 0.0,
            ema_slow: 0.0,
            ema_baseline: 0.0,
            last_ts: 0,
            initialized: false,
            count: 0,
        }
    }

    fn update(&mut self, ts_ms: i64, price: f64) {
        self.count += 1;
        if !self.initialized {
            self.ema_fast = price;
            self.ema_slow = price;
            self.ema_baseline = price;
            self.last_ts = ts_ms;
            self.initialized = true;
            return;
        }

        let dt_ms = (ts_ms - self.last_ts).max(1) as f64;
        self.last_ts = ts_ms;

        // Time-based EMA decay: alpha = 1 - exp(-dt / halflife)
        let alpha_fast = 1.0 - (-dt_ms / 1000.0).exp();      // ~1s halflife
        let alpha_slow = 1.0 - (-dt_ms / 5000.0).exp();       // ~5s halflife
        let alpha_base = 1.0 - (-dt_ms / 30000.0).exp();      // ~30s halflife

        self.ema_fast += alpha_fast * (price - self.ema_fast);
        self.ema_slow += alpha_slow * (price - self.ema_slow);
        self.ema_baseline += alpha_base * (price - self.ema_baseline);
    }

    /// Compute momentum signal: fast EMA deviation from slow, normalized by baseline.
    /// Returns [-1, 1].
    fn compute(&self) -> f64 {
        if !self.initialized || self.count < 50 || self.ema_baseline == 0.0 {
            return 0.0;
        }
        // Fast - Slow deviation, normalized by baseline price
        let dev = (self.ema_fast - self.ema_slow) / self.ema_baseline;
        // Scale: 1bp deviation = 0.1 score, 10bp = 1.0
        (dev * 10000.0 / 10.0).clamp(-1.0, 1.0)
    }
}

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

// ── Main engine ──────────────────────────────────────────────

/// Enhanced micro-alpha engine with multi-timeframe signals.
pub struct MicroAlpha {
    config: MicroAlphaConfig,
    trades: VecDeque<TradeRecord>,
    // Windowed running stats for large trade detection (decremented on eviction)
    window_qty_sum: f64,
    window_trade_count: u64,
    // Large trade events: (ts_ms, signed_notional)
    large_trades: VecDeque<(i64, f64)>,
    // Price samples for acceleration: (ts_ms, price)
    price_samples: VecDeque<(i64, f64)>,
    // V2: enhanced components
    cluster: ClusterState,
    momentum: PriceMomentum,
    // Latest depth snapshot
    depth: Option<DepthSnapshot>,
}

impl MicroAlpha {
    pub fn new(config: MicroAlphaConfig) -> Self {
        Self {
            config,
            trades: VecDeque::with_capacity(4096),
            window_qty_sum: 0.0,
            window_trade_count: 0,
            large_trades: VecDeque::with_capacity(128),
            price_samples: VecDeque::with_capacity(512),
            cluster: ClusterState::new(),
            momentum: PriceMomentum::new(),
            depth: None,
        }
    }

    /// Update depth snapshot from @depth5 stream.
    pub fn update_depth(&mut self, snapshot: DepthSnapshot) {
        self.depth = Some(snapshot);
    }

    /// Process a single aggTrade event.
    pub fn push_trade(&mut self, ts_ms: i64, price: f64, qty: f64, is_buy: bool) {
        let notional = price * qty;

        // Update momentum EMA
        self.momentum.update(ts_ms, price);

        // Update windowed running stats for large trade detection
        self.window_qty_sum += qty;
        self.window_trade_count += 1;
        let mean_qty = if self.window_trade_count > 0 {
            self.window_qty_sum / self.window_trade_count as f64
        } else {
            0.0
        };

        // Detect large trade
        if qty > mean_qty * self.config.large_trade_mult && self.window_trade_count > self.config.min_trades as u64 {
            let signed = if is_buy { notional } else { -notional };
            self.large_trades.push_back((ts_ms, signed));
            // V2: also push to cluster tracker
            self.cluster.push(ts_ms, is_buy, notional);
        }

        // Record trade
        self.trades.push_back(TradeRecord {
            ts_ms,
            qty,
            is_buy,
            notional,
        });

        // Record price sample (subsample: max 1 per 50ms for finer resolution)
        let should_sample = self.price_samples.back()
            .map(|(last_ts, _)| ts_ms - last_ts >= 50)
            .unwrap_or(true);
        if should_sample {
            self.price_samples.push_back((ts_ms, price));
        }

        // Evict old data (keep 60s for multi-timeframe)
        let cutoff = ts_ms - 60_000;
        while self.trades.front().map(|t| t.ts_ms < cutoff).unwrap_or(false) {
            if let Some(evicted) = self.trades.pop_front() {
                self.window_qty_sum -= evicted.qty;
                self.window_trade_count -= 1;
            }
        }
        while self.large_trades.front().map(|(ts, _)| *ts < cutoff).unwrap_or(false) {
            self.large_trades.pop_front();
        }
        let price_cutoff = ts_ms - 90_000;
        while self.price_samples.front().map(|(ts, _)| *ts < price_cutoff).unwrap_or(false) {
            self.price_samples.pop_front();
        }
    }

    /// Compute trade flow imbalance for a specific time window.
    fn flow_for_window(&self, now_ms: i64, window_ms: i64) -> (f64, usize, f64) {
        let cutoff = now_ms - window_ms;
        let (mut buy_not, mut sell_not) = (0.0, 0.0);
        let mut count = 0usize;
        let mut total_vol = 0.0;
        for t in self.trades.iter().rev() {
            if t.ts_ms < cutoff { break; }
            if t.is_buy { buy_not += t.notional; } else { sell_not += t.notional; }
            total_vol += t.notional;
            count += 1;
        }
        let total = buy_not + sell_not;
        let flow = if total > 0.0 { (buy_not - sell_not) / total } else { 0.0 };
        (flow, count, total_vol)
    }

    /// Compute enhanced multi-timeframe tick signal.
    pub fn compute_tick_signal(&self, now_ms: i64) -> TickSignal {
        // Multi-timeframe flow
        let (flow_3s, count_3s, _vol_3s) = self.flow_for_window(now_ms, 3_000);
        let (flow_10s, count_10s, vol_10s) = self.flow_for_window(now_ms, 10_000);
        let (flow_30s, _count_30s, _vol_30s) = self.flow_for_window(now_ms, 30_000);

        let valid = count_10s >= self.config.min_trades;
        if !valid {
            return TickSignal {
                flow_3s: 0.0, flow_10s: 0.0, flow_30s: 0.0,
                ensemble_score: 0.0, volume_spike: 0.0, large_cluster: 0.0,
                price_momentum: 0.0, depth_imbalance: 0.0, tick_score: 0.0,
                confidence: 0.0, trade_count: count_10s, valid: false,
            };
        }

        // Multi-timeframe ensemble: fast reacts, slow confirms
        // Only fire when all timeframes agree on direction
        let all_agree = (flow_3s > 0.0 && flow_10s > 0.0 && flow_30s > 0.0)
            || (flow_3s < 0.0 && flow_10s < 0.0 && flow_30s < 0.0);

        let ensemble_score = if all_agree {
            // Weighted: fast gets more weight (captures the edge before decay)
            0.5 * flow_3s + 0.3 * flow_10s + 0.2 * flow_30s
        } else {
            // Disagreement: only use if fast is very strong and mid confirms
            if flow_3s.abs() > 0.5 && flow_3s.signum() == flow_10s.signum() {
                0.6 * flow_3s + 0.4 * flow_10s
            } else {
                0.0 // conflicting signals — stay flat
            }
        };

        // Volume spike (10s window rate vs 60s baseline)
        let window_secs = 10.0;
        let window_rate = vol_10s / window_secs;
        let total_vol: f64 = self.trades.iter().map(|t| t.notional).sum();
        let total_span = self.trades.back()
            .and_then(|last| self.trades.front().map(|first| (last.ts_ms - first.ts_ms).max(1)))
            .unwrap_or(1) as f64 / 1000.0;
        let baseline_rate = total_vol / total_span;
        let volume_spike = if baseline_rate > 0.0 {
            (window_rate / baseline_rate).min(10.0)
        } else {
            1.0
        };

        // Large trade cluster signal
        let large_cluster = self.cluster.compute(now_ms);

        // Price momentum (EMA-based)
        let price_momentum = self.momentum.compute();

        // Depth imbalance (if available)
        let depth_imbalance = self.compute_depth_imbalance(now_ms);

        // ── Combine all signals ──────────────────────────────
        // Core: multi-TF flow ensemble (primary edge)
        let mut raw = 0.45 * ensemble_score
            + 0.25 * price_momentum
            + 0.20 * large_cluster
            + 0.10 * depth_imbalance;

        // Volume amplification (more aggressive than v1)
        if volume_spike > 1.5 {
            let amp = 1.0 + (volume_spike - 1.5).min(4.0) * 0.15; // up to 1.6x
            raw *= amp;
        }

        // Confidence: agreement across signals
        let signals = [ensemble_score, price_momentum, large_cluster, depth_imbalance];
        let n_agree = signals.iter().filter(|s| s.signum() == raw.signum() && s.abs() > 0.05).count();
        let confidence = match n_agree {
            4 => 1.0,
            3 => 0.8,
            2 => 0.5,
            _ => 0.3,
        };

        let tick_score = (raw * confidence).clamp(-1.0, 1.0);

        TickSignal {
            flow_3s,
            flow_10s,
            flow_30s,
            ensemble_score,
            volume_spike,
            large_cluster,
            price_momentum,
            depth_imbalance,
            tick_score,
            confidence,
            trade_count: count_10s,
            valid,
        }
    }

    fn compute_depth_imbalance(&self, now_ms: i64) -> f64 {
        match &self.depth {
            Some(d) if (now_ms - d.ts_ms) < 500 => {
                // Fresh depth (< 500ms old)
                let total = d.bid_qty + d.ask_qty;
                if total > 0.0 {
                    ((d.bid_qty - d.ask_qty) / total).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            }
            _ => 0.0, // No depth or stale
        }
    }

    // ── Legacy compute() for backward compatibility ──────────

    /// Compute all micro-alpha signals at current time (v1 API).
    pub fn compute(&self, now_ms: i64) -> MicroAlphaSignal {
        let cutoff = now_ms - self.config.window_ms;

        // Single pass over trades in window — no Vec allocation
        let (mut buy_not, mut sell_not) = (0.0, 0.0);
        let mut window_count = 0usize;
        let mut window_vol = 0.0;
        for t in self.trades.iter().rev() {
            if t.ts_ms < cutoff {
                break; // trades are time-ordered, no more in window
            }
            if t.is_buy {
                buy_not += t.notional;
            } else {
                sell_not += t.notional;
            }
            window_vol += t.notional;
            window_count += 1;
        }

        let valid = window_count >= self.config.min_trades;

        if !valid {
            return MicroAlphaSignal {
                trade_flow_imbalance: 0.0,
                volume_spike: 0.0,
                large_trade_signal: 0.0,
                price_acceleration: 0.0,
                micro_score: 0.0,
                trade_count: window_count,
                valid: false,
            };
        }

        // 1. Trade flow imbalance
        let total_not = buy_not + sell_not;
        let trade_flow_imbalance = if total_not > 0.0 {
            (buy_not - sell_not) / total_not
        } else {
            0.0
        };
        let window_secs = self.config.window_ms as f64 / 1000.0;
        let window_rate = window_vol / window_secs;

        let total_vol: f64 = self.trades.iter().map(|t| t.notional).sum();
        let total_span_ms = self.trades.back()
            .and_then(|last| self.trades.front().map(|first| last.ts_ms - first.ts_ms))
            .unwrap_or(1) as f64;
        let avg_rate = if total_span_ms > 0.0 {
            total_vol / (total_span_ms / 1000.0)
        } else {
            window_rate
        };
        let volume_spike = if avg_rate > 0.0 {
            (window_rate / avg_rate).min(10.0)
        } else {
            1.0
        };

        // 3. Large trade signal: exponentially decaying sum of signed notionals
        let mut large_trade_signal = 0.0;
        let decay_lambda = (2.0_f64).ln() / self.config.large_trade_decay_ms;
        let mut total_large_abs = 0.0;
        for (ts, signed_not) in &self.large_trades {
            if *ts >= cutoff {
                let age_ms = (now_ms - ts) as f64;
                let weight = (-decay_lambda * age_ms).exp();
                large_trade_signal += signed_not * weight;
                total_large_abs += signed_not.abs() * weight;
            }
        }
        let large_trade_signal = if total_large_abs > 0.0 {
            (large_trade_signal / total_large_abs).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 4. Price acceleration
        let price_acceleration = self.compute_price_acceleration(now_ms);

        // 5. Combined score — use V2 tick signal as micro_score
        let tick_sig = self.compute_tick_signal(now_ms);
        let micro_score = tick_sig.tick_score;

        MicroAlphaSignal {
            trade_flow_imbalance,
            volume_spike,
            large_trade_signal,
            price_acceleration,
            micro_score,
            trade_count: window_count,
            valid,
        }
    }

    fn compute_price_acceleration(&self, now_ms: i64) -> f64 {
        if self.price_samples.len() < 3 {
            return 0.0;
        }

        let window_ms = self.config.window_ms;
        let t_now = now_ms;
        let t_mid = now_ms - window_ms / 2;
        let t_old = now_ms - window_ms;

        let p_now = self.find_nearest_price(t_now);
        let p_mid = self.find_nearest_price(t_mid);
        let p_old = self.find_nearest_price(t_old);

        match (p_now, p_mid, p_old) {
            (Some(now), Some(mid), Some(old)) => {
                if old == 0.0 { return 0.0; }
                let v1 = (now - mid) / old;
                let v0 = (mid - old) / old;
                let accel = v1 - v0;
                (accel * 1000.0).clamp(-1.0, 1.0)
            }
            _ => 0.0,
        }
    }

    fn find_nearest_price(&self, target_ms: i64) -> Option<f64> {
        let mut best: Option<(i64, f64)> = None;
        for (ts, price) in &self.price_samples {
            let dist = (*ts - target_ms).abs();
            match best {
                Some((best_dist, _)) if dist < best_dist => {
                    best = Some((dist, *price));
                }
                None => {
                    best = Some((dist, *price));
                }
                _ => {
                    if let Some((best_dist, _)) = best {
                        if dist > best_dist {
                            break;
                        }
                    }
                }
            }
        }
        best.map(|(_, p)| p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_signals() {
        let ma = MicroAlpha::new(MicroAlphaConfig::default());
        let sig = ma.compute(1000);
        assert!(!sig.valid);
        assert_eq!(sig.micro_score, 0.0);
    }

    #[test]
    fn test_buy_flow_imbalance() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..10 {
            ma.push_trade(now - 5000 + i * 500, 50000.0, 0.1, true);
        }

        let sig = ma.compute(now);
        assert!(sig.valid);
        assert!(sig.trade_flow_imbalance > 0.9, "Expected strong buy imbalance, got {}", sig.trade_flow_imbalance);
        assert!(sig.micro_score > 0.0);
    }

    #[test]
    fn test_sell_flow_imbalance() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..10 {
            ma.push_trade(now - 5000 + i * 500, 50000.0, 0.1, false);
        }

        let sig = ma.compute(now);
        assert!(sig.valid);
        assert!(sig.trade_flow_imbalance < -0.9);
        assert!(sig.micro_score < 0.0);
    }

    #[test]
    fn test_large_trade_detection() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            large_trade_mult: 3.0,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..30 {
            ma.push_trade(now - 15000 + i * 500, 50000.0, 0.01, true);
        }
        ma.push_trade(now - 100, 50000.0, 1.0, true);

        let sig = ma.compute(now);
        assert!(sig.valid);
        assert!(sig.large_trade_signal > 0.0, "Expected positive large trade signal");
    }

    #[test]
    fn test_volume_spike() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 5_000,
            min_trades: 3,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..10 {
            ma.push_trade(now - 9000 + i * 500, 50000.0, 0.001, true);
        }
        for i in 0..10 {
            ma.push_trade(now - 4000 + i * 400, 50000.0, 0.1, true);
        }

        let sig = ma.compute(now);
        assert!(sig.valid);
        assert!(sig.volume_spike > 1.5, "Expected volume spike > 1.5, got {}", sig.volume_spike);
    }

    #[test]
    fn test_multi_timeframe_agreement() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        // Sustained buy pressure across 30s
        for i in 0..60 {
            ma.push_trade(now - 30000 + i * 500, 50000.0, 0.1, true);
        }

        let sig = ma.compute_tick_signal(now);
        assert!(sig.valid);
        assert!(sig.flow_3s > 0.5);
        assert!(sig.flow_10s > 0.5);
        assert!(sig.flow_30s > 0.5);
        assert!(sig.tick_score > 0.0, "Expected positive tick score for all-buy flow");
    }

    #[test]
    fn test_conflicting_timeframes() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        // Sell pressure in the past (30s)
        for i in 0..30 {
            ma.push_trade(now - 25000 + i * 500, 50000.0, 0.1, false);
        }
        // Buy pressure recently (3s)
        for i in 0..20 {
            ma.push_trade(now - 2500 + i * 125, 50000.0, 0.1, true);
        }

        let sig = ma.compute_tick_signal(now);
        assert!(sig.valid);
        // 3s should be buy, 30s should be sell → conflicting
        assert!(sig.flow_3s > 0.0);
        assert!(sig.flow_30s < 0.0);
        // Score should be dampened due to disagreement
        assert!(sig.tick_score.abs() < 0.5, "Expected dampened score, got {}", sig.tick_score);
    }

    #[test]
    fn test_tick_position_roundtrip() {
        let mut pos = TickPosition::new();

        // Open long
        let cost = pos.enter(1000, 50000.0, 0.1, 2e-4);
        assert!(cost < 0.0); // paid cost
        assert_eq!(pos.qty, 0.1);

        // Close long at profit
        let pnl = pos.enter(2000, 50100.0, -0.1, 2e-4);
        assert!(pnl > 0.0); // 50100-50000 = 100 * 0.1 = 10 - costs
        assert_eq!(pos.qty, 0.0);
        assert_eq!(pos.trade_count, 1);
        assert_eq!(pos.win_count, 1);
    }

    #[test]
    fn test_depth_imbalance() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        for i in 0..10 {
            ma.push_trade(now - 5000 + i * 500, 50000.0, 0.1, true);
        }

        // Push depth with bid-heavy imbalance
        ma.update_depth(DepthSnapshot {
            ts_ms: now - 100,
            best_bid: 49999.0,
            best_ask: 50001.0,
            bid_qty: 10.0,
            ask_qty: 2.0,
        });

        let sig = ma.compute_tick_signal(now);
        assert!(sig.depth_imbalance > 0.0, "Expected positive depth imbalance");
    }

    #[test]
    fn test_large_trade_cluster() {
        let mut ma = MicroAlpha::new(MicroAlphaConfig {
            window_ms: 10_000,
            large_trade_mult: 3.0,
            min_trades: 5,
            ..Default::default()
        });

        let now = 100_000i64;
        // Build baseline
        for i in 0..30 {
            ma.push_trade(now - 20000 + i * 500, 50000.0, 0.01, true);
        }
        // 3 consecutive large buy trades in 1s
        ma.push_trade(now - 800, 50000.0, 1.0, true);
        ma.push_trade(now - 400, 50000.0, 1.5, true);
        ma.push_trade(now - 100, 50000.0, 2.0, true);

        let sig = ma.compute_tick_signal(now);
        assert!(sig.large_cluster > 0.0, "Expected positive cluster signal, got {}", sig.large_cluster);
    }
}
