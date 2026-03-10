// micro_alpha.rs — Microstructure alpha signals for 1-second trading
//
// Computes trade flow imbalance, volume spikes, large trade detection,
// and price acceleration from aggTrade stream data.
//
// These signals are designed for sub-minute alpha exploration where
// traditional kline-based features are too slow.

use std::collections::VecDeque;

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

struct TradeRecord {
    ts_ms: i64,
    qty: f64,
    is_buy: bool,
    notional: f64,
}

/// Micro-alpha engine processing aggTrade data.
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
}

impl MicroAlpha {
    pub fn new(config: MicroAlphaConfig) -> Self {
        Self {
            config,
            trades: VecDeque::with_capacity(2048),
            window_qty_sum: 0.0,
            window_trade_count: 0,
            large_trades: VecDeque::with_capacity(128),
            price_samples: VecDeque::with_capacity(256),
        }
    }

    /// Process a single aggTrade event.
    pub fn push_trade(&mut self, ts_ms: i64, price: f64, qty: f64, is_buy: bool) {
        let notional = price * qty;

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
        }

        // Record trade
        self.trades.push_back(TradeRecord {
            ts_ms,
            qty,
            is_buy,
            notional,
        });

        // Record price sample (subsample: max 1 per 100ms)
        let should_sample = self.price_samples.back()
            .map(|(last_ts, _)| ts_ms - last_ts >= 100)
            .unwrap_or(true);
        if should_sample {
            self.price_samples.push_back((ts_ms, price));
        }

        // Evict old data (keep 2x window for stats stability)
        // Decrement windowed stats when evicting trades
        let cutoff = ts_ms - self.config.window_ms * 2;
        while self.trades.front().map(|t| t.ts_ms < cutoff).unwrap_or(false) {
            if let Some(evicted) = self.trades.pop_front() {
                self.window_qty_sum -= evicted.qty;
                self.window_trade_count -= 1;
            }
        }
        while self.large_trades.front().map(|(ts, _)| *ts < cutoff).unwrap_or(false) {
            self.large_trades.pop_front();
        }
        let price_cutoff = ts_ms - self.config.window_ms * 3;
        while self.price_samples.front().map(|(ts, _)| *ts < price_cutoff).unwrap_or(false) {
            self.price_samples.pop_front();
        }
    }

    /// Compute all micro-alpha signals at current time.
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

        // 1. Trade flow imbalance: (buy_notional - sell_notional) / total_notional
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
            (window_rate / avg_rate).min(10.0) // cap at 10x
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
        // Normalize to [-1, 1]
        let large_trade_signal = if total_large_abs > 0.0 {
            (large_trade_signal / total_large_abs).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // 4. Price acceleration: 2nd derivative using 3 time samples
        let price_acceleration = self.compute_price_acceleration(now_ms);

        // 5. Combined score: weighted sum
        // Weights tuned for 1s BTC:
        //   trade_flow (0.35) - strongest short-term predictor
        //   large_trade (0.30) - high conviction events
        //   volume_spike as amplifier (multiplicative, not additive)
        //   price_accel (0.35) - momentum continuation
        let raw_score = 0.35 * trade_flow_imbalance
            + 0.30 * large_trade_signal
            + 0.35 * price_acceleration;

        // Volume spike amplifies the signal (above 2x volume = stronger signal)
        let vol_amp = if volume_spike > 2.0 {
            1.0 + (volume_spike - 2.0).min(3.0) * 0.2 // up to 1.6x amplification
        } else {
            1.0
        };

        let micro_score = (raw_score * vol_amp).clamp(-1.0, 1.0);

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
        // Need at least 3 price samples spread across the window
        if self.price_samples.len() < 3 {
            return 0.0;
        }

        let window_ms = self.config.window_ms;
        let t_now = now_ms;
        let t_mid = now_ms - window_ms / 2;
        let t_old = now_ms - window_ms;

        // Find closest prices to each time point
        let p_now = self.find_nearest_price(t_now);
        let p_mid = self.find_nearest_price(t_mid);
        let p_old = self.find_nearest_price(t_old);

        match (p_now, p_mid, p_old) {
            (Some(now), Some(mid), Some(old)) => {
                if old == 0.0 { return 0.0; }
                // velocity = (now - mid) / dt, old_velocity = (mid - old) / dt
                // acceleration = (velocity - old_velocity) / dt
                // Normalize by price level
                let v1 = (now - mid) / old; // recent velocity (normalized)
                let v0 = (mid - old) / old; // older velocity (normalized)
                let accel = v1 - v0;
                // Clamp to reasonable range (±0.1% acceleration per window)
                (accel * 1000.0).clamp(-1.0, 1.0) // scale up for sensitivity
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
                    // If distance is increasing and we already have a candidate,
                    // we passed the optimal point (samples are sorted by time)
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
        // All buys
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
        // All sells
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
        // Small trades first (establish baseline)
        for i in 0..30 {
            ma.push_trade(now - 15000 + i * 500, 50000.0, 0.01, true);
        }
        // One large buy trade
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
        // Low volume early
        for i in 0..10 {
            ma.push_trade(now - 9000 + i * 500, 50000.0, 0.001, true);
        }
        // High volume recently
        for i in 0..10 {
            ma.push_trade(now - 4000 + i * 400, 50000.0, 0.1, true);
        }

        let sig = ma.compute(now);
        assert!(sig.valid);
        assert!(sig.volume_spike > 1.5, "Expected volume spike > 1.5, got {}", sig.volume_spike);
    }
}
