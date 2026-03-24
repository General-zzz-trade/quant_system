// micro_alpha_compute.inc.rs — Legacy compute() + price helpers for MicroAlpha.
// Included by micro_alpha.rs via include!() macro.

impl MicroAlpha {
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
