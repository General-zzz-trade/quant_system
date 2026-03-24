// unified_predictor_signal.inc.rs — Signal pipeline + prediction methods.
// Included by unified_predictor.rs via include!() macro.

impl RustUnifiedPredictor {
    #[inline]
    pub(crate) fn predict_ensemble(&mut self) -> f64 {
        if self.main_models.len() == 1 {
            return self.main_models[0].predict(&self.features_buf, &mut self.model_buf);
        }
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for (model, &w) in self.main_models.iter().zip(self.ensemble_weights.iter()) {
            let score = model.predict(&self.features_buf, &mut self.model_buf);
            weighted_sum += score * w;
            weight_sum += w;
        }
        if weight_sum > 0.0 { weighted_sum / weight_sum } else { 0.0 }
    }

    pub(crate) fn apply_signal_pipeline(
        &mut self,
        symbol: &str,
        raw_score: f64,
        close: f64,
        hour_key: i64,
        cfg: &CfgSnapshot,
    ) -> f64 {
        // No constraints → raw score
        if cfg.min_hold <= 0 {
            return raw_score;
        }

        let gate_w = cfg.monthly_gate_window;
        let state = self.bridge_states.entry(symbol.to_string())
            .or_insert_with(|| SymbolState::new(self.zscore_window, gate_w));

        // Monthly gate check (shared)
        let gate_ok = if cfg.monthly_gate {
            update_monthly_gate(
                &mut state.close_history, &mut state.gate_last_hour,
                close, hour_key, gate_w,
            )
        } else {
            true
        };

        // Z-score normalization
        let z = if self.zscore_warmup > 0 {
            match zscore_from_buf(
                &mut state.zscore_buf, &mut state.zscore_last_hour,
                raw_score, hour_key, self.zscore_window, self.zscore_warmup,
            ) {
                Some(z) => z,
                None => {
                    // Warmup: increment hold counter to match backtest behavior.
                    // In backtest, min-hold runs over warmup bars (raw=0.0) starting
                    // at hold_count=1, so by bar k the count is k+1. Replicating
                    // that here ensures the first post-warmup bar has the same
                    // hold state in both paths.
                    if state.hold_counter == 0 {
                        state.hold_counter = 1;
                    } else {
                        state.hold_counter += 1;
                    }
                    return state.position;
                }
            }
        } else {
            raw_score
        };

        // Long-only clip + discretize (shared)
        let z = long_only_clip(z, cfg.long_only);
        let desired = discretize(z, cfg.deadzone);

        // Min-hold + trend-hold (shared)
        let prev_pos = state.position;
        let hold_count = if state.hold_counter == 0 { cfg.min_hold } else { state.hold_counter };

        // Get trend value from features buffer
        let trend_val = if cfg.trend_follow && cfg.trend_indicator_idx < N_FEATURES as u16 {
            self.features_buf[cfg.trend_indicator_idx as usize]
        } else {
            f64::NAN
        };

        let (mut score, new_hold) = enforce_hold_step(
            desired, prev_pos, hold_count, cfg.min_hold,
            cfg.trend_follow, trend_val, cfg.trend_threshold, cfg.max_hold,
        );
        state.hold_counter = new_hold;
        if score != prev_pos {
            state.position = score;
        }

        // Bear regime handling (monthly gate failed)
        if !gate_ok {
            if self.bear_model.is_some() {
                let bear_score = self.bear_model.as_ref().unwrap()
                    .predict(&self.features_buf, &mut self.model_buf);
                let prob = bear_score + 0.5;
                if bear_score > 0.0 && !cfg.bear_thresholds.is_empty() {
                    score = 0.0;
                    for &(thresh, sig) in &cfg.bear_thresholds {
                        if prob > thresh {
                            score = sig;
                            break;
                        }
                    }
                } else {
                    score = 0.0;
                }
            } else if score != 0.0 {
                score = 0.0;
            }
            if score != state.position {
                state.position = score;
                state.hold_counter = 1;
            }
        }

        // Vol-adaptive sizing (shared)
        if let Some(vt) = cfg.vol_target {
            if cfg.vol_feature_idx < N_FEATURES as u16 {
                let vol_val = self.features_buf[cfg.vol_feature_idx as usize];
                score = vol_scale(score, vol_val, vt);
            }
        }

        score
    }

    pub(crate) fn predict_short(
        &mut self,
        symbol: &str,
        hour_key: i64,
        cfg: &CfgSnapshot,
    ) -> f64 {
        let short_model = match &self.short_model {
            Some(m) => m,
            None => return 0.0,
        };

        // Check for NaN in features (skip if any NaN in model features)
        let has_nan = short_model.feature_map.iter().any(|&idx| {
            if (idx as usize) < N_FEATURES {
                self.features_buf[idx as usize].is_nan()
            } else {
                true
            }
        });
        if has_nan {
            return 0.0;
        }

        let raw = short_model.predict(&self.features_buf, &mut self.model_buf);

        if cfg.min_hold <= 0 {
            return raw.min(0.0);
        }

        let gate_w = cfg.monthly_gate_window;
        let state = self.bridge_states.entry(symbol.to_string())
            .or_insert_with(|| SymbolState::new(self.zscore_window, gate_w));

        // Z-score for short buffer
        let z = match zscore_from_buf(
            &mut state.short_zscore_buf, &mut state.short_zscore_last_hour,
            raw, hour_key, self.zscore_window, self.zscore_warmup,
        ) {
            Some(z) => z,
            None => return state.short_position,
        };

        // Short-only discretize (shared)
        let desired = discretize_short(z, cfg.deadzone);

        // Min-hold enforcement (shared)
        let prev = state.short_position;
        let hold = if state.short_hold_counter == 0 { cfg.min_hold } else { state.short_hold_counter };

        let (output, new_hold) = enforce_short_hold_step(desired, prev, hold, cfg.min_hold);
        state.short_hold_counter = new_hold;
        if output != prev {
            state.short_position = output;
        }

        // Vol-adaptive sizing (shared)
        let mut score = output;
        if let Some(vt) = cfg.vol_target {
            if cfg.vol_feature_idx < N_FEATURES as u16 {
                let vol_val = self.features_buf[cfg.vol_feature_idx as usize];
                score = vol_scale(score, vol_val, vt);
            }
        }
        score
    }
}
