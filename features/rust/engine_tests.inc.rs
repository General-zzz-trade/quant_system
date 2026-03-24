// tests.inc.rs — Included by engine.rs via include!() macro.
#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine_with_bars(n: usize) -> BarState {
        let mut state = BarState::new();
        for i in 0..n {
            let price = 100.0 + (i as f64) * 0.1;
            state.push(
                price, 1000.0 + i as f64, price + 0.5, price - 0.5, price - 0.1,
                (i % 24) as i32, (i % 7) as i32,
                0.0001, 100.0, 500.0, 50000.0, 25000.0,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, -1,
                f64::NAN, f64::NAN,
            );
        }
        state
    }

    #[test]
    fn test_checkpoint_restore_features_match() {
        // Build original engine with 100 bars
        let mut original = BarState::new();
        let mut original_momentum = f64::NAN;
        for i in 0..100 {
            let price = 100.0 + (i as f64) * 0.1;
            original.push(
                price, 1000.0 + i as f64, price + 0.5, price - 0.5, price - 0.1,
                (i % 24) as i32, (i % 7) as i32,
                0.0001, 100.0, 500.0, 50000.0, 25000.0,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, -1,
                f64::NAN, f64::NAN,
            );
            original.prev_momentum = original_momentum;
            let mut out = [f64::NAN; N_FEATURES];
            original.get_features(&mut out);
            original_momentum = out[F_MA_CROSS_10_30];
            original.prev_momentum = original_momentum;
        }

        // Serialize bar history
        let bars = original.get_bar_history();
        let bars_json: Vec<serde_json::Value> = bars.iter().map(|b| {
            serde_json::json!({
                "c": b.close, "v": b.volume, "h": b.high, "l": b.low, "o": b.open,
                "hour": b.hour, "dow": b.dow,
                "fr": b.funding_rate, "trades": b.trades,
                "tbv": b.taker_buy_volume, "qv": b.quote_volume, "tbqv": b.taker_buy_quote_volume,
                "oi": b.open_interest, "ls": b.ls_ratio, "spot": b.spot_close, "fg": b.fear_greed,
                "iv": b.implied_vol, "pcr": b.put_call_ratio,
                "oc_fi": b.oc_flow_in, "oc_fo": b.oc_flow_out,
                "oc_s": b.oc_supply, "oc_a": b.oc_addr, "oc_t": b.oc_tx, "oc_h": b.oc_hashrate,
                "liq_tv": b.liq_total_vol, "liq_bv": b.liq_buy_vol, "liq_sv": b.liq_sell_vol, "liq_c": b.liq_count,
                "mp_ff": b.mempool_fastest_fee, "mp_ef": b.mempool_economy_fee, "mp_s": b.mempool_size,
                "m_dxy": b.macro_dxy, "m_spx": b.macro_spx, "m_vix": b.macro_vix, "m_day": b.macro_day,
                "sv": b.social_volume, "ss": b.sentiment_score,
            })
        }).collect();

        let checkpoint = serde_json::json!({
            "version": 1,
            "bar_count": original.bar_count,
            "bars": bars_json,
        });
        let json_str = serde_json::to_string(&checkpoint).unwrap();

        // Restore into fresh engine
        let mut restored = BarState::new();
        let mut restored_momentum = f64::NAN;
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let parsed_bars = parsed.get("bars").unwrap().as_array().unwrap();

        for bar in parsed_bars {
            restored.push(
                bar["c"].as_f64().unwrap(),
                bar["v"].as_f64().unwrap_or(0.0),
                bar["h"].as_f64().unwrap_or(0.0),
                bar["l"].as_f64().unwrap_or(0.0),
                bar["o"].as_f64().unwrap_or(0.0),
                bar["hour"].as_i64().unwrap_or(-1) as i32,
                bar["dow"].as_i64().unwrap_or(-1) as i32,
                bar.get("fr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("trades").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("tbv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("qv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("tbqv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("oi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("ls").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("spot").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("fg").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("iv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("pcr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_fi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_fo").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_s").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_a").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_t").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_h").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_tv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_bv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_sv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_c").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_ff").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_ef").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_s").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_dxy").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_spx").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_vix").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_day").and_then(|v| v.as_i64()).unwrap_or(-1),
                bar.get("sv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("ss").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
            );
            restored.prev_momentum = restored_momentum;
            let mut out = [f64::NAN; N_FEATURES];
            restored.get_features(&mut out);
            restored_momentum = out[F_MA_CROSS_10_30];
            restored.prev_momentum = restored_momentum;
        }

        assert_eq!(original.bar_count, restored.bar_count);

        // Push one more bar to both and compare features
        let next_price = 100.0 + 100.0 * 0.1;
        let push_one = |state: &mut BarState, mom: &mut f64| {
            state.push(
                next_price, 1100.0, next_price + 0.5, next_price - 0.5, next_price - 0.1,
                4, 3, 0.0001, 100.0, 500.0, 50000.0, 25000.0,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN,
                f64::NAN, f64::NAN, f64::NAN, -1,
                f64::NAN, f64::NAN,
            );
            state.prev_momentum = *mom;
            let mut out = [f64::NAN; N_FEATURES];
            state.get_features(&mut out);
            *mom = out[F_MA_CROSS_10_30];
            state.prev_momentum = *mom;
            out
        };

        let orig_features = push_one(&mut original, &mut original_momentum);
        let rest_features = push_one(&mut restored, &mut restored_momentum);

        for i in 0..N_FEATURES {
            let a = orig_features[i];
            let b = rest_features[i];
            if a.is_nan() && b.is_nan() { continue; }
            assert!(
                (a - b).abs() < 1e-10,
                "Feature {} ({}) mismatch: original={}, restored={}",
                i, FEATURE_NAMES[i], a, b,
            );
        }
    }

    #[test]
    fn test_checkpoint_bar_count() {
        let state = make_engine_with_bars(50);
        assert_eq!(state.bar_count, 50);
        assert_eq!(state.get_bar_history().len(), 50);
    }

    #[test]
    fn test_checkpoint_cap_720() {
        let state = make_engine_with_bars(800);
        assert_eq!(state.bar_count, 800);
        assert_eq!(state.get_bar_history().len(), 720);
    }
}
