// unified_predictor_pymethods.inc.rs — PyO3 push/predict/checkpoint methods.
// Included by unified_predictor.rs via include!() macro.

#[pymethods]
impl RustUnifiedPredictor {
    /// Update cached external data for a symbol.
    /// Only call when source values change (funding: 8h, OI: 5m, macro: 1d, etc).
    /// push_bar_predict() will use these cached values automatically.
    #[pyo3(signature = (
        symbol,
        hour=-1, dow=-1,
        funding_rate=f64::NAN,
        trades=0.0,
        taker_buy_volume=0.0,
        quote_volume=0.0,
        taker_buy_quote_volume=0.0,
        open_interest=f64::NAN,
        ls_ratio=f64::NAN,
        spot_close=f64::NAN,
        fear_greed=f64::NAN,
        implied_vol=f64::NAN,
        put_call_ratio=f64::NAN,
        oc_flow_in=f64::NAN, oc_flow_out=f64::NAN,
        oc_supply=f64::NAN, oc_addr=f64::NAN,
        oc_tx=f64::NAN, oc_hashrate=f64::NAN,
        liq_total_vol=f64::NAN, liq_buy_vol=f64::NAN, liq_sell_vol=f64::NAN,
        liq_count=f64::NAN,
        mempool_fastest_fee=f64::NAN, mempool_economy_fee=f64::NAN,
        mempool_size=f64::NAN,
        macro_dxy=f64::NAN, macro_spx=f64::NAN, macro_vix=f64::NAN,
        macro_day=-1_i64,
        social_volume=f64::NAN, sentiment_score=f64::NAN,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn push_external_data(
        &mut self,
        symbol: &str,
        hour: i32, dow: i32,
        funding_rate: f64,
        trades: f64,
        taker_buy_volume: f64,
        quote_volume: f64,
        taker_buy_quote_volume: f64,
        open_interest: f64,
        ls_ratio: f64,
        spot_close: f64,
        fear_greed: f64,
        implied_vol: f64,
        put_call_ratio: f64,
        oc_flow_in: f64, oc_flow_out: f64,
        oc_supply: f64, oc_addr: f64,
        oc_tx: f64, oc_hashrate: f64,
        liq_total_vol: f64, liq_buy_vol: f64, liq_sell_vol: f64,
        liq_count: f64,
        mempool_fastest_fee: f64, mempool_economy_fee: f64,
        mempool_size: f64,
        macro_dxy: f64, macro_spx: f64, macro_vix: f64,
        macro_day: i64,
        social_volume: f64, sentiment_score: f64,
    ) {
        let ext = self.external_data.entry(symbol.to_string())
            .or_insert_with(ExternalData::default);
        ext.hour = hour;
        ext.dow = dow;
        ext.funding_rate = funding_rate;
        ext.trades = trades;
        ext.taker_buy_volume = taker_buy_volume;
        ext.quote_volume = quote_volume;
        ext.taker_buy_quote_volume = taker_buy_quote_volume;
        ext.open_interest = open_interest;
        ext.ls_ratio = ls_ratio;
        ext.spot_close = spot_close;
        ext.fear_greed = fear_greed;
        ext.implied_vol = implied_vol;
        ext.put_call_ratio = put_call_ratio;
        ext.oc_flow_in = oc_flow_in;
        ext.oc_flow_out = oc_flow_out;
        ext.oc_supply = oc_supply;
        ext.oc_addr = oc_addr;
        ext.oc_tx = oc_tx;
        ext.oc_hashrate = oc_hashrate;
        ext.liq_total_vol = liq_total_vol;
        ext.liq_buy_vol = liq_buy_vol;
        ext.liq_sell_vol = liq_sell_vol;
        ext.liq_count = liq_count;
        ext.mempool_fastest_fee = mempool_fastest_fee;
        ext.mempool_economy_fee = mempool_economy_fee;
        ext.mempool_size = mempool_size;
        ext.macro_dxy = macro_dxy;
        ext.macro_spx = macro_spx;
        ext.macro_vix = macro_vix;
        ext.macro_day = macro_day;
        ext.social_volume = social_volume;
        ext.sentiment_score = sentiment_score;
    }

    /// Slim push: uses cached external data. Call push_external_data() first.
    /// Only needs OHLCV + hour_key per bar.
    #[pyo3(signature = (symbol, close, volume, high, low, open, hour_key))]
    fn push_bar_predict<'py>(
        &mut self,
        py: Python<'py>,
        symbol: &str,
        close: f64, volume: f64, high: f64, low: f64, open: f64,
        hour_key: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ext = self.external_data.get(symbol).cloned().unwrap_or_default();
        self.push_bar_and_predict(
            py, symbol, close, volume, high, low, open, hour_key,
            ext.hour, ext.dow, ext.funding_rate,
            ext.trades, ext.taker_buy_volume,
            ext.quote_volume, ext.taker_buy_quote_volume,
            ext.open_interest, ext.ls_ratio,
            ext.spot_close, ext.fear_greed,
            ext.implied_vol, ext.put_call_ratio,
            ext.oc_flow_in, ext.oc_flow_out,
            ext.oc_supply, ext.oc_addr,
            ext.oc_tx, ext.oc_hashrate,
            ext.liq_total_vol, ext.liq_buy_vol, ext.liq_sell_vol,
            ext.liq_count,
            ext.mempool_fastest_fee, ext.mempool_economy_fee,
            ext.mempool_size,
            ext.macro_dxy, ext.macro_spx, ext.macro_vix,
            ext.macro_day,
            ext.social_volume, ext.sentiment_score,
        )
    }

    /// Push bar data and return prediction signal in one call.
    ///
    /// Returns dict: {"ml_score": f64, "ml_short_score": f64, "raw_score": f64}
    /// All signal processing (z-score, min-hold, monthly gate, bear model, vol sizing)
    /// happens in Rust with zero Python overhead.
    #[pyo3(signature = (
        symbol, close, volume, high, low, open, hour_key,
        hour=-1, dow=-1,
        funding_rate=f64::NAN,
        trades=0.0,
        taker_buy_volume=0.0,
        quote_volume=0.0,
        taker_buy_quote_volume=0.0,
        open_interest=f64::NAN,
        ls_ratio=f64::NAN,
        spot_close=f64::NAN,
        fear_greed=f64::NAN,
        implied_vol=f64::NAN,
        put_call_ratio=f64::NAN,
        oc_flow_in=f64::NAN, oc_flow_out=f64::NAN,
        oc_supply=f64::NAN, oc_addr=f64::NAN,
        oc_tx=f64::NAN, oc_hashrate=f64::NAN,
        liq_total_vol=f64::NAN, liq_buy_vol=f64::NAN, liq_sell_vol=f64::NAN,
        liq_count=f64::NAN,
        mempool_fastest_fee=f64::NAN, mempool_economy_fee=f64::NAN,
        mempool_size=f64::NAN,
        macro_dxy=f64::NAN, macro_spx=f64::NAN, macro_vix=f64::NAN,
        macro_day=-1_i64,
        social_volume=f64::NAN, sentiment_score=f64::NAN,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn push_bar_and_predict<'py>(
        &mut self,
        py: Python<'py>,
        symbol: &str,
        close: f64, volume: f64, high: f64, low: f64, open: f64,
        hour_key: i64,
        hour: i32, dow: i32,
        funding_rate: f64,
        trades: f64,
        taker_buy_volume: f64,
        quote_volume: f64,
        taker_buy_quote_volume: f64,
        open_interest: f64,
        ls_ratio: f64,
        spot_close: f64,
        fear_greed: f64,
        implied_vol: f64,
        put_call_ratio: f64,
        oc_flow_in: f64, oc_flow_out: f64,
        oc_supply: f64, oc_addr: f64,
        oc_tx: f64, oc_hashrate: f64,
        liq_total_vol: f64, liq_buy_vol: f64, liq_sell_vol: f64,
        liq_count: f64,
        mempool_fastest_fee: f64, mempool_economy_fee: f64,
        mempool_size: f64,
        macro_dxy: f64, macro_spx: f64, macro_vix: f64,
        macro_day: i64,
        social_volume: f64, sentiment_score: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        // 1) Ensure engine exists
        let engine = self.engines.entry(symbol.to_string())
            .or_insert_with(BarState::new);

        // 2) Push bar data → updates all rolling state
        engine.push(
            close, volume, high, low, open,
            hour, dow,
            funding_rate, trades,
            taker_buy_volume, quote_volume, taker_buy_quote_volume,
            open_interest, ls_ratio, spot_close, fear_greed,
            implied_vol, put_call_ratio,
            oc_flow_in, oc_flow_out,
            oc_supply, oc_addr,
            oc_tx, oc_hashrate,
            liq_total_vol, liq_buy_vol, liq_sell_vol,
            liq_count,
            mempool_fastest_fee, mempool_economy_fee,
            mempool_size,
            macro_dxy, macro_spx, macro_vix,
            macro_day,
            social_volume, sentiment_score,
        );

        // 3) Get features (writes to internal buffer, no allocation)
        engine.get_features(&mut self.features_buf);

        // 4) ML prediction (ensemble: weighted average of all main models)
        let raw_score = self.predict_ensemble();

        // 5) Extract config snapshot (avoids borrow conflict with &mut self)
        let cfg_snap = self.configs.get(symbol).map(|c| CfgSnapshot {
            min_hold: c.min_hold,
            deadzone: c.deadzone,
            long_only: c.long_only,
            trend_follow: c.trend_follow,
            trend_threshold: c.trend_threshold,
            trend_indicator_idx: c.trend_indicator_idx,
            max_hold: c.max_hold,
            monthly_gate: c.monthly_gate,
            monthly_gate_window: c.monthly_gate_window,
            vol_target: c.vol_target,
            vol_feature_idx: c.vol_feature_idx,
            bear_thresholds: c.bear_thresholds.clone(),
        }).unwrap_or_default();

        // 6) Signal constraints
        let score = self.apply_signal_pipeline(
            symbol, raw_score, close, hour_key, &cfg_snap,
        );

        // 7) Short model (if configured)
        let short_score = if self.short_model.is_some() {
            self.predict_short(symbol, hour_key, &cfg_snap)
        } else {
            0.0
        };

        // 8) Build result dict
        let dict = PyDict::new(py);
        dict.set_item("ml_score", score)?;
        dict.set_item("ml_short_score", short_score)?;
        dict.set_item("raw_score", raw_score)?;
        Ok(dict)
    }

    /// Get features as PyDict, skipping NaN values (no None in result).
    /// This avoids the Python-side `{k:v for k,v in d.items() if v is not None}` filter.
    fn get_features<'py>(&self, py: Python<'py>, symbol: &str) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(engine) = self.engines.get(symbol) {
            let mut buf = [f64::NAN; N_FEATURES];
            engine.get_features(&mut buf);
            for (i, &name) in FEATURE_NAMES.iter().enumerate() {
                let v = buf[i];
                if !v.is_nan() {
                    dict.set_item(name, v)?;
                }
            }
        }
        Ok(dict)
    }

    /// Combined push_bar + predict + get_features in one call.
    /// Returns (prediction_dict, features_dict) — eliminates separate get_features call.
    #[pyo3(signature = (symbol, close, volume, high, low, open, hour_key))]
    fn push_bar_predict_features<'py>(
        &mut self,
        py: Python<'py>,
        symbol: &str,
        close: f64, volume: f64, high: f64, low: f64, open: f64,
        hour_key: i64,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        // 1. Push bar + predict (reuses cached external data)
        let ext = self.external_data.get(symbol).cloned().unwrap_or_default();

        let engine = self.engines.entry(symbol.to_string())
            .or_insert_with(BarState::new);

        engine.push(
            close, volume, high, low, open,
            ext.hour, ext.dow,
            ext.funding_rate, ext.trades,
            ext.taker_buy_volume, ext.quote_volume, ext.taker_buy_quote_volume,
            ext.open_interest, ext.ls_ratio, ext.spot_close, ext.fear_greed,
            ext.implied_vol, ext.put_call_ratio,
            ext.oc_flow_in, ext.oc_flow_out,
            ext.oc_supply, ext.oc_addr,
            ext.oc_tx, ext.oc_hashrate,
            ext.liq_total_vol, ext.liq_buy_vol, ext.liq_sell_vol,
            ext.liq_count,
            ext.mempool_fastest_fee, ext.mempool_economy_fee,
            ext.mempool_size,
            ext.macro_dxy, ext.macro_spx, ext.macro_vix,
            ext.macro_day,
            ext.social_volume, ext.sentiment_score,
        );

        engine.get_features(&mut self.features_buf);

        let raw_score = self.predict_ensemble();

        let cfg_snap = self.configs.get(symbol).map(|c| CfgSnapshot {
            min_hold: c.min_hold, deadzone: c.deadzone, long_only: c.long_only,
            trend_follow: c.trend_follow, trend_threshold: c.trend_threshold,
            trend_indicator_idx: c.trend_indicator_idx, max_hold: c.max_hold,
            monthly_gate: c.monthly_gate, monthly_gate_window: c.monthly_gate_window,
            vol_target: c.vol_target, vol_feature_idx: c.vol_feature_idx,
            bear_thresholds: c.bear_thresholds.clone(),
        }).unwrap_or_default();

        let score = self.apply_signal_pipeline(symbol, raw_score, close, hour_key, &cfg_snap);

        let short_score = if self.short_model.is_some() {
            self.predict_short(symbol, hour_key, &cfg_snap)
        } else {
            0.0
        };

        // 2. Build prediction dict
        let pred_dict = PyDict::new(py);
        pred_dict.set_item("ml_score", score)?;
        pred_dict.set_item("ml_short_score", short_score)?;
        pred_dict.set_item("raw_score", raw_score)?;

        // 3. Build features dict (skip NaN — no None values)
        let feat_dict = PyDict::new(py);
        for (i, &name) in FEATURE_NAMES.iter().enumerate() {
            let v = self.features_buf[i];
            if !v.is_nan() {
                feat_dict.set_item(name, v)?;
            }
        }

        Ok((pred_dict, feat_dict))
    }

    /// Get current position for a symbol.
    fn get_position(&self, symbol: &str) -> f64 {
        self.bridge_states.get(symbol)
            .map(|s| s.position)
            .unwrap_or(0.0)
    }

    /// Force-set position (for regime sync).
    fn set_position(&mut self, symbol: &str, position: f64, hold: i32) {
        if let Some(state) = self.bridge_states.get_mut(symbol) {
            state.position = position;
            state.hold_counter = hold;
        }
    }

    /// Checkpoint: serialize all state for persistence.
    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (sym, state) in &self.bridge_states {
            let d = PyDict::new(py);
            // Serialize z-score buffer
            let zb: Vec<f64> = state.zscore_buf.iter().copied().collect();
            d.set_item("zscore_buf", zb)?;
            d.set_item("zscore_last_hour", state.zscore_last_hour)?;
            d.set_item("position", state.position)?;
            d.set_item("hold_counter", state.hold_counter)?;
            let ch: Vec<f64> = state.close_history.iter().copied().collect();
            d.set_item("close_history", ch)?;
            d.set_item("gate_last_hour", state.gate_last_hour)?;
            // Short state
            let sb: Vec<f64> = state.short_zscore_buf.iter().copied().collect();
            d.set_item("short_zscore_buf", sb)?;
            d.set_item("short_zscore_last_hour", state.short_zscore_last_hour)?;
            d.set_item("short_position", state.short_position)?;
            d.set_item("short_hold_counter", state.short_hold_counter)?;
            result.set_item(sym.as_str(), d)?;
        }
        Ok(result)
    }

    /// Restore from checkpoint.
    fn restore(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        for item in data.iter() {
            let sym: String = item.0.extract()?;
            let d: &Bound<'_, PyDict> = item.1.downcast()?;

            let gate_w = self.configs.get(&sym)
                .map(|c| c.monthly_gate_window)
                .unwrap_or(480);
            let state = self.bridge_states.entry(sym)
                .or_insert_with(|| SymbolState::new(self.zscore_window, gate_w));

            if let Ok(zb) = d.get_item("zscore_buf") {
                if let Some(zb) = zb {
                    let buf: Vec<f64> = zb.extract()?;
                    state.zscore_buf.clear();
                    for v in buf { state.zscore_buf.push_back(v); }
                }
            }
            if let Ok(Some(v)) = d.get_item("zscore_last_hour") {
                state.zscore_last_hour = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("position") {
                state.position = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("hold_counter") {
                state.hold_counter = v.extract()?;
            }
            if let Ok(Some(ch)) = d.get_item("close_history") {
                let buf: Vec<f64> = ch.extract()?;
                state.close_history.clear();
                for v in buf { state.close_history.push_back(v); }
            }
            if let Ok(Some(v)) = d.get_item("gate_last_hour") {
                state.gate_last_hour = v.extract()?;
            }
            if let Ok(Some(sb)) = d.get_item("short_zscore_buf") {
                let buf: Vec<f64> = sb.extract()?;
                state.short_zscore_buf.clear();
                for v in buf { state.short_zscore_buf.push_back(v); }
            }
            if let Ok(Some(v)) = d.get_item("short_zscore_last_hour") {
                state.short_zscore_last_hour = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("short_position") {
                state.short_position = v.extract()?;
            }
            if let Ok(Some(v)) = d.get_item("short_hold_counter") {
                state.short_hold_counter = v.extract()?;
            }
        }
        Ok(())
    }
}
