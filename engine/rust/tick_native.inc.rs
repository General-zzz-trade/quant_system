// tick_native.inc.rs — Native (Python-free) types and methods for standalone binary

/// Result of processing a tick without Python dependency.
pub struct NativeTickResult {
    pub advanced: bool,
    pub changed: bool,
    pub event_index: i64,
    pub ml_score: f64,
    pub ml_short_score: f64,
    pub raw_score: f64,
    pub last_event_id: Option<String>,
    pub last_ts: Option<String>,
    pub features_buf: [f64; N_FEATURES],
}

impl RustTickProcessor {
    /// Create a tick processor without PyO3 (for standalone binary).
    pub fn create_native(
        symbols: Vec<String>,
        currency: String,
        balance: f64,
        model_paths: Vec<String>,
        ensemble_weights: Option<Vec<f64>>,
        bear_model_path: Option<&str>,
        short_model_path: Option<&str>,
        zscore_window: usize,
        zscore_warmup: usize,
    ) -> Result<Self, String> {
        if model_paths.is_empty() {
            return Err("At least one model path required".to_string());
        }

        let mut main_models = Vec::with_capacity(model_paths.len());
        for path in &model_paths {
            let model = LoadedModel::load_from_path(path)?;
            main_models.push(model);
        }

        let weights = match ensemble_weights {
            Some(w) if w.len() == main_models.len() => w,
            _ => vec![1.0 / main_models.len() as f64; main_models.len()],
        };

        let bear_model = match bear_model_path {
            Some(p) => Some(LoadedModel::load_from_path(p)?),
            None => None,
        };

        let short_model = match short_model_path {
            Some(p) => Some(LoadedModel::load_from_path(p)?),
            None => None,
        };

        let balance_i64 = (balance * SCALE as f64) as i64;
        let mut markets_map = HashMap::new();
        let mut positions_map = HashMap::new();
        for sym in &symbols {
            markets_map.insert(sym.clone(), RustMarketState::empty(sym.clone()));
            positions_map.insert(sym.clone(), RustPositionState::empty(sym.clone()));
        }

        let account = RustAccountState {
            currency,
            balance: balance_i64,
            margin_used: 0,
            margin_available: 0,
            realized_pnl: 0,
            unrealized_pnl: 0,
            fees_paid: 0,
            last_ts: None,
        };

        let portfolio = RustPortfolioState {
            total_equity: "0".to_string(),
            cash_balance: "0".to_string(),
            realized_pnl: "0".to_string(),
            unrealized_pnl: "0".to_string(),
            fees_paid: "0".to_string(),
            gross_exposure: "0".to_string(),
            net_exposure: "0".to_string(),
            leverage: Some("0".to_string()),
            margin_used: "0".to_string(),
            margin_available: "0".to_string(),
            margin_ratio: None,
            symbols: vec![],
            last_ts: None,
        };

        let risk = RustRiskState {
            blocked: false,
            halted: false,
            level: None,
            message: None,
            flags: vec![],
            equity_peak: "0".to_string(),
            drawdown_pct: "0".to_string(),
            last_ts: None,
        };

        let risk_limits = RustRiskLimits {
            max_leverage: "5".to_string(),
            max_position_notional: None,
            max_drawdown_pct: "0.30".to_string(),
            block_on_equity_le_zero: true,
        };

        let mut tp = Self {
            engines: HashMap::new(),
            features_buf: [f64::NAN; N_FEATURES],
            model_buf: Vec::with_capacity(128),
            main_models,
            ensemble_weights: weights,
            bear_model,
            short_model,
            external_data: HashMap::new(),
            bridge_states: HashMap::new(),
            zscore_window,
            zscore_warmup,
            configs: HashMap::new(),
            markets: markets_map,
            positions: positions_map,
            account,
            portfolio,
            risk,
            risk_limits,
            event_index: 0,
            mr: RustMarketReducer,
            pr: RustPositionReducer,
            ar: RustAccountReducer,
            last_event_id: None,
            last_ts: None,
        };

        tp.refresh_derived();
        Ok(tp)
    }

    /// Process a tick without Python dependency. Returns NativeTickResult.
    pub fn process_tick_native(
        &mut self,
        symbol: &str,
        close: f64,
        volume: f64,
        high: f64,
        low: f64,
        open: f64,
        hour_key: i64,
        ts: Option<String>,
    ) -> NativeTickResult {
        // 1) Push bar to feature engine
        let ext = self.external_data.get(symbol).cloned().unwrap_or_default();
        let engine = self
            .engines
            .entry(symbol.to_string())
            .or_insert_with(BarState::new);

        engine.push(
            close, volume, high, low, open, ext.hour, ext.dow, ext.funding_rate, ext.trades,
            ext.taker_buy_volume, ext.quote_volume, ext.taker_buy_quote_volume,
            ext.open_interest, ext.ls_ratio, ext.spot_close, ext.fear_greed,
            ext.implied_vol, ext.put_call_ratio,
            ext.oc_flow_in, ext.oc_flow_out,
            ext.oc_supply, ext.oc_addr, ext.oc_tx, ext.oc_hashrate,
            ext.liq_total_vol, ext.liq_buy_vol, ext.liq_sell_vol, ext.liq_count,
            ext.mempool_fastest_fee, ext.mempool_economy_fee, ext.mempool_size,
            ext.macro_dxy, ext.macro_spx, ext.macro_vix, ext.macro_day,
            ext.social_volume, ext.sentiment_score,
        );

        // 2) Get features
        engine.get_features(&mut self.features_buf);

        // 3) ML prediction (ensemble)
        let raw_score = self.predict_ensemble();

        // 4) Signal pipeline
        let cfg_snap = self
            .configs
            .get(symbol)
            .map(|c| CfgSnapshot {
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
            })
            .unwrap_or_default();

        let ml_score = self.apply_signal_pipeline(symbol, raw_score, close, hour_key, &cfg_snap);

        let ml_short_score = if self.short_model.is_some() {
            self.predict_short(symbol, hour_key, &cfg_snap)
        } else {
            0.0
        };

        // 5) Update market state via reducer
        self.ensure_symbol(symbol);
        let market_event = RustMarketEvent {
            symbol: symbol.to_string(),
            open: Fd8::from_f64(open).raw(),
            high: Fd8::from_f64(high).raw(),
            low: Fd8::from_f64(low).raw(),
            close: Fd8::from_f64(close).raw(),
            volume: Fd8::from_f64(volume).raw(),
            ts: ts.clone(),
        };

        let market = self.markets.get(symbol).unwrap().clone();
        let m_res = self.mr.reduce_rust_market(&market, &market_event);
        self.markets.insert(symbol.to_string(), m_res.state);
        self.event_index += 1;
        if let Some(ref t) = ts {
            self.last_ts = Some(t.clone());
        }

        // 6) Refresh portfolio + risk
        if m_res.changed {
            self.refresh_derived();
        }

        NativeTickResult {
            advanced: true,
            changed: m_res.changed,
            event_index: self.event_index,
            ml_score,
            ml_short_score,
            raw_score,
            last_event_id: self.last_event_id.clone(),
            last_ts: self.last_ts.clone(),
            features_buf: self.features_buf,
        }
    }

    /// Get current market state for a symbol.
    pub fn get_market_native(&self, symbol: &str) -> Option<&RustMarketState> {
        self.markets.get(symbol)
    }

    /// Get current position state for a symbol.
    pub fn get_position_native(&self, symbol: &str) -> Option<&RustPositionState> {
        self.positions.get(symbol)
    }

    /// Get current account state.
    pub fn account_native(&self) -> &RustAccountState {
        &self.account
    }

    /// Get current portfolio state.
    pub fn portfolio_native(&self) -> &RustPortfolioState {
        &self.portfolio
    }

    /// Get current risk state.
    pub fn risk_native(&self) -> &RustRiskState {
        &self.risk
    }

    /// Configure per-symbol signal constraints (native variant).
    pub fn configure_symbol_native(
        &mut self,
        symbol: &str,
        min_hold: i32,
        deadzone: f64,
        long_only: bool,
        trend_follow: bool,
        trend_threshold: f64,
        trend_indicator: &str,
        max_hold: i32,
        monthly_gate: bool,
        monthly_gate_window: usize,
        vol_target: Option<f64>,
        vol_feature: &str,
        bear_thresholds: Vec<(f64, f64)>,
    ) {
        let trend_indicator_idx = FEATURE_NAMES
            .iter()
            .position(|&n| n == trend_indicator)
            .map(|i| i as u16)
            .unwrap_or(u16::MAX);

        let vol_feature_idx = FEATURE_NAMES
            .iter()
            .position(|&n| n == vol_feature)
            .map(|i| i as u16)
            .unwrap_or(u16::MAX);

        self.configs.insert(
            symbol.to_string(),
            SymbolConfig {
                min_hold,
                deadzone,
                long_only,
                trend_follow,
                trend_threshold,
                trend_indicator_idx,
                max_hold,
                monthly_gate,
                monthly_gate_window,
                vol_target,
                vol_feature_idx,
                bear_thresholds,
            },
        );

        if !self.engines.contains_key(symbol) {
            self.engines.insert(symbol.to_string(), BarState::new());
        }
        if !self.bridge_states.contains_key(symbol) {
            self.bridge_states.insert(
                symbol.to_string(),
                SymbolState::new(self.zscore_window, monthly_gate_window),
            );
        }
    }

    /// Push external data (native variant, same fields as PyO3 version).
    pub fn push_external_data_native(&mut self, symbol: &str, ext: ExternalData) {
        self.external_data.insert(symbol.to_string(), ext);
    }

    /// Serialize checkpoint to JSON string (bar history + signal state).
    pub fn checkpoint_native(&self) -> Result<String, String> {
        use serde_json::{json, Value};

        let mut symbols_data = serde_json::Map::new();

        for (sym, engine) in &self.engines {
            let bars = engine.get_bar_history();
            let bars_json: Vec<Value> = bars.iter().map(|b| {
                json!({
                    "c": b.close, "v": b.volume, "h": b.high,
                    "l": b.low, "o": b.open,
                    "hour": b.hour, "dow": b.dow,
                    "fr": b.funding_rate, "trades": b.trades,
                    "tbv": b.taker_buy_volume, "qv": b.quote_volume,
                    "tbqv": b.taker_buy_quote_volume,
                    "oi": b.open_interest, "ls": b.ls_ratio,
                    "spot": b.spot_close, "fg": b.fear_greed,
                    "iv": b.implied_vol, "pcr": b.put_call_ratio,
                    "oc_fi": b.oc_flow_in, "oc_fo": b.oc_flow_out,
                    "oc_s": b.oc_supply, "oc_a": b.oc_addr,
                    "oc_t": b.oc_tx, "oc_h": b.oc_hashrate,
                    "liq_tv": b.liq_total_vol, "liq_bv": b.liq_buy_vol,
                    "liq_sv": b.liq_sell_vol, "liq_c": b.liq_count,
                    "mp_ff": b.mempool_fastest_fee, "mp_ef": b.mempool_economy_fee,
                    "mp_s": b.mempool_size,
                    "m_dxy": b.macro_dxy, "m_spx": b.macro_spx,
                    "m_vix": b.macro_vix, "m_day": b.macro_day,
                    "sv": b.social_volume, "ss": b.sentiment_score,
                })
            }).collect();

            // Signal state
            let signal_state = self.bridge_states.get(sym).map(|s| {
                json!({
                    "zscore_buf": s.zscore_buf.iter().copied().collect::<Vec<_>>(),
                    "zscore_last_hour": s.zscore_last_hour,
                    "position": s.position,
                    "hold_counter": s.hold_counter,
                    "close_history": s.close_history.iter().copied().collect::<Vec<_>>(),
                    "gate_last_hour": s.gate_last_hour,
                    "short_zscore_buf": s.short_zscore_buf.iter().copied().collect::<Vec<_>>(),
                    "short_zscore_last_hour": s.short_zscore_last_hour,
                    "short_position": s.short_position,
                    "short_hold_counter": s.short_hold_counter,
                })
            });

            symbols_data.insert(sym.clone(), json!({
                "bars": bars_json,
                "signal": signal_state,
            }));
        }

        let checkpoint = json!({
            "version": 1,
            "symbols": symbols_data,
            "event_index": self.event_index,
        });

        serde_json::to_string(&checkpoint)
            .map_err(|e| format!("Checkpoint serialize error: {}", e))
    }

    /// Restore from checkpoint JSON. Replays stored bars silently.
    pub fn restore_checkpoint_native(&mut self, json: &str) -> Result<usize, String> {
        let checkpoint: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| format!("Checkpoint parse error: {}", e))?;

        let symbols = checkpoint.get("symbols")
            .and_then(|v| v.as_object())
            .ok_or("No 'symbols' in checkpoint")?;

        let mut total_replayed = 0usize;

        for (sym, data) in symbols {
            // Only restore symbols we're tracking
            if !self.configs.contains_key(sym) {
                continue;
            }

            // Restore signal state first
            if let Some(signal) = data.get("signal").filter(|v| !v.is_null()) {
                let gate_w = self.configs.get(sym)
                    .map(|c| c.monthly_gate_window)
                    .unwrap_or(480);
                let state = self.bridge_states
                    .entry(sym.clone())
                    .or_insert_with(|| SymbolState::new(self.zscore_window, gate_w));

                if let Some(arr) = signal.get("zscore_buf").and_then(|v| v.as_array()) {
                    state.zscore_buf.clear();
                    for v in arr { if let Some(f) = v.as_f64() { state.zscore_buf.push_back(f); } }
                }
                state.zscore_last_hour = signal.get("zscore_last_hour").and_then(|v| v.as_i64()).unwrap_or(-1);
                state.position = signal.get("position").and_then(|v| v.as_f64()).unwrap_or(0.0);
                state.hold_counter = signal.get("hold_counter").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                if let Some(arr) = signal.get("close_history").and_then(|v| v.as_array()) {
                    state.close_history.clear();
                    for v in arr { if let Some(f) = v.as_f64() { state.close_history.push_back(f); } }
                }
                state.gate_last_hour = signal.get("gate_last_hour").and_then(|v| v.as_i64()).unwrap_or(-1);
                if let Some(arr) = signal.get("short_zscore_buf").and_then(|v| v.as_array()) {
                    state.short_zscore_buf.clear();
                    for v in arr { if let Some(f) = v.as_f64() { state.short_zscore_buf.push_back(f); } }
                }
                state.short_zscore_last_hour = signal.get("short_zscore_last_hour").and_then(|v| v.as_i64()).unwrap_or(-1);
                state.short_position = signal.get("short_position").and_then(|v| v.as_f64()).unwrap_or(0.0);
                state.short_hold_counter = signal.get("short_hold_counter").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            }

            // Replay bars silently
            if let Some(bars) = data.get("bars").and_then(|v| v.as_array()) {
                for bar in bars {
                    let close = bar.get("c").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let volume = bar.get("v").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let high = bar.get("h").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let low = bar.get("l").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let open = bar.get("o").and_then(|v| v.as_f64()).unwrap_or(0.0);

                    if close == 0.0 { continue; }

                    let engine = self.engines
                        .entry(sym.clone())
                        .or_insert_with(BarState::new);

                    engine.push(
                        close, volume, high, low, open,
                        bar.get("hour").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
                        bar.get("dow").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
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

                    total_replayed += 1;
                }
            }
        }

        Ok(total_replayed)
    }
}
