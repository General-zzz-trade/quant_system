// tick_pymethods_state.inc.rs — PyO3 state access, export/load, checkpoint/restore methods
//
// Included by tick_processor.rs via include!() macro.
// Separated from the main #[pymethods] block to keep each file under 500 lines.

#[pymethods]
impl RustTickProcessor {
    /// Configure per-symbol signal constraints (same API as RustUnifiedPredictor).
    #[pyo3(signature = (
        symbol,
        min_hold=0, deadzone=0.5, long_only=false,
        trend_follow=false, trend_threshold=0.0,
        trend_indicator="tf4h_close_vs_ma20",
        max_hold=120,
        monthly_gate=false, monthly_gate_window=480,
        vol_target=None, vol_feature="atr_norm_14",
        bear_thresholds=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn configure_symbol(
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
        bear_thresholds: Option<Vec<(f64, f64)>>,
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
                bear_thresholds: bear_thresholds.unwrap_or_default(),
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

    /// Update cached external data for a symbol.
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
        hour: i32,
        dow: i32,
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
        oc_flow_in: f64,
        oc_flow_out: f64,
        oc_supply: f64,
        oc_addr: f64,
        oc_tx: f64,
        oc_hashrate: f64,
        liq_total_vol: f64,
        liq_buy_vol: f64,
        liq_sell_vol: f64,
        liq_count: f64,
        mempool_fastest_fee: f64,
        mempool_economy_fee: f64,
        mempool_size: f64,
        macro_dxy: f64,
        macro_spx: f64,
        macro_vix: f64,
        macro_day: i64,
        social_volume: f64,
        sentiment_score: f64,
    ) {
        let ext = self
            .external_data
            .entry(symbol.to_string())
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

    /// Process a fill event (position + account state update).
    fn process_fill(&mut self, py: Python<'_>, event: &Bound<'_, PyAny>) -> PyResult<RustProcessResult> {
        if let Ok(fe) = event.downcast::<RustFillEvent>() {
            let fe = fe.borrow();
            let sym = &fe.symbol;
            self.ensure_symbol(sym);

            let position = self.positions.get(sym).unwrap().clone();
            let p_res = self.pr.reduce_rust_fill(&position, &fe)?;
            let a_res = self.ar.reduce_rust_fill(&self.account, &fe);

            let changed = p_res.changed || a_res.changed;
            self.positions.insert(sym.clone(), p_res.state);
            self.account = a_res.state;
            self.event_index += 1;
            if let Some(ref ts) = fe.ts {
                self.last_ts = Some(ts.clone());
            }
            if changed {
                self.refresh_derived();
            }

            return Ok(RustProcessResult {
                advanced: true,
                changed,
                event_index: self.event_index,
                kind: "FILL".to_string(),
            });
        }

        // Slow path: Python event
        let sym = event
            .getattr("symbol")
            .ok()
            .and_then(|s| s.extract::<String>().ok())
            .unwrap_or_default();
        self.ensure_symbol(&sym);

        let position = self.positions.get(&sym)
            .expect("ensure_symbol must insert before get")
            .clone();
        let p_res = self.pr.reduce_inner(py, &position, event)?;
        let a_res = self.ar.reduce_inner(py, &self.account, event)?;

        let changed = p_res.changed || a_res.changed;
        self.positions.insert(sym, p_res.state);
        self.account = a_res.state;
        self.event_index += 1;
        if changed {
            self.refresh_derived();
        }

        Ok(RustProcessResult {
            advanced: true,
            changed,
            event_index: self.event_index,
            kind: "FILL".to_string(),
        })
    }

    /// Process a funding event (account state update).
    fn process_funding(
        &mut self,
        _py: Python<'_>,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustProcessResult> {
        if let Ok(fe) = event.downcast::<RustFundingEvent>() {
            let fe = fe.borrow();
            let a_res = self.ar.reduce_rust_funding(&self.account, &fe);

            let changed = a_res.changed;
            self.account = a_res.state;
            self.event_index += 1;
            if let Some(ref ts) = fe.ts {
                self.last_ts = Some(ts.clone());
            }
            if changed {
                self.refresh_derived();
            }

            return Ok(RustProcessResult {
                advanced: true,
                changed,
                event_index: self.event_index,
                kind: "FUNDING".to_string(),
            });
        }

        // Funding events should always be RustFundingEvent on the fast path
        Ok(RustProcessResult {
            advanced: false,
            changed: false,
            event_index: self.event_index,
            kind: "FUNDING".to_string(),
        })
    }

    // ── State access ──

    fn get_markets(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (sym, state) in &self.markets {
            dict.set_item(sym, state.clone().into_pyobject(py)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    fn get_positions(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (sym, state) in &self.positions {
            dict.set_item(sym, state.clone().into_pyobject(py)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    fn get_account(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.account.clone().into_pyobject(py)?.into_any().unbind())
    }

    fn get_portfolio(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.portfolio.clone().into_pyobject(py)?.into_any().unbind())
    }

    fn get_risk(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.risk.clone().into_pyobject(py)?.into_any().unbind())
    }

    fn export_state(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("markets", self.get_markets(py)?)?;
        out.set_item("positions", self.get_positions(py)?)?;
        out.set_item("account", self.account.clone().into_pyobject(py)?)?;
        out.set_item("portfolio", self.portfolio.clone().into_pyobject(py)?)?;
        out.set_item("risk", self.risk.clone().into_pyobject(py)?)?;
        out.set_item("event_index", self.event_index)?;
        out.set_item("last_event_id", self.last_event_id.clone())?;
        out.set_item("last_ts", self.last_ts.clone())?;
        Ok(out.into_any().unbind())
    }

    #[pyo3(signature = (markets, positions, account, *, event_index, last_event_id=None, last_ts=None, portfolio=None, risk=None))]
    fn load_exported(
        &mut self,
        markets: &Bound<'_, PyDict>,
        positions: &Bound<'_, PyDict>,
        account: &RustAccountState,
        event_index: i64,
        last_event_id: Option<String>,
        last_ts: Option<String>,
        portfolio: Option<&Bound<'_, PyAny>>,
        risk: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut next_markets = HashMap::new();
        let mut next_positions = HashMap::new();
        for (key, val) in markets.iter() {
            let sym: String = key.extract()?;
            let state: RustMarketState = val.extract()?;
            next_markets.insert(sym, state);
        }
        for (key, val) in positions.iter() {
            let sym: String = key.extract()?;
            let state: RustPositionState = val.extract()?;
            next_positions.insert(sym, state);
        }
        self.markets = next_markets;
        self.positions = next_positions;
        self.account = account.clone();
        self.event_index = event_index;
        self.last_event_id = last_event_id;
        self.last_ts = last_ts;
        self.portfolio = match portfolio {
            Some(obj) if !obj.is_none() => obj.extract::<RustPortfolioState>()?,
            _ => compute_portfolio_from(
                &self.markets,
                &self.positions,
                &self.account,
                &self.last_ts,
            ),
        };
        self.risk = match risk {
            Some(obj) if !obj.is_none() => obj.extract::<RustRiskState>()?,
            _ => compute_risk_from(
                &self.portfolio,
                &self.risk_limits,
                &self.positions,
                &self.risk,
                &self.last_ts,
            ),
        };
        Ok(())
    }

    #[getter]
    fn event_index(&self) -> i64 {
        self.event_index
    }

    #[getter]
    fn last_event_id(&self) -> Option<String> {
        self.last_event_id.clone()
    }

    #[getter]
    fn last_ts_prop(&self) -> Option<String> {
        self.last_ts.clone()
    }

    /// Checkpoint: serialize signal state for persistence.
    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (sym, state) in &self.bridge_states {
            let d = PyDict::new(py);
            let zb: Vec<f64> = state.zscore_buf.iter().copied().collect();
            d.set_item("zscore_buf", zb)?;
            d.set_item("zscore_last_hour", state.zscore_last_hour)?;
            d.set_item("position", state.position)?;
            d.set_item("hold_counter", state.hold_counter)?;
            let ch: Vec<f64> = state.close_history.iter().copied().collect();
            d.set_item("close_history", ch)?;
            d.set_item("gate_last_hour", state.gate_last_hour)?;
            let sb: Vec<f64> = state.short_zscore_buf.iter().copied().collect();
            d.set_item("short_zscore_buf", sb)?;
            d.set_item("short_zscore_last_hour", state.short_zscore_last_hour)?;
            d.set_item("short_position", state.short_position)?;
            d.set_item("short_hold_counter", state.short_hold_counter)?;
            result.set_item(sym.as_str(), d)?;
        }
        Ok(result)
    }

    /// Restore signal state from checkpoint.
    fn restore(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        for item in data.iter() {
            let sym: String = item.0.extract()?;
            let d: &Bound<'_, PyDict> = item.1.downcast()?;

            let gate_w = self
                .configs
                .get(&sym)
                .map(|c| c.monthly_gate_window)
                .unwrap_or(480);
            let state = self
                .bridge_states
                .entry(sym)
                .or_insert_with(|| SymbolState::new(self.zscore_window, gate_w));

            if let Ok(Some(zb)) = d.get_item("zscore_buf") {
                let buf: Vec<f64> = zb.extract()?;
                state.zscore_buf.clear();
                for v in buf {
                    state.zscore_buf.push_back(v);
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
                for v in buf {
                    state.close_history.push_back(v);
                }
            }
            if let Ok(Some(v)) = d.get_item("gate_last_hour") {
                state.gate_last_hour = v.extract()?;
            }
            if let Ok(Some(sb)) = d.get_item("short_zscore_buf") {
                let buf: Vec<f64> = sb.extract()?;
                state.short_zscore_buf.clear();
                for v in buf {
                    state.short_zscore_buf.push_back(v);
                }
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
