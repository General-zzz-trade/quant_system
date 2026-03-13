// tick_processor.rs — Full hot-path: features + predict + state update + export in one Rust call
//
// Merges RustUnifiedPredictor + RustStateStore into a single struct,
// eliminating ~10 Python↔Rust boundary crossings per tick.
//
// Hot path: process_tick() ~80μs total vs ~1020μs with Python glue.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::feature_engine::{BarState, N_FEATURES, FEATURE_NAMES};
use crate::fixed_decimal::Fd8;
use crate::inference_bridge::SymbolState;
use crate::constraint_pipeline::{
    discretize, enforce_hold_step, enforce_short_hold_step,
    long_only_clip, discretize_short, update_monthly_gate, vol_scale,
};
use crate::rust_events::{RustFillEvent, RustFundingEvent, RustMarketEvent};
use crate::state_reducers::{RustAccountReducer, RustMarketReducer, RustPositionReducer};
use crate::state_store::{compute_portfolio_from, compute_risk_from, RustProcessResult};
use crate::state_types::{
    RustAccountState, RustMarketState, RustPortfolioState, RustPositionState, RustRiskLimits,
    RustRiskState,
};
use crate::unified_predictor::{CfgSnapshot, ExternalData, LoadedModel, SymbolConfig};

const SCALE: i64 = 100_000_000;

/// Result of processing a single market tick through the full pipeline.
#[pyclass(name = "RustTickResult", frozen)]
pub struct RustTickResult {
    #[pyo3(get)]
    pub advanced: bool,
    #[pyo3(get)]
    pub changed: bool,
    #[pyo3(get)]
    pub event_index: i64,
    #[pyo3(get)]
    pub ml_score: f64,
    #[pyo3(get)]
    pub ml_short_score: f64,
    #[pyo3(get)]
    pub raw_score: f64,
    #[pyo3(get)]
    pub last_event_id: Option<String>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
    // State exports (PyObject, built once at construction)
    #[pyo3(get)]
    pub markets: PyObject,
    #[pyo3(get)]
    pub positions: PyObject,
    #[pyo3(get)]
    pub account: PyObject,
    #[pyo3(get)]
    pub portfolio: PyObject,
    #[pyo3(get)]
    pub risk: PyObject,
    // Features buffer (not exposed directly)
    features_buf: [f64; N_FEATURES],
    // Pre-built features dict (set by process_tick_full)
    #[pyo3(get)]
    features_dict: Option<PyObject>,
}

#[pymethods]
impl RustTickResult {
    /// Export features as PyDict, skipping NaN values.
    fn get_features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (i, &name) in FEATURE_NAMES.iter().enumerate() {
            let v = self.features_buf[i];
            if !v.is_nan() {
                dict.set_item(name, v)?;
            }
        }
        Ok(dict)
    }
}

/// Unified tick processor: features + prediction + state update in one struct.
///
/// Merges RustUnifiedPredictor (feature engine + ML models + signal pipeline)
/// with RustStateStore (market/position/account/portfolio/risk state).
#[pyclass(name = "RustTickProcessor")]
pub struct RustTickProcessor {
    // ── Predictor side ──
    engines: HashMap<String, BarState>,
    features_buf: [f64; N_FEATURES],
    model_buf: Vec<f64>,
    main_models: Vec<LoadedModel>,
    ensemble_weights: Vec<f64>,
    bear_model: Option<LoadedModel>,
    short_model: Option<LoadedModel>,
    external_data: HashMap<String, ExternalData>,
    bridge_states: HashMap<String, SymbolState>,
    zscore_window: usize,
    zscore_warmup: usize,
    configs: HashMap<String, SymbolConfig>,

    // ── State store side ──
    markets: HashMap<String, RustMarketState>,
    positions: HashMap<String, RustPositionState>,
    account: RustAccountState,
    portfolio: RustPortfolioState,
    risk: RustRiskState,
    risk_limits: RustRiskLimits,
    event_index: i64,
    mr: RustMarketReducer,
    pr: RustPositionReducer,
    ar: RustAccountReducer,
    last_event_id: Option<String>,
    last_ts: Option<String>,
}

#[pymethods]
impl RustTickProcessor {
    /// Create a tick processor from model paths + state store config.
    #[staticmethod]
    #[pyo3(signature = (
        symbols,
        currency,
        balance,
        model_paths,
        ensemble_weights=None,
        bear_model_path=None,
        short_model_path=None,
        zscore_window=720,
        zscore_warmup=180,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create(
        symbols: Vec<String>,
        currency: String,
        balance: f64,
        model_paths: Vec<String>,
        ensemble_weights: Option<Vec<f64>>,
        bear_model_path: Option<&str>,
        short_model_path: Option<&str>,
        zscore_window: usize,
        zscore_warmup: usize,
    ) -> PyResult<Self> {
        if model_paths.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "At least one model path required",
            ));
        }

        // Load models
        let mut main_models = Vec::with_capacity(model_paths.len());
        for path in &model_paths {
            let model = LoadedModel::load_from_path(path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
            main_models.push(model);
        }

        let weights = match ensemble_weights {
            Some(w) if w.len() == main_models.len() => w,
            _ => vec![1.0 / main_models.len() as f64; main_models.len()],
        };

        let bear_model = match bear_model_path {
            Some(p) => Some(
                LoadedModel::load_from_path(p)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?,
            ),
            None => None,
        };

        let short_model = match short_model_path {
            Some(p) => Some(
                LoadedModel::load_from_path(p)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?,
            ),
            None => None,
        };

        // Initialize state store
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

    /// Full hot path with pre-built features dict.
    ///
    /// Combines process_tick + get_features + ml_score injection into one call.
    /// Eliminates Python-side dict operations (~35μs saved).
    ///
    /// Returns RustTickResult with features dict already containing:
    ///   close, volume, ml_score (if warmup_done), ml_short_score (if non-zero)
    #[pyo3(signature = (symbol, close, volume, high, low, open, hour_key, warmup_done=true, ts=None))]
    #[allow(clippy::too_many_arguments)]
    fn process_tick_full(
        &mut self,
        py: Python<'_>,
        symbol: &str,
        close: f64,
        volume: f64,
        high: f64,
        low: f64,
        open: f64,
        hour_key: i64,
        warmup_done: bool,
        ts: Option<String>,
    ) -> PyResult<RustTickResult> {
        let mut result = self.process_tick(py, symbol, close, volume, high, low, open, hour_key, ts)?;
        // Pre-build features dict inside Rust (avoids Python dict ops)
        let dict = PyDict::new(py);
        for (i, &name) in FEATURE_NAMES.iter().enumerate() {
            let v = result.features_buf[i];
            if !v.is_nan() {
                dict.set_item(name, v)?;
            }
        }
        dict.set_item("close", close)?;
        dict.set_item("volume", volume)?;
        if warmup_done {
            dict.set_item("ml_score", result.ml_score)?;
            if result.ml_short_score != 0.0 {
                dict.set_item("ml_short_score", result.ml_short_score)?;
            }
        }
        result.features_dict = Some(dict.into_any().unbind());
        Ok(result)
    }

    /// Core hot path: push bar → compute features → predict → update state → export.
    ///
    /// Returns RustTickResult with ML scores + exported state.
    /// Single Python→Rust call replaces ~10 separate calls.
    #[pyo3(signature = (symbol, close, volume, high, low, open, hour_key, ts=None))]
    #[allow(clippy::too_many_arguments)]
    fn process_tick(
        &mut self,
        py: Python<'_>,
        symbol: &str,
        close: f64,
        volume: f64,
        high: f64,
        low: f64,
        open: f64,
        hour_key: i64,
        ts: Option<String>,
    ) -> PyResult<RustTickResult> {
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
        let close_i64 = Fd8::from_f64(close).raw();
        let market_event = RustMarketEvent {
            symbol: symbol.to_string(),
            open: Fd8::from_f64(open).raw(),
            high: Fd8::from_f64(high).raw(),
            low: Fd8::from_f64(low).raw(),
            close: close_i64,
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

        // 7) Export state to Python objects
        let markets_py = {
            let dict = PyDict::new(py);
            for (sym, state) in &self.markets {
                dict.set_item(sym, state.clone().into_pyobject(py)?)?;
            }
            dict.into_any().unbind()
        };
        let positions_py = {
            let dict = PyDict::new(py);
            for (sym, state) in &self.positions {
                dict.set_item(sym, state.clone().into_pyobject(py)?)?;
            }
            dict.into_any().unbind()
        };
        let account_py = self.account.clone().into_pyobject(py)?.into_any().unbind();
        let portfolio_py = self.portfolio.clone().into_pyobject(py)?.into_any().unbind();
        let risk_py = self.risk.clone().into_pyobject(py)?.into_any().unbind();

        Ok(RustTickResult {
            advanced: true,
            changed: m_res.changed,
            event_index: self.event_index,
            ml_score,
            ml_short_score,
            raw_score,
            last_event_id: self.last_event_id.clone(),
            last_ts: self.last_ts.clone(),
            markets: markets_py,
            positions: positions_py,
            account: account_py,
            portfolio: portfolio_py,
            risk: risk_py,
            features_buf: self.features_buf,
            features_dict: None,
        })
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

        let position = self.positions.get(&sym).unwrap().clone();
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

// ── Internal methods (non-PyO3) ──

impl RustTickProcessor {
    fn ensure_symbol(&mut self, sym: &str) {
        if !self.markets.contains_key(sym) {
            self.markets
                .insert(sym.to_string(), RustMarketState::empty(sym.to_string()));
        }
        if !self.positions.contains_key(sym) {
            self.positions
                .insert(sym.to_string(), RustPositionState::empty(sym.to_string()));
        }
    }

    fn refresh_derived(&mut self) {
        self.portfolio = compute_portfolio_from(
            &self.markets,
            &self.positions,
            &self.account,
            &self.last_ts,
        );
        self.risk = compute_risk_from(
            &self.portfolio,
            &self.risk_limits,
            &self.positions,
            &self.risk,
            &self.last_ts,
        );
    }

    #[inline]
    fn predict_ensemble(&mut self) -> f64 {
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
        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    fn apply_signal_pipeline(
        &mut self,
        symbol: &str,
        raw_score: f64,
        close: f64,
        hour_key: i64,
        cfg: &CfgSnapshot,
    ) -> f64 {
        use crate::inference_bridge::zscore_from_buf;

        if cfg.min_hold <= 0 {
            return raw_score;
        }

        let gate_w = cfg.monthly_gate_window;
        let state = self
            .bridge_states
            .entry(symbol.to_string())
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
                &mut state.zscore_buf,
                &mut state.zscore_last_hour,
                raw_score,
                hour_key,
                self.zscore_window,
                self.zscore_warmup,
            ) {
                Some(z) => z,
                None => {
                    // Warmup: increment hold counter to match backtest behavior.
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
        let hold_count = if state.hold_counter == 0 {
            cfg.min_hold
        } else {
            state.hold_counter
        };

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

        // Bear regime handling
        if !gate_ok {
            if self.bear_model.is_some() {
                let bear_score = self
                    .bear_model
                    .as_ref()
                    .unwrap()
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

    fn predict_short(&mut self, symbol: &str, hour_key: i64, cfg: &CfgSnapshot) -> f64 {
        use crate::inference_bridge::zscore_from_buf;

        let short_model = match &self.short_model {
            Some(m) => m,
            None => return 0.0,
        };

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
        let state = self
            .bridge_states
            .entry(symbol.to_string())
            .or_insert_with(|| SymbolState::new(self.zscore_window, gate_w));

        let z = match zscore_from_buf(
            &mut state.short_zscore_buf,
            &mut state.short_zscore_last_hour,
            raw,
            hour_key,
            self.zscore_window,
            self.zscore_warmup,
        ) {
            Some(z) => z,
            None => return state.short_position,
        };

        // Short-only discretize (shared)
        let desired = discretize_short(z, cfg.deadzone);

        // Min-hold enforcement (shared)
        let prev = state.short_position;
        let hold = if state.short_hold_counter == 0 {
            cfg.min_hold
        } else {
            state.short_hold_counter
        };

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

// ── Native (Python-free) types and methods for standalone binary ──

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

    /// Phase 5: Serialize checkpoint to JSON string (bar history + signal state).
    ///
    /// Stores the last 720 raw bars per symbol plus bridge signal state,
    /// enabling warm restart without 3-hour warmup delay.
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

    /// Phase 5: Restore from checkpoint JSON. Replays stored bars silently.
    ///
    /// Returns the number of bars replayed.
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
