// tick_processor.rs — Full hot-path: features + predict + state update + export in one Rust call
//
// Merges RustUnifiedPredictor + RustStateStore into a single struct,
// eliminating ~10 Python↔Rust boundary crossings per tick.
//
// Hot path: process_tick() ~80μs total vs ~1020μs with Python glue.
//
// Split into multiple files for maintainability (all <500 lines):
//   tick_result.rs                — RustTickResult PyO3 output type
//   tick_processor.rs             — struct + create/configure/process_tick (this file)
//   tick_pymethods_state.inc.rs   — PyO3 state access, fill/funding, checkpoint/restore
//   tick_signal.inc.rs            — internal signal pipeline methods
//   tick_native.inc.rs            — native (Python-free) methods

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::features::engine::{BarState, N_FEATURES, FEATURE_NAMES};
use crate::state::fixed_decimal::Fd8;
use crate::decision::inference_bridge::SymbolState;
use crate::decision::constraint_pipeline::{
    discretize, enforce_hold_step, enforce_short_hold_step,
    long_only_clip, discretize_short, update_monthly_gate, vol_scale,
};
use crate::event::data_events::{RustFillEvent, RustFundingEvent, RustMarketEvent};
use crate::state::reducers::{RustAccountReducer, RustMarketReducer, RustPositionReducer};
use crate::state::store::RustProcessResult;
use crate::state::store_compute::{compute_portfolio_from, compute_risk_from};
use crate::state::types::{
    RustAccountState, RustMarketState, RustPortfolioState, RustPositionState, RustRiskLimits,
    RustRiskState,
};
use crate::decision::ml::unified_predictor::{CfgSnapshot, ExternalData, LoadedModel, SymbolConfig};

// Re-export RustTickResult from sibling module
pub use crate::engine::tick_result::RustTickResult;

const SCALE: i64 = 100_000_000;

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

    /// Core hot path: push bar -> compute features -> predict -> update state -> export.
    ///
    /// Returns RustTickResult with ML scores + exported state.
    /// Single Python->Rust call replaces ~10 separate calls.
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
}

// ── PyO3 state access, fill/funding, checkpoint/restore ──
include!("tick_pymethods_state.inc.rs");

// ── Internal signal pipeline methods ──
include!("tick_signal.inc.rs");

// ── Native (Python-free) types and methods for standalone binary ──
include!("tick_native.inc.rs");
