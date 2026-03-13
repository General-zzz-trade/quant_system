// unified_predictor.rs — Zero-copy feature→predict→signal pipeline
//
// Merges BarState (features) + Tree prediction + signal constraints
// into a single Rust call, eliminating Python dict construction and
// per-feature extraction overhead on the hot path.
//
// Hot path: push_bar_and_predict() ~50μs total vs ~885μs with Python glue.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::feature_engine::{BarState, N_FEATURES, FEATURE_NAMES};
use crate::tree_predict::{Tree, Node, NodeJson, TreeJson, ModelJson};
use crate::inference_bridge::{SymbolState, zscore_from_buf};
use crate::constraint_pipeline::{
    discretize, enforce_hold_step, enforce_short_hold_step,
    long_only_clip, discretize_short, update_monthly_gate, vol_scale,
};

// ── Internal model representation ──

pub(crate) struct LoadedModel {
    pub(crate) trees: Vec<Tree>,
    pub(crate) is_classifier: bool,
    pub(crate) base_score: f64,
    pub(crate) format: String,
    pub(crate) feature_map: Vec<u16>,
    pub(crate) num_model_features: usize,
}

impl LoadedModel {
    pub(crate) fn load_from_path(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Cannot read {}: {}", path, e))?;
        Self::load_from_json(&data)
    }

    fn load_from_json(json_str: &str) -> Result<Self, String> {
        let model_json: ModelJson = serde_json::from_str(json_str)
            .map_err(|e| format!("Invalid JSON: {}", e))?;

        let trees: Vec<Tree> = model_json.trees.iter().map(|tj| {
            let nodes: Vec<Node> = tj.nodes.iter().map(|nj| match nj {
                NodeJson::Split { feature, threshold, default_left, nan_as_zero, left, right } => {
                    Node::Split {
                        feature: *feature as u16,
                        threshold: *threshold,
                        default_left: *default_left,
                        nan_as_zero: *nan_as_zero,
                        left: *left as u32,
                        right: *right as u32,
                    }
                }
                NodeJson::Leaf { value } => Node::Leaf(*value),
            }).collect();
            Tree { shrinkage: tj.shrinkage, nodes: nodes }
        }).collect();

        // Build feature index map: model feature name → FEATURE_NAMES index
        let feature_map: Vec<u16> = model_json.features.iter().map(|name| {
            FEATURE_NAMES.iter().position(|&n| n == name.as_str())
                .map(|i| i as u16)
                .unwrap_or(u16::MAX)
        }).collect();

        Ok(Self {
            trees,
            is_classifier: model_json.is_classifier,
            base_score: model_json.base_score,
            format: model_json.format,
            feature_map,
            num_model_features: model_json.num_features,
        })
    }

    #[inline]
    pub(crate) fn predict(&self, engine_features: &[f64; N_FEATURES], model_buf: &mut Vec<f64>) -> f64 {
        // Remap engine features to model feature order (zero-copy from engine)
        model_buf.resize(self.num_model_features, f64::NAN);
        for (i, &idx) in self.feature_map.iter().enumerate() {
            model_buf[i] = if idx < N_FEATURES as u16 {
                engine_features[idx as usize]
            } else {
                f64::NAN
            };
        }

        // Sum tree predictions
        let mut sum = if self.format == "xgb" { self.base_score } else { 0.0 };
        for tree in &self.trees {
            sum += tree.predict(model_buf);
        }

        if self.is_classifier {
            let prob = 1.0 / (1.0 + (-sum).exp());
            prob - 0.5
        } else {
            sum
        }
    }
}

// ── Per-symbol cached external data ──

#[derive(Clone)]
pub struct ExternalData {
    pub(crate) hour: i32,
    pub(crate) dow: i32,
    pub(crate) funding_rate: f64,
    pub(crate) trades: f64,
    pub(crate) taker_buy_volume: f64,
    pub(crate) quote_volume: f64,
    pub(crate) taker_buy_quote_volume: f64,
    pub(crate) open_interest: f64,
    pub(crate) ls_ratio: f64,
    pub(crate) spot_close: f64,
    pub(crate) fear_greed: f64,
    pub(crate) implied_vol: f64,
    pub(crate) put_call_ratio: f64,
    pub(crate) oc_flow_in: f64, pub(crate) oc_flow_out: f64,
    pub(crate) oc_supply: f64, pub(crate) oc_addr: f64,
    pub(crate) oc_tx: f64, pub(crate) oc_hashrate: f64,
    pub(crate) liq_total_vol: f64, pub(crate) liq_buy_vol: f64, pub(crate) liq_sell_vol: f64,
    pub(crate) liq_count: f64,
    pub(crate) mempool_fastest_fee: f64, pub(crate) mempool_economy_fee: f64,
    pub(crate) mempool_size: f64,
    pub(crate) macro_dxy: f64, pub(crate) macro_spx: f64, pub(crate) macro_vix: f64,
    pub(crate) macro_day: i64,
    pub(crate) social_volume: f64, pub(crate) sentiment_score: f64,
}

impl Default for ExternalData {
    fn default() -> Self {
        Self {
            hour: -1, dow: -1,
            funding_rate: f64::NAN,
            trades: 0.0, taker_buy_volume: 0.0,
            quote_volume: 0.0, taker_buy_quote_volume: 0.0,
            open_interest: f64::NAN, ls_ratio: f64::NAN,
            spot_close: f64::NAN, fear_greed: f64::NAN,
            implied_vol: f64::NAN, put_call_ratio: f64::NAN,
            oc_flow_in: f64::NAN, oc_flow_out: f64::NAN,
            oc_supply: f64::NAN, oc_addr: f64::NAN,
            oc_tx: f64::NAN, oc_hashrate: f64::NAN,
            liq_total_vol: f64::NAN, liq_buy_vol: f64::NAN, liq_sell_vol: f64::NAN,
            liq_count: f64::NAN,
            mempool_fastest_fee: f64::NAN, mempool_economy_fee: f64::NAN,
            mempool_size: f64::NAN,
            macro_dxy: f64::NAN, macro_spx: f64::NAN, macro_vix: f64::NAN,
            macro_day: -1,
            social_volume: f64::NAN, sentiment_score: f64::NAN,
        }
    }
}

// ── Per-symbol config ──

#[derive(Clone)]
pub(crate) struct SymbolConfig {
    pub(crate) min_hold: i32,
    pub(crate) deadzone: f64,
    pub(crate) long_only: bool,
    pub(crate) trend_follow: bool,
    pub(crate) trend_threshold: f64,
    pub(crate) trend_indicator_idx: u16,
    pub(crate) max_hold: i32,
    pub(crate) monthly_gate: bool,
    pub(crate) monthly_gate_window: usize,
    pub(crate) vol_target: Option<f64>,
    pub(crate) vol_feature_idx: u16,
    pub(crate) bear_thresholds: Vec<(f64, f64)>,
}

// ── Config snapshot (owned, avoids borrow conflict) ──

pub(crate) struct CfgSnapshot {
    pub(crate) min_hold: i32,
    pub(crate) deadzone: f64,
    pub(crate) long_only: bool,
    pub(crate) trend_follow: bool,
    pub(crate) trend_threshold: f64,
    pub(crate) trend_indicator_idx: u16,
    pub(crate) max_hold: i32,
    pub(crate) monthly_gate: bool,
    pub(crate) monthly_gate_window: usize,
    pub(crate) vol_target: Option<f64>,
    pub(crate) vol_feature_idx: u16,
    pub(crate) bear_thresholds: Vec<(f64, f64)>,
}

impl Default for CfgSnapshot {
    fn default() -> Self {
        Self {
            min_hold: 0, deadzone: 0.5, long_only: false,
            trend_follow: false, trend_threshold: 0.0,
            trend_indicator_idx: u16::MAX, max_hold: 120,
            monthly_gate: false, monthly_gate_window: 480,
            vol_target: None, vol_feature_idx: u16::MAX,
            bear_thresholds: Vec::new(),
        }
    }
}

// ── Main struct ──

#[pyclass]
pub struct RustUnifiedPredictor {
    pub(crate) engines: HashMap<String, BarState>,
    pub(crate) features_buf: [f64; N_FEATURES],
    pub(crate) model_buf: Vec<f64>,
    pub(crate) main_models: Vec<LoadedModel>,
    pub(crate) ensemble_weights: Vec<f64>,
    pub(crate) bear_model: Option<LoadedModel>,
    pub(crate) short_model: Option<LoadedModel>,
    pub(crate) external_data: HashMap<String, ExternalData>,
    pub(crate) bridge_states: HashMap<String, SymbolState>,
    pub(crate) zscore_window: usize,
    pub(crate) zscore_warmup: usize,
    pub(crate) configs: HashMap<String, SymbolConfig>,
}

#[pymethods]
impl RustUnifiedPredictor {
    /// Create a unified predictor from model paths and config.
    ///
    /// model_paths: list of JSON model file paths (ensemble if >1)
    /// ensemble_weights: optional weights for ensemble (equal if not provided)
    #[staticmethod]
    #[pyo3(signature = (
        model_paths,
        ensemble_weights=None,
        bear_model_path=None,
        short_model_path=None,
        zscore_window=720,
        zscore_warmup=180,
    ))]
    fn create(
        model_paths: Vec<String>,
        ensemble_weights: Option<Vec<f64>>,
        bear_model_path: Option<&str>,
        short_model_path: Option<&str>,
        zscore_window: usize,
        zscore_warmup: usize,
    ) -> PyResult<Self> {
        if model_paths.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("At least one model path required"));
        }

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
            Some(p) => Some(LoadedModel::load_from_path(p)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?),
            None => None,
        };

        let short_model = match short_model_path {
            Some(p) => Some(LoadedModel::load_from_path(p)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?),
            None => None,
        };

        Ok(Self {
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
        })
    }

    /// Configure per-symbol signal constraints.
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
        let trend_indicator_idx = FEATURE_NAMES.iter()
            .position(|&n| n == trend_indicator)
            .map(|i| i as u16)
            .unwrap_or(u16::MAX);

        let vol_feature_idx = FEATURE_NAMES.iter()
            .position(|&n| n == vol_feature)
            .map(|i| i as u16)
            .unwrap_or(u16::MAX);

        self.configs.insert(symbol.to_string(), SymbolConfig {
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
        });

        // Ensure engine + bridge state exist
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

        // 4) ML prediction (main model)
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

// ── Internal methods (non-PyO3) ──

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
