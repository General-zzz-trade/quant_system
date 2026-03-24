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

use crate::features::engine::{BarState, N_FEATURES, FEATURE_NAMES};
use crate::decision::ml::tree_predict::{Tree, Node, NodeJson, TreeJson, ModelJson};
use crate::decision::inference_bridge::{SymbolState, zscore_from_buf};
use crate::decision::constraint_pipeline::{
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

}

// ── PyO3 push/predict/checkpoint methods — see unified_predictor_pymethods.inc.rs ──

include!("unified_predictor_pymethods.inc.rs");

// ── Internal methods (non-PyO3) — see unified_predictor_signal.inc.rs ──

include!("unified_predictor_signal.inc.rs");
