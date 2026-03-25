// regime_detector.rs — Rust migration of Python regime detection
//
// Consolidates:
//   regime/volatility.py  — VolatilityRegimeDetector
//   regime/trend.py        — TrendRegimeDetector
//   regime/composite.py    — CompositeRegimeDetector
//   regime/param_router.py — RegimeParamRouter

use pyo3::prelude::*;
use std::collections::{HashMap, VecDeque};

// ── Vol/trend classification helpers ────────────────────────────────────────

fn classify_vol(
    vol: f64,
    vov: Option<f64>,
    vol_history: &VecDeque<f64>,
    vov_history: &VecDeque<f64>,
    min_bars: usize,
) -> Option<(String, f64)> {
    if vol_history.len() < min_bars {
        return None;
    }

    // Sort for percentiles
    let mut vol_sorted: Vec<f64> = vol_history.iter().copied().collect();
    vol_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vol_sorted.len();
    // Match Python: int(n * 0.25) == (n * 25) / 100 (integer truncation)
    let p25 = vol_sorted[(n * 25) / 100];
    let p75 = vol_sorted[(n * 75) / 100];
    let p95 = vol_sorted[(n * 95) / 100];

    // Crisis detection: vol_of_vol above 95th percentile
    let is_crisis = if let Some(v) = vov {
        if vov_history.len() >= min_bars {
            let mut vov_sorted: Vec<f64> = vov_history.iter().copied().collect();
            vov_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let vov_p95 = vov_sorted[(vov_sorted.len() * 95) / 100];
            v > vov_p95
        } else {
            false
        }
    } else {
        false
    };

    if is_crisis {
        Some(("crisis".to_string(), 1.0))
    } else if vol > p95 {
        Some(("high_vol".to_string(), 0.9))
    } else if vol > p75 {
        Some(("high_vol".to_string(), 0.7))
    } else if vol < p25 {
        Some(("low_vol".to_string(), 0.2))
    } else {
        let band = (p75 - p25).max(1e-12);
        let score = 0.3 + 0.4 * ((vol - p25) / band);
        Some(("normal_vol".to_string(), score))
    }
}

fn classify_trend(
    ma20: f64,
    ma50: f64,
    adx: f64,
    adx_strong: f64,
    adx_ranging: f64,
) -> (String, f64) {
    let up20 = ma20 > 0.0;
    let up50 = ma50 > 0.0;
    let agree = up20 == up50;

    if !agree || adx < adx_ranging {
        ("ranging".to_string(), 0.2)
    } else if up20 && up50 {
        if adx >= adx_strong {
            ("strong_up".to_string(), (adx / 50.0).min(1.0))
        } else {
            ("weak_up".to_string(), adx / 50.0)
        }
    } else {
        if adx >= adx_strong {
            ("strong_down".to_string(), (adx / 50.0).min(1.0))
        } else {
            ("weak_down".to_string(), adx / 50.0)
        }
    }
}

// ── RustRegimeResult ─────────────────────────────────────────────────────────

#[pyclass(name = "RustRegimeResult", frozen)]
pub struct RustRegimeResult {
    #[pyo3(get)]
    pub value: String, // "trend_label|vol_label"
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub vol_label: String,
    #[pyo3(get)]
    pub trend_label: String,
    #[pyo3(get)]
    pub is_favorable: bool, // "strong" in trend && vol in {low_vol, normal_vol}
    #[pyo3(get)]
    pub is_crisis: bool, // vol == "crisis"
}

// ── RustCompositeRegimeDetector ──────────────────────────────────────────────

#[pyclass(name = "RustCompositeRegimeDetector")]
pub struct RustCompositeRegimeDetector {
    vol_window: usize,   // 720
    vol_min_bars: usize, // 30
    vol_history: VecDeque<f64>,
    vov_history: VecDeque<f64>,
    adx_strong: f64,  // 25.0
    adx_ranging: f64, // 15.0
}

#[pymethods]
impl RustCompositeRegimeDetector {
    #[new]
    #[pyo3(signature = (vol_window=720, vol_min_bars=30, adx_strong=25.0, adx_ranging=15.0))]
    pub fn new(
        vol_window: usize,
        vol_min_bars: usize,
        adx_strong: f64,
        adx_ranging: f64,
    ) -> Self {
        Self {
            vol_window,
            vol_min_bars,
            vol_history: VecDeque::with_capacity(vol_window),
            vov_history: VecDeque::with_capacity(vol_window),
            adx_strong,
            adx_ranging,
        }
    }

    /// Detect regime from feature dict.
    /// Features: parkinson_vol, vol_of_vol, bb_width_20, close_vs_ma20, close_vs_ma50, adx_14
    /// Returns None if < min_bars or missing required features.
    pub fn detect(&mut self, features: HashMap<String, f64>) -> Option<RustRegimeResult> {
        // Extract features
        let parkinson = features.get("parkinson_vol").copied()?;
        if !parkinson.is_finite() {
            return None;
        }
        let vol = parkinson.abs();

        let vov = features
            .get("vol_of_vol")
            .copied()
            .filter(|v| v.is_finite())
            .map(|v| v.abs());

        // Push to histories (bounded)
        if self.vol_history.len() >= self.vol_window {
            self.vol_history.pop_front();
        }
        self.vol_history.push_back(vol);

        if let Some(v) = vov {
            if self.vov_history.len() >= self.vol_window {
                self.vov_history.pop_front();
            }
            self.vov_history.push_back(v);
        }

        if self.vol_history.len() < self.vol_min_bars {
            return None;
        }

        // Classify vol
        let (vol_label, vol_score) = classify_vol(
            vol,
            vov,
            &self.vol_history,
            &self.vov_history,
            self.vol_min_bars,
        )?;

        // Extract trend features (require both MAs)
        let ma20 = features
            .get("close_vs_ma20")
            .copied()
            .filter(|v| v.is_finite())?;
        let ma50 = features
            .get("close_vs_ma50")
            .copied()
            .filter(|v| v.is_finite())?;
        let adx = features
            .get("adx_14")
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);

        let (trend_label, trend_score) =
            classify_trend(ma20, ma50, adx, self.adx_strong, self.adx_ranging);

        let is_crisis = vol_label == "crisis";
        // Match Python: "strong" in self.trend and self.vol in ("low_vol", "normal_vol")
        let is_favorable = (vol_label == "low_vol" || vol_label == "normal_vol")
            && trend_label.contains("strong");

        let score = if is_crisis {
            0.8 * vol_score + 0.2 * trend_score
        } else {
            0.5 * vol_score + 0.5 * trend_score
        };

        let value = format!("{}|{}", trend_label, vol_label);

        Some(RustRegimeResult {
            value,
            score,
            vol_label,
            trend_label,
            is_favorable,
            is_crisis,
        })
    }
}

// ── RustRegimeParams ─────────────────────────────────────────────────────────

#[pyclass(name = "RustRegimeParams", frozen)]
pub struct RustRegimeParams {
    #[pyo3(get)]
    pub deadzone: f64,
    #[pyo3(get)]
    pub min_hold: i32,
    #[pyo3(get)]
    pub max_hold: i32,
    #[pyo3(get)]
    pub position_scale: f64,
}

// ── RustRegimeParamRouter ────────────────────────────────────────────────────

#[pyclass(name = "RustRegimeParamRouter")]
pub struct RustRegimeParamRouter {
    // (trend, vol) → (deadzone, min_hold, max_hold, position_scale)
    params: HashMap<(String, String), (f64, i32, i32, f64)>,
    fallback: (f64, i32, i32, f64),
}

#[pymethods]
impl RustRegimeParamRouter {
    #[new]
    pub fn new() -> Self {
        let mut params = HashMap::new();
        // Exact matches from DEFAULT_PARAMS in Python
        params.insert(
            ("strong_up".to_string(), "low_vol".to_string()),
            (0.3, 18, 60, 1.0),
        );
        params.insert(
            ("strong_up".to_string(), "normal_vol".to_string()),
            (0.5, 18, 60, 0.8),
        );
        params.insert(
            ("strong_down".to_string(), "low_vol".to_string()),
            (0.3, 18, 60, 1.0),
        );
        params.insert(
            ("strong_down".to_string(), "normal_vol".to_string()),
            (0.5, 18, 60, 0.8),
        );
        params.insert(
            ("weak_up".to_string(), "normal_vol".to_string()),
            (0.8, 24, 96, 0.6),
        );
        params.insert(
            ("weak_down".to_string(), "normal_vol".to_string()),
            (0.8, 24, 96, 0.6),
        );
        params.insert(
            ("weak_up".to_string(), "low_vol".to_string()),
            (0.5, 24, 96, 0.7),
        );
        params.insert(
            ("weak_down".to_string(), "low_vol".to_string()),
            (0.5, 24, 96, 0.7),
        );
        params.insert(
            ("ranging".to_string(), "low_vol".to_string()),
            (1.0, 24, 96, 0.5),
        );
        params.insert(
            ("ranging".to_string(), "normal_vol".to_string()),
            (1.2, 24, 96, 0.4),
        );
        params.insert(
            ("ranging".to_string(), "high_vol".to_string()),
            (1.5, 24, 48, 0.3),
        );
        // Wildcard entries using "*" as trend key
        params.insert(
            ("*".to_string(), "crisis".to_string()),
            (2.5, 48, 96, 0.1),
        );
        params.insert(
            ("*".to_string(), "high_vol".to_string()),
            (1.5, 24, 60, 0.3),
        );

        Self {
            params,
            fallback: (1.0, 24, 96, 0.5),
        }
    }

    /// Route (trend, vol) to params.
    /// Lookup order: exact → ("*", vol) wildcard → (trend, "*") wildcard → fallback
    pub fn route(&self, trend: &str, vol: &str) -> RustRegimeParams {
        // 1. Exact match
        let key = (trend.to_string(), vol.to_string());
        if let Some(&(d, min_h, max_h, ps)) = self.params.get(&key) {
            return RustRegimeParams {
                deadzone: d,
                min_hold: min_h,
                max_hold: max_h,
                position_scale: ps,
            };
        }
        // 2. Wildcard trend ("*", vol)
        let wild_trend = ("*".to_string(), vol.to_string());
        if let Some(&(d, min_h, max_h, ps)) = self.params.get(&wild_trend) {
            return RustRegimeParams {
                deadzone: d,
                min_hold: min_h,
                max_hold: max_h,
                position_scale: ps,
            };
        }
        // 3. Wildcard vol (trend, "*")
        let wild_vol = (trend.to_string(), "*".to_string());
        if let Some(&(d, min_h, max_h, ps)) = self.params.get(&wild_vol) {
            return RustRegimeParams {
                deadzone: d,
                min_hold: min_h,
                max_hold: max_h,
                position_scale: ps,
            };
        }
        // 4. Fallback
        let (d, min_h, max_h, ps) = self.fallback;
        RustRegimeParams {
            deadzone: d,
            min_hold: min_h,
            max_hold: max_h,
            position_scale: ps,
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────
include!("detector_tests.inc.rs");
