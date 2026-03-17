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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(parkinson: f64, ma20: f64, ma50: f64, adx: f64) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("parkinson_vol".to_string(), parkinson);
        m.insert("close_vs_ma20".to_string(), ma20);
        m.insert("close_vs_ma50".to_string(), ma50);
        m.insert("adx_14".to_string(), adx);
        m
    }

    fn make_features_with_vov(
        parkinson: f64,
        vov: f64,
        ma20: f64,
        ma50: f64,
        adx: f64,
    ) -> HashMap<String, f64> {
        let mut m = make_features(parkinson, ma20, ma50, adx);
        m.insert("vol_of_vol".to_string(), vov);
        m
    }

    #[test]
    fn test_insufficient_bars_returns_none() {
        let mut det = RustCompositeRegimeDetector::new(720, 30, 25.0, 15.0);
        let features = make_features(0.01, 0.01, 0.01, 20.0);
        // Only push 5 bars (< 30 min_bars)
        for _ in 0..5 {
            det.detect(features.clone());
        }
        assert!(det.detect(features).is_none());
    }

    #[test]
    fn test_vol_low() {
        // Fill with 40 bars of identical low vol → should be "low_vol"
        // (all values equal → p25 == p75 == p95 == same value → vol < p25 is false,
        //  but vol > p95 is false too. With all same, everything falls into normal_vol band.
        // To get low_vol we need vol strictly below p25 — use mixed values.)
        let mut det = RustCompositeRegimeDetector::new(720, 30, 25.0, 15.0);
        // Push 35 bars of high vol first, then test with very low
        let high_features = make_features(0.1, 0.01, 0.01, 20.0);
        let low_features = make_features(0.001, 0.01, 0.01, 20.0);
        for _ in 0..35 {
            det.detect(high_features.clone());
        }
        // Now push a low vol — it should be below p25
        let result = det.detect(low_features);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.vol_label, "low_vol");
    }

    #[test]
    fn test_vol_crisis() {
        // Crisis: vol_of_vol above its own p95
        let mut det = RustCompositeRegimeDetector::new(720, 30, 25.0, 15.0);
        // Push 35 bars with low vov first
        let normal_feat = make_features_with_vov(0.01, 0.001, 0.01, 0.01, 20.0);
        for _ in 0..35 {
            det.detect(normal_feat.clone());
        }
        // Now push a bar with very high vov (above p95 of history)
        let crisis_feat = make_features_with_vov(0.01, 100.0, 0.01, 0.01, 20.0);
        let result = det.detect(crisis_feat);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.vol_label, "crisis");
        assert!(r.is_crisis);
    }

    #[test]
    fn test_trend_strong_up() {
        // ma20>0, ma50>0, adx=30 → "strong_up"
        let (label, score) = classify_trend(0.01, 0.01, 30.0, 25.0, 15.0);
        assert_eq!(label, "strong_up");
        assert!((score - (30.0_f64 / 50.0).min(1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_trend_ranging_disagree() {
        // MAs disagree → ranging
        let (label, _) = classify_trend(0.01, -0.01, 30.0, 25.0, 15.0);
        assert_eq!(label, "ranging");
    }

    #[test]
    fn test_trend_ranging_low_adx() {
        // Both MAs up but ADX < ranging threshold → ranging
        let (label, _) = classify_trend(0.01, 0.01, 10.0, 25.0, 15.0);
        assert_eq!(label, "ranging");
    }

    #[test]
    fn test_trend_strong_down() {
        let (label, _) = classify_trend(-0.01, -0.01, 30.0, 25.0, 15.0);
        assert_eq!(label, "strong_down");
    }

    #[test]
    fn test_trend_weak_up() {
        let (label, _) = classify_trend(0.01, 0.01, 20.0, 25.0, 15.0);
        assert_eq!(label, "weak_up");
    }

    #[test]
    fn test_trend_weak_down() {
        let (label, _) = classify_trend(-0.01, -0.01, 20.0, 25.0, 15.0);
        assert_eq!(label, "weak_down");
    }

    #[test]
    fn test_param_router_exact() {
        let router = RustRegimeParamRouter::new();
        let p = router.route("strong_up", "low_vol");
        assert!((p.deadzone - 0.3).abs() < 1e-9);
        assert_eq!(p.min_hold, 18);
        assert_eq!(p.max_hold, 60);
        assert!((p.position_scale - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_param_router_exact_strong_up_normal_vol() {
        let router = RustRegimeParamRouter::new();
        let p = router.route("strong_up", "normal_vol");
        assert!((p.deadzone - 0.5).abs() < 1e-9);
        assert_eq!(p.min_hold, 18);
        assert!((p.position_scale - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_param_router_wildcard_trend() {
        let router = RustRegimeParamRouter::new();
        // "strong_up" + "crisis" → no exact match → wildcard ("*", "crisis")
        let p = router.route("strong_up", "crisis");
        assert!((p.deadzone - 2.5).abs() < 1e-9);
        assert_eq!(p.min_hold, 48);
        assert!((p.position_scale - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_param_router_wildcard_high_vol() {
        let router = RustRegimeParamRouter::new();
        // "strong_up" + "high_vol" → exact matches ranging/high_vol only,
        // but for strong_up there's no exact entry. Check wildcard ("*", "high_vol")
        // Wait — "ranging" has exact ("ranging", "high_vol"), but "strong_up"+"high_vol" does not.
        // So it falls through to ("*", "high_vol")
        let p = router.route("strong_up", "high_vol");
        assert!((p.deadzone - 1.5).abs() < 1e-9);
        assert_eq!(p.min_hold, 24);
        assert_eq!(p.max_hold, 60);
    }

    #[test]
    fn test_param_router_fallback() {
        let router = RustRegimeParamRouter::new();
        let p = router.route("unknown_trend", "unknown_vol");
        assert!((p.deadzone - 1.0).abs() < 1e-9);
        assert_eq!(p.min_hold, 24);
        assert_eq!(p.max_hold, 96);
        assert!((p.position_scale - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_missing_parkinson_returns_none() {
        let mut det = RustCompositeRegimeDetector::new(720, 30, 25.0, 15.0);
        let features: HashMap<String, f64> = [
            ("close_vs_ma20".to_string(), 0.01),
            ("close_vs_ma50".to_string(), 0.01),
        ]
        .iter()
        .cloned()
        .collect();
        // No parkinson_vol → None immediately
        assert!(det.detect(features).is_none());
    }

    #[test]
    fn test_missing_ma_returns_none() {
        // Has parkinson but no MA → None after enough bars
        let mut det = RustCompositeRegimeDetector::new(720, 30, 25.0, 15.0);
        let mut m = HashMap::new();
        m.insert("parkinson_vol".to_string(), 0.01_f64);
        for _ in 0..35 {
            det.detect(m.clone());
        }
        // Should return None because close_vs_ma20 and close_vs_ma50 are missing
        assert!(det.detect(m).is_none());
    }

    #[test]
    fn test_composite_value_format() {
        // Verify the composite value string is "trend|vol"
        let mut det = RustCompositeRegimeDetector::new(720, 30, 25.0, 15.0);
        let features = make_features(0.01, 0.01, 0.01, 30.0);
        for _ in 0..35 {
            det.detect(features.clone());
        }
        let result = det.detect(features).unwrap();
        // value should be "trend_label|vol_label"
        let parts: Vec<&str> = result.value.split('|').collect();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], result.trend_label);
        assert_eq!(parts[1], result.vol_label);
    }

    #[test]
    fn test_is_favorable_strong_trend_low_vol() {
        // Fill with high vol, then test low vol + strong trend → is_favorable = true
        let mut det = RustCompositeRegimeDetector::new(720, 30, 25.0, 15.0);
        let high_vol = make_features(0.1, 0.01, 0.01, 30.0);
        let low_vol_strong = make_features(0.001, 0.01, 0.01, 30.0);
        for _ in 0..35 {
            det.detect(high_vol.clone());
        }
        let r = det.detect(low_vol_strong).unwrap();
        assert_eq!(r.trend_label, "strong_up");
        assert_eq!(r.vol_label, "low_vol");
        assert!(r.is_favorable);
    }

    #[test]
    fn test_is_favorable_false_ranging() {
        let mut det = RustCompositeRegimeDetector::new(720, 30, 25.0, 15.0);
        // ranging trend → is_favorable = false
        let features = make_features(0.01, 0.01, -0.01, 20.0);
        for _ in 0..35 {
            det.detect(features.clone());
        }
        let r = det.detect(features).unwrap();
        assert_eq!(r.trend_label, "ranging");
        assert!(!r.is_favorable);
    }
}
