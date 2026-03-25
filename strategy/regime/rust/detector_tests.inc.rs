// detector_tests.inc.rs — Unit tests for regime detector.
// Included by detector.rs via include!().

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
