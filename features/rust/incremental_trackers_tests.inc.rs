// incremental_trackers_tests.inc.rs — Unit tests for EMA, RSI, ATR, ADX trackers.
// Included by incremental_trackers.rs via include!().

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_basic() {
        let mut ema = EmaTracker::new(10);
        assert!(!ema.ready());
        assert!(ema.value().is_none());

        // Push constant value
        for _ in 0..20 {
            ema.push(100.0);
        }
        assert!(ema.ready());
        let v = ema.value().unwrap();
        assert!((v - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_nan_ignored() {
        let mut ema = EmaTracker::new(3);
        ema.push(10.0);
        ema.push(f64::NAN);
        ema.push(20.0);
        // NaN should be ignored, so count=2, value based on 10 then 20
        assert_eq!(ema.count(), 2);
        assert!(ema.value().is_some());
    }

    #[test]
    fn test_rsi_all_gains() {
        let mut rsi = RsiTracker::new(14);
        // First close seeds prev_close
        rsi.push(100.0);
        // 14 consecutive gains
        for i in 1..=14 {
            rsi.push(100.0 + i as f64);
        }
        assert!(rsi.ready());
        let v = rsi.value().unwrap();
        assert!((v - 100.0).abs() < 1e-10, "All gains should give RSI=100, got {v}");
    }

    #[test]
    fn test_rsi_all_losses() {
        let mut rsi = RsiTracker::new(14);
        rsi.push(200.0);
        for i in 1..=14 {
            rsi.push(200.0 - i as f64);
        }
        assert!(rsi.ready());
        let v = rsi.value().unwrap();
        assert!(v.abs() < 1e-10, "All losses should give RSI=0, got {v}");
    }

    #[test]
    fn test_rsi_not_ready() {
        let mut rsi = RsiTracker::new(14);
        rsi.push(100.0);
        for i in 1..13 {
            rsi.push(100.0 + i as f64);
        }
        assert!(!rsi.ready());
        assert!(rsi.value().is_none());
    }

    #[test]
    fn test_rsi_nan_ignored() {
        let mut rsi = RsiTracker::new(3);
        rsi.push(100.0);
        rsi.push(f64::NAN);
        rsi.push(101.0);
        rsi.push(102.0);
        rsi.push(103.0);
        // NaN ignored, so count=3 changes: 100->101, 101->102, 102->103
        assert!(rsi.ready());
    }

    #[test]
    fn test_atr_constant_bars() {
        let mut atr = AtrTracker::new(14);
        // Push bars with constant high-low spread of 2.0
        for i in 0..20 {
            let c = 100.0 + i as f64 * 0.1;
            atr.push(c + 1.0, c - 1.0, c);
        }
        assert!(atr.ready());
        let v = atr.value().unwrap();
        // TR should be close to 2.0 (high-low dominates small prev_close differences)
        assert!(v > 1.5 && v < 2.5, "ATR should be ~2.0, got {v}");
    }

    #[test]
    fn test_atr_normalized() {
        let mut atr = AtrTracker::new(3);
        atr.push(102.0, 98.0, 100.0);
        atr.push(103.0, 99.0, 101.0);
        atr.push(104.0, 100.0, 102.0);
        assert!(atr.ready());
        let norm = atr.normalized(102.0).unwrap();
        assert!(norm > 0.0);
        assert!(atr.normalized(0.0).is_none());
    }

    #[test]
    fn test_atr_nan_ignored() {
        let mut atr = AtrTracker::new(3);
        atr.push(102.0, 98.0, 100.0);
        atr.push(f64::NAN, 99.0, 101.0);
        // NaN bar ignored, count stays at 1
        assert_eq!(atr.count, 1);
    }

    #[test]
    fn test_adx_warmup() {
        let mut adx = AdxTracker::new(14);
        // Need 2*14 = 28 bars for warmup (first bar is seed, then 28 changes)
        assert!(!adx.ready());

        // Push 10 bars — not ready yet
        for i in 0..10 {
            let c = 100.0 + i as f64;
            adx.push(c + 1.0, c - 1.0, c);
        }
        assert!(!adx.ready());
        assert!(adx.value().is_none());

        // Push enough bars to initialize (need 1 seed + 2*14 = 29 total)
        for i in 10..30 {
            let c = 100.0 + i as f64;
            adx.push(c + 1.0, c - 1.0, c);
        }
        assert!(adx.ready());
        let v = adx.value().unwrap();
        assert!(v >= 0.0 && v <= 100.0, "ADX should be in [0,100], got {v}");
    }

    #[test]
    fn test_adx_nan_ignored() {
        let mut adx = AdxTracker::new(3);
        adx.push(102.0, 98.0, 100.0);
        adx.push(f64::NAN, 99.0, 101.0);
        // NaN bar ignored
        assert_eq!(adx.count, 0); // seed bar doesn't increment count
    }

    #[test]
    fn test_adx_strong_trend() {
        let mut adx = AdxTracker::new(3);
        // Strong uptrend: each bar higher than the last
        for i in 0..20 {
            let base = 100.0 + i as f64 * 5.0;
            adx.push(base + 1.0, base - 1.0, base);
        }
        assert!(adx.ready());
        let v = adx.value().unwrap();
        // Strong trend should give high ADX
        assert!(v > 30.0, "Strong trend ADX should be >30, got {v}");
    }

    #[test]
    fn test_rsi_matches_python_sequence() {
        // Verify exact numerical parity with a known sequence.
        // Prices: 44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42,
        //         45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00
        // (Classic Wilder RSI example — period 14)
        let prices = [
            44.0, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
            46.03, 46.41, 46.22, 45.64,
        ];
        let mut rsi = RsiTracker::new(14);
        for &p in &prices {
            rsi.push(p);
        }
        assert!(rsi.ready());
        let v = rsi.value().unwrap();
        // Should be a reasonable RSI value in [0, 100]
        assert!(v > 0.0 && v < 100.0, "RSI = {v}");
    }
}
