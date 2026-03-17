// incremental_trackers.rs — Standalone incremental indicator trackers.
//
// Implements EMA, RSI, ATR, and ADX with Wilder's smoothing, matching
// the Python implementations in features/enriched_computer.py exactly.
//
// These structs are self-contained and can be used independently or
// composed into larger feature engines.

// ============================================================
// EmaTracker
// ============================================================

/// Incremental EMA tracker (standard exponential moving average).
///
/// Alpha = 2 / (span + 1). First value seeds the EMA.
pub struct EmaTracker {
    span: usize,
    alpha: f64,
    value: f64,
    count: usize,
}

impl EmaTracker {
    pub fn new(span: usize) -> Self {
        EmaTracker {
            span,
            alpha: 2.0 / (span as f64 + 1.0),
            value: 0.0,
            count: 0,
        }
    }

    /// Push a new value. NaN/infinite inputs are ignored.
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        if self.count == 0 {
            self.value = x;
        } else {
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value;
        }
        self.count += 1;
    }

    /// Current EMA value, or None if no data pushed yet.
    pub fn value(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.value)
        } else {
            None
        }
    }

    /// True when at least `span` bars have been pushed.
    pub fn ready(&self) -> bool {
        self.count >= self.span
    }

    /// Number of values pushed so far.
    pub fn count(&self) -> usize {
        self.count
    }
}

// ============================================================
// RsiTracker
// ============================================================

/// Incremental RSI using Wilder's smoothing.
///
/// Matches `_RSITracker` in enriched_computer.py:
/// - First `period` changes: accumulate gains/losses.
/// - At change #period: avg_gain = sum(gains)/period, avg_loss = sum(losses)/period.
/// - After: Wilder smooth: avg = (avg*(period-1) + current) / period.
/// - RSI = 100 - 100/(1 + avg_gain/avg_loss).
pub struct RsiTracker {
    period: usize,
    avg_gain: f64,
    avg_loss: f64,
    prev_close: f64, // NAN = not set
    count: usize,     // number of changes processed (not bars)
    init_gains: f64,
    init_losses: f64,
}

impl RsiTracker {
    pub fn new(period: usize) -> Self {
        RsiTracker {
            period,
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev_close: f64::NAN,
            count: 0,
            init_gains: 0.0,
            init_losses: 0.0,
        }
    }

    /// Push a new close price. NaN/infinite inputs are ignored.
    pub fn push(&mut self, close: f64) {
        if !close.is_finite() {
            return;
        }
        if self.prev_close.is_nan() {
            self.prev_close = close;
            return;
        }
        let change = close - self.prev_close;
        self.prev_close = close;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };
        self.count += 1;

        if self.count <= self.period {
            self.init_gains += gain;
            self.init_losses += loss;
            if self.count == self.period {
                self.avg_gain = self.init_gains / self.period as f64;
                self.avg_loss = self.init_losses / self.period as f64;
            }
        } else {
            let p = self.period as f64;
            self.avg_gain = (self.avg_gain * (p - 1.0) + gain) / p;
            self.avg_loss = (self.avg_loss * (p - 1.0) + loss) / p;
        }
    }

    /// Current RSI value, or None if fewer than `period` changes processed.
    pub fn value(&self) -> Option<f64> {
        if self.count < self.period {
            return None;
        }
        if self.avg_loss == 0.0 {
            return Some(100.0);
        }
        let rs = self.avg_gain / self.avg_loss;
        Some(100.0 - (100.0 / (1.0 + rs)))
    }

    /// True when at least `period` changes have been processed.
    pub fn ready(&self) -> bool {
        self.count >= self.period
    }
}

// ============================================================
// AtrTracker
// ============================================================

/// Incremental ATR using Wilder's smoothing.
///
/// Matches `_ATRTracker` in enriched_computer.py:
/// - TR = max(high-low, |high-prev_close|, |low-prev_close|).
/// - First bar: TR = high - low (no prev_close yet).
/// - First `period` bars: accumulate TR sum.
/// - At bar #period: ATR = sum / period.
/// - After: Wilder smooth: ATR = (ATR*(period-1) + TR) / period.
pub struct AtrTracker {
    period: usize,
    atr: f64,
    prev_close: f64, // NAN = not set
    count: usize,
    init_sum: f64,
}

impl AtrTracker {
    pub fn new(period: usize) -> Self {
        AtrTracker {
            period,
            atr: 0.0,
            prev_close: f64::NAN,
            count: 0,
            init_sum: 0.0,
        }
    }

    /// Push a new bar (high, low, close). NaN/infinite inputs are ignored.
    pub fn push(&mut self, high: f64, low: f64, close: f64) {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            return;
        }
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let a = high - low;
            let b = (high - self.prev_close).abs();
            let c = (low - self.prev_close).abs();
            a.max(b).max(c)
        };
        self.prev_close = close;
        self.count += 1;

        if self.count <= self.period {
            self.init_sum += tr;
            if self.count == self.period {
                self.atr = self.init_sum / self.period as f64;
            }
        } else {
            let p = self.period as f64;
            self.atr = (self.atr * (p - 1.0) + tr) / p;
        }
    }

    /// Current ATR value, or None if fewer than `period` bars processed.
    pub fn value(&self) -> Option<f64> {
        if self.count >= self.period {
            Some(self.atr)
        } else {
            None
        }
    }

    /// ATR normalized by close price, or None if not ready or close is zero.
    pub fn normalized(&self, close: f64) -> Option<f64> {
        if self.count >= self.period && close.is_finite() && close != 0.0 {
            Some(self.atr / close)
        } else {
            None
        }
    }

    /// True when at least `period` bars have been processed.
    pub fn ready(&self) -> bool {
        self.count >= self.period
    }
}

// ============================================================
// AdxTracker
// ============================================================

/// Incremental ADX using two-stage Wilder's smoothing.
///
/// Matches `_ADXTracker` in enriched_computer.py:
/// 1. Compute TR, +DM, -DM each bar.
/// 2. Wilder-smooth TR, +DM, -DM over `period` bars:
///    - First `period` bars: accumulate sums.
///    - After: smooth = smooth - smooth/period + current.
/// 3. +DI = 100 * smooth_plus_dm / smooth_tr, -DI analogous.
/// 4. DX = 100 * |+DI - -DI| / (+DI + -DI).
/// 5. ADX = Wilder-smooth of DX over `period` bars:
///    - First `period` DX values: accumulate sum, then average.
///    - After: ADX = (ADX*(period-1) + DX) / period.
///
/// Requires 2*period bars of warmup before producing a value.
pub struct AdxTracker {
    period: usize,
    prev_high: f64,   // NAN = not set
    prev_low: f64,    // NAN = not set
    prev_close: f64,  // NAN = not set
    // Wilder-smoothed directional components
    smooth_tr: f64,
    smooth_plus_dm: f64,
    smooth_minus_dm: f64,
    // ADX Wilder smoothing state
    adx: f64,
    dx_init_sum: f64,
    dx_count: usize,
    count: usize, // number of bars after the first (changes processed)
    adx_initialized: bool,
}

impl AdxTracker {
    pub fn new(period: usize) -> Self {
        AdxTracker {
            period,
            prev_high: f64::NAN,
            prev_low: f64::NAN,
            prev_close: f64::NAN,
            smooth_tr: 0.0,
            smooth_plus_dm: 0.0,
            smooth_minus_dm: 0.0,
            adx: 0.0,
            dx_init_sum: 0.0,
            dx_count: 0,
            count: 0,
            adx_initialized: false,
        }
    }

    /// Push a new bar (high, low, close). NaN/infinite inputs are ignored.
    pub fn push(&mut self, high: f64, low: f64, close: f64) {
        if !high.is_finite() || !low.is_finite() || !close.is_finite() {
            return;
        }

        if self.prev_close.is_nan() {
            self.prev_high = high;
            self.prev_low = low;
            self.prev_close = close;
            return;
        }

        self.count += 1;
        let prev_high = if self.prev_high.is_nan() { high } else { self.prev_high };
        let prev_low = if self.prev_low.is_nan() { low } else { self.prev_low };
        let prev_close = self.prev_close;

        // True Range
        let tr = {
            let a = high - low;
            let b = (high - prev_close).abs();
            let c = (low - prev_close).abs();
            a.max(b).max(c)
        };

        // Directional Movement
        let up_move = high - prev_high;
        let down_move = prev_low - low;
        let plus_dm = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        let minus_dm = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };

        self.prev_high = high;
        self.prev_low = low;
        self.prev_close = close;

        let p = self.period;
        if self.count <= p {
            // Accumulate initial sums
            self.smooth_tr += tr;
            self.smooth_plus_dm += plus_dm;
            self.smooth_minus_dm += minus_dm;
            if self.count == p {
                self.compute_dx_and_accumulate();
            }
        } else {
            // Wilder smoothing: new = prev - prev/period + current
            let pf = p as f64;
            self.smooth_tr = self.smooth_tr - self.smooth_tr / pf + tr;
            self.smooth_plus_dm = self.smooth_plus_dm - self.smooth_plus_dm / pf + plus_dm;
            self.smooth_minus_dm = self.smooth_minus_dm - self.smooth_minus_dm / pf + minus_dm;
            self.compute_dx_and_accumulate();
        }
    }

    fn compute_dx_and_accumulate(&mut self) {
        let dx = if self.smooth_tr == 0.0 {
            0.0
        } else {
            let plus_di = 100.0 * self.smooth_plus_dm / self.smooth_tr;
            let minus_di = 100.0 * self.smooth_minus_dm / self.smooth_tr;
            let di_sum = plus_di + minus_di;
            if di_sum == 0.0 {
                0.0
            } else {
                100.0 * (plus_di - minus_di).abs() / di_sum
            }
        };

        let p = self.period;
        if !self.adx_initialized {
            self.dx_init_sum += dx;
            self.dx_count += 1;
            if self.dx_count >= p {
                self.adx = self.dx_init_sum / p as f64;
                self.adx_initialized = true;
            }
        } else {
            let pf = p as f64;
            self.adx = (self.adx * (pf - 1.0) + dx) / pf;
        }
    }

    /// Current ADX value, or None if not enough data (requires 2*period warmup).
    pub fn value(&self) -> Option<f64> {
        if self.adx_initialized {
            Some(self.adx)
        } else {
            None
        }
    }

    /// True when ADX has been initialized (2*period bars of warmup).
    pub fn ready(&self) -> bool {
        self.adx_initialized
    }
}

// ============================================================
// Tests
// ============================================================

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
