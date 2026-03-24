//! Fast 1-minute feature computation — single-pass over N bars.
//! Port of ext/rolling/fast_1m_features.hpp
//! Output: N x 15 row-major matrix (matching FAST_FEATURE_NAMES order).

use pyo3::prelude::*;

const N_FAST: usize = 15;

// Feature column indices
const F_RET_1: usize = 0;
const F_RET_3: usize = 1;
const F_RET_5: usize = 2;
const F_RET_10: usize = 3;
const F_RSI_6: usize = 4;
const F_VOL_5: usize = 5;
const F_VOL_20: usize = 6;
const F_TAKER_IMBALANCE: usize = 7;
const F_TRADE_INTENSITY: usize = 8;
const F_CVD_10: usize = 9;
const F_BODY_RATIO: usize = 10;
const F_UPPER_SHADOW: usize = 11;
const F_LOWER_SHADOW: usize = 12;
const F_VOL_RATIO_20: usize = 13;
const F_AGGRESSIVE_FLOW_ZSCORE: usize = 14;

// ── Circular buffer for rolling stats ────────────────────────

struct CircBuf {
    buf: Vec<f64>,
    capacity: usize,
    sum: f64,
    sq_sum: f64,
    head: usize,
    count: usize,
}

impl CircBuf {
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            capacity,
            sum: 0.0,
            sq_sum: 0.0,
            head: 0,
            count: 0,
        }
    }

    fn push(&mut self, x: f64) {
        if self.count >= self.capacity {
            let old = self.buf[self.head];
            self.sum -= old;
            self.sq_sum -= old * old;
        } else {
            self.count += 1;
        }
        self.buf[self.head] = x;
        self.sum += x;
        self.sq_sum += x * x;
        self.head = (self.head + 1) % self.capacity;
    }

    fn full(&self) -> bool {
        self.count >= self.capacity
    }

    fn mean(&self) -> f64 {
        self.sum / self.count as f64
    }

    fn std(&self) -> f64 {
        let m = self.mean();
        let v = self.sq_sum / self.count as f64 - m * m;
        if v > 0.0 { v.sqrt() } else { 0.0 }
    }
}

// ── Ring buffer for rolling sum (no variance) ────────────────

struct RingSum {
    buf: Vec<f64>,
    capacity: usize,
    sum: f64,
    head: usize,
    count: usize,
}

impl RingSum {
    fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            capacity,
            sum: 0.0,
            head: 0,
            count: 0,
        }
    }

    fn push(&mut self, x: f64) {
        if self.count >= self.capacity {
            self.sum -= self.buf[self.head];
        } else {
            self.count += 1;
        }
        self.buf[self.head] = x;
        self.sum += x;
        self.head = (self.head + 1) % self.capacity;
    }

    fn full(&self) -> bool {
        self.count >= self.capacity
    }

    fn total(&self) -> f64 {
        self.sum
    }
}

/// Compute 15 fast 1-minute features in a single pass.
/// Returns (n, 15) matrix as Vec<Vec<f64>>.
#[pyfunction]
#[pyo3(signature = (opens, highs, lows, closes, volumes, trades, taker_buy_volumes))]
pub fn cpp_compute_fast_1m_features(
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    trades: Vec<f64>,
    taker_buy_volumes: Vec<f64>,
) -> Vec<Vec<f64>> {
    let n = closes.len();
    if n == 0 {
        return vec![];
    }

    let mut result: Vec<Vec<f64>> = (0..n).map(|_| vec![f64::NAN; N_FAST]).collect();

    // RSI-6 state (EWM, span=6 → alpha = 2/7)
    let rsi_alpha = 2.0 / 7.0;
    let mut avg_gain = 0.0_f64;
    let mut avg_loss = 0.0_f64;
    let mut rsi_started = false;

    // EMA for trades (span=20 → alpha = 2/21)
    let ema_trades_alpha = 2.0 / 21.0;
    let mut ema_trades = 0.0_f64;
    let mut ema_trades_started = false;

    // EMA for volume (span=20)
    let ema_vol_alpha = 2.0 / 21.0;
    let mut ema_vol = 0.0_f64;
    let mut ema_vol_started = false;

    // Rolling windows
    let mut vol5_buf = CircBuf::new(5);
    let mut vol20_buf = CircBuf::new(20);
    let mut cvd10_buf = RingSum::new(10);
    let mut afs_buf = CircBuf::new(24);

    for i in 0..n {
        let row = &mut result[i];
        let c = closes[i];
        let o = opens[i];
        let h = highs[i];
        let l = lows[i];
        let v = volumes[i];
        let t = trades[i];
        let tb = taker_buy_volumes[i];

        // Returns
        if i >= 1 {
            let prev = closes[i - 1];
            if prev != 0.0 { row[F_RET_1] = c / prev - 1.0; }
        }
        if i >= 3 {
            let prev = closes[i - 3];
            if prev != 0.0 { row[F_RET_3] = c / prev - 1.0; }
        }
        if i >= 5 {
            let prev = closes[i - 5];
            if prev != 0.0 { row[F_RET_5] = c / prev - 1.0; }
        }
        if i >= 10 {
            let prev = closes[i - 10];
            if prev != 0.0 { row[F_RET_10] = c / prev - 1.0; }
        }

        // RSI-6
        if i >= 1 {
            let pct = if closes[i - 1] != 0.0 { c / closes[i - 1] - 1.0 } else { 0.0 };
            let gain = if pct > 0.0 { pct } else { 0.0 };
            let loss = if pct < 0.0 { -pct } else { 0.0 };
            if !rsi_started {
                avg_gain = gain;
                avg_loss = loss;
                rsi_started = true;
            } else {
                avg_gain = rsi_alpha * gain + (1.0 - rsi_alpha) * avg_gain;
                avg_loss = rsi_alpha * loss + (1.0 - rsi_alpha) * avg_loss;
            }
            if avg_loss > 0.0 {
                let rs = avg_gain / avg_loss;
                row[F_RSI_6] = 100.0 - 100.0 / (1.0 + rs);
            } else if avg_gain > 0.0 {
                row[F_RSI_6] = 100.0;
            } else {
                row[F_RSI_6] = 50.0;
            }
        }

        // Volatility (rolling std of pct_change)
        if i >= 1 {
            let pct = if closes[i - 1] != 0.0 { c / closes[i - 1] - 1.0 } else { 0.0 };
            vol5_buf.push(pct);
            vol20_buf.push(pct);
            if vol5_buf.count >= 3 { row[F_VOL_5] = vol5_buf.std(); }
            if vol20_buf.count >= 10 { row[F_VOL_20] = vol20_buf.std(); }
        }

        // Taker imbalance
        let taker_ratio = if v > 0.0 { tb / v } else { 0.5 };
        row[F_TAKER_IMBALANCE] = 2.0 * taker_ratio - 1.0;

        // Trade intensity
        if !ema_trades_started {
            ema_trades = t;
            ema_trades_started = true;
        } else {
            ema_trades = ema_trades_alpha * t + (1.0 - ema_trades_alpha) * ema_trades;
        }
        if ema_trades > 0.0 {
            row[F_TRADE_INTENSITY] = t / ema_trades;
        }

        // CVD-10
        let delta = tb - (v - tb); // buy - sell volume
        cvd10_buf.push(delta);

        if !ema_vol_started {
            ema_vol = v;
            ema_vol_started = true;
        } else {
            ema_vol = ema_vol_alpha * v + (1.0 - ema_vol_alpha) * ema_vol;
        }

        if cvd10_buf.full() {
            let denom = ema_vol * 10.0;
            if denom > 0.0 {
                row[F_CVD_10] = cvd10_buf.total() / denom;
            }
        }

        // Candle structure
        let body = (c - o).abs();
        let full_range = h - l;
        if full_range > 0.0 {
            row[F_BODY_RATIO] = body / full_range;
            row[F_UPPER_SHADOW] = (h - c.max(o)) / full_range;
            row[F_LOWER_SHADOW] = (c.min(o) - l) / full_range;
        } else {
            row[F_BODY_RATIO] = 0.0;
            row[F_UPPER_SHADOW] = 0.0;
            row[F_LOWER_SHADOW] = 0.0;
        }

        // Vol ratio 20
        if ema_vol > 0.0 {
            row[F_VOL_RATIO_20] = v / ema_vol;
        }

        // Aggressive flow z-score (24-bar)
        let tbr = if v > 0.0 { tb / v } else { 0.5 };
        afs_buf.push(tbr);
        if afs_buf.full() {
            let s = afs_buf.std();
            if s > 0.0 {
                row[F_AGGRESSIVE_FLOW_ZSCORE] = (tbr - afs_buf.mean()) / s;
            }
        }
    }

    result
}

/// Return the 15 feature names in order.
#[pyfunction]
pub fn cpp_fast_1m_feature_names() -> Vec<String> {
    vec![
        "ret_1".into(), "ret_3".into(), "ret_5".into(), "ret_10".into(),
        "rsi_6".into(),
        "vol_5".into(), "vol_20".into(),
        "taker_imbalance".into(),
        "trade_intensity".into(),
        "cvd_10".into(),
        "body_ratio".into(), "upper_shadow".into(), "lower_shadow".into(),
        "vol_ratio_20".into(),
        "aggressive_flow_zscore".into(),
    ]
}
