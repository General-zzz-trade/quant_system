//! Multi-timeframe features: resample 1h OHLCV to 4h, compute 10 features,
//! map back to 1h with anti-lookahead (uses group G-1).
//! Port of ext/rolling/multi_timeframe.hpp

use pyo3::prelude::*;

const FOUR_HOURS_MS: i64 = 4 * 3600 * 1000;
const N_FEAT: usize = 10;

// ── Internal helpers (NaN-aware, matching C++ mtf:: namespace) ───

fn mtf_ema(arr: &[f64], span: usize, out: &mut [f64]) {
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut started = false;
    let mut prev = 0.0;
    for i in 0..arr.len() {
        out[i] = f64::NAN;
        if arr[i].is_nan() {
            continue;
        }
        if !started {
            out[i] = arr[i];
            prev = arr[i];
            started = true;
        } else {
            out[i] = alpha * arr[i] + (1.0 - alpha) * prev;
            prev = out[i];
        }
    }
}

fn mtf_sma(arr: &[f64], window: usize, out: &mut [f64]) {
    let n = arr.len();
    let mut cumsum = vec![0.0; n];
    let mut cs = 0.0;
    for i in 0..n {
        cs += if arr[i].is_nan() { 0.0 } else { arr[i] };
        cumsum[i] = cs;
    }
    for i in 0..window.saturating_sub(1).min(n) {
        out[i] = f64::NAN;
    }
    for i in (window - 1)..n {
        let s = cumsum[i] - if i >= window { cumsum[i - window] } else { 0.0 };
        out[i] = s / window as f64;
    }
}

fn mtf_rolling_std(arr: &[f64], window: usize, out: &mut [f64]) {
    let n = arr.len();
    let min_valid = window / 2;
    for i in 0..n {
        out[i] = f64::NAN;
        if i < window - 1 {
            continue;
        }
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        let mut valid = 0usize;
        for k in (i + 1 - window)..=i {
            if !arr[k].is_nan() {
                sum += arr[k];
                sumsq += arr[k] * arr[k];
                valid += 1;
            }
        }
        if valid < min_valid {
            continue;
        }
        let mean = sum / valid as f64;
        let var = (sumsq - valid as f64 * mean * mean) / (valid as f64 - 1.0);
        out[i] = if var > 0.0 { var.sqrt() } else { 0.0 };
    }
}

#[allow(dead_code)]
struct Group4H {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    _volume: f64,
    _group_id: i64,
}

/// Compute 10 four-hour features from 1h OHLCV data.
/// Returns (n, 10) matrix as Vec<Vec<f64>> (row-major).
/// Anti-lookahead: each 1h bar gets features from the PREVIOUS 4h group.
#[pyfunction]
#[pyo3(signature = (timestamps, opens, highs, lows, closes, volumes))]
pub fn cpp_compute_4h_features(
    timestamps: Vec<i64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
) -> Vec<Vec<f64>> {
    let n = timestamps.len();
    if n == 0 {
        return vec![];
    }

    // Step 1: Resample to 4h bars
    let group_keys: Vec<i64> = timestamps.iter().map(|&t| t / FOUR_HOURS_MS).collect();

    let mut bars: Vec<Group4H> = Vec::new();
    let mut group_to_idx: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();

    let mut i = 0;
    while i < n {
        let gid = group_keys[i];
        let o = opens[i];
        let mut h = highs[i];
        let mut l = lows[i];
        let mut c = closes[i];
        let mut v = volumes[i];
        let mut j = i + 1;
        while j < n && group_keys[j] == gid {
            if highs[j] > h { h = highs[j]; }
            if lows[j] < l { l = lows[j]; }
            c = closes[j];
            v += volumes[j];
            j += 1;
        }
        group_to_idx.insert(gid, bars.len());
        bars.push(Group4H { open: o, high: h, low: l, close: c, _volume: v, _group_id: gid });
        i = j;
    }

    let n4 = bars.len();

    let close_4h: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let high_4h: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low_4h: Vec<f64> = bars.iter().map(|b| b.low).collect();

    // Step 2: Compute 10 features on 4h bars
    let mut ret_1 = vec![f64::NAN; n4];
    let mut ret_3 = vec![f64::NAN; n4];
    let mut ret_6 = vec![f64::NAN; n4];
    let mut rsi_14 = vec![f64::NAN; n4];
    let mut macd_hist = vec![f64::NAN; n4];
    let mut bb_pctb = vec![f64::NAN; n4];
    let mut atr_norm = vec![f64::NAN; n4];
    let mut vol_20 = vec![f64::NAN; n4];
    let mut close_vs_ma20 = vec![f64::NAN; n4];
    let mut mean_rev = vec![f64::NAN; n4];

    // Returns
    for i in 1..n4 {
        ret_1[i] = close_4h[i] / close_4h[i - 1] - 1.0;
    }
    for i in 3..n4 {
        ret_3[i] = close_4h[i] / close_4h[i - 3] - 1.0;
    }
    for i in 6..n4 {
        ret_6[i] = close_4h[i] / close_4h[i - 6] - 1.0;
    }

    // RSI-14
    let mut pct = vec![f64::NAN; n4];
    for i in 1..n4 {
        pct[i] = close_4h[i] / close_4h[i - 1] - 1.0;
    }

    let mut gains = vec![0.0; n4];
    let mut losses = vec![0.0; n4];
    for i in 0..n4 {
        if !pct[i].is_nan() {
            if pct[i] > 0.0 { gains[i] = pct[i]; }
            else if pct[i] < 0.0 { losses[i] = -pct[i]; }
        }
    }
    let mut avg_gain = vec![0.0; n4];
    let mut avg_loss = vec![0.0; n4];
    mtf_ema(&gains, 14, &mut avg_gain);
    mtf_ema(&losses, 14, &mut avg_loss);

    for i in 0..n4 {
        if !avg_gain[i].is_nan() && !avg_loss[i].is_nan() {
            if avg_loss[i] < 1e-15 {
                rsi_14[i] = 100.0;
            } else {
                let rs = avg_gain[i] / avg_loss[i];
                rsi_14[i] = 100.0 - 100.0 / (1.0 + rs);
            }
        }
    }

    // MACD (12, 26, 9)
    let mut ema12 = vec![0.0; n4];
    let mut ema26 = vec![0.0; n4];
    let mut macd_line = vec![0.0; n4];
    let mut signal_line = vec![0.0; n4];
    mtf_ema(&close_4h, 12, &mut ema12);
    mtf_ema(&close_4h, 26, &mut ema26);
    for i in 0..n4 {
        macd_line[i] = ema12[i] - ema26[i];
    }
    mtf_ema(&macd_line, 9, &mut signal_line);
    for i in 0..n4 {
        macd_hist[i] = macd_line[i] - signal_line[i];
        if close_4h[i] > 0.0 && !macd_hist[i].is_nan() {
            macd_hist[i] /= close_4h[i];
        }
    }

    // Bollinger %B (20, 2)
    let mut ma20 = vec![0.0; n4];
    let mut std20 = vec![0.0; n4];
    mtf_sma(&close_4h, 20, &mut ma20);
    mtf_rolling_std(&close_4h, 20, &mut std20);
    for i in 0..n4 {
        if !ma20[i].is_nan() && !std20[i].is_nan() && std20[i] > 1e-15 {
            let upper = ma20[i] + 2.0 * std20[i];
            let lower = ma20[i] - 2.0 * std20[i];
            bb_pctb[i] = (close_4h[i] - lower) / (upper - lower);
        }
    }

    // ATR normalized (14)
    let mut tr = vec![f64::NAN; n4];
    let mut atr_raw = vec![0.0; n4];
    for i in 1..n4 {
        let hl = high_4h[i] - low_4h[i];
        let hc = (high_4h[i] - close_4h[i - 1]).abs();
        let lc = (low_4h[i] - close_4h[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    mtf_ema(&tr, 14, &mut atr_raw);
    for i in 0..n4 {
        if !atr_raw[i].is_nan() && close_4h[i] > 0.0 {
            atr_norm[i] = atr_raw[i] / close_4h[i];
        }
    }

    // Volatility (rolling_std of pct, 20)
    mtf_rolling_std(&pct, 20, &mut vol_20);

    // Close vs MA20
    for i in 0..n4 {
        if !ma20[i].is_nan() && ma20[i] > 0.0 {
            close_vs_ma20[i] = close_4h[i] / ma20[i] - 1.0;
        }
    }

    // Mean reversion z-score
    for i in 0..n4 {
        if !ma20[i].is_nan() && !std20[i].is_nan() && std20[i] > 1e-15 {
            mean_rev[i] = (close_4h[i] - ma20[i]) / std20[i];
        }
    }

    // Step 3: Map back to 1h (anti-lookahead: use G-1)
    let feat_arrays: [&[f64]; N_FEAT] = [
        &ret_1, &ret_3, &ret_6, &rsi_14,
        &macd_hist, &bb_pctb, &atr_norm,
        &vol_20, &close_vs_ma20, &mean_rev,
    ];

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let g = group_keys[i];
        let mut row = vec![f64::NAN; N_FEAT];
        if let Some(&idx) = group_to_idx.get(&(g - 1)) {
            for f in 0..N_FEAT {
                row[f] = feat_arrays[f][idx];
            }
        }
        result.push(row);
    }

    result
}

/// Return the 10 feature names in order.
#[pyfunction]
pub fn cpp_4h_feature_names() -> Vec<String> {
    vec![
        "tf4h_ret_1".into(), "tf4h_ret_3".into(), "tf4h_ret_6".into(), "tf4h_rsi_14".into(),
        "tf4h_macd_hist".into(), "tf4h_bb_pctb_20".into(), "tf4h_atr_norm_14".into(),
        "tf4h_vol_20".into(), "tf4h_close_vs_ma20".into(), "tf4h_mean_reversion_20".into(),
    ]
}
