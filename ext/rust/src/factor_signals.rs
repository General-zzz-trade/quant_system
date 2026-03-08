//! Factor signal computation in Rust.
//!
//! Replaces Python z-score/slope math in decision/signals/factors/*.py.

use pyo3::prelude::*;

/// Momentum signal: cumulative return → z-score.
///
/// Returns (side, score, confidence).
#[pyfunction]
#[pyo3(signature = (closes, lookback=20))]
pub fn rust_momentum_score(closes: Vec<f64>, lookback: usize) -> (String, f64, f64) {
    if closes.len() < lookback + 1 {
        return ("flat".into(), 0.0, 0.0);
    }
    let start = closes.len() - lookback - 1;
    let recent = &closes[start..];
    let mut rets = Vec::with_capacity(lookback);
    for i in 1..recent.len() {
        if recent[i - 1] != 0.0 {
            rets.push(recent[i] / recent[i - 1] - 1.0);
        }
    }
    if rets.len() < 2 {
        return ("flat".into(), 0.0, 0.0);
    }
    let mut cum_ret = 1.0;
    for &r in &rets {
        cum_ret *= 1.0 + r;
    }
    cum_ret -= 1.0;

    let n = rets.len() as f64;
    let mean = rets.iter().sum::<f64>() / n;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
    let std = if var > 0.0 { var.sqrt() } else { 1e-10 };
    let z = if std > 1e-10 { cum_ret / std } else { 0.0 };

    let score = (z * 1e6).round() / 1e6;
    let conf = ((z.abs() / 3.0).min(1.0) * 1e4).round() / 1e4;
    let side = if score > 0.0 {
        "buy"
    } else if score < 0.0 {
        "sell"
    } else {
        "flat"
    };
    (side.into(), score, conf)
}

/// Volatility signal: mean-reversion on realized volatility.
///
/// Returns (side, score, confidence).
#[pyfunction]
#[pyo3(signature = (closes, lookback=20))]
pub fn rust_volatility_score(closes: Vec<f64>, lookback: usize) -> (String, f64, f64) {
    if closes.len() < lookback + 1 {
        return ("flat".into(), 0.0, 0.0);
    }
    let start = closes.len() - lookback - 1;
    let recent = &closes[start..];
    let mut rets = Vec::with_capacity(lookback);
    for i in 1..recent.len() {
        if recent[i - 1] != 0.0 {
            rets.push(recent[i] / recent[i - 1] - 1.0);
        }
    }
    if rets.len() < 4 {
        return ("flat".into(), 0.0, 0.0);
    }

    fn vol(rs: &[f64]) -> f64 {
        let n = rs.len() as f64;
        let mean = rs.iter().sum::<f64>() / n;
        let var = rs.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        if var > 0.0 { var.sqrt() } else { 0.0 }
    }

    let half = rets.len() / 2;
    let recent_vol = vol(&rets[half..]);
    let full_vol = vol(&rets);

    if full_vol < 1e-12 {
        return ("flat".into(), 0.0, 0.0);
    }

    let z = (recent_vol - full_vol) / full_vol;
    // Invert: expanding vol → sell
    let score = ((-z) * 1e6).round() / 1e6;
    let conf = (z.abs().min(1.0) * 1e4).round() / 1e4;
    let side = if score > 0.0 {
        "buy"
    } else if score < 0.0 {
        "sell"
    } else {
        "flat"
    };
    (side.into(), score, conf)
}

/// Liquidity signal: volume z-score.
///
/// Returns (side, score, confidence).
#[pyfunction]
#[pyo3(signature = (volumes, lookback=20))]
pub fn rust_liquidity_score(volumes: Vec<f64>, lookback: usize) -> (String, f64, f64) {
    if volumes.len() < lookback {
        return ("flat".into(), 0.0, 0.0);
    }
    let start = volumes.len() - lookback;
    let recent = &volumes[start..];
    let n = recent.len() as f64;
    let mean = recent.iter().sum::<f64>() / n;
    let var = recent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = if var > 0.0 { var.sqrt() } else { 1e-10 };
    let current = recent[recent.len() - 1];
    let z = if std > 1e-10 { (current - mean) / std } else { 0.0 };

    let score = (z * 1e6).round() / 1e6;
    let conf = ((z.abs() / 3.0).min(1.0) * 1e4).round() / 1e4;
    let side = if score > 0.0 {
        "buy"
    } else if score < 0.0 {
        "sell"
    } else {
        "flat"
    };
    (side.into(), score, conf)
}

/// Volume-price divergence signal.
///
/// Returns (side, score, confidence).
#[pyfunction]
#[pyo3(signature = (closes, volumes, lookback=10))]
pub fn rust_volume_price_div_score(
    closes: Vec<f64>,
    volumes: Vec<f64>,
    lookback: usize,
) -> (String, f64, f64) {
    if closes.len() < lookback + 1 || volumes.len() < lookback + 1 {
        return ("flat".into(), 0.0, 0.0);
    }
    let c_start = closes.len() - lookback - 1;
    let recent_c = &closes[c_start..];
    let v_start = volumes.len() - lookback;
    let recent_v = &volumes[v_start..];

    if recent_c[0] == 0.0 {
        return ("flat".into(), 0.0, 0.0);
    }
    let price_change = (recent_c[recent_c.len() - 1] - recent_c[0]) / recent_c[0].abs();

    let n = recent_v.len() as f64;
    let mean_v = recent_v.iter().sum::<f64>() / n;
    if mean_v == 0.0 {
        return ("flat".into(), 0.0, 0.0);
    }

    if price_change.abs() < 1e-6 {
        return ("flat".into(), 0.0, 0.0);
    }

    let mean_i = (n - 1.0) / 2.0;
    let num: f64 = recent_v
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64 - mean_i) * (v - mean_v))
        .sum();
    let den: f64 = (0..recent_v.len())
        .map(|i| (i as f64 - mean_i).powi(2))
        .sum();
    let vol_slope = if den > 0.0 { (num / den) / mean_v } else { 0.0 };

    let direction = if price_change > 0.0 { 1.0 } else { -1.0 };
    let divergence = vol_slope * direction;
    let score = (divergence * 100.0 * 1e6).round() / 1e6;
    let conf = ((divergence.abs() * 10.0).min(1.0) * 1e4).round() / 1e4;
    let side = if score > 0.0 {
        "buy"
    } else if score < 0.0 {
        "sell"
    } else {
        "flat"
    };
    (side.into(), score, conf)
}

/// Compute ADX (Average Directional Index) from OHLC bars.
///
/// Uses Wilder's smoothing. Returns NaN for warmup period (first 2*window bars).
#[pyfunction]
#[pyo3(signature = (highs, lows, closes, window=14))]
pub fn rust_adx(
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    window: usize,
) -> Vec<f64> {
    let n = highs.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 || window == 0 || n <= window {
        return out;
    }

    // Step 1: +DM, -DM, TR
    let mut plus_dm = vec![0.0_f64; n];
    let mut minus_dm = vec![0.0_f64; n];
    let mut tr_list = vec![highs[0] - lows[0]];

    for i in 1..n {
        let up_move = highs[i] - highs[i - 1];
        let down_move = lows[i - 1] - lows[i];
        plus_dm[i] = if up_move > down_move && up_move > 0.0 {
            up_move
        } else {
            0.0
        };
        minus_dm[i] = if down_move > up_move && down_move > 0.0 {
            down_move
        } else {
            0.0
        };
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
        tr_list.push(tr);
    }

    // Step 2: Wilder's smoothing initial sums
    let mut sm_plus: f64 = plus_dm[..window].iter().sum();
    let mut sm_minus: f64 = minus_dm[..window].iter().sum();
    let mut sm_tr: f64 = tr_list[..window].iter().sum();

    let di_dx = |sp: f64, sm: f64, st: f64| -> Option<f64> {
        if st == 0.0 {
            return None;
        }
        let plus_di = 100.0 * sp / st;
        let minus_di = 100.0 * sm / st;
        let denom = plus_di + minus_di;
        if denom == 0.0 {
            return Some(0.0);
        }
        Some(100.0 * (plus_di - minus_di).abs() / denom)
    };

    let mut dx_values: Vec<f64> = Vec::new();
    if let Some(dx) = di_dx(sm_plus, sm_minus, sm_tr) {
        dx_values.push(dx);
    }

    for i in window..n {
        sm_plus = sm_plus - sm_plus / window as f64 + plus_dm[i];
        sm_minus = sm_minus - sm_minus / window as f64 + minus_dm[i];
        sm_tr = sm_tr - sm_tr / window as f64 + tr_list[i];
        if let Some(dx) = di_dx(sm_plus, sm_minus, sm_tr) {
            dx_values.push(dx);
        }
    }

    if dx_values.len() < window {
        return out;
    }

    // Step 4: ADX = smoothed average of DX
    let mut adx_val: f64 = dx_values[..window].iter().sum::<f64>() / window as f64;
    let start_idx = 2 * window - 1;
    if start_idx < n {
        out[start_idx] = adx_val;
    }

    for j in window..dx_values.len() {
        adx_val = (adx_val * (window as f64 - 1.0) + dx_values[j]) / window as f64;
        let idx = window + j;
        if idx < n {
            out[idx] = adx_val;
        }
    }

    out
}

/// Compute carry signal score from funding rate.
///
/// Returns (side, score, confidence).
#[pyfunction]
pub fn rust_carry_score(funding_rate: f64) -> (String, f64, f64) {
    let score = ((-funding_rate * 10000.0) * 1e6).round() / 1e6;
    let conf = ((funding_rate.abs() * 10000.0).min(1.0) * 1e4).round() / 1e4;
    let side = if score > 0.0 {
        "buy"
    } else if score < 0.0 {
        "sell"
    } else {
        "flat"
    };
    (side.into(), score, conf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_momentum_flat_insufficient_data() {
        let (side, score, conf) = rust_momentum_score(vec![1.0, 2.0], 20);
        assert_eq!(side, "flat");
        assert_eq!(score, 0.0);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_momentum_uptrend() {
        let closes: Vec<f64> = (0..25).map(|i| 100.0 + i as f64).collect();
        let (side, score, _) = rust_momentum_score(closes, 20);
        assert_eq!(side, "buy");
        assert!(score > 0.0);
    }

    #[test]
    fn test_momentum_downtrend() {
        let closes: Vec<f64> = (0..25).map(|i| 100.0 - i as f64 * 0.5).collect();
        let (side, score, _) = rust_momentum_score(closes, 20);
        assert_eq!(side, "sell");
        assert!(score < 0.0);
    }

    #[test]
    fn test_volatility_flat() {
        let (side, _, _) = rust_volatility_score(vec![1.0; 5], 20);
        assert_eq!(side, "flat");
    }

    #[test]
    fn test_liquidity_insufficient() {
        let (side, _, _) = rust_liquidity_score(vec![1.0; 5], 20);
        assert_eq!(side, "flat");
    }

    #[test]
    fn test_liquidity_high_volume() {
        let mut vols = vec![100.0; 20];
        vols[19] = 500.0; // spike
        let (side, score, _) = rust_liquidity_score(vols, 20);
        assert_eq!(side, "buy");
        assert!(score > 0.0);
    }

    #[test]
    fn test_vpd_flat_no_change() {
        let closes = vec![100.0; 15];
        let volumes = vec![1000.0; 15];
        let (side, _, _) = rust_volume_price_div_score(closes, volumes, 10);
        assert_eq!(side, "flat");
    }

    #[test]
    fn test_adx_basic() {
        let n = 50;
        let highs: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 + 2.0).collect();
        let lows: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 - 2.0).collect();
        let closes: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let result = rust_adx(highs, lows, closes, 14);
        assert_eq!(result.len(), n);
        // First 27 (2*14-1) should be NaN
        assert!(result[26].is_nan());
        assert!(result[27].is_finite());
        assert!(result[27] > 0.0);
    }

    #[test]
    fn test_adx_too_short() {
        let result = rust_adx(vec![1.0; 5], vec![0.5; 5], vec![0.8; 5], 14);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_carry_positive_funding() {
        let (side, score, _) = rust_carry_score(0.001);
        assert_eq!(side, "sell");
        assert!(score < 0.0);
    }

    #[test]
    fn test_carry_negative_funding() {
        let (side, score, _) = rust_carry_score(-0.001);
        assert_eq!(side, "buy");
        assert!(score > 0.0);
    }
}
