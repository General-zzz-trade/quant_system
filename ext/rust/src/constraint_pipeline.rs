// constraint_pipeline.rs — Shared constraint primitives for signal processing.
//
// ALL signal constraint logic lives here. Both batch (backtest) and incremental
// (live/tick) paths call these same functions, eliminating semantic drift.
//
// Pipeline: raw_score → z-score → long_only clip → discretize → min-hold
//           → trend-hold → monthly gate → vol-adaptive sizing

use std::collections::VecDeque;

// ── Atomic operations (pure, no state) ──────────────────────

/// Discretize a z-score to {-1, 0, 1} using deadzone threshold.
#[inline]
pub fn discretize(z: f64, deadzone: f64) -> f64 {
    if z > deadzone {
        1.0
    } else if z < -deadzone {
        -1.0
    } else {
        0.0
    }
}

/// Apply long-only clip before discretization: clamp negative z to 0.
#[inline]
pub fn long_only_clip(z: f64, long_only: bool) -> f64 {
    if long_only { z.max(0.0) } else { z }
}

/// Single-step min-hold + trend-hold enforcement.
///
/// Given a desired signal and the previous state, returns (output_signal, new_hold_count).
/// Handles both min-hold lockout and trend-based position extension.
///
/// `hold_count` should be pre-adjusted: if state.hold_counter == 0, pass min_hold.
#[inline]
pub fn enforce_hold_step(
    desired: f64,
    prev_signal: f64,
    hold_count: i32,
    min_hold: i32,
    trend_follow: bool,
    trend_val: f64,
    trend_threshold: f64,
    max_hold: i32,
) -> (f64, i32) {
    // Min-hold lockout
    if hold_count < min_hold {
        return (prev_signal, hold_count + 1);
    }

    // Trend hold: extend long when raw goes flat but trend is up
    if trend_follow && desired == 0.0 && prev_signal > 0.0 && hold_count < max_hold {
        if !trend_val.is_nan() && trend_val > trend_threshold {
            return (prev_signal, hold_count + 1);
        }
    }

    // Allow change
    if desired != prev_signal {
        (desired, 1)
    } else {
        (desired, hold_count + 1)
    }
}

/// Vol-adaptive scaling: scale signal by (vol_target / realized_vol), capped at 1.0.
#[inline]
pub fn vol_scale(signal: f64, vol_val: f64, vol_target: f64) -> f64 {
    if signal != 0.0 && !vol_val.is_nan() && vol_val > 1e-8 {
        signal * (vol_target / vol_val).min(1.0)
    } else {
        signal
    }
}

/// Monthly gate check: returns true if trading is allowed (close > SMA).
/// During warmup (history_len < window), always returns true.
#[inline]
pub fn check_monthly_gate_inline(close: f64, close_sum: f64, history_len: usize, window: usize) -> bool {
    if history_len < window {
        return true; // warmup — allow trading
    }
    close > close_sum / window as f64
}

/// Update close history buffer on new hour boundary.
/// Returns (is_gate_ok, close_sum) for monthly gate check.
pub fn update_monthly_gate(
    close_history: &mut VecDeque<f64>,
    gate_last_hour: &mut i64,
    close: f64,
    hour_key: i64,
    window: usize,
) -> bool {
    if hour_key != *gate_last_hour {
        if close_history.len() >= window {
            close_history.pop_front();
        }
        close_history.push_back(close);
        *gate_last_hour = hour_key;
    }
    if close_history.len() < window {
        return true; // warmup
    }
    let sum: f64 = close_history.iter().sum();
    close > sum / window as f64
}

/// Short-only discretization: z < -deadzone → -1, else 0.
#[inline]
pub fn discretize_short(z: f64, deadzone: f64) -> f64 {
    if z < -deadzone { -1.0 } else { 0.0 }
}

/// Single-step short min-hold enforcement (simplified, no trend-hold).
#[inline]
pub fn enforce_short_hold_step(
    desired: f64,
    prev_signal: f64,
    hold_count: i32,
    min_hold: i32,
) -> (f64, i32) {
    if hold_count < min_hold {
        return (prev_signal, hold_count + 1);
    }
    if desired != prev_signal {
        (desired, 1)
    } else {
        (desired, hold_count + 1)
    }
}


// ── Batch operations (for backtest array processing) ─────────

/// Batch: rolling z-score → long_only clip → discretize over an array.
/// Returns discrete signal array {-1, 0, 1}.
pub fn zscore_discretize_array(
    y_pred: &[f64],
    deadzone: f64,
    zscore_window: usize,
    zscore_warmup: usize,
    long_only: bool,
) -> Vec<f64> {
    let n = y_pred.len();
    if n == 0 {
        return vec![];
    }
    let warmup = std::cmp::min(zscore_warmup, zscore_window);

    let mut buf = vec![0.0_f64; zscore_window];
    let mut buf_idx: usize = 0;
    let mut buf_count: usize = 0;
    let mut raw = vec![0.0_f64; n];

    for i in 0..n {
        buf[buf_idx] = y_pred[i];
        buf_idx = (buf_idx + 1) % zscore_window;
        buf_count = std::cmp::min(buf_count + 1, zscore_window);

        if buf_count < warmup {
            continue;
        }

        // Compute mean and population std
        let cnt = buf_count;
        let mut sum = 0.0_f64;
        let mut sum2 = 0.0_f64;
        for j in 0..cnt {
            sum += buf[j];
            sum2 += buf[j] * buf[j];
        }
        let mu = sum / cnt as f64;
        let mut var = sum2 / cnt as f64 - mu * mu;
        if var < 0.0 { var = 0.0; }
        let std_val = var.sqrt();

        if std_val < 1e-12 {
            continue;
        }

        let z = (y_pred[i] - mu) / std_val;
        let z = long_only_clip(z, long_only);
        raw[i] = discretize(z, deadzone);
    }

    raw
}

/// Batch: min-hold + trend-hold enforcement over an array.
pub fn enforce_hold_array(
    raw: &[f64],
    min_hold: i32,
    trend_follow: bool,
    trend_values: Option<&[f64]>,
    trend_threshold: f64,
    max_hold: i32,
) -> Vec<f64> {
    let n = raw.len();
    if n == 0 {
        return vec![];
    }

    let mut signal = vec![0.0_f64; n];
    signal[0] = raw[0];
    let mut hold_count: i32 = 1;

    for i in 1..n {
        let trend_val = if trend_follow {
            trend_values.and_then(|tv| {
                if i < tv.len() { Some(tv[i]) } else { None }
            }).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        let (sig, new_hold) = enforce_hold_step(
            raw[i], signal[i - 1], hold_count, min_hold,
            trend_follow, trend_val, trend_threshold, max_hold,
        );
        signal[i] = sig;
        hold_count = new_hold;
    }

    signal
}

/// Batch: compute bear mask (close <= SMA). True = bear (block longs).
pub fn compute_bear_mask(closes: &[f64], ma_window: usize) -> Vec<bool> {
    let n = closes.len();
    let mut mask = vec![false; n];
    if n < ma_window {
        return mask;
    }

    // Cumsum
    let mut cs = vec![0.0_f64; n];
    cs[0] = closes[0];
    for i in 1..n {
        cs[i] = cs[i - 1] + closes[i];
    }

    for i in ma_window..n {
        let ma = (cs[i] - cs[i - ma_window]) / ma_window as f64;
        mask[i] = closes[i] <= ma;
    }

    mask
}

/// Batch: vol-adaptive sizing over an array.
pub fn vol_scale_array(signal: &mut [f64], vol_values: &[f64], vol_target: f64) {
    for i in 0..signal.len() {
        if i < vol_values.len() {
            signal[i] = vol_scale(signal[i], vol_values[i], vol_target);
        }
    }
}


/// Batch: single-pass min-hold + trend-hold + monthly gate override.
///
/// Matches live `apply_signal_pipeline` semantics exactly:
///   1. enforce_hold_step runs on the raw discretized signal
///   2. If gate_mask[i] is true (bear regime), the output is overridden
///      AFTER min-hold (bypassing min-hold protection), and hold_count
///      resets to 1 on position change.
///
/// This eliminates the previous backtest approach of:
///   min-hold pass → gate zeros signal → re-apply min-hold (second pass)
/// which produced different behavior from live.
pub fn enforce_hold_with_gate_array(
    raw: &[f64],
    min_hold: i32,
    trend_follow: bool,
    trend_values: Option<&[f64]>,
    trend_threshold: f64,
    max_hold: i32,
    gate_mask: Option<&[bool]>,
    gate_scores: Option<&[f64]>,
) -> Vec<f64> {
    let n = raw.len();
    if n == 0 {
        return vec![];
    }

    let mut signal = vec![0.0_f64; n];
    let mut hold_count: i32 = 1;

    // Bar 0: set initial signal, then apply gate if needed
    signal[0] = raw[0];
    if let Some(mask) = gate_mask {
        if mask.len() > 0 && mask[0] {
            let gate_val = gate_scores.map_or(0.0, |gs| if gs.len() > 0 { gs[0] } else { 0.0 });
            if gate_val != signal[0] {
                signal[0] = gate_val;
                hold_count = 1;
            }
        }
    }

    for i in 1..n {
        let trend_val = if trend_follow {
            trend_values.and_then(|tv| {
                if i < tv.len() { Some(tv[i]) } else { None }
            }).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };

        // Step 1: Normal min-hold + trend-hold
        let (mut sig, mut new_hold) = enforce_hold_step(
            raw[i], signal[i - 1], hold_count, min_hold,
            trend_follow, trend_val, trend_threshold, max_hold,
        );

        // Step 2: Gate override (bypasses min-hold, matches live behavior)
        let gated = gate_mask.map_or(false, |m| i < m.len() && m[i]);
        if gated {
            let gate_val = gate_scores.map_or(0.0, |gs| {
                if i < gs.len() { gs[i] } else { 0.0 }
            });
            if gate_val != sig {
                sig = gate_val;
                new_hold = 1;
            }
        }

        signal[i] = sig;
        hold_count = new_hold;
    }

    signal
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discretize() {
        assert_eq!(discretize(0.6, 0.5), 1.0);
        assert_eq!(discretize(-0.6, 0.5), -1.0);
        assert_eq!(discretize(0.3, 0.5), 0.0);
        assert_eq!(discretize(0.5, 0.5), 0.0); // boundary: not >
        assert_eq!(discretize(-0.5, 0.5), 0.0); // boundary: not <
    }

    #[test]
    fn test_long_only_clip() {
        assert_eq!(long_only_clip(-1.0, true), 0.0);
        assert_eq!(long_only_clip(1.0, true), 1.0);
        assert_eq!(long_only_clip(-1.0, false), -1.0);
    }

    #[test]
    fn test_enforce_hold_step_lockout() {
        // During min-hold lockout, previous signal is maintained
        let (sig, hold) = enforce_hold_step(1.0, -1.0, 3, 10, false, f64::NAN, 0.0, 120);
        assert_eq!(sig, -1.0);
        assert_eq!(hold, 4);
    }

    #[test]
    fn test_enforce_hold_step_change() {
        // After min-hold, signal can change
        let (sig, hold) = enforce_hold_step(1.0, -1.0, 10, 10, false, f64::NAN, 0.0, 120);
        assert_eq!(sig, 1.0);
        assert_eq!(hold, 1);
    }

    #[test]
    fn test_enforce_hold_step_trend_extend() {
        // Trend hold: extend long when trend is up
        let (sig, hold) = enforce_hold_step(0.0, 1.0, 30, 10, true, 0.5, 0.0, 120);
        assert_eq!(sig, 1.0); // held
        assert_eq!(hold, 31);
    }

    #[test]
    fn test_enforce_hold_step_trend_no_extend_at_max() {
        // Don't extend past max_hold
        let (sig, hold) = enforce_hold_step(0.0, 1.0, 120, 10, true, 0.5, 0.0, 120);
        assert_eq!(sig, 0.0); // released
        assert_eq!(hold, 1);
    }

    #[test]
    fn test_vol_scale() {
        assert!((vol_scale(1.0, 0.02, 0.01) - 0.5).abs() < 1e-10);
        assert!((vol_scale(1.0, 0.005, 0.01) - 1.0).abs() < 1e-10); // capped at 1.0
        assert_eq!(vol_scale(0.0, 0.02, 0.01), 0.0); // zero signal unchanged
        assert_eq!(vol_scale(1.0, f64::NAN, 0.01), 1.0); // NaN vol unchanged
    }

    #[test]
    fn test_discretize_short() {
        assert_eq!(discretize_short(-0.6, 0.5), -1.0);
        assert_eq!(discretize_short(0.6, 0.5), 0.0);
        assert_eq!(discretize_short(0.0, 0.5), 0.0);
    }

    #[test]
    fn test_enforce_hold_array_basic() {
        let raw = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0];
        let signal = enforce_hold_array(&raw, 3, false, None, 0.0, 120);
        // hold_count starts at 1 for raw[0]=0
        // i=1: hold=1<3 → keep 0, hold=2
        // i=2: hold=2<3 → keep 0, hold=3
        // i=3: hold=3>=3, desired=1.0 != prev=0.0 → 1.0, hold=1
        // i=4: hold=1<3 → keep 1.0, hold=2
        // i=5: hold=2<3 → keep 1.0, hold=3
        // i=6: hold=3>=3, desired=-1.0 != prev=1.0 → -1.0, hold=1
        assert_eq!(signal, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_compute_bear_mask_warmup() {
        let closes = vec![100.0, 101.0, 99.0];
        let mask = compute_bear_mask(&closes, 5);
        assert_eq!(mask, vec![false, false, false]); // all warmup
    }

    #[test]
    fn test_enforce_hold_with_gate_no_gate() {
        // Without gate, should behave identically to enforce_hold_array
        let raw = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0];
        let signal_no_gate = enforce_hold_with_gate_array(&raw, 3, false, None, 0.0, 120, None, None);
        let signal_plain = enforce_hold_array(&raw, 3, false, None, 0.0, 120);
        assert_eq!(signal_no_gate, signal_plain);
    }

    #[test]
    fn test_enforce_hold_with_gate_overrides_mid_hold() {
        // Gate overrides signal even during min-hold lockout
        // raw:       [0, 0, 1, 1, 0]
        // min_hold=3
        // Without gate: [0, 0, 0, 1, 1]  (hold locks bar 2)
        // Gate at bar 3: [0, 0, 0, 0, 0]  (gate overrides bar 3 to 0)
        let raw = vec![0.0, 0.0, 1.0, 1.0, 0.0];
        let gate_mask = vec![false, false, false, true, false];
        let signal = enforce_hold_with_gate_array(
            &raw, 3, false, None, 0.0, 120,
            Some(&gate_mask), None,
        );
        // bar 0: raw=0, signal=0, hold=1
        // bar 1: hold=1<3, keep 0, hold=2
        // bar 2: hold=2<3, keep 0, hold=3
        // bar 3: hold=3>=3, desired=1!=0 → 1, hold=1; then gate → 0, hold=1
        // bar 4: hold=1<3, keep 0, hold=2
        assert_eq!(signal, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_enforce_hold_with_gate_uses_gate_scores() {
        // Gate replaces signal with bear model score
        let raw = vec![1.0, 1.0, 1.0, 1.0];
        let gate_mask = vec![false, false, true, true];
        let gate_scores = vec![0.0, 0.0, -1.0, -1.0];
        let signal = enforce_hold_with_gate_array(
            &raw, 1, false, None, 0.0, 120,
            Some(&gate_mask), Some(&gate_scores),
        );
        // bar 0: raw=1, no gate → 1, hold=1
        // bar 1: hold>=1, desired=1, same → 1, hold=2
        // bar 2: hold>=1, desired=1, same → 1, hold=3; gate → -1, hold=1
        // bar 3: hold=1>=1, desired=1, !=prev(-1) → 1, hold=1; gate → -1, hold=1
        assert_eq!(signal, vec![1.0, 1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_enforce_hold_with_gate_matches_live_semantics() {
        // Scenario: holding long for 5 bars, gate triggers at bar 5
        // Live behavior: gate overrides immediately (bypasses min-hold)
        // Old backtest: gate zeros signal, then re-min-hold would lock it
        let raw = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let gate_mask = vec![false, false, false, false, false, true, true, true];
        let signal = enforce_hold_with_gate_array(
            &raw, 3, false, None, 0.0, 120,
            Some(&gate_mask), None,
        );
        // bar 5: min-hold ok (hold>=3), desired=0!=1 → 0, hold=1; gate → 0 (same), hold=1
        // bar 6: hold=1<3, keep 0; gate → 0 (same)
        // bar 7: hold=2<3, keep 0; gate → 0 (same)
        assert_eq!(signal[5], 0.0);
        assert_eq!(signal[6], 0.0);
        assert_eq!(signal[7], 0.0);
    }
}
