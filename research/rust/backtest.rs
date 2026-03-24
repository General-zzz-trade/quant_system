// backtest_engine.rs — PyO3 port of C++ backtest_engine.hpp (948 lines)
// Components: pred_to_signal, regime_switch, cost_model, trade_sim, metrics

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::Deserialize;

use crate::decision::constraint_pipeline::{
    zscore_discretize_array, enforce_hold_array, enforce_hold_with_gate_array,
    compute_bear_mask, vol_scale_array,
};

// ── Configuration ────────────────────────────────────────────

#[derive(Deserialize)]
struct BacktestConfig {
    // Signal generation
    #[serde(default = "default_deadzone")]
    deadzone: f64,
    #[serde(default = "default_min_hold")]
    min_hold: i32,
    #[serde(default = "default_zscore_window")]
    zscore_window: i32,
    #[serde(default = "default_zscore_warmup")]
    zscore_warmup: i32,

    // Regime switch
    #[serde(default)]
    use_regime_switch: bool,
    #[serde(default = "default_ma_window")]
    ma_window: i32,
    #[serde(default)]
    bear_thresholds: Vec<[f64; 2]>,

    // Vol-adaptive sizing
    #[serde(default)]
    vol_adaptive: bool,
    #[serde(default)]
    vol_target: f64,

    // DD circuit breaker
    #[serde(default)]
    dd_breaker: bool,
    #[serde(default = "default_dd_limit")]
    dd_limit: f64,
    #[serde(default = "default_dd_cooldown")]
    dd_cooldown: i32,

    // Monthly gate
    #[serde(default)]
    monthly_gate: bool,

    // Long only
    #[serde(default)]
    long_only: bool,

    // Trend hold (matches live inference_bridge.rs)
    #[serde(default)]
    trend_follow: bool,
    #[serde(default)]
    trend_threshold: f64,
    #[serde(default = "default_max_hold")]
    max_hold: i32,

    // Cost model
    #[serde(default)]
    realistic_cost: bool,
    #[serde(default = "default_cost_per_trade")]
    cost_per_trade: f64,
    #[serde(default = "default_maker_fee_bps")]
    maker_fee_bps: f64,
    #[serde(default = "default_taker_fee_bps")]
    taker_fee_bps: f64,
    #[serde(default = "default_taker_ratio")]
    taker_ratio: f64,
    #[serde(default = "default_impact_eta")]
    impact_eta: f64,
    #[serde(default = "default_spread_multiplier")]
    spread_multiplier: f64,
    #[serde(default = "default_max_participation")]
    max_participation: f64,
    #[serde(default = "default_capital")]
    capital: f64,
}

fn default_deadzone() -> f64 { 0.5 }
fn default_min_hold() -> i32 { 24 }
fn default_zscore_window() -> i32 { 720 }
fn default_zscore_warmup() -> i32 { 180 }
fn default_ma_window() -> i32 { 480 }
fn default_dd_limit() -> f64 { -0.15 }
fn default_dd_cooldown() -> i32 { 48 }
fn default_cost_per_trade() -> f64 { 6e-4 }
fn default_maker_fee_bps() -> f64 { 2.0 }
fn default_taker_fee_bps() -> f64 { 4.0 }
fn default_taker_ratio() -> f64 { 0.7 }
fn default_impact_eta() -> f64 { 0.5 }
fn default_spread_multiplier() -> f64 { 0.05 }
fn default_max_participation() -> f64 { 0.10 }
fn default_max_hold() -> i32 { 120 }
fn default_capital() -> f64 { 10000.0 }

fn parse_config(json: &str) -> BacktestConfig {
    if json.is_empty() {
        return serde_json::from_str("{}").unwrap();
    }
    serde_json::from_str(json).unwrap_or_else(|_| serde_json::from_str("{}").unwrap())
}

// ── Component 1: pred_to_signal ──────────────────────────────

fn pred_to_signal_impl(
    y_pred: &[f64],
    deadzone: f64,
    min_hold: i32,
    zscore_window: i32,
    zscore_warmup: i32,
    long_only: bool,
    trend_follow: bool,
    trend_values: Option<&[f64]>,
    trend_threshold: f64,
    max_hold: i32,
) -> Vec<f64> {
    // Step 1: Rolling z-score → long_only clip → discretize (shared)
    let raw = zscore_discretize_array(
        y_pred, deadzone,
        zscore_window as usize, zscore_warmup as usize,
        long_only,
    );

    // Step 2: Min-hold + trend-hold enforcement (shared)
    enforce_hold_array(&raw, min_hold, trend_follow, trend_values, trend_threshold, max_hold)
}

// ── Component 2: compute_bear_mask ───────────────────────────

// compute_bear_mask is now in constraint_pipeline.rs

// ── Component 2b: prob_to_score ──────────────────────────────

fn prob_to_score(prob: f64, thresholds: &[[f64; 2]]) -> f64 {
    if thresholds.is_empty() {
        return if prob > 0.5 { -1.0 } else { 0.0 };
    }
    for t in thresholds {
        if prob > t[0] {
            return t[1];
        }
    }
    0.0
}

// ── Component 2c: apply_dd_breaker ───────────────────────────

fn apply_dd_breaker(signal: &mut [f64], closes: &[f64], dd_limit: f64, cooldown: i32) {
    let n = signal.len();
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut cool_remaining: i32 = 0;

    for i in 0..n {
        if cool_remaining > 0 {
            signal[i] = 0.0;
            cool_remaining -= 1;
            continue;
        }

        if i < closes.len() - 1 {
            let ret = (closes[i + 1] - closes[i]) / closes[i];
            equity *= 1.0 + signal[i] * ret;
        }
        if equity > peak {
            peak = equity;
        }
        let dd = (equity - peak) / peak;

        if dd < dd_limit {
            cool_remaining = cooldown;
            signal[i] = 0.0;
        }
    }
}

// ── Component 2d: apply_regime_switch ─────────────────────────

/// Post-processing: vol-adaptive sizing and DD circuit breaker.
/// Gate/bear logic is now handled in the single-pass enforce_hold_with_gate_array
/// inside run_backtest_impl, matching live semantics exactly.
fn apply_post_processing(
    signal: &mut [f64],
    closes: &[f64],
    vol_values: Option<&[f64]>,
    cfg: &BacktestConfig,
) {
    // Vol-adaptive sizing (shared)
    if cfg.vol_adaptive {
        if let Some(vv) = vol_values {
            vol_scale_array(signal, vv, cfg.vol_target);
        }
    }

    // DD circuit breaker (backtest-only — live uses real-time KillSwitch)
    if cfg.dd_breaker {
        apply_dd_breaker(signal, closes, cfg.dd_limit, cfg.dd_cooldown);
    }
}

// ── Component 3: compute_costs ───────────────────────────────

struct CostResult {
    total_cost: Vec<f64>,
    clipped_signal: Vec<f64>,
}

fn compute_costs_flat(signal: &[f64], cost_per_trade: f64) -> CostResult {
    let n = signal.len();
    let mut total_cost = vec![0.0_f64; n];
    let clipped_signal = signal.to_vec();

    let mut prev = 0.0_f64;
    for i in 0..n {
        let turnover = (signal[i] - prev).abs();
        total_cost[i] = turnover * cost_per_trade;
        prev = signal[i];
    }

    CostResult { total_cost, clipped_signal }
}

fn compute_costs_realistic(
    signal: &[f64],
    closes: &[f64],
    volumes: &[f64],
    volatility: &[f64],
    cfg: &BacktestConfig,
) -> CostResult {
    let n = signal.len();
    let mut fee_cost = vec![0.0_f64; n];
    let mut impact_cost = vec![0.0_f64; n];
    let mut spread_cost = vec![0.0_f64; n];
    let mut total_cost = vec![0.0_f64; n];
    let mut clipped_signal = vec![0.0_f64; n];

    let notional = cfg.capital;

    // Step 1: Compute raw turnover and max position change
    let mut turnover_raw = vec![0.0_f64; n];
    let mut max_pos_change = vec![0.0_f64; n];
    let mut prev_sig = 0.0_f64;
    let mut has_excess = false;

    for i in 0..n {
        turnover_raw[i] = (signal[i] - prev_sig).abs();
        prev_sig = signal[i];

        if closes[i] > 0.0 {
            max_pos_change[i] = (volumes[i] * cfg.max_participation * closes[i]) / notional;
        } else {
            max_pos_change[i] = 1e30;
        }
        if turnover_raw[i] > max_pos_change[i] {
            has_excess = true;
        }
    }

    // Step 2: Volume participation clipping
    if has_excess {
        clipped_signal[0] = signal[0].clamp(-max_pos_change[0], max_pos_change[0]);
        for i in 1..n {
            let delta = signal[i] - clipped_signal[i - 1];
            let md = max_pos_change[i];
            let delta_clipped = delta.clamp(-md, md);
            clipped_signal[i] = clipped_signal[i - 1] + delta_clipped;
        }
    } else {
        clipped_signal.copy_from_slice(signal);
    }

    // Recompute turnover from clipped signal
    let mut turnover = vec![0.0_f64; n];
    let mut prev_c = 0.0_f64;
    for i in 0..n {
        turnover[i] = (clipped_signal[i] - prev_c).abs();
        prev_c = clipped_signal[i];
    }

    // Step 3: Trading fees
    let blended_fee = (cfg.taker_ratio * cfg.taker_fee_bps
        + (1.0 - cfg.taker_ratio) * cfg.maker_fee_bps) / 10000.0;
    for i in 0..n {
        fee_cost[i] = turnover[i] * blended_fee;
    }

    // Step 4: Market impact (Almgren-Chriss sqrt)
    for i in 0..n {
        let safe_vol_notional = (volumes[i] * closes[i]).max(1.0);
        let participation = (turnover[i] * notional) / safe_vol_notional;
        let vol_val = volatility[i];
        let sigma_daily = if !vol_val.is_nan() && vol_val > 0.0 {
            vol_val * 24.0_f64.sqrt()
        } else {
            0.0
        };
        impact_cost[i] = cfg.impact_eta * sigma_daily * participation.max(0.0).sqrt();
    }

    // Step 5: Bid-ask spread
    for i in 0..n {
        let vol_val = volatility[i];
        let spread_bps = if vol_val.is_nan() { 0.0 } else { cfg.spread_multiplier * vol_val };
        spread_cost[i] = turnover[i] * spread_bps / 2.0;
    }

    // Total
    for i in 0..n {
        total_cost[i] = fee_cost[i] + impact_cost[i] + spread_cost[i];
    }

    CostResult { total_cost, clipped_signal }
}

// ── Component 4: simulate_trades ─────────────────────────────

struct TradeResult {
    net_pnl: Vec<f64>,
    equity: Vec<f64>,
}

fn simulate_trades(
    signal: &[f64],
    closes: &[f64],
    cost: &[f64],
    funding_rates: Option<&[f64]>,
    funding_ts: Option<&[i64]>,
    bar_timestamps: Option<&[i64]>,
    initial_capital: f64,
) -> TradeResult {
    let n_trade = std::cmp::min(signal.len(), closes.len().saturating_sub(1));
    let mut net_pnl = vec![0.0_f64; n_trade];
    let mut funding_cost = vec![0.0_f64; n_trade];
    let mut equity = vec![0.0_f64; n_trade + 1];
    equity[0] = initial_capital;

    // Funding: forward-scan merge
    let mut f_idx: usize = 0;
    let mut current_rate = 0.0_f64;

    let has_funding = funding_rates.is_some()
        && funding_ts.is_some()
        && bar_timestamps.is_some();

    for i in 0..n_trade {
        // Update funding rate
        if has_funding {
            let fr = funding_rates.unwrap();
            let ft = funding_ts.unwrap();
            let bt = bar_timestamps.unwrap();
            let ts = bt[i];
            while f_idx < fr.len() && ft[f_idx] <= ts {
                current_rate = fr[f_idx];
                f_idx += 1;
            }
            if signal[i] != 0.0 {
                funding_cost[i] = signal[i] * current_rate / 8.0;
            }
        }

        let ret = (closes[i + 1] - closes[i]) / closes[i];
        let gross = signal[i] * ret;
        net_pnl[i] = gross - cost[i] - funding_cost[i];
        equity[i + 1] = equity[i] * (1.0 + net_pnl[i]);
    }

    TradeResult { net_pnl, equity }
}

// ── Component 5: compute_metrics — see backtest_metrics.inc.rs ──

include!("backtest_metrics.inc.rs");

// ── Main entry: run_backtest ─────────────────────────────────

fn run_backtest_impl(
    timestamps: &[i64],
    closes: &[f64],
    volumes: Option<&[f64]>,
    vol_20: Option<&[f64]>,
    y_pred: &[f64],
    bear_probs: Option<&[f64]>,
    vol_values: Option<&[f64]>,
    funding_rates: Option<&[f64]>,
    funding_ts: Option<&[i64]>,
    trend_values: Option<&[f64]>,
    cfg: &BacktestConfig,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, BacktestMetrics) {
    let n = y_pred.len();

    // Step 1: Rolling z-score → long_only clip → discretize (shared)
    let raw = zscore_discretize_array(
        y_pred, cfg.deadzone,
        cfg.zscore_window as usize, cfg.zscore_warmup as usize,
        cfg.long_only,
    );

    // Step 2: Compute gate mask + gate scores (for unified single-pass)
    let needs_gate = cfg.monthly_gate || cfg.use_regime_switch;
    let gate_mask = if needs_gate {
        Some(compute_bear_mask(closes, cfg.ma_window as usize))
    } else {
        None
    };

    let gate_scores: Option<Vec<f64>> = if let Some(bp) = bear_probs {
        // Pre-compute bear model replacement scores
        Some(bp.iter().map(|&p| prob_to_score(p, &cfg.bear_thresholds)).collect())
    } else if cfg.monthly_gate {
        // Simple monthly gate: zero signal in bear regime (no bear model)
        None  // enforce_hold_with_gate_array defaults to 0.0
    } else {
        None
    };

    // Step 3: Single-pass min-hold + trend-hold + gate override
    // Matches live apply_signal_pipeline: min-hold runs first, then gate
    // overrides the output (bypassing min-hold). No re-min-hold second pass.
    let mut signal = if gate_mask.is_some() {
        enforce_hold_with_gate_array(
            &raw, cfg.min_hold,
            cfg.trend_follow, trend_values, cfg.trend_threshold, cfg.max_hold,
            gate_mask.as_deref(), gate_scores.as_deref(),
        )
    } else {
        enforce_hold_array(
            &raw, cfg.min_hold,
            cfg.trend_follow, trend_values, cfg.trend_threshold, cfg.max_hold,
        )
    };

    // Step 4: Vol-adaptive sizing + DD breaker
    if cfg.vol_adaptive || cfg.dd_breaker {
        apply_post_processing(&mut signal, closes, vol_values, cfg);
    }

    // Step 3: Cost computation
    let cost_result = if cfg.realistic_cost && volumes.is_some() && vol_20.is_some() {
        let cr = compute_costs_realistic(
            &signal, closes, volumes.unwrap(), vol_20.unwrap(), cfg,
        );
        // Update signal with clipped version
        signal.copy_from_slice(&cr.clipped_signal);
        cr
    } else {
        compute_costs_flat(&signal, cfg.cost_per_trade)
    };

    // Step 4: Trade simulation
    let trade = simulate_trades(
        &signal, closes, &cost_result.total_cost,
        funding_rates, funding_ts,
        Some(timestamps),
        cfg.capital,
    );

    // Step 5: Metrics
    let mut metrics = compute_metrics(
        &signal, &trade.net_pnl, &trade.equity,
        Some(timestamps),
    );

    // Set total_cost from cost_result
    let tc: f64 = cost_result.total_cost.iter().sum();
    metrics.total_cost = tc;

    // Truncate signal to n (it already is n)
    let _ = n;

    (signal, trade.equity, trade.net_pnl, metrics)
}

// ── PyO3 entry points — see backtest_pyo3.inc.rs ──

include!("backtest_pyo3.inc.rs");
