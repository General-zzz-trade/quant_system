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

    // ── Live-equivalent sizing (D10 #64) ──────────────────────────
    // Previously ``simulate_trades`` used the discretized signal
    // (-1/0/+1) as the literal position fraction of capital.  That is
    // *not* what AdaptivePositionSizer does in live: live notional =
    // equity × tier_cap × leverage × ic_scale × z_scale × regime_mult.
    //
    // These knobs replicate the live path inside backtest so PnL
    // scales comparably.  Default 1.0 preserves legacy behaviour for
    // backward compatibility with existing research scripts.
    #[serde(default = "default_sizer_fraction")]
    sizer_tier_cap: f64,        // matches _TIER_WEIGHTS entry (micro=0.65, medium=0.45)
    #[serde(default = "default_sizer_leverage")]
    sizer_leverage: f64,        // 3.0 live, 10.0 demo
    #[serde(default = "default_sizer_fraction")]
    sizer_ic_scale: f64,        // GREEN=1.2 YELLOW=0.8 RED=0.4 — treated as constant per run
    #[serde(default)]
    sizer_regime_gated: bool,   // if true, reduce pos when vol_ma_ratio > 1.15
    #[serde(default = "default_regime_mult_mid")]
    sizer_regime_mid_mult: f64, // size mult when 0.85 < vol_ratio ≤ 1.15
    #[serde(default = "default_regime_mult_high")]
    sizer_regime_high_mult: f64, // size mult when vol_ratio > 1.15

    // ── Latency slippage (D10 #65) ─────────────────────────────────
    #[serde(default)]
    latency_ms: f64,            // simulated order→fill delay
    #[serde(default = "default_latency_drift_frac")]
    latency_drift_frac: f64,    // fraction of next-bar range charged as slip

    // ── State restart simulation (D10 #66) ─────────────────────────
    // Periodically zero out the signal for `zscore_warmup` bars to
    // model the real-world effect of alpha_main process restarts on
    // the live rolling z-score window.  0 = disabled (default).
    #[serde(default)]
    restart_every_bars: i32,
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

// Live-equivalent sizer defaults. ``1.0`` keeps the legacy
// ``pos=signal*1*1*1`` behaviour for scripts that do not opt in.
fn default_sizer_fraction() -> f64 { 1.0 }
fn default_sizer_leverage() -> f64 { 1.0 }
fn default_regime_mult_mid() -> f64 { 1.0 }
fn default_regime_mult_high() -> f64 { 1.0 }
// Default latency slip fraction of the next-bar range.
// 0.0 = no slippage (legacy); 0.3 ≈ 300ms at typical 1s bar feed.
fn default_latency_drift_frac() -> f64 { 0.0 }

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
    effective_signal: Vec<f64>,  // signal after sizer + regime scaling
}

/// Rolling std of log returns over ``win`` bars.  Used to detect vol
/// regime inside the backtest so the sizer ``regime_gated`` flag can
/// replicate the live ``vol_ma_ratio_5_20`` behaviour.
fn rolling_std(xs: &[f64], win: usize) -> Vec<f64> {
    let n = xs.len();
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let lo = if i + 1 > win { i + 1 - win } else { 0 };
        let slice = &xs[lo..=i];
        if slice.len() < 2 {
            out[i] = 0.0;
            continue;
        }
        let mean: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
        let var: f64 = slice.iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>() / (slice.len() - 1) as f64;
        out[i] = var.sqrt();
    }
    out
}

/// Live-equivalent ``vol_ma_ratio_5_20``: ratio of the 5-bar log-return
/// stdev over the 20-bar log-return stdev.  Values ~1.0 = stable,
/// >1.15 = vol spike (strategy's "high_vol" bucket).
fn compute_vol_ratio(closes: &[f64]) -> Vec<f64> {
    let n = closes.len();
    let mut logret = vec![0.0_f64; n];
    for i in 1..n {
        if closes[i - 1] > 0.0 {
            logret[i] = (closes[i] / closes[i - 1]).ln();
        }
    }
    let vol5 = rolling_std(&logret, 5);
    let vol20 = rolling_std(&logret, 20);
    let mut out = vec![1.0_f64; n];
    for i in 0..n {
        if vol20[i] > 1e-9 {
            out[i] = vol5[i] / vol20[i];
        }
    }
    out
}

fn simulate_trades(
    signal: &[f64],
    closes: &[f64],
    cost: &[f64],
    funding_rates: Option<&[f64]>,
    funding_ts: Option<&[i64]>,
    bar_timestamps: Option<&[i64]>,
    cfg: &BacktestConfig,
) -> TradeResult {
    let n_trade = std::cmp::min(signal.len(), closes.len().saturating_sub(1));
    let mut net_pnl = vec![0.0_f64; n_trade];
    let mut funding_cost = vec![0.0_f64; n_trade];
    let mut equity = vec![0.0_f64; n_trade + 1];
    let mut effective_signal = vec![0.0_f64; n_trade];
    equity[0] = cfg.capital;

    // Funding: forward-scan merge
    let mut f_idx: usize = 0;
    let mut current_rate = 0.0_f64;

    let has_funding = funding_rates.is_some()
        && funding_ts.is_some()
        && bar_timestamps.is_some();

    // Pre-compute the live-equivalent notional multiplier:
    //   pos_fraction = signal × tier_cap × leverage × ic_scale × regime
    // ic_scale is treated as a constant for the run (backtest has no
    // time-varying IC oracle) — callers override it for stress tests.
    let base_notional_mult = cfg.sizer_tier_cap * cfg.sizer_leverage * cfg.sizer_ic_scale;

    // Vol regime detection — only if the caller opted in.  Computing
    // the 5/20-bar std is cheap and is shared across the bar loop below.
    let vol_ratio: Option<Vec<f64>> = if cfg.sizer_regime_gated {
        Some(compute_vol_ratio(closes))
    } else {
        None
    };

    let mut prev_pos_frac = 0.0_f64;

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
        }

        // Step 1: Apply sizer chain — produces live-comparable notional
        let mut pos_frac = signal[i] * base_notional_mult;

        // Step 2: Regime gating — when opted in, reduce size in
        // mid/high vol regimes.  Mirrors the D10 proposal to protect
        // the strategy from its documented mid/high-vol losses.
        if let Some(ref vr) = vol_ratio {
            let ratio = vr[i];
            if ratio > 1.15 {
                pos_frac *= cfg.sizer_regime_high_mult;
            } else if ratio > 0.85 {
                pos_frac *= cfg.sizer_regime_mid_mult;
            }
            // Low vol (<=0.85) keeps full size.
        }

        effective_signal[i] = pos_frac;

        // Step 3: Funding cost scales with *actual* position size
        if has_funding && pos_frac != 0.0 {
            // Funding rate is per 8h; backtest bars are 1h so divide by 8.
            funding_cost[i] = pos_frac * current_rate / 8.0;
        }

        // Step 4: Gross return on the effective notional fraction.
        let ret = (closes[i + 1] - closes[i]) / closes[i];
        let gross = pos_frac * ret;

        // Step 5: Scale the already-computed cost by the same notional
        // multiplier applied to gross.  The upstream cost_result was
        // built from the raw discretized signal (±1) so if we don't
        // scale here, leveraging up via sizer_tier_cap/leverage/ic
        // appears artificially free.  This keeps fee/impact/spread
        // proportional to real notional traded.
        let scaled_cost = cost[i] * base_notional_mult.abs();

        // Step 6: Latency slip — charged on turnover, proportional to
        // the absolute next-bar move (price moved this much during the
        // delay between decision and fill).  This is symmetric (always
        // adverse) because the market is not on our side.
        let latency_cost = if cfg.latency_drift_frac > 0.0 {
            let pos_delta = (pos_frac - prev_pos_frac).abs();
            let next_move = ((closes[i + 1] - closes[i]) / closes[i]).abs();
            pos_delta * cfg.latency_drift_frac * next_move
        } else {
            0.0
        };

        net_pnl[i] = gross - scaled_cost - funding_cost[i] - latency_cost;
        equity[i + 1] = equity[i] * (1.0 + net_pnl[i]);
        prev_pos_frac = pos_frac;
    }

    TradeResult { net_pnl, equity, effective_signal }
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

    // Step 4b: State restart simulation (D10 #66).
    //
    // Real production experiences alpha_main process restarts on model
    // hot-reload, weekly retrain SIGHUP, systemd restart, and ad-hoc
    // manual intervention.  Each restart clears the rolling z-score
    // window (pre-OnlineRidge-checkpoint era) or at minimum warms up
    // from a batch-synced buffer, taking ``zscore_warmup`` bars before
    // discretize() starts emitting non-zero signal.  This section
    // replicates that effect by forcing signal=0 for zscore_warmup bars
    // starting at each restart boundary.
    //
    // Observed 2026-04-11 live: the 22:55 UTC restart saw ETH z drop
    // from +2.33 (pre) to +0.62 (post) because the batch-sync buffer's
    // mean/std were computed over different recent data than the live
    // rolling window.  This switch exposes strategies that depend on
    // persistent high z-scores; those strategies will lose alpha after
    // each restart and the backtest should reflect that loss.
    if cfg.restart_every_bars > 0 {
        let period = cfg.restart_every_bars as usize;
        let warmup = cfg.zscore_warmup as usize;
        let n_bars = signal.len();
        let mut next_restart = period;
        while next_restart < n_bars {
            let end = std::cmp::min(next_restart + warmup, n_bars);
            for k in next_restart..end {
                signal[k] = 0.0;
            }
            next_restart += period;
        }
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

    // Step 4: Trade simulation with live-equivalent sizing
    let trade = simulate_trades(
        &signal, closes, &cost_result.total_cost,
        funding_rates, funding_ts,
        Some(timestamps),
        cfg,
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
