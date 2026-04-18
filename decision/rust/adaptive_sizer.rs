//! Adaptive position sizer: equity-tier + IC-health + regime-aware sizing in Rust.
//!
//! Mirrors `AdaptivePositionSizer.target_qty()` from `decision/sizing/adaptive.py`.
//! Pure math — no state, no IO.

use pyo3::prelude::*;

// ── Tier weights (must match Python _TIER_WEIGHTS exactly) ──────────

/// Lookup base cap for (tier, runner_key).  Returns DEFAULT_CAP if not found.
#[inline]
fn tier_cap(tier: &str, runner_key: &str) -> f64 {
    const DEFAULT_CAP: f64 = 0.15;
    // Must match Python _TIER_WEIGHTS in decision/sizing/adaptive.py
    match tier {
        // Micro: equity < 500 — D13 portfolio config (commit 929270c).
        // 12-month OOS on $400: joint Sharpe +6.36, Return +451%, MaxDD -18.2%.
        // BTC 2x + ETH 1x at OKX_LEVERAGE=10 — symmetric (0.075,0.075) was
        // strictly worse (Sharpe +5.95).  ec8bb15 had reverted ETH to 0.65
        // by accident; restored 2026-04-18 after the oversize-short incident.
        "micro" => match runner_key {
            "BTCUSDT" => 0.20,    // 2x effective: 0.20 × 10
            "ETHUSDT" => 0.10,    // 1x effective: 0.10 × 10  (D13)
            "SOLUSDT" => 0.40,
            "BTCUSDT_4h" => 0.0,  // signal_only
            "ETHUSDT_4h" => 0.0,
            _ => DEFAULT_CAP,
        },
        "medium" => match runner_key {
            "BTCUSDT" => 0.45,
            "ETHUSDT" => 0.45,
            "SOLUSDT" => 0.30,
            "BTCUSDT_4h" => 0.0,  // signal_only
            "ETHUSDT_4h" => 0.0,
            _ => DEFAULT_CAP,
        },
        "large" => match runner_key {
            "BTCUSDT" => 0.10,
            "ETHUSDT" => 0.10,
            "SOLUSDT" => 0.08,
            "BTCUSDT_4h" => 0.0,
            "ETHUSDT_4h" => 0.0,
            _ => DEFAULT_CAP,
        },
        _ => DEFAULT_CAP,
    }
}

/// Determine equity tier.
#[inline]
fn equity_tier(equity: f64) -> &'static str {
    if equity < 500.0 {
        "micro"
    } else if equity < 10_000.0 {
        "medium"
    } else {
        "large"
    }
}

/// Floor `size` to the nearest `step_size` increment.
///
/// Matches Python's `Decimal.quantize(ROUND_DOWN)` behavior:
/// compute number of decimal places from step_size, then truncate.
#[inline]
fn round_to_step(size: f64, step_size: f64) -> f64 {
    if step_size <= 0.0 {
        return size;
    }
    let decimals = (-step_size.log10()).floor().max(0.0) as u32;
    let factor = 10_f64.powi(decimals as i32);
    (size * factor).floor() / factor
}

/// Compute target position quantity (Rust fast path).
///
/// Mirrors `AdaptivePositionSizer.target_qty()` exactly:
///   1. Equity tier → base cap
///   2. Regime discount (×0.6 if inactive)
///   3. IC health scaling
///   4. Notional → quantity (× z_scale)
///   5. Clamp [min_size, max_qty]
///   6. Round down to step_size
///
/// NaN guards: returns `round_to_step(min_size, step_size)` on NaN equity/price.
#[pyfunction]
#[pyo3(signature = (runner_key, equity, price, step_size, min_size, max_qty, weight, leverage, ic_scale, regime_active, z_scale))]
pub fn rust_adaptive_target_qty(
    runner_key: &str,
    equity: f64,
    price: f64,
    step_size: f64,
    min_size: f64,
    max_qty: f64,
    weight: f64,
    leverage: f64,
    ic_scale: f64,
    regime_active: bool,
    z_scale: f64,
) -> f64 {
    // NaN guard
    if equity.is_nan() || price.is_nan() {
        return round_to_step(min_size, step_size);
    }

    // Early exit on invalid equity/price
    if equity <= 0.0 || price <= 0.0 {
        return round_to_step(min_size, step_size);
    }

    // 1. Tier-based cap
    let tier = equity_tier(equity);
    let mut base_cap = tier_cap(tier, runner_key);

    // 2. Regime discount
    if !regime_active {
        base_cap *= 0.6;
    }

    // 3. IC health scaling
    let per_sym_cap = base_cap * ic_scale;

    // 4. Notional → quantity
    let notional = equity * per_sym_cap * leverage * weight;
    let mut size = notional / price * z_scale;

    // 5. Clamp — respect cap=0.0 (disabled symbol in this tier)
    if base_cap > 0.0 {
        if size < min_size {
            size = min_size;
        }
    } else {
        size = 0.0;
    }
    if max_qty > 0.0 && size > max_qty {
        size = max_qty;
    }

    // 6. Round to step
    round_to_step(size, step_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_to_step_basic() {
        assert_eq!(round_to_step(0.0155, 0.001), 0.015);
        assert_eq!(round_to_step(0.0199, 0.001), 0.019);
        assert_eq!(round_to_step(1.999, 0.01), 1.99);
    }

    #[test]
    fn test_round_to_step_zero() {
        assert_eq!(round_to_step(1.234, 0.0), 1.234);
    }

    #[test]
    fn test_equity_tier() {
        assert_eq!(equity_tier(100.0), "micro");
        assert_eq!(equity_tier(499.9), "micro");
        assert_eq!(equity_tier(500.0), "medium");
        assert_eq!(equity_tier(9999.9), "medium");
        assert_eq!(equity_tier(10_000.0), "large");
    }

    #[test]
    fn test_tier_cap_known_keys() {
        // D13 micro tier: BTC 2x + ETH 1x at OKX_LEVERAGE=10
        assert!((tier_cap("micro", "BTCUSDT") - 0.20).abs() < 1e-9);
        assert!((tier_cap("micro", "ETHUSDT") - 0.10).abs() < 1e-9);
        assert_eq!(tier_cap("medium", "BTCUSDT"), 0.45);
        assert_eq!(tier_cap("large", "ETHUSDT"), 0.10);
        // 4h is signal_only — no position
        assert_eq!(tier_cap("medium", "BTCUSDT_4h"), 0.0);
    }

    #[test]
    fn test_tier_cap_sol() {
        // SOLUSDT now has explicit entries in all tiers.
        assert_eq!(tier_cap("micro", "SOLUSDT"), 0.40);
        assert_eq!(tier_cap("medium", "SOLUSDT"), 0.30);
        assert_eq!(tier_cap("large", "SOLUSDT"), 0.08);
        assert_eq!(tier_cap("unknown", "BTCUSDT"), 0.15);
    }

    #[test]
    fn test_micro_tier_reaches_btc_min_lot_at_10x() {
        // D13 balanced config: micro BTC cap 0.20 × 10x leverage.
        // Must reach the OKX minSz (0.0001 BTC = 0.01 contracts).
        // 397 × 0.20 × 10 / 72900 = 0.01089 BTC → 108 × minSz.
        let qty = rust_adaptive_target_qty(
            "BTCUSDT", 397.0, 72900.0, 0.0001, 0.0001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        assert!(qty >= 0.0001, "micro BTC must reach 0.0001 BTC min lot at 10x, got {qty}");
    }

    #[test]
    fn test_basic_sizing_micro() {
        // D13 micro BTC cap=0.20, lev=10, ic=1, z=1, regime=active:
        // notional = 400 * 0.20 * 10 * 1 = 800
        // size = 800 / 60000 ≈ 0.01333 → round_to_step(0.001) = 0.013
        let qty = rust_adaptive_target_qty(
            "BTCUSDT", 400.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        assert_eq!(qty, 0.013);
    }

    #[test]
    fn test_zero_equity() {
        let qty = rust_adaptive_target_qty(
            "BTCUSDT", 0.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        assert_eq!(qty, 0.001);
    }

    #[test]
    fn test_zero_price() {
        let qty = rust_adaptive_target_qty(
            "BTCUSDT", 5000.0, 0.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        assert_eq!(qty, 0.001);
    }

    #[test]
    fn test_nan_equity() {
        let qty = rust_adaptive_target_qty(
            "BTCUSDT", f64::NAN, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        assert_eq!(qty, 0.001);
    }

    #[test]
    fn test_nan_price() {
        let qty = rust_adaptive_target_qty(
            "BTCUSDT", 5000.0, f64::NAN, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        assert_eq!(qty, 0.001);
    }

    #[test]
    fn test_regime_inactive_reduces() {
        let active = rust_adaptive_target_qty(
            "BTCUSDT", 2000.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        let inactive = rust_adaptive_target_qty(
            "BTCUSDT", 2000.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, false, 1.0,
        );
        assert!(active > inactive);
    }

    #[test]
    fn test_ic_scaling() {
        // Use ETHUSDT (1h primary, cap > 0) — 4h runners are now
        // signal_only (cap=0.0) so they floor to min_size regardless
        // of ic_scale and cannot demonstrate scaling.
        let green = rust_adaptive_target_qty(
            "ETHUSDT", 1000.0, 3000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.2, true, 1.0,
        );
        let red = rust_adaptive_target_qty(
            "ETHUSDT", 1000.0, 3000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 0.4, true, 1.0,
        );
        assert!(green > red);
    }

    #[test]
    fn test_max_qty_clamp() {
        // Use BTCUSDT (cap>0) so clamp is actually exercised —
        // BTCUSDT_4h has cap=0.0 so qty would be min_size, not hit the clamp.
        let qty = rust_adaptive_target_qty(
            "BTCUSDT", 50000.0, 60000.0, 0.001, 0.001, 0.01,
            1.0, 10.0, 1.0, true, 1.0,
        );
        assert!(qty <= 0.01);
    }

    #[test]
    fn test_z_scale() {
        let base = rust_adaptive_target_qty(
            "BTCUSDT", 5000.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        let scaled = rust_adaptive_target_qty(
            "BTCUSDT", 5000.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 2.0,
        );
        // z_scale=2.0 should give ~2x the quantity
        assert!(scaled > base);
    }
}
