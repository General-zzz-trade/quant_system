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
    match tier {
        "small" => match runner_key {
            "BTCUSDT" => 0.25,
            "ETHUSDT" => 0.25,
            "BTCUSDT_4h" => 0.35,
            "ETHUSDT_4h" => 0.30,
            _ => DEFAULT_CAP,
        },
        "medium" => match runner_key {
            "BTCUSDT" => 0.18,
            "ETHUSDT" => 0.18,
            "BTCUSDT_4h" => 0.25,
            "ETHUSDT_4h" => 0.20,
            _ => DEFAULT_CAP,
        },
        "large" => match runner_key {
            "BTCUSDT" => 0.12,
            "ETHUSDT" => 0.12,
            "BTCUSDT_4h" => 0.18,
            "ETHUSDT_4h" => 0.15,
            _ => DEFAULT_CAP,
        },
        _ => DEFAULT_CAP,
    }
}

/// Determine equity tier.
#[inline]
fn equity_tier(equity: f64) -> &'static str {
    if equity < 500.0 {
        "small"
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

    // 5. Clamp
    if size < min_size {
        size = min_size;
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
        assert_eq!(equity_tier(100.0), "small");
        assert_eq!(equity_tier(499.9), "small");
        assert_eq!(equity_tier(500.0), "medium");
        assert_eq!(equity_tier(9999.9), "medium");
        assert_eq!(equity_tier(10_000.0), "large");
    }

    #[test]
    fn test_tier_cap_known_keys() {
        assert_eq!(tier_cap("small", "BTCUSDT_4h"), 0.35);
        assert_eq!(tier_cap("medium", "BTCUSDT"), 0.18);
        assert_eq!(tier_cap("large", "ETHUSDT"), 0.12);
    }

    #[test]
    fn test_tier_cap_fallback() {
        assert_eq!(tier_cap("small", "SOLUSDT"), 0.15);
        assert_eq!(tier_cap("unknown", "BTCUSDT"), 0.15);
    }

    #[test]
    fn test_basic_sizing_small() {
        // small tier, BTCUSDT_4h cap=0.35, lev=10, weight=1, ic=1, z=1
        // notional = 400 * 0.35 * 10 * 1 = 1400
        // size = 1400 / 60000 = 0.02333... → round_to_step(0.001) = 0.023
        let qty = rust_adaptive_target_qty(
            "BTCUSDT_4h", 400.0, 60000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        assert_eq!(qty, 0.023);
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
        let green = rust_adaptive_target_qty(
            "ETHUSDT_4h", 1000.0, 3000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 1.2, true, 1.0,
        );
        let red = rust_adaptive_target_qty(
            "ETHUSDT_4h", 1000.0, 3000.0, 0.001, 0.001, 0.0,
            1.0, 10.0, 0.4, true, 1.0,
        );
        assert!(green > red);
    }

    #[test]
    fn test_max_qty_clamp() {
        let qty = rust_adaptive_target_qty(
            "BTCUSDT_4h", 50000.0, 60000.0, 0.001, 0.001, 0.01,
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
