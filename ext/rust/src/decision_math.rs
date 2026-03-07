//! Decision math: position sizing and allocation constraints in Rust.
//!
//! Pure math functions — no state, no IO.
//! All use f64 arithmetic; Python side converts Decimal↔f64 at boundary.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Quantize to 3 decimal places, rounding down.
#[inline]
fn q3(x: f64) -> f64 {
    (x * 1000.0).floor() / 1000.0
}

/// Floor to lot size.
#[inline]
fn floor_to_lot(qty: f64, lot_size: f64) -> f64 {
    if lot_size <= 0.0 {
        return qty;
    }
    let steps = (qty / lot_size).floor();
    steps * lot_size
}

/// Fixed fraction position sizing.
///
/// target_qty = (equity * fraction * |weight|) / price
/// Optionally floors to lot_size.
///
/// Returns 0.0 if price <= 0.
#[pyfunction]
#[pyo3(signature = (equity, price, fraction, weight, lot_size=0.0))]
pub fn rust_fixed_fraction_qty(
    equity: f64,
    price: f64,
    fraction: f64,
    weight: f64,
    lot_size: f64,
) -> f64 {
    if price <= 0.0 {
        return 0.0;
    }
    let notional = equity * fraction * weight.abs();
    let qty = notional / price;
    if lot_size > 0.0 {
        floor_to_lot(qty, lot_size)
    } else {
        qty
    }
}

/// Volatility-adjusted position sizing.
///
/// target_qty = (equity * risk_fraction * weight) / (volatility * price)
/// Quantized to 3 decimal places (floor).
///
/// Returns 0.0 if price <= 0 or vol <= 0.
#[pyfunction]
#[pyo3(signature = (equity, price, volatility, risk_fraction, weight))]
pub fn rust_volatility_adjusted_qty(
    equity: f64,
    price: f64,
    volatility: f64,
    risk_fraction: f64,
    weight: f64,
) -> f64 {
    if price <= 0.0 || volatility <= 0.0 {
        return 0.0;
    }
    let risk_budget = equity * risk_fraction * weight;
    let qty = risk_budget / (volatility * price);
    q3(qty)
}

/// Apply allocation constraints: keep top-N by |weight|, renormalize.
///
/// Returns a new dict with at most max_positions entries,
/// weights renormalized so sum(|w|) = 1.
#[pyfunction]
#[pyo3(signature = (weights, max_positions))]
pub fn rust_apply_allocation_constraints(
    py: Python<'_>,
    weights: &Bound<'_, PyDict>,
    max_positions: usize,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);

    if weights.is_empty() || max_positions == 0 {
        return Ok(result.into_any().unbind());
    }

    // Collect (symbol, weight) pairs
    let mut items: Vec<(String, f64)> = Vec::new();
    for (k, v) in weights.iter() {
        let sym: String = k.extract()?;
        let w: f64 = v.extract::<f64>().or_else(|_| {
            v.str()
                .and_then(|s| s.to_str().map(|s| s.to_string()))
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
        })?;
        items.push((sym, w));
    }

    // Sort by |weight| descending
    items.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));

    // Keep top N
    items.truncate(max_positions);

    // Compute total |weight|
    let total: f64 = items.iter().map(|(_, w)| w.abs()).sum();

    if total <= 0.0 {
        for (sym, _) in &items {
            result.set_item(sym, 0.0)?;
        }
    } else {
        for (sym, w) in &items {
            result.set_item(sym, w / total)?;
        }
    }

    Ok(result.into_any().unbind())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_fraction_basic() {
        let qty = rust_fixed_fraction_qty(10000.0, 50000.0, 0.02, 1.0, 0.0);
        assert!((qty - 0.004).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_fraction_lot_size() {
        let qty = rust_fixed_fraction_qty(10000.0, 50000.0, 0.02, 1.0, 0.001);
        assert_eq!(qty, 0.004);
    }

    #[test]
    fn test_fixed_fraction_zero_price() {
        assert_eq!(rust_fixed_fraction_qty(10000.0, 0.0, 0.02, 1.0, 0.0), 0.0);
    }

    #[test]
    fn test_volatility_adjusted_basic() {
        let qty = rust_volatility_adjusted_qty(10000.0, 50000.0, 0.02, 0.02, 1.0);
        // 10000 * 0.02 * 1.0 / (0.02 * 50000) = 200 / 1000 = 0.2
        assert_eq!(qty, 0.2);
    }

    #[test]
    fn test_volatility_adjusted_quantize() {
        let qty = rust_volatility_adjusted_qty(10000.0, 30000.0, 0.015, 0.02, 1.0);
        // 10000 * 0.02 / (0.015 * 30000) = 200 / 450 = 0.4444...
        assert_eq!(qty, 0.444); // floor to 3 decimals
    }

    #[test]
    fn test_volatility_zero_vol() {
        assert_eq!(rust_volatility_adjusted_qty(10000.0, 50000.0, 0.0, 0.02, 1.0), 0.0);
    }
}
