//! Decision signal computation in Rust.
//!
//! Provides rebalance intent generation and feature signal scoring.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Compute rebalance intents from target weights and current positions.
///
/// For each symbol where |drift| >= drift_threshold, generates a rebalance
/// intent with delta qty = drift * equity / price.
///
/// Optionally filters by cost: if cost_bps * |drift| > cost_threshold, skip.
///
/// Returns list of dicts with: symbol, side, target_qty, drift.
#[pyfunction]
#[pyo3(signature = (target_weights, current_positions, prices, equity, drift_threshold=0.05, cost_bps=10.0, cost_threshold=0.0))]
pub fn rust_compute_rebalance_intents(
    py: Python<'_>,
    target_weights: &Bound<'_, PyDict>,
    current_positions: &Bound<'_, PyDict>,
    prices: &Bound<'_, PyDict>,
    equity: f64,
    drift_threshold: f64,
    cost_bps: f64,
    cost_threshold: f64,
) -> PyResult<PyObject> {
    let result = PyList::empty(py);

    if equity <= 0.0 {
        return Ok(result.into_any().unbind());
    }

    for (sym_obj, tw_obj) in target_weights.iter() {
        let sym: String = sym_obj.extract()?;
        let target_w: f64 = tw_obj.extract()?;

        // Get current weight: qty * price / equity
        let qty: f64 = current_positions
            .get_item(&sym)?
            .map(|v| v.extract::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);
        let price: f64 = prices
            .get_item(&sym)?
            .map(|v| v.extract::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);

        if price <= 0.0 {
            continue;
        }

        let current_w = (qty * price) / equity;
        let drift = target_w - current_w;

        if drift.abs() < drift_threshold {
            continue;
        }

        // Cost filter
        if cost_threshold > 0.0 {
            let cost_frac = cost_bps / 10000.0;
            let expected_cost = drift.abs() * cost_frac;
            if expected_cost > cost_threshold * drift.abs() {
                continue;
            }
        }

        let delta_notional = drift * equity;
        let delta_qty = (delta_notional / price).abs();
        let side = if delta_notional > 0.0 { "buy" } else { "sell" };

        let intent = PyDict::new(py);
        intent.set_item("symbol", &sym)?;
        intent.set_item("side", side)?;
        intent.set_item("target_qty", delta_qty)?;
        intent.set_item("drift", drift)?;
        result.append(intent)?;
    }

    Ok(result.into_any().unbind())
}

/// Compute feature-based trading signal from momentum, volatility, and VWAP ratio.
///
/// Returns (side, score, confidence) tuple:
/// - side: "buy", "sell", or "flat"
/// - score: [-1, 1] composite signal strength
/// - confidence: [0.2, 1.0] reduced by high volatility
#[pyfunction]
#[pyo3(signature = (momentum, volatility, vwap_ratio, momentum_threshold=0.001, vol_penalty_factor=2.0, vwap_weight=0.3))]
pub fn rust_compute_feature_signal(
    momentum: f64,
    volatility: f64,
    vwap_ratio: f64,
    momentum_threshold: f64,
    vol_penalty_factor: f64,
    vwap_weight: f64,
) -> (String, f64, f64) {
    // Base score from momentum
    let (side, raw_score) = if momentum.abs() < momentum_threshold {
        ("flat".to_string(), 0.0)
    } else if momentum > 0.0 {
        let s = (momentum / momentum_threshold).min(5.0) / 5.0;
        ("buy".to_string(), s)
    } else {
        let s = (momentum / momentum_threshold).max(-5.0) / 5.0;
        ("sell".to_string(), s)
    };

    // VWAP confirmation
    let vwap_bonus = if side != "flat" {
        let vwap_dev = vwap_ratio - 1.0;
        let confirms = (side == "buy" && vwap_dev > 0.0) || (side == "sell" && vwap_dev < 0.0);
        if confirms {
            vwap_weight * vwap_dev.abs()
        } else {
            -vwap_weight * vwap_dev.abs() * 0.5
        }
    } else {
        0.0
    };

    let score = raw_score + vwap_bonus;

    // Confidence: reduced by high volatility
    let confidence = if volatility > 0.0 {
        let penalty = (volatility * vol_penalty_factor * 100.0).min(0.8);
        (1.0 - penalty).max(0.2)
    } else {
        1.0
    };

    (side, score, confidence)
}

/// Compute rolling Sharpe ratio from a returns array.
///
/// Sharpe = mean(returns) / std(returns) * sqrt(annualize_factor)
/// Returns None if fewer than min_obs observations.
#[pyfunction]
#[pyo3(signature = (returns, window=60, annualize_factor=252.0, min_obs=10))]
pub fn rust_rolling_sharpe(
    returns: Vec<f64>,
    window: usize,
    annualize_factor: f64,
    min_obs: usize,
) -> Option<f64> {
    let n = returns.len();
    if n < min_obs {
        return None;
    }
    let start = if n > window { n - window } else { 0 };
    let slice = &returns[start..];
    let count = slice.len() as f64;
    let mean = slice.iter().sum::<f64>() / count;
    let var = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (count - 1.0).max(1.0);
    let std = var.max(0.0).sqrt();
    if std < 1e-12 {
        return Some(if mean > 0.0 { 99.0 } else { 0.0 });
    }
    Some(mean / std * annualize_factor.sqrt())
}

/// Compute maximum drawdown from an equity curve or returns array.
///
/// If `is_returns` is true, treats input as period returns and builds equity curve.
/// Otherwise treats input as absolute equity values.
#[pyfunction]
#[pyo3(signature = (values, is_returns=true))]
pub fn rust_max_drawdown(values: Vec<f64>, is_returns: bool) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut peak = if is_returns { 1.0 } else { values[0] };
    let mut eq = peak;
    let mut max_dd = 0.0_f64;

    for (i, &v) in values.iter().enumerate() {
        if is_returns {
            eq *= 1.0 + v;
        } else {
            eq = v;
            if i == 0 {
                continue;
            }
        }
        if eq > peak {
            peak = eq;
        }
        if peak > 0.0 {
            let dd = (peak - eq) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }
    }

    max_dd
}

/// Compute strategy weights based on method and performance data.
///
/// Methods:
/// - "equal": equal weight
/// - "sharpe": weight proportional to max(rolling_sharpe, 0)
/// - "inverse_vol": weight inversely proportional to volatility
///
/// performances: dict[str, dict] where inner dict has "returns" (list[f64])
/// min_weight, max_weight: bounds for clamping
///
/// Returns dict[str, f64] of normalized weights.
#[pyfunction]
#[pyo3(signature = (method, performances, min_weight=0.05, max_weight=0.5))]
pub fn rust_strategy_weights<'py>(
    py: Python<'py>,
    method: &str,
    performances: &Bound<'_, PyDict>,
    min_weight: f64,
    max_weight: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let n = performances.len();
    if n == 0 {
        return Ok(result);
    }

    let mut names: Vec<String> = Vec::with_capacity(n);
    let mut raw_weights: Vec<f64> = Vec::with_capacity(n);

    match method {
        "equal" => {
            let w = 1.0 / n as f64;
            for (k, _) in performances.iter() {
                names.push(k.extract()?);
                raw_weights.push(w);
            }
        }
        "sharpe" => {
            for (k, v) in performances.iter() {
                names.push(k.extract()?);
                let inner: &Bound<'_, PyDict> = v.downcast()?;
                let returns: Vec<f64> = inner
                    .get_item("returns")?
                    .map(|v| v.extract().unwrap_or_default())
                    .unwrap_or_default();
                let sr = compute_sharpe(&returns, 60, 252.0, 10);
                raw_weights.push(sr.unwrap_or(0.0).max(0.0));
            }
            let total: f64 = raw_weights.iter().sum();
            if total > 0.0 {
                for w in raw_weights.iter_mut() {
                    *w /= total;
                }
            } else {
                let eq = 1.0 / n as f64;
                for w in raw_weights.iter_mut() {
                    *w = eq;
                }
            }
        }
        "inverse_vol" => {
            for (k, v) in performances.iter() {
                names.push(k.extract()?);
                let inner: &Bound<'_, PyDict> = v.downcast()?;
                let returns: Vec<f64> = inner
                    .get_item("returns")?
                    .map(|v| v.extract().unwrap_or_default())
                    .unwrap_or_default();
                if returns.len() < 10 {
                    raw_weights.push(1.0);
                } else {
                    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                        / (returns.len() as f64 - 1.0).max(1.0);
                    let vol = var.max(0.0).sqrt();
                    raw_weights.push(1.0 / vol.max(1e-8));
                }
            }
            let total: f64 = raw_weights.iter().sum();
            if total > 0.0 {
                for w in raw_weights.iter_mut() {
                    *w /= total;
                }
            } else {
                let eq = 1.0 / n as f64;
                for w in raw_weights.iter_mut() {
                    *w = eq;
                }
            }
        }
        _ => {
            let w = 1.0 / n as f64;
            for (k, _) in performances.iter() {
                names.push(k.extract()?);
                raw_weights.push(w);
            }
        }
    }

    // Clamp weights and renormalize
    for w in raw_weights.iter_mut() {
        *w = w.max(min_weight).min(max_weight);
    }
    let total: f64 = raw_weights.iter().sum();
    if total > 0.0 {
        for w in raw_weights.iter_mut() {
            *w /= total;
        }
    }

    for (name, w) in names.iter().zip(raw_weights.iter()) {
        result.set_item(name, *w)?;
    }

    Ok(result)
}

fn compute_sharpe(returns: &[f64], window: usize, annualize: f64, min_obs: usize) -> Option<f64> {
    let n = returns.len();
    if n < min_obs {
        return None;
    }
    let start = if n > window { n - window } else { 0 };
    let slice = &returns[start..];
    let count = slice.len() as f64;
    let mean = slice.iter().sum::<f64>() / count;
    let var = slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (count - 1.0).max(1.0);
    let std = var.max(0.0).sqrt();
    if std < 1e-12 {
        return Some(if mean > 0.0 { 99.0 } else { 0.0 });
    }
    Some(mean / std * annualize.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_signal_flat() {
        let (side, score, conf) = rust_compute_feature_signal(
            0.0005, 0.0, 1.0, 0.001, 2.0, 0.3,
        );
        assert_eq!(side, "flat");
        assert_eq!(score, 0.0);
        assert_eq!(conf, 1.0);
    }

    #[test]
    fn test_feature_signal_buy() {
        let (side, score, conf) = rust_compute_feature_signal(
            0.003, 0.01, 1.01, 0.001, 2.0, 0.3,
        );
        assert_eq!(side, "buy");
        assert!(score > 0.0);
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_feature_signal_sell() {
        let (side, score, _conf) = rust_compute_feature_signal(
            -0.003, 0.01, 0.99, 0.001, 2.0, 0.3,
        );
        assert_eq!(side, "sell");
        assert!(score < 0.0);
    }

    #[test]
    fn test_feature_signal_high_vol_low_confidence() {
        let (_, _, conf) = rust_compute_feature_signal(
            0.003, 0.1, 1.0, 0.001, 2.0, 0.3,
        );
        // 0.1 * 2.0 * 100 = 20 → capped at 0.8 penalty → conf = 0.2
        assert!((conf - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_feature_signal_vwap_confirms_buy() {
        let (_, score_with, _) = rust_compute_feature_signal(
            0.003, 0.0, 1.05, 0.001, 2.0, 0.3,
        );
        let (_, score_without, _) = rust_compute_feature_signal(
            0.003, 0.0, 1.0, 0.001, 2.0, 0.3,
        );
        assert!(score_with > score_without);
    }

    #[test]
    fn test_feature_signal_vwap_contradicts_buy() {
        let (_, score_contra, _) = rust_compute_feature_signal(
            0.003, 0.0, 0.95, 0.001, 2.0, 0.3,
        );
        let (_, score_neutral, _) = rust_compute_feature_signal(
            0.003, 0.0, 1.0, 0.001, 2.0, 0.3,
        );
        assert!(score_contra < score_neutral);
    }

    #[test]
    fn test_rolling_sharpe_positive() {
        let rets: Vec<f64> = (0..30).map(|_| 0.01).collect();
        let sr = rust_rolling_sharpe(rets, 60, 252.0, 10).unwrap();
        assert!(sr > 0.0);
    }

    #[test]
    fn test_rolling_sharpe_not_enough_data() {
        let rets: Vec<f64> = vec![0.01; 5];
        assert!(rust_rolling_sharpe(rets, 60, 252.0, 10).is_none());
    }

    #[test]
    fn test_max_drawdown_basic() {
        let rets = vec![0.1, 0.1, -0.3, 0.05];
        let dd = rust_max_drawdown(rets, true);
        assert!(dd > 0.2);
    }

    #[test]
    fn test_max_drawdown_no_loss() {
        let rets = vec![0.01, 0.02, 0.01];
        let dd = rust_max_drawdown(rets, true);
        assert_eq!(dd, 0.0);
    }

    #[test]
    fn test_max_drawdown_empty() {
        assert_eq!(rust_max_drawdown(vec![], true), 0.0);
    }
}
