//! Portfolio allocation constraint math in Rust.
//!
//! Replaces Decimal-heavy Python constraint loops with f64 math.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Allocate portfolio: apply weights -> notional -> qty, with constraints.
///
/// Args:
///   target_weights: dict[str, f64] — signed weights per symbol
///   prices: dict[str, f64] — mark prices
///   equity: f64 — account equity
///   current_qty: dict[str, f64] — current position qty (signed)
///   max_weight: Optional[f64] — per-symbol absolute weight cap
///   max_notional_per_symbol: Optional[f64] — per-symbol notional cap
///   max_gross_leverage: Optional[f64] — portfolio gross leverage cap
///   turnover_cap: Optional[f64] — max turnover (delta_notional / equity)
///   allow_short: bool — whether to allow negative weights
///
/// Returns: dict[str, {weight, notional, qty}]
#[pyfunction]
#[pyo3(signature = (target_weights, prices, equity, current_qty=None, max_weight=None, max_notional_per_symbol=None, max_gross_leverage=None, turnover_cap=None, allow_short=true))]
pub fn rust_allocate_portfolio<'py>(
    py: Python<'py>,
    target_weights: &Bound<'_, PyDict>,
    prices: &Bound<'_, PyDict>,
    equity: f64,
    current_qty: Option<&Bound<'_, PyDict>>,
    max_weight: Option<f64>,
    max_notional_per_symbol: Option<f64>,
    max_gross_leverage: Option<f64>,
    turnover_cap: Option<f64>,
    allow_short: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);

    if equity <= 0.0 {
        return Ok(result);
    }

    // 1) Parse weights and apply short filter + max_weight
    let mut symbols: Vec<String> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    for (k, v) in target_weights.iter() {
        let sym: String = k.extract()?;
        let mut w: f64 = v.extract()?;

        if !allow_short && w < 0.0 {
            w = 0.0;
        }

        if let Some(cap) = max_weight {
            if w.abs() > cap {
                w = w.signum() * cap;
            }
        }

        symbols.push(sym);
        weights.push(w);
    }

    // 2) Extract prices once, compute notional and qty
    let mut px_vec: Vec<f64> = Vec::with_capacity(symbols.len());
    let mut notionals: Vec<f64> = Vec::with_capacity(symbols.len());
    let mut qtys: Vec<f64> = Vec::with_capacity(symbols.len());

    for (i, sym) in symbols.iter().enumerate() {
        let px: f64 = prices
            .get_item(sym)?
            .map(|v| v.extract::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);
        px_vec.push(px);
        if px <= 0.0 {
            notionals.push(0.0);
            qtys.push(0.0);
            continue;
        }
        let w = weights[i];
        notionals.push(w.abs() * equity);
        qtys.push((w * equity) / px);
    }

    // 3) Gross leverage cap
    if let Some(max_lev) = max_gross_leverage {
        if max_lev > 0.0 {
            let gross: f64 = notionals.iter().sum();
            let lev = gross / equity;
            if lev > max_lev && gross > 0.0 {
                let scale = max_lev / lev;
                for i in 0..symbols.len() {
                    weights[i] *= scale;
                    notionals[i] *= scale;
                    qtys[i] *= scale;
                }
            }
        }
    }

    // 4) Per-symbol notional cap
    if let Some(max_not) = max_notional_per_symbol {
        if max_not > 0.0 {
            for i in 0..symbols.len() {
                if notionals[i] > max_not && px_vec[i] > 0.0 {
                    let sign = weights[i].signum();
                    notionals[i] = max_not;
                    weights[i] = sign * max_not / equity;
                    qtys[i] = (sign * max_not) / px_vec[i];
                }
            }
        }
    }

    // 5) Turnover cap
    if let Some(tc) = turnover_cap {
        if tc > 0.0 {
            if let Some(cq) = current_qty {
                // Extract current quantities once
                let mut cur_qtys: Vec<f64> = Vec::with_capacity(symbols.len());
                for sym in symbols.iter() {
                    let cur_q: f64 = cq
                        .get_item(sym)?
                        .map(|v| v.extract::<f64>().unwrap_or(0.0))
                        .unwrap_or(0.0);
                    cur_qtys.push(cur_q);
                }

                let mut total_delta = 0.0_f64;
                for i in 0..symbols.len() {
                    let cn = cur_qtys[i].abs() * px_vec[i];
                    total_delta += (notionals[i] - cn).abs();
                }
                let turnover = if equity > 0.0 { total_delta / equity } else { 0.0 };
                if turnover > tc && total_delta > 0.0 {
                    let scale = tc / turnover;
                    for i in 0..symbols.len() {
                        if px_vec[i] <= 0.0 {
                            continue;
                        }
                        let cur_signed_not = cur_qtys[i] * px_vec[i];
                        let tgt_signed_not = qtys[i] * px_vec[i];
                        let delta = tgt_signed_not - cur_signed_not;
                        let new_signed_not = cur_signed_not + delta * scale;
                        qtys[i] = new_signed_not / px_vec[i];
                        notionals[i] = new_signed_not.abs();
                        weights[i] = new_signed_not / equity;
                    }
                }
            }
        }
    }

    // 6) Build result dict
    for i in 0..symbols.len() {
        let entry = PyDict::new(py);
        entry.set_item("weight", weights[i])?;
        entry.set_item("notional", notionals[i])?;
        entry.set_item("qty", qtys[i])?;
        result.set_item(&symbols[i], entry)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_signum() {
        assert_eq!(1.0_f64.signum(), 1.0);
        assert_eq!((-1.0_f64).signum(), -1.0);
        // Rust: 0.0_f64.signum() == 1.0 (positive zero)
        assert_eq!(0.0_f64.signum(), 1.0);
    }
}
