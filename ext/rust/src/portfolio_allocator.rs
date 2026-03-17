//! Portfolio allocation constraint math in Rust.
//!
//! Replaces Decimal-heavy Python constraint loops with f64 math.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

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

// ============================================================
// RustPortfolioAllocator — stateful allocator with full pipeline
// ============================================================

/// Helper: round qty DOWN to nearest lot step (floor).
/// Returns signed result preserving the sign of qty.
fn round_to_lot(qty: f64, step: f64) -> f64 {
    if step <= 0.0 || !step.is_finite() {
        return qty;
    }
    let sign = qty.signum();
    let abs_qty = qty.abs();
    let n = (abs_qty / step).floor();
    sign * n * step
}

/// Helper: parse a Bound<PyDict> into HashMap<String, f64>.
fn parse_dict(d: &Bound<'_, PyDict>) -> PyResult<HashMap<String, f64>> {
    let mut map = HashMap::with_capacity(d.len());
    for (k, v) in d.iter() {
        let sym: String = k.extract()?;
        let val: f64 = v.extract()?;
        map.insert(sym, val);
    }
    Ok(map)
}

#[pyclass(name = "RustPortfolioAllocator")]
pub struct RustPortfolioAllocator {
    max_gross_leverage: f64,
    max_net_leverage: f64,
    max_notional_per_symbol: f64,
    max_concentration: f64,
    min_trade_notional: f64,
    turnover_cap: f64,
    lot_sizes: HashMap<String, f64>,
}

#[pymethods]
impl RustPortfolioAllocator {
    #[new]
    #[pyo3(signature = (
        max_gross_leverage=3.0,
        max_net_leverage=1.0,
        max_notional_per_symbol=5000.0,
        max_concentration=0.4,
        min_trade_notional=5.0,
        turnover_cap=1.0
    ))]
    fn new(
        max_gross_leverage: f64,
        max_net_leverage: f64,
        max_notional_per_symbol: f64,
        max_concentration: f64,
        min_trade_notional: f64,
        turnover_cap: f64,
    ) -> Self {
        RustPortfolioAllocator {
            max_gross_leverage,
            max_net_leverage,
            max_notional_per_symbol,
            max_concentration,
            min_trade_notional,
            turnover_cap,
            lot_sizes: HashMap::new(),
        }
    }

    /// Set lot size (qty step) for a symbol.
    fn set_lot_size(&mut self, symbol: &str, step: f64) {
        self.lot_sizes.insert(symbol.to_string(), step);
    }

    /// Full allocation pipeline: target weights -> constrained positions -> trade intents.
    ///
    /// Returns dict with "trades" list and "diagnostics" dict.
    #[pyo3(signature = (target_weights, current_qty, prices, equity))]
    fn allocate<'py>(
        &self,
        py: Python<'py>,
        target_weights: &Bound<'_, PyDict>,
        current_qty: &Bound<'_, PyDict>,
        prices: &Bound<'_, PyDict>,
        equity: f64,
    ) -> PyResult<PyObject> {
        self.allocate_inner(py, target_weights, current_qty, prices, equity, self.turnover_cap)
    }

    /// Rebalance convenience: compute trades with explicit turnover cap.
    #[pyo3(signature = (target_weights, current_qty, prices, equity, max_turnover_pct))]
    fn rebalance<'py>(
        &self,
        py: Python<'py>,
        target_weights: &Bound<'_, PyDict>,
        current_qty: &Bound<'_, PyDict>,
        prices: &Bound<'_, PyDict>,
        equity: f64,
        max_turnover_pct: f64,
    ) -> PyResult<PyObject> {
        self.allocate_inner(py, target_weights, current_qty, prices, equity, max_turnover_pct)
    }

    /// Scale a single order qty to fit constraints. Returns scale factor in [0, 1].
    #[pyo3(signature = (symbol, qty, equity, price))]
    fn scale_order(&self, symbol: &str, qty: f64, equity: f64, price: f64) -> f64 {
        let _ = symbol; // symbol reserved for future per-symbol limits
        if !qty.is_finite() || !equity.is_finite() || !price.is_finite()
            || equity <= 0.0 || price <= 0.0
        {
            return 0.0;
        }
        let notional = qty.abs() * price;
        if notional <= 0.0 {
            return 1.0;
        }
        let mut scale = 1.0_f64;

        // Apply notional cap
        if notional > self.max_notional_per_symbol {
            scale = scale.min(self.max_notional_per_symbol / notional);
        }

        // Apply leverage cap
        let leverage = notional / equity;
        if leverage > self.max_gross_leverage {
            scale = scale.min(self.max_gross_leverage * equity / notional);
        }

        scale
    }
}

impl RustPortfolioAllocator {
    /// Inner allocation logic shared by allocate() and rebalance().
    fn allocate_inner<'py>(
        &self,
        py: Python<'py>,
        target_weights: &Bound<'_, PyDict>,
        current_qty: &Bound<'_, PyDict>,
        prices: &Bound<'_, PyDict>,
        equity: f64,
        effective_turnover_cap: f64,
    ) -> PyResult<PyObject> {
        if !equity.is_finite() || equity <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "equity must be positive finite",
            ));
        }

        // 1. Parse inputs
        let tw = parse_dict(target_weights)?;
        let cq = parse_dict(current_qty)?;
        let px = parse_dict(prices)?;

        // Collect symbols from target_weights (deterministic order via Vec)
        let symbols: Vec<String> = tw.keys().cloned().collect();

        // Compute gross leverage BEFORE for diagnostics
        let gross_before: f64 = symbols
            .iter()
            .map(|s| {
                let cur = cq.get(s).copied().unwrap_or(0.0);
                let p = px.get(s).copied().unwrap_or(0.0);
                cur.abs() * p
            })
            .sum();
        let gross_lev_before = gross_before / equity;

        // 2. Convert target weights to target notionals and qty
        let mut target_notionals: HashMap<String, f64> = HashMap::new();
        let mut target_qtys: HashMap<String, f64> = HashMap::new();

        for s in &symbols {
            let w = tw.get(s).copied().unwrap_or(0.0);
            let p = px.get(s).copied().unwrap_or(0.0);
            if p <= 0.0 || !p.is_finite() {
                target_notionals.insert(s.clone(), 0.0);
                target_qtys.insert(s.clone(), 0.0);
                continue;
            }
            let notional = w * equity; // signed notional
            target_notionals.insert(s.clone(), notional);
            target_qtys.insert(s.clone(), notional / p);
        }

        // 3. Apply gross leverage cap: if sum(|notional|) / equity > max_gross_leverage,
        //    scale ALL positions down proportionally.
        let gross_sum: f64 = target_notionals.values().map(|n| n.abs()).sum();
        let gross_lev = gross_sum / equity;
        if gross_lev > self.max_gross_leverage && gross_sum > 0.0 {
            let scale = self.max_gross_leverage / gross_lev;
            for s in &symbols {
                if let Some(n) = target_notionals.get_mut(s) {
                    *n *= scale;
                }
                if let Some(q) = target_qtys.get_mut(s) {
                    *q *= scale;
                }
            }
        }

        // 4. Apply net leverage cap: if |sum(notional)| / equity > max_net_leverage,
        //    scale all down proportionally.
        let net_sum: f64 = target_notionals.values().sum();
        let net_lev = net_sum.abs() / equity;
        if net_lev > self.max_net_leverage && net_sum.abs() > 0.0 {
            let scale = self.max_net_leverage / net_lev;
            for s in &symbols {
                if let Some(n) = target_notionals.get_mut(s) {
                    *n *= scale;
                }
                if let Some(q) = target_qtys.get_mut(s) {
                    *q *= scale;
                }
            }
        }

        // 5. Apply per-symbol notional cap: clamp each |notional| to max_notional_per_symbol.
        for s in &symbols {
            let n = target_notionals.get(s).copied().unwrap_or(0.0);
            if n.abs() > self.max_notional_per_symbol {
                let sign = n.signum();
                let capped = sign * self.max_notional_per_symbol;
                target_notionals.insert(s.clone(), capped);
                let p = px.get(s).copied().unwrap_or(0.0);
                if p > 0.0 {
                    target_qtys.insert(s.clone(), capped / p);
                }
            }
        }

        // 6. Apply concentration limit: |notional_i| / sum(|notional|) <= max_concentration.
        //    Iterate until converged (one pass usually suffices).
        for _ in 0..5 {
            let total_abs: f64 = target_notionals.values().map(|n| n.abs()).sum();
            if total_abs <= 0.0 {
                break;
            }
            let mut changed = false;
            for s in &symbols {
                let n = target_notionals.get(s).copied().unwrap_or(0.0);
                let conc = n.abs() / total_abs;
                if conc > self.max_concentration {
                    let max_abs = self.max_concentration * total_abs;
                    let sign = n.signum();
                    let capped = sign * max_abs;
                    target_notionals.insert(s.clone(), capped);
                    let p = px.get(s).copied().unwrap_or(0.0);
                    if p > 0.0 {
                        target_qtys.insert(s.clone(), capped / p);
                    }
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        // 7. Compute deltas: delta_i = target_qty_i - current_qty_i
        let mut deltas: HashMap<String, f64> = HashMap::new();
        for s in &symbols {
            let tgt = target_qtys.get(s).copied().unwrap_or(0.0);
            let cur = cq.get(s).copied().unwrap_or(0.0);
            deltas.insert(s.clone(), tgt - cur);
        }

        // 8. Round deltas to lot sizes (floor toward zero)
        for s in &symbols {
            if let Some(step) = self.lot_sizes.get(s) {
                if let Some(d) = deltas.get_mut(s) {
                    *d = round_to_lot(*d, *step);
                }
            }
        }

        // 9. Filter sub-minimum trades: skip if |delta * price| < min_trade_notional
        for s in &symbols {
            let d = deltas.get(s).copied().unwrap_or(0.0);
            let p = px.get(s).copied().unwrap_or(0.0);
            if (d.abs() * p) < self.min_trade_notional {
                deltas.insert(s.clone(), 0.0);
            }
        }

        // 10. Apply turnover cap: if sum(|delta * price|) / equity > turnover_cap, scale all deltas
        let total_trade_notional: f64 = symbols
            .iter()
            .map(|s| {
                let d = deltas.get(s).copied().unwrap_or(0.0);
                let p = px.get(s).copied().unwrap_or(0.0);
                d.abs() * p
            })
            .sum();
        let turnover = total_trade_notional / equity;
        if turnover > effective_turnover_cap && total_trade_notional > 0.0 {
            let scale = effective_turnover_cap / turnover;
            for s in &symbols {
                if let Some(d) = deltas.get_mut(s) {
                    *d *= scale;
                }
            }
            // Re-round after scaling
            for s in &symbols {
                if let Some(step) = self.lot_sizes.get(s) {
                    if let Some(d) = deltas.get_mut(s) {
                        *d = round_to_lot(*d, *step);
                    }
                }
            }
        }

        // Build trades list
        let trades = PyList::empty(py);
        for s in &symbols {
            let d = deltas.get(s).copied().unwrap_or(0.0);
            if d == 0.0 {
                continue;
            }
            let p = px.get(s).copied().unwrap_or(0.0);
            let notional_delta = d.abs() * p;

            // 11. Determine reduce_only
            let cur = cq.get(s).copied().unwrap_or(0.0);
            let tgt = target_qtys.get(s).copied().unwrap_or(0.0);
            let mut reduce_only = tgt.abs() < cur.abs();
            // Also reduce_only if delta opposes current position sign
            if cur != 0.0 && ((cur > 0.0 && d < 0.0) || (cur < 0.0 && d > 0.0)) {
                reduce_only = true;
            }

            // 12. Determine side
            let side = if d > 0.0 { "buy" } else { "sell" };

            let trade = PyDict::new(py);
            trade.set_item("symbol", s.as_str())?;
            trade.set_item("qty_delta", d)?;
            trade.set_item("notional_delta", notional_delta)?;
            trade.set_item("side", side)?;
            trade.set_item("reduce_only", reduce_only)?;
            trades.append(trade)?;
        }

        // Compute gross leverage after
        let gross_after: f64 = symbols
            .iter()
            .map(|s| {
                let cur = cq.get(s).copied().unwrap_or(0.0);
                let d = deltas.get(s).copied().unwrap_or(0.0);
                let p = px.get(s).copied().unwrap_or(0.0);
                (cur + d).abs() * p
            })
            .sum();
        let gross_lev_after = gross_after / equity;

        // Net leverage after
        let net_after: f64 = symbols
            .iter()
            .map(|s| {
                let cur = cq.get(s).copied().unwrap_or(0.0);
                let d = deltas.get(s).copied().unwrap_or(0.0);
                let p = px.get(s).copied().unwrap_or(0.0);
                (cur + d) * p
            })
            .sum();
        let net_lev_after = net_after.abs() / equity;

        let diagnostics = PyDict::new(py);
        diagnostics.set_item("equity", equity)?;
        diagnostics.set_item("gross_leverage_before", gross_lev_before)?;
        diagnostics.set_item("gross_leverage_after", gross_lev_after)?;
        diagnostics.set_item("net_leverage_after", net_lev_after)?;
        diagnostics.set_item("turnover", turnover)?;
        diagnostics.set_item("total_trade_notional", total_trade_notional)?;
        diagnostics.set_item("num_trades", trades.len())?;

        // 13. Build result dict
        let result = PyDict::new(py);
        result.set_item("trades", trades)?;
        result.set_item("diagnostics", diagnostics)?;

        Ok(result.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signum() {
        assert_eq!(1.0_f64.signum(), 1.0);
        assert_eq!((-1.0_f64).signum(), -1.0);
        // Rust: 0.0_f64.signum() == 1.0 (positive zero)
        assert_eq!(0.0_f64.signum(), 1.0);
    }

    #[test]
    fn test_round_to_lot() {
        // Positive qty, step=0.1 -> floor
        assert!((round_to_lot(1.57, 0.1) - 1.5).abs() < 1e-12);
        // Negative qty -> floor toward zero
        assert!((round_to_lot(-1.57, 0.1) - (-1.5)).abs() < 1e-12);
        // Exact multiple
        assert!((round_to_lot(2.0, 0.5) - 2.0).abs() < 1e-12);
        // Step=0 -> return qty unchanged
        assert!((round_to_lot(1.57, 0.0) - 1.57).abs() < 1e-12);
        // Large step
        assert!((round_to_lot(7.0, 10.0) - 0.0).abs() < 1e-12);
        assert!((round_to_lot(15.0, 10.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_scale_order_notional_cap() {
        let alloc = RustPortfolioAllocator::new(3.0, 1.0, 5000.0, 0.4, 5.0, 1.0);
        // qty=10, price=1000 -> notional=10000 > 5000 cap
        let scale = alloc.scale_order("ETH", 10.0, 100000.0, 1000.0);
        assert!((scale - 0.5).abs() < 1e-9); // 5000/10000
    }

    #[test]
    fn test_scale_order_leverage_cap() {
        let alloc = RustPortfolioAllocator::new(3.0, 1.0, 50000.0, 0.4, 5.0, 1.0);
        // qty=10, price=1000 -> notional=10000, equity=1000 -> leverage=10 > 3
        let scale = alloc.scale_order("ETH", 10.0, 1000.0, 1000.0);
        assert!((scale - 0.3).abs() < 1e-9); // 3*1000/10000
    }

    #[test]
    fn test_scale_order_invalid_inputs() {
        let alloc = RustPortfolioAllocator::new(3.0, 1.0, 5000.0, 0.4, 5.0, 1.0);
        assert_eq!(alloc.scale_order("X", f64::NAN, 1000.0, 100.0), 0.0);
        assert_eq!(alloc.scale_order("X", 1.0, 0.0, 100.0), 0.0);
        assert_eq!(alloc.scale_order("X", 1.0, 1000.0, 0.0), 0.0);
        assert_eq!(alloc.scale_order("X", 1.0, -1.0, 100.0), 0.0);
    }
}
