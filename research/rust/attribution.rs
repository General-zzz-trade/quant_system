//! Attribution module — P&L breakdown, cost attribution, signal-level attribution.
//!
//! Migrates computation from Python `attribution/pnl.py`, `attribution/cost.py`,
//! and `attribution/signal_attribution.py`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

// ── Orderbook bar-level aggregation ─────────────────────────────────────────

/// Aggregate orderbook snapshot metrics into bar-level features.
///
/// imbalances, spreads_bps, depth_ratios: lists of floats collected during one bar.
/// Returns dict with 6 features: ob_imbalance_mean, ob_imbalance_slope,
///   ob_spread_mean_bps, ob_spread_max_bps, ob_depth_ratio_mean, ob_pressure_score.
#[pyfunction]
pub fn rust_flush_orderbook_bar<'py>(
    py: Python<'py>,
    imbalances: Vec<f64>,
    spreads_bps: Vec<f64>,
    depth_ratios: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);

    // Imbalance mean + slope
    if !imbalances.is_empty() {
        let n = imbalances.len() as f64;
        let imb_mean = imbalances.iter().sum::<f64>() / n;
        result.set_item("ob_imbalance_mean", imb_mean)?;

        // Linear regression slope
        if imbalances.len() >= 3 {
            let ni = imbalances.len();
            let x_mean = (ni - 1) as f64 / 2.0;
            let mut num = 0.0_f64;
            let mut denom = 0.0_f64;
            for (i, v) in imbalances.iter().enumerate() {
                let dx = i as f64 - x_mean;
                num += dx * (v - imb_mean);
                denom += dx * dx;
            }
            result.set_item(
                "ob_imbalance_slope",
                if denom > 0.0 { num / denom } else { 0.0 },
            )?;
        }

        // Pressure score
        if !spreads_bps.is_empty() {
            let spread_mean = spreads_bps.iter().sum::<f64>() / spreads_bps.len() as f64;
            let spread_norm = (spread_mean / 20.0).min(1.0);
            result.set_item("ob_pressure_score", imb_mean * (1.0 - spread_norm))?;
        }
    }

    // Spread stats
    if !spreads_bps.is_empty() {
        let spread_mean = spreads_bps.iter().sum::<f64>() / spreads_bps.len() as f64;
        let spread_max = spreads_bps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        result.set_item("ob_spread_mean_bps", spread_mean)?;
        result.set_item("ob_spread_max_bps", spread_max)?;
    }

    // Depth ratio mean
    if !depth_ratios.is_empty() {
        result.set_item(
            "ob_depth_ratio_mean",
            depth_ratios.iter().sum::<f64>() / depth_ratios.len() as f64,
        )?;
    }

    Ok(result)
}

// ── Fill extraction helpers ─────────────────────────────────────────────────

struct Fill {
    symbol: String,
    qty: f64,
    price: f64,
    fee: f64,
    side: String,
    order_id: String,
}

fn extract_fill(obj: &Bound<'_, PyAny>) -> PyResult<Fill> {
    let symbol = obj
        .get_item("symbol")
        .and_then(|v| v.extract::<String>())
        .unwrap_or_default();
    let qty = obj
        .get_item("qty")
        .and_then(|v| v.extract::<f64>())
        .unwrap_or(0.0);
    let price = obj
        .get_item("price")
        .and_then(|v| v.extract::<f64>())
        .unwrap_or(0.0);
    let fee = obj
        .get_item("fee")
        .and_then(|v| v.extract::<f64>())
        .unwrap_or(0.0);
    let side = obj
        .get_item("side")
        .and_then(|v| v.extract::<String>())
        .unwrap_or_else(|_| "buy".to_string());
    let order_id = obj
        .get_item("order_id")
        .and_then(|v| v.extract::<String>())
        .unwrap_or_default();
    Ok(Fill {
        symbol,
        qty,
        price,
        fee,
        side,
        order_id,
    })
}

// ── Position tracking kernel ────────────────────────────────────────────────

struct PnlAccumulator {
    realized: f64,
    fees: f64,
    by_symbol: HashMap<String, f64>,
    positions: HashMap<String, (f64, f64)>, // symbol → (qty, avg_price)
    wins: i64,
    closing_trades: i64,
}

impl PnlAccumulator {
    fn new() -> Self {
        Self {
            realized: 0.0,
            fees: 0.0,
            by_symbol: HashMap::new(),
            positions: HashMap::new(),
            wins: 0,
            closing_trades: 0,
        }
    }

    fn process_fill(&mut self, fill: &Fill) {
        let signed_qty = if fill.side == "buy" {
            fill.qty
        } else {
            -fill.qty
        };
        self.fees += fill.fee;

        let (cur_qty, cur_avg) = self
            .positions
            .get(&fill.symbol)
            .copied()
            .unwrap_or((0.0, 0.0));
        let new_qty = cur_qty + signed_qty;

        // Closing portion → realized P&L
        if cur_qty != 0.0 && (cur_qty > 0.0) != (signed_qty > 0.0) {
            let closed_qty = signed_qty.abs().min(cur_qty.abs());
            let sign = if cur_qty > 0.0 { 1.0 } else { -1.0 };
            let pnl = closed_qty * (fill.price - cur_avg) * sign;
            self.realized += pnl;
            *self.by_symbol.entry(fill.symbol.clone()).or_insert(0.0) += pnl;
            self.closing_trades += 1;
            if pnl > 0.0 {
                self.wins += 1;
            }
        }

        // Update average price
        let new_avg = if new_qty.abs() > cur_qty.abs() {
            let total_cost = cur_avg * cur_qty.abs() + fill.price * signed_qty.abs();
            if new_qty != 0.0 {
                total_cost / new_qty.abs()
            } else {
                0.0
            }
        } else {
            cur_avg
        };
        self.positions.insert(fill.symbol.clone(), (new_qty, new_avg));
    }

    fn unrealized(&self, current_prices: &HashMap<String, f64>) -> f64 {
        let mut unrealized = 0.0;
        for (symbol, (qty, avg)) in &self.positions {
            if *qty != 0.0 {
                if let Some(&price) = current_prices.get(symbol) {
                    unrealized += qty * (price - avg);
                }
            }
        }
        unrealized
    }
}

// ── rust_compute_pnl ────────────────────────────────────────────────────────

/// Compute P&L breakdown from a list of fill dicts.
///
/// Each fill dict: {symbol, side, qty, price, fee}.
/// Returns dict: {total_pnl, realized_pnl, unrealized_pnl, fee_cost, funding_pnl, by_symbol}.
#[pyfunction]
#[pyo3(signature = (fills, current_prices=None))]
pub fn rust_compute_pnl<'py>(
    py: Python<'py>,
    fills: &Bound<'py, PyList>,
    current_prices: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut acc = PnlAccumulator::new();

    for item in fills.iter() {
        let fill = extract_fill(&item)?;
        acc.process_fill(&fill);
    }

    let prices_map: HashMap<String, f64> = if let Some(cp) = current_prices {
        cp.iter()
            .filter_map(|(k, v)| {
                let key = k.extract::<String>().ok()?;
                let val = v.extract::<f64>().ok()?;
                Some((key, val))
            })
            .collect()
    } else {
        HashMap::new()
    };

    let unrealized = acc.unrealized(&prices_map);

    let result = PyDict::new(py);
    result.set_item("total_pnl", acc.realized + unrealized - acc.fees)?;
    result.set_item("realized_pnl", acc.realized)?;
    result.set_item("unrealized_pnl", unrealized)?;
    result.set_item("fee_cost", acc.fees)?;
    result.set_item("funding_pnl", 0.0)?;

    let by_sym = PyDict::new(py);
    for (sym, pnl) in &acc.by_symbol {
        by_sym.set_item(sym, *pnl)?;
    }
    result.set_item("by_symbol", by_sym)?;

    Ok(result)
}

// ── rust_compute_cost_attribution ───────────────────────────────────────────

/// Compute cost attribution (fees + slippage) in basis points.
///
/// Returns dict: {total_cost_bps, fee_bps, slippage_bps, market_impact_bps}.
#[pyfunction]
#[pyo3(signature = (fills, reference_prices=None))]
pub fn rust_compute_cost_attribution<'py>(
    py: Python<'py>,
    fills: &Bound<'py, PyList>,
    reference_prices: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut total_fees = 0.0_f64;
    let mut total_notional = 0.0_f64;

    let ref_map: HashMap<String, f64> = if let Some(rp) = reference_prices {
        rp.iter()
            .filter_map(|(k, v)| {
                let key = k.extract::<String>().ok()?;
                let val = v.extract::<f64>().ok()?;
                Some((key, val))
            })
            .collect()
    } else {
        HashMap::new()
    };

    // First pass: fees + notional
    let mut parsed_fills: Vec<Fill> = Vec::new();
    for item in fills.iter() {
        let fill = extract_fill(&item)?;
        total_fees += fill.fee;
        total_notional += fill.qty * fill.price;
        parsed_fills.push(fill);
    }

    let fee_bps = if total_notional > 0.0 {
        total_fees / total_notional * 10_000.0
    } else {
        0.0
    };

    // Slippage
    let slip_bps = if !ref_map.is_empty() {
        let mut total_slip = 0.0_f64;
        let mut slip_notional = 0.0_f64;
        for fill in &parsed_fills {
            let ref_price = ref_map.get(&fill.symbol).copied().unwrap_or(fill.price);
            if ref_price > 0.0 {
                let slip = (fill.price - ref_price).abs() / ref_price;
                let notional = fill.qty * fill.price;
                total_slip += slip * notional;
                slip_notional += notional;
            }
        }
        if slip_notional > 0.0 {
            total_slip / slip_notional * 10_000.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    let result = PyDict::new(py);
    result.set_item("total_cost_bps", fee_bps + slip_bps)?;
    result.set_item("fee_bps", fee_bps)?;
    result.set_item("slippage_bps", slip_bps)?;
    result.set_item("market_impact_bps", 0.0)?;

    Ok(result)
}

// ── rust_attribute_by_signal ────────────────────────────────────────────────

/// Attribute fills to signal origins via intent→order→fill chain.
///
/// Returns dict: {
///   by_signal: {origin: {origin, realized_pnl, unrealized_pnl, fee_cost, trade_count, win_rate}},
///   total_pnl: f64,
///   unattributed_pnl: f64,
/// }
#[pyfunction]
#[pyo3(signature = (intents, orders, fills, current_prices=None))]
pub fn rust_attribute_by_signal<'py>(
    py: Python<'py>,
    intents: &Bound<'py, PyList>,
    orders: &Bound<'py, PyList>,
    fills: &Bound<'py, PyList>,
    current_prices: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    // Build lookup: intent_id → origin
    let mut intent_origin: HashMap<String, String> = HashMap::new();
    for item in intents.iter() {
        let iid = item
            .get_item("intent_id")
            .and_then(|v| v.extract::<String>())
            .unwrap_or_default();
        let origin = item
            .get_item("origin")
            .and_then(|v| v.extract::<String>())
            .unwrap_or_default();
        intent_origin.insert(iid, origin);
    }

    // Build lookup: order_id → intent_id
    let mut order_intent: HashMap<String, String> = HashMap::new();
    for item in orders.iter() {
        let oid = item
            .get_item("order_id")
            .and_then(|v| v.extract::<String>())
            .unwrap_or_default();
        let iid = item
            .get_item("intent_id")
            .and_then(|v| v.extract::<String>())
            .unwrap_or_default();
        order_intent.insert(oid, iid);
    }

    // Parse current prices
    let prices_map: HashMap<String, f64> = if let Some(cp) = current_prices {
        cp.iter()
            .filter_map(|(k, v)| {
                let key = k.extract::<String>().ok()?;
                let val = v.extract::<f64>().ok()?;
                Some((key, val))
            })
            .collect()
    } else {
        HashMap::new()
    };

    // Group fills by origin
    let mut origin_fills: HashMap<String, Vec<Fill>> = HashMap::new();
    let mut unattributed_fills: Vec<Fill> = Vec::new();

    for item in fills.iter() {
        let fill = extract_fill(&item)?;
        let intent_id = order_intent.get(&fill.order_id).cloned().unwrap_or_default();
        let origin = if !intent_id.is_empty() {
            intent_origin.get(&intent_id).cloned().unwrap_or_default()
        } else {
            String::new()
        };

        if !origin.is_empty() {
            origin_fills
                .entry(origin)
                .or_insert_with(Vec::new)
                .push(fill);
        } else {
            unattributed_fills.push(fill);
        }
    }

    // Compute P&L per origin
    let by_signal = PyDict::new(py);
    let mut total_pnl = 0.0_f64;

    for (origin, o_fills) in &origin_fills {
        let mut acc = PnlAccumulator::new();
        for fill in o_fills {
            acc.process_fill(fill);
        }
        let unrealized = acc.unrealized(&prices_map);
        let win_rate = if acc.closing_trades > 0 {
            acc.wins as f64 / acc.closing_trades as f64
        } else {
            0.0
        };
        let pnl = acc.realized + unrealized - acc.fees;
        total_pnl += pnl;

        let sig_dict = PyDict::new(py);
        sig_dict.set_item("origin", origin)?;
        sig_dict.set_item("realized_pnl", acc.realized)?;
        sig_dict.set_item("unrealized_pnl", unrealized)?;
        sig_dict.set_item("fee_cost", acc.fees)?;
        sig_dict.set_item("trade_count", o_fills.len())?;
        sig_dict.set_item("win_rate", win_rate)?;
        by_signal.set_item(origin, sig_dict)?;
    }

    // Unattributed
    let mut unattributed_total = 0.0_f64;
    if !unattributed_fills.is_empty() {
        let mut acc = PnlAccumulator::new();
        for fill in &unattributed_fills {
            acc.process_fill(fill);
        }
        let unrealized = acc.unrealized(&prices_map);
        unattributed_total = acc.realized + unrealized - acc.fees;
        total_pnl += unattributed_total;
    }

    let result = PyDict::new(py);
    result.set_item("by_signal", by_signal)?;
    result.set_item("total_pnl", total_pnl)?;
    result.set_item("unattributed_pnl", unattributed_total)?;

    Ok(result)
}
