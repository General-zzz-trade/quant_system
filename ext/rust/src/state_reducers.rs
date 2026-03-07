use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

use crate::state_types::{
    RustAccountState, RustMarketState, RustPortfolioState, RustPositionState, RustReducerResult,
    RustRiskLimits, RustRiskState,
};

// ===========================================================================
// Helper functions
// ===========================================================================

/// Normalize event_type: check header.event_type, then event.event_type.
/// Handles enum-like objects with .value or .name attributes.
fn get_event_type(event: &Bound<'_, PyAny>) -> PyResult<String> {
    // Try header.event_type first
    let raw = if let Ok(header) = event.getattr("header") {
        if !header.is_none() {
            if let Ok(et) = header.getattr("event_type") {
                if !et.is_none() {
                    Some(et.into())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    let raw: Bound<'_, PyAny> = match raw {
        Some(r) => r,
        None => match event.getattr("event_type") {
            Ok(et) if !et.is_none() => et,
            _ => return Ok(String::new()),
        },
    };

    // Handle enum-like: .value > .name > str()
    let val = if let Ok(v) = raw.getattr("value") {
        v
    } else if let Ok(n) = raw.getattr("name") {
        n
    } else {
        raw
    };

    let s: String = val.str()?.to_string();
    Ok(s.trim().to_lowercase())
}

/// Extract symbol from event. Checks event.symbol, then event.bar.symbol.
fn get_symbol(event: &Bound<'_, PyAny>, default: &str) -> PyResult<String> {
    if let Ok(sym) = event.getattr("symbol") {
        if !sym.is_none() {
            return Ok(sym.str()?.to_string());
        }
    }
    // Nested bar.symbol
    if let Ok(bar) = event.getattr("bar") {
        if !bar.is_none() {
            if let Ok(sym) = bar.getattr("symbol") {
                if !sym.is_none() {
                    return Ok(sym.str()?.to_string());
                }
            }
        }
    }
    Ok(default.to_string())
}

/// Extract timestamp as Option<String> (ISO format or whatever Python gives us).
/// Checks header.ts, then event.ts.
fn get_event_ts(event: &Bound<'_, PyAny>) -> PyResult<Option<PyObject>> {
    // Try header.ts first
    if let Ok(header) = event.getattr("header") {
        if !header.is_none() {
            if let Ok(ts) = header.getattr("ts") {
                if !ts.is_none() {
                    return Ok(Some(ts.unbind()));
                }
            }
        }
    }
    if let Ok(ts) = event.getattr("ts") {
        if !ts.is_none() {
            return Ok(Some(ts.unbind()));
        }
    }
    Ok(None)
}

/// Convert a Python value to f64. Handles int, float, Decimal, str.
fn to_f64(val: &Bound<'_, PyAny>) -> PyResult<f64> {
    // Try direct float extraction first (fastest path)
    if let Ok(f) = val.extract::<f64>() {
        return Ok(f);
    }
    // Fall back to str -> parse
    let s = val.str()?.to_string();
    s.parse::<f64>().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("cannot convert to f64: {} ({})", s, e))
    })
}

/// Try to extract an optional attribute as f64. Returns Ok(None) if attr missing or None.
fn opt_attr_f64(event: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<f64>> {
    match event.getattr(name) {
        Ok(v) if !v.is_none() => Ok(Some(to_f64(&v)?)),
        _ => Ok(None),
    }
}

/// Format f64 back to string with sufficient precision, stripping trailing zeros.
fn fmt_decimal(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let s = format!("{:.18}", v);
    // Strip trailing zeros after decimal point
    if s.contains('.') {
        let s = s.trim_end_matches('0');
        let s = s.trim_end_matches('.');
        s.to_string()
    } else {
        s
    }
}

/// Apply side to qty: buy/long -> +abs, sell/short -> -abs.
fn signed_qty_f64(qty: f64, side: &str) -> f64 {
    let s = side.trim().to_lowercase();
    match s.as_str() {
        "buy" | "long" => qty.abs(),
        "sell" | "short" => -qty.abs(),
        _ => qty,
    }
}

/// Convert Python ts object to string representation for Rust state.
/// Passes through to Python str() if present.
fn ts_to_opt_string(py: Python<'_>, ts: &Option<PyObject>) -> PyResult<Option<String>> {
    match ts {
        Some(t) => {
            let bound = t.bind(py);
            if bound.is_none() {
                Ok(None)
            } else {
                Ok(Some(bound.str()?.to_string()))
            }
        }
        None => Ok(None),
    }
}

// ===========================================================================
// RustMarketReducer
// ===========================================================================

#[pyclass(name = "RustMarketReducer")]
pub struct RustMarketReducer;

#[pymethods]
impl RustMarketReducer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn reduce(
        &self,
        py: Python<'_>,
        state: &RustMarketState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let et = get_event_type(event)?;
        let sym = get_symbol(event, &state.symbol)?;

        if sym != state.symbol {
            return Ok(RustReducerResult {
                state: state.clone().into_pyobject(py)?.into_any().unbind(),
                changed: false,
                note: None,
            });
        }

        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        // Bar-like events
        if matches!(et.as_str(), "market" | "market_bar" | "bar" | "marketbar") {
            let mut o = opt_attr_f64(event, "open")?;
            let mut h = opt_attr_f64(event, "high")?;
            let mut l = opt_attr_f64(event, "low")?;
            let mut c = opt_attr_f64(event, "close")?;
            let mut v = opt_attr_f64(event, "volume")?;

            // Support nested bar object
            if c.is_none() {
                if let Ok(bar) = event.getattr("bar") {
                    if !bar.is_none() {
                        if o.is_none() {
                            o = opt_attr_f64(&bar, "open")?;
                        }
                        if h.is_none() {
                            h = opt_attr_f64(&bar, "high")?;
                        }
                        if l.is_none() {
                            l = opt_attr_f64(&bar, "low")?;
                        }
                        c = opt_attr_f64(&bar, "close")?;
                        if v.is_none() {
                            v = opt_attr_f64(&bar, "volume")?;
                        }
                    }
                }
            }

            if let Some(c_f) = c {
                // Validate close > 0 and finite
                if c_f <= 0.0 || !c_f.is_finite() {
                    return Ok(RustReducerResult {
                        state: state.clone().into_pyobject(py)?.into_any().unbind(),
                        changed: false,
                        note: None,
                    });
                }
            }

            // Validate high >= low
            if let (Some(h_f), Some(l_f)) = (h, l) {
                if h_f < l_f {
                    return Ok(RustReducerResult {
                        state: state.clone().into_pyobject(py)?.into_any().unbind(),
                        changed: false,
                        note: None,
                    });
                }
            }

            // Validate volume >= 0
            if let Some(v_f) = v {
                if v_f < 0.0 {
                    return Ok(RustReducerResult {
                        state: state.clone().into_pyobject(py)?.into_any().unbind(),
                        changed: false,
                        note: None,
                    });
                }
            }

            if let Some(c_f) = c {
                let o_s = fmt_decimal(o.unwrap_or(c_f));
                let h_s = fmt_decimal(h.unwrap_or(c_f));
                let l_s = fmt_decimal(l.unwrap_or(c_f));
                let c_s = fmt_decimal(c_f);
                let v_s = v.map(fmt_decimal);

                let new_state = RustMarketState {
                    symbol: state.symbol.clone(),
                    last_price: Some(c_s.clone()),
                    open: Some(o_s),
                    high: Some(h_s),
                    low: Some(l_s),
                    close: Some(c_s),
                    volume: v_s,
                    last_ts: ts_str.or_else(|| state.last_ts.clone()),
                };
                return Ok(RustReducerResult {
                    state: new_state.into_pyobject(py)?.into_any().unbind(),
                    changed: true,
                    note: Some("market_bar".to_string()),
                });
            }

            // No close: check for price (tick fallback)
            let price = opt_attr_f64(event, "price")?;
            if let Some(p) = price {
                let new_state = RustMarketState {
                    symbol: state.symbol.clone(),
                    last_price: Some(fmt_decimal(p)),
                    open: state.open.clone(),
                    high: state.high.clone(),
                    low: state.low.clone(),
                    close: state.close.clone(),
                    volume: state.volume.clone(),
                    last_ts: ts_str.or_else(|| state.last_ts.clone()),
                };
                return Ok(RustReducerResult {
                    state: new_state.into_pyobject(py)?.into_any().unbind(),
                    changed: true,
                    note: Some("market_tick".to_string()),
                });
            }
            return Err(pyo3::exceptions::PyValueError::new_err(
                "market event missing close/price",
            ));
        }

        // Tick-like events
        if matches!(et.as_str(), "market_tick" | "tick") {
            let price = opt_attr_f64(event, "price")?;
            match price {
                Some(p) => {
                    let new_state = RustMarketState {
                        symbol: state.symbol.clone(),
                        last_price: Some(fmt_decimal(p)),
                        open: state.open.clone(),
                        high: state.high.clone(),
                        low: state.low.clone(),
                        close: state.close.clone(),
                        volume: state.volume.clone(),
                        last_ts: ts_str.or_else(|| state.last_ts.clone()),
                    };
                    Ok(RustReducerResult {
                        state: new_state.into_pyobject(py)?.into_any().unbind(),
                        changed: true,
                        note: Some("market_tick".to_string()),
                    })
                }
                None => Err(pyo3::exceptions::PyValueError::new_err(
                    "tick event missing price",
                )),
            }
        } else {
            // Unrecognized event type
            Ok(RustReducerResult {
                state: state.clone().into_pyobject(py)?.into_any().unbind(),
                changed: false,
                note: None,
            })
        }
    }
}

// ===========================================================================
// RustPositionReducer
// ===========================================================================

#[pyclass(name = "RustPositionReducer")]
pub struct RustPositionReducer;

#[pymethods]
impl RustPositionReducer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn reduce(
        &self,
        py: Python<'_>,
        state: &RustPositionState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let et = get_event_type(event)?;

        if !matches!(et.as_str(), "fill" | "trade_fill" | "execution_fill") {
            return Ok(RustReducerResult {
                state: state.clone().into_pyobject(py)?.into_any().unbind(),
                changed: false,
                note: None,
            });
        }

        let sym = get_symbol(event, &state.symbol)?;
        if sym != state.symbol {
            return Ok(RustReducerResult {
                state: state.clone().into_pyobject(py)?.into_any().unbind(),
                changed: false,
                note: None,
            });
        }

        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        // Extract qty
        let qty_raw = opt_attr_f64(event, "qty")?
            .or(opt_attr_f64(event, "quantity")?)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("fill event missing qty/quantity")
            })?;

        // Extract side
        let side_obj = event.getattr("side").ok();
        let side_str = match &side_obj {
            Some(s) if !s.is_none() => s.str()?.to_string(),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "fill event missing side",
                ))
            }
        };

        let qty = signed_qty_f64(qty_raw, &side_str);
        if qty == 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fill qty cannot be 0",
            ));
        }

        // Extract price
        let price = opt_attr_f64(event, "price")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("fill event missing price")
        })?;
        if price <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fill price must be > 0",
            ));
        }

        let prev_qty: f64 = state.qty.parse().unwrap_or(0.0);
        let new_qty = prev_qty + qty;
        let price_s = fmt_decimal(price);

        // 1) flat
        if new_qty.abs() < 1e-18 {
            let new_state = RustPositionState {
                symbol: state.symbol.clone(),
                qty: "0".to_string(),
                avg_price: None,
                last_price: Some(price_s),
                last_ts: ts_str.or_else(|| state.last_ts.clone()),
            };
            return Ok(RustReducerResult {
                state: new_state.into_pyobject(py)?.into_any().unbind(),
                changed: true,
                note: Some("position_flat".to_string()),
            });
        }

        // 2) opening or adding (same direction)
        if prev_qty == 0.0 || (prev_qty > 0.0) == (qty > 0.0) {
            let new_avg = if prev_qty == 0.0 || state.avg_price.is_none() {
                price
            } else {
                let old_avg: f64 = state
                    .avg_price
                    .as_ref()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(price);
                (prev_qty.abs() * old_avg + qty.abs() * price) / new_qty.abs()
            };

            let new_state = RustPositionState {
                symbol: state.symbol.clone(),
                qty: fmt_decimal(new_qty),
                avg_price: Some(fmt_decimal(new_avg)),
                last_price: Some(price_s),
                last_ts: ts_str.or_else(|| state.last_ts.clone()),
            };
            return Ok(RustReducerResult {
                state: new_state.into_pyobject(py)?.into_any().unbind(),
                changed: true,
                note: Some("position_add".to_string()),
            });
        }

        // 3) reducing (opposite direction, not crossing zero)
        if qty.abs() < prev_qty.abs() {
            let new_state = RustPositionState {
                symbol: state.symbol.clone(),
                qty: fmt_decimal(new_qty),
                avg_price: state.avg_price.clone(),
                last_price: Some(price_s),
                last_ts: ts_str.or_else(|| state.last_ts.clone()),
            };
            return Ok(RustReducerResult {
                state: new_state.into_pyobject(py)?.into_any().unbind(),
                changed: true,
                note: Some("position_reduce".to_string()),
            });
        }

        // 4) crossing zero -> reverse: avg resets to fill price
        let new_state = RustPositionState {
            symbol: state.symbol.clone(),
            qty: fmt_decimal(new_qty),
            avg_price: Some(price_s.clone()),
            last_price: Some(price_s),
            last_ts: ts_str.or_else(|| state.last_ts.clone()),
        };
        Ok(RustReducerResult {
            state: new_state.into_pyobject(py)?.into_any().unbind(),
            changed: true,
            note: Some("position_reverse".to_string()),
        })
    }
}

// ===========================================================================
// RustAccountReducer
// ===========================================================================

#[pyclass(name = "RustAccountReducer")]
pub struct RustAccountReducer;

#[pymethods]
impl RustAccountReducer {
    #[new]
    fn new() -> Self {
        Self
    }

    fn reduce(
        &self,
        py: Python<'_>,
        state: &RustAccountState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let et = get_event_type(event)?;

        if et == "funding" {
            return self.reduce_funding(py, state, event);
        }

        if !matches!(et.as_str(), "fill" | "trade_fill" | "execution_fill") {
            return Ok(RustReducerResult {
                state: state.clone().into_pyobject(py)?.into_any().unbind(),
                changed: false,
                note: None,
            });
        }

        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        let fee = opt_attr_f64(event, "fee")?.unwrap_or(0.0);
        let realized = opt_attr_f64(event, "realized_pnl")?.unwrap_or(0.0);
        let cash_delta = opt_attr_f64(event, "cash_delta")?.unwrap_or(0.0);
        let margin_change = opt_attr_f64(event, "margin_change")?.unwrap_or(0.0);

        let balance: f64 = state.balance.parse().unwrap_or(0.0);
        let margin_used: f64 = state.margin_used.parse().unwrap_or(0.0);
        let realized_pnl: f64 = state.realized_pnl.parse().unwrap_or(0.0);
        let unrealized_pnl: f64 = state.unrealized_pnl.parse().unwrap_or(0.0);
        let fees_paid: f64 = state.fees_paid.parse().unwrap_or(0.0);

        let new_balance = balance + realized + cash_delta - fee;
        let new_margin_used = margin_used + margin_change;
        let new_margin_available = new_balance - new_margin_used;

        let new_state = RustAccountState {
            currency: state.currency.clone(),
            balance: fmt_decimal(new_balance),
            margin_used: fmt_decimal(new_margin_used),
            margin_available: fmt_decimal(new_margin_available),
            realized_pnl: fmt_decimal(realized_pnl + realized),
            unrealized_pnl: fmt_decimal(unrealized_pnl),
            fees_paid: fmt_decimal(fees_paid + fee),
            last_ts: ts_str.or_else(|| state.last_ts.clone()),
        };
        Ok(RustReducerResult {
            state: new_state.into_pyobject(py)?.into_any().unbind(),
            changed: true,
            note: Some("fill_account".to_string()),
        })
    }
}

impl RustAccountReducer {
    fn reduce_funding(
        &self,
        py: Python<'_>,
        state: &RustAccountState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        let funding_rate = opt_attr_f64(event, "funding_rate")?;
        let mark_price = opt_attr_f64(event, "mark_price")?;
        let position_qty = opt_attr_f64(event, "position_qty")?;

        // If any required field is missing or position is zero, no-op
        match (funding_rate, mark_price, position_qty) {
            (Some(fr), Some(mp), Some(pq)) if pq != 0.0 => {
                let balance: f64 = state.balance.parse().unwrap_or(0.0);
                let margin_used: f64 = state.margin_used.parse().unwrap_or(0.0);
                let realized_pnl: f64 = state.realized_pnl.parse().unwrap_or(0.0);
                let unrealized_pnl: f64 = state.unrealized_pnl.parse().unwrap_or(0.0);
                let fees_paid: f64 = state.fees_paid.parse().unwrap_or(0.0);

                // funding_payment > 0 means account pays
                let funding_payment = pq * mp * fr;
                let new_balance = balance - funding_payment;
                let new_margin_available = new_balance - margin_used;

                let new_state = RustAccountState {
                    currency: state.currency.clone(),
                    balance: fmt_decimal(new_balance),
                    margin_used: fmt_decimal(margin_used),
                    margin_available: fmt_decimal(new_margin_available),
                    realized_pnl: fmt_decimal(realized_pnl - funding_payment),
                    unrealized_pnl: fmt_decimal(unrealized_pnl),
                    fees_paid: fmt_decimal(fees_paid + funding_payment.abs()),
                    last_ts: ts_str.or_else(|| state.last_ts.clone()),
                };
                Ok(RustReducerResult {
                    state: new_state.into_pyobject(py)?.into_any().unbind(),
                    changed: true,
                    note: Some("funding_settlement".to_string()),
                })
            }
            _ => Ok(RustReducerResult {
                state: state.clone().into_pyobject(py)?.into_any().unbind(),
                changed: false,
                note: None,
            }),
        }
    }
}

// ===========================================================================
// RustPortfolioReducer
// ===========================================================================

/// Portfolio is derived state: recomputed from account + positions + market on every event.
/// Stores Python callables (get_account, get_positions, get_market) to fetch current state.
#[pyclass(name = "RustPortfolioReducer")]
pub struct RustPortfolioReducer {
    get_account: PyObject,
    get_positions: PyObject,
    get_market: PyObject,
}

#[pymethods]
impl RustPortfolioReducer {
    #[new]
    #[pyo3(signature = (*, get_account, get_positions, get_market))]
    fn new(get_account: PyObject, get_positions: PyObject, get_market: PyObject) -> Self {
        Self {
            get_account,
            get_positions,
            get_market,
        }
    }

    fn reduce(
        &self,
        py: Python<'_>,
        state: &RustPortfolioState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        // Call the Python callables to get current state
        let account_obj = self.get_account.call0(py)?;
        let account: RustAccountState = account_obj.extract(py)?;

        let positions_obj = self.get_positions.call0(py)?;
        let positions: &Bound<'_, PyDict> = positions_obj.downcast_bound(py)?;

        let market_obj = self.get_market.call0(py)?;
        let market: RustMarketState = market_obj.extract(py)?;

        // Compute portfolio from components
        let balance: f64 = account.balance.parse().unwrap_or(0.0);
        let margin_used: f64 = account.margin_used.parse().unwrap_or(0.0);
        let margin_available: f64 = account.margin_available.parse().unwrap_or(0.0);
        let realized_pnl: f64 = account.realized_pnl.parse().unwrap_or(0.0);
        let fees_paid: f64 = account.fees_paid.parse().unwrap_or(0.0);

        let market_price: Option<f64> = market
            .last_price
            .as_ref()
            .and_then(|s| s.parse().ok());

        let mut gross: f64 = 0.0;
        let mut net: f64 = 0.0;
        let mut unreal: f64 = 0.0;
        let mut syms: Vec<String> = Vec::new();

        for (key, val) in positions.iter() {
            let sym: String = key.extract()?;
            let pos: RustPositionState = val.extract()?;
            let qty: f64 = pos.qty.parse().unwrap_or(0.0);
            if qty == 0.0 {
                continue;
            }

            // Mark price: market for primary symbol, otherwise pos.last_price
            let mark = if sym == market.symbol {
                market_price
            } else {
                None
            }
            .or_else(|| {
                pos.last_price
                    .as_ref()
                    .and_then(|s| s.parse().ok())
            });

            if let Some(m) = mark {
                let notional = qty.abs() * m;
                gross += notional;
                net += qty * m;

                if let Some(ref avg_s) = pos.avg_price {
                    if let Ok(avg) = avg_s.parse::<f64>() {
                        unreal += (m - avg) * qty;
                    }
                }
            }

            syms.push(sym);
        }

        syms.sort();

        let total_equity = balance + unreal;
        let leverage = if total_equity > 0.0 && gross != 0.0 {
            Some(fmt_decimal(gross / total_equity))
        } else if total_equity > 0.0 {
            Some("0".to_string())
        } else {
            None
        };

        let margin_ratio = if margin_used > 0.0 && total_equity > 0.0 {
            Some(fmt_decimal(total_equity / margin_used))
        } else {
            None
        };

        let new_state = RustPortfolioState {
            total_equity: fmt_decimal(total_equity),
            cash_balance: fmt_decimal(balance),
            realized_pnl: fmt_decimal(realized_pnl),
            unrealized_pnl: fmt_decimal(unreal),
            fees_paid: fmt_decimal(fees_paid),
            gross_exposure: fmt_decimal(gross),
            net_exposure: fmt_decimal(net),
            leverage,
            margin_used: fmt_decimal(margin_used),
            margin_available: fmt_decimal(margin_available),
            margin_ratio,
            symbols: syms,
            last_ts: ts_str,
        };

        // Check if changed by comparing key fields
        let changed = new_state.total_equity != state.total_equity
            || new_state.gross_exposure != state.gross_exposure
            || new_state.net_exposure != state.net_exposure
            || new_state.unrealized_pnl != state.unrealized_pnl
            || new_state.leverage != state.leverage
            || new_state.symbols != state.symbols
            || new_state.cash_balance != state.cash_balance
            || new_state.realized_pnl != state.realized_pnl
            || new_state.fees_paid != state.fees_paid
            || new_state.margin_used != state.margin_used
            || new_state.margin_available != state.margin_available
            || new_state.margin_ratio != state.margin_ratio;

        Ok(RustReducerResult {
            state: new_state.into_pyobject(py)?.into_any().unbind(),
            changed,
            note: Some("portfolio_recompute".to_string()),
        })
    }
}

// ===========================================================================
// RustRiskReducer
// ===========================================================================

#[pyclass(name = "RustRiskReducer")]
pub struct RustRiskReducer {
    limits: RustRiskLimits,
    get_portfolio: PyObject,
    get_positions: PyObject,
}

#[pymethods]
impl RustRiskReducer {
    #[new]
    #[pyo3(signature = (*, limits, get_portfolio, get_positions))]
    fn new(limits: RustRiskLimits, get_portfolio: PyObject, get_positions: PyObject) -> Self {
        Self {
            limits,
            get_portfolio,
            get_positions,
        }
    }

    fn reduce(
        &self,
        py: Python<'_>,
        state: &RustRiskState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let et = get_event_type(event)?;
        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        // Get current portfolio
        let portfolio_obj = self.get_portfolio.call0(py)?;
        let portfolio: RustPortfolioState = portfolio_obj.extract(py)?;
        let equity: f64 = portfolio.total_equity.parse().unwrap_or(0.0);

        // Parse limits
        let max_leverage: f64 = self.limits.max_leverage.parse().unwrap_or(5.0);
        let max_drawdown_pct: f64 = self.limits.max_drawdown_pct.parse().unwrap_or(0.30);
        let max_position_notional: Option<f64> = self
            .limits
            .max_position_notional
            .as_ref()
            .and_then(|s| s.parse().ok());

        // Peak/drawdown tracking
        let mut peak: f64 = state.equity_peak.parse().unwrap_or(0.0);
        if equity > peak {
            peak = equity;
        }

        let mut dd: f64 = 0.0;
        if peak > 0.0 {
            dd = (peak - equity) / peak;
            if dd < 0.0 {
                dd = 0.0;
            }
        }

        let mut halted = state.halted;
        let mut level = state.level.clone();
        let mut message = state.message.clone();
        let mut flags: HashSet<String> = state.flags.iter().cloned().collect();

        // Manual risk events
        if et == "risk" {
            if let Ok(lvl) = event.getattr("level") {
                if !lvl.is_none() {
                    let l = lvl.str()?.to_string().trim().to_lowercase();
                    level = Some(l.clone());
                    if l == "block" {
                        flags.insert("risk_block_event".to_string());
                    }
                }
            }
            if let Ok(msg) = event.getattr("message") {
                if !msg.is_none() {
                    message = Some(msg.str()?.to_string());
                }
            }
        } else if et == "control" {
            let cmd = event
                .getattr("command")
                .ok()
                .filter(|v| !v.is_none())
                .or_else(|| event.getattr("action").ok().filter(|v| !v.is_none()));

            if let Some(c) = cmd {
                let cmd_s = c.str()?.to_string().trim().to_lowercase();
                match cmd_s.as_str() {
                    "halt" | "pause" | "stop" => {
                        halted = true;
                        flags.insert("manual_halt".to_string());
                    }
                    "resume" | "unpause" | "start" => {
                        halted = false;
                        flags.remove("manual_halt");
                    }
                    _ => {}
                }
            }
        }

        // Limit checks
        if self.limits.block_on_equity_le_zero && equity <= 0.0 {
            flags.insert("equity_le_zero".to_string());
        }

        let lev: Option<f64> = portfolio.leverage.as_ref().and_then(|s| s.parse().ok());
        if let Some(l) = lev {
            if l > max_leverage {
                flags.insert("max_leverage".to_string());
            }
        }

        if let Some(cap) = max_position_notional {
            let positions_obj = self.get_positions.call0(py)?;
            let positions: &Bound<'_, PyDict> = positions_obj.downcast_bound(py)?;
            for (_key, val) in positions.iter() {
                let pos: RustPositionState = val.extract()?;
                let qty: f64 = pos.qty.parse().unwrap_or(0.0);
                if qty == 0.0 {
                    continue;
                }
                if let Some(ref lp) = pos.last_price {
                    if let Ok(mark) = lp.parse::<f64>() {
                        let notional = qty.abs() * mark;
                        if notional > cap {
                            let sym = &pos.symbol;
                            flags.insert(format!("max_position_notional:{}", sym));
                            break;
                        }
                    }
                }
            }
        }

        if dd > max_drawdown_pct {
            flags.insert("max_drawdown".to_string());
        }

        // blocked = halted or any critical flags
        let blocked = halted
            || (!flags.is_empty()
                && (flags.contains("risk_block_event")
                    || flags.contains("equity_le_zero")
                    || flags.contains("max_drawdown")
                    || flags.contains("max_leverage")
                    || flags.iter().any(|f| f.starts_with("max_position_notional"))));

        let mut new_flags: Vec<String> = flags.into_iter().collect();
        new_flags.sort();

        let new_state = RustRiskState {
            blocked,
            halted,
            level,
            message,
            flags: new_flags.clone(),
            equity_peak: fmt_decimal(peak),
            drawdown_pct: fmt_decimal(dd),
            last_ts: ts_str.or_else(|| state.last_ts.clone()),
        };

        // Check changed
        let changed = new_state.blocked != state.blocked
            || new_state.halted != state.halted
            || new_state.level != state.level
            || new_state.message != state.message
            || new_state.flags != state.flags
            || new_state.equity_peak != state.equity_peak
            || new_state.drawdown_pct != state.drawdown_pct
            || new_state.last_ts != state.last_ts;

        Ok(RustReducerResult {
            state: new_state.into_pyobject(py)?.into_any().unbind(),
            changed,
            note: Some("risk_eval".to_string()),
        })
    }
}
