use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

use crate::fixed_decimal::{Fd8, fd8_from_pyany, opt_fd8_from_pyany};
use crate::rust_events::{RustMarketEvent, RustFillEvent, RustFundingEvent};
use crate::state_types::{
    RustAccountState, RustMarketState, RustPortfolioState, RustPositionState, RustReducerResult,
    RustRiskLimits, RustRiskState,
};

/// Pure Rust result type — no PyObject allocation.
pub struct InnerReducerResult<S> {
    pub state: S,
    pub changed: bool,
    pub note: Option<String>,
}

// ===========================================================================
// Helper functions
// ===========================================================================

fn get_event_type(event: &Bound<'_, PyAny>) -> PyResult<String> {
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

fn get_symbol(event: &Bound<'_, PyAny>, default: &str) -> PyResult<String> {
    if let Ok(sym) = event.getattr("symbol") {
        if !sym.is_none() {
            return Ok(sym.str()?.to_string());
        }
    }
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

fn get_event_ts(event: &Bound<'_, PyAny>) -> PyResult<Option<PyObject>> {
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

/// Apply side to qty: buy/long -> +abs, sell/short -> -abs.
fn signed_qty(qty: Fd8, side: &str) -> Fd8 {
    let s = side.trim().to_lowercase();
    match s.as_str() {
        "buy" | "long" => qty.abs(),
        "sell" | "short" => -qty.abs(),
        _ => qty,
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

    pub fn reduce(
        &self,
        py: Python<'_>,
        state: &RustMarketState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let inner = self.reduce_inner(py, state, event)?;
        Ok(RustReducerResult {
            state: inner.state.into_pyobject(py)?.into_any().unbind(),
            changed: inner.changed,
            note: inner.note,
        })
    }
}

impl RustMarketReducer {
    /// Fast path: reduce a RustMarketEvent directly (no getattr overhead).
    pub fn reduce_rust_market(
        &self,
        state: &RustMarketState,
        event: &RustMarketEvent,
    ) -> InnerReducerResult<RustMarketState> {
        if event.symbol != state.symbol {
            return InnerReducerResult { state: state.clone(), changed: false, note: None };
        }
        let c = Fd8::from_raw(event.close);
        if !c.is_positive() {
            return InnerReducerResult { state: state.clone(), changed: false, note: None };
        }
        let h = Fd8::from_raw(event.high);
        let l = Fd8::from_raw(event.low);
        if h < l {
            return InnerReducerResult { state: state.clone(), changed: false, note: None };
        }
        let v = Fd8::from_raw(event.volume);
        if v.is_negative() {
            return InnerReducerResult { state: state.clone(), changed: false, note: None };
        }
        InnerReducerResult {
            state: RustMarketState {
                symbol: state.symbol.clone(),
                last_price: Some(event.close),
                open: Some(event.open),
                high: Some(event.high),
                low: Some(event.low),
                close: Some(event.close),
                volume: Some(event.volume),
                last_ts: event.ts.clone().or_else(|| state.last_ts.clone()),
            },
            changed: true,
            note: Some("market_bar".to_string()),
        }
    }

    pub fn reduce_inner(
        &self,
        py: Python<'_>,
        state: &RustMarketState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<InnerReducerResult<RustMarketState>> {
        let et = get_event_type(event)?;
        let sym = get_symbol(event, &state.symbol)?;

        if sym != state.symbol {
            return Ok(InnerReducerResult {
                state: state.clone(),
                changed: false,
                note: None,
            });
        }

        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        // Bar-like events
        if matches!(et.as_str(), "market" | "market_bar" | "bar" | "marketbar") {
            let mut o = opt_fd8_from_pyany(event, "open")?;
            let mut h = opt_fd8_from_pyany(event, "high")?;
            let mut l = opt_fd8_from_pyany(event, "low")?;
            let mut c = opt_fd8_from_pyany(event, "close")?;
            let mut v = opt_fd8_from_pyany(event, "volume")?;

            // Support nested bar object
            if c.is_none() {
                if let Ok(bar) = event.getattr("bar") {
                    if !bar.is_none() {
                        if o.is_none() { o = opt_fd8_from_pyany(&bar, "open")?; }
                        if h.is_none() { h = opt_fd8_from_pyany(&bar, "high")?; }
                        if l.is_none() { l = opt_fd8_from_pyany(&bar, "low")?; }
                        c = opt_fd8_from_pyany(&bar, "close")?;
                        if v.is_none() { v = opt_fd8_from_pyany(&bar, "volume")?; }
                    }
                }
            }

            if let Some(c_fd) = c {
                if !c_fd.is_positive() {
                    return Ok(InnerReducerResult { state: state.clone(), changed: false, note: None });
                }
                if let (Some(h_fd), Some(l_fd)) = (h, l) {
                    if h_fd < l_fd {
                        return Ok(InnerReducerResult { state: state.clone(), changed: false, note: None });
                    }
                }
                if let Some(v_fd) = v {
                    if v_fd.is_negative() {
                        return Ok(InnerReducerResult { state: state.clone(), changed: false, note: None });
                    }
                }

                let new_state = RustMarketState {
                    symbol: state.symbol.clone(),
                    last_price: Some(c_fd.raw()),
                    open: Some(o.unwrap_or(c_fd).raw()),
                    high: Some(h.unwrap_or(c_fd).raw()),
                    low: Some(l.unwrap_or(c_fd).raw()),
                    close: Some(c_fd.raw()),
                    volume: v.map(|fd| fd.raw()),
                    last_ts: ts_str.or_else(|| state.last_ts.clone()),
                };
                return Ok(InnerReducerResult {
                    state: new_state,
                    changed: true,
                    note: Some("market_bar".to_string()),
                });
            }

            let price = opt_fd8_from_pyany(event, "price")?;
            if let Some(p) = price {
                let new_state = RustMarketState {
                    symbol: state.symbol.clone(),
                    last_price: Some(p.raw()),
                    open: state.open,
                    high: state.high,
                    low: state.low,
                    close: state.close,
                    volume: state.volume,
                    last_ts: ts_str.or_else(|| state.last_ts.clone()),
                };
                return Ok(InnerReducerResult {
                    state: new_state,
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
            let price = opt_fd8_from_pyany(event, "price")?;
            match price {
                Some(p) => {
                    let new_state = RustMarketState {
                        symbol: state.symbol.clone(),
                        last_price: Some(p.raw()),
                        open: state.open,
                        high: state.high,
                        low: state.low,
                        close: state.close,
                        volume: state.volume,
                        last_ts: ts_str.or_else(|| state.last_ts.clone()),
                    };
                    Ok(InnerReducerResult {
                        state: new_state,
                        changed: true,
                        note: Some("market_tick".to_string()),
                    })
                }
                None => Err(pyo3::exceptions::PyValueError::new_err(
                    "tick event missing price",
                )),
            }
        } else {
            Ok(InnerReducerResult {
                state: state.clone(),
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

    pub fn reduce(
        &self,
        py: Python<'_>,
        state: &RustPositionState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let inner = self.reduce_inner(py, state, event)?;
        Ok(RustReducerResult {
            state: inner.state.into_pyobject(py)?.into_any().unbind(),
            changed: inner.changed,
            note: inner.note,
        })
    }
}

impl RustPositionReducer {
    /// Fast path: reduce a RustFillEvent directly (no getattr overhead).
    pub fn reduce_rust_fill(
        &self,
        state: &RustPositionState,
        event: &RustFillEvent,
    ) -> PyResult<InnerReducerResult<RustPositionState>> {
        if event.symbol != state.symbol {
            return Ok(InnerReducerResult { state: state.clone(), changed: false, note: None });
        }

        let qty_raw = Fd8::from_raw(event.qty);
        let qty = signed_qty(qty_raw, &event.side);
        if qty.is_zero() {
            return Err(pyo3::exceptions::PyValueError::new_err("fill qty cannot be 0"));
        }
        let price = Fd8::from_raw(event.price);
        if !price.is_positive() {
            return Err(pyo3::exceptions::PyValueError::new_err("fill price must be > 0"));
        }

        let prev_qty = Fd8::from_raw(state.qty);
        let new_qty = prev_qty + qty;

        if new_qty.abs().raw() < 1 {
            return Ok(InnerReducerResult {
                state: RustPositionState {
                    symbol: state.symbol.clone(), qty: 0, avg_price: None,
                    last_price: Some(price.raw()),
                    last_ts: event.ts.clone().or_else(|| state.last_ts.clone()),
                },
                changed: true,
                note: Some("position_flat".to_string()),
            });
        }
        if prev_qty.is_zero() || (prev_qty.is_positive() == qty.is_positive()) {
            let new_avg = if prev_qty.is_zero() || state.avg_price.is_none() {
                price
            } else {
                let old_avg = Fd8::from_raw(state.avg_price.unwrap_or(price.raw()));
                (prev_qty.abs() * old_avg + qty.abs() * price) / new_qty.abs()
            };
            return Ok(InnerReducerResult {
                state: RustPositionState {
                    symbol: state.symbol.clone(), qty: new_qty.raw(),
                    avg_price: Some(new_avg.raw()), last_price: Some(price.raw()),
                    last_ts: event.ts.clone().or_else(|| state.last_ts.clone()),
                },
                changed: true,
                note: Some("position_add".to_string()),
            });
        }
        if qty.abs() < prev_qty.abs() {
            return Ok(InnerReducerResult {
                state: RustPositionState {
                    symbol: state.symbol.clone(), qty: new_qty.raw(),
                    avg_price: state.avg_price, last_price: Some(price.raw()),
                    last_ts: event.ts.clone().or_else(|| state.last_ts.clone()),
                },
                changed: true,
                note: Some("position_reduce".to_string()),
            });
        }
        Ok(InnerReducerResult {
            state: RustPositionState {
                symbol: state.symbol.clone(), qty: new_qty.raw(),
                avg_price: Some(price.raw()), last_price: Some(price.raw()),
                last_ts: event.ts.clone().or_else(|| state.last_ts.clone()),
            },
            changed: true,
            note: Some("position_reverse".to_string()),
        })
    }

    pub fn reduce_inner(
        &self,
        py: Python<'_>,
        state: &RustPositionState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<InnerReducerResult<RustPositionState>> {
        let et = get_event_type(event)?;

        if !matches!(et.as_str(), "fill" | "trade_fill" | "execution_fill") {
            return Ok(InnerReducerResult { state: state.clone(), changed: false, note: None });
        }

        let sym = get_symbol(event, &state.symbol)?;
        if sym != state.symbol {
            return Ok(InnerReducerResult { state: state.clone(), changed: false, note: None });
        }

        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        let qty_raw = opt_fd8_from_pyany(event, "qty")?
            .or(opt_fd8_from_pyany(event, "quantity")?)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("fill event missing qty/quantity")
            })?;

        let side_obj = event.getattr("side").ok();
        let side_str = match &side_obj {
            Some(s) if !s.is_none() => s.str()?.to_string(),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "fill event missing side",
                ))
            }
        };

        let qty = signed_qty(qty_raw, &side_str);
        if qty.is_zero() {
            return Err(pyo3::exceptions::PyValueError::new_err("fill qty cannot be 0"));
        }

        let price = opt_fd8_from_pyany(event, "price")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("fill event missing price")
        })?;
        if !price.is_positive() {
            return Err(pyo3::exceptions::PyValueError::new_err("fill price must be > 0"));
        }

        let prev_qty = Fd8::from_raw(state.qty);
        let new_qty = prev_qty + qty;

        // 1) flat
        if new_qty.abs().raw() < 1 {
            return Ok(InnerReducerResult {
                state: RustPositionState {
                    symbol: state.symbol.clone(), qty: 0, avg_price: None,
                    last_price: Some(price.raw()),
                    last_ts: ts_str.or_else(|| state.last_ts.clone()),
                },
                changed: true,
                note: Some("position_flat".to_string()),
            });
        }

        // 2) opening or adding (same direction)
        if prev_qty.is_zero() || (prev_qty.is_positive() == qty.is_positive()) {
            let new_avg = if prev_qty.is_zero() || state.avg_price.is_none() {
                price
            } else {
                let old_avg = Fd8::from_raw(state.avg_price.unwrap_or(price.raw()));
                let num = prev_qty.abs() * old_avg + qty.abs() * price;
                num / new_qty.abs()
            };
            return Ok(InnerReducerResult {
                state: RustPositionState {
                    symbol: state.symbol.clone(), qty: new_qty.raw(),
                    avg_price: Some(new_avg.raw()), last_price: Some(price.raw()),
                    last_ts: ts_str.or_else(|| state.last_ts.clone()),
                },
                changed: true,
                note: Some("position_add".to_string()),
            });
        }

        // 3) reducing (opposite direction, not crossing zero)
        if qty.abs() < prev_qty.abs() {
            return Ok(InnerReducerResult {
                state: RustPositionState {
                    symbol: state.symbol.clone(), qty: new_qty.raw(),
                    avg_price: state.avg_price, last_price: Some(price.raw()),
                    last_ts: ts_str.or_else(|| state.last_ts.clone()),
                },
                changed: true,
                note: Some("position_reduce".to_string()),
            });
        }

        // 4) crossing zero -> reverse
        Ok(InnerReducerResult {
            state: RustPositionState {
                symbol: state.symbol.clone(), qty: new_qty.raw(),
                avg_price: Some(price.raw()), last_price: Some(price.raw()),
                last_ts: ts_str.or_else(|| state.last_ts.clone()),
            },
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

    pub fn reduce(
        &self,
        py: Python<'_>,
        state: &RustAccountState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let inner = self.reduce_inner(py, state, event)?;
        Ok(RustReducerResult {
            state: inner.state.into_pyobject(py)?.into_any().unbind(),
            changed: inner.changed,
            note: inner.note,
        })
    }
}

impl RustAccountReducer {
    /// Fast path: reduce a RustFillEvent directly (no getattr overhead).
    pub fn reduce_rust_fill(
        &self,
        state: &RustAccountState,
        event: &RustFillEvent,
    ) -> InnerReducerResult<RustAccountState> {
        let fee = Fd8::from_raw(event.fee);
        let realized = Fd8::from_raw(event.realized_pnl);
        let cash_delta = Fd8::from_raw(event.cash_delta);
        let margin_change = Fd8::from_raw(event.margin_change);

        let balance = Fd8::from_raw(state.balance);
        let margin_used = Fd8::from_raw(state.margin_used);
        let realized_pnl = Fd8::from_raw(state.realized_pnl);
        let fees_paid = Fd8::from_raw(state.fees_paid);

        let new_balance = balance + realized + cash_delta - fee;
        let new_margin_used = margin_used + margin_change;
        let new_margin_available = new_balance - new_margin_used;

        InnerReducerResult {
            state: RustAccountState {
                currency: state.currency.clone(),
                balance: new_balance.raw(),
                margin_used: new_margin_used.raw(),
                margin_available: new_margin_available.raw(),
                realized_pnl: (realized_pnl + realized).raw(),
                unrealized_pnl: state.unrealized_pnl,
                fees_paid: (fees_paid + fee).raw(),
                last_ts: event.ts.clone().or_else(|| state.last_ts.clone()),
            },
            changed: true,
            note: Some("fill_account".to_string()),
        }
    }

    /// Fast path: reduce a RustFundingEvent directly.
    pub fn reduce_rust_funding(
        &self,
        state: &RustAccountState,
        event: &RustFundingEvent,
    ) -> InnerReducerResult<RustAccountState> {
        let fr = Fd8::from_raw(event.funding_rate);
        let mp = Fd8::from_raw(event.mark_price);
        let pq = Fd8::from_raw(event.position_qty);

        if pq.is_zero() {
            return InnerReducerResult { state: state.clone(), changed: false, note: None };
        }

        let balance = Fd8::from_raw(state.balance);
        let margin_used = Fd8::from_raw(state.margin_used);
        let realized_pnl = Fd8::from_raw(state.realized_pnl);
        let fees_paid = Fd8::from_raw(state.fees_paid);

        let funding_payment = pq * mp * fr;
        let new_balance = balance - funding_payment;
        let new_margin_available = new_balance - margin_used;

        InnerReducerResult {
            state: RustAccountState {
                currency: state.currency.clone(),
                balance: new_balance.raw(),
                margin_used: margin_used.raw(),
                margin_available: new_margin_available.raw(),
                realized_pnl: (realized_pnl - funding_payment).raw(),
                unrealized_pnl: state.unrealized_pnl,
                fees_paid: (fees_paid + funding_payment.abs()).raw(),
                last_ts: event.ts.clone().or_else(|| state.last_ts.clone()),
            },
            changed: true,
            note: Some("funding_settlement".to_string()),
        }
    }

    pub fn reduce_inner(
        &self,
        py: Python<'_>,
        state: &RustAccountState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<InnerReducerResult<RustAccountState>> {
        let et = get_event_type(event)?;

        if et == "funding" {
            return self.reduce_funding_inner(py, state, event);
        }

        if !matches!(et.as_str(), "fill" | "trade_fill" | "execution_fill") {
            return Ok(InnerReducerResult { state: state.clone(), changed: false, note: None });
        }

        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        let fee = opt_fd8_from_pyany(event, "fee")?.unwrap_or(Fd8::ZERO);
        let realized = opt_fd8_from_pyany(event, "realized_pnl")?.unwrap_or(Fd8::ZERO);
        let cash_delta = opt_fd8_from_pyany(event, "cash_delta")?.unwrap_or(Fd8::ZERO);
        let margin_change = opt_fd8_from_pyany(event, "margin_change")?.unwrap_or(Fd8::ZERO);

        let balance = Fd8::from_raw(state.balance);
        let margin_used = Fd8::from_raw(state.margin_used);
        let realized_pnl = Fd8::from_raw(state.realized_pnl);
        let unrealized_pnl = Fd8::from_raw(state.unrealized_pnl);
        let fees_paid = Fd8::from_raw(state.fees_paid);

        let new_balance = balance + realized + cash_delta - fee;
        let new_margin_used = margin_used + margin_change;
        let new_margin_available = new_balance - new_margin_used;

        Ok(InnerReducerResult {
            state: RustAccountState {
                currency: state.currency.clone(),
                balance: new_balance.raw(),
                margin_used: new_margin_used.raw(),
                margin_available: new_margin_available.raw(),
                realized_pnl: (realized_pnl + realized).raw(),
                unrealized_pnl: unrealized_pnl.raw(),
                fees_paid: (fees_paid + fee).raw(),
                last_ts: ts_str.or_else(|| state.last_ts.clone()),
            },
            changed: true,
            note: Some("fill_account".to_string()),
        })
    }

    fn reduce_funding_inner(
        &self,
        py: Python<'_>,
        state: &RustAccountState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<InnerReducerResult<RustAccountState>> {
        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        let funding_rate = opt_fd8_from_pyany(event, "funding_rate")?;
        let mark_price = opt_fd8_from_pyany(event, "mark_price")?;
        let position_qty = opt_fd8_from_pyany(event, "position_qty")?;

        match (funding_rate, mark_price, position_qty) {
            (Some(fr), Some(mp), Some(pq)) if !pq.is_zero() => {
                let balance = Fd8::from_raw(state.balance);
                let margin_used = Fd8::from_raw(state.margin_used);
                let realized_pnl = Fd8::from_raw(state.realized_pnl);
                let unrealized_pnl = Fd8::from_raw(state.unrealized_pnl);
                let fees_paid = Fd8::from_raw(state.fees_paid);

                let funding_payment = pq * mp * fr;
                let new_balance = balance - funding_payment;
                let new_margin_available = new_balance - margin_used;

                Ok(InnerReducerResult {
                    state: RustAccountState {
                        currency: state.currency.clone(),
                        balance: new_balance.raw(),
                        margin_used: margin_used.raw(),
                        margin_available: new_margin_available.raw(),
                        realized_pnl: (realized_pnl - funding_payment).raw(),
                        unrealized_pnl: unrealized_pnl.raw(),
                        fees_paid: (fees_paid + funding_payment.abs()).raw(),
                        last_ts: ts_str.or_else(|| state.last_ts.clone()),
                    },
                    changed: true,
                    note: Some("funding_settlement".to_string()),
                })
            }
            _ => Ok(InnerReducerResult { state: state.clone(), changed: false, note: None }),
        }
    }
}

// ===========================================================================
// RustPortfolioReducer
// ===========================================================================

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

        let account_obj = self.get_account.call0(py)?;
        let account: RustAccountState = account_obj.extract(py)?;

        let positions_obj = self.get_positions.call0(py)?;
        let positions: &Bound<'_, PyDict> = positions_obj.downcast_bound(py)?;

        let market_obj = self.get_market.call0(py)?;
        let market: RustMarketState = market_obj.extract(py)?;

        // Read i64 fields from core types
        let balance = Fd8::from_raw(account.balance);
        let margin_used = Fd8::from_raw(account.margin_used);
        let margin_available = Fd8::from_raw(account.margin_available);
        let realized_pnl = Fd8::from_raw(account.realized_pnl);
        let fees_paid = Fd8::from_raw(account.fees_paid);

        let market_price: Option<Fd8> = market.last_price.map(Fd8::from_raw);

        let mut gross = Fd8::ZERO;
        let mut net = Fd8::ZERO;
        let mut unreal = Fd8::ZERO;
        let mut syms: Vec<String> = Vec::new();

        for (key, val) in positions.iter() {
            let sym: String = key.extract()?;
            let pos: RustPositionState = val.extract()?;
            let qty = Fd8::from_raw(pos.qty);
            if qty.is_zero() {
                continue;
            }

            let mark = if sym == market.symbol {
                market_price
            } else {
                None
            }
            .or_else(|| pos.last_price.map(Fd8::from_raw));

            if let Some(m) = mark {
                let notional = qty.abs() * m;
                gross = gross + notional;
                net = net + qty * m;

                if let Some(avg_raw) = pos.avg_price {
                    let avg = Fd8::from_raw(avg_raw);
                    unreal = unreal + (m - avg) * qty;
                }
            }

            syms.push(sym);
        }

        syms.sort();

        let total_equity = balance + unreal;
        let te_f = total_equity.to_f64();
        let ge_f = gross.to_f64();

        let leverage = if te_f > 0.0 && ge_f != 0.0 {
            Some(Fd8::from_f64(ge_f / te_f).to_string_stripped())
        } else if te_f > 0.0 {
            Some("0".to_string())
        } else {
            None
        };

        let mu_f = margin_used.to_f64();
        let margin_ratio = if mu_f > 0.0 && te_f > 0.0 {
            Some(Fd8::from_f64(te_f / mu_f).to_string_stripped())
        } else {
            None
        };

        let new_state = RustPortfolioState {
            total_equity: total_equity.to_string_stripped(),
            cash_balance: balance.to_string_stripped(),
            realized_pnl: realized_pnl.to_string_stripped(),
            unrealized_pnl: unreal.to_string_stripped(),
            fees_paid: fees_paid.to_string_stripped(),
            gross_exposure: gross.to_string_stripped(),
            net_exposure: net.to_string_stripped(),
            leverage,
            margin_used: margin_used.to_string_stripped(),
            margin_available: margin_available.to_string_stripped(),
            margin_ratio,
            symbols: syms,
            last_ts: ts_str,
        };

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

        let portfolio_obj = self.get_portfolio.call0(py)?;
        let portfolio: RustPortfolioState = portfolio_obj.extract(py)?;
        let equity: f64 = portfolio.total_equity.parse().unwrap_or(0.0);

        let max_leverage: f64 = self.limits.max_leverage.parse().unwrap_or(5.0);
        let max_drawdown_pct: f64 = self.limits.max_drawdown_pct.parse().unwrap_or(0.30);
        let max_position_notional: Option<f64> = self
            .limits
            .max_position_notional
            .as_ref()
            .and_then(|s| s.parse().ok());

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
                let qty = Fd8::from_raw(pos.qty);
                if qty.is_zero() {
                    continue;
                }
                if let Some(lp_raw) = pos.last_price {
                    let mark = Fd8::from_raw(lp_raw);
                    let notional = (qty.abs() * mark).to_f64();
                    if notional > cap {
                        let sym = &pos.symbol;
                        flags.insert(format!("max_position_notional:{}", sym));
                        break;
                    }
                }
            }
        }

        if dd > max_drawdown_pct {
            flags.insert("max_drawdown".to_string());
        }

        let blocked = halted
            || (!flags.is_empty()
                && (flags.contains("risk_block_event")
                    || flags.contains("equity_le_zero")
                    || flags.contains("max_drawdown")
                    || flags.contains("max_leverage")
                    || flags.iter().any(|f| f.starts_with("max_position_notional"))));

        let mut new_flags: Vec<String> = flags.into_iter().collect();
        new_flags.sort();

        let fmt_peak = Fd8::from_f64(peak).to_string_stripped();
        let fmt_dd = Fd8::from_f64(dd).to_string_stripped();

        let new_state = RustRiskState {
            blocked,
            halted,
            level,
            message,
            flags: new_flags.clone(),
            equity_peak: fmt_peak,
            drawdown_pct: fmt_dd,
            last_ts: ts_str.or_else(|| state.last_ts.clone()),
        };

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
