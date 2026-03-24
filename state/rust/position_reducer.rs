use pyo3::prelude::*;

use crate::state::fixed_decimal::{Fd8, opt_fd8_from_pyany};
use crate::event::data_events::RustFillEvent;
use crate::state::position_state::RustPositionState;
use crate::state::reducer_result::RustReducerResult;
use crate::state::reducer_helpers::{
    InnerReducerResult, get_event_type, get_symbol, get_event_ts, ts_to_opt_string, signed_qty,
};

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
