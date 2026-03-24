use pyo3::prelude::*;

use crate::state::fixed_decimal::{Fd8, opt_fd8_from_pyany};
use crate::event::data_events::RustMarketEvent;
use crate::state::market_state::RustMarketState;
use crate::state::reducer_result::RustReducerResult;
use crate::state::reducer_helpers::{
    InnerReducerResult, get_event_type, get_symbol, get_event_ts, ts_to_opt_string,
};

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
