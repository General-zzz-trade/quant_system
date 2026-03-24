use pyo3::prelude::*;

use crate::state::fixed_decimal::{Fd8, opt_fd8_from_pyany};
use crate::event::data_events::{RustFillEvent, RustFundingEvent};
use crate::state::account_state::RustAccountState;
use crate::state::reducer_result::RustReducerResult;
use crate::state::reducer_helpers::{
    InnerReducerResult, get_event_type, get_event_ts, ts_to_opt_string,
};

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
