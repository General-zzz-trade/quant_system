use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::state::fixed_decimal::Fd8;
use crate::state::market_state::RustMarketState;
use crate::state::position_state::RustPositionState;
use crate::state::account_state::RustAccountState;
use crate::state::portfolio_state::RustPortfolioState;
use crate::state::reducer_result::RustReducerResult;
use crate::state::reducer_helpers::{get_event_ts, ts_to_opt_string};

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
