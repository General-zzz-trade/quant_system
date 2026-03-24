use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use crate::state::fixed_decimal::Fd8;
use crate::state::type_helpers::{opt_str_eq, opt_str_repr};
use crate::state::market_state::RustMarketState;
use crate::state::position_state::RustPositionState;
use crate::state::account_state::RustAccountState;

// ===========================================================================
// PortfolioState (keeps String — not on pipeline hot path)
// ===========================================================================
#[pyclass(name = "RustPortfolioState", frozen)]
#[derive(Clone)]
pub struct RustPortfolioState {
    #[pyo3(get)]
    pub total_equity: String,
    #[pyo3(get)]
    pub cash_balance: String,
    #[pyo3(get)]
    pub realized_pnl: String,
    #[pyo3(get)]
    pub unrealized_pnl: String,
    #[pyo3(get)]
    pub fees_paid: String,
    #[pyo3(get)]
    pub gross_exposure: String,
    #[pyo3(get)]
    pub net_exposure: String,
    #[pyo3(get)]
    pub leverage: Option<String>,
    #[pyo3(get)]
    pub margin_used: String,
    #[pyo3(get)]
    pub margin_available: String,
    #[pyo3(get)]
    pub margin_ratio: Option<String>,
    #[pyo3(get)]
    pub symbols: Vec<String>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustPortfolioState {
    #[new]
    #[pyo3(signature = (
        total_equity,
        cash_balance,
        realized_pnl,
        unrealized_pnl,
        fees_paid,
        gross_exposure,
        net_exposure,
        leverage,
        margin_used,
        margin_available,
        margin_ratio,
        symbols=vec![],
        last_ts=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        total_equity: String,
        cash_balance: String,
        realized_pnl: String,
        unrealized_pnl: String,
        fees_paid: String,
        gross_exposure: String,
        net_exposure: String,
        leverage: Option<String>,
        margin_used: String,
        margin_available: String,
        margin_ratio: Option<String>,
        symbols: Vec<String>,
        last_ts: Option<String>,
    ) -> Self {
        Self {
            total_equity,
            cash_balance,
            realized_pnl,
            unrealized_pnl,
            fees_paid,
            gross_exposure,
            net_exposure,
            leverage,
            margin_used,
            margin_available,
            margin_ratio,
            symbols,
            last_ts,
        }
    }

    /// Compute a PortfolioState from account + positions + market data.
    /// Now reads i64 fields from core types.
    #[staticmethod]
    #[pyo3(signature = (account, positions, market))]
    fn compute(
        account: &RustAccountState,
        positions: &Bound<'_, PyDict>,
        market: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let balance = Fd8::from_raw(account.balance);
        let margin_used = Fd8::from_raw(account.margin_used);
        let margin_available = Fd8::from_raw(account.margin_available);
        let realized_pnl = Fd8::from_raw(account.realized_pnl);
        let fees_paid = Fd8::from_raw(account.fees_paid);

        let mut gross_exposure = Fd8::ZERO;
        let mut net_exposure = Fd8::ZERO;
        let mut total_unrealized = Fd8::ZERO;
        let mut symbols: Vec<String> = Vec::new();
        let mut latest_ts: Option<String> = account.last_ts.clone();

        // Build market price lookup
        let mut market_prices: HashMap<String, Fd8> = HashMap::new();
        for (key, val) in market.iter() {
            let sym: String = key.extract()?;
            let ms: RustMarketState = val.extract()?;
            if let Some(p) = ms.last_price {
                market_prices.insert(sym, Fd8::from_raw(p));
            }
        }

        for (key, val) in positions.iter() {
            let sym: String = key.extract()?;
            let pos: RustPositionState = val.extract()?;
            let qty = Fd8::from_raw(pos.qty);
            if qty.is_zero() {
                continue;
            }
            symbols.push(sym.clone());

            let mark = pos
                .last_price
                .map(Fd8::from_raw)
                .or_else(|| market_prices.get(&sym).copied())
                .or_else(|| pos.avg_price.map(Fd8::from_raw))
                .unwrap_or(Fd8::ZERO);

            let notional = qty.abs() * mark;
            gross_exposure = gross_exposure + notional;
            net_exposure = net_exposure + qty * mark;

            if let Some(avg_raw) = pos.avg_price {
                let avg = Fd8::from_raw(avg_raw);
                total_unrealized = total_unrealized + qty * (mark - avg);
            }

            if pos.last_ts.is_some() && (latest_ts.is_none() || pos.last_ts > latest_ts) {
                latest_ts = pos.last_ts.clone();
            }
        }

        symbols.sort();

        let total_equity = balance + total_unrealized;
        let te_f = total_equity.to_f64();
        let ge_f = gross_exposure.to_f64();
        let mu_f = margin_used.to_f64();

        let leverage = if te_f.abs() > 1e-12 {
            Some(format!("{}", ge_f / te_f))
        } else {
            None
        };
        let margin_ratio = if mu_f > 1e-12 && te_f.abs() > 1e-12 {
            Some(format!("{}", te_f / mu_f))
        } else {
            None
        };

        Ok(Self {
            total_equity: total_equity.to_string_stripped(),
            cash_balance: balance.to_string_stripped(),
            realized_pnl: realized_pnl.to_string_stripped(),
            unrealized_pnl: total_unrealized.to_string_stripped(),
            fees_paid: fees_paid.to_string_stripped(),
            gross_exposure: gross_exposure.to_string_stripped(),
            net_exposure: net_exposure.to_string_stripped(),
            leverage,
            margin_used: margin_used.to_string_stripped(),
            margin_available: margin_available.to_string_stripped(),
            margin_ratio,
            symbols,
            last_ts: latest_ts,
        })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("total_equity", &self.total_equity)?;
        d.set_item("cash_balance", &self.cash_balance)?;
        d.set_item("realized_pnl", &self.realized_pnl)?;
        d.set_item("unrealized_pnl", &self.unrealized_pnl)?;
        d.set_item("fees_paid", &self.fees_paid)?;
        d.set_item("gross_exposure", &self.gross_exposure)?;
        d.set_item("net_exposure", &self.net_exposure)?;
        d.set_item("leverage", &self.leverage)?;
        d.set_item("margin_used", &self.margin_used)?;
        d.set_item("margin_available", &self.margin_available)?;
        d.set_item("margin_ratio", &self.margin_ratio)?;
        let sym_list = PyList::new(py, &self.symbols)?;
        d.set_item("symbols", sym_list)?;
        d.set_item("last_ts", &self.last_ts)?;
        Ok(d)
    }

    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let symbols: Vec<String> = d.get_item("symbols")?
            .map(|v| v.extract().unwrap_or_default())
            .unwrap_or_default();
        Ok(Self {
            total_equity: d.get_item("total_equity")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            cash_balance: d.get_item("cash_balance")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            realized_pnl: d.get_item("realized_pnl")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            unrealized_pnl: d.get_item("unrealized_pnl")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            fees_paid: d.get_item("fees_paid")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            gross_exposure: d.get_item("gross_exposure")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            net_exposure: d.get_item("net_exposure")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            leverage: d.get_item("leverage")?.and_then(|v| v.extract().ok()),
            margin_used: d.get_item("margin_used")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            margin_available: d.get_item("margin_available")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            margin_ratio: d.get_item("margin_ratio")?.and_then(|v| v.extract().ok()),
            symbols,
            last_ts: d.get_item("last_ts")?.and_then(|v| v.extract().ok()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RustPortfolioState(equity='{}', cash='{}', gross_exp='{}', net_exp='{}', leverage={}, symbols={:?})",
            self.total_equity,
            self.cash_balance,
            self.gross_exposure,
            self.net_exposure,
            opt_str_repr(&self.leverage),
            self.symbols,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.total_equity == other.total_equity
            && self.cash_balance == other.cash_balance
            && self.realized_pnl == other.realized_pnl
            && self.unrealized_pnl == other.unrealized_pnl
            && self.fees_paid == other.fees_paid
            && self.gross_exposure == other.gross_exposure
            && self.net_exposure == other.net_exposure
            && opt_str_eq(&self.leverage, &other.leverage)
            && self.margin_used == other.margin_used
            && self.margin_available == other.margin_available
            && opt_str_eq(&self.margin_ratio, &other.margin_ratio)
            && self.symbols == other.symbols
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}
