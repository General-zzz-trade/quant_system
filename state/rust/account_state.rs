use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::state::fixed_decimal::SCALE;
use crate::state::type_helpers::{i64_repr, opt_str_eq, opt_str_repr};

// ===========================================================================
// AccountState — i64 fixed-point (x10^8)
// ===========================================================================
#[pyclass(name = "RustAccountState", frozen)]
#[derive(Clone)]
pub struct RustAccountState {
    #[pyo3(get)]
    pub currency: String,
    #[pyo3(get)]
    pub balance: i64,
    #[pyo3(get)]
    pub margin_used: i64,
    #[pyo3(get)]
    pub margin_available: i64,
    #[pyo3(get)]
    pub realized_pnl: i64,
    #[pyo3(get)]
    pub unrealized_pnl: i64,
    #[pyo3(get)]
    pub fees_paid: i64,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustAccountState {
    #[new]
    #[pyo3(signature = (currency, balance, margin_used=0, margin_available=0, realized_pnl=0, unrealized_pnl=0, fees_paid=0, last_ts=None))]
    fn new(
        currency: String,
        balance: i64,
        margin_used: i64,
        margin_available: i64,
        realized_pnl: i64,
        unrealized_pnl: i64,
        fees_paid: i64,
        last_ts: Option<String>,
    ) -> Self {
        Self {
            currency,
            balance,
            margin_used,
            margin_available,
            realized_pnl,
            unrealized_pnl,
            fees_paid,
            last_ts,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*, currency, balance))]
    fn initial(currency: String, balance: i64) -> Self {
        Self {
            currency,
            balance,
            margin_used: 0,
            margin_available: 0,
            realized_pnl: 0,
            unrealized_pnl: 0,
            fees_paid: 0,
            last_ts: None,
        }
    }

    #[pyo3(signature = (*, balance, margin_used, realized_pnl, unrealized_pnl, fees_paid, ts=None))]
    fn with_update(
        &self,
        balance: i64,
        margin_used: i64,
        realized_pnl: i64,
        unrealized_pnl: i64,
        fees_paid: i64,
        ts: Option<String>,
    ) -> Self {
        Self {
            currency: self.currency.clone(),
            balance,
            margin_used,
            margin_available: self.margin_available,
            realized_pnl,
            unrealized_pnl,
            fees_paid,
            last_ts: ts,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("currency", &self.currency)?;
        d.set_item("balance", self.balance)?;
        d.set_item("margin_used", self.margin_used)?;
        d.set_item("margin_available", self.margin_available)?;
        d.set_item("realized_pnl", self.realized_pnl)?;
        d.set_item("unrealized_pnl", self.unrealized_pnl)?;
        d.set_item("fees_paid", self.fees_paid)?;
        d.set_item("last_ts", &self.last_ts)?;
        Ok(d)
    }

    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            currency: d.get_item("currency")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "USDT".to_string()),
            balance: d.get_item("balance")?.and_then(|v| v.extract().ok()).unwrap_or(0),
            margin_used: d.get_item("margin_used")?.and_then(|v| v.extract().ok()).unwrap_or(0),
            margin_available: d.get_item("margin_available")?.and_then(|v| v.extract().ok()).unwrap_or(0),
            realized_pnl: d.get_item("realized_pnl")?.and_then(|v| v.extract().ok()).unwrap_or(0),
            unrealized_pnl: d.get_item("unrealized_pnl")?.and_then(|v| v.extract().ok()).unwrap_or(0),
            fees_paid: d.get_item("fees_paid")?.and_then(|v| v.extract().ok()).unwrap_or(0),
            last_ts: d.get_item("last_ts")?.and_then(|v| v.extract().ok()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RustAccountState(currency='{}', balance='{}', margin_used='{}', realized_pnl='{}', unrealized_pnl='{}', fees_paid='{}', last_ts={})",
            self.currency,
            i64_repr(self.balance),
            i64_repr(self.margin_used),
            i64_repr(self.realized_pnl),
            i64_repr(self.unrealized_pnl),
            i64_repr(self.fees_paid),
            opt_str_repr(&self.last_ts),
        )
    }

    // Float accessors for Python consumers
    #[getter]
    fn balance_f(&self) -> f64 {
        self.balance as f64 / SCALE as f64
    }

    #[getter]
    fn margin_used_f(&self) -> f64 {
        self.margin_used as f64 / SCALE as f64
    }

    #[getter]
    fn margin_available_f(&self) -> f64 {
        self.margin_available as f64 / SCALE as f64
    }

    #[getter]
    fn realized_pnl_f(&self) -> f64 {
        self.realized_pnl as f64 / SCALE as f64
    }

    #[getter]
    fn unrealized_pnl_f(&self) -> f64 {
        self.unrealized_pnl as f64 / SCALE as f64
    }

    #[getter]
    fn fees_paid_f(&self) -> f64 {
        self.fees_paid as f64 / SCALE as f64
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.currency == other.currency
            && self.balance == other.balance
            && self.margin_used == other.margin_used
            && self.margin_available == other.margin_available
            && self.realized_pnl == other.realized_pnl
            && self.unrealized_pnl == other.unrealized_pnl
            && self.fees_paid == other.fees_paid
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}
