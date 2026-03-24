use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::state::fixed_decimal::SCALE;
use crate::state::type_helpers::{i64_repr, opt_i64_repr, opt_str_eq, opt_str_repr};

// ===========================================================================
// PositionState — i64 fixed-point (x10^8)
// ===========================================================================
#[pyclass(name = "RustPositionState", frozen)]
#[derive(Clone)]
pub struct RustPositionState {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub qty: i64,
    #[pyo3(get)]
    pub avg_price: Option<i64>,
    #[pyo3(get)]
    pub last_price: Option<i64>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustPositionState {
    #[new]
    #[pyo3(signature = (symbol, qty=0, avg_price=None, last_price=None, last_ts=None))]
    fn new(
        symbol: String,
        qty: i64,
        avg_price: Option<i64>,
        last_price: Option<i64>,
        last_ts: Option<String>,
    ) -> Self {
        Self { symbol, qty, avg_price, last_price, last_ts }
    }

    #[staticmethod]
    pub fn empty(symbol: String) -> Self {
        Self {
            symbol,
            qty: 0,
            avg_price: None,
            last_price: None,
            last_ts: None,
        }
    }

    #[getter]
    fn is_flat(&self) -> bool {
        self.qty == 0
    }

    #[pyo3(signature = (*, qty, avg_price, last_price, ts=None))]
    fn with_update(
        &self,
        qty: i64,
        avg_price: Option<i64>,
        last_price: Option<i64>,
        ts: Option<String>,
    ) -> Self {
        Self {
            symbol: self.symbol.clone(),
            qty,
            avg_price,
            last_price,
            last_ts: ts,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("symbol", &self.symbol)?;
        d.set_item("qty", self.qty)?;
        d.set_item("avg_price", self.avg_price)?;
        d.set_item("last_price", self.last_price)?;
        d.set_item("last_ts", &self.last_ts)?;
        Ok(d)
    }

    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            symbol: d.get_item("symbol")?.map(|v| v.extract()).transpose()?.unwrap_or_default(),
            qty: d.get_item("qty")?.and_then(|v| v.extract().ok()).unwrap_or(0),
            avg_price: d.get_item("avg_price")?.and_then(|v| v.extract().ok()),
            last_price: d.get_item("last_price")?.and_then(|v| v.extract().ok()),
            last_ts: d.get_item("last_ts")?.and_then(|v| v.extract().ok()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RustPositionState(symbol='{}', qty='{}', avg_price={}, last_price={}, last_ts={})",
            self.symbol,
            i64_repr(self.qty),
            opt_i64_repr(&self.avg_price),
            opt_i64_repr(&self.last_price),
            opt_str_repr(&self.last_ts),
        )
    }

    // Float accessors for Python consumers
    #[getter]
    fn qty_f(&self) -> f64 {
        self.qty as f64 / SCALE as f64
    }

    #[getter]
    fn avg_price_f(&self) -> Option<f64> {
        self.avg_price.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn last_price_f(&self) -> Option<f64> {
        self.last_price.map(|v| v as f64 / SCALE as f64)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.symbol == other.symbol
            && self.qty == other.qty
            && self.avg_price == other.avg_price
            && self.last_price == other.last_price
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}
