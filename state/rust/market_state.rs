use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::state::fixed_decimal::SCALE;
use crate::state::type_helpers::{opt_i64_repr, opt_str_eq, opt_str_repr};

// ===========================================================================
// MarketState — i64 fixed-point (x10^8)
// ===========================================================================
#[pyclass(name = "RustMarketState", frozen)]
#[derive(Clone)]
pub struct RustMarketState {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub last_price: Option<i64>,
    #[pyo3(get)]
    pub open: Option<i64>,
    #[pyo3(get)]
    pub high: Option<i64>,
    #[pyo3(get)]
    pub low: Option<i64>,
    #[pyo3(get)]
    pub close: Option<i64>,
    #[pyo3(get)]
    pub volume: Option<i64>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustMarketState {
    #[new]
    #[pyo3(signature = (symbol, last_price=None, open=None, high=None, low=None, close=None, volume=None, last_ts=None))]
    fn new(
        symbol: String,
        last_price: Option<i64>,
        open: Option<i64>,
        high: Option<i64>,
        low: Option<i64>,
        close: Option<i64>,
        volume: Option<i64>,
        last_ts: Option<String>,
    ) -> Self {
        Self { symbol, last_price, open, high, low, close, volume, last_ts }
    }

    #[staticmethod]
    pub fn empty(symbol: String) -> Self {
        Self {
            symbol,
            last_price: None,
            open: None,
            high: None,
            low: None,
            close: None,
            volume: None,
            last_ts: None,
        }
    }

    #[pyo3(signature = (*, price, ts=None))]
    fn with_tick(&self, price: i64, ts: Option<String>) -> Self {
        Self {
            symbol: self.symbol.clone(),
            last_price: Some(price),
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            last_ts: ts,
        }
    }

    #[pyo3(signature = (*, o, h, l, c, v, ts=None))]
    fn with_bar(
        &self,
        o: i64,
        h: i64,
        l: i64,
        c: i64,
        v: i64,
        ts: Option<String>,
    ) -> Self {
        Self {
            symbol: self.symbol.clone(),
            last_price: Some(c),
            open: Some(o),
            high: Some(h),
            low: Some(l),
            close: Some(c),
            volume: Some(v),
            last_ts: ts,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("symbol", &self.symbol)?;
        d.set_item("last_price", self.last_price)?;
        d.set_item("open", self.open)?;
        d.set_item("high", self.high)?;
        d.set_item("low", self.low)?;
        d.set_item("close", self.close)?;
        d.set_item("volume", self.volume)?;
        d.set_item("last_ts", &self.last_ts)?;
        Ok(d)
    }

    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            symbol: d.get_item("symbol")?.map(|v| v.extract()).transpose()?.unwrap_or_default(),
            last_price: d.get_item("last_price")?.and_then(|v| v.extract().ok()),
            open: d.get_item("open")?.and_then(|v| v.extract().ok()),
            high: d.get_item("high")?.and_then(|v| v.extract().ok()),
            low: d.get_item("low")?.and_then(|v| v.extract().ok()),
            close: d.get_item("close")?.and_then(|v| v.extract().ok()),
            volume: d.get_item("volume")?.and_then(|v| v.extract().ok()),
            last_ts: d.get_item("last_ts")?.and_then(|v| v.extract().ok()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RustMarketState(symbol='{}', last_price={}, open={}, high={}, low={}, close={}, volume={}, last_ts={})",
            self.symbol,
            opt_i64_repr(&self.last_price),
            opt_i64_repr(&self.open),
            opt_i64_repr(&self.high),
            opt_i64_repr(&self.low),
            opt_i64_repr(&self.close),
            opt_i64_repr(&self.volume),
            opt_str_repr(&self.last_ts),
        )
    }

    // Float accessors for Python consumers (i64 / SCALE -> f64)
    #[getter]
    fn last_price_f(&self) -> Option<f64> {
        self.last_price.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn open_f(&self) -> Option<f64> {
        self.open.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn high_f(&self) -> Option<f64> {
        self.high.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn low_f(&self) -> Option<f64> {
        self.low.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn close_f(&self) -> Option<f64> {
        self.close.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn volume_f(&self) -> Option<f64> {
        self.volume.map(|v| v as f64 / SCALE as f64)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.symbol == other.symbol
            && self.last_price == other.last_price
            && self.open == other.open
            && self.high == other.high
            && self.low == other.low
            && self.close == other.close
            && self.volume == other.volume
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}
