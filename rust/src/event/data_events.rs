//! Rust-native event types for the hot path.
//!
//! These replace Python SimpleNamespace / dataclass events in the pipeline.
//! When reducers receive a Rust event, they read fields directly instead of
//! going through PyO3 getattr — eliminating FFI overhead entirely.

use pyo3::prelude::*;

use crate::state::fixed_decimal::{Fd8, SCALE};

// ===========================================================================
// RustMarketEvent — replaces MarketEvent / SimpleNamespace for bar data
// ===========================================================================

#[pyclass(name = "RustMarketEvent", frozen)]
#[derive(Clone, Debug)]
pub struct RustMarketEvent {
    #[pyo3(get)]
    pub symbol: String,
    // Fd8 i64 fields (×10^8)
    pub open: i64,
    pub high: i64,
    pub low: i64,
    pub close: i64,
    pub volume: i64,
    #[pyo3(get)]
    pub ts: Option<String>,
}

#[pymethods]
impl RustMarketEvent {
    #[new]
    #[pyo3(signature = (symbol, open, high, low, close, volume, ts=None))]
    fn new(
        symbol: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        ts: Option<String>,
    ) -> Self {
        Self {
            symbol,
            open: Fd8::from_f64(open).raw(),
            high: Fd8::from_f64(high).raw(),
            low: Fd8::from_f64(low).raw(),
            close: Fd8::from_f64(close).raw(),
            volume: Fd8::from_f64(volume).raw(),
            ts,
        }
    }

    #[getter]
    fn open(&self) -> String {
        Fd8::from_raw(self.open).to_string_stripped()
    }

    #[getter]
    fn high(&self) -> String {
        Fd8::from_raw(self.high).to_string_stripped()
    }

    #[getter]
    fn low(&self) -> String {
        Fd8::from_raw(self.low).to_string_stripped()
    }

    #[getter]
    fn close(&self) -> String {
        Fd8::from_raw(self.close).to_string_stripped()
    }

    #[getter]
    fn volume(&self) -> String {
        Fd8::from_raw(self.volume).to_string_stripped()
    }

    #[getter]
    fn open_f(&self) -> f64 {
        self.open as f64 / SCALE as f64
    }

    #[getter]
    fn high_f(&self) -> f64 {
        self.high as f64 / SCALE as f64
    }

    #[getter]
    fn low_f(&self) -> f64 {
        self.low as f64 / SCALE as f64
    }

    #[getter]
    fn close_f(&self) -> f64 {
        self.close as f64 / SCALE as f64
    }

    #[getter]
    fn volume_f(&self) -> f64 {
        self.volume as f64 / SCALE as f64
    }

    /// event_type property for compat with Python event detection
    #[getter]
    fn event_type(&self) -> &str {
        "market"
    }

    /// header = None (Rust events don't carry headers in the reducer hot path)
    #[getter]
    fn header(&self) -> Option<PyObject> {
        None
    }

    fn __repr__(&self) -> String {
        format!(
            "RustMarketEvent(symbol={}, close={}, ts={:?})",
            self.symbol,
            Fd8::from_raw(self.close).to_string_stripped(),
            self.ts,
        )
    }
}

// ===========================================================================
// RustFillEvent — replaces FillEvent / SimpleNamespace for fill data
// ===========================================================================

#[pyclass(name = "RustFillEvent", frozen)]
#[derive(Clone, Debug)]
pub struct RustFillEvent {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: String,
    // Fd8 i64 fields (×10^8)
    pub qty: i64,
    pub price: i64,
    pub fee: i64,
    pub realized_pnl: i64,
    pub margin_change: i64,
    pub cash_delta: i64,
    #[pyo3(get)]
    pub ts: Option<String>,
}

#[pymethods]
impl RustFillEvent {
    #[new]
    #[pyo3(signature = (symbol, side, qty, price, fee=0.0, realized_pnl=0.0, margin_change=0.0, cash_delta=0.0, ts=None))]
    fn new(
        symbol: String,
        side: String,
        qty: f64,
        price: f64,
        fee: f64,
        realized_pnl: f64,
        margin_change: f64,
        cash_delta: f64,
        ts: Option<String>,
    ) -> Self {
        // Normalize side
        let side_norm = match side.to_lowercase().as_str() {
            "buy" | "long" => "buy".to_string(),
            "sell" | "short" => "sell".to_string(),
            _ => side.to_lowercase(),
        };
        Self {
            symbol,
            side: side_norm,
            qty: Fd8::from_f64(qty.abs()).raw(),
            price: Fd8::from_f64(price).raw(),
            fee: Fd8::from_f64(fee).raw(),
            realized_pnl: Fd8::from_f64(realized_pnl).raw(),
            margin_change: Fd8::from_f64(margin_change).raw(),
            cash_delta: Fd8::from_f64(cash_delta).raw(),
            ts,
        }
    }

    #[getter]
    fn qty(&self) -> String {
        Fd8::from_raw(self.qty).to_string_stripped()
    }

    #[getter]
    fn price(&self) -> String {
        Fd8::from_raw(self.price).to_string_stripped()
    }

    #[getter]
    fn fee(&self) -> String {
        Fd8::from_raw(self.fee).to_string_stripped()
    }

    #[getter]
    fn realized_pnl(&self) -> String {
        Fd8::from_raw(self.realized_pnl).to_string_stripped()
    }

    #[getter]
    fn margin_change(&self) -> String {
        Fd8::from_raw(self.margin_change).to_string_stripped()
    }

    #[getter]
    fn cash_delta(&self) -> String {
        Fd8::from_raw(self.cash_delta).to_string_stripped()
    }

    #[getter]
    fn qty_f(&self) -> f64 {
        self.qty as f64 / SCALE as f64
    }

    #[getter]
    fn price_f(&self) -> f64 {
        self.price as f64 / SCALE as f64
    }

    #[getter]
    fn event_type(&self) -> &str {
        "fill"
    }

    #[getter]
    fn header(&self) -> Option<PyObject> {
        None
    }

    fn __repr__(&self) -> String {
        format!(
            "RustFillEvent(symbol={}, side={}, qty={}, price={})",
            self.symbol,
            self.side,
            Fd8::from_raw(self.qty).to_string_stripped(),
            Fd8::from_raw(self.price).to_string_stripped(),
        )
    }
}

// ===========================================================================
// RustFundingEvent — replaces FundingEvent / SimpleNamespace for funding
// ===========================================================================

#[pyclass(name = "RustFundingEvent", frozen)]
#[derive(Clone, Debug)]
pub struct RustFundingEvent {
    #[pyo3(get)]
    pub symbol: String,
    pub funding_rate: i64,
    pub mark_price: i64,
    pub position_qty: i64,
    #[pyo3(get)]
    pub ts: Option<String>,
}

#[pymethods]
impl RustFundingEvent {
    #[new]
    #[pyo3(signature = (symbol, funding_rate, mark_price, position_qty, ts=None))]
    fn new(
        symbol: String,
        funding_rate: f64,
        mark_price: f64,
        position_qty: f64,
        ts: Option<String>,
    ) -> Self {
        Self {
            symbol,
            funding_rate: Fd8::from_f64(funding_rate).raw(),
            mark_price: Fd8::from_f64(mark_price).raw(),
            position_qty: Fd8::from_f64(position_qty).raw(),
            ts,
        }
    }

    #[getter]
    fn funding_rate(&self) -> String {
        Fd8::from_raw(self.funding_rate).to_string_stripped()
    }

    #[getter]
    fn mark_price(&self) -> String {
        Fd8::from_raw(self.mark_price).to_string_stripped()
    }

    #[getter]
    fn position_qty(&self) -> String {
        Fd8::from_raw(self.position_qty).to_string_stripped()
    }

    #[getter]
    fn funding_rate_f(&self) -> f64 {
        self.funding_rate as f64 / SCALE as f64
    }

    #[getter]
    fn mark_price_f(&self) -> f64 {
        self.mark_price as f64 / SCALE as f64
    }

    #[getter]
    fn position_qty_f(&self) -> f64 {
        self.position_qty as f64 / SCALE as f64
    }

    #[getter]
    fn event_type(&self) -> &str {
        "funding"
    }

    #[getter]
    fn header(&self) -> Option<PyObject> {
        None
    }

    fn __repr__(&self) -> String {
        format!(
            "RustFundingEvent(symbol={}, rate={}, mark={})",
            self.symbol,
            Fd8::from_raw(self.funding_rate).to_string_stripped(),
            Fd8::from_raw(self.mark_price).to_string_stripped(),
        )
    }
}
