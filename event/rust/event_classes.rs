use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::event::header::RustEventHeader;

// ============================================================
// RustSignalEvent — framework signal event with causation header
// ============================================================

#[pyclass(name = "RustSignalEvent", frozen)]
pub struct RustSignalEvent {
    #[pyo3(get)]
    pub header: Py<RustEventHeader>,
    #[pyo3(get)]
    pub signal_id: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: String, // "long", "short", "flat"
    #[pyo3(get)]
    pub strength: f64, // z-score or confidence
}

#[pymethods]
impl RustSignalEvent {
    #[new]
    fn new(
        header: Py<RustEventHeader>,
        signal_id: String,
        symbol: String,
        side: String,
        strength: f64,
    ) -> Self {
        Self {
            header,
            signal_id,
            symbol,
            side,
            strength,
        }
    }

    #[getter]
    fn event_type(&self) -> &str {
        "signal"
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("header", self.header.bind(py).borrow().serialize_to_dict(py)?)?;
        d.set_item("signal_id", &self.signal_id)?;
        d.set_item("symbol", &self.symbol)?;
        d.set_item("side", &self.side)?;
        d.set_item("strength", self.strength)?;
        d.set_item("event_type", "signal")?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustSignalEvent(signal_id='{}', symbol='{}', side='{}', strength={})",
            self.signal_id, self.symbol, self.side, self.strength
        )
    }
}

// ============================================================
// RustIntentEvent — trading intent before order generation
// ============================================================

#[pyclass(name = "RustIntentEvent", frozen)]
pub struct RustIntentEvent {
    #[pyo3(get)]
    pub header: Py<RustEventHeader>,
    #[pyo3(get)]
    pub intent_id: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: String, // "buy", "sell"
    #[pyo3(get)]
    pub target_qty: f64,
    #[pyo3(get)]
    pub reason_code: String,
    #[pyo3(get)]
    pub origin: String,
}

#[pymethods]
impl RustIntentEvent {
    #[new]
    fn new(
        header: Py<RustEventHeader>,
        intent_id: String,
        symbol: String,
        side: String,
        target_qty: f64,
        reason_code: String,
        origin: String,
    ) -> Self {
        Self {
            header,
            intent_id,
            symbol,
            side,
            target_qty,
            reason_code,
            origin,
        }
    }

    #[getter]
    fn event_type(&self) -> &str {
        "intent"
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("header", self.header.bind(py).borrow().serialize_to_dict(py)?)?;
        d.set_item("intent_id", &self.intent_id)?;
        d.set_item("symbol", &self.symbol)?;
        d.set_item("side", &self.side)?;
        d.set_item("target_qty", self.target_qty)?;
        d.set_item("reason_code", &self.reason_code)?;
        d.set_item("origin", &self.origin)?;
        d.set_item("event_type", "intent")?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustIntentEvent(intent_id='{}', symbol='{}', side='{}', qty={})",
            self.intent_id, self.symbol, self.side, self.target_qty
        )
    }
}

// ============================================================
// RustOrderEvent — order placed on venue
// ============================================================

#[pyclass(name = "RustOrderEvent", frozen)]
pub struct RustOrderEvent {
    #[pyo3(get)]
    pub header: Py<RustEventHeader>,
    #[pyo3(get)]
    pub order_id: String,
    #[pyo3(get)]
    pub intent_id: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: String, // "buy", "sell"
    #[pyo3(get)]
    pub qty: f64,
    #[pyo3(get)]
    pub price: Option<f64>, // None = market order
}

#[pymethods]
impl RustOrderEvent {
    #[new]
    #[pyo3(signature = (header, order_id, intent_id, symbol, side, qty, price=None))]
    fn new(
        header: Py<RustEventHeader>,
        order_id: String,
        intent_id: String,
        symbol: String,
        side: String,
        qty: f64,
        price: Option<f64>,
    ) -> Self {
        Self {
            header,
            order_id,
            intent_id,
            symbol,
            side,
            qty,
            price,
        }
    }

    #[getter]
    fn event_type(&self) -> &str {
        "order"
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("header", self.header.bind(py).borrow().serialize_to_dict(py)?)?;
        d.set_item("order_id", &self.order_id)?;
        d.set_item("intent_id", &self.intent_id)?;
        d.set_item("symbol", &self.symbol)?;
        d.set_item("side", &self.side)?;
        d.set_item("qty", self.qty)?;
        match self.price {
            Some(p) => d.set_item("price", p)?,
            None => d.set_item("price", py.None())?,
        }
        d.set_item("event_type", "order")?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        match self.price {
            Some(p) => format!(
                "RustOrderEvent(order_id='{}', symbol='{}', side='{}', qty={}, price={})",
                self.order_id, self.symbol, self.side, self.qty, p
            ),
            None => format!(
                "RustOrderEvent(order_id='{}', symbol='{}', side='{}', qty={}, price=MARKET)",
                self.order_id, self.symbol, self.side, self.qty
            ),
        }
    }
}

// ============================================================
// RustRiskEvent — risk rule evaluation result
// ============================================================

#[pyclass(name = "RustRiskEvent", frozen)]
pub struct RustRiskEvent {
    #[pyo3(get)]
    pub header: Py<RustEventHeader>,
    #[pyo3(get)]
    pub rule_id: String,
    #[pyo3(get)]
    pub level: String, // "info", "warn", "block"
    #[pyo3(get)]
    pub message: String,
}

#[pymethods]
impl RustRiskEvent {
    #[new]
    fn new(
        header: Py<RustEventHeader>,
        rule_id: String,
        level: String,
        message: String,
    ) -> Self {
        Self {
            header,
            rule_id,
            level,
            message,
        }
    }

    #[getter]
    fn event_type(&self) -> &str {
        "risk"
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("header", self.header.bind(py).borrow().serialize_to_dict(py)?)?;
        d.set_item("rule_id", &self.rule_id)?;
        d.set_item("level", &self.level)?;
        d.set_item("message", &self.message)?;
        d.set_item("event_type", "risk")?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustRiskEvent(rule_id='{}', level='{}', message='{}')",
            self.rule_id, self.level, self.message
        )
    }
}

// ============================================================
// RustControlEvent — system control commands
// ============================================================

#[pyclass(name = "RustControlEvent", frozen)]
pub struct RustControlEvent {
    #[pyo3(get)]
    pub header: Py<RustEventHeader>,
    #[pyo3(get)]
    pub command: String, // "halt", "reduce_only", "resume", "flush", "shutdown"
    #[pyo3(get)]
    pub reason: String,
}

#[pymethods]
impl RustControlEvent {
    #[new]
    fn new(
        header: Py<RustEventHeader>,
        command: String,
        reason: String,
    ) -> Self {
        Self {
            header,
            command,
            reason,
        }
    }

    #[getter]
    fn event_type(&self) -> &str {
        "control"
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("header", self.header.bind(py).borrow().serialize_to_dict(py)?)?;
        d.set_item("command", &self.command)?;
        d.set_item("reason", &self.reason)?;
        d.set_item("event_type", "control")?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustControlEvent(command='{}', reason='{}')",
            self.command, self.reason
        )
    }
}
