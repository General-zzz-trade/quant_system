use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// ============================================================
// Enum value lists (string constants, Python-compatible)
// ============================================================

#[pyfunction]
pub fn rust_event_types() -> Vec<String> {
    vec![
        "market".into(),
        "signal".into(),
        "intent".into(),
        "order".into(),
        "fill".into(),
        "risk".into(),
        "control".into(),
        "funding".into(),
    ]
}

#[pyfunction]
pub fn rust_sides() -> Vec<String> {
    vec!["buy".into(), "sell".into()]
}

#[pyfunction]
pub fn rust_signal_sides() -> Vec<String> {
    vec!["buy".into(), "sell".into(), "flat".into()]
}

#[pyfunction]
pub fn rust_venues() -> Vec<String> {
    vec!["BINANCE".into(), "BYBIT".into(), "SIM".into()]
}

#[pyfunction]
pub fn rust_order_types() -> Vec<String> {
    vec!["market".into(), "limit".into(), "stop".into(), "stop_limit".into()]
}

#[pyfunction]
pub fn rust_time_in_force() -> Vec<String> {
    vec!["GTC".into(), "IOC".into(), "FOK".into(), "GTX".into()]
}

// ============================================================
// RustSignalResult — mirrors decision/types.py SignalResult
// ============================================================

#[pyclass(frozen)]
#[derive(Clone)]
pub struct RustSignalResult {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: String, // "buy" | "sell" | "flat"
    #[pyo3(get)]
    pub score: f64, // signed strength
    #[pyo3(get)]
    pub confidence: f64, // [0, 1]
    #[pyo3(get)]
    pub meta: Option<HashMap<String, String>>,
}

#[pymethods]
impl RustSignalResult {
    #[new]
    #[pyo3(signature = (symbol, side, score, confidence=1.0, meta=None))]
    fn new(
        symbol: String,
        side: String,
        score: f64,
        confidence: f64,
        meta: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let valid_sides = ["buy", "sell", "flat"];
        if !valid_sides.contains(&side.as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid side '{}', must be one of: buy, sell, flat",
                side
            )));
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "confidence must be in [0, 1], got {}",
                confidence
            )));
        }
        Ok(Self {
            symbol,
            side,
            score,
            confidence,
            meta,
        })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("symbol", &self.symbol)?;
        d.set_item("side", &self.side)?;
        d.set_item("score", self.score)?;
        d.set_item("confidence", self.confidence)?;
        match &self.meta {
            Some(m) => {
                let md = PyDict::new(py);
                for (k, v) in m {
                    md.set_item(k, v)?;
                }
                d.set_item("meta", md)?;
            },
            None => d.set_item("meta", py.None())?,
        }
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustSignalResult(symbol='{}', side='{}', score={}, confidence={})",
            self.symbol, self.side, self.score, self.confidence
        )
    }
}

// ============================================================
// RustDecisionOutput — mirrors decision/types.py DecisionOutput
// ============================================================

#[pyclass(frozen)]
#[derive(Clone)]
pub struct RustDecisionOutput {
    #[pyo3(get)]
    pub ts_iso: String, // ISO timestamp string
    #[pyo3(get)]
    pub strategy_id: String,
    #[pyo3(get)]
    pub targets: Vec<RustTargetPosition>,
    #[pyo3(get)]
    pub orders: Vec<RustOrderSpec>,
}

#[pymethods]
impl RustDecisionOutput {
    #[new]
    fn new(
        ts_iso: String,
        strategy_id: String,
        targets: Vec<RustTargetPosition>,
        orders: Vec<RustOrderSpec>,
    ) -> Self {
        Self {
            ts_iso,
            strategy_id,
            targets,
            orders,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ts", &self.ts_iso)?;
        d.set_item("strategy_id", &self.strategy_id)?;
        let target_dicts: Vec<_> = self
            .targets
            .iter()
            .map(|t| t.to_dict(py))
            .collect::<PyResult<Vec<_>>>()?;
        d.set_item("targets", target_dicts)?;
        let order_dicts: Vec<_> = self
            .orders
            .iter()
            .map(|o| o.to_dict(py))
            .collect::<PyResult<Vec<_>>>()?;
        d.set_item("orders", order_dicts)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustDecisionOutput(ts='{}', strategy_id='{}', targets={}, orders={})",
            self.ts_iso,
            self.strategy_id,
            self.targets.len(),
            self.orders.len()
        )
    }
}

// ============================================================
// RustTargetPosition — mirrors decision/types.py TargetPosition
// ============================================================

#[pyclass(frozen)]
#[derive(Clone)]
pub struct RustTargetPosition {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub target_qty: f64, // signed target position qty
    #[pyo3(get)]
    pub reason_code: String,
    #[pyo3(get)]
    pub origin: String,
}

#[pymethods]
impl RustTargetPosition {
    #[new]
    #[pyo3(signature = (symbol, target_qty, reason_code="signal".to_string(), origin="decision".to_string()))]
    fn new(symbol: String, target_qty: f64, reason_code: String, origin: String) -> Self {
        Self {
            symbol,
            target_qty,
            reason_code,
            origin,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("symbol", &self.symbol)?;
        d.set_item("target_qty", self.target_qty)?;
        d.set_item("reason_code", &self.reason_code)?;
        d.set_item("origin", &self.origin)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustTargetPosition(symbol='{}', target_qty={}, reason_code='{}')",
            self.symbol, self.target_qty, self.reason_code
        )
    }
}

// ============================================================
// RustOrderSpec — mirrors decision/types.py OrderSpec
// ============================================================

#[pyclass(frozen)]
#[derive(Clone)]
pub struct RustOrderSpec {
    #[pyo3(get)]
    pub order_id: String,
    #[pyo3(get)]
    pub intent_id: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: String, // "buy" | "sell"
    #[pyo3(get)]
    pub qty: String, // Decimal as string (precision-safe)
    #[pyo3(get)]
    pub order_type: String, // "market" | "limit" | "stop" | "stop_limit"
    #[pyo3(get)]
    pub price: Option<String>, // Decimal as string, None for market orders
    #[pyo3(get)]
    pub tif: String, // time in force: GTC, IOC, FOK, GTX
    #[pyo3(get)]
    pub client_order_id: Option<String>,
}

#[pymethods]
impl RustOrderSpec {
    #[new]
    #[pyo3(signature = (order_id, intent_id, symbol, side, qty, order_type="limit".to_string(), price=None, tif="GTC".to_string(), client_order_id=None))]
    fn new(
        order_id: String,
        intent_id: String,
        symbol: String,
        side: String,
        qty: String,
        order_type: String,
        price: Option<String>,
        tif: String,
        client_order_id: Option<String>,
    ) -> PyResult<Self> {
        let valid_sides = ["buy", "sell"];
        if !valid_sides.contains(&side.as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid side '{}', must be buy or sell",
                side
            )));
        }
        let valid_types = ["market", "limit", "stop", "stop_limit"];
        if !valid_types.contains(&order_type.as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid order_type '{}', must be one of: market, limit, stop, stop_limit",
                order_type
            )));
        }
        let valid_tif = ["GTC", "IOC", "FOK", "GTX"];
        if !valid_tif.contains(&tif.as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid tif '{}', must be one of: GTC, IOC, FOK, GTX",
                tif
            )));
        }
        Ok(Self {
            order_id,
            intent_id,
            symbol,
            side,
            qty,
            order_type,
            price,
            tif,
            client_order_id,
        })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("order_id", &self.order_id)?;
        d.set_item("intent_id", &self.intent_id)?;
        d.set_item("symbol", &self.symbol)?;
        d.set_item("side", &self.side)?;
        d.set_item("qty", &self.qty)?;
        d.set_item("order_type", &self.order_type)?;
        match &self.price {
            Some(p) => d.set_item("price", p)?,
            None => d.set_item("price", py.None())?,
        }
        d.set_item("tif", &self.tif)?;
        match &self.client_order_id {
            Some(id) => d.set_item("client_order_id", id)?,
            None => d.set_item("client_order_id", py.None())?,
        }
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustOrderSpec(order_id='{}', symbol='{}', side='{}', qty='{}', type='{}')",
            self.order_id, self.symbol, self.side, self.qty, self.order_type
        )
    }
}
