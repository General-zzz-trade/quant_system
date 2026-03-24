use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::state::type_helpers::{opt_str_eq, opt_str_repr};

// ===========================================================================
// RiskLimits (keeps String — not on pipeline hot path)
// ===========================================================================
#[pyclass(name = "RustRiskLimits", frozen)]
#[derive(Clone)]
pub struct RustRiskLimits {
    #[pyo3(get)]
    pub max_leverage: String,
    #[pyo3(get)]
    pub max_position_notional: Option<String>,
    #[pyo3(get)]
    pub max_drawdown_pct: String,
    #[pyo3(get)]
    pub block_on_equity_le_zero: bool,
}

#[pymethods]
impl RustRiskLimits {
    #[new]
    #[pyo3(signature = (max_leverage="5".to_string(), max_position_notional=None, max_drawdown_pct="0.30".to_string(), block_on_equity_le_zero=true))]
    fn new(
        max_leverage: String,
        max_position_notional: Option<String>,
        max_drawdown_pct: String,
        block_on_equity_le_zero: bool,
    ) -> Self {
        Self {
            max_leverage,
            max_position_notional,
            max_drawdown_pct,
            block_on_equity_le_zero,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustRiskLimits(max_leverage='{}', max_position_notional={}, max_drawdown_pct='{}', block_on_equity_le_zero={})",
            self.max_leverage,
            opt_str_repr(&self.max_position_notional),
            self.max_drawdown_pct,
            self.block_on_equity_le_zero,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.max_leverage == other.max_leverage
            && opt_str_eq(&self.max_position_notional, &other.max_position_notional)
            && self.max_drawdown_pct == other.max_drawdown_pct
            && self.block_on_equity_le_zero == other.block_on_equity_le_zero
    }
}

// ===========================================================================
// RiskState (keeps String — not on pipeline hot path)
// ===========================================================================
#[pyclass(name = "RustRiskState", frozen)]
#[derive(Clone)]
pub struct RustRiskState {
    #[pyo3(get)]
    pub blocked: bool,
    #[pyo3(get)]
    pub halted: bool,
    #[pyo3(get)]
    pub level: Option<String>,
    #[pyo3(get)]
    pub message: Option<String>,
    #[pyo3(get)]
    pub flags: Vec<String>,
    #[pyo3(get)]
    pub equity_peak: String,
    #[pyo3(get)]
    pub drawdown_pct: String,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustRiskState {
    #[new]
    #[pyo3(signature = (blocked=false, halted=false, level=None, message=None, flags=vec![], equity_peak="0".to_string(), drawdown_pct="0".to_string(), last_ts=None))]
    fn new(
        blocked: bool,
        halted: bool,
        level: Option<String>,
        message: Option<String>,
        flags: Vec<String>,
        equity_peak: String,
        drawdown_pct: String,
        last_ts: Option<String>,
    ) -> Self {
        Self {
            blocked,
            halted,
            level,
            message,
            flags,
            equity_peak,
            drawdown_pct,
            last_ts,
        }
    }

    #[pyo3(signature = (*, blocked, halted, level, message, flags, equity_peak, drawdown_pct, ts=None))]
    fn with_update(
        &self,
        blocked: bool,
        halted: bool,
        level: Option<String>,
        message: Option<String>,
        flags: Vec<String>,
        equity_peak: String,
        drawdown_pct: String,
        ts: Option<String>,
    ) -> Self {
        Self {
            blocked,
            halted,
            level,
            message,
            flags,
            equity_peak,
            drawdown_pct,
            last_ts: ts,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("blocked", self.blocked)?;
        d.set_item("halted", self.halted)?;
        d.set_item("level", &self.level)?;
        d.set_item("message", &self.message)?;
        let flags_list = PyList::new(py, &self.flags)?;
        d.set_item("flags", flags_list)?;
        d.set_item("equity_peak", &self.equity_peak)?;
        d.set_item("drawdown_pct", &self.drawdown_pct)?;
        d.set_item("last_ts", &self.last_ts)?;
        Ok(d)
    }

    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let flags: Vec<String> = d.get_item("flags")?
            .map(|v| v.extract().unwrap_or_default())
            .unwrap_or_default();
        Ok(Self {
            blocked: d.get_item("blocked")?.and_then(|v| v.extract().ok()).unwrap_or(false),
            halted: d.get_item("halted")?.and_then(|v| v.extract().ok()).unwrap_or(false),
            level: d.get_item("level")?.and_then(|v| v.extract().ok()),
            message: d.get_item("message")?.and_then(|v| v.extract().ok()),
            flags,
            equity_peak: d.get_item("equity_peak")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            drawdown_pct: d.get_item("drawdown_pct")?.map(|v| v.extract()).transpose()?.unwrap_or_else(|| "0".to_string()),
            last_ts: d.get_item("last_ts")?.and_then(|v| v.extract().ok()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RustRiskState(blocked={}, halted={}, level={}, equity_peak='{}', drawdown_pct='{}', flags={:?})",
            self.blocked,
            self.halted,
            opt_str_repr(&self.level),
            self.equity_peak,
            self.drawdown_pct,
            self.flags,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.blocked == other.blocked
            && self.halted == other.halted
            && opt_str_eq(&self.level, &other.level)
            && opt_str_eq(&self.message, &other.message)
            && self.flags == other.flags
            && self.equity_peak == other.equity_peak
            && self.drawdown_pct == other.drawdown_pct
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}
