//! RustStateStore: unified pipeline with state on the Rust heap.
//!
//! State lives in Rust memory. Python calls process_event() per event.
//! Only exports to Python dict on demand (decision cycles).
//!
//! This eliminates per-event Python↔Rust state conversion entirely.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::fixed_decimal::Fd8;
use crate::pipeline::detect_kind_inner;
use crate::rust_events::{RustMarketEvent, RustFillEvent, RustFundingEvent};
use crate::state_types::{RustAccountState, RustMarketState, RustPositionState};
use crate::state_reducers::{RustAccountReducer, RustMarketReducer, RustPositionReducer};

/// Result of processing a single event.
#[pyclass(name = "RustProcessResult", frozen)]
#[derive(Clone)]
pub struct RustProcessResult {
    #[pyo3(get)]
    pub advanced: bool,
    #[pyo3(get)]
    pub changed: bool,
    #[pyo3(get)]
    pub event_index: i64,
    #[pyo3(get)]
    pub kind: String,
}

/// Persistent state store that keeps all state on the Rust heap.
///
/// Usage:
///   handle = RustStateStore(["BTCUSDT"], "USDT", balance_i64)
///   result = handle.process_event(event, "BTCUSDT")
///   snapshot = handle.snapshot_dict()  # only when needed
#[pyclass(name = "RustStateStore")]
pub struct RustStateStore {
    markets: HashMap<String, RustMarketState>,
    positions: HashMap<String, RustPositionState>,
    account: RustAccountState,
    event_index: i64,
    mr: RustMarketReducer,
    pr: RustPositionReducer,
    ar: RustAccountReducer,
    last_event_id: Option<String>,
    last_ts: Option<String>,
}

#[pymethods]
impl RustStateStore {
    #[new]
    #[pyo3(signature = (symbols, currency, balance))]
    fn new(symbols: Vec<String>, currency: String, balance: i64) -> Self {
        let mut markets = HashMap::new();
        let mut positions = HashMap::new();
        for sym in &symbols {
            markets.insert(sym.clone(), RustMarketState::empty(sym.clone()));
            positions.insert(sym.clone(), RustPositionState::empty(sym.clone()));
        }
        Self {
            markets,
            positions,
            account: RustAccountState {
                currency,
                balance,
                margin_used: 0,
                margin_available: 0,
                realized_pnl: 0,
                unrealized_pnl: 0,
                fees_paid: 0,
                last_ts: None,
            },
            event_index: 0,
            mr: RustMarketReducer,
            pr: RustPositionReducer,
            ar: RustAccountReducer,
            last_event_id: None,
            last_ts: None,
        }
    }

    /// Process a single event. Returns a lightweight result.
    /// State mutations happen in-place on the Rust heap — no Python conversion.
    ///
    /// Fast path: if event is a RustMarketEvent/RustFillEvent/RustFundingEvent,
    /// reads fields directly from Rust struct — zero PyO3 getattr overhead.
    #[pyo3(signature = (event, symbol_default))]
    fn process_event(
        &mut self,
        py: Python<'_>,
        event: &Bound<'_, PyAny>,
        symbol_default: &str,
    ) -> PyResult<RustProcessResult> {
        // ── Fast path: try native Rust event types (zero getattr overhead) ──
        if let Ok(me) = event.downcast::<RustMarketEvent>() {
            return self.process_rust_market(&me.borrow());
        }
        if let Ok(fe) = event.downcast::<RustFillEvent>() {
            return self.process_rust_fill(&fe.borrow());
        }
        if let Ok(fe) = event.downcast::<RustFundingEvent>() {
            return self.process_rust_funding(&fe.borrow());
        }

        // ── Slow path: Python event objects (getattr) ──
        let kind = detect_kind_inner(event);

        // Non-fact events
        if !matches!(kind.as_str(), "MARKET" | "FILL" | "FUNDING" | "ORDER") {
            return Ok(RustProcessResult {
                advanced: false,
                changed: false,
                event_index: self.event_index,
                kind: String::new(),
            });
        }

        // Determine symbol
        let sym = if let Ok(s) = event.getattr("symbol") {
            if !s.is_none() {
                s.str()?.to_string()
            } else {
                symbol_default.to_string()
            }
        } else {
            symbol_default.to_string()
        };

        self.ensure_symbol(&sym);

        let market = self.markets.get(&sym).unwrap().clone();
        let position = self.positions.get(&sym).unwrap().clone();

        // Apply reducers — reduce_inner returns native Rust structs, no PyObject roundtrip
        let m_res = self.mr.reduce_inner(py, &market, event)?;
        let a_res = self.ar.reduce_inner(py, &self.account, event)?;
        let p_res = self.pr.reduce_inner(py, &position, event)?;

        let any_changed = m_res.changed || a_res.changed || p_res.changed;

        self.positions.insert(sym.clone(), p_res.state);
        self.markets.insert(sym, m_res.state);
        self.account = a_res.state;

        self.event_index += 1;

        // Track event id/ts
        if let Ok(header) = event.getattr("header") {
            if !header.is_none() {
                if let Ok(eid) = header.getattr("event_id") {
                    if !eid.is_none() {
                        if let Ok(s) = eid.extract::<String>() {
                            self.last_event_id = Some(s);
                        }
                    }
                }
                if let Ok(ts) = header.getattr("ts") {
                    if !ts.is_none() {
                        self.last_ts = Some(ts.str()?.to_string());
                    }
                }
            }
        }

        Ok(RustProcessResult {
            advanced: true,
            changed: any_changed,
            event_index: self.event_index,
            kind,
        })
    }

    /// Export current state as Python-accessible objects.
    /// Call this only on decision cycles, not every event.
    #[pyo3(signature = ())]
    fn get_markets(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (sym, state) in &self.markets {
            dict.set_item(sym, state.clone().into_pyobject(py)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    #[pyo3(signature = ())]
    fn get_positions(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (sym, state) in &self.positions {
            dict.set_item(sym, state.clone().into_pyobject(py)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    #[pyo3(signature = ())]
    fn get_account(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.account.clone().into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (symbol))]
    fn get_market(&self, py: Python<'_>, symbol: &str) -> PyResult<PyObject> {
        match self.markets.get(symbol) {
            Some(state) => Ok(state.clone().into_pyobject(py)?.into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    #[pyo3(signature = (symbol))]
    fn get_position(&self, py: Python<'_>, symbol: &str) -> PyResult<PyObject> {
        match self.positions.get(symbol) {
            Some(state) => Ok(state.clone().into_pyobject(py)?.into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    #[getter]
    fn event_index(&self) -> i64 {
        self.event_index
    }

    #[getter]
    fn last_event_id(&self) -> Option<String> {
        self.last_event_id.clone()
    }

    #[getter]
    fn last_ts(&self) -> Option<String> {
        self.last_ts.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RustStateStore(symbols={:?}, event_index={}, balance={})",
            self.markets.keys().collect::<Vec<_>>(),
            self.event_index,
            Fd8::from_raw(self.account.balance).to_string_stripped(),
        )
    }
}

// Private helper methods (not exposed to Python)
impl RustStateStore {
    fn ensure_symbol(&mut self, sym: &str) {
        if !self.markets.contains_key(sym) {
            self.markets.insert(sym.to_string(), RustMarketState::empty(sym.to_string()));
        }
        if !self.positions.contains_key(sym) {
            self.positions.insert(sym.to_string(), RustPositionState::empty(sym.to_string()));
        }
    }

    fn process_rust_market(&mut self, event: &RustMarketEvent) -> PyResult<RustProcessResult> {
        let sym = &event.symbol;
        self.ensure_symbol(sym);

        let market = self.markets.get(sym).unwrap().clone();
        let m_res = self.mr.reduce_rust_market(&market, event);

        let changed = m_res.changed;
        self.markets.insert(sym.clone(), m_res.state);
        self.event_index += 1;
        if let Some(ref ts) = event.ts {
            self.last_ts = Some(ts.clone());
        }

        Ok(RustProcessResult {
            advanced: true,
            changed,
            event_index: self.event_index,
            kind: "MARKET".to_string(),
        })
    }

    fn process_rust_fill(&mut self, event: &RustFillEvent) -> PyResult<RustProcessResult> {
        let sym = &event.symbol;
        self.ensure_symbol(sym);

        let position = self.positions.get(sym).unwrap().clone();
        let p_res = self.pr.reduce_rust_fill(&position, event)?;
        let a_res = self.ar.reduce_rust_fill(&self.account, event);

        let changed = p_res.changed || a_res.changed;
        self.positions.insert(sym.clone(), p_res.state);
        self.account = a_res.state;
        self.event_index += 1;
        if let Some(ref ts) = event.ts {
            self.last_ts = Some(ts.clone());
        }

        Ok(RustProcessResult {
            advanced: true,
            changed,
            event_index: self.event_index,
            kind: "FILL".to_string(),
        })
    }

    fn process_rust_funding(&mut self, event: &RustFundingEvent) -> PyResult<RustProcessResult> {
        let sym = &event.symbol;
        self.ensure_symbol(sym);

        let a_res = self.ar.reduce_rust_funding(&self.account, event);

        let changed = a_res.changed;
        self.account = a_res.state;
        self.event_index += 1;
        if let Some(ref ts) = event.ts {
            self.last_ts = Some(ts.clone());
        }

        Ok(RustProcessResult {
            advanced: true,
            changed,
            event_index: self.event_index,
            kind: "FUNDING".to_string(),
        })
    }
}
