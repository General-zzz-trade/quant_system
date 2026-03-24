// saga_manager.rs — RustSagaManager: Rust backend for engine/saga.py
//
// Order lifecycle saga state machine with TTL expiry and audit trail.
// Mirrors Python SagaManager 1:1 for parity.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, VecDeque};

// ── Saga state enum, internal types, transition validator ──
include!("saga_types.inc.rs");

// ============================================================
// RustSagaManager
// ============================================================

#[pyclass(name = "RustSagaManager")]
pub struct RustSagaManager {
    sagas: HashMap<String, OrderSaga>,
    completed: HashMap<String, OrderSaga>,
    completed_order: VecDeque<String>,
    max_completed: usize,
    default_ttl_sec: f64,
}

impl RustSagaManager {
    /// Move a saga from active to completed, evicting oldest if at capacity.
    fn move_to_completed(&mut self, order_id: &str) {
        if let Some(saga) = self.sagas.remove(order_id) {
            let oid = saga.order_id.clone();
            self.completed.insert(oid.clone(), saga);
            self.completed_order.push_back(oid);
            self.evict_completed();
        }
    }

    /// Evict oldest completed sagas beyond max_completed.
    fn evict_completed(&mut self) {
        while self.completed.len() > self.max_completed {
            if let Some(oldest) = self.completed_order.pop_front() {
                self.completed.remove(&oldest);
            } else {
                break;
            }
        }
    }

    /// Find a saga in active store, returning mutable ref.
    fn find_active(&mut self, order_id: &str) -> PyResult<&mut OrderSaga> {
        self.sagas.get_mut(order_id).ok_or_else(|| {
            PyValueError::new_err(format!("unknown saga: {}", order_id))
        })
    }

    /// Build a Python dict representing a saga.
    fn saga_to_dict<'py>(py: Python<'py>, saga: &OrderSaga) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("order_id", &saga.order_id)?;
        d.set_item("intent_id", &saga.intent_id)?;
        d.set_item("state", saga.state.as_str())?;
        d.set_item("symbol", &saga.symbol)?;
        d.set_item("side", &saga.side)?;
        d.set_item("qty", saga.qty)?;
        d.set_item("filled_qty", saga.filled_qty)?;
        d.set_item("avg_fill_price", saga.avg_fill_price)?;
        d.set_item("fill_count", saga.fill_count)?;
        d.set_item("created_at", saga.created_at)?;
        d.set_item("submitted_at", saga.submitted_at)?;
        d.set_item("last_transition_at", saga.last_transition_at)?;
        d.set_item("ttl_sec", saga.ttl_sec)?;
        d.set_item("remaining_qty", (saga.qty - saga.filled_qty).max(0.0))?;
        let fill_ratio = if saga.qty > 0.0 {
            saga.filled_qty / saga.qty
        } else {
            0.0
        };
        d.set_item("fill_ratio", fill_ratio)?;
        d.set_item("is_terminal", saga.state.is_terminal())?;
        // meta as dict
        let meta_dict = PyDict::new(py);
        for (k, v) in &saga.meta {
            meta_dict.set_item(k, v)?;
        }
        d.set_item("meta", meta_dict)?;
        Ok(d)
    }

    /// Build transition history as a list of dicts.
    fn history_to_list<'py>(
        py: Python<'py>,
        history: &[SagaTransition],
    ) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Bound<'py, PyDict>> = history
            .iter()
            .map(|t| {
                let d = PyDict::new(py);
                d.set_item("from", t.from.as_str()).unwrap();
                d.set_item("to", t.to.as_str()).unwrap();
                d.set_item("reason", &t.reason).unwrap();
                d.set_item("timestamp", t.timestamp).unwrap();
                d
            })
            .collect();
        Ok(PyList::new(py, items)?)
    }
}

#[pymethods]
impl RustSagaManager {
    #[new]
    #[pyo3(signature = (max_completed=10000, default_ttl_sec=120.0))]
    fn new(max_completed: usize, default_ttl_sec: f64) -> Self {
        RustSagaManager {
            sagas: HashMap::new(),
            completed: HashMap::new(),
            completed_order: VecDeque::new(),
            max_completed,
            default_ttl_sec,
        }
    }

    /// Create a new saga in PENDING state.
    #[pyo3(signature = (order_id, intent_id, symbol, side, qty, ttl_sec=None, timestamp=0.0))]
    fn create(
        &mut self,
        order_id: &str,
        intent_id: &str,
        symbol: &str,
        side: &str,
        qty: f64,
        ttl_sec: Option<f64>,
        timestamp: f64,
    ) -> PyResult<()> {
        if self.sagas.contains_key(order_id) || self.completed.contains_key(order_id) {
            return Err(PyValueError::new_err(format!(
                "saga already exists: {}",
                order_id
            )));
        }

        let saga = OrderSaga {
            order_id: order_id.to_string(),
            intent_id: intent_id.to_string(),
            state: SagaState::Pending,
            symbol: symbol.to_string(),
            side: side.to_string(),
            qty,
            filled_qty: 0.0,
            avg_fill_price: 0.0,
            fill_count: 0,
            created_at: timestamp,
            submitted_at: None,
            last_transition_at: timestamp,
            ttl_sec: ttl_sec.unwrap_or(self.default_ttl_sec),
            history: Vec::new(),
            meta: HashMap::new(),
        };
        self.sagas.insert(order_id.to_string(), saga);
        Ok(())
    }

    /// Transition to a new state with validation.
    /// Returns the new state string.
    #[pyo3(signature = (order_id, new_state, reason="", timestamp=0.0))]
    fn transition(
        &mut self,
        order_id: &str,
        new_state: &str,
        reason: &str,
        timestamp: f64,
    ) -> PyResult<String> {
        let target = SagaState::from_str(new_state)?;

        // Must be in active sagas
        let saga = self.sagas.get_mut(order_id).ok_or_else(|| {
            PyValueError::new_err(format!("unknown saga: {}", order_id))
        })?;

        // Validate transition
        if !is_valid_transition(saga.state, target) {
            return Err(PyValueError::new_err(format!(
                "invalid transition: {} -> {} for order {}",
                saga.state.as_str(),
                target.as_str(),
                order_id
            )));
        }

        // Record transition in history
        saga.history.push(SagaTransition {
            from: saga.state,
            to: target,
            reason: reason.to_string(),
            timestamp,
        });

        // Update state
        saga.state = target;
        saga.last_transition_at = timestamp;

        // Set submitted_at when transitioning TO Submitted
        if target == SagaState::Submitted {
            saga.submitted_at = Some(timestamp);
        }

        let state_str = target.as_str().to_string();

        // If terminal, move to completed
        if target.is_terminal() {
            self.move_to_completed(order_id);
        }

        Ok(state_str)
    }

    /// Record a fill (partial or full).
    /// Returns the new state string.
    #[pyo3(signature = (order_id, fill_qty, fill_price, timestamp=0.0))]
    fn record_fill(
        &mut self,
        order_id: &str,
        fill_qty: f64,
        fill_price: f64,
        timestamp: f64,
    ) -> PyResult<String> {
        // Validate state allows fill recording (ACKED or PARTIAL_FILL)
        {
            let saga = self.sagas.get(order_id).ok_or_else(|| {
                PyValueError::new_err(format!("unknown saga: {}", order_id))
            })?;

            if saga.state != SagaState::Acked && saga.state != SagaState::PartialFill {
                return Err(PyValueError::new_err(format!(
                    "cannot record fill in state {} for {}",
                    saga.state.as_str(),
                    order_id
                )));
            }
        }

        // Update fill tracking
        let (fully_filled, new_state_str);
        {
            let saga = self.sagas.get_mut(order_id).unwrap();
            let total_value =
                saga.avg_fill_price * saga.filled_qty + fill_price * fill_qty;
            saga.filled_qty += fill_qty;
            saga.fill_count += 1;
            if saga.filled_qty > 0.0 {
                saga.avg_fill_price = total_value / saga.filled_qty;
            }
            fully_filled = saga.filled_qty >= saga.qty;
        }

        // Auto-transition to PARTIAL_FILL or FILLED
        if fully_filled {
            new_state_str =
                self.transition(order_id, "filled", "fully filled", timestamp)?;
        } else {
            new_state_str =
                self.transition(order_id, "partial_fill", "partial fill", timestamp)?;
        }

        Ok(new_state_str)
    }

    /// Check for TTL-expired SUBMITTED sagas.
    /// Returns list of expired order IDs.
    /// Matches Python tick(): transitions expired to CANCELLED.
    fn check_timeouts(&mut self, now_ts: f64) -> Vec<String> {
        // Collect expired order IDs first
        let expired: Vec<String> = self
            .sagas
            .iter()
            .filter(|(_, saga)| {
                saga.state == SagaState::Submitted
                    && saga.submitted_at.is_some()
                    && (now_ts - saga.submitted_at.unwrap()) > saga.ttl_sec
            })
            .map(|(oid, _)| oid.clone())
            .collect();

        // Transition each to CANCELLED (matching Python's tick() behavior)
        for oid in &expired {
            // transition may fail if saga was already moved, but we ignore that
            let _ = self.transition(oid, "cancelled", "ttl_expired", now_ts);
            // Python also explicitly moves to completed after CANCELLED;
            // CANCELLED is not terminal in our model, so move explicitly.
            if self.sagas.contains_key(oid.as_str()) {
                self.move_to_completed(oid);
            }
        }

        expired
    }

    /// Get current state of a saga (active or completed).
    fn get_state(&self, order_id: &str) -> PyResult<String> {
        if let Some(saga) = self.sagas.get(order_id) {
            return Ok(saga.state.as_str().to_string());
        }
        if let Some(saga) = self.completed.get(order_id) {
            return Ok(saga.state.as_str().to_string());
        }
        Err(PyValueError::new_err(format!(
            "unknown saga: {}",
            order_id
        )))
    }

    /// Get saga details as a Python dict, or None if not found.
    fn get_saga(&self, py: Python<'_>, order_id: &str) -> PyResult<Option<PyObject>> {
        let saga = self
            .sagas
            .get(order_id)
            .or_else(|| self.completed.get(order_id));

        match saga {
            Some(s) => {
                let d = Self::saga_to_dict(py, s)?;
                Ok(Some(d.into()))
            }
            None => Ok(None),
        }
    }

    /// Count of active (non-terminal) sagas.
    fn active_count(&self) -> usize {
        self.sagas.len()
    }

    /// Count of completed sagas.
    fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Get all active order IDs for a symbol.
    fn by_symbol(&self, symbol: &str) -> Vec<String> {
        self.sagas
            .values()
            .filter(|s| s.symbol == symbol)
            .map(|s| s.order_id.clone())
            .collect()
    }

    /// Get all order IDs (active + completed) for an intent.
    fn by_intent(&self, intent_id: &str) -> Vec<String> {
        let mut result: Vec<String> = self
            .sagas
            .values()
            .filter(|s| s.intent_id == intent_id)
            .map(|s| s.order_id.clone())
            .collect();
        for saga in self.completed.values() {
            if saga.intent_id == intent_id {
                result.push(saga.order_id.clone());
            }
        }
        result
    }

    /// Get transition history as list of dicts.
    fn get_history(&self, py: Python<'_>, order_id: &str) -> PyResult<PyObject> {
        let saga = self
            .sagas
            .get(order_id)
            .or_else(|| self.completed.get(order_id));

        match saga {
            Some(s) => {
                let list = Self::history_to_list(py, &s.history)?;
                Ok(list.into())
            }
            None => Err(PyValueError::new_err(format!(
                "unknown saga: {}",
                order_id
            ))),
        }
    }

    /// Prune completed sagas beyond max_completed.
    /// Returns number of sagas pruned.
    fn prune_completed(&mut self) -> usize {
        let before = self.completed.len();
        self.evict_completed();
        before - self.completed.len()
    }
}
