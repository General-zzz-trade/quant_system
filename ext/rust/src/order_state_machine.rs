use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(name = "RustOrderTransition", frozen)]
#[derive(Clone, Debug)]
pub struct RustOrderTransition {
    from_status: String,
    to_status: String,
    ts_ms: i64,
    reason: String,
}

#[pymethods]
impl RustOrderTransition {
    #[getter]
    fn from_status(&self) -> &str {
        &self.from_status
    }

    #[getter]
    fn to_status(&self) -> &str {
        &self.to_status
    }

    #[getter]
    fn ts_ms(&self) -> i64 {
        self.ts_ms
    }

    #[getter]
    fn reason(&self) -> &str {
        &self.reason
    }
}

#[pyclass(name = "RustOrderState", frozen)]
#[derive(Clone, Debug)]
pub struct RustOrderState {
    order_id: String,
    client_order_id: Option<String>,
    symbol: String,
    side: String,
    order_type: String,
    status: String,
    qty: String,
    price: Option<String>,
    filled_qty: String,
    avg_price: Option<String>,
    last_update_ts: i64,
    transitions: Vec<RustOrderTransition>,
}

#[pymethods]
impl RustOrderState {
    #[getter]
    fn order_id(&self) -> &str {
        &self.order_id
    }

    #[getter]
    fn client_order_id(&self) -> Option<String> {
        self.client_order_id.clone()
    }

    #[getter]
    fn symbol(&self) -> &str {
        &self.symbol
    }

    #[getter]
    fn side(&self) -> &str {
        &self.side
    }

    #[getter]
    fn order_type(&self) -> &str {
        &self.order_type
    }

    #[getter]
    fn status(&self) -> &str {
        &self.status
    }

    #[getter]
    fn qty(&self) -> &str {
        &self.qty
    }

    #[getter]
    fn price(&self) -> Option<String> {
        self.price.clone()
    }

    #[getter]
    fn filled_qty(&self) -> &str {
        &self.filled_qty
    }

    #[getter]
    fn avg_price(&self) -> Option<String> {
        self.avg_price.clone()
    }

    #[getter]
    fn last_update_ts(&self) -> i64 {
        self.last_update_ts
    }

    #[getter]
    fn transitions(&self) -> Vec<RustOrderTransition> {
        self.transitions.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RustOrderState(order_id='{}', status='{}', symbol='{}', side='{}')",
            self.order_id, self.status, self.symbol, self.side
        )
    }
}

#[pyclass(name = "RustOrderStateMachine")]
pub struct RustOrderStateMachine {
    orders: HashMap<String, RustOrderState>,
    archived: HashMap<String, RustOrderState>,
}

#[pymethods]
impl RustOrderStateMachine {
    #[new]
    fn new() -> Self {
        Self {
            orders: HashMap::new(),
            archived: HashMap::new(),
        }
    }

    #[pyo3(signature = (order_id, symbol, side, order_type, qty, client_order_id=None, price=None))]
    fn register(
        &mut self,
        order_id: String,
        symbol: String,
        side: String,
        order_type: String,
        qty: String,
        client_order_id: Option<String>,
        price: Option<String>,
    ) -> PyResult<RustOrderState> {
        if self.orders.contains_key(&order_id) {
            return Err(PyRuntimeError::new_err(format!(
                "order {} already registered",
                order_id
            )));
        }
        let state = RustOrderState {
            order_id: order_id.clone(),
            client_order_id,
            symbol,
            side,
            order_type,
            status: "pending_new".to_string(),
            qty,
            price,
            filled_qty: "0".to_string(),
            avg_price: None,
            last_update_ts: 0,
            transitions: Vec::new(),
        };
        self.orders.insert(order_id, state.clone());
        Ok(state)
    }

    #[pyo3(signature = (order_id, new_status, filled_qty=None, avg_price=None, ts_ms=0, reason="".to_string()))]
    fn transition(
        &mut self,
        order_id: String,
        new_status: String,
        filled_qty: Option<String>,
        avg_price: Option<String>,
        ts_ms: i64,
        reason: String,
    ) -> PyResult<RustOrderState> {
        let state = match self.orders.get_mut(&order_id) {
            Some(state) => state,
            None => {
                return Err(PyRuntimeError::new_err(format!(
                    "unknown order: {}",
                    order_id
                )));
            }
        };

        let normalized = normalize_status(&new_status)?;
        if !is_transition_allowed(&state.status, &normalized) {
            return Err(PyRuntimeError::new_err(format!(
                "order {}: {} -> {} not allowed",
                order_id, state.status, normalized
            )));
        }

        state.transitions.push(RustOrderTransition {
            from_status: state.status.clone(),
            to_status: normalized.clone(),
            ts_ms,
            reason,
        });
        state.status = normalized;
        if let Some(v) = filled_qty {
            state.filled_qty = v;
        }
        if let Some(v) = avg_price {
            state.avg_price = Some(v);
        }
        state.last_update_ts = ts_ms;

        let out = state.clone();
        if is_terminal_status(&out.status) {
            self.archived.insert(order_id.clone(), out.clone());
            self.orders.remove(&order_id);
        }
        Ok(out)
    }

    fn get(&self, order_id: String) -> Option<RustOrderState> {
        self.orders
            .get(&order_id)
            .cloned()
            .or_else(|| self.archived.get(&order_id).cloned())
    }

    fn active_orders(&self) -> Vec<RustOrderState> {
        self.orders.values().cloned().collect()
    }

    fn active_count(&self) -> usize {
        self.orders.len()
    }
}

fn normalize_status(status: &str) -> PyResult<String> {
    let normalized = status.trim().to_ascii_lowercase();
    if is_known_status(&normalized) {
        Ok(normalized)
    } else {
        Err(PyRuntimeError::new_err(format!(
            "unknown order status: {}",
            status
        )))
    }
}

fn is_known_status(status: &str) -> bool {
    matches!(
        status,
        "pending_new"
            | "new"
            | "partially_filled"
            | "filled"
            | "pending_cancel"
            | "canceled"
            | "rejected"
            | "expired"
    )
}

fn is_terminal_status(status: &str) -> bool {
    matches!(status, "filled" | "canceled" | "rejected" | "expired")
}

fn is_transition_allowed(from_status: &str, to_status: &str) -> bool {
    match from_status {
        "pending_new" => matches!(to_status, "new" | "rejected" | "partially_filled" | "filled"),
        "new" => matches!(
            to_status,
            "partially_filled" | "filled" | "pending_cancel" | "canceled" | "expired"
        ),
        "partially_filled" => matches!(to_status, "partially_filled" | "filled" | "pending_cancel" | "canceled"),
        "pending_cancel" => matches!(to_status, "canceled" | "filled" | "partially_filled" | "rejected"),
        "filled" | "canceled" | "rejected" | "expired" => false,
        _ => false,
    }
}
