// risk_gate.inc.rs — RustRiskGate pre-execution risk gate.
// Included by engine.rs via include!() macro.

#[pyclass]
pub struct RustRiskGate {
    max_open_orders: u32,
    max_order_notional: Option<f64>,
    max_position_notional: Option<f64>,
    max_portfolio_notional: Option<f64>,
}

#[pymethods]
impl RustRiskGate {
    #[new]
    #[pyo3(signature = (
        max_open_orders=10,
        max_order_notional=None,
        max_position_notional=None,
        max_portfolio_notional=None,
    ))]
    fn new(
        max_open_orders: u32,
        max_order_notional: Option<f64>,
        max_position_notional: Option<f64>,
        max_portfolio_notional: Option<f64>,
    ) -> Self {
        Self {
            max_open_orders,
            max_order_notional,
            max_position_notional,
            max_portfolio_notional,
        }
    }

    /// Pre-execution risk gate check.
    /// Returns (allowed, reason_or_none).
    #[pyo3(signature = (
        *,
        symbol,
        side,
        qty,
        price,
        open_order_count=0,
        position_notional=0.0,
        portfolio_notional=0.0,
        kill_switch_armed=false,
    ))]
    #[allow(unused_variables)]
    fn check(
        &self,
        symbol: &str,
        side: &str,
        qty: f64,
        price: f64,
        open_order_count: u32,
        position_notional: f64,
        portfolio_notional: f64,
        kill_switch_armed: bool,
    ) -> (bool, Option<String>) {
        // 1. Kill switch
        if kill_switch_armed {
            return (false, Some("kill_switch_active".to_string()));
        }

        // 2. Open order limit
        if open_order_count >= self.max_open_orders {
            return (
                false,
                Some(format!(
                    "max_open_orders: {}>={}",
                    open_order_count, self.max_open_orders
                )),
            );
        }

        let notional = (qty * price).abs();

        // 3. Order notional
        if let Some(max_on) = self.max_order_notional {
            if notional > max_on {
                return (
                    false,
                    Some(format!(
                        "order_notional: {:.2}>{:.2}",
                        notional, max_on
                    )),
                );
            }
        }

        // 4. Position notional
        if let Some(max_pn) = self.max_position_notional {
            let projected = position_notional + notional;
            if projected > max_pn {
                return (
                    false,
                    Some(format!(
                        "position_notional: {:.2}>{:.2}",
                        projected, max_pn
                    )),
                );
            }
        }

        // 5. Portfolio notional
        if let Some(max_port) = self.max_portfolio_notional {
            let projected = portfolio_notional + notional;
            if projected > max_port {
                return (
                    false,
                    Some(format!(
                        "portfolio_notional: {:.2}>{:.2}",
                        projected, max_port
                    )),
                );
            }
        }

        (true, None)
    }
}
