// order_limiter.inc.rs — RustOrderLimiter: rate + notional limits.
// Included by engine.rs via include!() macro.

#[pyclass]
pub struct RustOrderLimiter {
    max_order_qty: Option<f64>,
    max_order_notional: Option<f64>,
    max_position_notional: Option<f64>,
    max_orders_per_sec: Option<f64>,
    max_daily_orders: Option<u64>,
    max_daily_notional: Option<f64>,

    order_timestamps: Vec<f64>,    // monotonic timestamps for rate limiting
    daily_order_count: u64,
    daily_notional: f64,
    day_start: f64,
    start: Instant,
}

impl RustOrderLimiter {
    fn mono_now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn maybe_reset_daily(&mut self, now: f64) {
        if now - self.day_start > 86400.0 {
            self.daily_order_count = 0;
            self.daily_notional = 0.0;
            self.day_start = now;
        }
    }
}

#[pymethods]
impl RustOrderLimiter {
    #[new]
    #[pyo3(signature = (
        max_order_qty=None,
        max_order_notional=None,
        max_position_notional=None,
        max_orders_per_sec=None,
        max_daily_orders=None,
        max_daily_notional=None,
    ))]
    fn new(
        max_order_qty: Option<f64>,
        max_order_notional: Option<f64>,
        max_position_notional: Option<f64>,
        max_orders_per_sec: Option<f64>,
        max_daily_orders: Option<u64>,
        max_daily_notional: Option<f64>,
    ) -> Self {
        let start = Instant::now();
        Self {
            max_order_qty,
            max_order_notional,
            max_position_notional,
            max_orders_per_sec,
            max_daily_orders,
            max_daily_notional,
            order_timestamps: Vec::new(),
            daily_order_count: 0,
            daily_notional: 0.0,
            day_start: start.elapsed().as_secs_f64(),
            start,
        }
    }

    /// Pre-flight check. Returns (allowed, reason_or_none).
    #[pyo3(signature = (*, qty, price, current_position_notional=0.0))]
    fn check(
        &mut self,
        qty: f64,
        price: f64,
        current_position_notional: f64,
    ) -> (bool, Option<String>) {
        let now = self.mono_now();
        let notional = qty * price;

        self.maybe_reset_daily(now);

        // max_order_qty
        if let Some(max_q) = self.max_order_qty {
            if qty > max_q {
                return (
                    false,
                    Some(format!("max_order_qty: qty={} > max={}", qty, max_q)),
                );
            }
        }

        // max_order_notional
        if let Some(max_n) = self.max_order_notional {
            if notional > max_n {
                return (
                    false,
                    Some(format!(
                        "max_order_notional: notional={:.2} > max={:.2}",
                        notional, max_n
                    )),
                );
            }
        }

        // max_position_notional
        if let Some(max_p) = self.max_position_notional {
            let projected = current_position_notional + notional;
            if projected > max_p {
                return (
                    false,
                    Some(format!(
                        "max_position_notional: projected={:.2} > max={:.2}",
                        projected, max_p
                    )),
                );
            }
        }

        // max_orders_per_second (rate limit)
        if let Some(max_rate) = self.max_orders_per_sec {
            let cutoff = now - 1.0;
            self.order_timestamps.retain(|&t| t > cutoff);
            if self.order_timestamps.len() as f64 >= max_rate {
                return (
                    false,
                    Some(format!(
                        "max_orders_per_second: rate={}/s",
                        self.order_timestamps.len()
                    )),
                );
            }
        }

        // max_daily_orders
        if let Some(max_d) = self.max_daily_orders {
            if self.daily_order_count >= max_d {
                return (
                    false,
                    Some(format!(
                        "max_daily_orders: count={}",
                        self.daily_order_count
                    )),
                );
            }
        }

        // max_daily_notional
        if let Some(max_dn) = self.max_daily_notional {
            if self.daily_notional + notional > max_dn {
                return (
                    false,
                    Some(format!(
                        "max_daily_notional: total={:.2}",
                        self.daily_notional + notional
                    )),
                );
            }
        }

        (true, None)
    }

    /// Record an order (after it passes check and is sent).
    fn record_order(&mut self, notional: f64) {
        let now = self.mono_now();
        self.order_timestamps.push(now);
        self.daily_order_count += 1;
        self.daily_notional += notional;
    }

    /// Reset daily counters.
    fn reset_daily(&mut self) {
        self.daily_order_count = 0;
        self.daily_notional = 0.0;
        self.day_start = self.mono_now();
    }
}
