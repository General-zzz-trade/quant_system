use pyo3::prelude::*;

/// Token-bucket pool. Mirrors rate_limit_policy.py _Pool.
struct Pool {
    capacity: f64,
    refill_per_sec: f64,
    tokens: f64,
    last_ts: f64,
}

impl Pool {
    fn new(capacity: f64, refill_per_sec: f64) -> Self {
        Self {
            capacity,
            refill_per_sec,
            tokens: capacity,
            last_ts: 0.0,
        }
    }

    fn try_consume(&mut self, weight: f64, now: f64) -> bool {
        let dt = now - self.last_ts;
        self.last_ts = now;
        self.tokens = (self.tokens + dt * self.refill_per_sec).min(self.capacity);
        if self.tokens >= weight {
            self.tokens -= weight;
            true
        } else {
            false
        }
    }

    fn sync_from_header(&mut self, used_weight: i64) {
        let remaining = self.capacity - used_weight as f64;
        if remaining < self.tokens {
            self.tokens = remaining.max(0.0);
        }
    }
}

/// Binance rate limit policy with two token-bucket pools.
/// Mirrors execution/adapters/binance/rate_limit_policy.py.
#[pyclass]
pub struct RustRateLimitPolicy {
    order_pool: Pool,
    weight_pool: Pool,
}

fn is_order_endpoint(path: &str) -> bool {
    matches!(
        path,
        "/fapi/v1/order" | "/fapi/v1/batchOrders" | "/fapi/v1/allOpenOrders"
    )
}

fn endpoint_weight(path: &str) -> f64 {
    match path {
        "/fapi/v1/order" | "/fapi/v1/batchOrders" | "/fapi/v1/allOpenOrders" => 0.0,
        "/fapi/v1/ticker/price" | "/fapi/v1/ticker/bookTicker" | "/fapi/v1/exchangeInfo" | "/fapi/v1/openOrders" => 1.0,
        "/fapi/v2/account" | "/fapi/v2/balance" | "/fapi/v2/positionRisk" | "/fapi/v1/allOrders" | "/fapi/v1/userTrades" | "/fapi/v1/klines" | "/fapi/v1/depth" => 5.0,
        "/fapi/v1/income" => 30.0,
        _ => 1.0,
    }
}

#[pymethods]
impl RustRateLimitPolicy {
    #[new]
    fn new() -> Self {
        Self {
            order_pool: Pool::new(10.0, 10.0),
            weight_pool: Pool::new(1200.0, 20.0),
        }
    }

    /// Check if request to path is allowed. `now` is monotonic time.
    fn check(&mut self, path: &str, now: f64) -> bool {
        if is_order_endpoint(path) {
            self.order_pool.try_consume(1.0, now)
        } else {
            let w = endpoint_weight(path);
            self.weight_pool.try_consume(w, now)
        }
    }

    /// Calibrate weight pool from X-MBX-USED-WEIGHT-1M header.
    fn sync_used_weight(&mut self, used_weight: i64) {
        self.weight_pool.sync_from_header(used_weight);
    }
}
