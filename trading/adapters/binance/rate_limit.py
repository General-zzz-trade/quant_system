"""Client-side Binance USD-M rate limiting: an order pool and a request-weight pool (token buckets).

Pure-Python port of the Rust ``RustRateLimitPolicy`` (common_rust/rate_limiter.rs) that the adapter
used before migration step 8: same capacities, refill rates, endpoint weights and arithmetic, with an
injectable monotonic clock. tests/unit/trading/test_binance_rate_limit.py checks parity with Rust.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

ORDER_ENDPOINTS = frozenset({"/fapi/v1/order", "/fapi/v1/batchOrders", "/fapi/v1/allOpenOrders"})
_WEIGHT_1 = frozenset({"/fapi/v1/ticker/price", "/fapi/v1/ticker/bookTicker", "/fapi/v1/exchangeInfo",
                       "/fapi/v1/openOrders"})
_WEIGHT_5 = frozenset({"/fapi/v2/account", "/fapi/v2/balance", "/fapi/v2/positionRisk", "/fapi/v1/allOrders",
                       "/fapi/v1/userTrades", "/fapi/v1/klines", "/fapi/v1/depth"})


def endpoint_weight(path: str) -> float:
    if path in ORDER_ENDPOINTS:
        return 0.0
    if path in _WEIGHT_5:
        return 5.0
    if path == "/fapi/v1/income":
        return 30.0
    return 1.0      # _WEIGHT_1 and every unknown endpoint


class _Pool:
    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        self.capacity, self.refill_per_sec = capacity, refill_per_sec
        self.tokens, self.last_ts = capacity, 0.0

    def try_consume(self, weight: float, now: float) -> bool:
        dt = now - self.last_ts
        self.last_ts = now
        self.tokens = min(self.tokens + dt * self.refill_per_sec, self.capacity)
        if self.tokens >= weight:
            self.tokens -= weight
            return True
        return False

    def sync_from_header(self, used_weight: int) -> None:
        remaining = self.capacity - float(used_weight)
        if remaining < self.tokens:
            self.tokens = max(remaining, 0.0)


class BinanceRateLimitPolicy:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._orders = _Pool(10.0, 10.0)
        self._weight = _Pool(1200.0, 20.0)
        self._clock = clock
        self._lock = threading.Lock()

    def check(self, path: str) -> bool:
        with self._lock:
            now = self._clock()
            if path in ORDER_ENDPOINTS:
                return self._orders.try_consume(1.0, now)
            return self._weight.try_consume(endpoint_weight(path), now)

    def sync_used_weight(self, used_weight: int) -> None:
        with self._lock:
            self._weight.sync_from_header(used_weight)


def make_rate_limit_policy() -> BinanceRateLimitPolicy:
    return BinanceRateLimitPolicy()
