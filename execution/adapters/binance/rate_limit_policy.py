# execution/adapters/binance/rate_limit_policy.py
"""Per-endpoint rate limiting for Binance API.

Binance uses two independent rate limit pools:
- Order rate limit: 10 orders/second, 100 orders/10s
- Request weight limit: 1200 weight/minute (auto-ban at 2400)

This module maps endpoints to their weight costs and provides
per-pool token bucket enforcement.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


# Binance Futures endpoint weights (approximation)
# See: https://binance-docs.github.io/apidocs/futures/en/#limits
ORDER_ENDPOINTS = {
    "/fapi/v1/order",
    "/fapi/v1/batchOrders",
    "/fapi/v1/allOpenOrders",
}

ENDPOINT_WEIGHTS: Dict[str, int] = {
    # Order endpoints
    "/fapi/v1/order": 0,       # counted separately as order rate
    "/fapi/v1/batchOrders": 0,
    "/fapi/v1/allOpenOrders": 0,
    # Query endpoints
    "/fapi/v1/ticker/price": 1,
    "/fapi/v1/ticker/bookTicker": 1,
    "/fapi/v2/account": 5,
    "/fapi/v2/balance": 5,
    "/fapi/v2/positionRisk": 5,
    "/fapi/v1/openOrders": 1,
    "/fapi/v1/allOrders": 5,
    "/fapi/v1/userTrades": 5,
    "/fapi/v1/income": 30,
    "/fapi/v1/klines": 5,
    "/fapi/v1/depth": 5,
    "/fapi/v1/exchangeInfo": 1,
}

DEFAULT_WEIGHT = 1


@dataclass
class _Pool:
    """Token-bucket rate limiter for a single pool."""
    capacity: float
    refill_per_sec: float
    tokens: float = field(init=False)
    last_ts: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_ts = time.monotonic()

    def try_consume(self, weight: float) -> bool:
        with self._lock:
            now = time.monotonic()
            dt = now - self.last_ts
            self.last_ts = now
            self.tokens = min(self.capacity, self.tokens + dt * self.refill_per_sec)
            if self.tokens >= weight:
                self.tokens -= weight
                return True
            return False

    def sync_from_header(self, used_weight: int) -> None:
        """Calibrate from X-MBX-USED-WEIGHT response header."""
        with self._lock:
            remaining = self.capacity - used_weight
            if remaining < self.tokens:
                self.tokens = max(0.0, remaining)


@dataclass
class BinanceRateLimitPolicy:
    """Enforces Binance's per-pool rate limits.

    Two pools:
    - order_pool: 10 orders/second (burst 10)
    - weight_pool: 1200 weight/minute (20/sec refill)
    """

    order_pool: _Pool = field(default_factory=lambda: _Pool(capacity=10.0, refill_per_sec=10.0))
    weight_pool: _Pool = field(default_factory=lambda: _Pool(capacity=1200.0, refill_per_sec=20.0))

    def check(self, path: str) -> bool:
        """Check if request to path is allowed. Returns True if OK."""
        if path in ORDER_ENDPOINTS:
            return self.order_pool.try_consume(1.0)

        weight = ENDPOINT_WEIGHTS.get(path, DEFAULT_WEIGHT)
        return self.weight_pool.try_consume(weight)

    def sync_used_weight(self, used_weight: int) -> None:
        """Calibrate weight pool from X-MBX-USED-WEIGHT-1M header."""
        self.weight_pool.sync_from_header(used_weight)
