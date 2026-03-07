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
from typing import Dict

from _quant_hotpath import RustRateLimitPolicy as _RustPolicy


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


class BinanceRateLimitPolicy:
    """Rust-accelerated rate limiter with same API as BinanceRateLimitPolicy."""

    def __init__(self) -> None:
        self._inner = _RustPolicy()
        self._lock = threading.Lock()

    def check(self, path: str) -> bool:
        with self._lock:
            return self._inner.check(path, time.monotonic())

    def sync_used_weight(self, used_weight: int) -> None:
        with self._lock:
            self._inner.sync_used_weight(used_weight)


def make_rate_limit_policy() -> BinanceRateLimitPolicy:
    """Factory: returns Rust-backed rate limit policy."""
    return BinanceRateLimitPolicy()
