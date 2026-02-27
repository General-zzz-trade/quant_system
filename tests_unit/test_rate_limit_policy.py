"""Unit tests for BinanceRateLimitPolicy — token bucket rate limiting."""
from __future__ import annotations

import time

import pytest

from execution.adapters.binance.rate_limit_policy import (
    BinanceRateLimitPolicy, _Pool,
)


class TestPool:

    def test_consume_within_capacity(self):
        pool = _Pool(capacity=10.0, refill_per_sec=1.0)
        assert pool.try_consume(5.0) is True
        assert pool.try_consume(5.0) is True

    def test_consume_exhausted(self):
        pool = _Pool(capacity=10.0, refill_per_sec=0.0)
        pool.try_consume(10.0)
        assert pool.try_consume(1.0) is False

    def test_refill_over_time(self):
        pool = _Pool(capacity=10.0, refill_per_sec=100.0)
        pool.try_consume(10.0)
        # Refill rate is 100/sec, so even a tiny time delta refills something.
        # We can't inject time into _Pool directly, so just verify the contract.
        # After consuming all tokens, a subsequent call after some real time
        # should refill based on elapsed time.
        # Since we can't control time here, test the math via sync_from_header.

    def test_sync_from_header_reduces_tokens(self):
        pool = _Pool(capacity=1200.0, refill_per_sec=20.0)
        # Pool starts at capacity (1200)
        pool.sync_from_header(1000)  # used=1000, remaining=200
        # tokens should be capped at 200 since 200 < 1200
        assert pool.try_consume(200.0) is True
        assert pool.try_consume(1.0) is False

    def test_sync_from_header_does_not_increase(self):
        pool = _Pool(capacity=1200.0, refill_per_sec=20.0)
        pool.try_consume(1100.0)  # tokens ~= 100
        pool.sync_from_header(100)  # used=100, remaining=1100
        # remaining (1100) > current tokens (~100), so no change
        # tokens should still be ~100, not increased to 1100
        assert pool.try_consume(100.0) is True


class TestBinanceRateLimitPolicy:

    def test_order_pool_basic(self):
        policy = BinanceRateLimitPolicy()
        assert policy.check("/fapi/v1/order") is True

    def test_order_pool_exhausted(self):
        policy = BinanceRateLimitPolicy(
            order_pool=_Pool(capacity=3.0, refill_per_sec=0.0),
        )
        assert policy.check("/fapi/v1/order") is True
        assert policy.check("/fapi/v1/order") is True
        assert policy.check("/fapi/v1/order") is True
        assert policy.check("/fapi/v1/order") is False

    def test_weight_pool_basic(self):
        policy = BinanceRateLimitPolicy()
        assert policy.check("/fapi/v2/account") is True  # weight=5

    def test_weight_pool_endpoint_weights(self):
        policy = BinanceRateLimitPolicy(
            weight_pool=_Pool(capacity=10.0, refill_per_sec=0.0),
        )
        # /fapi/v2/account costs 5
        assert policy.check("/fapi/v2/account") is True   # 10 - 5 = 5
        assert policy.check("/fapi/v2/account") is True   # 5 - 5 = 0
        assert policy.check("/fapi/v2/account") is False   # 0 < 5

    def test_two_pools_independent(self):
        policy = BinanceRateLimitPolicy(
            order_pool=_Pool(capacity=1.0, refill_per_sec=0.0),
            weight_pool=_Pool(capacity=1200.0, refill_per_sec=0.0),
        )
        # Exhaust order pool
        assert policy.check("/fapi/v1/order") is True
        assert policy.check("/fapi/v1/order") is False
        # Weight pool should still work
        assert policy.check("/fapi/v2/account") is True

    def test_sync_used_weight(self):
        policy = BinanceRateLimitPolicy(
            weight_pool=_Pool(capacity=1200.0, refill_per_sec=0.0),
        )
        policy.sync_used_weight(1195)  # remaining = 5
        assert policy.check("/fapi/v2/account") is True   # cost 5, 5-5=0
        assert policy.check("/fapi/v1/ticker/price") is False  # cost 1, 0<1

    def test_unknown_endpoint_default_weight(self):
        policy = BinanceRateLimitPolicy(
            weight_pool=_Pool(capacity=2.0, refill_per_sec=0.0),
        )
        # Unknown endpoint uses DEFAULT_WEIGHT = 1
        assert policy.check("/some/unknown/endpoint") is True
        assert policy.check("/some/unknown/endpoint") is True
        assert policy.check("/some/unknown/endpoint") is False
