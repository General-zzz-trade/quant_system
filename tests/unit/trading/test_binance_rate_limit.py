"""The pure-Python Binance rate limiter matches the Rust policy it replaced (migration step 8)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from trading.adapters.binance.rate_limit import BinanceRateLimitPolicy, endpoint_weight

PATHS = ["/fapi/v1/order", "/fapi/v1/batchOrders", "/fapi/v1/allOpenOrders", "/fapi/v1/ticker/price",
         "/fapi/v1/exchangeInfo", "/fapi/v1/openOrders", "/fapi/v2/account", "/fapi/v2/positionRisk",
         "/fapi/v1/userTrades", "/fapi/v1/klines", "/fapi/v1/depth", "/fapi/v1/income", "/fapi/v1/unknown"]


RUST_DECISIONS = Path(__file__).resolve().parents[2] / "fixtures" / "rust_rate_limit_decisions.json"


@pytest.mark.parametrize("seed", range(5))
def test_matches_the_rust_policy_on_random_traffic(seed):
    """The Rust policy's decisions on this traffic were recorded on 2026-11-15, before the Rust extension was
    deleted (one character per check, "1" allowed): the comparison this test made live until then."""
    recorded = json.loads(RUST_DECISIONS.read_text())[str(seed)]
    rng = random.Random(seed)
    now = {"t": 1000.0}
    python = BinanceRateLimitPolicy(clock=lambda: now["t"])
    decisions = []
    for _ in range(3000):
        now["t"] += rng.choice([0.0, 0.0, 0.001, 0.01, 0.05, 0.2, 1.0, 5.0])
        if rng.random() < 0.02:
            python.sync_used_weight(rng.randrange(0, 2400))
            continue
        path = rng.choice(PATHS)
        # Open orders of one symbol: the Rust policy weighed every open-order list 1; since ADR-0011 an
        # account-wide list weighs 40 (tests/characterization/test_runtime_account_check.py), a per-symbol one 1.
        params = {"symbol": "BTCUSDT"} if path == "/fapi/v1/openOrders" else None
        decisions.append("1" if python.check(path, params) else "0")
    assert "".join(decisions) == recorded
    assert set(recorded) == {"0", "1"}                       # the traffic exercised both outcomes


def test_order_pool_allows_ten_per_second_and_refills():
    now = {"t": 100.0}
    policy = BinanceRateLimitPolicy(clock=lambda: now["t"])
    assert [policy.check("/fapi/v1/order") for _ in range(11)] == [True] * 10 + [False]
    now["t"] += 0.25                                  # refills 2.5 orders
    assert [policy.check("/fapi/v1/order") for _ in range(3)] == [True, True, False]


def test_weights_and_header_sync():
    assert [endpoint_weight(p) for p in ("/fapi/v1/order", "/fapi/v2/account", "/fapi/v1/income", "/x")] == \
        [0.0, 5.0, 30.0, 1.0]
    now = {"t": 100.0}
    policy = BinanceRateLimitPolicy(clock=lambda: now["t"])
    assert policy.check("/fapi/v1/ticker/price")     # the first request starts the refill clock
    policy.sync_used_weight(1198)                     # the exchange says only 2 weight is left
    assert [policy.check("/fapi/v1/ticker/price") for _ in range(3)] == [True, True, False]
