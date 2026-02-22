"""Tests for payload digest stability."""
from __future__ import annotations

from decimal import Decimal

from execution.adapters.binance.dedup_keys import payload_digest_for_fill


def test_digest_deterministic():
    kwargs = dict(
        symbol="BTCUSDT", order_id="123", trade_id="456",
        side="BUY", qty=Decimal("0.1"), price=Decimal("50000"),
        fee=Decimal("0.01"), fee_asset="USDT", ts_ms=1700000000000,
    )
    d1 = payload_digest_for_fill(**kwargs)
    d2 = payload_digest_for_fill(**kwargs)
    assert d1 == d2


def test_digest_changes_with_price():
    base = dict(
        symbol="BTCUSDT", order_id="123", trade_id="456",
        side="BUY", qty=Decimal("0.1"), price=Decimal("50000"),
        fee=Decimal("0.01"), fee_asset="USDT", ts_ms=1700000000000,
    )
    d1 = payload_digest_for_fill(**base)
    d2 = payload_digest_for_fill(**{**base, "price": Decimal("50001")})
    assert d1 != d2
