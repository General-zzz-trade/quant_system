"""Tests for fill processing."""
from __future__ import annotations

from decimal import Decimal
from execution.models.fills import CanonicalFill


def test_canonical_fill_creation():
    fill = CanonicalFill(
        venue="binance", symbol="BTCUSDT",
        order_id="o1", trade_id="t1", fill_id="f1",
        side="buy", qty=Decimal("0.1"), price=Decimal("50000"),
        fee=Decimal("0.5"), fee_asset="USDT", ts_ms=1700000000000,
        payload_digest="abc123",
    )
    assert fill.fill_id == "f1"
    assert fill.qty == Decimal("0.1")
    assert fill.venue == "binance"
