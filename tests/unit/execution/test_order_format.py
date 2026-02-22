"""Tests for order format validation."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from execution.models.commands import SubmitOrderCommand


def test_submit_order_command():
    cmd = SubmitOrderCommand(
        command_id="c1", created_ts=datetime.now(), actor="test",
        venue="binance", symbol="BTCUSDT", idempotency_key="ik1",
        side="buy", order_type="limit", qty=Decimal("0.1"),
        price=Decimal("50000"), tif="GTC",
    )
    assert cmd.symbol == "BTCUSDT"
    assert cmd.side == "buy"
