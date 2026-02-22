from __future__ import annotations

import pytest
from decimal import Decimal

from execution.bridge.request_ids import RequestIdFactory
from execution.models.commands import (
    make_submit_order_command,
    make_cancel_order_command,
)


def test_make_submit_command_is_stable_for_same_logical_id() -> None:
    rid = RequestIdFactory(namespace="qsys", run_id="run-001", deterministic=True)

    c1 = make_submit_order_command(
        rid=rid,
        actor="strategy:ema",
        venue="binance",
        symbol="BTCUSDT",
        strategy="ema",
        logical_id="sig-1",
        side="buy",
        order_type="limit",
        qty="1",
        price="100",
        tif="GTC",
    )
    c2 = make_submit_order_command(
        rid=rid,
        actor="strategy:ema",
        venue="binance",
        symbol="BTCUSDT",
        strategy="ema",
        logical_id="sig-1",
        side="buy",
        order_type="limit",
        qty=Decimal("1"),
        price=Decimal("100"),
        tif="GTC",
    )

    assert c1.client_order_id == c2.client_order_id
    assert c1.idempotency_key == c2.idempotency_key


def test_limit_requires_price() -> None:
    rid = RequestIdFactory()
    with pytest.raises(ValueError):
        make_submit_order_command(
            rid=rid,
            actor="x",
            venue="binance",
            symbol="BTCUSDT",
            strategy="ema",
            logical_id="sig-2",
            side="buy",
            order_type="limit",
            qty="1",
            price=None,
        )


def test_cancel_requires_order_id_or_client_order_id() -> None:
    with pytest.raises(ValueError):
        make_cancel_order_command(actor="x", venue="binance", symbol="BTCUSDT")

    c = make_cancel_order_command(
        actor="x",
        venue="binance",
        symbol="BTCUSDT",
        client_order_id="qsys-run-ema-BTCUSDT-aaaa",
        reason="user",
    )
    assert c.idempotency_key
