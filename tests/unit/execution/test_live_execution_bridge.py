# tests/unit/execution/test_live_execution_bridge.py
"""Tests for LiveExecutionBridge — routing, fill conversion, edge cases."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, List

import pytest

from execution.bridge.live_execution_bridge import LiveExecutionBridge, LiveExecutionConfig


# ── Stubs ────────────────────────────────────────────────────


def _order_event(
    *,
    symbol: str = "BTCUSDT",
    side: str = "buy",
    qty: str = "1.0",
    price: str | None = "40000",
    order_id: str = "ord-001",
) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="order",
        order_id=order_id,
        symbol=symbol,
        side=side,
        qty=Decimal(qty),
        price=Decimal(price) if price else None,
        venue="binance",
        command_id="cmd-001",
        idempotency_key="idem-001",
    )


class _FakeAck:
    def __init__(self, *, ok: bool = True, result: dict | None = None, status: str = "ACCEPTED",
                 venue: str = "binance", command_id: str = "cmd-001", error: str | None = None):
        self.ok = ok
        self.status = status
        self.result = result or {}
        self.venue = venue
        self.command_id = command_id
        self.error = error


class _FakeBridge:
    def __init__(self, ack: _FakeAck | None = None):
        self.ack = ack or _FakeAck()
        self.submitted: List[Any] = []

    def submit(self, cmd: Any) -> _FakeAck:
        self.submitted.append(cmd)
        return self.ack


class _FakeAlgoAdapter:
    def __init__(self, fills: list | None = None):
        self.fills = fills or []
        self.received: List[Any] = []

    def send_order(self, order_event: Any) -> list:
        self.received.append(order_event)
        return self.fills


# ── Tests: Routing ───────────────────────────────────────────


class TestOrderRouting:
    def test_small_order_goes_to_direct_execution(self):
        bridge = _FakeBridge(_FakeAck(ok=True, result={"price": "40000", "qty": "1.0"}))
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            config=LiveExecutionConfig(large_order_notional=Decimal("50000")),
        )

        order = _order_event(qty="1.0", price="40000")  # notional=40000 < 50000
        results = list(live.send_order(order))

        assert len(results) == 1
        assert results[0].event_type == "fill"
        assert len(bridge.submitted) == 1

    def test_large_order_routes_to_algo_adapter(self):
        algo = _FakeAlgoAdapter(fills=[SimpleNamespace(event_type="fill", symbol="BTCUSDT")])
        bridge = _FakeBridge()
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            algo_adapter=algo,
            config=LiveExecutionConfig(large_order_notional=Decimal("10000")),
        )

        order = _order_event(qty="1.0", price="40000")  # notional=40000 >= 10000
        results = list(live.send_order(order))

        assert len(results) == 1
        assert len(algo.received) == 1
        assert len(bridge.submitted) == 0  # not direct

    def test_large_order_without_algo_falls_to_direct(self):
        bridge = _FakeBridge(_FakeAck(ok=True, result={"price": "40000", "qty": "1.0"}))
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            algo_adapter=None,  # no algo
            config=LiveExecutionConfig(large_order_notional=Decimal("10000")),
        )

        order = _order_event(qty="1.0", price="40000")  # notional=40000 >= 10000 but no algo
        results = list(live.send_order(order))

        assert len(results) == 1
        assert len(bridge.submitted) == 1

    def test_no_price_uses_qty_as_notional(self):
        bridge = _FakeBridge(_FakeAck(ok=True, result={}))
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            config=LiveExecutionConfig(large_order_notional=Decimal("5")),
        )

        order = _order_event(qty="1.0", price=None)  # notional=1.0 < 5 -> direct
        results = list(live.send_order(order))

        assert len(results) == 1
        assert len(bridge.submitted) == 1


# ── Tests: Ack to fill conversion ────────────────────────────


class TestAckToFill:
    def test_fill_from_ack_with_result_fields(self):
        ack = _FakeAck(ok=True, result={"price": "41000", "qty": "0.5", "fee": "1.23"})
        order = _order_event(symbol="ETHUSDT", side="sell", order_id="o-99")

        fill = LiveExecutionBridge._ack_to_fill(order, ack)

        assert fill.event_type == "fill"
        assert fill.symbol == "ETHUSDT"
        assert fill.side == "sell"
        assert fill.order_id == "o-99"
        assert fill.price == Decimal("41000")
        assert fill.qty == Decimal("0.5")
        assert fill.fee == Decimal("1.23")
        assert fill.fill_id.startswith("bridge-fill-")

    def test_fill_falls_back_to_order_event_fields(self):
        ack = _FakeAck(ok=True, result={})
        order = _order_event(qty="2.0", price="3000")

        fill = LiveExecutionBridge._ack_to_fill(order, ack)

        assert fill.price == Decimal("3000")
        assert fill.qty == Decimal("2.0")
        assert fill.fee == Decimal("0")


# ── Tests: Rejected orders ───────────────────────────────────


class TestRejected:
    def test_rejected_ack_returns_empty(self):
        bridge = _FakeBridge(_FakeAck(ok=False, status="REJECTED", error="insufficient_balance"))
        live = LiveExecutionBridge(execution_bridge=bridge)

        results = list(live.send_order(_order_event()))

        assert len(results) == 0

    def test_dispatcher_emit_called_on_fill(self):
        emitted: List[Any] = []
        bridge = _FakeBridge(_FakeAck(ok=True, result={"price": "40000", "qty": "1.0"}))
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            dispatcher_emit=emitted.append,
        )

        list(live.send_order(_order_event()))

        assert len(emitted) == 1
        assert emitted[0].event_type == "fill"

    def test_order_count_increments(self):
        bridge = _FakeBridge(_FakeAck(ok=True, result={}))
        live = LiveExecutionBridge(execution_bridge=bridge)

        assert live.order_count == 0
        list(live.send_order(_order_event()))
        assert live.order_count == 1
        list(live.send_order(_order_event()))
        assert live.order_count == 2
