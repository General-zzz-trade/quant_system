# tests/unit/execution/test_live_execution_bridge.py
"""Tests for LiveExecutionBridge — routing, fill conversion, edge cases."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, List


from execution.bridge.live_execution_bridge import LiveExecutionBridge, LiveExecutionConfig
from monitoring.alerts.manager import AlertManager


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


class _RecordingSink:
    def __init__(self) -> None:
        self.alerts: List[Any] = []

    def emit(self, alert: Any) -> None:
        self.alerts.append(alert)


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
        assert results[0].event_type == "FILL"
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

        assert fill.event_type == "FILL"
        assert fill.symbol == "ETHUSDT"
        assert fill.side == "sell"
        assert fill.order_id == "o-99"
        assert fill.price == 41000.0
        assert fill.qty == 0.5
        assert fill.quantity == 0.5
        assert fill.fee == 1.23
        assert fill.fill_id.startswith("bridge-fill-")
        assert fill.payload_digest
        assert len(fill.payload_digest) == 16
        assert fill.venue == "binance"
        assert fill.header.event_type == "FILL"
        assert fill.header.event_id is None

    def test_fill_falls_back_to_order_event_fields(self):
        ack = _FakeAck(ok=True, result={})
        order = _order_event(qty="2.0", price="3000")

        fill = LiveExecutionBridge._ack_to_fill(order, ack)

        assert fill.price == 3000.0
        assert fill.qty == 2.0
        assert fill.quantity == 2.0
        assert fill.fee == 0.0

    def test_fill_identity_is_stable_for_same_ack_and_order(self):
        ack = _FakeAck(ok=True, result={"price": "41000", "qty": "0.5", "fee": "1.23"})
        order = _order_event(symbol="ETHUSDT", side="sell", order_id="o-99")

        fill1 = LiveExecutionBridge._ack_to_fill(order, ack)
        fill2 = LiveExecutionBridge._ack_to_fill(order, ack)

        assert fill1.fill_id == fill2.fill_id
        assert fill1.payload_digest == fill2.payload_digest


# ── Tests: Rejected orders ───────────────────────────────────


class TestRejected:
    def test_rejected_ack_returns_empty(self):
        bridge = _FakeBridge(_FakeAck(ok=False, status="REJECTED", error="insufficient_balance"))
        live = LiveExecutionBridge(execution_bridge=bridge)

        results = list(live.send_order(_order_event()))

        assert len(results) == 0

    def test_on_reject_receives_canonical_rejection(self):
        rejected: List[Any] = []
        bridge = _FakeBridge(_FakeAck(ok=False, status="REJECTED", error="insufficient_balance"))
        live = LiveExecutionBridge(execution_bridge=bridge, on_reject=rejected.append)

        results = list(live.send_order(_order_event(symbol="ETHUSDT")))

        assert len(results) == 0
        assert len(rejected) == 1
        rej = rejected[0]
        assert rej.status == "REJECTED"
        assert rej.venue == "binance"
        assert rej.symbol == "ETHUSDT"
        assert rej.reason == "insufficient_balance"
        assert rej.retryable is False

    def test_on_reject_event_receives_event_like_rejection(self):
        rejected_events: List[Any] = []
        bridge = _FakeBridge(_FakeAck(ok=False, status="FAILED", error="retryable:TimeoutError:timeout"))
        live = LiveExecutionBridge(execution_bridge=bridge, on_reject_event=rejected_events.append)

        results = list(live.send_order(_order_event(symbol="BTCUSDT")))

        assert len(results) == 0
        assert len(rejected_events) == 1
        rej = rejected_events[0]
        assert rej.event_type == "EXECUTION_REJECT"
        assert rej.header.event_type == "EXECUTION_REJECT"
        assert rej.status == "FAILED"
        assert rej.venue == "binance"
        assert rej.symbol == "BTCUSDT"
        assert rej.retryable is True

    def test_rejected_ack_emits_structured_alert(self):
        sink = _RecordingSink()
        bridge = _FakeBridge(_FakeAck(ok=False, status="REJECTED", error="insufficient_balance"))
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            alert_manager=AlertManager(sink=sink),
        )

        results = list(live.send_order(_order_event(symbol="ETHUSDT")))

        assert len(results) == 0
        assert len(sink.alerts) == 1
        alert = sink.alerts[0]
        assert alert.title == "execution-rejected"
        assert alert.meta["symbol"] == "ETHUSDT"
        assert alert.meta["status"] == "REJECTED"
        assert alert.meta["reason_family"] == "balance"
        assert alert.meta["routing_key"] == "binance:ETHUSDT:rejected:balance"
        assert alert.meta["retryable"] is False

    def test_rejected_ack_emits_to_incident_logger(self):
        incidents: List[Any] = []
        bridge = _FakeBridge(_FakeAck(ok=False, status="REJECTED", error="insufficient_balance"))
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            incident_logger=incidents.append,
        )

        results = list(live.send_order(_order_event(symbol="ETHUSDT")))

        assert len(results) == 0
        assert len(incidents) == 1
        assert incidents[0].title == "execution-rejected"
        assert incidents[0].meta["category"] == "execution_rejection"
        assert incidents[0].meta["reason_family"] == "balance"

    def test_success_ack_emits_synthetic_fill_alert(self):
        sink = _RecordingSink()
        bridge = _FakeBridge(_FakeAck(ok=True, result={"price": "40000", "qty": "1.0"}))
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            alert_manager=AlertManager(sink=sink),
        )

        results = list(live.send_order(_order_event(symbol="ETHUSDT")))

        assert len(results) == 1
        assert len(sink.alerts) == 1
        alert = sink.alerts[0]
        assert alert.title == "execution-synthetic-fill"
        assert alert.meta["category"] == "execution_fill"
        assert alert.meta["synthetic"] is True

    def test_success_ack_emits_synthetic_fill_to_incident_logger(self):
        incidents: List[Any] = []
        bridge = _FakeBridge(_FakeAck(ok=True, result={"price": "40000", "qty": "1.0"}))
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            incident_logger=incidents.append,
        )

        results = list(live.send_order(_order_event(symbol="ETHUSDT")))

        assert len(results) == 1
        assert len(incidents) == 1
        assert incidents[0].title == "execution-synthetic-fill"
        assert incidents[0].meta["category"] == "execution_fill"

    def test_dispatcher_emit_called_on_fill(self):
        emitted: List[Any] = []
        bridge = _FakeBridge(_FakeAck(ok=True, result={"price": "40000", "qty": "1.0"}))
        live = LiveExecutionBridge(
            execution_bridge=bridge,
            dispatcher_emit=emitted.append,
        )

        list(live.send_order(_order_event()))

        assert len(emitted) == 1
        assert emitted[0].event_type == "FILL"

    def test_order_count_increments(self):
        bridge = _FakeBridge(_FakeAck(ok=True, result={}))
        live = LiveExecutionBridge(execution_bridge=bridge)

        assert live.order_count == 0
        list(live.send_order(_order_event()))
        assert live.order_count == 1
        list(live.send_order(_order_event()))
        assert live.order_count == 2


# ── Tests: ExecutionBridge retry + circuit breaker ───────────


class TestExecutionBridgeRetry:
    def test_transient_failure_retried(self):
        from engine.execution_bridge import ExecutionBridge
        call_count = 0

        class FlakeyAdapter:
            def send_order(self, event):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("transient")
                return []

        bridge = ExecutionBridge(
            adapter=FlakeyAdapter(),
            dispatcher_emit=lambda e: None,
            max_retries=3,
            retry_base_delay=0.01,
        )
        bridge.handle_event("fake_order")
        assert call_count == 3

    def test_permanent_failure_opens_circuit(self):
        from engine.execution_bridge import ExecutionBridge

        class AlwaysFailAdapter:
            def send_order(self, event):
                raise ConnectionError("down")

        bridge = ExecutionBridge(
            adapter=AlwaysFailAdapter(),
            dispatcher_emit=lambda e: None,
            max_retries=2,
            retry_base_delay=0.01,
            cb_failure_threshold=3,
            cb_cooldown_seconds=10.0,
        )
        for _ in range(3):
            try:
                bridge.handle_event("order")
            except Exception:
                pass
        assert bridge.circuit_open
