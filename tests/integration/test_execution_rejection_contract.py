from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from execution.bridge.live_execution_bridge import LiveExecutionBridge
from execution.ingress.router import FillIngressRouter
from execution.store.event_log import InMemoryEventLog
from monitoring.alerts.manager import AlertManager
from runner.live_runner import LiveRunner
from risk.kill_switch import KillSwitch


class _Ack:
    def __init__(
        self,
        *,
        ok: bool,
        status: str,
        error: str,
        command_id: str = "cmd-001",
        venue: str = "binance",
        result: dict | None = None,
    ) -> None:
        self.ok = ok
        self.status = status
        self.error = error
        self.command_id = command_id
        self.venue = venue
        self.result = result or {}


class _Bridge:
    def __init__(self, ack: _Ack) -> None:
        self.ack = ack
        self.submit_calls = 0

    def submit(self, _cmd: object) -> _Ack:
        self.submit_calls += 1
        return self.ack


class _SequencedBridge:
    def __init__(self, acks: list[_Ack]) -> None:
        self.acks = list(acks)
        self.submit_calls = 0

    def submit(self, _cmd: object) -> _Ack:
        self.submit_calls += 1
        if not self.acks:
            raise RuntimeError("no ack configured")
        return self.acks.pop(0)


class _RecordingSink:
    def __init__(self) -> None:
        self.alerts: list[object] = []

    def emit(self, alert: object) -> None:
        self.alerts.append(alert)


def _order_event(
    *,
    order_id: str = "ord-001",
    symbol: str = "BTCUSDT",
    side: str = "buy",
    qty: str = "1.0",
    price: str = "40000",
    venue: str = "binance",
    command_id: str = "cmd-001",
) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="order",
        order_id=order_id,
        symbol=symbol,
        side=side,
        qty=Decimal(qty),
        price=Decimal(price),
        venue=venue,
        command_id=command_id,
    )


def test_rejected_order_emits_rejection_event_but_no_fill_or_state_change() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="bridge:test")
    fill_events: list[object] = []
    reject_events: list[object] = []

    live = LiveExecutionBridge(
        execution_bridge=_Bridge(_Ack(ok=False, status="REJECTED", error="insufficient_balance")),
        dispatcher_emit=lambda event: (fill_events.append(event), router.ingest_canonical_fill(event)),
        on_reject_event=reject_events.append,
    )

    results = list(live.send_order(_order_event()))

    assert results == []
    assert fill_events == []
    assert len(reject_events) == 1

    rejection = reject_events[0]
    assert rejection.event_type == "EXECUTION_REJECT"
    assert rejection.header.event_type == "EXECUTION_REJECT"
    assert rejection.status == "REJECTED"
    assert rejection.venue == "binance"
    assert rejection.symbol == "BTCUSDT"
    assert rejection.reason == "insufficient_balance"
    assert rejection.retryable is False

    state = coord.get_state_view()
    assert float(state["positions"]["BTCUSDT"].qty) == 0.0
    assert state["event_index"] == 0


def test_failed_order_emits_retryable_rejection_event_without_fill() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="ETHUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="bridge:test")
    fill_events: list[object] = []
    reject_events: list[object] = []

    live = LiveExecutionBridge(
        execution_bridge=_Bridge(_Ack(ok=False, status="FAILED", error="retryable:TimeoutError:timeout")),
        dispatcher_emit=lambda event: (fill_events.append(event), router.ingest_canonical_fill(event)),
        on_reject_event=reject_events.append,
    )

    results = list(live.send_order(_order_event(symbol="ETHUSDT", command_id="cmd-eth")))

    assert results == []
    assert fill_events == []
    assert len(reject_events) == 1
    assert reject_events[0].status == "FAILED"
    assert reject_events[0].retryable is True

    state = coord.get_state_view()
    assert float(state["positions"]["ETHUSDT"].qty) == 0.0
    assert state["event_index"] == 0


def test_failed_then_successful_retry_flows_into_ingress_and_reconcile_cleanly() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="bridge:test")
    fill_events: list[object] = []
    reject_events: list[object] = []

    live = LiveExecutionBridge(
        execution_bridge=_SequencedBridge([
            _Ack(ok=False, status="FAILED", error="retryable:TimeoutError:timeout", command_id="cmd-retry"),
            _Ack(
                ok=True,
                status="ACCEPTED",
                error="",
                command_id="cmd-retry",
                result={"price": "40000", "qty": "1.0", "fee": "0.0"},
            ),
        ]),
        dispatcher_emit=lambda event: (fill_events.append(event), router.ingest_canonical_fill(event)),
        on_reject_event=reject_events.append,
    )

    order = _order_event(order_id="ord-retry", command_id="cmd-retry")

    first = list(live.send_order(order))
    second = list(live.send_order(order))

    assert first == []
    assert len(second) == 1
    assert len(reject_events) == 1
    assert reject_events[0].status == "FAILED"
    assert reject_events[0].retryable is True
    assert len(fill_events) == 1
    assert fill_events[0].fill_id == second[0].fill_id

    state = coord.get_state_view()
    assert float(state["positions"]["BTCUSDT"].qty) == 1.0
    assert state["event_index"] == 1


def test_rejection_enters_alert_observation_chain_without_entering_ingress() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="bridge:test")
    fill_events: list[object] = []
    reject_events: list[object] = []
    sink = _RecordingSink()

    live = LiveExecutionBridge(
        execution_bridge=_Bridge(_Ack(ok=False, status="REJECTED", error="insufficient_balance")),
        dispatcher_emit=lambda event: (fill_events.append(event), router.ingest_canonical_fill(event)),
        on_reject_event=reject_events.append,
        alert_manager=AlertManager(sink=sink),
    )

    results = list(live.send_order(_order_event()))

    assert results == []
    assert fill_events == []
    assert len(reject_events) == 1
    assert len(sink.alerts) == 1
    alert = sink.alerts[0]
    assert alert.title == "execution-rejected"
    assert alert.meta["status"] == "REJECTED"
    assert alert.meta["reason_family"] == "balance"
    assert alert.meta["routing_key"] == "binance:BTCUSDT:rejected:balance"
    assert alert.meta["retryable"] is False

    state = coord.get_state_view()
    assert float(state["positions"]["BTCUSDT"].qty) == 0.0
    assert state["event_index"] == 0


def test_rejection_enters_runner_persistent_ops_timeline_without_entering_ingress() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="bridge:test")
    sink = _RecordingSink()
    runner = LiveRunner(
        loop=SimpleNamespace(),
        coordinator=SimpleNamespace(),
        runtime=SimpleNamespace(),
        kill_switch=KillSwitch(),
        alert_manager=AlertManager(sink=sink),
        event_log=InMemoryEventLog(),
    )

    live = LiveExecutionBridge(
        execution_bridge=_Bridge(_Ack(ok=False, status="REJECTED", error="insufficient_balance")),
        dispatcher_emit=lambda event: router.ingest_canonical_fill(event),
        incident_logger=runner._emit_execution_incident,
    )

    results = list(live.send_order(_order_event()))
    timeline = runner.ops_timeline()

    assert results == []
    assert any(row["kind"] == "execution_incident" and row["category"] == "execution_rejection" for row in timeline)
    assert any(row["title"] == "execution-rejected" for row in timeline)

    state = coord.get_state_view()
    assert float(state["positions"]["BTCUSDT"].qty) == 0.0
    assert state["event_index"] == 0
