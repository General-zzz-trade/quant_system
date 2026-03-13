from __future__ import annotations

import time
from decimal import Decimal
from types import SimpleNamespace

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from execution.algo_adapter import _make_fill_event
from execution.algo_adapter import AlgoConfig, AlgoExecutionAdapter
from execution.bridge.live_execution_bridge import LiveExecutionBridge
from execution.ingress.router import FillIngressRouter
from execution.store.event_log import InMemoryEventLog
from execution.reconcile.controller import ReconcileController
from monitoring.alerts.manager import AlertManager
from runner.live_runner import LiveRunner
from risk.kill_switch import KillSwitch


class _Ack:
    def __init__(self, *, ok: bool = True, result: dict | None = None, status: str = "ACCEPTED") -> None:
        self.ok = ok
        self.status = status
        self.result = result or {}
        self.venue = "binance"
        self.command_id = "cmd-001"
        self.error = None


class _Bridge:
    def __init__(self, ack: _Ack) -> None:
        self.ack = ack
        self.submit_calls = 0

    def submit(self, _cmd: object) -> _Ack:
        self.submit_calls += 1
        return self.ack


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
    intent_id: str = "int-001",
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
        intent_id=intent_id,
    )


def test_direct_bridge_synthetic_fill_is_idempotent_through_ingress_and_reconcile() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="bridge:test")
    bridge = _Bridge(_Ack(ok=True, result={"price": "40000", "qty": "1.0", "fee": "0.1"}))
    live = LiveExecutionBridge(execution_bridge=bridge, dispatcher_emit=lambda event: router.ingest_canonical_fill(event))

    order = _order_event()

    first = list(live.send_order(order))
    second = list(live.send_order(order))

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].fill_id == second[0].fill_id
    assert first[0].payload_digest == second[0].payload_digest
    assert bridge.submit_calls == 2

    state = coord.get_state_view()
    assert float(state["positions"]["BTCUSDT"].qty) == 1.0

    report = ReconcileController().reconcile(
        venue="binance",
        local_positions={"BTCUSDT": Decimal("1.0")},
        venue_positions={"BTCUSDT": Decimal("1.0")},
        local_fill_ids={first[0].fill_id},
        venue_fill_ids={first[0].fill_id},
        fill_symbol="BTCUSDT",
    )
    assert report.ok


def test_algo_synthetic_fills_keep_stable_distinct_ids_for_reconcile() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="algo:test")
    order = _order_event(order_id="ord-algo", qty="3.0", intent_id="intent-42")

    fill1 = _make_fill_event(order, Decimal("40000"), Decimal("1.0"), fill_seq=1)
    fill1_dup = _make_fill_event(order, Decimal("40000"), Decimal("1.0"), fill_seq=1)
    fill2 = _make_fill_event(order, Decimal("40100"), Decimal("2.0"), fill_seq=2)

    assert fill1.fill_id == fill1_dup.fill_id
    assert fill1.payload_digest == fill1_dup.payload_digest
    assert fill2.fill_id != fill1.fill_id

    assert router.ingest_canonical_fill(fill1) is True
    assert router.ingest_canonical_fill(fill1_dup) is False
    assert router.ingest_canonical_fill(fill2) is True

    state = coord.get_state_view()
    assert float(state["positions"]["BTCUSDT"].qty) == 3.0

    report = ReconcileController().reconcile(
        venue="binance",
        local_positions={"BTCUSDT": Decimal("3.0")},
        venue_positions={"BTCUSDT": Decimal("3.0")},
        local_fill_ids={fill1.fill_id, fill2.fill_id},
        venue_fill_ids={fill1.fill_id, fill2.fill_id},
        fill_symbol="BTCUSDT",
    )
    assert report.ok


def test_bridge_synthetic_fill_enters_runner_persistent_ops_timeline() -> None:
    sink = _RecordingSink()
    runner = LiveRunner(
        loop=SimpleNamespace(),
        coordinator=SimpleNamespace(),
        runtime=SimpleNamespace(),
        kill_switch=KillSwitch(),
        alert_manager=AlertManager(sink=sink),
        event_log=InMemoryEventLog(),
    )
    bridge = _Bridge(_Ack(ok=True, result={"price": "40000", "qty": "1.0", "fee": "0.1"}))
    live = LiveExecutionBridge(
        execution_bridge=bridge,
        incident_logger=runner._emit_execution_incident,
    )

    fills = list(live.send_order(_order_event()))
    timeline = runner.ops_timeline()

    assert len(fills) == 1
    assert any(row["kind"] == "execution_incident" and row["category"] == "execution_fill" for row in timeline)
    assert any(row["title"] == "execution-synthetic-fill" for row in timeline)


def test_algo_synthetic_fill_enters_runner_persistent_ops_timeline() -> None:
    sink = _RecordingSink()
    runner = LiveRunner(
        loop=SimpleNamespace(),
        coordinator=SimpleNamespace(),
        runtime=SimpleNamespace(),
        kill_switch=KillSwitch(),
        alert_manager=AlertManager(sink=sink),
        event_log=InMemoryEventLog(),
    )
    emitted: list[object] = []
    adapter = AlgoExecutionAdapter(
        submit_fn=lambda _symbol, _side, _qty: Decimal("40000"),
        dispatcher_emit=emitted.append,
        incident_logger=runner._emit_execution_incident,
        cfg=AlgoConfig(
            large_order_notional=Decimal("100"),
            default_algo="twap",
            twap_slices=2,
            twap_duration_sec=0.01,
            tick_interval_sec=0.05,
        ),
    )

    adapter.send_order(_order_event(order_id="ord-algo-tl", qty="2.0", price="1000", intent_id="intent-tl"))
    deadline = time.monotonic() + 3.0
    while len(emitted) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    adapter.stop()

    timeline = runner.ops_timeline()

    assert len(emitted) == 2
    fill_rows = [row for row in timeline if row["kind"] == "execution_incident" and row["category"] == "execution_fill"]
    assert len(fill_rows) >= 2
    assert all(row["title"] == "execution-synthetic-fill" for row in fill_rows[:2])
