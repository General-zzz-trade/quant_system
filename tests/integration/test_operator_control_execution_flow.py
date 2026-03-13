from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from execution.bridge.live_execution_bridge import LiveExecutionBridge
from execution.ingress.router import FillIngressRouter
from execution.reconcile.controller import ReconcileController
from risk.kill_switch_bridge import KillSwitchBridge
from runner.live_runner import LiveRunner
from risk.kill_switch import KillSwitch


class _Ack:
    def __init__(
        self,
        *,
        ok: bool = True,
        status: str = "ACCEPTED",
        command_id: str = "cmd-001",
        result: dict | None = None,
    ) -> None:
        self.ok = ok
        self.status = status
        self.command_id = command_id
        self.venue = "binance"
        self.error = None
        self.result = result or {"price": "40000", "qty": "1.0", "fee": "0.1"}


class _Bridge:
    def __init__(self, ack: _Ack | None = None) -> None:
        self.ack = ack or _Ack()
        self.submit_calls = 0

    def submit(self, _cmd: object) -> _Ack:
        self.submit_calls += 1
        return self.ack


def _order_event(
    *,
    order_id: str = "ord-001",
    symbol: str = "BTCUSDT",
    side: str = "buy",
    qty: str = "1.0",
    price: str = "40000",
    venue: str = "binance",
    command_id: str = "cmd-001",
    reduce_only: bool = False,
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
        reduce_only=reduce_only,
    )


def _make_runner_with_kill_switch() -> LiveRunner:
    return LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
    )


def _guarded_live_bridge(runner: LiveRunner, router: FillIngressRouter, bridge: _Bridge) -> KillSwitchBridge:
    live = LiveExecutionBridge(
        execution_bridge=bridge,
        dispatcher_emit=lambda event: router.ingest_canonical_fill(event),
    )
    return KillSwitchBridge(inner=live, kill_switch=runner.kill_switch)


def test_halt_then_resume_allows_later_successful_fill_and_reconcile() -> None:
    runner = _make_runner_with_kill_switch()
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="control:test")
    guarded = _guarded_live_bridge(runner, router, _Bridge())
    order = _order_event(order_id="ord-halt", command_id="cmd-halt")

    runner.apply_control(SimpleNamespace(command="halt", reason="manual stop"))
    first = list(guarded.send_order(order))
    assert first == []

    runner.apply_control(SimpleNamespace(command="resume", reason="manual resume"))
    second = list(guarded.send_order(order))

    assert len(second) == 1
    state = coord.get_state_view()
    assert float(state["positions"]["BTCUSDT"].qty) == 1.0

    report = ReconcileController().reconcile(
        venue="binance",
        local_positions={"BTCUSDT": Decimal("1.0")},
        venue_positions={"BTCUSDT": Decimal("1.0")},
        local_fill_ids={second[0].fill_id},
        venue_fill_ids={second[0].fill_id},
        fill_symbol="BTCUSDT",
    )
    assert report.ok


def test_reduce_only_blocks_opening_order_but_allows_flagged_order_into_ingress() -> None:
    runner = _make_runner_with_kill_switch()
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="control:test")
    guarded = _guarded_live_bridge(runner, router, _Bridge())

    runner.apply_control(SimpleNamespace(command="reduce_only", reason="manual ro"))

    blocked = list(guarded.send_order(_order_event(order_id="ord-blocked", command_id="cmd-blocked", reduce_only=False)))
    allowed = list(guarded.send_order(_order_event(order_id="ord-allowed", command_id="cmd-allowed", reduce_only=True)))

    assert blocked == []
    assert len(allowed) == 1
    state = coord.get_state_view()
    assert float(state["positions"]["BTCUSDT"].qty) == 1.0

    report = ReconcileController().reconcile(
        venue="binance",
        local_positions={"BTCUSDT": Decimal("1.0")},
        venue_positions={"BTCUSDT": Decimal("1.0")},
        local_fill_ids={allowed[0].fill_id},
        venue_fill_ids={allowed[0].fill_id},
        fill_symbol="BTCUSDT",
    )
    assert report.ok
