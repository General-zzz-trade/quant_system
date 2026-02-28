"""End-to-end: market_event -> pipeline -> snapshot -> decision -> order -> execution -> fill -> state.

Tests the full causal chain with mock transport and mock venue.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Set

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge


class MockDecisionModule:
    """Issues a buy order when close > 40100, only on MARKET snapshots (not fill re-triggers)."""

    def __init__(self) -> None:
        self.snapshots: List[Any] = []
        self._ordered_bars: Set[int] = set()

    def decide(self, snapshot: Any) -> list:
        self.snapshots.append(snapshot)
        # Only trade on MARKET events, not FILL re-triggers
        if snapshot.event_type.upper() != "MARKET":
            return []
        # Deduplicate: one order per bar
        if snapshot.bar_index in self._ordered_bars:
            return []
        close = snapshot.markets[snapshot.symbol].close
        if close > 40100:
            self._ordered_bars.add(snapshot.bar_index)
            return [SimpleNamespace(
                event_type="ORDER",
                EVENT_TYPE="order",
                symbol=snapshot.symbol,
                venue="mock",
                command_id=f"cmd_{snapshot.bar_index}",
                idempotency_key=f"idem_{snapshot.bar_index}",
                side="buy",
                qty=0.01,
                price=close,
                order_type="limit",
            )]
        return []


class MockVenueAdapter:
    """Mock venue adapter that returns fill events. Implements ExecutionAdapter protocol."""

    def __init__(self) -> None:
        self.orders: List[Any] = []
        self._fill_seq = 0

    def send_order(self, order_event: Any) -> list:
        self.orders.append(order_event)
        self._fill_seq += 1
        return [SimpleNamespace(
            event_type="FILL",
            EVENT_TYPE="fill",
            symbol=getattr(order_event, "symbol", "BTCUSDT"),
            side=getattr(order_event, "side", "buy"),
            qty=getattr(order_event, "qty", 0.01),
            price=getattr(order_event, "price", 40000.0),
            fee=0.0,
            realized_pnl=0.0,
            margin_change=0.0,
            header=SimpleNamespace(
                event_id=f"fill_{self._fill_seq}",
                ts=datetime(2024, 1, 1, 1, 0),
            ),
        )]


def _market(close: float, idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="MARKET",
        symbol="BTCUSDT",
        open=close, high=close + 5, low=close - 5, close=close,
        volume=100.0,
        ts=datetime(2024, 1, 1, 0, idx),
        header=SimpleNamespace(event_id=f"e{idx}", ts=datetime(2024, 1, 1, 0, idx)),
    )


def test_full_trading_flow():
    """Market event triggers decision, which sends order to venue, producing fill that updates position."""
    mock_venue = MockVenueAdapter()
    mock_decision = MockDecisionModule()

    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT",),
        currency="USDT",
        starting_balance=10000.0,
    )
    coord = EngineCoordinator(cfg=cfg)

    def _emit(ev: Any) -> None:
        coord.emit(ev, actor="bridge")

    decision_bridge = DecisionBridge(dispatcher_emit=_emit, modules=[mock_decision])
    exec_bridge = ExecutionBridge(adapter=mock_venue, dispatcher_emit=_emit)

    coord.attach_decision_bridge(decision_bridge)
    coord.attach_execution_bridge(exec_bridge)
    coord.start()

    # Emit market events: first two below threshold, then above
    coord.emit(_market(40000.0, 0), actor="test")
    coord.emit(_market(40050.0, 1), actor="test")
    coord.emit(_market(40150.0, 2), actor="test")  # triggers buy
    coord.emit(_market(40200.0, 3), actor="test")  # triggers buy

    # Decision module received snapshots (market + fill re-triggers)
    assert len(mock_decision.snapshots) >= 4

    # Venue received orders for bars where close > 40100
    assert len(mock_venue.orders) >= 1
    assert getattr(mock_venue.orders[0], "side", None) == "buy"
    assert getattr(mock_venue.orders[0], "symbol", None) == "BTCUSDT"

    # State advanced past market + fill events
    view = coord.get_state_view()
    assert view["event_index"] >= 4

    # Position exists (fills were processed)
    pos = view["positions"].get("BTCUSDT")
    assert pos is not None
    assert float(pos.qty) != 0


def test_non_triggering_market_no_orders():
    """Market events below threshold don't produce orders."""
    mock_venue = MockVenueAdapter()
    mock_decision = MockDecisionModule()

    cfg = CoordinatorConfig(
        symbol_default="BTCUSDT",
        symbols=("BTCUSDT",),
        currency="USDT",
    )
    coord = EngineCoordinator(cfg=cfg)

    def _emit(ev: Any) -> None:
        coord.emit(ev, actor="bridge")

    decision_bridge = DecisionBridge(dispatcher_emit=_emit, modules=[mock_decision])
    exec_bridge = ExecutionBridge(adapter=mock_venue, dispatcher_emit=_emit)
    coord.attach_decision_bridge(decision_bridge)
    coord.attach_execution_bridge(exec_bridge)
    coord.start()

    for i in range(5):
        coord.emit(_market(39000.0 + i * 10, i), actor="test")

    assert len(mock_venue.orders) == 0
