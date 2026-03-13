"""Replay vs live path equivalence — state, signals, and orders."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.replay import EventReplay, ReplayConfig
from event.header import EventHeader
from event.types import EventType, MarketEvent, OrderEvent
from execution.sim.replay_adapter import ReplayExecutionAdapter
from runner.replay_runner import run_replay_from_events


# ============================================================
# Shared fixtures
# ============================================================

def _events() -> list[MarketEvent]:
    return [
        MarketEvent(
            header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="test"),
            ts=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("10"),
        ),
        MarketEvent(
            header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="test"),
            ts=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("99"),
            close=Decimal("101"),
            volume=Decimal("12"),
        ),
    ]


class SimpleBuyModule:
    """Buys on first bar above 100.5. Deterministic."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._bought = False

    def decide(self, snapshot: Any) -> Iterable[Any]:
        if isinstance(snapshot, dict):
            markets = snapshot.get("markets", {})
        else:
            markets = getattr(snapshot, "markets", None) or {}
        mkt = markets.get(self.symbol) if hasattr(markets, "get") else None
        if mkt is None:
            return []
        close = Decimal(str(getattr(mkt, "close", 0)))
        if close > Decimal("100.5") and not self._bought:
            self._bought = True
            return [OrderEvent(
                header=EventHeader.new_root(
                    event_type=EventType.ORDER, version=1, source="test_dm",
                ),
                order_id=str(uuid.uuid4()),
                intent_id="test_buy",
                symbol=self.symbol,
                side="BUY",
                qty=Decimal("0.01"),
                price=close,
            )]
        return []


# ============================================================
# Helpers
# ============================================================

def _run_live(events: list[MarketEvent], decision_modules=None) -> dict:
    coordinator = EngineCoordinator(
        cfg=CoordinatorConfig(symbol_default="BTCUSDT", currency="USDT")
    )

    captured_orders = []
    order_log = []

    if decision_modules is not None:
        def _price_source(sym):
            try:
                view = coordinator.get_state_view()
                mkt = view.get("markets", {}).get(sym)
                return Decimal(str(getattr(mkt, "close", 0))) if mkt else None
            except Exception:
                return None

        adapter = ReplayExecutionAdapter(price_source=_price_source)

        def _emit(ev, *, actor="live"):
            et = getattr(ev, "event_type", None)
            if getattr(et, "value", "") == "order":
                captured_orders.append(ev)
            coordinator.emit(ev, actor=actor)

        db = DecisionBridge(dispatcher_emit=_emit, modules=list(decision_modules))
        eb = ExecutionBridge(adapter=adapter, dispatcher_emit=_emit)
        coordinator.attach_decision_bridge(db)
        coordinator.attach_execution_bridge(eb)
    else:
        adapter = None

    coordinator.start()
    for event in events:
        coordinator.emit(event, actor="live")
    view = coordinator.get_state_view()
    coordinator.stop()

    return {
        "view": view,
        "order_log": adapter.order_log if adapter else [],
        "captured_orders": captured_orders,
    }


def _run_replay(events: list[MarketEvent], decision_modules=None) -> dict:
    if decision_modules is not None:
        result = run_replay_from_events(
            events=events,
            symbol="BTCUSDT",
            decision_modules=decision_modules,
        )
        return {
            "view": result.final_state,
            "order_log": result.order_log,
            "captured_orders": result.captured_orders,
        }

    coordinator = EngineCoordinator(
        cfg=CoordinatorConfig(symbol_default="BTCUSDT", currency="USDT")
    )
    coordinator.start()
    replay = EventReplay(
        dispatcher=coordinator.dispatcher,
        source=events,
        config=ReplayConfig(strict_order=False, actor="replay"),
    )
    replay.run()
    view = coordinator.get_state_view()
    coordinator.stop()
    return {"view": view, "order_log": [], "captured_orders": []}


# ============================================================
# Tests
# ============================================================

def test_replay_vs_live_state_equivalence() -> None:
    """State-only: replay and live produce same event_index, close, last_price."""
    events = _events()
    live = _run_live(events)
    replay = _run_replay(events)

    assert live["view"]["event_index"] == replay["view"]["event_index"] == 2
    assert live["view"]["market"].close == replay["view"]["market"].close == Decimal("101")
    assert live["view"]["market"].last_price == replay["view"]["market"].last_price == Decimal("101")


def test_replay_vs_live_order_equivalence() -> None:
    """Full-chain: same decision module produces same orders in live vs replay."""
    events = _events()

    live = _run_live(events, decision_modules=[SimpleBuyModule("BTCUSDT")])
    replay = _run_replay(events, decision_modules=[SimpleBuyModule("BTCUSDT")])

    # Same state
    assert live["view"]["event_index"] == replay["view"]["event_index"]
    assert live["view"]["market"].close == replay["view"]["market"].close

    # Same orders
    assert len(live["order_log"]) == len(replay["order_log"]), (
        f"Live: {len(live['order_log'])} orders, Replay: {len(replay['order_log'])}"
    )

    # bar 1 close=101 > 100.5 → 1 BUY order
    assert len(live["order_log"]) == 1
    assert live["order_log"][0]["side"] == "BUY"
    assert replay["order_log"][0]["side"] == "BUY"
    assert live["order_log"][0]["qty"] == replay["order_log"][0]["qty"]


def test_replay_vs_live_position_equivalence() -> None:
    """Full-chain: fills update position identically in live vs replay."""
    events = _events()

    live = _run_live(events, decision_modules=[SimpleBuyModule("BTCUSDT")])
    replay = _run_replay(events, decision_modules=[SimpleBuyModule("BTCUSDT")])

    live_pos = live["view"]["positions"].get("BTCUSDT")
    replay_pos = replay["view"]["positions"].get("BTCUSDT")

    # Both should have a position after the fill
    assert live_pos is not None, "Live path should have BTCUSDT position"
    assert replay_pos is not None, "Replay path should have BTCUSDT position"

    live_qty = getattr(live_pos, "qty", None) or getattr(live_pos, "quantity", None)
    replay_qty = getattr(replay_pos, "qty", None) or getattr(replay_pos, "quantity", None)

    if live_qty is not None and replay_qty is not None:
        assert float(live_qty) == float(replay_qty), (
            f"Position mismatch: live={live_qty}, replay={replay_qty}"
        )
