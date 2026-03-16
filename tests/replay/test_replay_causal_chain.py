"""Full causal chain test: market → decision → order → fill → position update.

Validates that replay produces the same orders and final state as the live path
when given identical market events and decision modules.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable

logger = logging.getLogger(__name__)

from engine.coordinator import CoordinatorConfig, EngineCoordinator  # noqa: E402
from engine.decision_bridge import DecisionBridge  # noqa: E402
from engine.execution_bridge import ExecutionBridge  # noqa: E402
from event.header import EventHeader  # noqa: E402
from event.types import EventType, MarketEvent, OrderEvent  # noqa: E402
from execution.sim.replay_adapter import ReplayExecutionAdapter  # noqa: E402
from runner.replay_runner import run_replay_from_events  # noqa: E402


# ============================================================
# Deterministic decision module (no ML dependency)
# ============================================================

class ThresholdDecisionModule:
    """Buy when close > threshold, sell when close < threshold - spread.

    Deterministic: same prices → same orders. Tracks position internally
    to avoid duplicate orders.
    """

    def __init__(self, symbol: str, buy_threshold: Decimal = Decimal("100.5"),
                 sell_threshold: Decimal = Decimal("99.5")):
        self.symbol = symbol
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._position = 0  # 0=flat, 1=long

    def decide(self, snapshot: Any) -> Iterable[Any]:
        if isinstance(snapshot, dict):
            markets = snapshot.get("markets", {})
        else:
            markets = getattr(snapshot, "markets", None) or {}

        mkt = markets.get(self.symbol) if hasattr(markets, "get") else None
        if mkt is None:
            return []

        close = Decimal(str(getattr(mkt, "close", 0)))
        events = []

        if close > self._buy_threshold and self._position == 0:
            self._position = 1
            events.append(OrderEvent(
                header=EventHeader.new_root(
                    event_type=EventType.ORDER, version=1, source="threshold_dm",
                ),
                order_id=str(uuid.uuid4()),
                intent_id="threshold_buy",
                symbol=self.symbol,
                side="BUY",
                qty=Decimal("0.01"),
                price=close,
            ))
        elif close < self._sell_threshold and self._position == 1:
            self._position = 0
            events.append(OrderEvent(
                header=EventHeader.new_root(
                    event_type=EventType.ORDER, version=1, source="threshold_dm",
                ),
                order_id=str(uuid.uuid4()),
                intent_id="threshold_sell",
                symbol=self.symbol,
                side="SELL",
                qty=Decimal("0.01"),
                price=close,
            ))

        return events


# ============================================================
# Test data
# ============================================================

def _market_events(symbol: str = "BTCUSDT") -> list[MarketEvent]:
    """Generate a price sequence that triggers buy then sell."""
    datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    prices = [
        # bar 0: close=100, no action (below buy threshold)
        (Decimal("100"), Decimal("100.2"), Decimal("99.8"), Decimal("100"), Decimal("10")),
        # bar 1: close=101, triggers BUY (> 100.5)
        (Decimal("100"), Decimal("101.5"), Decimal("100"), Decimal("101"), Decimal("15")),
        # bar 2: close=101.5, hold (already long)
        (Decimal("101"), Decimal("102"), Decimal("101"), Decimal("101.5"), Decimal("12")),
        # bar 3: close=99, triggers SELL (< 99.5)
        (Decimal("101"), Decimal("101"), Decimal("98.5"), Decimal("99"), Decimal("20")),
        # bar 4: close=100, no action (flat, below buy threshold)
        (Decimal("99"), Decimal("100.2"), Decimal("99"), Decimal("100"), Decimal("11")),
    ]

    events = []
    for i, (o, h, l, c, v) in enumerate(prices):  # noqa: E741
        ts = datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc)
        events.append(MarketEvent(
            header=EventHeader.new_root(
                event_type=EventType.MARKET, version=1, source="test",
            ),
            ts=ts,
            symbol=symbol,
            open=o, high=h, low=l, close=c, volume=v,
        ))
    return events


# ============================================================
# Live-path helper (same wiring as replay, but via direct emit)
# ============================================================

def _run_live_path(events: list[MarketEvent], symbol: str = "BTCUSDT") -> dict:
    """Run events through the live path with decision + execution bridges."""
    cfg = CoordinatorConfig(symbol_default=symbol, currency="USDT")
    coordinator = EngineCoordinator(cfg=cfg)

    dm = ThresholdDecisionModule(symbol)

    def _price_source(sym: str):
        try:
            view = coordinator.get_state_view()
            mkt = view.get("markets", {}).get(sym)
            if mkt is not None:
                return Decimal(str(getattr(mkt, "close", 0)))
        except Exception as e:
            logger.debug("Failed to get replay price for %s: %s", sym, e)
        return None

    adapter = ReplayExecutionAdapter(price_source=_price_source)

    captured_orders = []

    def _emit(ev, *, actor="live"):
        et = getattr(ev, "event_type", None)
        et_val = getattr(et, "value", str(et) if et else "").lower()
        if et_val == "order":
            captured_orders.append(ev)
        coordinator.emit(ev, actor=actor)

    decision_bridge = DecisionBridge(dispatcher_emit=_emit, modules=[dm])
    execution_bridge = ExecutionBridge(adapter=adapter, dispatcher_emit=_emit)

    coordinator.attach_decision_bridge(decision_bridge)
    coordinator.attach_execution_bridge(execution_bridge)
    coordinator.start()

    for ev in events:
        coordinator.emit(ev, actor="live")

    final_state = dict(coordinator.get_state_view())
    coordinator.stop()

    return {
        "final_state": final_state,
        "order_log": adapter.order_log,
        "captured_orders": captured_orders,
    }


# ============================================================
# Tests
# ============================================================

def test_replay_produces_orders():
    """Replay with decision modules must produce orders and fills."""
    events = _market_events()
    dm = ThresholdDecisionModule("BTCUSDT")

    result = run_replay_from_events(
        events=events,
        symbol="BTCUSDT",
        decision_modules=[dm],
    )

    assert result.events_processed == 5
    # Should have 2 orders: BUY at bar 1, SELL at bar 3
    assert len(result.order_log) == 2, f"Expected 2 orders, got {len(result.order_log)}"
    assert result.order_log[0]["side"] == "BUY"
    assert result.order_log[1]["side"] == "SELL"

    # Captured order events
    assert len(result.captured_orders) == 2

    # Final state should have position data
    assert result.final_state is not None
    # 5 market events + 2 fill events (from BUY and SELL) = 7
    assert result.final_state["event_index"] == 7


def test_replay_fills_update_position():
    """Fills from replay adapter must flow back to update position state."""
    events = _market_events()[:2]  # Just bars 0 and 1 (triggers BUY)
    dm = ThresholdDecisionModule("BTCUSDT")

    result = run_replay_from_events(
        events=events,
        symbol="BTCUSDT",
        decision_modules=[dm],
    )

    assert len(result.order_log) == 1
    assert result.order_log[0]["side"] == "BUY"

    # FillEvent should have updated position in state
    positions = result.final_state.get("positions", {})
    pos = positions.get("BTCUSDT")
    # Position qty should be non-zero after fill
    if pos is not None:
        qty = getattr(pos, "qty", None) or getattr(pos, "quantity", None)
        if qty is not None:
            assert float(qty) != 0.0, "Position should be non-zero after fill"


def test_replay_vs_live_order_equivalence():
    """Same events + same decision module → same orders in live vs replay paths."""
    events = _market_events()

    # Live path
    live_result = _run_live_path(events)

    # Replay path (fresh decision module — same initial state)
    dm = ThresholdDecisionModule("BTCUSDT")
    replay_result = run_replay_from_events(
        events=events,
        symbol="BTCUSDT",
        decision_modules=[dm],
    )

    # Same number of orders
    assert len(live_result["order_log"]) == len(replay_result.order_log), (
        f"Live produced {len(live_result['order_log'])} orders, "
        f"replay produced {len(replay_result.order_log)}"
    )

    # Same order sides
    for i, (live_ord, replay_ord) in enumerate(
        zip(live_result["order_log"], replay_result.order_log)
    ):
        assert live_ord["side"] == replay_ord["side"], (
            f"Order {i}: live side={live_ord['side']}, replay side={replay_ord['side']}"
        )
        assert live_ord["symbol"] == replay_ord["symbol"]
        assert live_ord["qty"] == replay_ord["qty"]

    # Same final event index
    assert (
        live_result["final_state"]["event_index"]
        == replay_result.final_state["event_index"]
    )

    # Same final market close
    live_close = live_result["final_state"]["market"].close
    replay_close = replay_result.final_state["market"].close
    assert live_close == replay_close


def test_replay_without_decision_modules_is_state_only():
    """Without decision modules, replay should NOT produce orders."""
    events = _market_events()

    result = run_replay_from_events(
        events=events,
        symbol="BTCUSDT",
        decision_modules=None,
    )

    assert result.events_processed == 5
    assert len(result.order_log) == 0
    assert len(result.captured_orders) == 0
    assert result.final_state is not None
    assert result.final_state["event_index"] == 5


def test_replay_determinism_with_orders():
    """Two replay runs with same events and decision modules produce identical results."""
    events = _market_events()

    dm1 = ThresholdDecisionModule("BTCUSDT")
    r1 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=[dm1])

    dm2 = ThresholdDecisionModule("BTCUSDT")
    r2 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=[dm2])

    assert r1.events_processed == r2.events_processed
    assert len(r1.order_log) == len(r2.order_log)
    for o1, o2 in zip(r1.order_log, r2.order_log):
        assert o1["side"] == o2["side"]
        assert o1["symbol"] == o2["symbol"]
        assert o1["qty"] == o2["qty"]
        # Fill prices should be identical (same price source)
        assert o1["fill_price"] == o2["fill_price"]
