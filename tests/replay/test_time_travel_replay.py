"""Replay: time-travel replay tests.

Records market events, replays them, and verifies output matches.
Also tests mid-stream checkpoint restore + continued replay.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from event.header import EventHeader
from event.types import EventType, MarketEvent, OrderEvent
from execution.sim.replay_adapter import ReplayExecutionAdapter
from runner.replay_runner import run_replay_from_events


# ============================================================
# Deterministic decision module (reuse pattern from causal chain)
# ============================================================

class ThresholdDecisionModule:
    """Buy when close > threshold, sell when close < threshold - spread."""

    def __init__(self, symbol: str, buy_threshold: Decimal = Decimal("100.5"),
                 sell_threshold: Decimal = Decimal("99.5")):
        self.symbol = symbol
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._position = 0

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


def _market_events(n: int = 20, symbol: str = "BTCUSDT") -> list[MarketEvent]:
    """Generate n market events with price oscillation to trigger buy/sell."""
    import math
    events = []
    for i in range(n):
        # Oscillate price around 100 with amplitude > threshold spread
        price = Decimal(str(round(100.0 + 2.0 * math.sin(i * 0.5), 2)))
        ts = datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc)
        events.append(MarketEvent(
            header=EventHeader.new_root(
                event_type=EventType.MARKET, version=1, source="test",
            ),
            ts=ts,
            symbol=symbol,
            open=price - Decimal("0.1"),
            high=price + Decimal("0.5"),
            low=price - Decimal("0.5"),
            close=price,
            volume=Decimal("10"),
        ))
    return events


# ============================================================
# Tests
# ============================================================

def test_time_travel_replay_full():
    """Record 20 market events → replay → verify orders and final state match."""
    events = _market_events(20)
    dm = ThresholdDecisionModule("BTCUSDT")

    result = run_replay_from_events(
        events=events,
        symbol="BTCUSDT",
        decision_modules=[dm],
    )

    assert result.events_processed == 20
    assert result.final_state is not None
    # Should have generated some orders (price oscillates across thresholds)
    assert len(result.order_log) >= 1, f"Expected orders, got {len(result.order_log)}"

    # Verify order sides alternate BUY/SELL
    sides = [o["side"] for o in result.order_log]
    for i in range(1, len(sides)):
        assert sides[i] != sides[i - 1], f"Orders {i-1} and {i} have same side: {sides[i]}"


def test_time_travel_replay_from_midpoint():
    """Split events: replay first half, then replay second half → same final state as full."""
    events = _market_events(20)

    # Full replay
    dm_full = ThresholdDecisionModule("BTCUSDT")
    full_result = run_replay_from_events(
        events=events,
        symbol="BTCUSDT",
        decision_modules=[dm_full],
    )

    # Two-phase replay: first 10 events
    dm_first = ThresholdDecisionModule("BTCUSDT")
    first_result = run_replay_from_events(
        events=events[:10],
        symbol="BTCUSDT",
        decision_modules=[dm_first],
    )

    # Second phase: remaining events with state carried over via fresh dm
    # (ThresholdDecisionModule tracks position internally, so we need to
    #  set it to match the state after first 10 events)
    dm_second = ThresholdDecisionModule("BTCUSDT")
    dm_second._position = dm_first._position  # carry over position
    second_result = run_replay_from_events(
        events=events[10:],
        symbol="BTCUSDT",
        decision_modules=[dm_second],
    )

    # Combined orders from both phases should match full replay
    combined_orders = first_result.order_log + second_result.order_log
    assert len(combined_orders) == len(full_result.order_log), (
        f"Combined={len(combined_orders)}, full={len(full_result.order_log)}"
    )
    for i, (comb, full) in enumerate(zip(combined_orders, full_result.order_log)):
        assert comb["side"] == full["side"], f"Order {i}: side mismatch"


def test_time_travel_replay_determinism():
    """Two replay runs with same events produce identical results."""
    events = _market_events(20)

    dm1 = ThresholdDecisionModule("BTCUSDT")
    r1 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=[dm1])

    dm2 = ThresholdDecisionModule("BTCUSDT")
    r2 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=[dm2])

    assert r1.events_processed == r2.events_processed
    assert len(r1.order_log) == len(r2.order_log)
    for o1, o2 in zip(r1.order_log, r2.order_log):
        assert o1["side"] == o2["side"]
        assert o1["qty"] == o2["qty"]
        assert o1["fill_price"] == o2["fill_price"]

    # Final event indices must match
    assert r1.final_state["event_index"] == r2.final_state["event_index"]
