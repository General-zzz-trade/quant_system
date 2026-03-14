"""Replay: snapshot hash consistency tests.

Verifies that running the same event sequence twice produces identical
state snapshots (hash equality).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable

from event.header import EventHeader
from event.types import EventType, MarketEvent, OrderEvent
from runner.replay_runner import run_replay_from_events


class ThresholdDecisionModule:
    """Deterministic decision module for hash consistency testing."""

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


def _market_events(n: int = 15, symbol: str = "BTCUSDT") -> list[MarketEvent]:
    """Generate n deterministic market events."""
    prices = [
        Decimal("100"), Decimal("101"), Decimal("101.5"), Decimal("99"),
        Decimal("98.5"), Decimal("100.8"), Decimal("101.2"), Decimal("99.3"),
        Decimal("98"), Decimal("100.6"), Decimal("101.1"), Decimal("100"),
        Decimal("99.2"), Decimal("101.3"), Decimal("100.1"),
    ]
    events = []
    for i in range(min(n, len(prices))):
        ts = datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc)
        c = prices[i]
        events.append(MarketEvent(
            header=EventHeader.new_root(
                event_type=EventType.MARKET, version=1, source="test",
            ),
            ts=ts,
            symbol=symbol,
            open=c - Decimal("0.1"),
            high=c + Decimal("0.5"),
            low=c - Decimal("0.5"),
            close=c,
            volume=Decimal("10"),
        ))
    return events


def _state_hash(state: dict) -> str:
    """Compute deterministic hash of state snapshot."""
    # Extract key state fields that should be deterministic
    key_fields = {}
    key_fields["event_index"] = state.get("event_index")

    mkt = state.get("market")
    if mkt is not None:
        key_fields["market_close"] = str(getattr(mkt, "close", ""))
        key_fields["market_high"] = str(getattr(mkt, "high", ""))
        key_fields["market_low"] = str(getattr(mkt, "low", ""))

    # Serialize deterministically
    canonical = json.dumps(key_fields, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_snapshot_hash_consistency():
    """Same event sequence run twice must produce identical state hash."""
    events = _market_events()

    dm1 = ThresholdDecisionModule("BTCUSDT")
    r1 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=[dm1])

    dm2 = ThresholdDecisionModule("BTCUSDT")
    r2 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=[dm2])

    h1 = _state_hash(r1.final_state)
    h2 = _state_hash(r2.final_state)

    assert h1 == h2, f"State hashes differ:\n  run1: {h1}\n  run2: {h2}"


def test_snapshot_hash_without_decisions():
    """State-only replay (no decision modules) also produces consistent hashes."""
    events = _market_events()

    r1 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=None)
    r2 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=None)

    h1 = _state_hash(r1.final_state)
    h2 = _state_hash(r2.final_state)

    assert h1 == h2, f"State hashes differ:\n  run1: {h1}\n  run2: {h2}"


def test_snapshot_hash_order_log_consistency():
    """Order logs from two identical runs must match exactly."""
    events = _market_events()

    dm1 = ThresholdDecisionModule("BTCUSDT")
    r1 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=[dm1])

    dm2 = ThresholdDecisionModule("BTCUSDT")
    r2 = run_replay_from_events(events=events, symbol="BTCUSDT", decision_modules=[dm2])

    assert len(r1.order_log) == len(r2.order_log)
    for o1, o2 in zip(r1.order_log, r2.order_log):
        assert o1["side"] == o2["side"]
        assert o1["symbol"] == o2["symbol"]
        assert o1["qty"] == o2["qty"]
        assert o1["fill_price"] == o2["fill_price"]
