"""Runtime contract tests — verify minimum field contracts from runtime_contracts.md.

Ensures IntentEvent, OrderEvent, FillEvent, and StateSnapshot meet
the documented minimum field requirements.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest

sys.path.insert(0, "/quant_system")


# ============================================================
# a. Snapshot minimum fields
# ============================================================

class TestSnapshotMinimumFields:
    def test_coordinator_snapshot_has_minimum_fields(self):
        """StateSnapshot from coordinator must have market, positions, account, event_index."""
        from engine.coordinator import CoordinatorConfig, EngineCoordinator
        from event.header import EventHeader
        from event.types import EventType, MarketEvent

        cfg = CoordinatorConfig(symbol_default="BTCUSDT", currency="USDT")
        coord = EngineCoordinator(cfg=cfg)
        coord.start()

        ev = MarketEvent(
            header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="test"),
            ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("100"), high=Decimal("101"),
            low=Decimal("99"), close=Decimal("100.5"),
            volume=Decimal("10"),
        )
        coord.emit(ev, actor="test")
        view = coord.get_state_view()
        coord.stop()

        # Minimum snapshot fields per runtime_contracts.md
        assert "event_index" in view
        assert view["event_index"] >= 1
        # Market state
        market = view.get("market") or view.get("markets", {}).get("BTCUSDT")
        assert market is not None
        assert hasattr(market, "close") or hasattr(market, "close_f")


# ============================================================
# b. IntentEvent minimum fields
# ============================================================

class TestIntentEventMinimumFields:
    def test_intent_event_has_required_fields(self):
        """IntentEvent must have intent_id, symbol, side, target_qty."""
        from event.header import EventHeader
        from event.types import EventType, IntentEvent

        intent = IntentEvent(
            header=EventHeader.new_root(event_type=EventType.INTENT, version=1, source="test"),
            intent_id="intent-001",
            symbol="BTCUSDT",
            side="buy",
            target_qty=Decimal("0.01"),
            reason_code="test_entry",
            origin="test_module",
        )

        assert intent.intent_id == "intent-001"
        assert intent.symbol == "BTCUSDT"
        assert intent.side == "buy"
        assert intent.target_qty == Decimal("0.01")
        assert intent.reason_code == "test_entry"
        assert intent.origin == "test_module"


# ============================================================
# c. OrderEvent minimum fields
# ============================================================

class TestOrderEventMinimumFields:
    def test_order_event_has_required_fields(self):
        """OrderEvent must have order_id, symbol, side, qty."""
        from event.header import EventHeader
        from event.types import EventType, OrderEvent

        order = OrderEvent(
            header=EventHeader.new_root(event_type=EventType.ORDER, version=1, source="test"),
            order_id="order-001",
            intent_id="intent-001",
            symbol="BTCUSDT",
            side="BUY",
            qty=Decimal("0.01"),
            price=Decimal("50000"),
        )

        assert order.order_id == "order-001"
        assert order.intent_id == "intent-001"
        assert order.symbol == "BTCUSDT"
        assert order.side == "BUY"
        assert order.qty == Decimal("0.01")
        assert order.price == Decimal("50000")


# ============================================================
# d. FillEvent minimum fields
# ============================================================

class TestFillEventMinimumFields:
    def test_fill_event_has_required_fields(self):
        """FillEvent must have fill_id, order_id, symbol, qty, price."""
        from event.header import EventHeader
        from event.types import EventType, FillEvent

        fill = FillEvent(
            header=EventHeader.new_root(event_type=EventType.FILL, version=1, source="test"),
            fill_id="fill-001",
            order_id="order-001",
            symbol="BTCUSDT",
            qty=Decimal("0.01"),
            price=Decimal("50000"),
        )

        assert fill.fill_id == "fill-001"
        assert fill.order_id == "order-001"
        assert fill.symbol == "BTCUSDT"
        assert fill.qty == Decimal("0.01")
        assert fill.price == Decimal("50000")

    def test_rust_fill_event_has_required_fields(self):
        """RustFillEvent must have symbol, side, qty, price."""
        _hp = pytest.importorskip("_quant_hotpath")
        fill = _hp.RustFillEvent(
            symbol="BTCUSDT",
            side="buy",
            qty=0.01,
            price=50000.0,
        )
        assert fill.symbol == "BTCUSDT"
        assert fill.side == "buy"
        assert float(fill.qty) == pytest.approx(0.01)
        assert float(fill.price) == pytest.approx(50000.0)


# ============================================================
# e. Replay path contract
# ============================================================

class TestReplayPathContract:
    def test_replay_preserves_event_semantics(self):
        """Replay should process events through same coordinator path as live."""
        from engine.coordinator import CoordinatorConfig, EngineCoordinator
        from event.header import EventHeader
        from event.types import EventType, MarketEvent
        from runner.replay_runner import run_replay_from_events

        events = []
        for i in range(5):
            events.append(MarketEvent(
                header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="test"),
                ts=datetime(2026, 1, 1, 0, i, tzinfo=timezone.utc),
                symbol="BTCUSDT",
                open=Decimal("100"), high=Decimal("101"),
                low=Decimal("99"), close=Decimal(str(100 + i)),
                volume=Decimal("10"),
            ))

        result = run_replay_from_events(events=events, symbol="BTCUSDT")

        assert result.events_processed == 5
        assert result.final_state is not None
        assert result.final_state["event_index"] == 5
        # Market state should reflect last bar
        market = result.final_state.get("market")
        assert market is not None
        assert float(getattr(market, "close", getattr(market, "close_f", 0))) == 104.0
