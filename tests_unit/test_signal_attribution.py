"""Tests for signal attribution: intent→order→fill chain attribution."""
from __future__ import annotations

import pytest
from attribution.signal_attribution import (
    SignalPnL,
    SignalAttributionReport,
    attribute_by_signal,
)
from attribution.tracker import AttributionTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intent(intent_id: str, origin: str, symbol: str = "BTCUSDT", side: str = "buy"):
    return {"intent_id": intent_id, "origin": origin, "symbol": symbol, "side": side}


def _order(order_id: str, intent_id: str, symbol: str = "BTCUSDT", side: str = "buy"):
    return {"order_id": order_id, "intent_id": intent_id, "symbol": symbol, "side": side}


def _fill(fill_id: str, order_id: str, symbol: str = "BTCUSDT",
          side: str = "buy", qty: float = 1.0, price: float = 100.0, fee: float = 0.0):
    return {
        "fill_id": fill_id, "order_id": order_id, "symbol": symbol,
        "side": side, "qty": qty, "price": price, "fee": fee,
    }


# ---------------------------------------------------------------------------
# attribute_by_signal
# ---------------------------------------------------------------------------

class TestAttributeBySignal:

    def test_single_signal_roundtrip(self):
        """Buy then sell under one signal → realized P&L attributed."""
        intents = [_intent("i1", "factor:momentum_20")]
        orders = [_order("o1", "i1"), _order("o2", "i1")]
        fills = [
            _fill("f1", "o1", side="buy", qty=1.0, price=100.0),
            _fill("f2", "o2", side="sell", qty=1.0, price=110.0),
        ]
        report = attribute_by_signal(intents, orders, fills)

        assert "factor:momentum_20" in report.by_signal
        sig = report.by_signal["factor:momentum_20"]
        assert sig.realized_pnl == pytest.approx(10.0)
        assert sig.trade_count == 2
        assert sig.win_rate == pytest.approx(1.0)

    def test_multiple_signals(self):
        """Two different signals → separate attribution."""
        intents = [
            _intent("i1", "factor:momentum_20"),
            _intent("i2", "factor:rsi_14"),
        ]
        orders = [
            _order("o1", "i1"),
            _order("o2", "i1"),
            _order("o3", "i2"),
            _order("o4", "i2"),
        ]
        fills = [
            _fill("f1", "o1", side="buy", qty=1.0, price=100.0),
            _fill("f2", "o2", side="sell", qty=1.0, price=110.0),
            _fill("f3", "o3", side="buy", qty=1.0, price=200.0),
            _fill("f4", "o4", side="sell", qty=1.0, price=195.0),
        ]
        report = attribute_by_signal(intents, orders, fills)

        assert len(report.by_signal) == 2
        mom = report.by_signal["factor:momentum_20"]
        rsi = report.by_signal["factor:rsi_14"]
        assert mom.realized_pnl == pytest.approx(10.0)
        assert rsi.realized_pnl == pytest.approx(-5.0)

    def test_unattributed_fills(self):
        """Fills with no matching intent → unattributed bucket."""
        intents = []
        orders = []
        fills = [
            _fill("f1", "orphan_order", side="buy", qty=1.0, price=100.0),
            _fill("f2", "orphan_order", side="sell", qty=1.0, price=105.0),
        ]
        report = attribute_by_signal(intents, orders, fills)

        assert len(report.by_signal) == 0
        assert report.unattributed_pnl == pytest.approx(5.0)

    def test_pnl_identity(self):
        """sum(by_signal PnL) + unattributed == total_pnl."""
        intents = [_intent("i1", "sig_a")]
        orders = [_order("o1", "i1"), _order("o2", "i1")]
        fills = [
            _fill("f1", "o1", side="buy", qty=2.0, price=50.0, fee=0.5),
            _fill("f2", "o2", side="sell", qty=2.0, price=55.0, fee=0.5),
            _fill("f3", "orphan", side="buy", qty=1.0, price=80.0, fee=0.1),
            _fill("f4", "orphan", side="sell", qty=1.0, price=82.0, fee=0.1),
        ]
        report = attribute_by_signal(intents, orders, fills)

        attributed_sum = sum(
            s.realized_pnl + s.unrealized_pnl - s.fee_cost
            for s in report.by_signal.values()
        )
        assert report.total_pnl == pytest.approx(attributed_sum + report.unattributed_pnl)

    def test_unrealized_pnl(self):
        """Open position → unrealized P&L from current_prices."""
        intents = [_intent("i1", "sig_a")]
        orders = [_order("o1", "i1")]
        fills = [_fill("f1", "o1", side="buy", qty=1.0, price=100.0)]
        report = attribute_by_signal(intents, orders, fills, current_prices={"BTCUSDT": 120.0})

        sig = report.by_signal["sig_a"]
        assert sig.unrealized_pnl == pytest.approx(20.0)
        assert sig.realized_pnl == pytest.approx(0.0)

    def test_fees_deducted(self):
        """Fees are tracked in fee_cost and reduce total_pnl."""
        intents = [_intent("i1", "sig_a")]
        orders = [_order("o1", "i1"), _order("o2", "i1")]
        fills = [
            _fill("f1", "o1", side="buy", qty=1.0, price=100.0, fee=1.0),
            _fill("f2", "o2", side="sell", qty=1.0, price=110.0, fee=1.0),
        ]
        report = attribute_by_signal(intents, orders, fills)

        sig = report.by_signal["sig_a"]
        assert sig.fee_cost == pytest.approx(2.0)
        assert sig.realized_pnl == pytest.approx(10.0)
        # total_pnl = realized - fees = 10 - 2 = 8
        assert report.total_pnl == pytest.approx(8.0)

    def test_empty_inputs(self):
        """No events → empty report."""
        report = attribute_by_signal([], [], [])
        assert report.total_pnl == pytest.approx(0.0)
        assert report.unattributed_pnl == pytest.approx(0.0)
        assert len(report.by_signal) == 0

    def test_losing_trade_win_rate(self):
        """Losing trade → win_rate = 0."""
        intents = [_intent("i1", "sig_a")]
        orders = [_order("o1", "i1"), _order("o2", "i1")]
        fills = [
            _fill("f1", "o1", side="buy", qty=1.0, price=100.0),
            _fill("f2", "o2", side="sell", qty=1.0, price=90.0),
        ]
        report = attribute_by_signal(intents, orders, fills)
        assert report.by_signal["sig_a"].win_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# AttributionTracker
# ---------------------------------------------------------------------------

class TestAttributionTracker:

    def test_tracker_basic(self):
        """Tracker accumulates events and produces report."""
        tracker = AttributionTracker()
        tracker.on_intent(_intent("i1", "sig_a"))
        tracker.on_order(_order("o1", "i1"))
        tracker.on_fill(_fill("f1", "o1", side="buy", qty=1.0, price=100.0))
        tracker.on_fill(_fill("f2", "o1", side="sell", qty=1.0, price=110.0))

        # Need a second order for the sell
        tracker.on_order(_order("o2", "i1"))
        # Re-do: order o1 has fills f1 and f2 both
        report = tracker.report()
        assert "sig_a" in report.by_signal

    def test_tracker_counts(self):
        tracker = AttributionTracker()
        tracker.on_intent(_intent("i1", "sig_a"))
        tracker.on_order(_order("o1", "i1"))
        tracker.on_fill(_fill("f1", "o1"))

        assert tracker.intent_count == 1
        assert tracker.order_count == 1
        assert tracker.fill_count == 1

    def test_tracker_on_event_routing(self):
        """on_event dispatches by type name."""

        class FakeIntent:
            intent_id = "i1"
            symbol = "BTCUSDT"
            side = "buy"
            origin = "sig_a"

        class FakeOrder:
            order_id = "o1"
            intent_id = "i1"
            symbol = "BTCUSDT"
            side = "buy"

        class FakeFill:
            fill_id = "f1"
            order_id = "o1"
            symbol = "BTCUSDT"
            side = "buy"
            qty = "1.0"
            price = "100.0"
            fee = "0"

        # Rename classes to match event type routing
        FakeIntent.__name__ = "IntentEvent"
        FakeOrder.__name__ = "OrderEvent"
        FakeFill.__name__ = "FillEvent"

        tracker = AttributionTracker()
        tracker.on_event(FakeIntent())
        tracker.on_event(FakeOrder())
        tracker.on_event(FakeFill())

        assert tracker.intent_count == 1
        assert tracker.order_count == 1
        assert tracker.fill_count == 1

    def test_tracker_with_current_prices(self):
        tracker = AttributionTracker()
        tracker.on_intent(_intent("i1", "sig_a"))
        tracker.on_order(_order("o1", "i1"))
        tracker.on_fill(_fill("f1", "o1", side="buy", qty=1.0, price=100.0))

        report = tracker.report(current_prices={"BTCUSDT": 150.0})
        sig = report.by_signal["sig_a"]
        assert sig.unrealized_pnl == pytest.approx(50.0)

    def test_multiple_signals_via_tracker(self):
        """Multiple signals tracked correctly."""
        tracker = AttributionTracker()

        tracker.on_intent(_intent("i1", "factor:momentum"))
        tracker.on_intent(_intent("i2", "factor:rsi"))
        tracker.on_order(_order("o1", "i1"))
        tracker.on_order(_order("o2", "i1"))
        tracker.on_order(_order("o3", "i2"))
        tracker.on_order(_order("o4", "i2"))

        tracker.on_fill(_fill("f1", "o1", side="buy", qty=1.0, price=100.0))
        tracker.on_fill(_fill("f2", "o2", side="sell", qty=1.0, price=120.0))
        tracker.on_fill(_fill("f3", "o3", side="buy", qty=1.0, price=200.0))
        tracker.on_fill(_fill("f4", "o4", side="sell", qty=1.0, price=190.0))

        report = tracker.report()
        assert len(report.by_signal) == 2
        assert report.by_signal["factor:momentum"].realized_pnl == pytest.approx(20.0)
        assert report.by_signal["factor:rsi"].realized_pnl == pytest.approx(-10.0)
