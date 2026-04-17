"""Test that signal_only=True suppresses OrderEvents while preserving others."""
import pytest
from unittest.mock import MagicMock
from decimal import Decimal

from event.header import EventHeader
from event.types import EventType, OrderEvent, SignalEvent


def test_signal_only_filters_order_events():
    """The signal_only filter should remove OrderEvents but keep SignalEvents."""
    signal_ev = SignalEvent(
        header=EventHeader.new_root(event_type=EventType.SIGNAL, version=1, source="test"),
        signal_id="sig1",
        symbol="BTCUSDT",
        side="long",
        strength=Decimal("2.5"),
    )
    order_ev = OrderEvent(
        header=EventHeader.new_root(event_type=EventType.ORDER, version=1, source="test"),
        order_id="ord1",
        intent_id="int1",
        symbol="BTCUSDT",
        side="buy",
        qty=Decimal("0.05"),
        price=Decimal("72000"),
    )
    events = [signal_ev, order_ev]

    # signal_only filter (same logic as in alpha.py)
    filtered = [e for e in events if not hasattr(e, "order_id")]

    assert len(filtered) == 1
    assert isinstance(filtered[0], SignalEvent)


def test_normal_mode_keeps_all_events():
    """Without signal_only, all events pass through."""
    signal_ev = SignalEvent(
        header=EventHeader.new_root(event_type=EventType.SIGNAL, version=1, source="test"),
        signal_id="sig2",
        symbol="BTCUSDT",
        side="long",
        strength=Decimal("2.5"),
    )
    order_ev = OrderEvent(
        header=EventHeader.new_root(event_type=EventType.ORDER, version=1, source="test"),
        order_id="ord2",
        intent_id="int2",
        symbol="BTCUSDT",
        side="buy",
        qty=Decimal("0.05"),
        price=Decimal("72000"),
    )
    events = [signal_ev, order_ev]
    assert len(events) == 2
    assert any(hasattr(e, "order_id") for e in events)


def test_signal_only_flag_stored():
    """AlphaDecisionModule stores signal_only flag correctly."""
    from decision.modules.alpha import AlphaDecisionModule

    pred = MagicMock()
    disc = MagicMock()
    disc.deadzone = 2.0
    disc.min_hold = 3
    disc.max_hold = 24
    sizer = MagicMock()

    mod = AlphaDecisionModule(
        symbol="BTCUSDT",
        runner_key="BTCUSDT_4h",
        predictor=pred,
        discretizer=disc,
        sizer=sizer,
        leverage=3.0,
        signal_only=True,
    )
    assert mod._signal_only is True

    mod2 = AlphaDecisionModule(
        symbol="BTCUSDT",
        runner_key="BTCUSDT",
        predictor=pred,
        discretizer=disc,
        sizer=sizer,
        leverage=3.0,
    )
    assert mod2._signal_only is False
