"""Extended tests for event.bus — EventBus subscribe/publish/unsubscribe."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, List, Optional

import pytest

from event.bus import EventBus, Subscription
from event.types import (
    ControlEvent,
    EventType,
    FillEvent,
    FundingEvent,
    MarketEvent,
    RiskEvent,
    SignalEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _H:
    event_type: str = "test"
    event_id: Optional[str] = None


def _make_market(symbol: str = "ETHUSDT") -> MarketEvent:
    return MarketEvent(
        header=_H(event_type="market", event_id="m1"),
        ts=datetime(2024, 6, 1, tzinfo=timezone.utc),
        symbol=symbol,
        open=Decimal("3000"),
        high=Decimal("3100"),
        low=Decimal("2900"),
        close=Decimal("3050"),
        volume=Decimal("500"),
    )


def _make_signal() -> SignalEvent:
    return SignalEvent(
        header=_H(event_type="signal", event_id="s1"),
        signal_id="sig-001",
        symbol="ETHUSDT",
        side="long",
        strength=Decimal("0.75"),
    )


def _make_fill() -> FillEvent:
    return FillEvent(
        header=_H(event_type="fill", event_id="f1"),
        fill_id="fl-001",
        order_id="ord-001",
        symbol="ETHUSDT",
        qty=Decimal("1.0"),
        price=Decimal("3050"),
    )


def _make_risk() -> RiskEvent:
    return RiskEvent(
        header=_H(event_type="risk", event_id="r1"),
        rule_id="max_dd",
        level="warn",
        message="drawdown 4%",
    )


def _make_control() -> ControlEvent:
    return ControlEvent(
        header=_H(event_type="control", event_id="c1"),
        command="halt",
        reason="manual",
    )


def _make_funding() -> FundingEvent:
    return FundingEvent(
        header=_H(event_type="funding", event_id="fu1"),
        ts=datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc),
        symbol="ETHUSDT",
        funding_rate=Decimal("0.0001"),
        mark_price=Decimal("3050"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBusSubscribePublishRouting:
    """Routing by event_type string key."""

    def test_publish_routes_to_correct_event_type(self) -> None:
        bus = EventBus()
        market_received: List[Any] = []
        signal_received: List[Any] = []
        bus.subscribe(handler=market_received.append, event_type=EventType.MARKET)
        bus.subscribe(handler=signal_received.append, event_type=EventType.SIGNAL)

        bus.publish(_make_market())
        bus.publish(_make_signal())

        assert len(market_received) == 1
        assert len(signal_received) == 1
        assert market_received[0].symbol == "ETHUSDT"

    def test_publish_routes_risk_and_control(self) -> None:
        bus = EventBus()
        risk_r: List[Any] = []
        ctrl_r: List[Any] = []
        bus.subscribe(handler=risk_r.append, event_type=EventType.RISK)
        bus.subscribe(handler=ctrl_r.append, event_type=EventType.CONTROL)

        bus.publish(_make_risk())
        bus.publish(_make_control())

        assert len(risk_r) == 1
        assert len(ctrl_r) == 1

    def test_publish_routes_funding_event(self) -> None:
        bus = EventBus()
        received: List[Any] = []
        bus.subscribe(handler=received.append, event_type=EventType.FUNDING)
        bus.publish(_make_funding())
        assert len(received) == 1
        assert received[0].funding_rate == Decimal("0.0001")


class TestBusSubscribeAny:
    """subscribe_any receives all events regardless of type."""

    def test_subscribe_any_receives_all_types(self) -> None:
        bus = EventBus()
        all_events: List[Any] = []
        bus.subscribe_any(all_events.append)

        bus.publish(_make_market())
        bus.publish(_make_signal())
        bus.publish(_make_fill())
        bus.publish(_make_risk())
        bus.publish(_make_control())
        bus.publish(_make_funding())

        assert len(all_events) == 6

    def test_subscribe_any_and_typed_both_fire(self) -> None:
        bus = EventBus()
        any_r: List[Any] = []
        typed_r: List[Any] = []
        bus.subscribe_any(any_r.append)
        bus.subscribe(handler=typed_r.append, event_type=EventType.MARKET)

        bus.publish(_make_market())
        assert len(any_r) == 1
        assert len(typed_r) == 1


class TestBusUnsubscribe:
    """unsubscribe removes handler; silent on missing handler."""

    def test_unsubscribe_by_event_type(self) -> None:
        bus = EventBus()
        received: List[Any] = []
        handler = received.append
        bus.subscribe(handler=handler, event_type=EventType.MARKET)
        bus.unsubscribe(handler=handler, event_type=EventType.MARKET)

        bus.publish(_make_market())
        assert len(received) == 0

    def test_unsubscribe_by_event_cls(self) -> None:
        bus = EventBus()
        received: List[Any] = []
        handler = received.append
        bus.subscribe(handler=handler, event_cls=MarketEvent)
        bus.unsubscribe(handler=handler, event_cls=MarketEvent)

        bus.publish(_make_market())
        assert len(received) == 0

    def test_unsubscribe_missing_handler_is_silent(self) -> None:
        bus = EventBus()
        # Should not raise
        bus.unsubscribe(handler=lambda e: None, event_type=EventType.MARKET)

    def test_unsubscribe_any(self) -> None:
        bus = EventBus()
        received: List[Any] = []
        handler = received.append
        bus.subscribe_any(handler)
        bus.unsubscribe_any(handler)
        bus.publish(_make_market())
        assert len(received) == 0

    def test_unsubscribe_requires_type_or_cls(self) -> None:
        bus = EventBus()
        with pytest.raises(ValueError, match="至少"):
            bus.unsubscribe(handler=lambda e: None)


class TestBusExceptionIsolation:
    """Handler exceptions propagate (bus does not catch)."""

    def test_handler_exception_propagates(self) -> None:
        bus = EventBus()

        def bad_handler(e: Any) -> None:
            raise RuntimeError("boom")

        bus.subscribe(handler=bad_handler, event_type=EventType.MARKET)
        with pytest.raises(RuntimeError, match="boom"):
            bus.publish(_make_market())

    def test_first_handler_exception_prevents_second(self) -> None:
        """Bus calls handlers sequentially; exception in first stops second."""
        bus = EventBus()
        order: List[int] = []

        def h1(e: Any) -> None:
            order.append(1)
            raise RuntimeError("h1 fail")

        def h2(e: Any) -> None:
            order.append(2)

        bus.subscribe(handler=h1, event_type=EventType.MARKET)
        bus.subscribe(handler=h2, event_type=EventType.MARKET)

        with pytest.raises(RuntimeError):
            bus.publish(_make_market())
        assert order == [1]


class TestBusEmptyAndEdge:
    """Empty bus publish and edge cases."""

    def test_empty_bus_publish_no_error(self) -> None:
        bus = EventBus()
        bus.publish(_make_market())  # Should not raise

    def test_subscribe_both_type_and_cls(self) -> None:
        """Handler registered for both type and cls receives event twice."""
        bus = EventBus()
        received: List[Any] = []
        handler = received.append
        bus.subscribe(handler=handler, event_type=EventType.MARKET, event_cls=MarketEvent)
        bus.publish(_make_market())
        # Handler is in both _by_type and _by_cls -> called twice
        assert len(received) == 2

    def test_subscription_dataclass(self) -> None:
        """Subscription is a frozen dataclass."""
        handler = lambda e: None  # noqa: E731
        sub = Subscription(handler=handler, event_type="market")
        assert sub.handler is handler
        assert sub.event_type == "market"
        assert sub.event_cls is None
