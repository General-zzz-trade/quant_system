# tests/unit/event/test_event_schema.py
"""EventBus and event type unit tests — subscribe, publish, routing, round-trip."""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, List, Optional

from event.types import (
    ControlEvent,
    EventType,
    FillEvent,
    IntentEvent,
    MarketEvent,
    OrderEvent,
    RiskEvent,
    SignalEvent,
)
from event.bus import EventBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Header:
    event_type: str = "test"
    ts: Optional[str] = None
    event_id: Optional[str] = None


def _market_header() -> _Header:
    return _Header(event_type="market", event_id="mkt-1")


def _fill_header() -> _Header:
    return _Header(event_type="fill", event_id="fill-1")


# ---------------------------------------------------------------------------
# Tests: EventType enum
# ---------------------------------------------------------------------------

class TestEventTypeEnum:
    def test_all_types_exist(self) -> None:
        expected = {"market", "signal", "intent", "order", "fill", "risk", "control", "funding"}
        actual = {e.value for e in EventType}
        assert actual == expected


# ---------------------------------------------------------------------------
# Tests: MarketEvent round-trip
# ---------------------------------------------------------------------------

class TestMarketEvent:
    def test_from_dict_iso_string(self) -> None:
        body = {
            "ts": "2024-01-15T12:00:00+00:00",
            "symbol": "BTCUSDT",
            "open": "42000",
            "high": "43000",
            "low": "41000",
            "close": "42500",
            "volume": "1000",
        }
        evt = MarketEvent.from_dict(header=_market_header(), body=body)
        assert evt.symbol == "BTCUSDT"
        assert evt.close == Decimal("42500")
        assert evt.ts.tzinfo is not None

    def test_from_dict_z_suffix(self) -> None:
        body = {
            "ts": "2024-01-15T12:00:00Z",
            "symbol": "BTCUSDT",
            "open": "42000", "high": "43000", "low": "41000",
            "close": "42500", "volume": "1000",
        }
        evt = MarketEvent.from_dict(header=_market_header(), body=body)
        assert evt.ts.tzinfo is not None

    def test_from_dict_naive_ts_raises(self) -> None:
        body = {
            "ts": "2024-01-15T12:00:00",
            "symbol": "BTCUSDT",
            "open": "42000", "high": "43000", "low": "41000",
            "close": "42500", "volume": "1000",
        }
        with pytest.raises(ValueError, match="tz-aware"):
            MarketEvent.from_dict(header=_market_header(), body=body)

    def test_to_dict(self) -> None:
        evt = MarketEvent(
            header=_market_header(),
            ts=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("42000"), high=Decimal("43000"),
            low=Decimal("41000"), close=Decimal("42500"),
            volume=Decimal("1000"),
        )
        d = evt.to_dict()
        assert d["symbol"] == "BTCUSDT"
        assert d["close"] == Decimal("42500")

    def test_version(self) -> None:
        evt = MarketEvent(
            header=_market_header(),
            ts=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("0"), high=Decimal("0"),
            low=Decimal("0"), close=Decimal("0"),
            volume=Decimal("0"),
        )
        assert evt.version == 1


# ---------------------------------------------------------------------------
# Tests: SignalEvent
# ---------------------------------------------------------------------------

class TestSignalEvent:
    def test_round_trip(self) -> None:
        body = {"signal_id": "sig-1", "symbol": "BTCUSDT", "side": "long", "strength": "0.85"}
        evt = SignalEvent.from_dict(header=_Header(event_type="signal"), body=body)
        assert evt.signal_id == "sig-1"
        assert evt.strength == Decimal("0.85")
        d = evt.to_dict()
        assert d["signal_id"] == "sig-1"


# ---------------------------------------------------------------------------
# Tests: IntentEvent
# ---------------------------------------------------------------------------

class TestIntentEvent:
    def test_round_trip(self) -> None:
        body = {
            "intent_id": "int-1", "symbol": "BTCUSDT", "side": "buy",
            "target_qty": "0.5", "reason_code": "signal", "origin": "strat_a",
        }
        evt = IntentEvent.from_dict(header=_Header(event_type="intent"), body=body)
        assert evt.target_qty == Decimal("0.5")
        d = evt.to_dict()
        assert d["origin"] == "strat_a"


# ---------------------------------------------------------------------------
# Tests: OrderEvent
# ---------------------------------------------------------------------------

class TestOrderEvent:
    def test_round_trip(self) -> None:
        body = {
            "order_id": "ord-1", "intent_id": "int-1", "symbol": "BTCUSDT",
            "side": "buy", "qty": "1.5", "price": "42000",
        }
        evt = OrderEvent.from_dict(header=_Header(event_type="order"), body=body)
        assert evt.qty == Decimal("1.5")
        assert evt.price == Decimal("42000")

    def test_price_none(self) -> None:
        body = {
            "order_id": "ord-1", "intent_id": "int-1", "symbol": "BTCUSDT",
            "side": "buy", "qty": "1.5", "price": None,
        }
        evt = OrderEvent.from_dict(header=_Header(event_type="order"), body=body)
        assert evt.price is None


# ---------------------------------------------------------------------------
# Tests: FillEvent
# ---------------------------------------------------------------------------

class TestFillEvent:
    def test_round_trip(self) -> None:
        body = {
            "fill_id": "fl-1", "order_id": "ord-1", "symbol": "BTCUSDT",
            "qty": "0.5", "price": "42500",
        }
        evt = FillEvent.from_dict(header=_fill_header(), body=body)
        assert evt.qty == Decimal("0.5")
        assert evt.price == Decimal("42500")
        d = evt.to_dict()
        assert d["fill_id"] == "fl-1"


# ---------------------------------------------------------------------------
# Tests: RiskEvent and ControlEvent
# ---------------------------------------------------------------------------

class TestRiskControlEvent:
    def test_risk_round_trip(self) -> None:
        body = {"rule_id": "r-1", "level": "warn", "message": "drawdown high"}
        evt = RiskEvent.from_dict(header=_Header(event_type="risk"), body=body)
        assert evt.level == "warn"
        d = evt.to_dict()
        assert d["message"] == "drawdown high"

    def test_control_round_trip(self) -> None:
        body = {"command": "halt", "reason": "manual stop"}
        evt = ControlEvent.from_dict(header=_Header(event_type="control"), body=body)
        assert evt.command == "halt"
        d = evt.to_dict()
        assert d["reason"] == "manual stop"

    def test_control_reduce_only_round_trip(self) -> None:
        body = {"command": "reduce_only", "reason": "manual reduce only"}
        evt = ControlEvent.from_dict(header=_Header(event_type="control"), body=body)
        assert evt.command == "reduce_only"
        d = evt.to_dict()
        assert d["reason"] == "manual reduce only"


# ---------------------------------------------------------------------------
# Tests: EventBus subscribe / publish
# ---------------------------------------------------------------------------

class TestEventBus:
    def test_subscribe_by_type(self) -> None:
        bus = EventBus()
        received: List[Any] = []
        # EventType.MARKET is the Enum member (not the string "market")
        bus.subscribe(handler=lambda e: received.append(e), event_type=EventType.MARKET)
        evt = MarketEvent(
            header=_market_header(),
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("0"), high=Decimal("0"),
            low=Decimal("0"), close=Decimal("0"), volume=Decimal("0"),
        )
        bus.publish(evt)
        assert len(received) == 1

    def test_subscribe_by_class(self) -> None:
        bus = EventBus()
        received: List[Any] = []
        bus.subscribe(handler=lambda e: received.append(e), event_cls=MarketEvent)
        evt = MarketEvent(
            header=_market_header(),
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("0"), high=Decimal("0"),
            low=Decimal("0"), close=Decimal("0"), volume=Decimal("0"),
        )
        bus.publish(evt)
        assert len(received) == 1

    def test_subscribe_any(self) -> None:
        bus = EventBus()
        received: List[Any] = []
        bus.subscribe_any(handler=lambda e: received.append(e))
        evt = MarketEvent(
            header=_market_header(),
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("0"), high=Decimal("0"),
            low=Decimal("0"), close=Decimal("0"), volume=Decimal("0"),
        )
        bus.publish(evt)
        assert len(received) == 1

    def test_subscribe_requires_type_or_cls(self) -> None:
        bus = EventBus()
        with pytest.raises(ValueError, match="至少"):
            bus.subscribe(handler=lambda e: None)

    def test_no_match_no_call(self) -> None:
        bus = EventBus()
        received: List[Any] = []
        bus.subscribe(handler=lambda e: received.append(e), event_type=EventType.SIGNAL)
        # Publish a market event → signal handler should not fire
        evt = MarketEvent(
            header=_market_header(),
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("0"), high=Decimal("0"),
            low=Decimal("0"), close=Decimal("0"), volume=Decimal("0"),
        )
        bus.publish(evt)
        assert len(received) == 0

    def test_multiple_handlers_order(self) -> None:
        bus = EventBus()
        order: List[int] = []
        bus.subscribe(handler=lambda e: order.append(1), event_type=EventType.MARKET)
        bus.subscribe(handler=lambda e: order.append(2), event_type=EventType.MARKET)
        evt = MarketEvent(
            header=_market_header(),
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("0"), high=Decimal("0"),
            low=Decimal("0"), close=Decimal("0"), volume=Decimal("0"),
        )
        bus.publish(evt)
        assert order == [1, 2]
