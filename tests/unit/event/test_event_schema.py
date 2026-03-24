# tests/unit/event/test_event_schema.py
"""Event type unit tests — construction, field access, routing."""
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
# Tests: MarketEvent
# ---------------------------------------------------------------------------

class TestMarketEvent:
    def test_construction_and_fields(self) -> None:
        evt = MarketEvent(
            header=_market_header(),
            ts=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("42000"),
            high=Decimal("43000"),
            low=Decimal("41000"),
            close=Decimal("42500"),
            volume=Decimal("1000"),
        )
        assert evt.symbol == "BTCUSDT"
        assert evt.close == Decimal("42500")
        assert evt.ts.tzinfo is not None

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

    def test_rust_backing(self) -> None:
        evt = MarketEvent(
            header=_market_header(),
            ts=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("42000"), high=Decimal("43000"),
            low=Decimal("41000"), close=Decimal("42500"),
            volume=Decimal("1000"),
        )
        assert evt.to_rust() is not None


# ---------------------------------------------------------------------------
# Tests: SignalEvent
# ---------------------------------------------------------------------------

class TestSignalEvent:
    def test_construction(self) -> None:
        evt = SignalEvent(
            header=_Header(event_type="signal"),
            signal_id="sig-1",
            symbol="BTCUSDT",
            side="long",
            strength=Decimal("0.85"),
        )
        assert evt.signal_id == "sig-1"
        assert evt.strength == Decimal("0.85")


# ---------------------------------------------------------------------------
# Tests: IntentEvent
# ---------------------------------------------------------------------------

class TestIntentEvent:
    def test_construction(self) -> None:
        evt = IntentEvent(
            header=_Header(event_type="intent"),
            intent_id="int-1",
            symbol="BTCUSDT",
            side="buy",
            target_qty=Decimal("0.5"),
            reason_code="signal",
            origin="strat_a",
        )
        assert evt.target_qty == Decimal("0.5")
        assert evt.origin == "strat_a"


# ---------------------------------------------------------------------------
# Tests: OrderEvent
# ---------------------------------------------------------------------------

class TestOrderEvent:
    def test_construction_with_price(self) -> None:
        evt = OrderEvent(
            header=_Header(event_type="order"),
            order_id="ord-1",
            intent_id="int-1",
            symbol="BTCUSDT",
            side="buy",
            qty=Decimal("1.5"),
            price=Decimal("42000"),
        )
        assert evt.qty == Decimal("1.5")
        assert evt.price == Decimal("42000")

    def test_price_none(self) -> None:
        evt = OrderEvent(
            header=_Header(event_type="order"),
            order_id="ord-1",
            intent_id="int-1",
            symbol="BTCUSDT",
            side="buy",
            qty=Decimal("1.5"),
            price=None,
        )
        assert evt.price is None


# ---------------------------------------------------------------------------
# Tests: FillEvent
# ---------------------------------------------------------------------------

class TestFillEvent:
    def test_construction(self) -> None:
        evt = FillEvent(
            header=_fill_header(),
            fill_id="fl-1",
            order_id="ord-1",
            symbol="BTCUSDT",
            qty=Decimal("0.5"),
            price=Decimal("42500"),
        )
        assert evt.qty == Decimal("0.5")
        assert evt.price == Decimal("42500")
        assert evt.fill_id == "fl-1"


# ---------------------------------------------------------------------------
# Tests: RiskEvent and ControlEvent
# ---------------------------------------------------------------------------

class TestRiskControlEvent:
    def test_risk_construction(self) -> None:
        evt = RiskEvent(
            header=_Header(event_type="risk"),
            rule_id="r-1",
            level="warn",
            message="drawdown high",
        )
        assert evt.level == "warn"
        assert evt.message == "drawdown high"

    def test_control_construction(self) -> None:
        evt = ControlEvent(
            header=_Header(event_type="control"),
            command="halt",
            reason="manual stop",
        )
        assert evt.command == "halt"
        assert evt.reason == "manual stop"

    def test_control_reduce_only(self) -> None:
        evt = ControlEvent(
            header=_Header(event_type="control"),
            command="reduce_only",
            reason="manual reduce only",
        )
        assert evt.command == "reduce_only"
        assert evt.reason == "manual reduce only"
