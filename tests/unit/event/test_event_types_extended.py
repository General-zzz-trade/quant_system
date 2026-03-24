"""Extended tests for event.events and event.domain — construction and domain types."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pytest

from event.types import (
    ControlEvent,
    EventType,
    FillEvent,
    FundingEvent,
    IntentEvent,
    MarketEvent,
    Money,
    OrderEvent,
    OrderType,
    Price,
    Qty,
    RiskEvent,
    Side,
    SignalEvent,
    Symbol,
    TimeInForce,
    Venue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _H:
    event_type: str = "test"
    event_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Tests: Event type construction with all fields
# ---------------------------------------------------------------------------


class TestEventConstruction:
    def test_market_event_all_fields(self) -> None:
        evt = MarketEvent(
            header=_H(event_type="market"),
            ts=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            symbol="ETHUSDT",
            open=Decimal("3000"),
            high=Decimal("3100"),
            low=Decimal("2900"),
            close=Decimal("3050"),
            volume=Decimal("500"),
        )
        assert evt.event_type == EventType.MARKET
        assert evt.symbol == "ETHUSDT"
        assert evt.version == 1

    def test_signal_event_all_fields(self) -> None:
        evt = SignalEvent(
            header=_H(),
            signal_id="sig-1",
            symbol="BTCUSDT",
            side="long",
            strength=Decimal("0.95"),
        )
        assert evt.event_type == EventType.SIGNAL
        assert evt.strength == Decimal("0.95")

    def test_intent_event_all_fields(self) -> None:
        evt = IntentEvent(
            header=_H(),
            intent_id="int-1",
            symbol="ETHUSDT",
            side="buy",
            target_qty=Decimal("10"),
            reason_code="signal",
            origin="model_v8",
        )
        assert evt.event_type == EventType.INTENT
        assert evt.reason_code == "signal"

    def test_order_event_with_price(self) -> None:
        evt = OrderEvent(
            header=_H(),
            order_id="ord-1",
            intent_id="int-1",
            symbol="BTCUSDT",
            side="buy",
            qty=Decimal("0.1"),
            price=Decimal("42000"),
        )
        assert evt.price == Decimal("42000")

    def test_order_event_market_order_price_none(self) -> None:
        evt = OrderEvent(
            header=_H(),
            order_id="ord-2",
            intent_id="int-2",
            symbol="BTCUSDT",
            side="sell",
            qty=Decimal("0.5"),
            price=None,
        )
        assert evt.price is None

    def test_fill_event_optional_side(self) -> None:
        evt = FillEvent(
            header=_H(),
            fill_id="fl-1",
            order_id="ord-1",
            symbol="ETHUSDT",
            qty=Decimal("1"),
            price=Decimal("3050"),
        )
        assert evt.side is None

    def test_fill_event_with_side(self) -> None:
        evt = FillEvent(
            header=_H(),
            fill_id="fl-2",
            order_id="ord-2",
            symbol="ETHUSDT",
            qty=Decimal("1"),
            price=Decimal("3050"),
            side="buy",
        )
        assert evt.side == "buy"

    def test_funding_event_all_fields(self) -> None:
        evt = FundingEvent(
            header=_H(),
            ts=datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc),
            symbol="ETHUSDT",
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("3050"),
        )
        assert evt.event_type == EventType.FUNDING
        assert evt.version == 1


# ---------------------------------------------------------------------------
# Tests: Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    def test_intent_fields(self) -> None:
        evt = IntentEvent(
            header=_H(),
            intent_id="int-rt",
            symbol="SUIUSDT",
            side="sell",
            target_qty=Decimal("100"),
            reason_code="rebalance",
            origin="portfolio_v2",
        )
        assert evt.intent_id == "int-rt"
        assert evt.target_qty == Decimal("100")
        assert evt.origin == "portfolio_v2"

    def test_risk_fields(self) -> None:
        evt = RiskEvent(
            header=_H(),
            rule_id="max_leverage",
            level="block",
            message="leverage exceeded 3x",
        )
        assert evt.level == "block"
        assert evt.message == "leverage exceeded 3x"

    def test_control_event_variants(self) -> None:
        for cmd in ["halt", "reduce_only", "resume", "flush", "shutdown"]:
            evt = ControlEvent(
                header=_H(),
                command=cmd,
                reason=f"testing {cmd}",
            )
            assert evt.command == cmd

    def test_funding_event_fields(self) -> None:
        evt = FundingEvent(
            header=_H(),
            ts=datetime(2024, 6, 1, 8, 0, 0, tzinfo=timezone.utc),
            symbol="ETHUSDT",
            funding_rate=Decimal("0.0002"),
            mark_price=Decimal("3100"),
        )
        assert evt.ts.tzinfo is not None
        assert evt.funding_rate == Decimal("0.0002")


# ---------------------------------------------------------------------------
# Tests: Domain value types
# ---------------------------------------------------------------------------


class TestDomainTypes:
    def test_side_enum(self) -> None:
        assert Side.BUY.value == "buy"
        assert Side.SELL.value == "sell"

    def test_symbol_normalized(self) -> None:
        s = Symbol(value="ethusdt")
        assert s.normalized == "ETHUSDT"
        assert str(s) == "ETHUSDT"

    def test_venue_enum(self) -> None:
        assert Venue.BINANCE.value == "BINANCE"
        assert Venue.BYBIT.value == "BYBIT"
        assert Venue.SIM.value == "SIM"

    def test_qty_of(self) -> None:
        q = Qty.of(1.5)
        assert q.value == Decimal("1.5")

    def test_price_of(self) -> None:
        p = Price.of("42000.50")
        assert p.value == Decimal("42000.50")

    def test_money_of(self) -> None:
        m = Money.of(100)
        assert m.amount == Decimal("100")
        assert m.currency == "USDT"

    def test_money_custom_currency(self) -> None:
        m = Money.of(50, currency="BTC")
        assert m.currency == "BTC"

    def test_order_type_enum(self) -> None:
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"

    def test_time_in_force_enum(self) -> None:
        assert TimeInForce.GTC.value == "GTC"
        assert TimeInForce.IOC.value == "IOC"
        assert TimeInForce.FOK.value == "FOK"
        assert TimeInForce.GTX.value == "GTX"
