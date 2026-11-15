from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from event.header import EventHeader
from event.types import EventType, FillEvent, IntentEvent, MarketEvent, OrderEvent
from state import AccountState, MarketState, PositionState
from state.snapshot import StateSnapshot

_SCALE = 100_000_000


def test_state_snapshot_contract_baseline() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snap = StateSnapshot.of(
        symbol="BTCUSDT",
        ts=ts,
        event_id="evt-1",
        event_type="market",
        bar_index=7,
        markets={
            "BTCUSDT": MarketState(
                symbol="BTCUSDT",
                last_price=101 * _SCALE,
                close=101 * _SCALE,
                last_ts=ts.isoformat(),
            )
        },
        positions={"BTCUSDT": PositionState.empty("BTCUSDT")},
        account=AccountState.initial(currency="USDT", balance=10000 * _SCALE),
        portfolio=None,
        risk=None,
    )

    assert snap.symbol == "BTCUSDT"
    assert snap.event_id == "evt-1"
    assert snap.event_type == "market"
    assert snap.bar_index == 7
    assert "BTCUSDT" in snap.markets
    assert "BTCUSDT" in snap.positions
    assert snap.account.currency == "USDT"


def test_event_minimum_fields_baseline() -> None:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)

    market = MarketEvent(
        header=EventHeader.new_root(
            event_type=EventType.MARKET,
            version=1,
            source="test",
        ),
        ts=ts,
        symbol="BTCUSDT",
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("10"),
    )
    intent = IntentEvent(
        header=EventHeader.new_root(
            event_type=EventType.INTENT,
            version=1,
            source="test",
        ),
        intent_id="intent-1",
        symbol="BTCUSDT",
        side="buy",
        target_qty=Decimal("1"),
        reason_code="signal",
        origin="test",
    )
    order = OrderEvent(
        header=EventHeader.new_root(
            event_type=EventType.ORDER,
            version=1,
            source="test",
        ),
        order_id="order-1",
        intent_id="intent-1",
        symbol="BTCUSDT",
        side="buy",
        qty=Decimal("1"),
        price=Decimal("100"),
    )
    fill = FillEvent(
        header=EventHeader.new_root(
            event_type=EventType.FILL,
            version=1,
            source="test",
        ),
        fill_id="fill-1",
        order_id="order-1",
        symbol="BTCUSDT",
        qty=Decimal("1"),
        price=Decimal("100"),
    )

    for event in (market, intent, order, fill):
        assert event.header.event_id
        assert event.header.ts_ns > 0
        assert event.header.source == "test"
