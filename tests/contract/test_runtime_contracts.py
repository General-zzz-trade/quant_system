from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from event.header import EventHeader
from event.types import EventType, FillEvent, IntentEvent, MarketEvent, OrderEvent
from runner.backtest_runner import MovingAverageCrossModule
from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.snapshot import StateSnapshot


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
                last_price=Decimal("101"),
                close=Decimal("101"),
                last_ts=ts,
            )
        },
        positions={"BTCUSDT": PositionState.empty("BTCUSDT")},
        account=AccountState.initial(currency="USDT", balance=Decimal("10000")),
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


def test_backtest_decision_events_follow_contract_baseline() -> None:
    mod = MovingAverageCrossModule(symbol="BTCUSDT", window=3, order_qty=Decimal("0.01"))

    def _market_snapshot(close: str):
        return StateSnapshot.of(
            symbol="BTCUSDT",
            ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
            event_id="mkt-evt",
            event_type="market",
            bar_index=1,
            markets={
                "BTCUSDT": MarketState(
                    symbol="BTCUSDT",
                    last_price=Decimal(close),
                    close=Decimal(close),
                )
            },
            positions={"BTCUSDT": PositionState.empty("BTCUSDT")},
            account=AccountState.initial(currency="USDT", balance=Decimal("10000")),
        )

    mod.decide(_market_snapshot("100"))
    mod.decide(_market_snapshot("100"))
    events = list(mod.decide(_market_snapshot("110")))

    assert len(events) == 2
    intent, order = events

    assert isinstance(intent, IntentEvent)
    assert isinstance(order, OrderEvent)
    assert isinstance(intent.header, EventHeader)
    assert isinstance(order.header, EventHeader)

    assert intent.intent_id
    assert intent.symbol == "BTCUSDT"
    assert intent.side == "buy"
    assert intent.target_qty == Decimal("0.01")
    assert intent.reason_code
    assert intent.origin

    assert order.order_id
    assert order.intent_id == intent.intent_id
    assert order.symbol == "BTCUSDT"
    assert order.side == "buy"
    assert order.qty == Decimal("0.01")
    assert order.header.parent_event_id == intent.header.event_id


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

