from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from event.bootstrap import bootstrap_event_layer
from event.store import InMemoryEventStore
from event.types import EventType, MarketEvent
from event.header import EventHeader


def _make_market_event(symbol: str = 'BTCUSDT') -> MarketEvent:
    header = EventHeader.new_root(
        event_type=EventType.MARKET,
        version=MarketEvent.VERSION,
        source='test',
    )
    return MarketEvent(
        header=header,
        ts=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        symbol=symbol,
        open=Decimal('100.0'),
        high=Decimal('105.0'),
        low=Decimal('99.0'),
        close=Decimal('103.0'),
        volume=Decimal('1000.0'),
    )


class TestInMemoryEventStore:
    def test_append_iter_and_size(self) -> None:
        bootstrap_event_layer()
        store = InMemoryEventStore()
        event = _make_market_event()

        store.append(event)

        events = list(store.iter_events())
        assert store.size() == 1
        assert events == [event]
