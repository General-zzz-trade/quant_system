# tests/unit/event/test_sqlite_event_store.py
"""Tests for SQLiteEventStore — persistent event store."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from event.bootstrap import bootstrap_event_layer
from event.store import EventStore, InMemoryEventStore, SQLiteEventStore
from event.types import EventType, MarketEvent
from event.header import EventHeader


def _make_market_event(symbol: str = "BTCUSDT") -> MarketEvent:
    """Create a minimal MarketEvent with real EventHeader for codec round-trip."""
    header = EventHeader.new_root(
        event_type=EventType.MARKET,
        version=MarketEvent.VERSION,
        source="test",
    )
    return MarketEvent(
        header=header,
        ts=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        symbol=symbol,
        open=Decimal("100.0"),
        high=Decimal("105.0"),
        low=Decimal("99.0"),
        close=Decimal("103.0"),
        volume=Decimal("1000.0"),
    )


@pytest.fixture(autouse=True)
def _bootstrap_codecs():
    """Ensure codec registry is populated for encode/decode."""
    bootstrap_event_layer()


class TestSQLiteEventStore:
    def test_append_and_iter(self, tmp_path):
        db_path = str(tmp_path / "events.db")
        store = SQLiteEventStore(db_path)

        event = _make_market_event()
        store.append(event)

        events = list(store.iter_events())
        assert len(events) == 1
        assert events[0].event_type == EventType.MARKET
        store.close()

    def test_size(self, tmp_path):
        db_path = str(tmp_path / "events.db")
        store = SQLiteEventStore(db_path)

        assert store.size() == 0
        store.append(_make_market_event())
        assert store.size() == 1
        store.append(_make_market_event("ETHUSDT"))
        assert store.size() == 2
        store.close()

    def test_rejects_non_base_event(self, tmp_path):
        db_path = str(tmp_path / "events.db")
        store = SQLiteEventStore(db_path)

        with pytest.raises(Exception, match="BaseEvent"):
            store.append({"not": "an event"})
        store.close()

    def test_persistence_across_reopen(self, tmp_path):
        db_path = str(tmp_path / "events.db")

        store1 = SQLiteEventStore(db_path)
        store1.append(_make_market_event())
        store1.append(_make_market_event("ETHUSDT"))
        store1.close()

        store2 = SQLiteEventStore(db_path)
        assert store2.size() == 2
        events = list(store2.iter_events())
        assert len(events) == 2
        store2.close()

    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "events.db")

        with SQLiteEventStore(db_path) as store:
            store.append(_make_market_event())
            assert store.size() == 1

    def test_auto_mkdir(self, tmp_path):
        db_path = str(tmp_path / "nested" / "dir" / "events.db")
        store = SQLiteEventStore(db_path)
        store.append(_make_market_event())
        assert store.size() == 1
        store.close()

    def test_implements_event_store_interface(self, tmp_path):
        db_path = str(tmp_path / "events.db")
        store = SQLiteEventStore(db_path)
        assert isinstance(store, EventStore)
        store.close()


class TestBootstrapWithStore:
    def test_bootstrap_default_inmemory(self):
        runtime, store = bootstrap_event_layer()
        assert isinstance(store, InMemoryEventStore)

    def test_bootstrap_with_custom_store(self, tmp_path):
        db_path = str(tmp_path / "events.db")
        custom_store = SQLiteEventStore(db_path)
        runtime, store = bootstrap_event_layer(store=custom_store)
        assert store is custom_store
        custom_store.close()
