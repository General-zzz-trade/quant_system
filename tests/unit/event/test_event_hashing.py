"""Tests for event hashing stability."""
from __future__ import annotations

from event.types import EventType


def test_event_type_values_stable():
    assert EventType.MARKET.value == "market"
    assert EventType.FILL.value == "fill"


def test_event_type_is_enum():
    assert hasattr(EventType, "__members__")
    assert len(EventType.__members__) >= 5
