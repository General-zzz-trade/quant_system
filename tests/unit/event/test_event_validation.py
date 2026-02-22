"""Tests for event validation."""
from __future__ import annotations

from event.types import EventType


def test_event_type_values():
    assert EventType.MARKET.value is not None
    assert EventType.FILL.value is not None
    assert EventType.ORDER.value is not None
