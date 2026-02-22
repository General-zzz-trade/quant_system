"""Event contract tests — verify event schema stability."""
from __future__ import annotations

from event.types import EventType


def test_event_type_enum_stable():
    assert EventType.MARKET.value == "market"
    assert EventType.FILL.value == "fill"
    assert EventType.ORDER.value == "order"
    assert EventType.SIGNAL.value == "signal"


def test_event_type_all_values():
    expected = {"market", "signal", "intent", "order", "fill", "risk", "control"}
    actual = {e.value for e in EventType}
    assert expected == actual
