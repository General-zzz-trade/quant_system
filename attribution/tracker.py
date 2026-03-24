"""Real-time attribution tracker — accumulates intent/order/fill events."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from attribution.signal_attribution import SignalAttributionReport, attribute_by_signal


class AttributionTracker:
    """Accumulates events and produces signal attribution on demand.

    Usage:
        tracker = AttributionTracker()
        tracker.on_intent(intent_event)
        tracker.on_order(order_event)
        tracker.on_fill(fill_event)
        report = tracker.report()
    """

    def __init__(self) -> None:
        self._intents: List[Dict[str, object]] = []
        self._orders: List[Dict[str, object]] = []
        self._fills: List[Dict[str, object]] = []

    def on_intent(self, event: Any) -> None:
        self._intents.append(_to_dict(event, ("intent_id", "symbol", "side", "origin")))

    def on_order(self, event: Any) -> None:
        self._orders.append(_to_dict(event, ("order_id", "intent_id", "symbol", "side")))

    def on_fill(self, event: Any) -> None:
        self._fills.append(_to_dict(event, ("fill_id", "order_id", "symbol", "side", "qty", "price", "fee")))

    def on_event(self, event: Any) -> None:
        """Route any event to the correct handler by type name or event_type."""
        etype = _event_type(event)
        if etype == "intent":
            self.on_intent(event)
        elif etype == "order":
            self.on_order(event)
        elif etype == "fill":
            self.on_fill(event)

    def report(
        self, current_prices: Optional[Mapping[str, float]] = None,
    ) -> SignalAttributionReport:
        return attribute_by_signal(self._intents, self._orders, self._fills, current_prices)

    @property
    def intent_count(self) -> int:
        return len(self._intents)

    @property
    def order_count(self) -> int:
        return len(self._orders)

    @property
    def fill_count(self) -> int:
        return len(self._fills)


def _event_type(event: Any) -> str:
    """Resolve event type from class name or .event_type attribute."""
    name = type(event).__name__
    for label in ("Intent", "Order", "Fill"):
        if label in name:
            return label.lower()
    et = getattr(event, "event_type", None)
    if et is not None:
        return str(et.value) if hasattr(et, "value") else str(et)
    return ""


def _to_dict(event: Any, fields: tuple[str, ...]) -> Dict[str, object]:
    """Extract named fields from event object; pass-through dicts."""
    if isinstance(event, dict):
        return event
    return {f: str(getattr(event, f, "")) for f in fields}
