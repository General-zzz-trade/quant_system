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
        """Record an IntentEvent (or dict-like)."""
        self._intents.append(_extract_intent(event))

    def on_order(self, event: Any) -> None:
        """Record an OrderEvent (or dict-like)."""
        self._orders.append(_extract_order(event))

    def on_fill(self, event: Any) -> None:
        """Record a FillEvent (or dict-like)."""
        self._fills.append(_extract_fill(event))

    def on_event(self, event: Any) -> None:
        """Route any event to the correct handler by type name or event_type."""
        type_name = type(event).__name__
        if type_name == "IntentEvent" or _get_event_type(event) == "intent":
            self.on_intent(event)
        elif type_name == "OrderEvent" or _get_event_type(event) == "order":
            self.on_order(event)
        elif type_name == "FillEvent" or _get_event_type(event) == "fill":
            self.on_fill(event)

    def report(
        self, current_prices: Optional[Mapping[str, float]] = None,
    ) -> SignalAttributionReport:
        """Generate signal attribution report from accumulated events."""
        return attribute_by_signal(
            self._intents, self._orders, self._fills, current_prices,
        )

    @property
    def intent_count(self) -> int:
        return len(self._intents)

    @property
    def order_count(self) -> int:
        return len(self._orders)

    @property
    def fill_count(self) -> int:
        return len(self._fills)


def _get_event_type(event: Any) -> str:
    """Extract event type string from event object."""
    et = getattr(event, "event_type", None)
    if et is not None:
        return str(et.value) if hasattr(et, "value") else str(et)
    return ""


def _extract_intent(event: Any) -> Dict[str, object]:
    """Extract intent fields from an event object or dict."""
    if isinstance(event, dict):
        return event
    return {
        "intent_id": str(getattr(event, "intent_id", "")),
        "symbol": str(getattr(event, "symbol", "")),
        "side": str(getattr(event, "side", "")),
        "origin": str(getattr(event, "origin", "")),
    }


def _extract_order(event: Any) -> Dict[str, object]:
    """Extract order fields from an event object or dict."""
    if isinstance(event, dict):
        return event
    return {
        "order_id": str(getattr(event, "order_id", "")),
        "intent_id": str(getattr(event, "intent_id", "")),
        "symbol": str(getattr(event, "symbol", "")),
        "side": str(getattr(event, "side", "")),
    }


def _extract_fill(event: Any) -> Dict[str, object]:
    """Extract fill fields from an event object or dict."""
    if isinstance(event, dict):
        return event
    return {
        "fill_id": str(getattr(event, "fill_id", "")),
        "order_id": str(getattr(event, "order_id", "")),
        "symbol": str(getattr(event, "symbol", "")),
        "side": str(getattr(event, "side", "")),
        "qty": str(getattr(event, "qty", "0")),
        "price": str(getattr(event, "price", "0")),
        "fee": str(getattr(event, "fee", "0")),
    }
