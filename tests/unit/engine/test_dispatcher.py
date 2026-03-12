# tests/unit/engine/test_dispatcher.py
"""EventDispatcher unit tests — routing rules, dedup, sequence tracking."""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from typing import Any, List

from engine.dispatcher import (
    DispatcherError,
    DuplicateEventError,
    EventDispatcher,
    Route,
)
from _quant_hotpath import rust_route_event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event(event_type: str, event_id: str = None) -> SimpleNamespace:
    header = SimpleNamespace(event_id=event_id, ts=None) if event_id else SimpleNamespace(ts=None)
    return SimpleNamespace(event_type=event_type, header=header)


def _event_name(name: str, event_id: str = None) -> SimpleNamespace:
    """Event using EVENT_TYPE string style (no event_type attr)."""
    header = SimpleNamespace(event_id=event_id, ts=None) if event_id else SimpleNamespace(ts=None)
    ns = SimpleNamespace(EVENT_TYPE=name, header=header)
    # Remove event_type attr that SimpleNamespace wouldn't normally have
    return ns


class _Collector:
    """Collects dispatched events for assertions."""
    def __init__(self) -> None:
        self.events: List[Any] = []

    def __call__(self, event: Any) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Tests: routing rules (EventType enum style)
# ---------------------------------------------------------------------------

class TestRoutingRules:
    def test_market_to_pipeline(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.PIPELINE, handler=c)
        d.dispatch(event=_event("market"))
        assert len(c.events) == 1

    def test_fill_to_pipeline(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.PIPELINE, handler=c)
        d.dispatch(event=_event("fill"))
        assert len(c.events) == 1

    def test_order_to_execution(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.EXECUTION, handler=c)
        d.dispatch(event=_event("order"))
        assert len(c.events) == 1

    def test_order_update_to_pipeline(self) -> None:
        """ORDER_UPDATE is a fact event → PIPELINE (not EXECUTION)."""
        d = EventDispatcher()
        c_pipe = _Collector()
        c_exec = _Collector()
        d.register(route=Route.PIPELINE, handler=c_pipe)
        d.register(route=Route.EXECUTION, handler=c_exec)
        d.dispatch(event=_event("order_update"))
        assert len(c_pipe.events) == 1
        assert len(c_exec.events) == 0

    def test_signal_to_decision(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.DECISION, handler=c)
        d.dispatch(event=_event("signal"))
        assert len(c.events) == 1

    def test_intent_to_decision(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.DECISION, handler=c)
        d.dispatch(event=_event("intent"))
        assert len(c.events) == 1

    def test_risk_to_decision(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.DECISION, handler=c)
        d.dispatch(event=_event("risk"))
        assert len(c.events) == 1

    def test_unknown_event_dropped(self) -> None:
        d = EventDispatcher()
        c_pipe = _Collector()
        c_dec = _Collector()
        c_exec = _Collector()
        d.register(route=Route.PIPELINE, handler=c_pipe)
        d.register(route=Route.DECISION, handler=c_dec)
        d.register(route=Route.EXECUTION, handler=c_exec)
        d.dispatch(event=_event("some_unknown_type"))
        assert len(c_pipe.events) == 0
        assert len(c_dec.events) == 0
        assert len(c_exec.events) == 0


# ---------------------------------------------------------------------------
# Tests: EVENT_TYPE string style routing
# ---------------------------------------------------------------------------

class TestStringStyleRouting:
    def test_market_bar_to_pipeline(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.PIPELINE, handler=c)
        e = SimpleNamespace(EVENT_TYPE="market_bar", header=SimpleNamespace(ts=None))
        d.dispatch(event=e)
        assert len(c.events) == 1

    def test_trade_fill_to_pipeline(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.PIPELINE, handler=c)
        e = SimpleNamespace(EVENT_TYPE="trade_fill", header=SimpleNamespace(ts=None))
        d.dispatch(event=e)
        assert len(c.events) == 1

    def test_order_report_to_pipeline(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.PIPELINE, handler=c)
        e = SimpleNamespace(EVENT_TYPE="order_report", header=SimpleNamespace(ts=None))
        d.dispatch(event=e)
        assert len(c.events) == 1

    def test_header_event_type_routes_when_event_attr_missing(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.PIPELINE, handler=c)
        e = SimpleNamespace(header=SimpleNamespace(event_type="market", ts=None))
        d.dispatch(event=e)
        assert len(c.events) == 1


class TestRustParity:
    def test_route_event_parity_for_event_object_shapes(self) -> None:
        d = EventDispatcher()
        cases = [
            _event("market"),
            _event("fill"),
            _event("signal"),
            _event("intent"),
            _event("order"),
            _event_name("order_report"),
            SimpleNamespace(header=SimpleNamespace(event_type="risk", ts=None)),
            SimpleNamespace(header=SimpleNamespace(event_type="order_update", ts=None)),
            SimpleNamespace(EVENT_TYPE="some_unknown_type", header=SimpleNamespace(ts=None)),
        ]

        route_map = {
            "pipeline": Route.PIPELINE,
            "decision": Route.DECISION,
            "execution": Route.EXECUTION,
            "drop": Route.DROP,
        }

        for event in cases:
            rust_route = route_map[rust_route_event(event)]
            py_route = d._route_for(event)
            assert rust_route == py_route

    def test_route_from_label_matches_route_for(self) -> None:
        d = EventDispatcher()
        event = SimpleNamespace(EVENT_TYPE="trade_fill", header=SimpleNamespace(ts=None))
        assert d._route_for(event) == EventDispatcher._route_from_name("trade_fill")


# ---------------------------------------------------------------------------
# Tests: deduplication
# ---------------------------------------------------------------------------

class TestDedup:
    def test_duplicate_event_id_raises(self) -> None:
        d = EventDispatcher()
        d.dispatch(event=_event("market", event_id="evt-1"))
        with pytest.raises(DuplicateEventError, match="evt-1"):
            d.dispatch(event=_event("market", event_id="evt-1"))

    def test_different_event_ids_ok(self) -> None:
        d = EventDispatcher()
        d.dispatch(event=_event("market", event_id="evt-1"))
        d.dispatch(event=_event("market", event_id="evt-2"))  # should not raise

    def test_no_event_id_no_dedup(self) -> None:
        """Events without event_id are never deduplicated."""
        d = EventDispatcher()
        d.dispatch(event=_event("market"))
        d.dispatch(event=_event("market"))  # should not raise


# ---------------------------------------------------------------------------
# Tests: sequence tracking
# ---------------------------------------------------------------------------

class TestSequence:
    def test_seq_increments(self) -> None:
        d = EventDispatcher()
        d.dispatch(event=_event("market"))
        d.dispatch(event=_event("fill"))
        assert d._seq == 2


# ---------------------------------------------------------------------------
# Tests: handler registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_multiple_handlers_per_route(self) -> None:
        d = EventDispatcher()
        c1 = _Collector()
        c2 = _Collector()
        d.register(route=Route.PIPELINE, handler=c1)
        d.register(route=Route.PIPELINE, handler=c2)
        d.dispatch(event=_event("market"))
        assert len(c1.events) == 1
        assert len(c2.events) == 1

    def test_handler_receives_event_object(self) -> None:
        d = EventDispatcher()
        c = _Collector()
        d.register(route=Route.PIPELINE, handler=c)
        evt = _event("market")
        d.dispatch(event=evt)
        assert c.events[0] is evt

    def test_handler_exception_propagates(self) -> None:
        d = EventDispatcher()

        def bad_handler(e: Any) -> None:
            raise ValueError("handler failed")

        d.register(route=Route.PIPELINE, handler=bad_handler)
        with pytest.raises(ValueError, match="handler failed"):
            d.dispatch(event=_event("market"))
