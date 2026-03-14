"""Tests for LiveEmitHandler — extracted from LiveRunner._emit closure."""
from __future__ import annotations

import sys
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/quant_system")

from runner.emit_handler import LiveEmitHandler


def _make_handler(**overrides) -> tuple[LiveEmitHandler, dict]:
    """Create LiveEmitHandler with mock dependencies."""
    mocks = {
        "coordinator": MagicMock(),
        "attribution_tracker": MagicMock(),
        "gate_chain": MagicMock(),
        "order_state_machine": MagicMock(),
        "timeout_tracker": MagicMock(),
        "event_recorder": MagicMock(),
        "live_signal_tracker": MagicMock(),
    }
    mocks.update(overrides)
    handler = LiveEmitHandler(**mocks)
    return handler, mocks


def _order_event(symbol="BTCUSDT", side="BUY", qty="0.01", price="50000"):
    ev = MagicMock()
    ev.event_type = MagicMock()
    ev.event_type.value = "ORDER"
    ev.symbol = symbol
    ev.side = side
    ev.qty = qty
    ev.price = price
    ev.order_id = "ord-123"
    ev.client_order_id = "cli-123"
    ev.order_type = "LIMIT"
    ev.quantity = qty
    return ev


def _fill_event(symbol="BTCUSDT", order_id="ord-123", qty="0.01", price="50000"):
    ev = MagicMock()
    ev.event_type = MagicMock()
    ev.event_type.value = "FILL"
    ev.symbol = symbol
    ev.order_id = order_id
    ev.qty = qty
    ev.price = price
    return ev


class TestOrderPath:
    def test_order_passes_gate_chain(self):
        handler, mocks = _make_handler()
        ev = _order_event()
        mocks["gate_chain"].process_with_audit.return_value = (ev, [])  # gate passes

        handler(ev)

        mocks["gate_chain"].process_with_audit.assert_called_once_with(ev, {})
        mocks["order_state_machine"].register.assert_called_once()
        mocks["timeout_tracker"].on_submit.assert_called_once()
        mocks["coordinator"].emit.assert_called_once()

    def test_order_rejected_by_gate(self):
        handler, mocks = _make_handler()
        ev = _order_event()
        from runner.gate_chain import GateResult
        mocks["gate_chain"].process_with_audit.return_value = (None, [("TestGate", GateResult(allowed=False, reason="blocked"))])

        handler(ev)

        mocks["gate_chain"].process_with_audit.assert_called_once()
        mocks["order_state_machine"].register.assert_not_called()
        mocks["timeout_tracker"].on_submit.assert_not_called()
        mocks["coordinator"].emit.assert_not_called()

    def test_order_osm_register_failure_doesnt_block(self):
        handler, mocks = _make_handler()
        ev = _order_event()
        mocks["gate_chain"].process_with_audit.return_value = (ev, [])
        mocks["order_state_machine"].register.side_effect = Exception("OSM error")

        handler(ev)

        # Should still emit to coordinator despite OSM failure
        mocks["coordinator"].emit.assert_called_once()

    def test_osm_register_failure_skips_timeout(self):
        """If OSM register fails, timeout_tracker should NOT be called."""
        handler, mocks = _make_handler()
        ev = _order_event()
        mocks["gate_chain"].process_with_audit.return_value = (ev, [])
        mocks["order_state_machine"].register.side_effect = Exception("OSM error")

        handler(ev)

        mocks["timeout_tracker"].on_submit.assert_not_called()


class TestFillPath:
    def test_fill_transitions_osm(self):
        handler, mocks = _make_handler()
        ev = _fill_event()

        handler(ev)

        mocks["timeout_tracker"].on_fill.assert_called_once_with("ord-123")
        mocks["order_state_machine"].transition.assert_called_once()
        mocks["coordinator"].emit.assert_called_once()

    def test_fill_records_to_event_recorder(self):
        handler, mocks = _make_handler()
        ev = _fill_event()

        handler(ev)

        mocks["event_recorder"].record_fill.assert_called_once_with(ev)

    def test_fill_tracks_signal(self):
        handler, mocks = _make_handler()
        ev = _fill_event(symbol="ETHUSDT")

        handler(ev)

        mocks["live_signal_tracker"].on_fill.assert_called_once_with(ev, origin="ETHUSDT")

    def test_fill_without_recorder(self):
        handler, mocks = _make_handler(event_recorder=None)
        ev = _fill_event()

        handler(ev)  # Should not raise

        mocks["coordinator"].emit.assert_called_once()

    def test_fill_without_signal_tracker(self):
        handler, mocks = _make_handler(live_signal_tracker=None)
        ev = _fill_event()

        handler(ev)  # Should not raise

        mocks["coordinator"].emit.assert_called_once()

    def test_partial_fill_uses_partially_filled_status(self):
        """Fill events with is_partial=True should transition to PARTIALLY_FILLED."""
        handler, mocks = _make_handler()
        ev = _fill_event()
        ev.is_partial = True

        handler(ev)

        from execution.state_machine.transitions import OrderStatus
        call_kwargs = mocks["order_state_machine"].transition.call_args
        assert call_kwargs.kwargs["new_status"] == OrderStatus.PARTIALLY_FILLED

    def test_partial_fill_via_status_field(self):
        """Fill events with status containing 'partial' should use PARTIALLY_FILLED."""
        handler, mocks = _make_handler()
        ev = _fill_event()
        ev.is_partial = None
        ev.status = "PARTIALLY_FILLED"

        handler(ev)

        from execution.state_machine.transitions import OrderStatus
        call_kwargs = mocks["order_state_machine"].transition.call_args
        assert call_kwargs.kwargs["new_status"] == OrderStatus.PARTIALLY_FILLED

    def test_full_fill_uses_filled_status(self):
        """Fill events without partial indicators should use FILLED status."""
        handler, mocks = _make_handler()
        ev = _fill_event()

        handler(ev)

        from execution.state_machine.transitions import OrderStatus
        call_kwargs = mocks["order_state_machine"].transition.call_args
        assert call_kwargs.kwargs["new_status"] == OrderStatus.FILLED


class TestAttribution:
    def test_accepted_order_and_fill_tracked(self):
        handler, mocks = _make_handler()

        order_ev = _order_event()
        mocks["gate_chain"].process_with_audit.return_value = (order_ev, [])
        handler(order_ev)

        fill_ev = _fill_event()
        handler(fill_ev)

        # Both events should be tracked by attribution
        assert mocks["attribution_tracker"].on_event.call_count == 2

    def test_rejected_order_not_attributed(self):
        """Orders rejected by gate chain should NOT be tracked by attribution."""
        handler, mocks = _make_handler()
        ev = _order_event()
        from runner.gate_chain import GateResult
        mocks["gate_chain"].process_with_audit.return_value = (None, [("TestGate", GateResult(allowed=False, reason="blocked"))])

        handler(ev)

        # Attribution should NOT have been called for rejected order
        mocks["attribution_tracker"].on_event.assert_not_called()

    def test_accepted_order_is_attributed(self):
        """Orders passing gate chain should be tracked by attribution."""
        handler, mocks = _make_handler()
        ev = _order_event()
        mocks["gate_chain"].process_with_audit.return_value = (ev, [])

        handler(ev)

        mocks["attribution_tracker"].on_event.assert_called_once_with(ev)

    def test_fill_event_attributed(self):
        """Fill events should still be attributed."""
        handler, mocks = _make_handler()
        ev = _fill_event()

        handler(ev)

        mocks["attribution_tracker"].on_event.assert_called_once_with(ev)

    def test_unknown_event_type_still_emitted(self):
        handler, mocks = _make_handler()
        ev = MagicMock()
        ev.event_type = MagicMock()
        ev.event_type.value = "MARKET"

        handler(ev)

        mocks["attribution_tracker"].on_event.assert_called_once_with(ev)
        mocks["coordinator"].emit.assert_called_once()
