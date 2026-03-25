"""Tests for engine/coordinator_handlers.py — event routing and dispatch."""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from engine.coordinator_handlers import (
    handle_pipeline_event,
    handle_execution_event,
    handle_market_tick_fast,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coord(**overrides):
    """Build a minimal coordinator-like namespace with mock attributes."""
    coord = SimpleNamespace(
        _tick_processors=None,
        _pipeline=MagicMock(),
        _decision_bridge=MagicMock(),
        _execution_bridge=MagicMock(),
        _feature_hook=None,
        _last_snapshot=None,
        _lock=MagicMock(),
        _cfg=SimpleNamespace(
            symbol_default="BTCUSDT",
            feature_hook=None,
            on_pipeline_output=None,
            on_snapshot=None,
            emit_on_non_advanced=False,
        ),
    )
    for k, v in overrides.items():
        setattr(coord, k, v)
    return coord


def _make_event(symbol="BTCUSDT", event_type="MARKET", close=50000):
    """Build a mock market event."""
    evt = MagicMock()
    evt.symbol = symbol
    evt.event_type = event_type
    evt.close = close
    evt.volume = 100
    evt.high = close + 100
    evt.low = close - 100
    evt.open = close
    evt.ts = datetime(2025, 6, 1, 12, 0, 0)
    evt.header = SimpleNamespace(event_id="evt-001", ts=evt.ts)
    return evt


# ---------------------------------------------------------------------------
# handle_execution_event
# ---------------------------------------------------------------------------

class TestHandleExecutionEvent:
    def test_drops_silently_when_bridge_is_none(self):
        """When execution bridge is detached (warmup), events are silently dropped."""
        coord = _make_coord(_execution_bridge=None)
        event = _make_event()
        # Should not raise
        handle_execution_event(coord, event)

    def test_routes_to_bridge_when_present(self):
        bridge = MagicMock()
        coord = _make_coord(_execution_bridge=bridge)
        event = _make_event()
        handle_execution_event(coord, event)
        bridge.handle_event.assert_called_once_with(event)

    def test_bridge_receives_exact_event_object(self):
        bridge = MagicMock()
        coord = _make_coord(_execution_bridge=bridge)
        event = _make_event()
        handle_execution_event(coord, event)
        assert bridge.handle_event.call_args[0][0] is event


# ---------------------------------------------------------------------------
# handle_pipeline_event — slow path (no tick_processors)
# ---------------------------------------------------------------------------

class TestHandlePipelineEventSlowPath:
    def test_routes_to_pipeline(self):
        pipeline = MagicMock()
        out = MagicMock()
        out.advanced = True
        out.snapshot = MagicMock()
        pipeline.apply.return_value = out
        coord = _make_coord(
            _tick_processors=None,
            _pipeline=pipeline,
        )
        event = _make_event()
        handle_pipeline_event(coord, event)
        pipeline.apply.assert_called_once()

    @patch("engine.coordinator_handlers._detect_kind", return_value="MARKET")
    def test_triggers_decision_bridge_on_market_advance(self, mock_kind):
        pipeline = MagicMock()
        out = MagicMock()
        out.advanced = True
        out.snapshot = MagicMock()
        pipeline.apply.return_value = out
        decision_bridge = MagicMock()
        coord = _make_coord(
            _tick_processors=None,
            _pipeline=pipeline,
            _decision_bridge=decision_bridge,
        )
        event = _make_event()
        handle_pipeline_event(coord, event)
        decision_bridge.on_pipeline_output.assert_called_once_with(out)

    @patch("engine.coordinator_handlers._detect_kind", return_value="MARKET")
    def test_no_decision_when_not_advanced(self, mock_kind):
        pipeline = MagicMock()
        out = MagicMock()
        out.advanced = False
        out.snapshot = None
        pipeline.apply.return_value = out
        decision_bridge = MagicMock()
        coord = _make_coord(
            _tick_processors=None,
            _pipeline=pipeline,
            _decision_bridge=decision_bridge,
        )
        event = _make_event()
        handle_pipeline_event(coord, event)
        decision_bridge.on_pipeline_output.assert_not_called()

    @patch("engine.coordinator_handlers._detect_kind", return_value="MARKET")
    def test_no_decision_when_bridge_is_none(self, mock_kind):
        pipeline = MagicMock()
        out = MagicMock()
        out.advanced = True
        out.snapshot = MagicMock()
        pipeline.apply.return_value = out
        coord = _make_coord(
            _tick_processors=None,
            _pipeline=pipeline,
            _decision_bridge=None,
        )
        event = _make_event()
        # Should not raise even without decision bridge
        handle_pipeline_event(coord, event)


# ---------------------------------------------------------------------------
# handle_pipeline_event — fast path (tick_processors present)
# ---------------------------------------------------------------------------

class TestHandlePipelineEventFastPath:
    @patch("engine.coordinator_handlers._detect_kind", return_value="MARKET")
    def test_routes_market_to_tick_fast(self, mock_kind):
        tp = MagicMock()
        coord = _make_coord(
            _tick_processors={"BTCUSDT": tp},
        )
        event = _make_event(symbol="BTCUSDT")

        with patch("engine.coordinator_handlers.handle_market_tick_fast") as mock_fast:
            handle_pipeline_event(coord, event)
            mock_fast.assert_called_once_with(coord, event, tp)

    @patch("engine.coordinator_handlers._detect_kind", return_value="FILL")
    def test_routes_fill_to_process_fill(self, mock_kind):
        tp = MagicMock()
        coord = _make_coord(
            _tick_processors={"BTCUSDT": tp},
        )
        event = _make_event(symbol="BTCUSDT")
        handle_pipeline_event(coord, event)
        tp.process_fill.assert_called_once_with(event)

    @patch("engine.coordinator_handlers._detect_kind", return_value="FUNDING")
    def test_routes_funding_to_process_funding(self, mock_kind):
        tp = MagicMock()
        coord = _make_coord(
            _tick_processors={"BTCUSDT": tp},
        )
        event = _make_event(symbol="BTCUSDT")
        handle_pipeline_event(coord, event)
        tp.process_funding.assert_called_once_with(event)


# ---------------------------------------------------------------------------
# handle_market_tick_fast
# ---------------------------------------------------------------------------

class TestHandleMarketTickFast:
    def test_calls_process_tick_full(self):
        tp = MagicMock()
        result = MagicMock()
        result.features_dict = {"rsi_14": 55.0}
        result.event_index = 1
        result.markets = {"BTCUSDT": MagicMock()}
        result.account = MagicMock()
        result.positions = {"BTCUSDT": MagicMock()}
        result.portfolio = MagicMock()
        result.risk = MagicMock()
        result.last_event_id = "evt-002"
        result.last_ts = None
        tp.process_tick_full.return_value = result

        coord = _make_coord(
            _decision_bridge=None,
            _feature_hook=None,
        )
        event = _make_event()
        handle_market_tick_fast(coord, event, tp)
        tp.process_tick_full.assert_called_once()

    def test_sets_last_snapshot(self):
        tp = MagicMock()
        result = MagicMock()
        result.features_dict = {"rsi_14": 55.0}
        result.event_index = 1
        result.markets = {"BTCUSDT": MagicMock()}
        result.account = MagicMock()
        result.positions = {"BTCUSDT": MagicMock()}
        result.portfolio = MagicMock()
        result.risk = MagicMock()
        result.last_event_id = None
        result.last_ts = None
        tp.process_tick_full.return_value = result

        coord = _make_coord(
            _decision_bridge=None,
            _feature_hook=None,
        )
        assert coord._last_snapshot is None
        handle_market_tick_fast(coord, _make_event(), tp)
        assert coord._last_snapshot is not None

    def test_triggers_decision_bridge(self):
        tp = MagicMock()
        result = MagicMock()
        result.features_dict = {}
        result.event_index = 1
        result.markets = {}
        result.account = MagicMock()
        result.positions = {}
        result.portfolio = MagicMock()
        result.risk = MagicMock()
        result.last_event_id = None
        result.last_ts = None
        tp.process_tick_full.return_value = result

        decision_bridge = MagicMock()
        coord = _make_coord(
            _decision_bridge=decision_bridge,
            _feature_hook=None,
        )
        handle_market_tick_fast(coord, _make_event(), tp)
        decision_bridge.on_pipeline_output.assert_called_once()
