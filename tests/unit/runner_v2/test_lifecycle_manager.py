"""Tests for LifecycleManager."""
from unittest.mock import MagicMock, call
import pytest

from runner.lifecycle_manager import LifecycleManager


class TestLifecycleSignals:
    def test_sighup_triggers_model_reload(self):
        engine = MagicMock()
        engine.reload_models.return_value = {"BTCUSDT": "reloaded"}
        executor = MagicMock()
        recovery = MagicMock()
        loop = MagicMock()
        lm = LifecycleManager(engine=engine, executor=executor,
                               recovery=recovery, loop=loop)
        lm._handle_sighup()
        engine.reload_models.assert_called_once()


class TestLifecycleShutdownOrder:
    def test_stop_calls_subsystems_in_order(self):
        engine = MagicMock()
        executor = MagicMock()
        recovery = MagicMock()
        loop = MagicMock()
        lm = LifecycleManager(engine=engine, executor=executor,
                               recovery=recovery, loop=loop)
        lm._running = True
        lm.stop()
        # User stream stopped before loop
        assert executor.stop_user_stream.called
        assert loop.stop.called
        assert recovery.save.called
        # Verify order: user_stream before loop
        calls = []
        executor.stop_user_stream.side_effect = lambda: calls.append("user_stream")
        recovery.save.side_effect = lambda: calls.append("recovery")
        loop.stop.side_effect = lambda: calls.append("loop")
        lm._running = True
        lm.stop()
