"""Tests for RecoveryManager."""
from unittest.mock import MagicMock

from runner.recovery_manager import RecoveryManager


class TestRecoveryManagerSave:
    def test_save_checkpoints_engine_and_risk(self):
        engine = MagicMock()
        engine.checkpoint.return_value = {"features": {}}
        risk = MagicMock()
        risk.checkpoint.return_value = {"kill_switch": {}}
        orders = MagicMock()
        rm = RecoveryManager(state_dir="/tmp/test_recovery_state",
                             engine=engine, risk=risk, orders=orders)
        rm.save()
        engine.checkpoint.assert_called_once()
        risk.checkpoint.assert_called_once()


class TestRecoveryManagerRestore:
    def test_restore_returns_false_when_no_state(self):
        engine = MagicMock()
        risk = MagicMock()
        orders = MagicMock()
        rm = RecoveryManager(state_dir="/tmp/nonexistent_recovery_dir_xyz",
                             engine=engine, risk=risk, orders=orders)
        assert rm.restore() is False


class TestRecoveryManagerReconcile:
    def test_reconcile_returns_list(self):
        engine = MagicMock()
        risk = MagicMock()
        orders = MagicMock()
        rm = RecoveryManager(state_dir="/tmp/test_recovery_state",
                             engine=engine, risk=risk, orders=orders)
        executor = MagicMock()
        executor.get_positions.return_value = []
        result = rm.reconcile_startup(executor)
        assert isinstance(result, list)
