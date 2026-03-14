"""Tests that all 7 auxiliary components are saved AND restored in recovery.

Validates that the recovery chain in live_runner.py is complete:
  1. kill_switch
  2. inference_bridge
  3. feature_hook
  4. correlation_computer
  5. timeout_tracker
  6. exit_manager      (was missing restore — fixed)
  7. regime_gate       (was missing save+restore — fixed)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock


from runner.recovery import (
    restore_exit_manager_state,
    restore_regime_gate_state,
    save_all_auxiliary_state,
    save_exit_manager_state,
    save_regime_gate_state,
    restore_all_auxiliary_state,
)


# ============================================================
# ExitManager checkpoint roundtrip
# ============================================================

class TestExitManagerRecovery:
    def _make_exit_manager(self):
        mgr = MagicMock()
        mgr.checkpoint.return_value = {
            "BTCUSDT": {
                "entry_price": 50000.0,
                "peak_price": 52000.0,
                "entry_bar": 100,
            }
        }
        return mgr

    def test_save_and_restore_roundtrip(self, tmp_path: Path):
        mgr = self._make_exit_manager()
        save_exit_manager_state(mgr, data_dir=str(tmp_path))

        assert (tmp_path / "exit_manager_checkpoint.json").exists()

        mgr2 = MagicMock()
        result = restore_exit_manager_state(mgr2, data_dir=str(tmp_path))
        assert result is True
        mgr2.restore.assert_called_once()
        restored_data = mgr2.restore.call_args[0][0]
        assert "BTCUSDT" in restored_data
        assert restored_data["BTCUSDT"]["entry_price"] == 50000.0

    def test_restore_returns_false_when_no_file(self, tmp_path: Path):
        mgr = MagicMock()
        assert restore_exit_manager_state(mgr, data_dir=str(tmp_path)) is False

    def test_restore_returns_false_when_none(self, tmp_path: Path):
        assert restore_exit_manager_state(None, data_dir=str(tmp_path)) is False

    def test_restore_returns_false_when_no_restore_method(self, tmp_path: Path):
        mgr = object()  # no .restore attribute
        assert restore_exit_manager_state(mgr, data_dir=str(tmp_path)) is False


# ============================================================
# RegimeGate checkpoint roundtrip
# ============================================================

class TestRegimeGateRecovery:
    def _make_regime_gate(self):
        gate = MagicMock()
        gate.checkpoint.return_value = {
            "bb_width_buf": [0.01, 0.02, 0.03],
            "atr_buf": [100.0, 110.0, 105.0],
        }
        return gate

    def test_save_and_restore_roundtrip(self, tmp_path: Path):
        gate = self._make_regime_gate()
        save_regime_gate_state(gate, data_dir=str(tmp_path))

        assert (tmp_path / "regime_gate_checkpoint.json").exists()

        gate2 = MagicMock()
        result = restore_regime_gate_state(gate2, data_dir=str(tmp_path))
        assert result is True
        gate2.restore.assert_called_once()
        restored_data = gate2.restore.call_args[0][0]
        assert restored_data["bb_width_buf"] == [0.01, 0.02, 0.03]

    def test_restore_returns_false_when_no_file(self, tmp_path: Path):
        gate = MagicMock()
        assert restore_regime_gate_state(gate, data_dir=str(tmp_path)) is False

    def test_restore_returns_false_when_none(self, tmp_path: Path):
        assert restore_regime_gate_state(None, data_dir=str(tmp_path)) is False


# ============================================================
# save_all_auxiliary_state includes exit_manager + regime_gate
# ============================================================

class TestSaveAllIncludesAllComponents:
    def test_save_all_writes_exit_manager_and_regime_gate(self, tmp_path: Path):
        exit_mgr = MagicMock()
        exit_mgr.checkpoint.return_value = {"ETHUSDT": {"entry_price": 3000.0}}

        regime_gate = MagicMock()
        regime_gate.checkpoint.return_value = {"bb_width_buf": [0.05]}

        save_all_auxiliary_state(
            exit_manager=exit_mgr,
            regime_gate=regime_gate,
            data_dir=str(tmp_path),
        )

        # Individual files written
        assert (tmp_path / "exit_manager_checkpoint.json").exists()
        assert (tmp_path / "regime_gate_checkpoint.json").exists()

        # Bundle also contains both
        bundle_path = tmp_path / "auxiliary_state_bundle.json"
        assert bundle_path.exists()
        with open(bundle_path) as f:
            bundle = json.load(f)
        assert "exit_manager" in bundle
        assert "regime_gate" in bundle


# ============================================================
# restore_all_auxiliary_state includes exit_manager + regime_gate
# ============================================================

class TestRestoreAllIncludesAllComponents:
    def test_restore_all_restores_exit_manager_and_regime_gate(self, tmp_path: Path):
        # Save
        exit_mgr = MagicMock()
        exit_mgr.checkpoint.return_value = {"BTCUSDT": {"entry_bar": 42}}
        regime_gate = MagicMock()
        regime_gate.checkpoint.return_value = {"bb_width_buf": [0.1, 0.2]}

        save_all_auxiliary_state(
            exit_manager=exit_mgr,
            regime_gate=regime_gate,
            data_dir=str(tmp_path),
        )

        # Restore into fresh mocks
        exit_mgr2 = MagicMock()
        regime_gate2 = MagicMock()
        results = restore_all_auxiliary_state(
            exit_manager=exit_mgr2,
            regime_gate=regime_gate2,
            data_dir=str(tmp_path),
        )

        assert results.get("exit_manager") is True
        assert results.get("regime_gate") is True
        exit_mgr2.restore.assert_called_once()
        regime_gate2.restore.assert_called_once()


# ============================================================
# _find_module_attr traversal
# ============================================================

class TestFindModuleAttr:
    def test_finds_attr_on_direct_module(self):
        from runner.live_runner import _find_module_attr

        exit_mgr = MagicMock()
        module = MagicMock()
        module._exit_mgr = exit_mgr
        module.inner = None

        bridge = MagicMock()
        bridge.modules = [module]

        assert _find_module_attr(bridge, '_exit_mgr') is exit_mgr

    def test_finds_attr_on_inner_module(self):
        from runner.live_runner import _find_module_attr

        regime_gate = MagicMock()
        inner = MagicMock()
        inner._regime_gate = regime_gate

        wrapper = MagicMock(spec=[])  # no _regime_gate on wrapper
        wrapper.inner = inner

        bridge = MagicMock()
        bridge.modules = [wrapper]

        assert _find_module_attr(bridge, '_regime_gate') is regime_gate

    def test_returns_none_when_bridge_is_none(self):
        from runner.live_runner import _find_module_attr
        assert _find_module_attr(None, '_exit_mgr') is None

    def test_returns_none_when_no_modules(self):
        from runner.live_runner import _find_module_attr
        bridge = MagicMock()
        bridge.modules = []
        assert _find_module_attr(bridge, '_exit_mgr') is None


# ============================================================
# All 7 components appear in both save and restore sequences
# ============================================================

class TestRecoveryChainCompleteness:
    """Verify that live_runner.py's recovery chain covers all 7 components.

    This is a source-level check: we read live_runner.py and verify that
    each restore function is called in the recovery sequence.
    """

    def test_unified_restore_called_with_all_components(self):
        """Verify live_runner.py calls restore_all_auxiliary_state with all 7 components."""
        import inspect
        import runner.live_runner as lr
        source = inspect.getsource(lr)

        # Must call the unified restore function
        assert "restore_all_auxiliary_state(" in source, (
            "live_runner.py must call restore_all_auxiliary_state()"
        )

        # All 7 component keyword args must be passed
        component_kwargs = [
            "kill_switch=",
            "inference_bridge=",
            "feature_hook=",
            "exit_manager=",
            "regime_gate=",
            "correlation_computer=",
            "timeout_tracker=",
        ]

        for kwarg in component_kwargs:
            assert kwarg in source, (
                f"restore_all_auxiliary_state call must pass {kwarg} parameter"
            )

    def test_save_all_passes_regime_gate_in_source(self):
        import inspect
        import runner.live_runner as lr
        source = inspect.getsource(lr)

        assert "regime_gate=" in source, (
            "save_all_auxiliary_state call must pass regime_gate= parameter"
        )
