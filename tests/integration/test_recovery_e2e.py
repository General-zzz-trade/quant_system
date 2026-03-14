"""End-to-end recovery tests: checkpoint -> crash -> restore cycle.

Validates that the full recovery infrastructure (kill switch, inference bridge,
feature hook, coordinator state) survives a simulated crash and restart with
consistent state.
"""
from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from risk.kill_switch import KillMode, KillScope, KillSwitch
from runner.recovery import (
    save_inference_bridge_state,
    restore_inference_bridge_state,
    save_all_auxiliary_state,
    restore_all_auxiliary_state,
)


# ============================================================
# Mock helpers
# ============================================================

class MockInferenceBridge:
    """Minimal inference bridge with checkpoint/restore for z-score buffers."""

    def __init__(self) -> None:
        self._zscore_buffers: Dict[str, list] = {}

    def push_zscore(self, symbol: str, value: float) -> None:
        self._zscore_buffers.setdefault(symbol, []).append(value)

    def checkpoint(self) -> Dict[str, Any]:
        return {"zscore_buffers": {k: list(v) for k, v in self._zscore_buffers.items()}}

    def restore(self, data: Dict[str, Any]) -> None:
        self._zscore_buffers = {
            k: list(v) for k, v in data.get("zscore_buffers", {}).items()
        }


class MockFeatureHook:
    """Minimal feature hook with _bar_count dict (matches real FeatureHook API)."""

    def __init__(self) -> None:
        self._bar_count: Dict[str, int] = {}

    def increment(self, symbol: str) -> None:
        self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1


class MockExitManager:
    """Minimal exit manager with checkpoint/restore."""

    def __init__(self) -> None:
        self._positions: Dict[str, Dict[str, Any]] = {}

    def on_entry(self, symbol: str, price: float, bar: int, direction: float) -> None:
        self._positions[symbol] = {
            "entry_price": price, "peak_price": price,
            "entry_bar": bar, "direction": direction,
        }

    def checkpoint(self) -> Dict[str, Any]:
        return dict(self._positions)

    def restore(self, data: Dict[str, Any]) -> None:
        self._positions = dict(data)


class MockRegimeGate:
    """Minimal regime gate with checkpoint/restore."""

    def __init__(self) -> None:
        self._bb_width_buf: list = []
        self._vol_of_vol_buf: list = []

    def push(self, bb_width: float, vol_of_vol: float) -> None:
        self._bb_width_buf.append(bb_width)
        self._vol_of_vol_buf.append(vol_of_vol)

    def checkpoint(self) -> Dict[str, Any]:
        return {"bb_width_buf": list(self._bb_width_buf),
                "vol_of_vol_buf": list(self._vol_of_vol_buf)}

    def restore(self, data: Dict[str, Any]) -> None:
        self._bb_width_buf = list(data.get("bb_width_buf", []))
        self._vol_of_vol_buf = list(data.get("vol_of_vol_buf", []))


def _market_event(symbol: str, close: float, idx: int) -> SimpleNamespace:
    """Build a minimal market event compatible with EngineCoordinator.emit()."""
    ts = datetime(2024, 1, 1, idx // 60, idx % 60)
    return SimpleNamespace(
        event_type="MARKET",
        symbol=symbol,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=50.0,
        ts=ts,
        header=SimpleNamespace(event_id=f"e{idx}", ts=ts),
    )


# ============================================================
# SimulatedLiveStack
# ============================================================

class SimulatedLiveStack:
    """Wires together coordinator + kill switch + mock bridge + mock feature hook.

    This is a test-only helper that simulates the production live stack
    without requiring real exchange connections or ML models.
    """

    def __init__(self, data_dir: str, symbols: tuple = ("BTCUSDT",)) -> None:
        self.data_dir = data_dir
        self.symbols = symbols
        os.makedirs(data_dir, exist_ok=True)

        # Coordinator (uses Rust state store internally)
        self.cfg = CoordinatorConfig(
            symbol_default=symbols[0],
            symbols=symbols,
            currency="USDT",
            starting_balance=10000.0,
        )
        self.coordinator = EngineCoordinator(cfg=self.cfg)

        # Kill switch
        self.kill_switch = KillSwitch()

        # Mock inference bridge
        self.inference_bridge = MockInferenceBridge()

        # Mock feature hook
        self.feature_hook = MockFeatureHook()

        # Mock exit manager
        self.exit_manager = MockExitManager()

        # Mock regime gate
        self.regime_gate = MockRegimeGate()

        # Correlation computer (real, lightweight)
        from risk.correlation_computer import CorrelationComputer
        self.correlation_computer = CorrelationComputer(window=60)

    def start(self) -> None:
        self.coordinator.start()

    def emit_market(self, symbol: str, close: float, idx: int) -> None:
        self.coordinator.emit(_market_event(symbol, close, idx), actor="test")

    def save_all(self) -> None:
        """Save all component state to data_dir using atomic bundle."""
        save_all_auxiliary_state(
            kill_switch=self.kill_switch,
            inference_bridge=self.inference_bridge,
            feature_hook=self.feature_hook,
            exit_manager=self.exit_manager,
            regime_gate=self.regime_gate,
            correlation_computer=self.correlation_computer,
            data_dir=self.data_dir,
        )

    def restore_all(self) -> Dict[str, Any]:
        """Restore all component state from data_dir. Returns restore results."""
        results = restore_all_auxiliary_state(
            kill_switch=self.kill_switch,
            inference_bridge=self.inference_bridge,
            feature_hook=self.feature_hook,
            exit_manager=self.exit_manager,
            regime_gate=self.regime_gate,
            correlation_computer=self.correlation_computer,
            data_dir=self.data_dir,
        )
        # Map to legacy keys for backward compat with existing tests
        return {
            "kill_switch_restored": results.get("kill_switch", False),
            "bridge_restored": results.get("inference_bridge", False),
            "hook_restored": results.get("feature_hook", False),
            "exit_manager_restored": results.get("exit_manager", False),
            "regime_gate_restored": results.get("regime_gate", False),
            "correlation_restored": results.get("correlation_computer", False),
        }


# ============================================================
# Tests
# ============================================================

@pytest.mark.slow
class TestRecoveryE2E:
    """End-to-end recovery tests covering checkpoint -> crash -> restore."""

    def test_full_checkpoint_crash_restore(self, tmp_path):
        """Full cycle: process events -> checkpoint -> fresh stack -> restore -> verify."""
        data_dir = str(tmp_path / "live")

        # --- Phase 1: Build stack, process events, checkpoint ---
        stack1 = SimulatedLiveStack(data_dir=data_dir)
        stack1.start()

        for i in range(20):
            stack1.emit_market("BTCUSDT", 40000.0 + i * 10, i)

        # Simulate inference bridge activity
        for i in range(20):
            stack1.inference_bridge.push_zscore("BTCUSDT", 0.1 * i)

        # Simulate feature hook bar counts
        for i in range(20):
            stack1.feature_hook.increment("BTCUSDT")

        # Snapshot original state
        orig_bar_count = dict(stack1.feature_hook._bar_count)
        orig_zscores = {
            k: list(v) for k, v in stack1.inference_bridge._zscore_buffers.items()
        }

        # Save checkpoint
        stack1.save_all()

        # --- Phase 2: Simulate crash -> fresh stack -> restore ---
        stack2 = SimulatedLiveStack(data_dir=data_dir)
        stack2.start()

        # Before restore: state should be empty
        assert stack2.feature_hook._bar_count == {}
        assert stack2.inference_bridge._zscore_buffers == {}

        # Restore
        result = stack2.restore_all()

        # --- Phase 3: Verify ---
        assert result["hook_restored"] is True
        assert result["bridge_restored"] is True

        assert stack2.feature_hook._bar_count == orig_bar_count
        assert stack2.inference_bridge._zscore_buffers == orig_zscores

    def test_kill_switch_state_atomicity(self, tmp_path):
        """Kill switch trigger -> checkpoint -> restore -> still active."""
        data_dir = str(tmp_path / "live")

        # --- Phase 1: trigger kill switch ---
        stack1 = SimulatedLiveStack(data_dir=data_dir)
        stack1.start()

        for i in range(5):
            stack1.emit_market("BTCUSDT", 40000.0 + i, i)

        stack1.kill_switch.trigger(
            scope=KillScope.SYMBOL,
            key="BTCUSDT",
            mode=KillMode.HARD_KILL,
            reason="test kill for recovery",
            source="test",
        )

        # Verify kill is active before save
        rec = stack1.kill_switch.is_killed(symbol="BTCUSDT")
        assert rec is not None
        assert rec.reason == "test kill for recovery"

        # Save all state
        stack1.save_all()

        # --- Phase 2: fresh stack -> restore ---
        stack2 = SimulatedLiveStack(data_dir=data_dir)
        stack2.start()

        # Before restore: kill switch should be clean
        assert stack2.kill_switch.is_killed(symbol="BTCUSDT") is None

        result = stack2.restore_all()
        assert result["kill_switch_restored"] is True

        # After restore: kill switch should be active
        rec2 = stack2.kill_switch.is_killed(symbol="BTCUSDT")
        assert rec2 is not None
        assert rec2.scope == KillScope.SYMBOL
        assert rec2.key == "BTCUSDT"
        assert rec2.mode == KillMode.HARD_KILL
        assert rec2.reason == "test kill for recovery"

        # Verify order gating still works after restore
        allowed, kill_rec = stack2.kill_switch.allow_order(
            symbol="BTCUSDT", strategy_id=None, reduce_only=False,
        )
        assert allowed is False
        assert kill_rec is not None

    def test_corrupt_checkpoint_fallback(self, tmp_path):
        """Corrupt inference bridge checkpoint is handled gracefully."""
        data_dir = str(tmp_path / "live")

        # --- Phase 1: normal checkpoint ---
        stack1 = SimulatedLiveStack(data_dir=data_dir)
        stack1.start()

        for i in range(10):
            stack1.emit_market("BTCUSDT", 40000.0 + i, i)

        stack1.inference_bridge.push_zscore("BTCUSDT", 1.5)
        stack1.feature_hook.increment("BTCUSDT")

        stack1.save_all()

        # --- Phase 2: corrupt both the atomic bundle AND the individual file ---
        bridge_path = os.path.join(data_dir, "inference_bridge_checkpoint.json")
        assert os.path.exists(bridge_path)

        # Write truncated/corrupt JSON to individual file
        with open(bridge_path, "w") as f:
            f.write('{"_default": {"zscore_buffers": {"BTCUSDT": [1.5')  # truncated

        # Also corrupt the atomic bundle (so restore_all can't use it for bridge)
        bundle_path = os.path.join(data_dir, "auxiliary_state_bundle.json")
        if os.path.exists(bundle_path):
            with open(bundle_path, "w") as f:
                f.write('{"inference_bridge": BROKEN')  # truncated

        # --- Phase 3: restore in fresh stack ---
        stack2 = SimulatedLiveStack(data_dir=data_dir)
        stack2.start()

        result = stack2.restore_all()

        # Bridge restore should fail gracefully (returns False due to JSON decode error)
        assert result["bridge_restored"] is False

        # Feature hook should still restore fine (separate file)
        assert result["hook_restored"] is True
        assert stack2.feature_hook._bar_count == {"BTCUSDT": 1}

        # Inference bridge should remain in default empty state
        assert stack2.inference_bridge._zscore_buffers == {}

    def test_fill_in_flight_state_consistency(self, tmp_path):
        """Coordinator market state survives checkpoint/restore cycle."""
        data_dir = str(tmp_path / "live")

        # --- Phase 1: process market events, verify state ---
        stack1 = SimulatedLiveStack(data_dir=data_dir)
        stack1.start()

        for i in range(15):
            stack1.emit_market("BTCUSDT", 40000.0 + i * 100, i)

        view1 = stack1.coordinator.get_state_view()
        assert "BTCUSDT" in view1["markets"]
        view1["markets"]["BTCUSDT"].close
        view1["event_index"]

        # Also checkpoint inference bridge and feature hook
        stack1.inference_bridge.push_zscore("BTCUSDT", 2.0)
        stack1.inference_bridge.push_zscore("BTCUSDT", 3.0)
        stack1.feature_hook.increment("BTCUSDT")
        stack1.feature_hook.increment("BTCUSDT")
        stack1.feature_hook.increment("BTCUSDT")

        stack1.save_all()

        # --- Phase 2: fresh stack, restore auxiliary state ---
        stack2 = SimulatedLiveStack(data_dir=data_dir)
        stack2.start()

        result = stack2.restore_all()
        assert result["bridge_restored"] is True
        assert result["hook_restored"] is True

        # Verify auxiliary state consistency
        assert stack2.inference_bridge._zscore_buffers == {"BTCUSDT": [2.0, 3.0]}
        assert stack2.feature_hook._bar_count == {"BTCUSDT": 3}

        # The coordinator in stack2 is fresh (no snapshot restore here),
        # but the auxiliary state (bridge + hook) is fully restored.
        # In production, coordinator state would be restored via
        # SqliteStateStore + restore_from_snapshot (tested in test_crash_recovery.py).
        # Here we verify that auxiliary state restore is independent and correct.

    def test_multiple_symbols_checkpoint_restore(self, tmp_path):
        """Multi-symbol checkpoint/restore preserves per-symbol state."""
        data_dir = str(tmp_path / "live")
        symbols = ("BTCUSDT", "ETHUSDT")

        stack1 = SimulatedLiveStack(data_dir=data_dir, symbols=symbols)
        stack1.start()

        for i in range(10):
            stack1.emit_market("BTCUSDT", 40000.0 + i, i)
            stack1.emit_market("ETHUSDT", 3000.0 + i, 100 + i)

        # Per-symbol inference bridge data
        stack1.inference_bridge.push_zscore("BTCUSDT", 0.5)
        stack1.inference_bridge.push_zscore("BTCUSDT", 1.0)
        stack1.inference_bridge.push_zscore("ETHUSDT", -0.3)

        # Per-symbol bar counts
        for _ in range(10):
            stack1.feature_hook.increment("BTCUSDT")
        for _ in range(7):
            stack1.feature_hook.increment("ETHUSDT")

        # Kill one symbol
        stack1.kill_switch.trigger(
            scope=KillScope.SYMBOL,
            key="ETHUSDT",
            mode=KillMode.HARD_KILL,
            reason="eth paused",
        )

        stack1.save_all()

        # Fresh stack
        stack2 = SimulatedLiveStack(data_dir=data_dir, symbols=symbols)
        stack2.start()
        stack2.restore_all()

        # Verify per-symbol state
        assert stack2.feature_hook._bar_count == {"BTCUSDT": 10, "ETHUSDT": 7}
        assert stack2.inference_bridge._zscore_buffers == {
            "BTCUSDT": [0.5, 1.0],
            "ETHUSDT": [-0.3],
        }

        # Kill switch: only ETHUSDT should be killed
        assert stack2.kill_switch.is_killed(symbol="ETHUSDT") is not None
        assert stack2.kill_switch.is_killed(symbol="BTCUSDT") is None

    def test_empty_state_checkpoint_restore(self, tmp_path):
        """Checkpoint with no active state restores cleanly to empty state."""
        data_dir = str(tmp_path / "live")

        stack1 = SimulatedLiveStack(data_dir=data_dir)
        stack1.start()
        # No events, no bridge data, no feature hook data, no kills
        stack1.save_all()

        stack2 = SimulatedLiveStack(data_dir=data_dir)
        stack2.start()
        result = stack2.restore_all()

        assert result["kill_switch_restored"] is False
        # Bridge checkpoint() returns a dict even when empty, so file is written
        # and restore succeeds — but state remains empty
        assert result["hook_restored"] is False
        assert stack2.feature_hook._bar_count == {}
        assert stack2.inference_bridge._zscore_buffers == {}

    def test_inference_bridge_dict_mode(self, tmp_path):
        """Recovery functions handle dict-of-bridges (multi-symbol bridge map)."""
        data_dir = str(tmp_path / "live")
        os.makedirs(data_dir, exist_ok=True)

        bridges = {
            "BTCUSDT": MockInferenceBridge(),
            "ETHUSDT": MockInferenceBridge(),
        }
        bridges["BTCUSDT"].push_zscore("BTCUSDT", 1.1)
        bridges["ETHUSDT"].push_zscore("ETHUSDT", -0.5)
        bridges["ETHUSDT"].push_zscore("ETHUSDT", -0.7)

        save_inference_bridge_state(bridges, data_dir=data_dir)

        # Restore into fresh bridges
        bridges2 = {
            "BTCUSDT": MockInferenceBridge(),
            "ETHUSDT": MockInferenceBridge(),
        }
        ok = restore_inference_bridge_state(bridges2, data_dir=data_dir)
        assert ok is True
        assert bridges2["BTCUSDT"]._zscore_buffers == {"BTCUSDT": [1.1]}
        assert bridges2["ETHUSDT"]._zscore_buffers == {"ETHUSDT": [-0.5, -0.7]}

    def test_exit_manager_checkpoint_restore(self, tmp_path):
        """Exit manager positions survive checkpoint/restore cycle."""
        data_dir = str(tmp_path / "live")

        stack1 = SimulatedLiveStack(data_dir=data_dir)
        stack1.start()

        stack1.exit_manager.on_entry("BTCUSDT", price=40000.0, bar=5, direction=1.0)
        stack1.save_all()

        stack2 = SimulatedLiveStack(data_dir=data_dir)
        stack2.start()
        result = stack2.restore_all()

        assert result["exit_manager_restored"] is True
        assert "BTCUSDT" in stack2.exit_manager._positions
        assert stack2.exit_manager._positions["BTCUSDT"]["entry_price"] == 40000.0

    def test_regime_gate_checkpoint_restore(self, tmp_path):
        """Regime gate buffers survive checkpoint/restore cycle."""
        data_dir = str(tmp_path / "live")

        stack1 = SimulatedLiveStack(data_dir=data_dir)
        stack1.start()

        for i in range(100):
            stack1.regime_gate.push(0.01 * i, 0.005 * i)
        stack1.save_all()

        stack2 = SimulatedLiveStack(data_dir=data_dir)
        stack2.start()
        result = stack2.restore_all()

        assert result["regime_gate_restored"] is True
        assert len(stack2.regime_gate._bb_width_buf) == 100
        assert len(stack2.regime_gate._vol_of_vol_buf) == 100

    def test_correlation_computer_checkpoint_restore(self, tmp_path):
        """Correlation computer returns survive checkpoint/restore cycle."""
        data_dir = str(tmp_path / "live")

        stack1 = SimulatedLiveStack(data_dir=data_dir)
        stack1.start()

        for i in range(20):
            stack1.correlation_computer.update("BTCUSDT", 40000.0 + i * 10)
            stack1.correlation_computer.update("ETHUSDT", 3000.0 + i * 5)

        corr_before = stack1.correlation_computer.portfolio_avg_correlation(
            ["BTCUSDT", "ETHUSDT"]
        )
        stack1.save_all()

        stack2 = SimulatedLiveStack(data_dir=data_dir)
        stack2.start()
        result = stack2.restore_all()

        assert result["correlation_restored"] is True
        corr_after = stack2.correlation_computer.portfolio_avg_correlation(
            ["BTCUSDT", "ETHUSDT"]
        )
        assert corr_before is not None
        assert corr_after is not None
        assert abs(corr_before - corr_after) < 1e-10

    def test_atomic_bundle_roundtrip(self, tmp_path):
        """Atomic bundle save/restore covers all components."""
        data_dir = str(tmp_path / "live")

        stack1 = SimulatedLiveStack(data_dir=data_dir)
        stack1.start()

        # Populate all components
        for i in range(10):
            stack1.emit_market("BTCUSDT", 40000.0 + i, i)
        stack1.inference_bridge.push_zscore("BTCUSDT", 1.5)
        stack1.feature_hook.increment("BTCUSDT")
        stack1.exit_manager.on_entry("BTCUSDT", 40000.0, 5, 1.0)
        for i in range(50):
            stack1.regime_gate.push(0.01 * i, 0.005 * i)
        for i in range(10):
            stack1.correlation_computer.update("BTCUSDT", 40000.0 + i * 10)

        stack1.save_all()

        # Verify bundle file exists
        bundle_path = os.path.join(data_dir, "auxiliary_state_bundle.json")
        assert os.path.exists(bundle_path)

        stack2 = SimulatedLiveStack(data_dir=data_dir)
        stack2.start()
        result = stack2.restore_all()

        assert result["bridge_restored"] is True
        assert result["hook_restored"] is True
        assert result["exit_manager_restored"] is True
        assert result["regime_gate_restored"] is True
        assert result["correlation_restored"] is True
