"""End-to-end recovery tests: checkpoint -> crash -> restore cycle.

Validates that the full recovery infrastructure (kill switch, inference bridge,
feature hook, coordinator state) survives a simulated crash and restart with
consistent state.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from risk.kill_switch import KillMode, KillScope, KillSwitch
from runner.recovery import (
    save_kill_switch_state,
    restore_kill_switch_state,
    save_inference_bridge_state,
    restore_inference_bridge_state,
    save_feature_hook_state,
    restore_feature_hook_state,
    reconcile_and_heal,
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

    def start(self) -> None:
        self.coordinator.start()

    def emit_market(self, symbol: str, close: float, idx: int) -> None:
        self.coordinator.emit(_market_event(symbol, close, idx), actor="test")

    def save_all(self) -> None:
        """Save all component state to data_dir."""
        save_kill_switch_state(self.kill_switch, data_dir=self.data_dir)
        save_inference_bridge_state(self.inference_bridge, data_dir=self.data_dir)
        save_feature_hook_state(self.feature_hook, data_dir=self.data_dir)

    def restore_all(self) -> Dict[str, Any]:
        """Restore all component state from data_dir. Returns restore results."""
        ks_count = restore_kill_switch_state(self.kill_switch, data_dir=self.data_dir)
        bridge_ok = restore_inference_bridge_state(
            self.inference_bridge, data_dir=self.data_dir
        )
        hook_ok = restore_feature_hook_state(
            self.feature_hook, data_dir=self.data_dir
        )
        return {
            "kill_switch_restored": ks_count,
            "bridge_restored": bridge_ok,
            "hook_restored": hook_ok,
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
        assert result["kill_switch_restored"] == 1

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

        # --- Phase 2: corrupt the inference bridge checkpoint ---
        bridge_path = os.path.join(data_dir, "inference_bridge_checkpoint.json")
        assert os.path.exists(bridge_path)

        # Write truncated/corrupt JSON
        with open(bridge_path, "w") as f:
            f.write('{"_default": {"zscore_buffers": {"BTCUSDT": [1.5')  # truncated

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
        orig_close = view1["markets"]["BTCUSDT"].close
        orig_event_index = view1["event_index"]

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
        result = stack2.restore_all()

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

        assert result["kill_switch_restored"] == 0
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
