"""End-to-end tests for crash recovery save/restore cycle.

Tests save_all_auxiliary_state() and restore_all_auxiliary_state() from
runner/recovery.py with mock components.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from runner.recovery import (
    save_all_auxiliary_state,
    restore_all_auxiliary_state,
    _AUXILIARY_BUNDLE_FILE,
)


# ── Helpers ──────────────────────────────────────────────────


class _MockCheckpointable:
    """Mock component with checkpoint/restore (exit_manager, regime_gate, etc.)."""

    def __init__(self, data: dict | None = None):
        self._data = data or {}
        self.restored_data = None

    def checkpoint(self) -> dict:
        return dict(self._data)

    def restore(self, data: dict) -> None:
        self.restored_data = data
        self._data = dict(data)


class _MockInferenceBridge:
    """Mock inference bridge with checkpoint/restore."""

    def __init__(self, z_scores: dict | None = None):
        self._z = z_scores or {"BTCUSDT": 1.5}
        self.restored_data = None

    def checkpoint(self) -> dict:
        return {"z_scores": dict(self._z)}

    def restore(self, data: dict) -> None:
        self.restored_data = data
        self._z = data.get("z_scores", {})


class _MockFeatureHook:
    """Mock feature hook with _bar_count attribute."""

    def __init__(self, bar_count: dict | None = None):
        self._bar_count = dict(bar_count) if bar_count else {}


class _MockDrawdownBreaker:
    """Mock drawdown breaker with checkpoint/restore_checkpoint."""

    def __init__(self, hwm: float = 0.0):
        self._hwm = hwm
        self.restored_data = None

    def checkpoint(self) -> dict:
        return {"equity_hwm": self._hwm, "state": "normal"}

    def restore_checkpoint(self, data: dict) -> None:
        self.restored_data = data
        self._hwm = data.get("equity_hwm", 0.0)


# ── Tests ────────────────────────────────────────────────────


def test_save_and_restore_all_auxiliary_state(tmp_path):
    """Full save -> restore roundtrip for all 8 components."""
    data_dir = str(tmp_path)

    # Create components with state
    inference_bridge = _MockInferenceBridge(z_scores={"BTCUSDT": 2.1, "ETHUSDT": -0.5})
    feature_hook = _MockFeatureHook(bar_count={"BTCUSDT": 100, "ETHUSDT": 50})
    exit_manager = _MockCheckpointable({"BTCUSDT": {"trail_pct": 0.02}})
    regime_gate = _MockCheckpointable({"bb_width_buf": [0.01, 0.02, 0.03]})
    correlation_computer = _MockCheckpointable({"last_prices": {"BTCUSDT": 50000}})
    timeout_tracker = _MockCheckpointable({"pending": {"ord-1": 1700000000}})
    drawdown_breaker = _MockDrawdownBreaker(hwm=55000.0)

    # Save
    save_all_auxiliary_state(
        inference_bridge=inference_bridge,
        feature_hook=feature_hook,
        exit_manager=exit_manager,
        regime_gate=regime_gate,
        correlation_computer=correlation_computer,
        timeout_tracker=timeout_tracker,
        drawdown_breaker=drawdown_breaker,
        data_dir=data_dir,
    )

    # Verify atomic bundle file exists
    bundle_path = os.path.join(data_dir, _AUXILIARY_BUNDLE_FILE)
    assert os.path.exists(bundle_path)
    with open(bundle_path) as f:
        bundle = json.load(f)
    assert "inference_bridge" in bundle
    assert "feature_hook" in bundle
    assert "exit_manager" in bundle
    assert "regime_gate" in bundle
    assert "drawdown_breaker" in bundle

    # Create fresh components
    fresh_bridge = _MockInferenceBridge(z_scores={})
    fresh_hook = _MockFeatureHook(bar_count={})
    fresh_exit = _MockCheckpointable()
    fresh_regime = _MockCheckpointable()
    fresh_corr = _MockCheckpointable()
    fresh_timeout = _MockCheckpointable()
    fresh_dd = _MockDrawdownBreaker()

    # Restore
    results = restore_all_auxiliary_state(
        inference_bridge=fresh_bridge,
        feature_hook=fresh_hook,
        exit_manager=fresh_exit,
        regime_gate=fresh_regime,
        correlation_computer=fresh_corr,
        timeout_tracker=fresh_timeout,
        drawdown_breaker=fresh_dd,
        data_dir=data_dir,
    )

    # Verify all restored
    assert results["inference_bridge"] is True
    assert results["feature_hook"] is True
    assert results["exit_manager"] is True
    assert results["regime_gate"] is True
    assert results["correlation_computer"] is True
    assert results["timeout_tracker"] is True
    assert results["drawdown_breaker"] is True

    # Verify actual data was restored
    assert fresh_bridge.restored_data["z_scores"]["BTCUSDT"] == 2.1
    assert fresh_hook._bar_count["BTCUSDT"] == 100
    assert fresh_exit.restored_data["BTCUSDT"]["trail_pct"] == 0.02
    assert fresh_regime.restored_data["bb_width_buf"] == [0.01, 0.02, 0.03]
    assert fresh_dd.restored_data["equity_hwm"] == 55000.0


def test_restore_falls_back_to_individual_files(tmp_path):
    """If atomic bundle missing, restore from individual files."""
    data_dir = str(tmp_path)

    # Save using individual file paths (simulating old-style saves)
    # We save only inference bridge individually -- the bundle won't exist
    bridge = _MockInferenceBridge(z_scores={"BTCUSDT": 3.0})
    from runner.recovery import save_inference_bridge_state
    save_inference_bridge_state(bridge, data_dir=data_dir)

    # Verify bundle does NOT exist
    bundle_path = os.path.join(data_dir, _AUXILIARY_BUNDLE_FILE)
    assert not os.path.exists(bundle_path)

    # Restore -- should fall back to individual file
    fresh_bridge = _MockInferenceBridge(z_scores={})
    results = restore_all_auxiliary_state(
        inference_bridge=fresh_bridge,
        data_dir=data_dir,
    )

    assert results["inference_bridge"] is True
    assert fresh_bridge.restored_data["z_scores"]["BTCUSDT"] == 3.0


def test_partial_bundle_corruption_handled(tmp_path):
    """If bundle JSON is truncated, restore gracefully fails and falls back."""
    data_dir = str(tmp_path)

    # Write a corrupted (truncated) bundle file
    bundle_path = os.path.join(data_dir, _AUXILIARY_BUNDLE_FILE)
    os.makedirs(data_dir, exist_ok=True)
    with open(bundle_path, "w") as f:
        f.write('{"inference_bridge": {"_default": {"z_scores":')  # truncated JSON

    fresh_bridge = _MockInferenceBridge(z_scores={})
    fresh_exit = _MockCheckpointable()

    # Should not raise -- graceful fallback
    results = restore_all_auxiliary_state(
        inference_bridge=fresh_bridge,
        exit_manager=fresh_exit,
        data_dir=data_dir,
    )

    # With corrupted bundle and no individual files, nothing should restore
    assert results.get("inference_bridge", False) is False
    assert results.get("exit_manager", False) is False
