"""Tests for LiveInferenceBridge.update_params() method."""
from __future__ import annotations

import pytest

_quant_hotpath = pytest.importorskip("_quant_hotpath")

from alpha.inference.bridge import LiveInferenceBridge
from alpha.base import Signal


class _DummyModel:
    """Minimal AlphaModel for testing."""
    name = "test_model"

    def predict(self, *, symbol, ts, features):
        return Signal(side="long", strength=0.5)

    def feature_names(self):
        return ["close"]


class TestUpdateParams:

    def _make_bridge(self, **kwargs):
        return LiveInferenceBridge(
            models=[_DummyModel()],
            min_hold_bars={"BTCUSDT": 12, "ETHUSDT": 24},
            deadzone={"BTCUSDT": 0.5, "ETHUSDT": 0.8},
            long_only_symbols=set(),
            **kwargs,
        )

    def test_update_deadzone(self):
        bridge = self._make_bridge()
        bridge.update_params("BTCUSDT", deadzone=1.5)
        assert bridge._deadzone["BTCUSDT"] == 1.5
        # ETH should be unchanged
        assert bridge._deadzone["ETHUSDT"] == 0.8

    def test_update_min_hold(self):
        bridge = self._make_bridge()
        bridge.update_params("BTCUSDT", min_hold=8)
        assert bridge._min_hold_bars["BTCUSDT"] == 8
        assert bridge._min_hold_bars["ETHUSDT"] == 24

    def test_update_max_hold(self):
        bridge = self._make_bridge()
        bridge.update_params("BTCUSDT", max_hold=60)
        assert bridge._max_hold == 60

    def test_update_long_only_add(self):
        bridge = self._make_bridge()
        bridge.update_params("BTCUSDT", long_only=True)
        assert "BTCUSDT" in bridge._long_only_symbols

    def test_update_long_only_remove(self):
        bridge = self._make_bridge()
        bridge._long_only_symbols.add("BTCUSDT")
        bridge.update_params("BTCUSDT", long_only=False)
        assert "BTCUSDT" not in bridge._long_only_symbols

    def test_partial_update_preserves_others(self):
        bridge = self._make_bridge()
        old_min_hold = bridge._min_hold_bars["BTCUSDT"]
        bridge.update_params("BTCUSDT", deadzone=2.0)
        assert bridge._min_hold_bars["BTCUSDT"] == old_min_hold

    def test_update_with_scalar_deadzone_converts_to_dict(self):
        bridge = LiveInferenceBridge(
            models=[_DummyModel()],
            deadzone=0.5,  # scalar
        )
        bridge.update_params("BTCUSDT", deadzone=1.5)
        assert isinstance(bridge._deadzone, dict)
        assert bridge._deadzone["BTCUSDT"] == 1.5
