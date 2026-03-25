"""Tests for RegimeGate — regime classification and position scaling."""
from __future__ import annotations

import pytest
from alpha.v11_config import RegimeGateConfig
from strategy.regime_gate import RegimeGate


@pytest.fixture
def disabled_gate():
    return RegimeGate(config=RegimeGateConfig(enabled=False))


@pytest.fixture
def enabled_gate():
    return RegimeGate(config=RegimeGateConfig(enabled=True, reduce_factor=0.3))


@pytest.fixture
def skip_gate():
    return RegimeGate(config=RegimeGateConfig(
        enabled=True,
        ranging_high_vol_action="skip",
    ))


class TestDisabledGate:
    def test_always_returns_normal(self, disabled_gate):
        label, scale = disabled_gate.evaluate({"bb_width_20": 100.0, "vol_of_vol": 50.0})
        assert label == "normal"
        assert scale == 1.0


class TestADXBasedRegime:
    def test_trending_high_adx(self, enabled_gate):
        gate = enabled_gate
        # Warm up the percentile buffers
        for i in range(150):
            gate.evaluate({"adx_14": 30.0, "bb_width_20": float(i % 10), "vol_of_vol": float(i % 10)})
        label, scale = gate.evaluate({"adx_14": 30.0, "bb_width_20": 5.0, "vol_of_vol": 5.0})
        assert label == "trending"
        assert scale == 1.0

    def test_ranging_low_adx_low_vol(self, enabled_gate):
        gate = enabled_gate
        # Warm up with spread of values — current value is in the middle
        for i in range(150):
            gate.evaluate({"adx_14": 15.0, "bb_width_20": float(i % 20), "vol_of_vol": float(i % 20)})
        # Value of 5 is well below 75th percentile of 0-19 range
        label, scale = gate.evaluate({"adx_14": 15.0, "bb_width_20": 5.0, "vol_of_vol": 5.0})
        assert label == "ranging"
        assert scale == 1.0

    def test_ranging_high_vol_reduces(self, enabled_gate):
        gate = enabled_gate
        # Warm up with low values, then spike
        for i in range(150):
            gate.evaluate({"adx_14": 15.0, "bb_width_20": 1.0, "vol_of_vol": 1.0})
        # Now high bb_width — should be in top percentile
        label, scale = gate.evaluate({"adx_14": 15.0, "bb_width_20": 100.0, "vol_of_vol": 1.0})
        assert label == "ranging_high_vol"
        assert scale == pytest.approx(0.3)


class TestSkipAction:
    def test_ranging_high_vol_skips(self, skip_gate):
        gate = skip_gate
        for i in range(150):
            gate.evaluate({"adx_14": 15.0, "bb_width_20": 1.0, "vol_of_vol": 1.0})
        label, scale = gate.evaluate({"adx_14": 15.0, "bb_width_20": 100.0, "vol_of_vol": 1.0})
        assert label == "ranging_high_vol"
        assert scale == 0.0


class TestFallbackNoADX:
    def test_no_adx_normal(self, enabled_gate):
        gate = enabled_gate
        for i in range(150):
            gate.evaluate({"bb_width_20": float(i % 20), "vol_of_vol": float(i % 20)})
        # Value 5 is well below 75th percentile of 0-19
        label, scale = gate.evaluate({"bb_width_20": 5.0, "vol_of_vol": 5.0})
        assert label == "normal"
        assert scale == 1.0

    def test_no_adx_high_vol_reduces(self, enabled_gate):
        gate = enabled_gate
        for i in range(150):
            gate.evaluate({"bb_width_20": float(i % 10), "vol_of_vol": float(i % 10)})
        # 100 is clearly above 75th percentile of 0-9
        label, scale = gate.evaluate({"bb_width_20": 100.0, "vol_of_vol": 1.0})
        assert label == "ranging_high_vol"
        assert scale == pytest.approx(0.3)
