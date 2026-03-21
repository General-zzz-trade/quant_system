"""Tests for LiquidationCascadeGate."""

from unittest.mock import MagicMock
from runner.gates.liquidation_cascade_gate import (
    LiquidationCascadeGate,
    LiquidationCascadeConfig,
)


def _make_ev(signal=0):
    ev = MagicMock()
    ev.metadata = {"signal": signal}
    return ev


class TestLiquidationCascadeGate:
    def test_normal_conditions_passthrough(self):
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 0.5,
            "oi_acceleration": 0.1,
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_extreme_liq_blocks(self):
        """liq_zscore >= 3.0 → block."""
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 3.5,
        })
        assert not r.allowed
        assert "block" in r.reason

    def test_danger_liq_plus_oi_unwind_blocks(self):
        """liq_zscore >= 2.0 AND oi_accel <= -2.0 → block."""
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 2.5,
            "oi_acceleration": -2.5,
        })
        assert not r.allowed

    def test_danger_liq_alone_scales(self):
        """liq_zscore >= 2.0 (without OI unwind) → 0.3x."""
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 2.5,
            "oi_acceleration": 0.0,
        })
        assert r.allowed
        assert r.scale == 0.3

    def test_caution_liq_scales(self):
        """liq_zscore >= 1.5 → 0.5x."""
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 1.8,
            "oi_acceleration": 0.0,
        })
        assert r.allowed
        assert r.scale == 0.5

    def test_oi_unwind_alone_scales(self):
        """oi_accel <= -2.0 (without high liq) → 0.5x."""
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 0.5,
            "oi_acceleration": -2.5,
        })
        assert r.allowed
        assert r.scale == 0.5

    def test_oi_caution_scales(self):
        """oi_accel <= -1.5 → 0.7x."""
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 0.5,
            "oi_acceleration": -1.8,
        })
        assert r.allowed
        assert r.scale == 0.7

    def test_cascade_score_scales(self):
        """High cascade_score → 0.5x."""
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 0.5,
            "oi_acceleration": 0.0,
            "liquidation_cascade_score": 3.0,
        })
        assert r.allowed
        assert r.scale == 0.5

    def test_contrarian_slight_boost(self):
        """Long after sell liquidations → slight boost."""
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 0.5,
            "oi_acceleration": 0.0,
            "liquidation_imbalance": -0.5,  # sell liquidations
        })
        assert r.allowed
        assert r.scale == 1.1

    def test_disabled(self):
        gate = LiquidationCascadeGate(LiquidationCascadeConfig(enabled=False))
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 5.0,
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_missing_data_defaults_to_zero(self):
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {})
        assert r.allowed
        assert r.scale == 1.0

    def test_nan_values_default(self):
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": float("nan"),
        })
        assert r.allowed
        assert r.scale == 1.0

    def test_stats(self):
        gate = LiquidationCascadeGate()
        gate.check(_make_ev(1), {"liquidation_volume_zscore_24": 3.5})
        gate.check(_make_ev(1), {"liquidation_volume_zscore_24": 2.0})
        gate.check(_make_ev(1), {"liquidation_volume_zscore_24": 0.5})
        stats = gate.stats
        assert stats["total_checks"] == 3
        assert stats["blocked"] == 1
        assert stats["scaled"] == 1

    def test_combined_min_scale(self):
        """Multiple triggers → take minimum scale."""
        gate = LiquidationCascadeGate()
        r = gate.check(_make_ev(1), {
            "liquidation_volume_zscore_24": 1.8,  # caution: 0.5
            "oi_acceleration": -2.5,               # danger: 0.5
            "liquidation_cascade_score": 3.0,      # caution: 0.5
        })
        assert r.allowed
        assert r.scale == 0.5  # min of all
