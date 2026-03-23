"""Tests for EntryScaler."""
from __future__ import annotations

import pytest


@pytest.fixture
def scaler():
    from portfolio.entry_scaler import EntryScaler
    return EntryScaler()


class TestBBScale:

    def test_long_oversold_boost(self, scaler):
        # Price well below BB → 1.2x
        closes = [100.0] * 12 + [90.0]  # last price far below mean
        assert scaler.bb_scale(1, closes, window=12) == 1.2

    def test_long_overbought_reduce(self, scaler):
        # Price well above BB → reduced (continuous: < 1.0)
        closes = [100.0] * 12 + [110.0]
        scale = scaler.bb_scale(1, closes, window=12)
        assert 0.75 <= scale < 1.0

    def test_short_overbought_boost(self, scaler):
        # Short into overbought → boosted (continuous: > 1.0)
        closes = [100.0] * 12 + [110.0]
        scale = scaler.bb_scale(-1, closes, window=12)
        assert scale > 1.0

    def test_short_oversold_reduce(self, scaler):
        # Short into oversold → reduced (continuous: < 1.0)
        closes = [100.0] * 12 + [90.0]
        scale = scaler.bb_scale(-1, closes, window=12)
        assert 0.75 <= scale < 1.0

    def test_zero_signal(self, scaler):
        closes = [100.0] * 20
        assert scaler.bb_scale(0, closes) == 1.0

    def test_insufficient_data(self, scaler):
        closes = [100.0] * 5
        assert scaler.bb_scale(1, closes, window=12) == 1.0

    def test_zero_std(self, scaler):
        # All same price → std=0 → 1.0
        closes = [100.0] * 20
        assert scaler.bb_scale(1, closes, window=12) == 1.0

    def test_moderate_oversold_long(self, scaler):
        # Slightly below mean → 1.0
        closes = [100.0, 101.0, 99.0, 100.5, 99.5, 100.0, 101.0, 99.0, 100.5, 99.5, 100.0, 101.0, 98.5]
        scale = scaler.bb_scale(1, closes, window=12)
        assert scale in (0.85, 1.0, 1.2)

    def test_return_range(self, scaler):
        """All returns should be in [0.6, 1.2]."""
        import random
        random.seed(42)
        for _ in range(100):
            closes = [100 + random.gauss(0, 5) for _ in range(20)]
            for sig in [1, -1]:
                s = scaler.bb_scale(sig, closes, window=12)
                assert 0.75 <= s <= 1.2


class TestLeverageScale:

    def test_no_drawdown(self, scaler):
        assert scaler.leverage_scale(0.0) == 1.0

    def test_mild_drawdown(self, scaler):
        assert scaler.leverage_scale(5.0) == 1.0

    def test_10pct_drawdown(self, scaler):
        assert scaler.leverage_scale(10.0) == 0.75

    def test_20pct_drawdown(self, scaler):
        assert scaler.leverage_scale(20.0) == 0.50

    def test_35pct_drawdown(self, scaler):
        assert scaler.leverage_scale(35.0) == 0.25

    def test_extreme_drawdown(self, scaler):
        assert scaler.leverage_scale(80.0) == 0.25

    def test_boundary_15pct(self, scaler):
        assert scaler.leverage_scale(15.0) == 0.75

    def test_boundary_25pct(self, scaler):
        assert scaler.leverage_scale(25.0) == 0.50


class TestAdaptiveHold:

    def test_normal_vol(self, scaler):
        mn, mx = scaler.adaptive_hold(18, 60, vol_ratio=1.0)
        assert mn == 18
        assert mx == 60

    def test_high_vol_longer_hold(self, scaler):
        mn, mx = scaler.adaptive_hold(18, 60, vol_ratio=2.0)
        assert mn > 18  # should be ~25
        assert mx > 60  # should be ~84

    def test_low_vol_shorter_hold(self, scaler):
        mn, mx = scaler.adaptive_hold(18, 60, vol_ratio=0.5)
        assert mn < 18  # should be ~12
        assert mx < 60  # should be ~42

    def test_min_hold_at_least_1(self, scaler):
        mn, mx = scaler.adaptive_hold(1, 5, vol_ratio=0.1)
        assert mn >= 1
        assert mx > mn


class TestVolAwareLeverage:

    def test_normal_vol_normal_dd(self, scaler):
        assert scaler.leverage_scale(15.0, vol_ratio=1.0) == 0.75

    def test_high_vol_same_dd_is_less_severe(self, scaler):
        # At vol_ratio=2.0, DD threshold widens: 10%*2=20%, so 15% DD < 20% → 1.0
        assert scaler.leverage_scale(15.0, vol_ratio=2.0) == 1.0

    def test_low_vol_same_dd_is_more_severe(self, scaler):
        # At vol_ratio=0.5, DD threshold tightens: 10%*0.5=5%, so 15% DD > 10% → 0.5
        assert scaler.leverage_scale(15.0, vol_ratio=0.5) == 0.50

    def test_backward_compat_no_vol_ratio(self, scaler):
        assert scaler.leverage_scale(10.0) == 0.75
        assert scaler.leverage_scale(20.0) == 0.50


class TestConfidenceCapScale:

    def test_weak_signal(self, scaler):
        # z just above dz → low confidence → 0.7x
        scale = scaler.confidence_cap_scale(z_score=0.55, base_dz=0.5)
        assert 0.7 <= scale <= 0.8

    def test_strong_signal(self, scaler):
        # z = 3x dz → high confidence → ~1.3x
        scale = scaler.confidence_cap_scale(z_score=1.5, base_dz=0.5)
        assert scale >= 1.2

    def test_zero_dz(self, scaler):
        assert scaler.confidence_cap_scale(1.0, 0.0) == 1.0

    def test_clamped(self, scaler):
        scale = scaler.confidence_cap_scale(10.0, 0.5)
        assert scale <= 1.3


class TestVolAdaptiveDeadzone:

    def test_normal_vol(self, scaler):
        # Ratio = 1.0 → no change
        assert scaler.vol_adaptive_deadzone(0.5, 0.006, 0.006) == 0.5

    def test_high_vol(self, scaler):
        # Ratio = 2.0 → deadzone doubles
        dz = scaler.vol_adaptive_deadzone(0.5, 0.012, 0.006)
        assert abs(dz - 1.0) < 1e-6

    def test_low_vol(self, scaler):
        # Ratio = 0.5 → deadzone halves
        dz = scaler.vol_adaptive_deadzone(0.5, 0.003, 0.006)
        assert abs(dz - 0.25) < 1e-6

    def test_extreme_high_vol_clamped(self, scaler):
        # Ratio > 2.0 → clamped to 2.0
        dz = scaler.vol_adaptive_deadzone(0.5, 0.1, 0.006)
        assert abs(dz - 1.0) < 1e-6

    def test_extreme_low_vol_clamped(self, scaler):
        # Ratio < 0.5 → clamped to 0.5
        dz = scaler.vol_adaptive_deadzone(0.5, 0.001, 0.006)
        assert abs(dz - 0.25) < 1e-6

    def test_zero_vol_median(self, scaler):
        assert scaler.vol_adaptive_deadzone(0.5, 0.01, 0.0) == 0.5

    def test_negative_vol_median(self, scaler):
        assert scaler.vol_adaptive_deadzone(0.5, 0.01, -1.0) == 0.5
