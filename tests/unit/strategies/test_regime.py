"""Tests for regime classification."""
from __future__ import annotations

import pytest

from strategies.multi_factor.feature_computer import MultiFactorFeatures
from strategies.multi_factor.regime import Regime, classify_regime


def _make_features(**overrides) -> MultiFactorFeatures:
    defaults = dict(
        sma_fast=None, sma_slow=None, sma_trend=None,
        rsi=None, macd=None, macd_signal=None, macd_hist=None,
        bb_upper=None, bb_middle=None, bb_lower=None, bb_pct=None,
        atr=None, atr_pct=None, atr_percentile=None, ma_slope=None,
        close=100.0, volume=1000.0,
    )
    defaults.update(overrides)
    return MultiFactorFeatures(**defaults)


class TestClassifyRegime:
    def test_none_when_missing_fields(self):
        f = _make_features()
        assert classify_regime(f) is None

    def test_none_when_sma_missing(self):
        f = _make_features(atr_percentile=50.0, ma_slope=0.002)
        assert classify_regime(f) is None

    def test_high_vol(self):
        f = _make_features(
            atr_percentile=90.0, ma_slope=0.005,
            sma_fast=105.0, sma_slow=100.0,
        )
        assert classify_regime(f, atr_extreme_pct=85.0) == Regime.HIGH_VOL

    def test_trending_up(self):
        f = _make_features(
            atr_percentile=50.0, ma_slope=0.005,
            sma_fast=105.0, sma_slow=100.0,
        )
        assert classify_regime(f) == Regime.TRENDING_UP

    def test_trending_down(self):
        f = _make_features(
            atr_percentile=50.0, ma_slope=-0.005,
            sma_fast=95.0, sma_slow=100.0,
        )
        assert classify_regime(f) == Regime.TRENDING_DOWN

    def test_ranging_default(self):
        f = _make_features(
            atr_percentile=50.0, ma_slope=0.0005,  # below threshold
            sma_fast=100.0, sma_slow=100.0,
        )
        assert classify_regime(f) == Regime.RANGING

    def test_slope_positive_but_sma_inverted_is_ranging(self):
        # ma_slope > threshold but sma_fast < sma_slow => not TRENDING_UP
        f = _make_features(
            atr_percentile=50.0, ma_slope=0.005,
            sma_fast=95.0, sma_slow=100.0,
        )
        result = classify_regime(f)
        # slope positive but fast < slow => falls through to TRENDING_DOWN check
        # slope is positive, so not < -threshold => RANGING
        assert result == Regime.RANGING

    def test_custom_thresholds(self):
        f = _make_features(
            atr_percentile=80.0, ma_slope=0.0001,
            sma_fast=101.0, sma_slow=100.0,
        )
        # Default slope_threshold=0.001 => this is below => RANGING
        assert classify_regime(f) == Regime.RANGING
        # With lower threshold => TRENDING_UP
        assert classify_regime(f, slope_threshold=0.00005) == Regime.TRENDING_UP
