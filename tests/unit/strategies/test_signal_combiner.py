"""Tests for signal_combiner — trend score, range score, combined signal."""
from __future__ import annotations

import pytest

from strategies.multi_factor.feature_computer import MultiFactorFeatures
from strategies.multi_factor.regime import Regime
from strategies.multi_factor.signal_combiner import (
    CombinedSignal,
    _clamp,
    _range_score,
    _trend_score,
    combine_signals,
)


def _make_features(**overrides) -> MultiFactorFeatures:
    """Build a MultiFactorFeatures with sensible defaults, overridable."""
    defaults = dict(
        sma_fast=None, sma_slow=None, sma_trend=None,
        rsi=None, macd=None, macd_signal=None, macd_hist=None,
        bb_upper=None, bb_middle=None, bb_lower=None, bb_pct=None,
        atr=None, atr_pct=None, atr_percentile=None, ma_slope=None,
        close=100.0, volume=1000.0,
    )
    defaults.update(overrides)
    return MultiFactorFeatures(**defaults)


# ── _clamp ───────────────────────────────────────────────────

class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_high(self):
        assert _clamp(5.0) == 1.0

    def test_clamp_low(self):
        assert _clamp(-3.0) == -1.0


# ── Trend score ──────────────────────────────────────────────

class TestTrendScore:
    def test_bullish_indicators(self):
        features = _make_features(
            sma_fast=105.0, sma_slow=100.0,  # fast > slow => positive ma_cross
            macd_hist=2.0, atr=1.0,           # positive MACD
            rsi=65.0,                          # RSI > 50 => positive
        )
        score, components = _trend_score(features)
        assert score > 0
        assert components["ma_cross"] > 0
        assert components["macd"] > 0
        assert components["rsi"] > 0

    def test_bearish_indicators(self):
        features = _make_features(
            sma_fast=95.0, sma_slow=100.0,   # fast < slow
            macd_hist=-2.0, atr=1.0,          # negative MACD
            rsi=35.0,                          # RSI < 50
        )
        score, components = _trend_score(features)
        assert score < 0
        assert components["ma_cross"] < 0
        assert components["macd"] < 0
        assert components["rsi"] < 0

    def test_rsi_overbought_caution(self):
        features = _make_features(rsi=85.0)
        _, components = _trend_score(features)
        assert components["rsi"] == -0.5  # overbought warning

    def test_rsi_oversold_caution(self):
        features = _make_features(rsi=15.0)
        _, components = _trend_score(features)
        assert components["rsi"] == 0.5  # oversold for shorts

    def test_all_none_features(self):
        features = _make_features()
        score, components = _trend_score(features)
        assert score == 0.0
        assert components["ma_cross"] == 0.0
        assert components["macd"] == 0.0
        assert components["rsi"] == 0.0


# ── Range score ──────────────────────────────────────────────

class TestRangeScore:
    def test_oversold_bb(self):
        features = _make_features(bb_pct=0.1, rsi=25.0, macd_hist=0.5, atr=1.0)
        score, components = _range_score(features)
        assert components["bb_pct"] > 0  # bb_pct low => buy signal
        assert components["rsi"] > 0     # RSI < 30 => buy
        assert score > 0

    def test_overbought_bb(self):
        features = _make_features(bb_pct=0.9, rsi=75.0, macd_hist=-0.5, atr=1.0)
        score, components = _range_score(features)
        assert components["bb_pct"] < 0  # bb_pct high => sell
        assert components["rsi"] < 0     # RSI > 70 => sell
        assert score < 0


# ── combine_signals ──────────────────────────────────────────

class TestCombineSignals:
    def test_high_vol_always_flat(self):
        features = _make_features(
            sma_fast=110.0, sma_slow=100.0,
            macd_hist=5.0, atr=1.0, rsi=70.0,
        )
        sig = combine_signals(features, Regime.HIGH_VOL)
        assert sig.direction == 0
        assert sig.strength == 0.0
        assert sig.components == {"reason": "high_vol"}

    def test_trending_up_long(self):
        features = _make_features(
            sma_fast=110.0, sma_slow=100.0,  # 10% diff => ma_score clamped to 1.0
            macd_hist=3.0, atr=1.0,
            rsi=60.0,
        )
        sig = combine_signals(features, Regime.TRENDING_UP, trend_threshold=0.3)
        assert sig.direction == 1
        assert sig.strength > 0.3

    def test_trending_down_short(self):
        features = _make_features(
            sma_fast=90.0, sma_slow=100.0,
            macd_hist=-3.0, atr=1.0,
            rsi=35.0,
        )
        sig = combine_signals(features, Regime.TRENDING_DOWN, trend_threshold=0.3)
        assert sig.direction == -1
        assert sig.strength > 0.3

    def test_below_threshold_flat(self):
        features = _make_features(
            sma_fast=100.1, sma_slow=100.0,
            macd_hist=0.01, atr=1.0,
            rsi=50.0,
        )
        sig = combine_signals(features, Regime.TRENDING_UP, trend_threshold=0.5)
        assert sig.direction == 0

    def test_ranging_uses_range_score(self):
        features = _make_features(bb_pct=0.05, rsi=20.0, macd_hist=1.0, atr=1.0)
        sig = combine_signals(features, Regime.RANGING, range_threshold=0.2)
        assert sig.direction == 1  # strong oversold => long

    def test_strength_clamped_0_1(self):
        features = _make_features(
            sma_fast=200.0, sma_slow=100.0,  # extreme diff
            macd_hist=100.0, atr=1.0,
            rsi=60.0,
        )
        sig = combine_signals(features, Regime.TRENDING_UP, trend_threshold=0.01)
        assert 0.0 <= sig.strength <= 1.0
