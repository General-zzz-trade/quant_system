# tests/unit/features/test_adx_enriched.py
"""Tests for ADX(14) in EnrichedFeatureComputer."""
from __future__ import annotations

import pytest

from features.enriched_computer import (
    ENRICHED_FEATURE_NAMES,
    EnrichedFeatureComputer,
    _ADXTracker,
)


class TestADXInFeatureNames:
    def test_adx_14_in_enriched_names(self):
        assert "adx_14" in ENRICHED_FEATURE_NAMES


class TestADXTracker:
    def test_constant_price_adx_zero(self):
        """ADX should be near 0 for constant prices (no trend)."""
        tracker = _ADXTracker(period=14)
        for _ in range(60):
            tracker.push(high=100.0, low=100.0, close=100.0)
        val = tracker.value
        assert val is not None
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_warmup_returns_none(self):
        """ADX needs 2*period bars warmup, should return None before that."""
        tracker = _ADXTracker(period=14)
        for i in range(20):
            tracker.push(high=100.0 + i, low=99.0 + i, close=100.0 + i)
        # Not enough bars for ADX initialization
        assert tracker.value is None

    def test_strong_uptrend_high_adx(self):
        """ADX should be high (>25) during a persistent uptrend."""
        tracker = _ADXTracker(period=14)
        for i in range(80):
            base = 100.0 + i * 2.0  # Strong uptrend
            tracker.push(high=base + 1.0, low=base - 0.5, close=base)
        val = tracker.value
        assert val is not None
        assert val > 25.0, f"ADX should be >25 for strong trend, got {val}"

    def test_strong_downtrend_high_adx(self):
        """ADX measures trend strength regardless of direction."""
        tracker = _ADXTracker(period=14)
        for i in range(80):
            base = 200.0 - i * 2.0  # Strong downtrend
            tracker.push(high=base + 0.5, low=base - 1.0, close=base)
        val = tracker.value
        assert val is not None
        assert val > 25.0, f"ADX should be >25 for strong downtrend, got {val}"

    def test_choppy_market_low_adx(self):
        """ADX should be lower for choppy, oscillating markets."""
        tracker = _ADXTracker(period=14)
        for i in range(80):
            # Oscillate up and down with equal amplitude
            offset = 5.0 if i % 2 == 0 else -5.0
            base = 100.0 + offset
            tracker.push(high=base + 2.0, low=base - 2.0, close=base)
        val = tracker.value
        assert val is not None
        # Choppy markets should have moderate-to-low ADX
        assert val < 50.0, f"ADX should be <50 for choppy market, got {val}"

    def test_adx_value_range(self):
        """ADX should always be in [0, 100]."""
        tracker = _ADXTracker(period=14)
        for i in range(60):
            base = 100.0 + i
            tracker.push(high=base + 3.0, low=base - 1.0, close=base)
        val = tracker.value
        assert val is not None
        assert 0.0 <= val <= 100.0


class TestADXInEnrichedComputer:
    def test_adx_available_after_warmup(self):
        """EnrichedFeatureComputer should output adx_14 after warmup."""
        comp = EnrichedFeatureComputer()
        for i in range(65):
            base = 100.0 + i * 0.5
            feats = comp.on_bar(
                "ETHUSDT",
                close=base,
                high=base + 1.0,
                low=base - 0.5,
                volume=1000.0,
                open_=base - 0.2,
            )
        assert "adx_14" in feats
        assert feats["adx_14"] is not None

    def test_adx_none_during_warmup(self):
        """ADX should be None during warmup period."""
        comp = EnrichedFeatureComputer()
        feats = comp.on_bar(
            "ETHUSDT", close=100.0, high=101.0, low=99.0, volume=1000.0,
        )
        assert "adx_14" in feats
        assert feats["adx_14"] is None

    def test_adx_matches_trend(self):
        """ADX should be higher in trending vs non-trending markets."""
        comp_trend = EnrichedFeatureComputer()
        comp_flat = EnrichedFeatureComputer()

        for i in range(80):
            # Trending
            base_t = 100.0 + i * 2.0
            feats_t = comp_trend.on_bar(
                "ETHUSDT", close=base_t, high=base_t + 1.0,
                low=base_t - 0.5, volume=1000.0,
            )
            # Flat
            feats_f = comp_flat.on_bar(
                "ETHUSDT", close=100.0, high=100.5,
                low=99.5, volume=1000.0,
            )

        adx_trend = feats_t["adx_14"]
        adx_flat = feats_f["adx_14"]
        assert adx_trend is not None
        # Trending should have meaningfully higher ADX
        if adx_flat is not None and adx_flat > 0:
            assert adx_trend > adx_flat
