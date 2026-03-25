"""Tests for CompositeRegimeDetector and CompositeRegimeLabel."""

from datetime import datetime

import pytest

from strategy.regime.composite import CompositeRegimeDetector, CompositeRegimeLabel
from strategy.regime.trend import TrendRegimeDetector
from strategy.regime.volatility import VolatilityRegimeDetector


def _ts():
    return datetime(2026, 1, 1)


class TestCompositeRegimeLabel:
    def test_is_favorable_strong_up_low_vol(self):
        label = CompositeRegimeLabel(vol="low_vol", trend="strong_up")
        assert label.is_favorable is True
        assert label.is_crisis is False

    def test_is_favorable_strong_down_normal_vol(self):
        label = CompositeRegimeLabel(vol="normal_vol", trend="strong_down")
        assert label.is_favorable is True

    def test_not_favorable_weak_trend(self):
        label = CompositeRegimeLabel(vol="low_vol", trend="weak_up")
        assert label.is_favorable is False

    def test_not_favorable_high_vol(self):
        label = CompositeRegimeLabel(vol="high_vol", trend="strong_up")
        assert label.is_favorable is False

    def test_is_crisis(self):
        label = CompositeRegimeLabel(vol="crisis", trend="strong_down")
        assert label.is_crisis is True
        assert label.is_favorable is False

    def test_frozen(self):
        label = CompositeRegimeLabel(vol="low_vol", trend="ranging")
        with pytest.raises(AttributeError):
            label.vol = "high_vol"


class TestCompositeRegimeDetector:
    def test_returns_none_when_both_none(self):
        detector = CompositeRegimeDetector()
        # No relevant features -> both sub-detectors return None
        result = detector.detect(symbol="ETHUSDT", ts=_ts(), features={})
        assert result is None

    def test_defaults_when_vol_only(self):
        """When only vol detector fires, trend defaults to ranging."""
        vol_det = VolatilityRegimeDetector(window=20, min_bars=5)
        trend_det = TrendRegimeDetector()
        detector = CompositeRegimeDetector(vol_detector=vol_det, trend_detector=trend_det)

        # Seed vol history
        for i in range(10):
            detector.detect(
                symbol="ETHUSDT",
                ts=_ts(),
                features={"parkinson_vol": 0.01 + i * 0.001},
            )
        result = detector.detect(
            symbol="ETHUSDT",
            ts=_ts(),
            features={"parkinson_vol": 0.015},
        )
        assert result is not None
        assert "ranging" in result.value
        composite = result.meta["composite"]
        assert composite.trend == "ranging"

    def test_defaults_when_trend_only(self):
        """When only trend detector fires, vol defaults to normal_vol."""
        detector = CompositeRegimeDetector()
        result = detector.detect(
            symbol="ETHUSDT",
            ts=_ts(),
            features={"close_vs_ma20": 0.02, "close_vs_ma50": 0.01, "adx_14": 30.0},
        )
        assert result is not None
        composite = result.meta["composite"]
        assert composite.vol == "normal_vol"
        assert composite.trend == "strong_up"

    def test_combined_value_format(self):
        detector = CompositeRegimeDetector()
        result = detector.detect(
            symbol="ETHUSDT",
            ts=_ts(),
            features={"close_vs_ma20": 0.02, "close_vs_ma50": 0.01, "adx_14": 30.0},
        )
        assert result is not None
        assert "|" in result.value
        assert result.value == "strong_up|normal_vol"

    def test_crisis_weighted_score(self):
        """Crisis regime should weight vol_score more heavily."""
        vol_det = VolatilityRegimeDetector(window=100, min_bars=5)
        detector = CompositeRegimeDetector(vol_detector=vol_det)

        # Seed with spread of vol and low vov so the 95th pctl is well-defined
        for i in range(50):
            detector.detect(
                symbol="ETHUSDT",
                ts=_ts(),
                features={"parkinson_vol": 0.01 + i * 0.001, "vol_of_vol": 0.001 + i * 0.0001},
            )
        # Spike vov well above 95th percentile to trigger crisis
        result = detector.detect(
            symbol="ETHUSDT",
            ts=_ts(),
            features={"parkinson_vol": 0.02, "vol_of_vol": 1.0},
        )
        assert result is not None
        composite = result.meta["composite"]
        assert composite.is_crisis

    def test_meta_contains_sub_labels(self):
        detector = CompositeRegimeDetector()
        result = detector.detect(
            symbol="ETHUSDT",
            ts=_ts(),
            features={"close_vs_ma20": 0.02, "close_vs_ma50": 0.01, "adx_14": 30.0},
        )
        assert result is not None
        assert "composite" in result.meta
        assert "vol_label" in result.meta
        assert "trend_label" in result.meta
        assert "is_favorable" in result.meta
        assert "is_crisis" in result.meta
