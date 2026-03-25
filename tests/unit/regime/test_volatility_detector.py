"""Tests for the percentile-based VolatilityRegimeDetector."""

from datetime import datetime

import pytest

from strategy.regime.volatility import VolatilityRegimeDetector


@pytest.fixture
def detector():
    return VolatilityRegimeDetector(window=100, min_bars=10)


def _ts():
    return datetime(2026, 1, 1)


def _features(parkinson_vol, vol_of_vol=None, bb_width_20=None):
    f = {"parkinson_vol": parkinson_vol}
    if vol_of_vol is not None:
        f["vol_of_vol"] = vol_of_vol
    if bb_width_20 is not None:
        f["bb_width_20"] = bb_width_20
    return f


class TestVolatilityRegimeDetector:
    def test_returns_none_before_min_bars(self, detector):
        """Should return None until min_bars of history accumulated."""
        for i in range(9):
            result = detector.detect(
                symbol="ETHUSDT", ts=_ts(), features=_features(0.01 + i * 0.001)
            )
            assert result is None

    def test_returns_label_after_min_bars(self, detector):
        for i in range(10):
            detector.detect(
                symbol="ETHUSDT", ts=_ts(), features=_features(0.01 + i * 0.001)
            )
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.012)
        )
        assert result is not None
        assert result.name == "volatility"

    def test_low_vol_classification(self, detector):
        """Values well below the bulk should classify as low_vol."""
        # Seed with a range of values
        for i in range(50):
            detector.detect(
                symbol="ETHUSDT", ts=_ts(), features=_features(0.01 + i * 0.001)
            )
        # Very low value
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.001)
        )
        assert result is not None
        assert result.value == "low_vol"

    def test_high_vol_classification(self, detector):
        """Values above 75th percentile should classify as high_vol."""
        for i in range(50):
            detector.detect(
                symbol="ETHUSDT", ts=_ts(), features=_features(0.01 + i * 0.001)
            )
        # Very high value
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.1)
        )
        assert result is not None
        assert result.value == "high_vol"

    def test_normal_vol_classification(self, detector):
        """Median value should classify as normal_vol."""
        for i in range(50):
            detector.detect(
                symbol="ETHUSDT", ts=_ts(), features=_features(0.01 + i * 0.001)
            )
        # Median value
        median = 0.01 + 25 * 0.001
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(median)
        )
        assert result is not None
        assert result.value == "normal_vol"

    def test_crisis_detection_via_vol_of_vol(self, detector):
        """Extreme vol_of_vol should trigger crisis classification."""
        # Seed with moderate vol and moderate vov
        for i in range(50):
            detector.detect(
                symbol="ETHUSDT",
                ts=_ts(),
                features=_features(0.02, vol_of_vol=0.005),
            )
        # Spike in vol_of_vol
        result = detector.detect(
            symbol="ETHUSDT",
            ts=_ts(),
            features=_features(0.02, vol_of_vol=0.5),
        )
        assert result is not None
        assert result.value == "crisis"
        assert result.score == 1.0

    def test_none_on_missing_parkinson(self, detector):
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features={"bb_width_20": 0.1}
        )
        assert result is None

    def test_none_on_nan_parkinson(self, detector):
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(float("nan"))
        )
        assert result is None

    def test_meta_contains_percentiles(self, detector):
        for i in range(20):
            detector.detect(
                symbol="ETHUSDT", ts=_ts(), features=_features(0.01 + i * 0.001)
            )
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.02)
        )
        assert result is not None
        assert "p25" in result.meta
        assert "p75" in result.meta
        assert "p95" in result.meta
        assert "bars" in result.meta

    def test_bb_width_in_meta(self, detector):
        for i in range(15):
            detector.detect(
                symbol="ETHUSDT",
                ts=_ts(),
                features=_features(0.01 + i * 0.001, bb_width_20=0.05),
            )
        result = detector.detect(
            symbol="ETHUSDT",
            ts=_ts(),
            features=_features(0.02, bb_width_20=0.06),
        )
        assert result is not None
        assert result.meta["bb_width_20"] == 0.06
