"""Tests for the ADX-based TrendRegimeDetector."""

from datetime import datetime

import pytest

from regime.trend import TrendRegimeDetector


@pytest.fixture
def detector():
    return TrendRegimeDetector(adx_strong=25.0, adx_ranging=15.0)


def _ts():
    return datetime(2026, 1, 1)


def _features(ma20, ma50, adx=None):
    f = {"close_vs_ma20": ma20, "close_vs_ma50": ma50}
    if adx is not None:
        f["adx_14"] = adx
    return f


class TestTrendRegimeDetector:
    def test_strong_up(self, detector):
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.02, 0.01, adx=30.0)
        )
        assert result is not None
        assert result.value == "strong_up"

    def test_weak_up(self, detector):
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.02, 0.01, adx=20.0)
        )
        assert result is not None
        assert result.value == "weak_up"

    def test_strong_down(self, detector):
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(-0.02, -0.01, adx=30.0)
        )
        assert result is not None
        assert result.value == "strong_down"

    def test_weak_down(self, detector):
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(-0.02, -0.01, adx=20.0)
        )
        assert result is not None
        assert result.value == "weak_down"

    def test_ranging_disagreement(self, detector):
        """MAs disagree on direction -> ranging."""
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.02, -0.01, adx=30.0)
        )
        assert result is not None
        assert result.value == "ranging"

    def test_ranging_low_adx(self, detector):
        """ADX below 15 -> ranging even if MAs agree."""
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.02, 0.01, adx=10.0)
        )
        assert result is not None
        assert result.value == "ranging"

    def test_missing_adx_defaults_to_zero(self, detector):
        """No adx_14 -> treated as 0 -> ranging."""
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.02, 0.01)
        )
        assert result is not None
        assert result.value == "ranging"

    def test_returns_none_on_missing_ma(self, detector):
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features={"close_vs_ma20": 0.02}
        )
        assert result is None

    def test_returns_none_on_nan(self, detector):
        result = detector.detect(
            symbol="ETHUSDT",
            ts=_ts(),
            features=_features(float("nan"), 0.01, adx=30.0),
        )
        assert result is None

    def test_meta_contains_inputs(self, detector):
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.02, 0.01, adx=30.0)
        )
        assert result is not None
        assert result.meta["close_vs_ma20"] == 0.02
        assert result.meta["close_vs_ma50"] == 0.01
        assert result.meta["adx_14"] == 30.0

    def test_score_capped_at_1(self, detector):
        result = detector.detect(
            symbol="ETHUSDT", ts=_ts(), features=_features(0.05, 0.03, adx=80.0)
        )
        assert result is not None
        assert result.score <= 1.0
