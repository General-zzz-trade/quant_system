"""Tests for strategy/regime/ — CompositeRegimeDetector + ParamRouter."""
from __future__ import annotations

from datetime import datetime

import pytest

from strategy.regime.base import RegimeLabel
from strategy.regime.composite import CompositeRegimeDetector, CompositeRegimeLabel
from strategy.regime.param_router import (
    DEFAULT_PARAMS,
    RegimeParamRouter,
    RegimeParams,
    _FALLBACK,
)
from strategy.regime.trend import TrendRegimeDetector
from strategy.regime.volatility import VolatilityRegimeDetector


NOW = datetime(2026, 3, 25, 12, 0, 0)


# ===========================================================================
# CompositeRegimeLabel
# ===========================================================================

class TestCompositeRegimeLabel:
    def test_favorable_strong_up_low_vol(self):
        label = CompositeRegimeLabel(vol="low_vol", trend="strong_up")
        assert label.is_favorable is True
        assert label.is_crisis is False

    def test_favorable_strong_down_normal_vol(self):
        label = CompositeRegimeLabel(vol="normal_vol", trend="strong_down")
        assert label.is_favorable is True

    def test_not_favorable_weak_trend(self):
        label = CompositeRegimeLabel(vol="low_vol", trend="weak_up")
        assert label.is_favorable is False

    def test_not_favorable_high_vol(self):
        label = CompositeRegimeLabel(vol="high_vol", trend="strong_up")
        assert label.is_favorable is False

    def test_crisis(self):
        label = CompositeRegimeLabel(vol="crisis", trend="strong_down")
        assert label.is_crisis is True
        assert label.is_favorable is False

    def test_frozen(self):
        label = CompositeRegimeLabel(vol="low_vol", trend="ranging")
        with pytest.raises(AttributeError):
            label.vol = "high_vol"  # type: ignore[misc]


# ===========================================================================
# TrendRegimeDetector
# ===========================================================================

class TestTrendRegimeDetector:
    def test_strong_up(self):
        det = TrendRegimeDetector()
        result = det.detect(symbol="BTCUSDT", ts=NOW, features={
            "close_vs_ma20": 0.05, "close_vs_ma50": 0.03, "adx_14": 30.0
        })
        assert result is not None
        assert result.value == "strong_up"

    def test_weak_up(self):
        det = TrendRegimeDetector()
        result = det.detect(symbol="BTCUSDT", ts=NOW, features={
            "close_vs_ma20": 0.05, "close_vs_ma50": 0.03, "adx_14": 20.0
        })
        assert result is not None
        assert result.value == "weak_up"

    def test_strong_down(self):
        det = TrendRegimeDetector()
        result = det.detect(symbol="BTCUSDT", ts=NOW, features={
            "close_vs_ma20": -0.05, "close_vs_ma50": -0.03, "adx_14": 30.0
        })
        assert result is not None
        assert result.value == "strong_down"

    def test_ranging_low_adx(self):
        det = TrendRegimeDetector()
        result = det.detect(symbol="BTCUSDT", ts=NOW, features={
            "close_vs_ma20": 0.05, "close_vs_ma50": 0.03, "adx_14": 10.0
        })
        assert result is not None
        assert result.value == "ranging"

    def test_ranging_disagreement(self):
        det = TrendRegimeDetector()
        result = det.detect(symbol="BTCUSDT", ts=NOW, features={
            "close_vs_ma20": 0.05, "close_vs_ma50": -0.03, "adx_14": 30.0
        })
        assert result is not None
        assert result.value == "ranging"

    def test_missing_features_returns_none(self):
        det = TrendRegimeDetector()
        assert det.detect(symbol="BTCUSDT", ts=NOW, features={}) is None

    def test_nan_features_returns_none(self):
        det = TrendRegimeDetector()
        assert det.detect(symbol="BTCUSDT", ts=NOW, features={
            "close_vs_ma20": float("nan"), "close_vs_ma50": 0.01
        }) is None

    def test_no_adx_defaults_to_zero(self):
        det = TrendRegimeDetector()
        # adx=0 < adx_ranging(15) → ranging
        result = det.detect(symbol="BTCUSDT", ts=NOW, features={
            "close_vs_ma20": 0.05, "close_vs_ma50": 0.03
        })
        assert result is not None
        assert result.value == "ranging"


# ===========================================================================
# VolatilityRegimeDetector
# ===========================================================================

class TestVolatilityRegimeDetector:
    def test_insufficient_bars_returns_none(self):
        det = VolatilityRegimeDetector(min_bars=30)
        result = det.detect(symbol="BTCUSDT", ts=NOW, features={"parkinson_vol": 0.01})
        assert result is None

    def test_low_vol_classification(self):
        det = VolatilityRegimeDetector(window=100, min_bars=5)
        # Push varied samples then a low one
        for i in range(10):
            det.detect(symbol="BTC", ts=NOW, features={"parkinson_vol": 0.01 * (i + 1)})
        result = det.detect(symbol="BTC", ts=NOW, features={"parkinson_vol": 0.001})
        assert result is not None
        assert result.value == "low_vol"

    def test_high_vol_classification(self):
        det = VolatilityRegimeDetector(window=100, min_bars=5)
        for i in range(10):
            det.detect(symbol="BTC", ts=NOW, features={"parkinson_vol": 0.01 * (i + 1)})
        result = det.detect(symbol="BTC", ts=NOW, features={"parkinson_vol": 0.15})
        assert result is not None
        assert result.value == "high_vol"

    def test_missing_parkinson_returns_none(self):
        det = VolatilityRegimeDetector()
        assert det.detect(symbol="BTC", ts=NOW, features={}) is None

    def test_nan_parkinson_returns_none(self):
        det = VolatilityRegimeDetector()
        assert det.detect(symbol="BTC", ts=NOW, features={"parkinson_vol": float("nan")}) is None


# ===========================================================================
# CompositeRegimeDetector (Python fallback)
# ===========================================================================

class TestCompositeRegimeDetector:
    @pytest.fixture()
    def detector(self):
        """Use Python fallback by injecting custom detectors."""
        return CompositeRegimeDetector(
            vol_detector=VolatilityRegimeDetector(window=100, min_bars=5),
            trend_detector=TrendRegimeDetector(),
        )

    def _warm_vol(self, detector, n=10):
        for i in range(n):
            detector.detect(symbol="BTC", ts=NOW, features={
                "parkinson_vol": 0.01 * (i + 1),
                "close_vs_ma20": 0.05,
                "close_vs_ma50": 0.03,
                "adx_14": 30.0,
            })

    def test_composite_returns_label(self, detector):
        self._warm_vol(detector)
        result = detector.detect(symbol="BTC", ts=NOW, features={
            "parkinson_vol": 0.05,
            "close_vs_ma20": 0.05,
            "close_vs_ma50": 0.03,
            "adx_14": 30.0,
        })
        assert result is not None
        assert isinstance(result, RegimeLabel)
        assert "|" in result.value  # composite format: "trend|vol"

    def test_insufficient_data_returns_none(self):
        det = CompositeRegimeDetector(
            vol_detector=VolatilityRegimeDetector(min_bars=999),
            trend_detector=TrendRegimeDetector(),
        )
        # No trend features and insufficient vol bars
        result = det.detect(symbol="BTC", ts=NOW, features={})
        assert result is None

    def test_crisis_regime_high_vol_weight(self):
        """Crisis detection requires vol_of_vol above 95th percentile of history."""
        det = CompositeRegimeDetector(
            vol_detector=VolatilityRegimeDetector(window=100, min_bars=5),
            trend_detector=TrendRegimeDetector(),
        )
        # Build vol and vov history with moderate values
        for i in range(30):
            det.detect(symbol="BTC", ts=NOW, features={
                "parkinson_vol": 0.02,
                "vol_of_vol": 0.01,
                "close_vs_ma20": -0.05,
                "close_vs_ma50": -0.03,
                "adx_14": 30.0,
            })
        # Now push extreme vol_of_vol — well above 95th percentile of 0.01
        result = det.detect(symbol="BTC", ts=NOW, features={
            "parkinson_vol": 0.02,
            "vol_of_vol": 10.0,
            "close_vs_ma20": -0.05,
            "close_vs_ma50": -0.03,
            "adx_14": 30.0,
        })
        assert result is not None
        meta = result.meta
        assert meta["is_crisis"] is True


# ===========================================================================
# RegimeParamRouter
# ===========================================================================

class TestRegimeParamRouter:
    def test_strong_up_low_vol(self):
        router = RegimeParamRouter(params=DEFAULT_PARAMS)
        p = router.route(CompositeRegimeLabel(vol="low_vol", trend="strong_up"))
        assert p.deadzone == 0.3
        assert p.position_scale == 1.0

    def test_crisis_wildcard(self):
        router = RegimeParamRouter(params=DEFAULT_PARAMS)
        p = router.route(CompositeRegimeLabel(vol="crisis", trend="strong_up"))
        assert p.deadzone == 2.5
        assert p.position_scale == 0.1

    def test_ranging_high_vol(self):
        router = RegimeParamRouter(params=DEFAULT_PARAMS)
        p = router.route(CompositeRegimeLabel(vol="high_vol", trend="ranging"))
        assert p.deadzone == 1.2
        assert p.position_scale == 0.4

    def test_unknown_regime_returns_fallback(self):
        router = RegimeParamRouter(params=DEFAULT_PARAMS)
        p = router.route(CompositeRegimeLabel(vol="unknown", trend="unknown"))
        assert p == _FALLBACK

    def test_custom_fallback(self):
        custom_fb = RegimeParams(deadzone=5.0, min_hold=100, max_hold=200, position_scale=0.01)
        router = RegimeParamRouter(params={}, fallback=custom_fb)
        p = router.route(CompositeRegimeLabel(vol="low_vol", trend="strong_up"))
        assert p == custom_fb

    def test_default_params_all_have_valid_fields(self):
        for key, p in DEFAULT_PARAMS.items():
            assert p.deadzone > 0
            assert p.min_hold > 0
            assert p.max_hold >= p.min_hold
            assert 0.0 <= p.position_scale <= 1.0

    def test_regime_params_frozen(self):
        p = RegimeParams(deadzone=1.0, min_hold=10, max_hold=20, position_scale=0.5)
        with pytest.raises(AttributeError):
            p.deadzone = 2.0  # type: ignore[misc]

    def test_wildcard_high_vol_match(self):
        router = RegimeParamRouter(params=DEFAULT_PARAMS)
        # "weak_down" + "high_vol" not in table directly but ("*", "high_vol") is
        p = router.route(CompositeRegimeLabel(vol="high_vol", trend="weak_down"))
        # Should match ("*", "high_vol")
        assert p.deadzone == 1.5
        assert p.position_scale == 0.3
