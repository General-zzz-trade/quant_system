"""Parity test: Rust regime detector vs Python.

Verifies that RustCompositeRegimeDetector and RustRegimeParamRouter
produce identical results to their Python equivalents for all key
regime scenarios.
"""
from __future__ import annotations

import pytest
from datetime import datetime

from regime.volatility import VolatilityRegimeDetector
from regime.trend import TrendRegimeDetector
from regime.composite import CompositeRegimeDetector, CompositeRegimeLabel
from regime.param_router import RegimeParamRouter, RegimeParams, DEFAULT_PARAMS

try:
    from _quant_hotpath import RustCompositeRegimeDetector, RustRegimeParamRouter
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not built")


def make_base_features(**overrides):
    base = {
        "parkinson_vol": 0.01,
        "vol_of_vol": 0.001,
        "bb_width_20": 0.02,
        "close_vs_ma20": 0.005,
        "close_vs_ma50": 0.003,
        "adx_14": 30.0,
    }
    base.update(overrides)
    return base


class TestVolLabelParity:
    """Same feature sequences → same vol labels between Rust and Python."""

    def test_vol_normal(self):
        rust = RustCompositeRegimeDetector()
        py_vol = VolatilityRegimeDetector()
        ts = datetime.utcnow()
        symbol = "ETHUSDT"

        features = make_base_features()
        rust_result = py_result = None

        for i in range(35):
            rust_result = rust.detect(dict(features))
            py_result = py_vol.detect(symbol=symbol, ts=ts, features=features)

        assert rust_result is not None
        assert py_result is not None
        assert rust_result.vol_label == py_result.value  # parity assertion added

    def test_low_vol_label_parity(self):
        """Fill with very low parkinson_vol → both classify as low_vol."""
        rust = RustCompositeRegimeDetector()
        py_vol = VolatilityRegimeDetector()
        ts = datetime.utcnow()
        symbol = "ETHUSDT"

        # Push 35 bars of high vol first to create a high baseline
        high_vol_features = make_base_features(parkinson_vol=0.1)
        for _ in range(35):
            rust.detect(dict(high_vol_features))
            py_vol.detect(symbol=symbol, ts=ts, features=high_vol_features)

        # Now test with very low vol — it should be below p25 in both
        low_vol_features = make_base_features(parkinson_vol=0.001)
        r = rust.detect(dict(low_vol_features))
        py_r = py_vol.detect(symbol=symbol, ts=ts, features=low_vol_features)

        assert r is not None
        assert py_r is not None
        assert r.vol_label == py_r.value  # Both should be "low_vol"


class TestTrendLabelParity:
    def test_strong_up_parity(self):
        rust = RustCompositeRegimeDetector()
        ts = datetime.utcnow()
        symbol = "ETHUSDT"
        py_trend = TrendRegimeDetector()

        features = make_base_features(close_vs_ma20=0.01, close_vs_ma50=0.01, adx_14=30.0)

        for _ in range(35):
            rust.detect(dict(features))
            py_trend.detect(symbol=symbol, ts=ts, features=features)

        r = rust.detect(dict(features))
        py_r = py_trend.detect(symbol=symbol, ts=ts, features=features)

        assert r is not None
        assert py_r is not None
        assert r.trend_label == py_r.value  # Both "strong_up"

    def test_ranging_parity(self):
        rust = RustCompositeRegimeDetector()
        ts = datetime.utcnow()
        symbol = "ETHUSDT"
        py_trend = TrendRegimeDetector()

        # MAs disagree → ranging
        features = make_base_features(close_vs_ma20=0.01, close_vs_ma50=-0.01, adx_14=20.0)

        for _ in range(35):
            rust.detect(dict(features))
            py_trend.detect(symbol=symbol, ts=ts, features=features)

        r = rust.detect(dict(features))
        py_r = py_trend.detect(symbol=symbol, ts=ts, features=features)

        assert r is not None
        assert r.trend_label == "ranging"
        assert py_r is not None
        assert py_r.value == "ranging"

    def test_strong_down_parity(self):
        rust = RustCompositeRegimeDetector()
        ts = datetime.utcnow()
        symbol = "ETHUSDT"
        py_trend = TrendRegimeDetector()

        features = make_base_features(close_vs_ma20=-0.01, close_vs_ma50=-0.01, adx_14=30.0)

        for _ in range(35):
            rust.detect(dict(features))
            py_trend.detect(symbol=symbol, ts=ts, features=features)

        r = rust.detect(dict(features))
        py_r = py_trend.detect(symbol=symbol, ts=ts, features=features)

        assert r is not None
        assert py_r is not None
        assert r.trend_label == py_r.value  # Both "strong_down"

    def test_weak_up_parity(self):
        rust = RustCompositeRegimeDetector()
        ts = datetime.utcnow()
        symbol = "ETHUSDT"
        py_trend = TrendRegimeDetector()

        # ADX between ranging (15) and strong (25) → weak_up
        features = make_base_features(close_vs_ma20=0.01, close_vs_ma50=0.01, adx_14=20.0)

        for _ in range(35):
            rust.detect(dict(features))
            py_trend.detect(symbol=symbol, ts=ts, features=features)

        r = rust.detect(dict(features))
        py_r = py_trend.detect(symbol=symbol, ts=ts, features=features)

        assert r is not None
        assert py_r is not None
        assert r.trend_label == py_r.value  # Both "weak_up"


class TestParamRouterParity:
    """All 13 DEFAULT_PARAMS entries + wildcards route identically in Rust and Python."""

    def test_all_entries(self):
        rust_router = RustRegimeParamRouter()
        py_router = RegimeParamRouter()

        for (trend, vol), py_params in DEFAULT_PARAMS.items():
            if trend == "*":
                # Test with a concrete trend that doesn't have an exact match
                test_trend = "strong_up" if vol == "crisis" else "ranging"
            else:
                test_trend = trend

            regime = CompositeRegimeLabel(vol=vol, trend=test_trend)
            py_result = py_router.route(regime)
            rust_result = rust_router.route(test_trend, vol)

            assert rust_result.deadzone == pytest.approx(py_result.deadzone, abs=1e-9), \
                f"deadzone mismatch for ({test_trend}, {vol})"
            assert rust_result.min_hold == py_result.min_hold, \
                f"min_hold mismatch for ({test_trend}, {vol})"
            assert rust_result.max_hold == py_result.max_hold, \
                f"max_hold mismatch for ({test_trend}, {vol})"
            assert rust_result.position_scale == pytest.approx(py_result.position_scale, abs=1e-9), \
                f"position_scale mismatch for ({test_trend}, {vol})"

    def test_wildcard_crisis(self):
        rust_router = RustRegimeParamRouter()
        py_router = RegimeParamRouter()

        regime = CompositeRegimeLabel(vol="crisis", trend="strong_up")
        py_p = py_router.route(regime)
        rust_p = rust_router.route("strong_up", "crisis")

        assert rust_p.deadzone == pytest.approx(py_p.deadzone, abs=1e-9)
        assert rust_p.min_hold == py_p.min_hold
        assert rust_p.max_hold == py_p.max_hold
        assert rust_p.position_scale == pytest.approx(py_p.position_scale, abs=1e-9)

    def test_wildcard_high_vol(self):
        rust_router = RustRegimeParamRouter()
        py_router = RegimeParamRouter()

        # strong_up + high_vol → no exact match → wildcard ("*", "high_vol")
        regime = CompositeRegimeLabel(vol="high_vol", trend="strong_up")
        py_p = py_router.route(regime)
        rust_p = rust_router.route("strong_up", "high_vol")

        assert rust_p.deadzone == pytest.approx(py_p.deadzone, abs=1e-9)
        assert rust_p.min_hold == py_p.min_hold
        assert rust_p.max_hold == py_p.max_hold
        assert rust_p.position_scale == pytest.approx(py_p.position_scale, abs=1e-9)

    def test_fallback(self):
        rust_router = RustRegimeParamRouter()
        py_router = RegimeParamRouter()

        regime = CompositeRegimeLabel(vol="unknown_vol", trend="unknown_trend")
        py_p = py_router.route(regime)
        rust_p = rust_router.route("unknown_trend", "unknown_vol")

        assert rust_p.deadzone == pytest.approx(py_p.deadzone, abs=1e-9)
        assert rust_p.min_hold == py_p.min_hold
        assert rust_p.max_hold == py_p.max_hold
        assert rust_p.position_scale == pytest.approx(py_p.position_scale, abs=1e-9)

    def test_exact_strong_down_normal_vol(self):
        rust_router = RustRegimeParamRouter()
        py_router = RegimeParamRouter()

        regime = CompositeRegimeLabel(vol="normal_vol", trend="strong_down")
        py_p = py_router.route(regime)
        rust_p = rust_router.route("strong_down", "normal_vol")

        assert rust_p.deadzone == pytest.approx(py_p.deadzone, abs=1e-9)
        assert rust_p.min_hold == py_p.min_hold

    def test_exact_ranging_high_vol(self):
        rust_router = RustRegimeParamRouter()
        py_router = RegimeParamRouter()

        regime = CompositeRegimeLabel(vol="high_vol", trend="ranging")
        py_p = py_router.route(regime)
        rust_p = rust_router.route("ranging", "high_vol")

        # ranging+high_vol has exact entry → deadzone=1.5, min_hold=24, max_hold=48
        assert rust_p.deadzone == pytest.approx(py_p.deadzone, abs=1e-9)
        assert rust_p.min_hold == py_p.min_hold
        assert rust_p.max_hold == py_p.max_hold


class TestCompositeDetectorParity:
    def test_composite_detector_uses_rust(self):
        """CompositeRegimeDetector should use Rust backend when no custom detectors."""
        det = CompositeRegimeDetector()
        assert det._rust_detector is not None

    def test_composite_fallback_on_custom_vol_detector(self):
        """When custom vol_detector is injected, Python path is used."""
        custom_vol = VolatilityRegimeDetector()
        det = CompositeRegimeDetector(vol_detector=custom_vol)
        assert det._rust_detector is None

    def test_composite_fallback_on_custom_trend_detector(self):
        """When custom trend_detector is injected, Python path is used."""
        custom_trend = TrendRegimeDetector()
        det = CompositeRegimeDetector(trend_detector=custom_trend)
        assert det._rust_detector is None

    def test_insufficient_bars_returns_none(self):
        rust = RustCompositeRegimeDetector()
        features = make_base_features()
        result = None
        for _ in range(5):
            result = rust.detect(dict(features))
        assert result is None  # < 30 bars

    def test_missing_parkinson_vol_returns_none(self):
        rust = RustCompositeRegimeDetector()
        features = {"close_vs_ma20": 0.01, "close_vs_ma50": 0.01}
        result = rust.detect(features)
        assert result is None

    def test_composite_result_has_correct_fields(self):
        """After enough bars, result has all expected fields."""
        rust = RustCompositeRegimeDetector()
        features = make_base_features(close_vs_ma20=0.01, close_vs_ma50=0.01, adx_14=30.0)
        result = None
        for _ in range(35):
            result = rust.detect(dict(features))
        assert result is not None
        assert isinstance(result.value, str)
        assert "|" in result.value
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert result.trend_label in ("strong_up", "weak_up", "ranging", "weak_down", "strong_down")
        assert result.vol_label in ("low_vol", "normal_vol", "high_vol", "crisis")
        assert isinstance(result.is_favorable, bool)
        assert isinstance(result.is_crisis, bool)

    def test_composite_detect_returns_regime_label(self):
        """CompositeRegimeDetector.detect() returns RegimeLabel with Rust backend."""
        from regime.base import RegimeLabel
        det = CompositeRegimeDetector()
        ts = datetime.utcnow()
        symbol = "ETHUSDT"
        features = make_base_features()
        result = None
        for _ in range(35):
            result = det.detect(symbol=symbol, ts=ts, features=features)
        assert result is not None
        assert isinstance(result, RegimeLabel)
        assert result.name == "composite"
        assert "|" in result.value
        assert result.meta is not None
        assert "composite" in result.meta
        assert isinstance(result.meta["composite"], CompositeRegimeLabel)

    def test_param_router_uses_rust(self):
        """RegimeParamRouter should use Rust backend when no custom params."""
        router = RegimeParamRouter()
        assert router._rust_router is not None

    def test_param_router_fallback_on_custom_params(self):
        """When custom params are injected, Python path is used."""
        custom = {("ranging", "low_vol"): RegimeParams(1.0, 24, 96, 0.5)}
        router = RegimeParamRouter(params=custom)
        assert router._rust_router is None

    def test_regime_label_composite_parity(self):
        """Full round-trip: CompositeRegimeDetector (Rust) matches value/trend/vol."""
        det_rust = CompositeRegimeDetector()  # uses Rust
        det_py = CompositeRegimeDetector(
            vol_detector=VolatilityRegimeDetector(),
            trend_detector=TrendRegimeDetector(),
        )  # forces Python path

        ts = datetime.utcnow()
        symbol = "ETHUSDT"
        features = make_base_features(
            parkinson_vol=0.01, close_vs_ma20=0.01, close_vs_ma50=0.01, adx_14=30.0
        )

        result_rust = result_py = None
        for _ in range(35):
            result_rust = det_rust.detect(symbol=symbol, ts=ts, features=features)
            result_py = det_py.detect(symbol=symbol, ts=ts, features=features)

        assert result_rust is not None
        assert result_py is not None

        # Both should produce same composite structure
        composite_rust = result_rust.meta["composite"]
        composite_py = result_py.meta["composite"]

        assert composite_rust.vol == composite_py.vol
        assert composite_rust.trend == composite_py.trend
        assert result_rust.value == result_py.value

        # Score parity: Rust and Python composite scores must agree
        assert result_rust.score == pytest.approx(result_py.score, rel=1e-6)
