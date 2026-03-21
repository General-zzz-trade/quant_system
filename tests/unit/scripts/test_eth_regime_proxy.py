"""Tests for ETHRegimeProxy — BTC regime labels for ETH param routing."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from regime.eth_regime_proxy import (
    ETHRegimeProxy,
    ETHRegimeProxyConfig,
    ETH_REGIME_PARAMS,
    ETH_FALLBACK,
)
from regime.composite import CompositeRegimeLabel


def _make_mock_detector(trend: str, vol: str):
    """Create a mock CompositeRegimeDetector that returns fixed labels."""
    from regime.base import RegimeLabel

    detector = MagicMock()
    composite = CompositeRegimeLabel(trend=trend, vol=vol)
    label = RegimeLabel(
        name="composite",
        ts=datetime.now(timezone.utc),
        value=f"{trend}|{vol}",
        score=0.5,
        meta={"composite": composite},
    )
    detector.detect.return_value = label
    return detector


class TestETHRegimeProxy:
    def test_default_fallback_params(self):
        proxy = ETHRegimeProxy()
        params = proxy.get_eth_params()
        assert params == ETH_FALLBACK

    def test_update_btc_regime_strong_up(self):
        detector = _make_mock_detector("strong_up", "low_vol")
        proxy = ETHRegimeProxy(btc_regime_detector=detector)
        proxy.update_btc_regime(vol=0.01, adx=35.0, close_vs_ma=0.03)
        params = proxy.get_eth_params()
        assert params.position_scale >= 0.8
        assert params.deadzone <= 0.5

    def test_update_btc_regime_crisis(self):
        detector = _make_mock_detector("ranging", "crisis")
        proxy = ETHRegimeProxy(btc_regime_detector=detector)
        proxy.update_btc_regime(vol=0.1, adx=10.0, close_vs_ma=-0.05)
        params = proxy.get_eth_params()
        assert params.position_scale <= 0.4

    def test_position_scale_gate_interface(self):
        detector = _make_mock_detector("weak_up", "normal_vol")
        proxy = ETHRegimeProxy(btc_regime_detector=detector)
        proxy.update_btc_regime(vol=0.02, adx=30.0, close_vs_ma=0.02)
        scale = proxy.position_scale("ETHUSDT")
        assert 0.0 < scale <= 1.0

    def test_disabled_returns_1(self):
        detector = _make_mock_detector("ranging", "crisis")
        proxy = ETHRegimeProxy(
            ETHRegimeProxyConfig(enabled=False),
            btc_regime_detector=detector,
        )
        proxy.update_btc_regime(vol=0.1)
        assert proxy.position_scale() == 1.0

    def test_current_regime_none_initially(self):
        proxy = ETHRegimeProxy()
        assert proxy.current_regime is None

    def test_current_regime_after_update(self):
        detector = _make_mock_detector("strong_up", "normal_vol")
        proxy = ETHRegimeProxy(btc_regime_detector=detector)
        proxy.update_btc_regime(vol=0.02)
        assert proxy.current_regime is not None
        assert proxy.current_regime.trend == "strong_up"
        assert proxy.current_regime.vol == "normal_vol"

    def test_stats(self):
        detector = _make_mock_detector("weak_down", "high_vol")
        proxy = ETHRegimeProxy(btc_regime_detector=detector)
        proxy.update_btc_regime(vol=0.02)
        stats = proxy.stats
        assert stats["enabled"]
        assert stats["updates"] == 1

    def test_eth_crisis_less_aggressive(self):
        """ETH crisis scale should be >= BTC crisis."""
        eth_crisis = ETH_REGIME_PARAMS.get(("*", "crisis"))
        from regime.param_router import DEFAULT_PARAMS
        btc_crisis = DEFAULT_PARAMS.get(("*", "crisis"))
        if eth_crisis and btc_crisis:
            assert eth_crisis.position_scale >= btc_crisis.position_scale

    def test_multiple_regime_transitions(self):
        det_bull = _make_mock_detector("strong_up", "low_vol")
        proxy = ETHRegimeProxy(btc_regime_detector=det_bull)
        proxy.update_btc_regime(vol=0.01)
        bull_scale = proxy.get_eth_params().position_scale

        det_crisis = _make_mock_detector("ranging", "crisis")
        proxy._btc_detector = det_crisis
        proxy.update_btc_regime(vol=0.1)
        crisis_scale = proxy.get_eth_params().position_scale

        assert bull_scale > crisis_scale

    def test_detector_returns_none(self):
        detector = MagicMock()
        detector.detect.return_value = None
        proxy = ETHRegimeProxy(btc_regime_detector=detector)
        proxy.update_btc_regime(vol=0.02)
        assert proxy.get_eth_params() == ETH_FALLBACK
