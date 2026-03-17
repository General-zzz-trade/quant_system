# tests/unit/scripts/test_alpha_runner_composite_regime.py
"""Tests for AlphaRunner composite regime integration."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock

from regime.base import RegimeLabel
from regime.composite import CompositeRegimeLabel

# Mock _quant_hotpath before importing AlphaRunner (it imports at module init)
_mock_hotpath = MagicMock()
if "_quant_hotpath" not in sys.modules:
    sys.modules["_quant_hotpath"] = _mock_hotpath

from scripts.ops.alpha_runner import AlphaRunner  # noqa: E402


# Minimal model_info for constructing AlphaRunner without real models
_MINIMAL_MODEL_INFO = {
    "model": MagicMock(predict=MagicMock(return_value=0.0)),
    "features": ["close", "volume"],
    "horizon_models": [],
    "lgbm_xgb_weight": 0.5,
    "config": {"version": "test_v1"},
    "deadzone": 0.3,
    "min_hold": 18,
    "max_hold": 96,
    "zscore_window": 720,
    "zscore_warmup": 180,
}


class _FakeAdapter:
    def get_ticker(self, symbol):
        return {}


def _build_runner(use_composite=False, symbol="ETHUSDT"):
    info = dict(_MINIMAL_MODEL_INFO)
    if use_composite:
        info["use_composite_regime"] = True
    return AlphaRunner(
        adapter=_FakeAdapter(),
        model_info=info,
        symbol=symbol,
    )


class TestCompositeRegimeDisabled:
    def test_default_no_composite(self):
        runner = _build_runner(use_composite=False)
        assert runner._use_composite_regime is False
        assert runner._composite_detector is None
        assert runner._param_router is None

    def test_check_regime_ignores_feat_dict(self):
        runner = _build_runner(use_composite=False)
        for i in range(25):
            runner._check_regime(100.0 + i * 0.1)
        runner._check_regime(100.0, feat_dict={"parkinson_vol": 0.02})
        assert runner._regime_params is None


class TestCompositeRegimeEnabled:
    def test_composite_enabled(self):
        runner = _build_runner(use_composite=True, symbol="BTCUSDT")
        assert runner._use_composite_regime is True
        assert runner._composite_detector is not None
        assert runner._param_router is not None

    def test_feat_dict_none_no_composite(self):
        runner = _build_runner(use_composite=True, symbol="BTCUSDT")
        for i in range(25):
            runner._check_regime(100.0 + i * 0.1)
        runner._check_regime(100.0, feat_dict=None)
        assert runner._regime_params is None

    def test_composite_updates_deadzone(self):
        runner = _build_runner(use_composite=True, symbol="BTCUSDT")
        for i in range(35):
            runner._check_regime(100.0 + i * 0.5)
        features = {
            "parkinson_vol": 0.02, "vol_of_vol": 0.005,
            "bb_width_20": 0.05, "close_vs_ma20": 0.03,
            "close_vs_ma50": 0.04, "adx_14": 30.0,
        }
        runner._check_regime(120.0, feat_dict=features)
        assert isinstance(runner._deadzone, float)

    def test_crisis_blocks_trading(self):
        runner = _build_runner(use_composite=True, symbol="BTCUSDT")
        for i in range(25):
            runner._check_regime(100.0 + i * 0.1)

        crisis = CompositeRegimeLabel(vol="crisis", trend="ranging")
        label = RegimeLabel(
            name="composite", ts=datetime.now(timezone.utc),
            value="ranging|crisis", score=0.95,
            meta={"composite": crisis, "is_crisis": True, "is_favorable": False},
        )
        runner._composite_detector.detect = MagicMock(return_value=label)
        runner._check_regime(100.0, feat_dict={"parkinson_vol": 0.1})
        assert runner._regime_active is False
        assert runner._regime_params is not None
        assert runner._regime_params.position_scale == 0.1

    def test_favorable_aggressive_params(self):
        runner = _build_runner(use_composite=True, symbol="BTCUSDT")
        for i in range(25):
            runner._check_regime(100.0 + i * 0.1)

        favorable = CompositeRegimeLabel(vol="low_vol", trend="strong_up")
        label = RegimeLabel(
            name="composite", ts=datetime.now(timezone.utc),
            value="strong_up|low_vol", score=0.8,
            meta={"composite": favorable, "is_crisis": False, "is_favorable": True},
        )
        runner._composite_detector.detect = MagicMock(return_value=label)
        runner._check_regime(100.0, feat_dict={"parkinson_vol": 0.01})
        assert runner._regime_active is True
        assert runner._deadzone == 0.3
        assert runner._min_hold == 18
        assert runner._regime_params.position_scale == 1.0

    def test_detector_returns_none_no_update(self):
        runner = _build_runner(use_composite=True, symbol="BTCUSDT")
        for i in range(25):
            runner._check_regime(100.0 + i * 0.1)
        runner._composite_detector.detect = MagicMock(return_value=None)
        runner._check_regime(100.0, feat_dict={"parkinson_vol": 0.01})
        assert runner._regime_params is None


class TestSymbolConfig:
    def test_btc_has_composite_regime(self):
        from scripts.ops.config import SYMBOL_CONFIG
        assert SYMBOL_CONFIG["BTCUSDT"].get("use_composite_regime") is True

    def test_eth_no_composite_regime(self):
        from scripts.ops.config import SYMBOL_CONFIG
        assert SYMBOL_CONFIG["ETHUSDT"].get("use_composite_regime") is None
