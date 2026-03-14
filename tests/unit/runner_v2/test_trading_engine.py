"""Tests for TradingEngine — uses mocks (no Rust build required)."""
from unittest.mock import MagicMock, patch
import pytest

from runner.trading_engine import TradingEngine


class TestTradingEngineOnBar:
    def test_returns_prediction_when_nonzero(self):
        hook = MagicMock()
        hook.on_bar.return_value = {"rsi_14": 0.5}
        bridge = MagicMock()
        bridge.predict.return_value = 0.8
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        result = engine.on_bar("BTCUSDT", {"close": 70000, "volume": 100})
        assert result == 0.8
        hook.on_bar.assert_called_once()
        bridge.predict.assert_called_once()

    def test_returns_none_when_zero_prediction(self):
        hook = MagicMock()
        hook.on_bar.return_value = {"rsi_14": 0.5}
        bridge = MagicMock()
        bridge.predict.return_value = 0.0
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        result = engine.on_bar("BTCUSDT", {"close": 70000, "volume": 100})
        assert result is None

    def test_returns_none_when_features_none(self):
        hook = MagicMock()
        hook.on_bar.return_value = None
        bridge = MagicMock()
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        result = engine.on_bar("BTCUSDT", {"close": 70000, "volume": 100})
        assert result is None
        bridge.predict.assert_not_called()


class TestTradingEngineReload:
    def test_reload_returns_status_dict(self):
        hook = MagicMock()
        bridge = MagicMock()
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="/tmp/nonexistent_model_dir")
        result = engine.reload_models()
        assert isinstance(result, dict)


class TestTradingEngineCheckpoint:
    def test_checkpoint_returns_dict_with_keys(self):
        hook = MagicMock()
        hook.checkpoint.return_value = {"bars": []}
        bridge = MagicMock()
        bridge.get_params.return_value = {"zscore": 1.0}
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        state = engine.checkpoint()
        assert "feature_hook" in state
        assert "inference" in state

    def test_restore_delegates(self):
        hook = MagicMock()
        bridge = MagicMock()
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        engine.restore({"feature_hook": {"bars": []}, "inference": {"zscore": 1.0}})
        hook.restore_checkpoint.assert_called_once()
