"""Tests for Phase 3 deep learning and online learning modules."""
from __future__ import annotations

import importlib
import sys
import unittest.mock
import pytest
from datetime import datetime
from unittest.mock import MagicMock


def _import_from(module_path: str, name: str):
    """Import a specific module file, bypassing __init__.py side effects."""
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── B3: Deep learning models ──

class TestLSTMAlpha:
    def test_init(self):
        mod = _import_from("alpha/models/lstm_alpha.py", "_lstm_test1")
        model = mod.LSTMAlphaModel(
            name="test_lstm",
            feature_names=("f1", "f2", "f3"),
            seq_len=10,
            hidden_size=32,
            num_layers=1,
        )
        assert model.name == "test_lstm"
        assert model.seq_len == 10

    def test_predict_no_model_returns_none(self):
        mod = _import_from("alpha/models/lstm_alpha.py", "_lstm_test2")
        model = mod.LSTMAlphaModel(feature_names=("f1",))
        result = model.predict(
            symbol="BTCUSDT",
            ts=datetime.now(),
            features={"f1": 1.0},
        )
        assert result is None  # No model built


class TestTransformerAlpha:
    def test_init(self):
        mod = _import_from("alpha/models/transformer_alpha.py", "_tfm_test1")
        model = mod.TransformerAlphaModel(
            name="test_tfm",
            feature_names=("a", "b"),
            seq_len=8,
            d_model=32,
            n_heads=2,
        )
        assert model.name == "test_tfm"
        assert model.d_model == 32

    def test_predict_no_model_returns_none(self):
        mod = _import_from("alpha/models/transformer_alpha.py", "_tfm_test2")
        model = mod.TransformerAlphaModel(feature_names=("a",))
        result = model.predict(
            symbol="ETHUSDT",
            ts=datetime.now(),
            features={"a": 1.0},
        )
        assert result is None


# ── B4: Online learning ──

def _mock_numpy():
    """Create a mock numpy module for tests."""
    np = MagicMock()
    np.array = lambda x: x  # passthrough
    return np


class TestIncrementalUpdater:
    def test_buffer_and_trigger(self):
        from alpha.online.incremental import IncrementalUpdater

        with unittest.mock.patch.dict(sys.modules, {"numpy": _mock_numpy()}):
            mock_model = MagicMock()
            mock_model.partial_fit = MagicMock()
            updater = IncrementalUpdater(model=mock_model, buffer_size=3)

            updater.add_observation([1.0, 2.0], 0.5)
            updater.add_observation([3.0, 4.0], 1.0)
            assert mock_model.partial_fit.call_count == 0

            updater.add_observation([5.0, 6.0], 1.5)
            assert mock_model.partial_fit.call_count == 1

    def test_update_count(self):
        from alpha.online.incremental import IncrementalUpdater

        with unittest.mock.patch.dict(sys.modules, {"numpy": _mock_numpy()}):
            mock_model = MagicMock()
            mock_model.partial_fit = MagicMock()
            updater = IncrementalUpdater(model=mock_model, buffer_size=2)

            updater.add_observation([1.0], 0.5)
            updater.add_observation([2.0], 1.0)
            assert updater.update_count == 1

    def test_force_update(self):
        from alpha.online.incremental import IncrementalUpdater

        with unittest.mock.patch.dict(sys.modules, {"numpy": _mock_numpy()}):
            mock_model = MagicMock()
            mock_model.partial_fit = MagicMock()
            updater = IncrementalUpdater(model=mock_model, buffer_size=100)

            updater.add_observation([1.0], 0.5)
            assert updater.update_count == 0
            updater.force_update()
            assert updater.update_count == 1


class TestRegimeModelSwitcher:
    def _make_model(self, name: str, side: str = "long", strength: float = 0.5):
        m = MagicMock()
        m.name = name
        sig = MagicMock()
        sig.side = side
        sig.strength = strength
        m.predict.return_value = sig
        return m

    def test_register_and_list(self):
        from alpha.online.regime_switch import RegimeModelSwitcher

        switcher = RegimeModelSwitcher()
        model = self._make_model("trend")
        switcher.register_model(model, regime_weights={"trending": 1.0})
        assert "trend" in switcher.models

    def test_set_regime(self):
        from alpha.online.regime_switch import RegimeModelSwitcher

        switcher = RegimeModelSwitcher()
        switcher.set_regime("volatile")
        assert switcher.current_regime == "volatile"

    def test_equal_weight_fallback(self):
        from alpha.online.regime_switch import RegimeModelSwitcher

        switcher = RegimeModelSwitcher()
        switcher.register_model(self._make_model("a"))
        switcher.register_model(self._make_model("b"))
        weights = switcher.get_active_weights()
        assert weights == {"a": 0.5, "b": 0.5}

    def test_ensemble_prediction(self):
        from alpha.online.regime_switch import RegimeModelSwitcher

        switcher = RegimeModelSwitcher()
        switcher.register_model(
            self._make_model("trend", "long", 0.8),
            regime_weights={"bullish": 0.7},
        )
        switcher.register_model(
            self._make_model("revert", "short", 0.3),
            regime_weights={"bullish": 0.3},
        )
        switcher.set_regime("bullish")

        result = switcher.predict_ensemble(
            symbol="BTCUSDT",
            ts=datetime.now(),
            features={"f": 1.0},
        )
        assert result is not None
        assert "_ensemble" in result
