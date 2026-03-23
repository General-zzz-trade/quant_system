"""Coverage wave 2 — unit tests for runner/, decision/, alpha/, execution/ modules."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import types

import numpy as np
import pytest


# ─── Helpers / stubs ──────────────────────────────────────────────────────────

def _make_rust_bridge_mock():
    m = MagicMock()
    m.checkpoint.return_value = {"state": "ok"}
    m.apply_constraints.return_value = 1.0
    m.check_monthly_gate.return_value = True
    m.get_position.return_value = 0.0
    m.set_position.return_value = None
    m.process_short_signal.return_value = -1.0
    m.reset.return_value = None
    m.restore.return_value = None
    return m


def _patch_quant_hotpath():
    """Add any missing attributes to the real _quant_hotpath for alpha/inference/bridge tests."""
    import _quant_hotpath as _real
    # bridge.py imports _RustBridge = RustInferenceBridge; ensure attribute exists
    if not hasattr(_real, "RustInferenceBridge"):
        _real.RustInferenceBridge = MagicMock(return_value=_make_rust_bridge_mock())
    if not hasattr(_real, "cpp_pred_to_signal"):
        _real.cpp_pred_to_signal = MagicMock(return_value=np.zeros(10))
    if not hasattr(_real, "RustTreePredictor"):
        _real.RustTreePredictor = MagicMock()
    return _real


_qhp = _patch_quant_hotpath()


def _stub_missing_modules():
    """Pre-stub transitive import deps that may not exist in test env."""
    stubs = {
        "data": None,
        "data.quality": None,
        "data.quality.live_validator": {"LiveBarValidator": type("LiveBarValidator", (), {})},
    }
    for mod_name, attrs in stubs.items():
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            if attrs:
                for k, v in attrs.items():
                    setattr(m, k, v)
            sys.modules[mod_name] = m
        elif attrs:
            existing = sys.modules[mod_name]
            for k, v in attrs.items():
                if not hasattr(existing, k):
                    setattr(existing, k, v)


_stub_missing_modules()


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: decision/hybrid_15m_executor.py
# ══════════════════════════════════════════════════════════════════════════════

from decision.hybrid_15m_executor import (  # noqa: E402
    _Bar15m, _BarAccumulator, MicroTiming, Hybrid15mExecutor,
)


class TestBar15m:
    def test_create(self):
        b = _Bar15m(open=100.0, high=105.0, low=99.0, close=102.0, volume=1000.0)
        assert b.open == 100.0
        assert b.timestamp == 0


class TestBarAccumulator:
    def test_partial_bars_return_none(self):
        acc = _BarAccumulator()
        b = _Bar15m(100, 101, 99, 100, 500)
        assert acc.push(b) is None
        assert acc.push(b) is None
        assert acc.push(b) is None

    def test_fourth_bar_completes(self):
        acc = _BarAccumulator()
        bars = [_Bar15m(100, 105, 98, 102, 100, timestamp=i) for i in range(4)]
        for b in bars[:3]:
            assert acc.push(b) is None
        result = acc.push(bars[3])
        assert result is not None
        assert result["open"] == 100.0
        assert result["high"] == 105.0
        assert result["low"] == 98.0
        assert result["close"] == 102.0
        assert result["volume"] == pytest.approx(400.0)

    def test_buffer_clears_after_completion(self):
        acc = _BarAccumulator()
        bars = [_Bar15m(i, i+1, i-1, i, 10) for i in range(5)]
        for b in bars[:4]:
            acc.push(b)
        assert len(acc.buffer) == 0
        r = acc.push(bars[4])
        assert r is None
        assert len(acc.buffer) == 1


class TestMicroTiming:
    def _populated(self, closes):
        mt = MicroTiming(ma_period=5, rsi_period=6)
        for c in closes:
            mt.update(c)
        return mt

    def test_not_ready_with_too_few(self):
        mt = MicroTiming(ma_period=5, rsi_period=6)
        for _ in range(5):
            mt.update(100.0)
        assert not mt.ready

    def test_ready_after_enough_bars(self):
        mt = MicroTiming(ma_period=5, rsi_period=6)
        for _ in range(7):
            mt.update(100.0)
        assert mt.ready

    def test_ma_calculation(self):
        mt = self._populated([100.0] * 10)
        assert mt.ma == pytest.approx(100.0)

    def test_ma_with_few_closes(self):
        mt = MicroTiming(ma_period=5, rsi_period=6)
        mt.update(50.0)
        assert mt.ma == 50.0

    def test_rsi_returns_50_when_not_ready(self):
        mt = MicroTiming(rsi_period=6)
        mt.update(100.0)
        assert mt.rsi == 50.0

    def test_rsi_flat_market(self):
        mt = self._populated([100.0] * 20)
        assert mt.rsi == 100.0

    def test_rsi_falling_market(self):
        closes = [100.0 - i for i in range(15)]
        mt = self._populated(closes)
        assert mt.rsi < 50.0

    def test_price_vs_ma_zero_ma(self):
        mt = MicroTiming()
        mt.update(0.0)
        assert mt.price_vs_ma == 0.0

    def test_price_vs_ma_above(self):
        closes = [100.0] * 9 + [110.0]
        mt = self._populated(closes)
        assert mt.price_vs_ma > 0.0

    def test_pullback_entry_direction_zero(self):
        mt = self._populated([100.0] * 15)
        assert mt.is_pullback_entry(0) is False

    def test_pullback_entry_not_ready(self):
        mt = MicroTiming()
        assert mt.is_pullback_entry(1) is False

    def test_momentum_entry_not_ready(self):
        mt = MicroTiming()
        assert mt.is_momentum_entry(1) is False

    def test_momentum_entry_direction_zero(self):
        mt = self._populated([100.0] * 15)
        assert mt.is_momentum_entry(0) is False

    def test_should_exit_micro_not_ready(self):
        mt = MicroTiming()
        assert mt.should_exit_micro(1) is False

    def test_should_exit_micro_direction_zero(self):
        mt = self._populated([100.0] * 15)
        assert mt.should_exit_micro(0) is False

    def test_pullback_entry_short_returns_bool(self):
        mt = self._populated([100.0] * 15)
        result = mt.is_pullback_entry(-1)
        assert isinstance(result, bool)

    def test_should_exit_long_returns_bool(self):
        closes = [100.0] * 5 + [120.0, 125.0, 130.0, 135.0, 140.0, 145.0]
        mt = self._populated(closes)
        result = mt.should_exit_micro(1)
        assert isinstance(result, bool)

    def test_should_exit_short_returns_bool(self):
        closes = [100.0] * 5 + [80.0, 75.0, 70.0, 65.0, 60.0, 55.0]
        mt = self._populated(closes)
        result = mt.should_exit_micro(-1)
        assert isinstance(result, bool)


class TestHybrid15mExecutor:
    def _bar(self, close=100.0, high=None, low=None):
        return _Bar15m(
            open=close, high=high or close * 1.01,
            low=low or close * 0.99, close=close, volume=1000.0,
        )

    def test_init_defaults(self):
        ex = Hybrid15mExecutor()
        assert ex.position == 0

    def test_no_signal_no_trade(self):
        ex = Hybrid15mExecutor(deadzone=0.5)
        result = ex.on_15m_bar(self._bar(100.0), z_1h=0.0)
        assert result is None
        assert ex.position == 0

    def test_strong_signal_immediate_entry_long(self):
        ex = Hybrid15mExecutor(deadzone=0.5, strong_signal_mult=2.0)
        result = ex.on_15m_bar(self._bar(100.0), z_1h=1.5)
        assert result is not None
        assert result["action"] == "entry"
        assert result["side"] == "long"
        assert result["reason"] == "strong_signal"
        assert ex.position == 1

    def test_strong_signal_immediate_entry_short(self):
        ex = Hybrid15mExecutor(deadzone=0.5, strong_signal_mult=2.0, long_only=False)
        result = ex.on_15m_bar(self._bar(100.0), z_1h=-1.5)
        assert result is not None
        assert result["action"] == "entry"
        assert result["side"] == "short"

    def test_long_only_no_short(self):
        ex = Hybrid15mExecutor(deadzone=0.5, long_only=True, strong_signal_mult=2.0)
        result = ex.on_15m_bar(self._bar(100.0), z_1h=-1.5)
        assert result is None
        assert ex.position == 0

    def test_timeout_entry(self):
        ex = Hybrid15mExecutor(deadzone=0.3, strong_signal_mult=100.0)
        result = None
        for _ in range(ex._max_wait):
            result = ex.on_15m_bar(self._bar(100.0), z_1h=0.4)
        assert result is not None
        assert result["reason"] == "timeout_entry"

    def test_max_hold_exit(self):
        ex = Hybrid15mExecutor(deadzone=0.5, strong_signal_mult=2.0, max_hold_15m=3, min_hold_15m=1)
        ex.on_15m_bar(self._bar(100.0), z_1h=1.5)
        assert ex.position == 1
        # Exit occurs when held >= max_hold=3 (on the 4th bar after entry)
        exits = []
        for _ in range(5):
            r = ex.on_15m_bar(self._bar(100.0), z_1h=0.6)
            if r is not None:
                exits.append(r)
        assert len(exits) >= 1
        assert exits[0]["reason"] == "max_hold"

    def test_trailing_stop_long(self):
        ex = Hybrid15mExecutor(
            deadzone=0.5, strong_signal_mult=2.0,
            min_hold_15m=1, trailing_stop_pct=0.01,
        )
        ex.on_15m_bar(self._bar(100.0), z_1h=1.5)
        assert ex.position == 1
        result = ex.on_15m_bar(self._bar(close=98.0, low=98.0), z_1h=0.6)
        assert result is not None
        assert "trailing_stop" in result["reason"]

    def test_trailing_stop_short(self):
        ex = Hybrid15mExecutor(
            deadzone=0.5, strong_signal_mult=2.0,
            min_hold_15m=1, trailing_stop_pct=0.01, long_only=False,
        )
        ex.on_15m_bar(self._bar(100.0), z_1h=-1.5)
        assert ex.position == -1
        result = ex.on_15m_bar(self._bar(close=101.5, high=101.5), z_1h=-0.6)
        assert result is not None
        assert "trailing_stop" in result["reason"]

    def test_signal_reversal_exit(self):
        ex = Hybrid15mExecutor(deadzone=0.3, strong_signal_mult=2.0, min_hold_15m=1)
        ex.on_15m_bar(self._bar(100.0), z_1h=1.0)
        result = ex.on_15m_bar(self._bar(100.0), z_1h=-1.0)
        assert result is not None
        assert result["reason"] == "signal_reversal"

    def test_deadzone_fade_exit(self):
        ex = Hybrid15mExecutor(deadzone=0.3, strong_signal_mult=2.0, min_hold_15m=1)
        ex.on_15m_bar(self._bar(100.0), z_1h=1.0)
        result = ex.on_15m_bar(self._bar(100.0), z_1h=0.1)
        assert result is not None
        assert result["reason"] == "deadzone_fade"

    def test_exit_clears_pending_direction(self):
        ex = Hybrid15mExecutor(deadzone=0.3, strong_signal_mult=2.0, min_hold_15m=1)
        ex.on_15m_bar(self._bar(100.0), z_1h=1.0)
        ex.on_15m_bar(self._bar(100.0), z_1h=0.1)
        assert ex._pending_direction == 0

    def test_reset(self):
        ex = Hybrid15mExecutor(deadzone=0.5, strong_signal_mult=2.0)
        ex.on_15m_bar(self._bar(100.0), z_1h=1.5)
        ex.reset()
        assert ex.position == 0
        assert ex._bar_count == 0

    def test_pending_direction_clears_on_no_signal(self):
        ex = Hybrid15mExecutor(deadzone=0.5, strong_signal_mult=100.0)
        ex.on_15m_bar(self._bar(100.0), z_1h=0.6)
        ex.on_15m_bar(self._bar(100.0), z_1h=0.0)
        assert ex._pending_direction == 0

    def test_min_hold_prevents_early_exit(self):
        ex = Hybrid15mExecutor(deadzone=0.3, strong_signal_mult=2.0, min_hold_15m=5)
        ex.on_15m_bar(self._bar(100.0), z_1h=1.0)
        result = ex.on_15m_bar(self._bar(100.0), z_1h=-1.0)
        assert result is None

    def test_entry_records_price(self):
        ex = Hybrid15mExecutor(deadzone=0.5, strong_signal_mult=2.0)
        ex.on_15m_bar(self._bar(200.0), z_1h=1.5)
        assert ex._entry_price == pytest.approx(200.0)

    def test_pending_z_updated(self):
        ex = Hybrid15mExecutor(deadzone=0.3, strong_signal_mult=100.0)
        ex.on_15m_bar(self._bar(100.0), z_1h=0.5)
        assert ex._pending_z == 0.5

    def test_direction_change_resets_wait(self):
        ex = Hybrid15mExecutor(deadzone=0.3, strong_signal_mult=100.0, long_only=False)
        ex.on_15m_bar(self._bar(100.0), z_1h=0.4)
        ex.on_15m_bar(self._bar(100.0), z_1h=-0.4)
        assert ex._wait_bars == 1

    def test_exit_side_is_opposite(self):
        ex = Hybrid15mExecutor(deadzone=0.3, strong_signal_mult=2.0, min_hold_15m=1)
        ex.on_15m_bar(self._bar(100.0), z_1h=1.0)
        result = ex.on_15m_bar(self._bar(100.0), z_1h=0.1)
        # When long (position=1), exit side is "short" (to close the long)
        assert result["side"] == "short"

    def test_exit_result_has_held_bars(self):
        ex = Hybrid15mExecutor(deadzone=0.3, strong_signal_mult=2.0, min_hold_15m=1)
        ex.on_15m_bar(self._bar(100.0), z_1h=1.0)
        result = ex.on_15m_bar(self._bar(100.0), z_1h=0.1)
        assert "held_bars" in result
        assert "held_hours" in result


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: alpha/models/lgbm_alpha.py
# ══════════════════════════════════════════════════════════════════════════════

from alpha.models.lgbm_alpha import LGBMAlphaModel  # noqa: E402


class TestLGBMAlphaModel:
    def test_predict_returns_none_when_no_model(self):
        m = LGBMAlphaModel(feature_names=["f1"])
        result = m.predict(symbol="BTC", ts=datetime.now(), features={"f1": 1.0})
        assert result is None

    def test_predict_uses_rust_predictor_if_available(self):
        m = LGBMAlphaModel(feature_names=["f1"])
        rust_pred = MagicMock()
        rust_pred.predict_dict.return_value = 0.8
        m._rust_predictor = rust_pred
        sig = m.predict(symbol="BTC", ts=datetime.now(), features={"f1": 1.0})
        assert sig is not None
        assert sig.side == "long"
        assert sig.strength == pytest.approx(0.8)

    def test_predict_short_signal(self):
        m = LGBMAlphaModel(feature_names=["f1"])
        rust_pred = MagicMock()
        rust_pred.predict_dict.return_value = -0.7
        m._rust_predictor = rust_pred
        sig = m.predict(symbol="BTC", ts=datetime.now(), features={"f1": 1.0})
        assert sig.side == "short"

    def test_predict_flat_within_threshold(self):
        m = LGBMAlphaModel(feature_names=["f1"], threshold=0.5)
        rust_pred = MagicMock()
        rust_pred.predict_dict.return_value = 0.3
        m._rust_predictor = rust_pred
        sig = m.predict(symbol="BTC", ts=datetime.now(), features={"f1": 1.0})
        assert sig.side == "flat"

    def test_predict_regressor(self):
        m = LGBMAlphaModel(feature_names=["f1"], _is_classifier=False)
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.6]
        m._model = mock_model
        sig = m.predict(symbol="ETH", ts=datetime.now(), features={"f1": 0.5})
        assert sig.side == "long"

    def test_predict_classifier(self):
        m = LGBMAlphaModel(feature_names=["f1"], _is_classifier=True)
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.8]])
        m._model = mock_model
        sig = m.predict(symbol="ETH", ts=datetime.now(), features={"f1": 0.5})
        assert sig.side == "long"

    def test_predict_strength_capped_at_1(self):
        m = LGBMAlphaModel(feature_names=["f1"])
        rust_pred = MagicMock()
        rust_pred.predict_dict.return_value = 5.0
        m._rust_predictor = rust_pred
        sig = m.predict(symbol="BTC", ts=datetime.now(), features={"f1": 1.0})
        assert sig.strength == pytest.approx(1.0)

    def test_fit_with_mock_lgb(self):
        lgb_mod = types.ModuleType("lightgbm")
        mock_regressor = MagicMock()
        mock_regressor_inst = MagicMock()
        # predict must return array of same length as val set
        mock_regressor_inst.predict.side_effect = lambda X_in: np.zeros(len(X_in))
        mock_regressor_inst.best_iteration_ = 50
        mock_regressor.return_value = mock_regressor_inst
        lgb_mod.LGBMRegressor = mock_regressor
        lgb_mod.early_stopping = MagicMock(return_value=MagicMock())
        lgb_mod.log_evaluation = MagicMock(return_value=MagicMock())

        n = 20
        X = np.ones((n, 2))
        y = np.ones(n) * 0.1

        with patch.dict(sys.modules, {"lightgbm": lgb_mod}):
            m = LGBMAlphaModel(feature_names=["f1", "f2"])
            metrics = m.fit(X, y, early_stopping_rounds=10)

        assert "val_mse" in metrics
        assert "direction_accuracy" in metrics
        assert "best_iteration" in metrics

    def test_fit_without_early_stopping(self):
        lgb_mod = types.ModuleType("lightgbm")
        mock_reg = MagicMock()
        mock_inst = MagicMock()
        mock_inst.predict.side_effect = lambda X_in: np.zeros(len(X_in))
        mock_reg.return_value = mock_inst
        lgb_mod.LGBMRegressor = mock_reg

        n = 10
        X = np.ones((n, 2))
        y = np.ones(n) * 0.1

        with patch.dict(sys.modules, {"lightgbm": lgb_mod}):
            m = LGBMAlphaModel(feature_names=["f1", "f2"])
            metrics = m.fit(X, y)
        assert "best_iteration" not in metrics

    def test_fit_classifier(self):
        lgb_mod = types.ModuleType("lightgbm")
        mock_clf = MagicMock()
        mock_clf_inst = MagicMock()
        # predict_proba must return array matching val size
        mock_clf_inst.predict_proba.side_effect = lambda X_in: np.array([[0.4, 0.6]] * len(X_in))
        mock_clf.return_value = mock_clf_inst
        lgb_mod.LGBMClassifier = mock_clf

        n = 20
        X = np.ones((n, 2))
        y_binary = np.array([0, 1] * 10)

        with patch.dict(sys.modules, {"lightgbm": lgb_mod}):
            m = LGBMAlphaModel(feature_names=["f1", "f2"])
            metrics = m.fit_classifier(X, y_binary)

        assert "val_logloss" in metrics
        assert m._is_classifier is True

    def test_fit_with_embargo_and_weights(self):
        lgb_mod = types.ModuleType("lightgbm")
        mock_reg = MagicMock()
        mock_inst = MagicMock()
        mock_inst.predict.side_effect = lambda X_in: np.zeros(len(X_in))
        mock_reg.return_value = mock_inst
        lgb_mod.LGBMRegressor = mock_reg

        n = 20
        X = np.ones((n, 2))
        y = np.ones(n) * 0.1
        w = np.ones(n)

        with patch.dict(sys.modules, {"lightgbm": lgb_mod}):
            m = LGBMAlphaModel(feature_names=["f1", "f2"])
            metrics = m.fit(X, y, embargo_bars=2, sample_weight=w)
        assert "val_mse" in metrics

    def test_try_load_rust_no_json(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        model_path.touch()
        m = LGBMAlphaModel()
        m._try_load_rust(model_path)
        assert m._rust_predictor is None

    def test_try_load_rust_exception_is_swallowed(self, tmp_path):
        json_path = tmp_path / "model.json"
        json_path.write_text("{}")
        pkl_path = tmp_path / "model.pkl"

        m = LGBMAlphaModel()
        # Patch RustTreePredictor.load to raise; the except block should swallow it
        import _quant_hotpath as real_qhp
        orig_cls = real_qhp.RustTreePredictor
        bad_cls = MagicMock()
        bad_cls.load.side_effect = RuntimeError("rust predictor fail")
        real_qhp.RustTreePredictor = bad_cls
        try:
            m._try_load_rust(pkl_path)
        finally:
            real_qhp.RustTreePredictor = orig_cls
        assert m._rust_predictor is None

    def test_fit_classifier_with_early_stopping(self):
        lgb_mod = types.ModuleType("lightgbm")
        mock_clf = MagicMock()
        mock_clf_inst = MagicMock()
        mock_clf_inst.predict_proba.side_effect = lambda X_in: np.array([[0.4, 0.6]] * len(X_in))
        mock_clf_inst.best_iteration_ = 30
        mock_clf.return_value = mock_clf_inst
        lgb_mod.LGBMClassifier = mock_clf
        lgb_mod.early_stopping = MagicMock(return_value=MagicMock())
        lgb_mod.log_evaluation = MagicMock(return_value=MagicMock())

        n = 20
        X = np.ones((n, 2))
        y_binary = np.array([0, 1] * 10)

        with patch.dict(sys.modules, {"lightgbm": lgb_mod}):
            m = LGBMAlphaModel(feature_names=["f1", "f2"])
            metrics = m.fit_classifier(X, y_binary, early_stopping_rounds=10)

        assert "best_iteration" in metrics


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: alpha/models/xgb_alpha.py
# ══════════════════════════════════════════════════════════════════════════════

from alpha.models.xgb_alpha import XGBAlphaModel  # noqa: E402


class TestXGBAlphaModel:
    def test_predict_none_when_no_model(self):
        m = XGBAlphaModel(feature_names=["f1"])
        result = m.predict(symbol="BTC", ts=datetime.now(), features={"f1": 1.0})
        assert result is None

    def test_predict_rust_path(self):
        m = XGBAlphaModel(feature_names=["f1"])
        rust_pred = MagicMock()
        rust_pred.predict_dict.return_value = 0.5
        m._rust_predictor = rust_pred
        sig = m.predict(symbol="BTC", ts=datetime.now(), features={"f1": 1.0})
        assert sig.side == "long"

    def test_predict_short(self):
        m = XGBAlphaModel(feature_names=["f1"])
        rust_pred = MagicMock()
        rust_pred.predict_dict.return_value = -0.9
        m._rust_predictor = rust_pred
        sig = m.predict(symbol="ETH", ts=datetime.now(), features={"f1": 0.0})
        assert sig.side == "short"

    def test_predict_flat(self):
        m = XGBAlphaModel(feature_names=["f1"], threshold=0.5)
        rust_pred = MagicMock()
        rust_pred.predict_dict.return_value = 0.2
        m._rust_predictor = rust_pred
        sig = m.predict(symbol="ETH", ts=datetime.now(), features={"f1": 0.0})
        assert sig.side == "flat"

    def test_predict_sklearn_model(self):
        m = XGBAlphaModel(feature_names=["f1"])
        mock_model = MagicMock(spec=[])  # no 'predict' in spec forces attribute access
        mock_model.predict = MagicMock(return_value=np.array([0.7]))
        m._model = mock_model

        xgb_mod = types.ModuleType("xgboost")
        xgb_mod.Booster = type("Booster", (), {})
        with patch.dict(sys.modules, {"xgboost": xgb_mod}):
            sig = m.predict(symbol="BTC", ts=datetime.now(), features={"f1": 1.0})
        assert sig.side == "long"

    def test_fit_mock(self):
        xgb_mod = types.ModuleType("xgboost")
        mock_reg = MagicMock()
        mock_inst = MagicMock()
        mock_inst.predict.side_effect = lambda X_in: np.zeros(len(X_in))
        mock_reg.return_value = mock_inst
        xgb_mod.XGBRegressor = mock_reg

        n = 10
        X = np.ones((n, 2))
        y = np.ones(n) * 0.1

        with patch.dict(sys.modules, {"xgboost": xgb_mod}):
            m = XGBAlphaModel(feature_names=["f1", "f2"])
            metrics = m.fit(X, y)

        assert "val_mse" in metrics

    def test_fit_with_early_stopping(self):
        xgb_mod = types.ModuleType("xgboost")
        mock_reg = MagicMock()
        mock_inst = MagicMock()
        mock_inst.predict.side_effect = lambda X_in: np.zeros(len(X_in))
        mock_inst.best_iteration = 42
        mock_reg.return_value = mock_inst
        xgb_mod.XGBRegressor = mock_reg

        n = 10
        X = np.ones((n, 2))
        y = np.ones(n) * 0.1

        with patch.dict(sys.modules, {"xgboost": xgb_mod}):
            m = XGBAlphaModel(feature_names=["f1", "f2"])
            metrics = m.fit(X, y, early_stopping_rounds=5)

        assert metrics.get("best_iteration") == 42.0

    def test_fit_sample_weight(self):
        xgb_mod = types.ModuleType("xgboost")
        mock_reg = MagicMock()
        mock_inst = MagicMock()
        mock_inst.predict.side_effect = lambda X_in: np.zeros(len(X_in))
        mock_reg.return_value = mock_inst
        xgb_mod.XGBRegressor = mock_reg

        n = 10
        X = np.ones((n, 2))
        y = np.ones(n)
        w = np.ones(n)

        with patch.dict(sys.modules, {"xgboost": xgb_mod}):
            m = XGBAlphaModel(feature_names=["f1", "f2"])
            metrics = m.fit(X, y, sample_weight=w)

        assert "val_mse" in metrics

    def test_fit_with_custom_params(self):
        xgb_mod = types.ModuleType("xgboost")
        mock_reg = MagicMock()
        mock_inst = MagicMock()
        mock_inst.predict.side_effect = lambda X_in: np.zeros(len(X_in))
        mock_reg.return_value = mock_inst
        xgb_mod.XGBRegressor = mock_reg

        n = 10
        X = np.ones((n, 2))
        y = np.ones(n)

        with patch.dict(sys.modules, {"xgboost": xgb_mod}):
            m = XGBAlphaModel(feature_names=["f1", "f2"])
            m.fit(X, y, params={"subsample": 0.8})

        call_kwargs = mock_reg.call_args[1]
        assert call_kwargs.get("subsample") == 0.8


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: alpha/training/regime_split.py
# ══════════════════════════════════════════════════════════════════════════════

from alpha.training.regime_split import (  # noqa: E402
    RegimeModelBundle, compute_vol_regime, train_regime_models,
    apply_vol_regime, detect_current_regime,
)


class TestComputeVolRegime:
    def test_too_few_samples_returns_ones(self):
        vol = np.array([0.1, 0.2, 0.3])
        regime = compute_vol_regime(vol)
        assert np.all(regime == 1)

    def test_tercile_binning(self):
        vol = np.linspace(0.01, 0.30, 90)
        regime = compute_vol_regime(vol)
        assert 0 in regime
        assert 1 in regime
        assert 2 in regime

    def test_nan_values_get_regime_1(self):
        vol = np.array([0.1] * 30 + [np.nan] * 10)
        regime = compute_vol_regime(vol)
        assert regime[-1] == 1

    def test_all_nan_falls_back(self):
        vol = np.full(40, np.nan)
        regime = compute_vol_regime(vol)
        assert np.all(regime == 1)


class TestApplyVolRegime:
    def test_basic(self):
        vol = np.array([0.05, 0.10, 0.20, np.nan])
        regime = apply_vol_regime(vol, p33=0.07, p67=0.15)
        assert regime[0] == 0
        assert regime[1] == 1
        assert regime[2] == 2
        assert regime[3] == 1


class TestDetectCurrentRegime:
    def test_empty_returns_mid(self):
        assert detect_current_regime([], 0.1, 0.2) == 1

    def test_low_regime(self):
        assert detect_current_regime([0.05], 0.1, 0.2) == 0

    def test_mid_regime(self):
        assert detect_current_regime([0.15], 0.1, 0.2) == 1

    def test_high_regime(self):
        assert detect_current_regime([0.25], 0.1, 0.2) == 2

    def test_nan_returns_mid(self):
        assert detect_current_regime([float("nan")], 0.1, 0.2) == 1

    def test_exactly_p33_is_low(self):
        assert detect_current_regime([0.10], 0.10, 0.20) == 0

    def test_exactly_p67_is_mid(self):
        assert detect_current_regime([0.20], 0.10, 0.20) == 1


class TestRegimeModelBundle:
    def _make_bundle(self):
        bundle = RegimeModelBundle(feature_names=("f1", "f2"))
        for regime in [0, 1, 2]:
            m = LGBMAlphaModel(name=f"r{regime}", feature_names=("f1", "f2"))
            mock_inner = MagicMock()
            mock_inner.predict.return_value = [0.5 * regime]
            m._model = mock_inner
            bundle.models[regime] = m
        return bundle

    def test_predict_uses_correct_model(self):
        bundle = self._make_bundle()
        result = bundle.predict_regime(1, {"f1": 0.1, "f2": 0.2})
        assert result is not None

    def test_predict_falls_back_to_fallback(self):
        bundle = RegimeModelBundle(feature_names=("f1",))
        fallback = LGBMAlphaModel(name="fallback", feature_names=("f1",))
        mock_inner = MagicMock()
        mock_inner.predict.return_value = [0.42]
        fallback._model = mock_inner
        bundle.fallback = fallback
        result = bundle.predict_regime(99, {"f1": 1.0})
        assert result == pytest.approx(0.42)

    def test_predict_returns_none_when_no_model_or_fallback(self):
        bundle = RegimeModelBundle(feature_names=("f1",))
        result = bundle.predict_regime(0, {"f1": 1.0})
        assert result is None

    def test_predict_fallback_no_inner_model_returns_none(self):
        bundle = RegimeModelBundle(feature_names=("f1",))
        fb = LGBMAlphaModel(name="fb", feature_names=("f1",))
        # _model is None
        bundle.fallback = fb
        result = bundle.predict_regime(99, {"f1": 1.0})
        assert result is None


class TestTrainRegimeModels:
    def test_train_with_mock_lgbm(self):
        lgb_mod = types.ModuleType("lightgbm")

        # Use a side_effect that returns the right shape based on call count
        def make_predict(inst):
            def predict(X_in):
                return np.zeros(len(X_in))
            return predict

        def make_inst():
            inst = MagicMock()
            inst.predict.side_effect = make_predict(inst)
            return inst

        mock_reg_cls = MagicMock(side_effect=lambda **kw: make_inst())
        lgb_mod.LGBMRegressor = mock_reg_cls

        n = 2000
        X = np.random.randn(n, 3)
        y = np.random.randn(n)
        regimes = np.array([i % 3 for i in range(n)])

        with patch.dict(sys.modules, {"lightgbm": lgb_mod}):
            bundle = train_regime_models(
                X, y, regimes, feature_names=["f1", "f2", "f3"],
                min_samples=100, early_stopping_rounds=0,
            )

        assert bundle.fallback is not None
        assert len(bundle.models) == 3

    def test_train_skips_small_regimes(self):
        lgb_mod = types.ModuleType("lightgbm")

        def make_inst():
            inst = MagicMock()
            inst.predict.side_effect = lambda X_in: np.zeros(len(X_in))
            return inst

        lgb_mod.LGBMRegressor = MagicMock(side_effect=lambda **kw: make_inst())

        n = 600
        X = np.random.randn(n, 2)
        y = np.random.randn(n)
        regimes = np.zeros(n, dtype=int)

        with patch.dict(sys.modules, {"lightgbm": lgb_mod}):
            bundle = train_regime_models(
                X, y, regimes, feature_names=["f1", "f2"],
                min_samples=500, early_stopping_rounds=0,
            )
        assert 0 in bundle.models
        assert 1 not in bundle.models
        assert 2 not in bundle.models

    def test_train_with_sample_weights(self):
        lgb_mod = types.ModuleType("lightgbm")

        def make_inst():
            inst = MagicMock()
            inst.predict.side_effect = lambda X_in: np.zeros(len(X_in))
            return inst

        lgb_mod.LGBMRegressor = MagicMock(side_effect=lambda **kw: make_inst())

        n = 1500
        X = np.random.randn(n, 2)
        y = np.random.randn(n)
        regimes = np.array([i % 3 for i in range(n)])
        w = np.ones(n)

        with patch.dict(sys.modules, {"lightgbm": lgb_mod}):
            bundle = train_regime_models(
                X, y, regimes, feature_names=["f1", "f2"],
                min_samples=100, early_stopping_rounds=0, sample_weight=w,
            )
        assert bundle.fallback is not None


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: alpha/signal_transform.py
# ══════════════════════════════════════════════════════════════════════════════

from alpha.signal_transform import enforce_min_hold  # noqa: E402


class TestEnforceMinHold:
    def test_basic_hold(self):
        # Signal starts at 1.0, then flips to -1.0 after min_hold=3 bars
        signal = np.array([1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
        out = enforce_min_hold(signal, min_hold=3)
        # First 3 bars held at 1.0 regardless of signal
        assert out[0] == 1.0
        assert out[1] == 1.0
        assert out[2] == 1.0
        # After hold completes, can now follow signal = -1.0
        assert out[3] == -1.0

    def test_min_hold_one(self):
        signal = np.array([1.0, -1.0, 1.0])
        out = enforce_min_hold(signal, min_hold=1)
        assert out[0] == 1.0
        assert out[1] == -1.0

    def test_flat_signal(self):
        signal = np.array([0.0, 0.0, 0.0])
        out = enforce_min_hold(signal, min_hold=5)
        assert np.all(out == 0.0)

    def test_holds_extend_when_same_signal(self):
        signal = np.array([1.0, 1.0, 1.0, 1.0, -1.0])
        out = enforce_min_hold(signal, min_hold=3)
        assert out[4] == -1.0

    def test_single_element(self):
        signal = np.array([1.0])
        out = enforce_min_hold(signal, min_hold=5)
        assert out[0] == 1.0

    def test_output_same_length(self):
        signal = np.array([1.0, -1.0, 0.0, 1.0, -1.0] * 4)
        out = enforce_min_hold(signal, min_hold=3)
        assert len(out) == len(signal)


class TestPredToSignalBinary:
    def test_binary_mode_sign(self):
        from alpha.signal_transform import pred_to_signal
        y_pred = np.array([0.6, 0.4, 0.5, 0.51, 0.49])
        out = pred_to_signal(y_pred, target_mode="binary", min_hold=1)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(-1.0)

    def test_binary_mode_deadzone(self):
        from alpha.signal_transform import pred_to_signal
        y_pred = np.array([0.51, 0.49])
        out = pred_to_signal(y_pred, target_mode="binary", min_hold=1)
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.0)

    def test_binary_mode_with_min_hold(self):
        from alpha.signal_transform import pred_to_signal
        y_pred = np.array([0.7, 0.3, 0.7, 0.3])
        out = pred_to_signal(y_pred, target_mode="binary", min_hold=2)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(1.0)  # held


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: alpha/strategy_config.py
# ══════════════════════════════════════════════════════════════════════════════

from alpha.strategy_config import SymbolStrategyConfig, SYMBOL_CONFIG, get_config  # noqa: E402


class TestStrategyConfig:
    def test_get_config_btcusdt(self):
        cfg = get_config("BTCUSDT")
        assert isinstance(cfg, SymbolStrategyConfig)
        assert cfg.deadzone == 0.5
        assert "basis" in cfg.fixed_features

    def test_get_config_ethusdt(self):
        cfg = get_config("ETHUSDT")
        assert cfg.min_hold == 24
        assert cfg.long_only is True

    def test_get_config_solusdt(self):
        cfg = get_config("SOLUSDT")
        assert cfg.n_flexible == 5
        assert cfg.deadzone == 1.0

    def test_get_config_missing_symbol_raises(self):
        with pytest.raises(KeyError):
            get_config("FAKECOIN")

    def test_symbol_config_is_frozen(self):
        cfg = get_config("BTCUSDT")
        with pytest.raises((AttributeError, TypeError)):
            cfg.deadzone = 99.0  # type: ignore

    def test_all_symbols_have_model_dir(self):
        for sym, cfg in SYMBOL_CONFIG.items():
            assert cfg.model_dir, f"{sym} has empty model_dir"

    def test_candidate_pool_non_empty(self):
        for sym, cfg in SYMBOL_CONFIG.items():
            assert len(cfg.candidate_pool) > 0, f"{sym} candidate_pool empty"

    def test_default_ensemble_true(self):
        cfg = get_config("ETHUSDT")
        assert cfg.ensemble is True


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: alpha/inference/bridge.py (uncovered branches)
# ══════════════════════════════════════════════════════════════════════════════

class TestLiveInferenceBridgeUncovered:
    def _make_bridge(self, **kw):
        rust_mock = _make_rust_bridge_mock()
        inference_engine_mock = MagicMock()
        inference_engine_mock.run.return_value = []

        with patch("alpha.inference.bridge._RustBridge", return_value=rust_mock), \
             patch("alpha.inference.bridge.InferenceEngine", return_value=inference_engine_mock):
            from alpha.inference.bridge import LiveInferenceBridge
            bridge = LiveInferenceBridge(models=[], **kw)
            bridge._rust = rust_mock
            bridge._engine = inference_engine_mock
            return bridge

    def _make_entry(self, side="long", strength=0.7, error=None):
        mock_signal = MagicMock()
        mock_signal.side = side
        mock_signal.strength = strength
        entry = MagicMock()
        entry.error = error
        entry.signal = mock_signal if error is None else None
        entry.latency_ms = 1.0
        entry.model_name = "test_model"
        return entry

    def test_enrich_no_results(self):
        bridge = self._make_bridge()
        features = {"close": 100.0, "f1": 0.5}
        result = bridge.enrich("BTCUSDT", datetime.now(timezone.utc), features)
        assert "ml_score" not in result

    def test_enrich_with_long_signal(self):
        bridge = self._make_bridge()
        bridge._engine.run.return_value = [self._make_entry("long", 0.7)]
        bridge._rust.apply_constraints.return_value = 0.7

        out = bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
        assert "ml_score" in out
        assert out["ml_score"] == pytest.approx(0.7)

    def test_enrich_short_signal_negates(self):
        bridge = self._make_bridge()
        bridge._engine.run.return_value = [self._make_entry("short", 0.6)]
        bridge._rust.apply_constraints.return_value = -0.6

        out = bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
        assert out.get("ml_score") == pytest.approx(-0.6)

    def test_enrich_flat_signal(self):
        bridge = self._make_bridge()
        bridge._engine.run.return_value = [self._make_entry("flat", 0.0)]
        bridge._rust.apply_constraints.return_value = 0.0

        out = bridge.enrich("BTCUSDT", None, {"close": 50.0})
        assert out.get("ml_score") == pytest.approx(0.0)

    def test_enrich_inference_error_skipped(self):
        bridge = self._make_bridge()
        bridge._engine.run.return_value = [self._make_entry(error="some error")]

        out = bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
        assert "ml_score" not in out

    def test_enrich_with_metrics_exporter(self):
        metrics = MagicMock()
        bridge = self._make_bridge(metrics_exporter=metrics)
        bridge._engine.run.return_value = [self._make_entry("long", 0.5)]
        bridge._rust.apply_constraints.return_value = 0.5

        bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
        assert metrics.observe_histogram.called

    def test_enrich_inference_error_with_metrics(self):
        metrics = MagicMock()
        bridge = self._make_bridge(metrics_exporter=metrics)
        bridge._engine.run.return_value = [self._make_entry(error="boom")]

        bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
        assert metrics.inc_counter.called

    def test_enrich_weighted_ensemble(self):
        bridge = self._make_bridge(ensemble_weights=[0.6, 0.4])
        bridge._engine.run.return_value = [
            self._make_entry("long", 0.8),
            self._make_entry("long", 0.4),
        ]
        bridge._rust.apply_constraints.return_value = 0.64

        out = bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
        assert "ml_score" in out

    def test_vol_adaptive_sizing(self):
        bridge = self._make_bridge(vol_target=0.02)
        bridge._engine.run.return_value = [self._make_entry("long", 1.0)]
        bridge._rust.apply_constraints.return_value = 1.0

        features = {"close": 100.0, "atr_norm_14": 0.04}
        out = bridge.enrich("BTCUSDT", datetime.now(timezone.utc), features)
        assert out.get("ml_score") == pytest.approx(0.5)

    def test_monthly_gate_gates_score(self):
        bridge = self._make_bridge(monthly_gate=True)
        bridge._rust.check_monthly_gate.return_value = False
        bridge._rust.apply_constraints.return_value = 1.0
        bridge._rust.get_position.return_value = 0.0
        bridge._engine.run.return_value = [self._make_entry("long", 1.0)]

        out = bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
        assert out.get("ml_score") == pytest.approx(0.0)

    def test_update_params_scalar_deadzone(self):
        bridge = self._make_bridge()
        bridge.update_params("BTCUSDT", deadzone=0.8)
        assert isinstance(bridge._deadzone, dict)
        assert bridge._deadzone["BTCUSDT"] == 0.8

    def test_update_params_dict_deadzone(self):
        bridge = self._make_bridge(deadzone={"BTCUSDT": 0.5})
        bridge.update_params("BTCUSDT", deadzone=0.9)
        assert bridge._deadzone["BTCUSDT"] == 0.9

    def test_update_params_min_hold(self):
        bridge = self._make_bridge()
        bridge.update_params("ETHUSDT", min_hold=18)
        assert bridge._min_hold_bars["ETHUSDT"] == 18

    def test_update_params_max_hold(self):
        bridge = self._make_bridge()
        bridge.update_params("ETHUSDT", max_hold=300)
        assert bridge._max_hold == 300

    def test_update_params_long_only_add(self):
        bridge = self._make_bridge()
        bridge.update_params("BTCUSDT", long_only=True)
        assert "BTCUSDT" in bridge._long_only_symbols

    def test_update_params_long_only_remove(self):
        bridge = self._make_bridge(long_only_symbols={"BTCUSDT"})
        bridge.update_params("BTCUSDT", long_only=False)
        assert "BTCUSDT" not in bridge._long_only_symbols

    def test_checkpoint_restore(self):
        bridge = self._make_bridge()
        ckpt = bridge.checkpoint()
        assert isinstance(ckpt, dict)
        bridge.restore(ckpt)
        bridge._rust.restore.assert_called_once()

    def test_update_models(self):
        bridge = self._make_bridge()
        new_models = [MagicMock()]
        bridge.update_models(new_models)
        bridge._engine.set_models.assert_called_once_with(new_models)
        bridge._rust.reset.assert_called_once()

    def test_bear_thresholds_must_be_descending(self):
        with patch("alpha.inference.bridge._RustBridge", return_value=_make_rust_bridge_mock()), \
             patch("alpha.inference.bridge.InferenceEngine"):
            from alpha.inference.bridge import LiveInferenceBridge
            with pytest.raises(ValueError, match="descending"):
                LiveInferenceBridge(
                    models=[],
                    bear_thresholds=[(0.5, 1.0), (0.8, 0.5)],
                )

    def test_short_model_path(self):
        bridge = self._make_bridge()
        short_m = MagicMock()
        short_sig = MagicMock()
        short_sig.side = "short"
        short_sig.strength = 0.5
        short_m.predict.return_value = short_sig
        bridge._short_model = short_m
        bridge._rust.process_short_signal.return_value = -0.5
        bridge._engine.run.return_value = []

        ts = datetime.now(timezone.utc)
        out = bridge.enrich("BTCUSDT", ts, {"close": 100.0})
        assert out.get("ml_short_score") == pytest.approx(-0.5)

    def test_short_model_nan_features_skips(self):
        bridge = self._make_bridge()
        short_m = MagicMock()
        bridge._short_model = short_m
        bridge._engine.run.return_value = []

        ts = datetime.now(timezone.utc)
        features = {"close": float("nan")}
        out = bridge.enrich("BTCUSDT", ts, features)
        assert out.get("ml_short_score") == pytest.approx(0.0)
        short_m.predict.assert_not_called()

    def test_resolve_dict_param(self):
        bridge = self._make_bridge(deadzone={"BTCUSDT": 0.7, "ETHUSDT": 0.5})
        assert bridge._resolve(bridge._deadzone, "BTCUSDT", 0.5) == 0.7
        assert bridge._resolve(bridge._deadzone, "SOLANA", 0.5) == 0.5

    def test_resolve_scalar_param(self):
        bridge = self._make_bridge(deadzone=0.6)
        assert bridge._resolve(bridge._deadzone, "anything", 0.5) == 0.6

    def test_resolve_none_returns_default(self):
        bridge = self._make_bridge()
        assert bridge._resolve(None, "BTC", 0.5) == 0.5

    def test_enrich_metrics_gauge_set(self):
        metrics = MagicMock()
        bridge = self._make_bridge(metrics_exporter=metrics)
        bridge._engine.run.return_value = [self._make_entry("long", 0.5)]
        bridge._rust.apply_constraints.return_value = 0.5

        bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
        assert metrics.set_gauge.called

    def test_enrich_ts_none_uses_utcnow(self):
        bridge = self._make_bridge()
        bridge._engine.run.return_value = [self._make_entry("long", 0.5)]
        bridge._rust.apply_constraints.return_value = 0.5
        # Should not raise when ts=None
        out = bridge.enrich("BTCUSDT", None, {"close": 100.0})
        assert "ml_score" in out

    def test_min_hold_zero_returns_raw_score(self):
        bridge = self._make_bridge()
        bridge._engine.run.return_value = [self._make_entry("long", 0.7)]
        bridge._rust.apply_constraints.return_value = 0.7
        # min_hold=0 (default) means constraints not applied via rust
        out = bridge.enrich("BTCUSDT", datetime.now(timezone.utc), {"close": 100.0})
        # apply_constraints only called when min_hold > 0
        assert "ml_score" in out


# ══════════════════════════════════════════════════════════════════════════════
# Section 9: execution/routing/execution_quality.py
# ══════════════════════════════════════════════════════════════════════════════

from execution.routing.execution_quality import (  # noqa: E402
    ExecutionRecord, ExecutionQualityTracker, _slippage_bps,
)


class TestSlippageBps:
    def test_buy_positive_slippage(self):
        slip = _slippage_bps("buy", Decimal("100"), Decimal("101"))
        assert slip == pytest.approx(100.0)

    def test_sell_positive_slippage(self):
        slip = _slippage_bps("sell", Decimal("100"), Decimal("99"))
        assert slip == pytest.approx(100.0)

    def test_zero_intended_price(self):
        slip = _slippage_bps("buy", Decimal("0"), Decimal("100"))
        assert slip == 0.0

    def test_no_slippage(self):
        assert _slippage_bps("buy", Decimal("100"), Decimal("100")) == 0.0

    def test_sell_favorable(self):
        # fill higher than intended → negative slippage for sell
        slip = _slippage_bps("sell", Decimal("100"), Decimal("101"))
        assert slip < 0.0


class TestExecutionQualityTracker:
    def _rec(self, order_id="1", symbol="BTCUSDT", side="buy",
             intended=50000, fill=50050, qty=1, venue="binance",
             submit_ts=0.0, fill_ts=0.01):
        return ExecutionRecord(
            order_id=order_id,
            symbol=symbol,
            side=side,
            intended_qty=Decimal(str(qty)),
            filled_qty=Decimal(str(qty)),
            intended_price=Decimal(str(intended)),
            avg_fill_price=Decimal(str(fill)),
            venue_id=venue,
            submit_ts=submit_ts,
            fill_ts=fill_ts,
        )

    def test_empty_tracker_returns_none(self):
        t = ExecutionQualityTracker()
        assert t.compute_metrics() is None

    def test_record_count(self):
        t = ExecutionQualityTracker()
        t.record(self._rec("1"))
        t.record(self._rec("2"))
        assert t.record_count == 2

    def test_compute_metrics_basic(self):
        t = ExecutionQualityTracker()
        t.record(self._rec(intended=100, fill=101, fill_ts=0.002))
        m = t.compute_metrics()
        assert m is not None
        assert m.total_orders == 1
        assert m.fill_rate == pytest.approx(1.0)
        assert m.avg_slippage_bps > 0.0

    def test_filter_by_venue(self):
        t = ExecutionQualityTracker()
        t.record(self._rec("1", venue="binance"))
        t.record(self._rec("2", venue="bybit"))
        m = t.compute_metrics(venue_id="binance")
        assert m.total_orders == 1

    def test_filter_by_symbol(self):
        t = ExecutionQualityTracker()
        t.record(self._rec("1", symbol="BTCUSDT"))
        t.record(self._rec("2", symbol="ETHUSDT"))
        m = t.compute_metrics(symbol="ETHUSDT")
        assert m.total_orders == 1

    def test_filter_missing_venue_returns_none(self):
        t = ExecutionQualityTracker()
        t.record(self._rec("1"))
        assert t.compute_metrics(venue_id="unknown") is None

    def test_max_history_trims_oldest(self):
        t = ExecutionQualityTracker(max_history=3)
        for i in range(5):
            t.record(self._rec(str(i)))
        assert t.record_count == 3

    def test_should_reduce_size_below_min_records(self):
        t = ExecutionQualityTracker()
        for i in range(3):
            t.record(self._rec(str(i), symbol="BTCUSDT"))
        assert t.should_reduce_size("BTCUSDT") == 1.0

    def test_should_reduce_size_above_threshold(self):
        t = ExecutionQualityTracker()
        for i in range(10):
            t.record(self._rec(str(i), symbol="ETHUSDT", intended=100, fill=105))
        result = t.should_reduce_size("ETHUSDT", threshold_bps=100.0)
        assert result < 1.0

    def test_should_reduce_size_halt(self):
        t = ExecutionQualityTracker()
        for i in range(10):
            t.record(self._rec(str(i), symbol="SOLUSDT", intended=100, fill=110))
        result = t.should_reduce_size("SOLUSDT", threshold_bps=100.0)
        assert result == 0.0

    def test_venue_comparison(self):
        t = ExecutionQualityTracker()
        t.record(self._rec("1", venue="binance"))
        t.record(self._rec("2", venue="bybit"))
        comp = t.venue_comparison()
        assert "binance" in comp
        assert "bybit" in comp

    def test_sell_slippage(self):
        t = ExecutionQualityTracker()
        t.record(self._rec("1", side="sell", intended=100, fill=99, symbol="SOLUSDT"))
        m = t.compute_metrics()
        assert m.avg_slippage_bps == pytest.approx(100.0)

    def test_metrics_worst_slippage(self):
        t = ExecutionQualityTracker()
        t.record(self._rec("1", intended=100, fill=101))
        t.record(self._rec("2", intended=100, fill=103))
        m = t.compute_metrics()
        assert m.worst_slippage_bps == pytest.approx(300.0)

    def test_metrics_median_slippage(self):
        t = ExecutionQualityTracker()
        for fill in [101, 102, 103]:
            t.record(self._rec(str(fill), intended=100, fill=fill))
        m = t.compute_metrics()
        assert m.median_slippage_bps == pytest.approx(200.0)

    def test_latency_computed(self):
        t = ExecutionQualityTracker()
        t.record(self._rec("1", submit_ts=0.0, fill_ts=0.005))
        m = t.compute_metrics()
        assert m.avg_latency_ms == pytest.approx(5.0)

    def test_filter_by_symbol_missing(self):
        t = ExecutionQualityTracker()
        t.record(self._rec("1", symbol="BTCUSDT"))
        assert t.compute_metrics(symbol="UNKNOWN") is None

    def test_total_slippage_cost_positive_for_buy(self):
        t = ExecutionQualityTracker()
        # Buy at 101 vs intended 100 → slippage cost = (101-100)*1 = 1
        t.record(self._rec("1", intended=100, fill=101, qty=1))
        m = t.compute_metrics()
        assert m.total_slippage_cost > 0


# ══════════════════════════════════════════════════════════════════════════════
# Section 10: execution/adapters/binance/mapper_fill.py
# ══════════════════════════════════════════════════════════════════════════════

from execution.adapters.binance.mapper_fill import BinanceFillMapper  # noqa: E402


class TestBinanceFillMapper:
    def _mapper(self):
        return BinanceFillMapper(venue="binance")

    def _futures_payload(self, **overrides):
        base = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1700000000000,
            "o": {
                "s": "BTCUSDT",
                "S": "BUY",
                "i": 123456,
                "c": "client_001",
                "t": 9999,
                "l": "0.001",
                "L": "50000.00",
                "n": "0.05",
                "N": "USDT",
                "T": 1700000000001,
                "m": False,
            },
        }
        for k, v in overrides.items():
            if k.startswith("o."):
                base["o"][k[2:]] = v
            else:
                base[k] = v
        return base

    def _spot_payload(self, **overrides):
        base = {
            "e": "executionReport",
            "s": "ETHUSDT",
            "S": "SELL",
            "i": 654321,
            "c": "client_002",
            "t": 8888,
            "l": "0.5",
            "L": "3000.00",
            "n": "0.30",
            "N": "ETH",
            "T": 1700000000002,
            "m": True,
        }
        base.update(overrides)
        return base

    def _rest_payload(self, **overrides):
        base = {
            "symbol": "SOLUSDT",
            "side": "BUY",
            "orderId": 111222,
            "clientOrderId": "c_003",
            "id": 77777,
            "qty": "10.0",
            "price": "100.0",
            "commission": "0.10",
            "commissionAsset": "SOL",
            "time": 1700000000003,
            "maker": True,
        }
        base.update(overrides)
        return base

    def test_map_futures_ws(self):
        fill = self._mapper().map_fill(self._futures_payload())
        assert fill.symbol == "BTCUSDT"
        assert fill.side == "buy"
        assert fill.qty == Decimal("0.001")
        assert fill.price == Decimal("50000.00")
        assert fill.liquidity == "taker"

    def test_map_spot_ws(self):
        fill = self._mapper().map_fill(self._spot_payload())
        assert fill.symbol == "ETHUSDT"
        assert fill.side == "sell"
        assert fill.liquidity == "maker"

    def test_map_rest_trade(self):
        fill = self._mapper().map_fill(self._rest_payload())
        assert fill.symbol == "SOLUSDT"
        assert fill.side == "buy"
        assert fill.qty == Decimal("10.0")

    def test_unsupported_payload_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            self._mapper().map_fill({"e": "unknown", "foo": "bar"})

    def test_missing_symbol_raises(self):
        payload = self._futures_payload()
        payload["o"]["s"] = None
        with pytest.raises(ValueError):
            self._mapper().map_fill(payload)

    def test_missing_order_id_raises(self):
        payload = self._futures_payload()
        payload["o"]["i"] = None
        with pytest.raises(ValueError, match="order_id"):
            self._mapper().map_fill(payload)

    def test_missing_trade_id_raises(self):
        payload = self._futures_payload()
        payload["o"]["t"] = None
        with pytest.raises(ValueError, match="trade_id"):
            self._mapper().map_fill(payload)

    def test_zero_qty_raises(self):
        payload = self._futures_payload()
        payload["o"]["l"] = "0"
        with pytest.raises(ValueError):
            self._mapper().map_fill(payload)

    def test_negative_qty_raises(self):
        payload = self._futures_payload()
        payload["o"]["l"] = "-1"
        with pytest.raises(ValueError):
            self._mapper().map_fill(payload)

    def test_zero_price_raises(self):
        payload = self._futures_payload()
        payload["o"]["L"] = "0"
        with pytest.raises(ValueError):
            self._mapper().map_fill(payload)

    def test_negative_fee_raises(self):
        payload = self._futures_payload()
        payload["o"]["n"] = "-0.5"
        with pytest.raises(ValueError):
            self._mapper().map_fill(payload)

    def test_zero_fee_ok(self):
        payload = self._futures_payload()
        payload["o"]["n"] = "0"
        fill = self._mapper().map_fill(payload)
        assert fill.fee == Decimal("0")

    def test_none_fee_ok(self):
        payload = self._futures_payload()
        payload["o"]["n"] = None
        fill = self._mapper().map_fill(payload)
        assert fill.fee == Decimal("0")

    def test_bool_qty_raises_type_error(self):
        from execution.adapters.binance.mapper_fill import _dec
        with pytest.raises(TypeError):
            _dec(True, "qty")

    def test_dec_invalid_string(self):
        from execution.adapters.binance.mapper_fill import _dec
        with pytest.raises(ValueError):
            _dec("abc", "qty")

    def test_dec_none_raises(self):
        from execution.adapters.binance.mapper_fill import _dec
        with pytest.raises(ValueError):
            _dec(None, "price")

    def test_int_ms_bool_raises(self):
        from execution.adapters.binance.mapper_fill import _int_ms
        with pytest.raises(TypeError):
            _int_ms(True, "ts_ms")

    def test_norm_side_buy(self):
        from execution.adapters.binance.mapper_fill import _norm_side
        assert _norm_side("BUY") == "buy"
        assert _norm_side("buy") == "buy"
        assert _norm_side("b") == "buy"

    def test_norm_side_sell(self):
        from execution.adapters.binance.mapper_fill import _norm_side
        assert _norm_side("SELL") == "sell"
        assert _norm_side("s") == "sell"

    def test_norm_side_invalid_raises(self):
        from execution.adapters.binance.mapper_fill import _norm_side
        with pytest.raises(ValueError):
            _norm_side("LONG")

    def test_norm_side_none_raises(self):
        from execution.adapters.binance.mapper_fill import _norm_side
        with pytest.raises(ValueError):
            _norm_side(None)

    def test_rest_uses_tradeid_fallback(self):
        payload = self._rest_payload()
        del payload["id"]
        payload["tradeId"] = 99999
        fill = self._mapper().map_fill(payload)
        assert fill.trade_id == "99999"

    def test_futures_missing_o_raises(self):
        payload = {"e": "ORDER_TRADE_UPDATE", "o": None}
        with pytest.raises(ValueError):
            self._mapper().map_fill(payload)

    def test_is_maker_none_liquidity_is_none(self):
        payload = self._futures_payload()
        payload["o"]["m"] = None
        fill = self._mapper().map_fill(payload)
        assert fill.liquidity is None

    def test_maker_true_liquidity(self):
        payload = self._futures_payload()
        payload["o"]["m"] = True
        fill = self._mapper().map_fill(payload)
        assert fill.liquidity == "maker"

    def test_symbol_normalized_uppercase(self):
        payload = self._spot_payload(s="btcusdt")
        payload["s"] = "btcusdt"
        fill = self._mapper().map_fill(payload)
        assert fill.symbol == "BTCUSDT"

    def test_fill_id_is_stable(self):
        payload = self._futures_payload()
        m = self._mapper()
        fill1 = m.map_fill(payload)
        fill2 = m.map_fill(payload)
        assert fill1.fill_id == fill2.fill_id

    def test_rest_is_maker_field(self):
        payload = self._rest_payload()
        payload["isMaker"] = True
        del payload["maker"]
        fill = self._mapper().map_fill(payload)
        assert fill.liquidity == "maker"


# ══════════════════════════════════════════════════════════════════════════════
# Section 11: execution/sim/shadow_adapter.py
# ══════════════════════════════════════════════════════════════════════════════

from execution.sim.shadow_adapter import ShadowExecutionAdapter  # noqa: E402


class TestShadowExecutionAdapter:
    def _make(self, price=Decimal("100.00")):
        return ShadowExecutionAdapter(price_source=lambda sym: price)

    def test_send_order_returns_empty(self):
        adapter = self._make()
        order = MagicMock()
        order.symbol = "BTCUSDT"
        order.side = "BUY"
        order.qty = Decimal("0.1")
        result = adapter.send_order(order)
        assert result == []

    def test_send_order_records_entry(self):
        adapter = self._make(price=Decimal("50000"))
        order = MagicMock()
        order.symbol = "BTCUSDT"
        order.side = "BUY"
        order.qty = Decimal("0.001")
        adapter.send_order(order)
        log = adapter.order_log
        assert len(log) == 1
        assert log[0]["symbol"] == "BTCUSDT"
        assert log[0]["simulated"] is True

    def test_buy_slippage_increases_price(self):
        adapter = self._make(price=Decimal("100"))
        order = MagicMock()
        order.symbol = "BTC"
        order.side = "BUY"
        order.qty = Decimal("1")
        adapter.send_order(order)
        fill_price = Decimal(adapter.order_log[0]["fill_price"])
        assert fill_price > Decimal("100")

    def test_sell_slippage_decreases_price(self):
        adapter = self._make(price=Decimal("100"))
        order = MagicMock()
        order.symbol = "ETH"
        order.side = "SELL"
        order.qty = Decimal("1")
        adapter.send_order(order)
        fill_price = Decimal(adapter.order_log[0]["fill_price"])
        assert fill_price < Decimal("100")

    def test_no_price_records_none(self):
        adapter = ShadowExecutionAdapter(price_source=lambda sym: None)
        order = MagicMock()
        order.symbol = "UNKNOWN"
        order.side = "BUY"
        order.qty = Decimal("1")
        adapter.send_order(order)
        assert adapter.order_log[0]["fill_price"] is None
        assert adapter.order_log[0]["fee"] is None

    def test_order_log_returns_copy(self):
        adapter = self._make()
        order = MagicMock()
        order.symbol = "X"
        order.side = "BUY"
        order.qty = Decimal("1")
        adapter.send_order(order)
        log = adapter.order_log
        log.clear()
        assert len(adapter.order_log) == 1

    def test_multiple_orders(self):
        adapter = self._make()
        for i in range(3):
            order = MagicMock()
            order.symbol = f"SYM{i}"
            order.side = "BUY"
            order.qty = Decimal("1")
            adapter.send_order(order)
        assert len(adapter.order_log) == 3

    def test_fee_is_computed(self):
        adapter = self._make(price=Decimal("1000"))
        order = MagicMock()
        order.symbol = "ETH"
        order.side = "BUY"
        order.qty = Decimal("1")
        adapter.send_order(order)
        fee = Decimal(adapter.order_log[0]["fee"])
        assert fee > Decimal("0")

    def test_ts_recorded(self):
        adapter = self._make()
        order = MagicMock()
        order.symbol = "BTC"
        order.side = "BUY"
        order.qty = Decimal("1")
        adapter.send_order(order)
        assert "ts" in adapter.order_log[0]


# ══════════════════════════════════════════════════════════════════════════════
# Section 12: execution/sim/venue_emulator.py
# ══════════════════════════════════════════════════════════════════════════════

from execution.sim.venue_emulator import VenueEmulator  # noqa: E402
from execution.sim.paper_broker import PaperBrokerConfig  # noqa: E402


class TestVenueEmulator:
    def _make(self, fill_price=Decimal("1000"), auto_fill=True):
        cfg = PaperBrokerConfig(initial_balance=Decimal("100000"))
        return VenueEmulator(
            venue="sim", config=cfg, auto_fill=auto_fill, fill_price=fill_price,
        )

    def test_submit_order_returns_accepted(self):
        em = self._make()
        cmd = MagicMock()
        cmd.symbol = "BTCUSDT"
        cmd.side = "buy"
        cmd.qty = Decimal("0.01")
        cmd.price = Decimal("1000")
        result = em.submit_order(cmd)
        assert result["status"] == "ACCEPTED"
        assert "order_id" in result

    def test_submit_order_records_submitted(self):
        em = self._make()
        cmd = MagicMock()
        cmd.symbol = "ETHUSDT"
        cmd.side = "buy"
        cmd.qty = Decimal("1")
        cmd.price = None
        em.submit_order(cmd)
        assert len(em.submitted) == 1
        assert em.submitted[0]["action"] == "submit"

    def test_submit_auto_fill_fills_order(self):
        em = self._make(fill_price=Decimal("1000"), auto_fill=True)
        cmd = MagicMock()
        cmd.symbol = "BTCUSDT"
        cmd.side = "buy"
        cmd.qty = Decimal("1")
        cmd.price = None
        em.submit_order(cmd)
        assert len(em.broker.fills) == 1

    def test_submit_no_auto_fill(self):
        em = self._make(auto_fill=False)
        cmd = MagicMock()
        cmd.symbol = "BTCUSDT"
        cmd.side = "buy"
        cmd.qty = Decimal("0.001")
        cmd.price = Decimal("1000")
        em.submit_order(cmd)
        assert len(em.broker.fills) == 0

    def test_cancel_order(self):
        em = self._make(auto_fill=False)
        cmd = MagicMock()
        cmd.symbol = "BTCUSDT"
        cmd.side = "buy"
        cmd.qty = Decimal("0.001")
        cmd.price = Decimal("1000")
        result = em.submit_order(cmd)
        order_id = result["order_id"]

        cancel_cmd = MagicMock()
        cancel_cmd.order_id = order_id
        cancel_result = em.cancel_order(cancel_cmd)
        assert cancel_result["status"] == "CANCELED"

    def test_cancel_nonexistent_order(self):
        em = self._make()
        cancel_cmd = MagicMock()
        cancel_cmd.order_id = "nonexistent_999"
        result = em.cancel_order(cancel_cmd)
        assert result["status"] == "FAILED"

    def test_broker_property(self):
        em = self._make()
        assert em.broker is not None

    def test_submitted_returns_copy(self):
        em = self._make()
        cmd = MagicMock()
        cmd.symbol = "BTC"
        cmd.side = "buy"
        cmd.qty = Decimal("1")
        cmd.price = None
        em.submit_order(cmd)
        s = em.submitted
        s.clear()
        assert len(em.submitted) == 1

    def test_default_fill_price_is_100(self):
        em = VenueEmulator()
        assert em._fill_price == Decimal("100")


# ══════════════════════════════════════════════════════════════════════════════
# Section 13: runner/live_paper_runner.py
# ══════════════════════════════════════════════════════════════════════════════

class TestLivePaperConfig:
    def test_defaults(self):
        from runner.live_paper_runner import LivePaperConfig
        cfg = LivePaperConfig()
        assert cfg.symbols == ("BTCUSDT",)
        assert cfg.starting_balance == 10000.0
        assert cfg.fee_bps == 4.0
        assert cfg.enable_monitoring is True

    def test_testnet_flag(self):
        from runner.live_paper_runner import LivePaperConfig
        cfg = LivePaperConfig(testnet=True)
        assert cfg.testnet is True

    def test_custom_symbols(self):
        from runner.live_paper_runner import LivePaperConfig
        cfg = LivePaperConfig(symbols=("ETHUSDT", "BTCUSDT"))
        assert len(cfg.symbols) == 2


class TestLivePaperRunnerUnit:
    def _make_runner(self, *, with_bridge=False, bridge_is_dict=False, with_ckpt=True):
        from runner.live_paper_runner import LivePaperRunner

        loop = MagicMock()
        coordinator = MagicMock()
        coordinator.get_state_view.return_value = {"event_index": 42, "phase": "running"}
        runtime = MagicMock()
        health = MagicMock()

        runner = object.__new__(LivePaperRunner)
        runner.loop = loop
        runner.coordinator = coordinator
        runner.runtime = runtime
        runner.health = health
        runner._fills = []
        runner._bar_count = 0
        runner._running = False
        runner._ckpt_stop = threading.Event()

        if with_bridge:
            if bridge_is_dict:
                mock_bridge_a = MagicMock()
                mock_bridge_a.checkpoint.return_value = {"data": "A"}
                mock_bridge_a.restore = MagicMock()
                mock_bridge_b = MagicMock()
                mock_bridge_b.checkpoint.return_value = {"data": "B"}
                mock_bridge_b.restore = MagicMock()
                runner.inference_bridge = {"BTC": mock_bridge_a, "ETH": mock_bridge_b}
            else:
                mock_bridge = MagicMock()
                mock_bridge.checkpoint.return_value = {"data": "ok"}
                mock_bridge.restore = MagicMock()
                runner.inference_bridge = mock_bridge
        else:
            runner.inference_bridge = None

        if with_ckpt and with_bridge:
            runner.checkpoint_path = MagicMock(spec=Path)
            runner.checkpoint_path.exists.return_value = True
            runner.checkpoint_path.parent = MagicMock()
        else:
            runner.checkpoint_path = None

        runner._on_fill = MagicMock()
        return runner

    def test_fills_property(self):
        runner = self._make_runner()
        runner._fills = [{"id": "1"}, {"id": "2"}]
        fills = runner.fills
        assert len(fills) == 2

    def test_event_index(self):
        runner = self._make_runner()
        assert runner.event_index == 42

    def test_log_status(self):
        runner = self._make_runner()
        runner._log_status()
        runner.coordinator.get_state_view.assert_called()

    def test_save_checkpoint_no_bridge(self):
        runner = self._make_runner(with_bridge=False)
        runner._save_checkpoint()  # No-op

    def test_save_checkpoint_with_bridge(self):
        runner = self._make_runner(with_bridge=True, with_ckpt=True)
        with patch("builtins.open", MagicMock()), patch("json.dump"):
            runner._save_checkpoint()
        runner.inference_bridge.checkpoint.assert_called()

    def test_save_checkpoint_dict_bridge(self):
        runner = self._make_runner(with_bridge=True, bridge_is_dict=True, with_ckpt=True)
        with patch("builtins.open", MagicMock()), patch("json.dump"):
            runner._save_checkpoint()
        for b in runner.inference_bridge.values():
            b.checkpoint.assert_called()

    def test_save_checkpoint_exception_swallowed(self):
        runner = self._make_runner(with_bridge=True, with_ckpt=True)
        runner.inference_bridge.checkpoint.side_effect = RuntimeError("disk full")
        runner._save_checkpoint()  # Should not raise

    def test_restore_checkpoint_no_bridge(self):
        runner = self._make_runner(with_bridge=False)
        runner._restore_checkpoint()

    def test_restore_checkpoint_no_path(self):
        runner = self._make_runner(with_bridge=True)
        runner.checkpoint_path = None
        runner._restore_checkpoint()

    def test_restore_checkpoint_missing_file(self):
        runner = self._make_runner(with_bridge=True, with_ckpt=True)
        runner.checkpoint_path.exists.return_value = False
        runner._restore_checkpoint()
        runner.inference_bridge.restore.assert_not_called()

    def test_restore_checkpoint_dict_bridge(self):
        runner = self._make_runner(with_bridge=True, bridge_is_dict=True, with_ckpt=True)
        data = {"BTC": {"data": "A"}, "ETH": {"data": "B"}}
        with patch("builtins.open", MagicMock()), patch("json.load", return_value=data):
            runner._restore_checkpoint()
        for b in runner.inference_bridge.values():
            b.restore.assert_called_once()

    def test_restore_checkpoint_exception_swallowed(self):
        runner = self._make_runner(with_bridge=True, with_ckpt=True)
        with patch("builtins.open", side_effect=IOError("read failed")):
            runner._restore_checkpoint()

    def test_restore_checkpoint_single_bridge(self):
        runner = self._make_runner(with_bridge=True, with_ckpt=True)
        data = {"data": "ok"}
        with patch("builtins.open", MagicMock()), patch("json.load", return_value=data):
            runner._restore_checkpoint()
        runner.inference_bridge.restore.assert_called_once_with(data)

    def test_stop_idempotent(self):
        runner = self._make_runner()
        runner._running = False
        runner.stop()
        runner.runtime.stop.assert_not_called()

    def test_stop_shuts_down_all_subsystems(self):
        runner = self._make_runner()
        runner._running = True
        runner.stop()
        runner.runtime.stop.assert_called_once()
        runner.loop.stop_background.assert_called_once()
        runner.coordinator.stop.assert_called_once()
        runner.health.stop.assert_called_once()

    def test_stop_without_health(self):
        runner = self._make_runner()
        runner.health = None
        runner._running = True
        runner.stop()
        runner.runtime.stop.assert_called_once()

    def test_apply_perf_tuning_no_raise(self):
        from runner.live_paper_runner import LivePaperRunner
        with patch("builtins.open", side_effect=FileNotFoundError()), \
             patch("os.sched_setaffinity", side_effect=OSError("not permitted")), \
             patch("os.cpu_count", return_value=2), \
             patch("os.nice", side_effect=PermissionError("not root")):
            LivePaperRunner._apply_perf_tuning()

    def test_apply_perf_tuning_with_nohz_cpus_range(self):
        from runner.live_paper_runner import LivePaperRunner
        mock_file = MagicMock()
        mock_file.__enter__ = lambda s: s
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = "2-3"
        with patch("builtins.open", return_value=mock_file), \
             patch("os.sched_setaffinity") as mock_aff, \
             patch("os.nice"):
            LivePaperRunner._apply_perf_tuning()
        mock_aff.assert_called_once_with(0, {2, 3})

    def test_apply_perf_tuning_with_single_nohz_cpu(self):
        from runner.live_paper_runner import LivePaperRunner
        mock_file = MagicMock()
        mock_file.__enter__ = lambda s: s
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = "4"
        with patch("builtins.open", return_value=mock_file), \
             patch("os.sched_setaffinity") as mock_aff, \
             patch("os.nice"):
            LivePaperRunner._apply_perf_tuning()
        mock_aff.assert_called_once_with(0, {4})

    def test_apply_perf_tuning_no_nohz_multi_cpu(self):
        from runner.live_paper_runner import LivePaperRunner
        with patch("builtins.open", side_effect=FileNotFoundError()), \
             patch("os.cpu_count", return_value=4), \
             patch("os.sched_setaffinity") as mock_aff, \
             patch("os.nice"):
            LivePaperRunner._apply_perf_tuning()
        mock_aff.assert_called_once_with(0, {3})

    def test_apply_perf_tuning_single_cpu_no_affinity(self):
        from runner.live_paper_runner import LivePaperRunner
        with patch("builtins.open", side_effect=FileNotFoundError()), \
             patch("os.cpu_count", return_value=1), \
             patch("os.sched_setaffinity") as mock_aff, \
             patch("os.nice"):
            LivePaperRunner._apply_perf_tuning()
        mock_aff.assert_not_called()

    def test_checkpoint_loop_stops_immediately(self):
        runner = self._make_runner(with_bridge=True, with_ckpt=True)
        runner._ckpt_stop.set()
        with patch.object(runner, "_save_checkpoint"):
            runner._checkpoint_loop()

    def test_class_has_required_methods(self):
        from runner.live_paper_runner import LivePaperRunner
        assert hasattr(LivePaperRunner, "build")
        assert hasattr(LivePaperRunner, "start")
        assert hasattr(LivePaperRunner, "stop")
        assert hasattr(LivePaperRunner, "_restore_checkpoint")
        assert hasattr(LivePaperRunner, "_save_checkpoint")
        assert hasattr(LivePaperRunner, "_apply_perf_tuning")

    def test_fills_returns_list_copy(self):
        runner = self._make_runner()
        runner._fills = [{"a": 1}]
        fills = runner.fills
        fills.append({"b": 2})
        assert len(runner._fills) == 1  # original unchanged
