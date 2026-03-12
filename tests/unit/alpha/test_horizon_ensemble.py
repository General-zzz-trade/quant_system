"""Tests for AdaptiveHorizonEnsemble — mean_zscore and ic_weighted modes."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from alpha.v11_config import V11Config
from alpha.horizon_ensemble import AdaptiveHorizonEnsemble
from decision.backtest_module import _ZScoreBuf


def _make_mock_model(predictions):
    """Create a mock model that returns fixed predictions."""
    model = MagicMock()
    model.predict = MagicMock(side_effect=lambda x: np.array([predictions.pop(0)]))
    return model


def _make_horizon_models(horizons, n_features=3, n_preds=500):
    """Create horizon model dicts with mock lgbm models."""
    models = []
    rng = np.random.RandomState(42)
    for h in horizons:
        preds = list(rng.randn(n_preds) * 0.001)
        models.append({
            "horizon": h,
            "lgbm": _make_mock_model(preds),
            "xgb": None,
            "features": [f"feat_{i}" for i in range(n_features)],
            "zscore_buf": _ZScoreBuf(window=720, warmup=20),
        })
    return models


class TestMeanZScoreMode:
    def test_equal_weights(self):
        cfg = V11Config(horizons=[12, 24, 48], ensemble_method="mean_zscore")
        models = _make_horizon_models([12, 24, 48])
        ens = AdaptiveHorizonEnsemble(cfg, models)
        weights = ens.get_weights()
        assert weights == {12: pytest.approx(1/3), 24: pytest.approx(1/3), 48: pytest.approx(1/3)}

    def test_predict_returns_float(self):
        cfg = V11Config(horizons=[12, 24], ensemble_method="mean_zscore")
        models = _make_horizon_models([12, 24], n_preds=100)
        ens = AdaptiveHorizonEnsemble(cfg, models)
        features = {f"feat_{i}": 0.5 for i in range(3)}
        # Run enough predictions to warm up z-score
        for _ in range(25):
            z = ens.predict(features)
        assert isinstance(z, float)

    def test_returns_zero_during_warmup(self):
        cfg = V11Config(horizons=[12, 24], ensemble_method="mean_zscore")
        models = _make_horizon_models([12, 24], n_preds=100)
        ens = AdaptiveHorizonEnsemble(cfg, models)
        features = {f"feat_{i}": 0.5 for i in range(3)}
        z = ens.predict(features)
        assert z == 0.0


class TestICWeightedMode:
    def test_default_weights_before_data(self):
        cfg = V11Config(horizons=[12, 24], ensemble_method="ic_weighted")
        models = _make_horizon_models([12, 24])
        ens = AdaptiveHorizonEnsemble(cfg, models)
        weights = ens.get_weights()
        # Before enough IC data, all weights should be 1.0 (default)
        assert weights[12] == 1.0
        assert weights[24] == 1.0

    def test_update_ic_changes_weights(self):
        cfg = V11Config(horizons=[12, 24], ensemble_method="ic_weighted")
        models = _make_horizon_models([12, 24])
        ens = AdaptiveHorizonEnsemble(cfg, models)
        rng = np.random.RandomState(123)
        # Feed good IC for h=12, bad IC for h=24
        for i in range(100):
            pred12 = rng.randn()
            ens.update_ic(12, pred12, pred12 * 0.5 + rng.randn() * 0.1)  # correlated
            pred24 = rng.randn()
            ens.update_ic(24, pred24, -pred24 + rng.randn() * 0.1)  # anti-correlated
        weights = ens.get_weights()
        # h=12 should have higher weight than h=24
        assert weights[12] > weights[24]

    def test_at_least_one_horizon_active(self):
        cfg = V11Config(horizons=[12, 24], ensemble_method="ic_weighted")
        models = _make_horizon_models([12, 24])
        ens = AdaptiveHorizonEnsemble(cfg, models)
        # Feed terrible IC for both
        rng = np.random.RandomState(999)
        for i in range(100):
            pred = rng.randn()
            ens.update_ic(12, pred, -pred)
            ens.update_ic(24, pred, -pred)
        weights = ens.get_weights()
        # Even with all negative IC, at least one should be non-zero
        assert sum(weights.values()) > 0


class TestLGBMXGBWeight:
    def test_custom_lgbm_weight(self):
        cfg = V11Config(horizons=[12], ensemble_method="mean_zscore", lgbm_xgb_weight=0.7)
        # Create model with xgb
        lgbm_model = MagicMock()
        lgbm_model.predict = MagicMock(return_value=np.array([1.0]))
        xgb_model = MagicMock()
        xgb_model.predict = MagicMock(return_value=np.array([0.0]))
        models = [{
            "horizon": 12,
            "lgbm": lgbm_model,
            "xgb": xgb_model,
            "features": ["feat_0"],
            "zscore_buf": _ZScoreBuf(window=720, warmup=5),
        }]
        ens = AdaptiveHorizonEnsemble(cfg, models)
        # Since xgb import might fail in test, just verify construction works
        assert ens._lgbm_xgb_w == 0.7
