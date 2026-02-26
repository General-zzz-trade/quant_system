"""Tests for stacking ensemble and meta-learner."""
from __future__ import annotations

import pytest

from decision.ensemble.meta_learner import LinearMetaLearner
from decision.ensemble.stacking import StackingConfig, StackingEnsemble


class TestStackingEnsemble:
    def test_fit_and_predict_linear(self):
        config = StackingConfig(
            base_signal_names=("sig_a", "sig_b"),
            meta_method="linear",
        )
        ensemble = StackingEnsemble(config)

        signals = {
            "sig_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sig_b": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
        target = [3.0, 3.0, 3.0, 3.0, 3.0]

        ensemble.fit(signals, target)
        assert ensemble.is_fitted

        pred = ensemble.predict({"sig_a": 3.0, "sig_b": 3.0})
        assert pred == pytest.approx(3.0, abs=0.1)

    def test_fit_and_predict_ridge(self):
        config = StackingConfig(
            base_signal_names=("x",),
            meta_method="ridge",
        )
        ensemble = StackingEnsemble(config)

        signals = {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}
        target = [2.0, 4.0, 6.0, 8.0, 10.0]

        ensemble.fit(signals, target)
        pred = ensemble.predict({"x": 3.0})
        assert pred == pytest.approx(6.0, abs=0.5)

    def test_equal_weight_method(self):
        config = StackingConfig(
            base_signal_names=("a", "b", "c"),
            meta_method="mean",
        )
        ensemble = StackingEnsemble(config)
        ensemble.fit({}, [])

        pred = ensemble.predict({"a": 3.0, "b": 6.0, "c": 9.0})
        assert pred == pytest.approx(6.0, abs=1e-6)

    def test_predict_before_fit_raises(self):
        config = StackingConfig(base_signal_names=("a",))
        ensemble = StackingEnsemble(config)

        with pytest.raises(RuntimeError, match="fit"):
            ensemble.predict({"a": 1.0})

    def test_weights_accessible(self):
        config = StackingConfig(
            base_signal_names=("x", "y"),
            meta_method="mean",
        )
        ensemble = StackingEnsemble(config)
        ensemble.fit({}, [])

        weights = ensemble.weights
        assert "x" in weights
        assert "y" in weights
        assert weights["x"] == pytest.approx(0.5)

    def test_unknown_meta_method_raises(self):
        config = StackingConfig(
            base_signal_names=("a",),
            meta_method="xgboost",
        )
        ensemble = StackingEnsemble(config)
        with pytest.raises(ValueError, match="Unknown meta_method"):
            ensemble.fit({"a": [1.0]}, [1.0])

    def test_single_signal_linear(self):
        config = StackingConfig(
            base_signal_names=("sig",),
            meta_method="linear",
        )
        ensemble = StackingEnsemble(config)

        signals = {"sig": [0.0, 1.0, 2.0, 3.0, 4.0]}
        target = [0.0, 2.0, 4.0, 6.0, 8.0]

        ensemble.fit(signals, target)
        pred = ensemble.predict({"sig": 5.0})
        assert pred == pytest.approx(10.0, abs=0.5)


class TestLinearMetaLearner:
    def test_fit_and_predict(self):
        ml = LinearMetaLearner(regularization=0.0)
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        ml.fit(X, y)
        pred = ml.predict([3.0])
        assert pred == pytest.approx(6.0, abs=0.1)

    def test_predict_before_fit_raises(self):
        ml = LinearMetaLearner()
        with pytest.raises(RuntimeError, match="fit"):
            ml.predict([1.0])

    def test_multi_feature(self):
        ml = LinearMetaLearner(regularization=0.001)
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]
        y = [1.0, 1.0, 2.0, 2.0, 2.0]

        ml.fit(X, y)
        pred = ml.predict([1.0, 1.0])
        assert pred == pytest.approx(2.0, abs=0.3)

    def test_weights_and_bias_accessible(self):
        ml = LinearMetaLearner(regularization=0.0)
        X = [[1.0], [2.0], [3.0]]
        y = [1.0, 2.0, 3.0]

        ml.fit(X, y)
        assert len(ml.weights) == 1
        assert ml.weights[0] == pytest.approx(1.0, abs=0.1)
        assert ml.bias == pytest.approx(0.0, abs=0.1)

    def test_regularization_shrinks_weights(self):
        ml_noreg = LinearMetaLearner(regularization=0.0)
        ml_reg = LinearMetaLearner(regularization=1.0)

        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        ml_noreg.fit(X, y)
        ml_reg.fit(X, y)

        assert abs(ml_reg.weights[0]) <= abs(ml_noreg.weights[0]) + 0.01
