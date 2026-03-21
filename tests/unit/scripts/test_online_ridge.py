"""Tests for OnlineRidge — recursive least squares with forgetting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pytest

from alpha.online_ridge import OnlineRidge


class TestOnlineRidge:
    def test_init(self):
        ridge = OnlineRidge(n_features=5)
        assert ridge.n_updates == 0
        assert ridge.predict(np.zeros(5)) == 0.0

    def test_load_from_weights(self):
        ridge = OnlineRidge(n_features=3)
        ridge.load_from_weights(np.array([1.0, 2.0, 3.0]), intercept=0.5)
        # Predict: [1,0,0] → 1*1 + 0.5 = 1.5
        assert abs(ridge.predict(np.array([1.0, 0.0, 0.0])) - 1.5) < 1e-10

    def test_load_from_weights_wrong_size(self):
        ridge = OnlineRidge(n_features=3)
        with pytest.raises(ValueError):
            ridge.load_from_weights(np.array([1.0, 2.0]))

    def test_predict_batch(self):
        ridge = OnlineRidge(n_features=2)
        ridge.load_from_weights(np.array([1.0, -1.0]), intercept=0.0)
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        preds = ridge.predict_batch(X)
        np.testing.assert_allclose(preds, [1.0, -1.0, 0.0])

    def test_update_converges(self):
        """Online RLS should converge to true weights on linear data."""
        np.random.seed(42)
        n_features = 3
        true_w = np.array([0.5, -0.3, 0.1])
        true_intercept = 0.02

        ridge = OnlineRidge(
            n_features=n_features,
            forgetting_factor=0.99,
            min_samples_for_update=10,
        )
        ridge.load_from_weights(np.zeros(n_features), intercept=0.0)

        # Generate data and update
        errors = []
        for _ in range(500):
            x = np.random.randn(n_features)
            y = np.dot(true_w, x) + true_intercept + np.random.randn() * 0.01
            err = ridge.update(x, y)
            errors.append(abs(err))

        # Weights should be close to true weights
        w = ridge.weights
        assert np.allclose(w, true_w, atol=0.1), f"Weights {w} not close to {true_w}"
        # Later errors should be smaller than early errors
        assert np.mean(errors[-50:]) < np.mean(errors[:50])

    def test_no_update_before_min_samples(self):
        ridge = OnlineRidge(n_features=2, min_samples_for_update=100)
        ridge.load_from_weights(np.array([1.0, 0.0]))
        initial_w = ridge.weights.copy()

        for _ in range(50):
            ridge.update(np.array([1.0, 0.0]), 0.5)

        # Weights should not change before min_samples
        np.testing.assert_array_equal(ridge.weights, initial_w)

    def test_nan_skip(self):
        ridge = OnlineRidge(n_features=2, min_samples_for_update=1)
        ridge.load_from_weights(np.array([1.0, 0.0]))
        initial_w = ridge.weights.copy()

        ridge.update(np.array([float("nan"), 1.0]), 1.0)
        np.testing.assert_array_equal(ridge.weights, initial_w)

        ridge.update(np.array([1.0, 0.0]), float("nan"))
        np.testing.assert_array_equal(ridge.weights, initial_w)

    def test_reset_to_static(self):
        ridge = OnlineRidge(n_features=2, min_samples_for_update=1)
        ridge.load_from_weights(np.array([1.0, 0.0]), intercept=0.5)

        # Update a few times
        for _ in range(10):
            ridge.update(np.array([1.0, 1.0]), 2.0)

        assert ridge.n_updates == 10
        assert ridge.weight_drift > 0

        # Reset
        ridge.reset_to_static()
        assert ridge.n_updates == 0
        np.testing.assert_array_equal(ridge.weights, np.array([1.0, 0.0]))
        assert ridge.intercept == 0.5

    def test_weight_drift(self):
        ridge = OnlineRidge(n_features=2, min_samples_for_update=1)
        ridge.load_from_weights(np.array([1.0, 0.0]))
        assert ridge.weight_drift == 0.0

        for _ in range(20):
            ridge.update(np.array([1.0, 0.0]), 2.0)

        assert ridge.weight_drift > 0

    def test_max_update_magnitude(self):
        """Large errors should be clamped."""
        ridge = OnlineRidge(
            n_features=2,
            min_samples_for_update=1,
            max_update_magnitude=0.01,
        )
        ridge.load_from_weights(np.array([0.0, 0.0]))

        # Single huge update
        ridge.update(np.array([1.0, 0.0]), 1000.0)

        # Weight change should be bounded
        assert abs(ridge.weights[0]) <= 0.02  # clamped

    def test_stats(self):
        ridge = OnlineRidge(n_features=3)
        ridge.load_from_weights(np.array([0.1, 0.2, 0.3]), intercept=0.01)
        stats = ridge.stats
        assert stats["n_updates"] == 0
        assert stats["weight_drift"] == 0.0
        assert stats["forgetting_factor"] == 0.997
