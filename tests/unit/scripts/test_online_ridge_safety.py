"""Safety tests for OnlineRidge — numerical stability, edge cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
from alpha.online_ridge import OnlineRidge


class TestOnlineRidgeNumericalSafety:
    def test_all_nan_update(self):
        """All-NaN input should be silently skipped."""
        ridge = OnlineRidge(n_features=3, min_samples_for_update=1)
        ridge.load_from_weights(np.array([1.0, 0.0, 0.0]))
        initial_w = ridge.weights.copy()
        ridge.update(np.array([float("nan")] * 3), 1.0)
        np.testing.assert_array_equal(ridge.weights, initial_w)

    def test_inf_input(self):
        """Inf input should be skipped (NaN check catches inf after dot product)."""
        ridge = OnlineRidge(n_features=2, min_samples_for_update=1)
        ridge.load_from_weights(np.array([1.0, 0.0]))
        ridge.update(np.array([float("inf"), 0.0]), 1.0)
        # Weights should still be finite
        assert np.all(np.isfinite(ridge.weights))

    def test_zero_denominator_safe(self):
        """When x·P·x ≈ 0, update should not crash."""
        ridge = OnlineRidge(n_features=2, min_samples_for_update=1)
        ridge.load_from_weights(np.zeros(2))
        # Zero feature vector
        ridge.update(np.array([0.0, 0.0]), 1.0)
        assert np.all(np.isfinite(ridge.weights))

    def test_very_large_target(self):
        """Large target values should be clamped by max_update_magnitude."""
        ridge = OnlineRidge(
            n_features=2, min_samples_for_update=1,
            max_update_magnitude=0.01,
        )
        ridge.load_from_weights(np.array([0.0, 0.0]))
        ridge.update(np.array([1.0, 0.0]), 1e10)
        # Weight change should be bounded
        assert np.linalg.norm(ridge.weights) < 0.1

    def test_p_matrix_stays_positive_definite(self):
        """P matrix should remain positive semi-definite after many updates."""
        np.random.seed(42)
        ridge = OnlineRidge(n_features=5, min_samples_for_update=1)
        ridge.load_from_weights(np.zeros(5))
        for _ in range(1000):
            x = np.random.randn(5) * 10
            y = np.random.randn() * 0.1
            ridge.update(x, y)
        # P should be symmetric and positive semi-definite
        P = ridge._P
        assert np.allclose(P, P.T, atol=1e-10), "P not symmetric"
        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues >= -1e-8), f"P not PSD: min eigenvalue = {eigenvalues.min()}"

    def test_weights_dont_explode(self):
        """After many updates, weights should remain bounded."""
        np.random.seed(123)
        ridge = OnlineRidge(n_features=3, min_samples_for_update=1)
        ridge.load_from_weights(np.zeros(3))
        for _ in range(5000):
            x = np.random.randn(3)
            y = np.random.randn() * 0.01
            ridge.update(x, y)
        assert np.linalg.norm(ridge.weights) < 10, "Weights exploded"

    def test_predict_with_nan_features(self):
        """Predict with NaN features should return NaN, not crash."""
        ridge = OnlineRidge(n_features=2)
        ridge.load_from_weights(np.array([1.0, 0.0]))
        result = ridge.predict(np.array([float("nan"), 1.0]))
        assert np.isnan(result)

    def test_concurrent_predict_update(self):
        """Simulate concurrent predict + update (no locks — should not crash)."""
        import threading
        ridge = OnlineRidge(n_features=3, min_samples_for_update=1)
        ridge.load_from_weights(np.array([0.1, 0.2, 0.3]))
        errors = []

        def updater():
            try:
                for _ in range(100):
                    ridge.update(np.random.randn(3), np.random.randn() * 0.01)
            except Exception as e:
                errors.append(e)

        def predictor():
            try:
                for _ in range(100):
                    ridge.predict(np.random.randn(3))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater), threading.Thread(target=predictor)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # May have race conditions but should not crash
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
