"""Tests for NN ensemble integration in walk-forward validation."""
import numpy as np
import pytest


class TestComputeEnsembleWeights:
    def setup_method(self):
        from scripts.walkforward_validate import _compute_ensemble_weights
        self.fn = _compute_ensemble_weights

    def test_positive_ics_weighted(self):
        np.random.seed(42)
        y = np.random.randn(100)
        # pred1 correlates well, pred2 is noise
        pred1 = y + np.random.randn(100) * 0.5
        pred2 = np.random.randn(100)
        weights = self.fn([pred1, pred2], y)
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-10
        assert weights[0] > weights[1]  # pred1 has better IC

    def test_all_zero_ic_equal_weights(self):
        y = np.random.randn(100)
        # Both have zero correlation
        pred1 = np.ones(100)  # constant → zero IC
        pred2 = np.ones(100) * 2
        weights = self.fn([pred1, pred2], y)
        assert abs(weights[0] - 0.5) < 1e-10
        assert abs(weights[1] - 0.5) < 1e-10

    def test_single_model(self):
        y = np.random.randn(50)
        pred = y * 2
        weights = self.fn([pred], y)
        assert len(weights) == 1
        assert abs(weights[0] - 1.0) < 1e-10

    def test_too_few_samples(self):
        y = np.random.randn(5)  # < 10 threshold
        pred1 = np.random.randn(5)
        pred2 = np.random.randn(5)
        weights = self.fn([pred1, pred2], y)
        # Should fallback to equal weights
        assert abs(weights[0] - 0.5) < 1e-10

    def test_nan_in_target(self):
        y = np.concatenate([np.random.randn(50), np.full(5, np.nan)])
        pred1 = np.random.randn(55)
        pred2 = np.random.randn(55)
        weights = self.fn([pred1, pred2], y)
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 1e-10


@pytest.mark.slow
class TestApplyNnEnsemble:
    def test_no_torch_returns_lgbm(self):
        """When torch is not available, should return LGBM predictions unchanged."""
        from scripts import walkforward_validate as wf

        lgbm_pred = np.random.randn(100)
        X_train = np.random.randn(500, 10)
        y_train = np.random.randn(500)
        X_test = np.random.randn(100, 10)

        original_flag = wf._HAS_TORCH
        try:
            wf._HAS_TORCH = False
            result = wf._apply_nn_ensemble(lgbm_pred, X_train, y_train, X_test)
            np.testing.assert_array_equal(result, lgbm_pred)
        finally:
            wf._HAS_TORCH = original_flag

    def test_too_short_data_returns_lgbm(self):
        """When data is shorter than seq_len, should return LGBM predictions."""
        from scripts import walkforward_validate as wf

        if not wf._HAS_TORCH:
            pytest.skip("torch not available")

        lgbm_pred = np.random.randn(10)
        X_train = np.random.randn(15, 5)  # < default seq_len=20
        y_train = np.random.randn(15)
        X_test = np.random.randn(10, 5)

        result = wf._apply_nn_ensemble(lgbm_pred, X_train, y_train, X_test, nn_seq_len=20)
        np.testing.assert_array_equal(result, lgbm_pred)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_output_length_matches_lgbm(self):
        """NN ensemble output must have same length as LGBM predictions."""
        from scripts import walkforward_validate as wf

        np.random.seed(123)
        n_train, n_test, n_feat = 500, 100, 8
        lgbm_pred = np.random.randn(n_test)
        X_train = np.random.randn(n_train, n_feat)
        y_train = np.random.randn(n_train)
        X_test = np.random.randn(n_test, n_feat)

        result = wf._apply_nn_ensemble(
            lgbm_pred, X_train, y_train, X_test,
            nn_seq_len=5, nn_epochs=2,
        )
        assert len(result) == len(lgbm_pred)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_ensemble_differs_from_lgbm_only(self):
        """With enough data, ensemble should differ from pure LGBM (not identical)."""
        from scripts import walkforward_validate as wf

        np.random.seed(42)
        n_train, n_test, n_feat = 300, 50, 5
        lgbm_pred = np.random.randn(n_test)
        X_train = np.random.randn(n_train, n_feat)
        y_train = np.random.randn(n_train)
        X_test = np.random.randn(n_test, n_feat)

        result = wf._apply_nn_ensemble(
            lgbm_pred, X_train, y_train, X_test,
            nn_seq_len=5, nn_epochs=2,
        )
        # At least some predictions should differ
        assert not np.allclose(result, lgbm_pred, atol=1e-6)
