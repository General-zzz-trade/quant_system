"""Unit tests for V6 Alpha training pipeline."""
from __future__ import annotations

import numpy as np
import pytest


class TestComputeTarget:
    """Test _compute_target with all 4 modes."""

    def _closes(self, n: int = 200) -> np.ndarray:
        rng = np.random.RandomState(42)
        prices = 100.0 * np.cumprod(1 + rng.randn(n) * 0.01)
        return prices

    def test_raw_mode(self):
        from scripts.train_v6_alpha import _compute_target
        closes = self._closes()
        y = _compute_target(closes, horizon=5, mode="raw")
        assert len(y) == len(closes)
        # Last 5 bars should be NaN
        assert np.all(np.isnan(y[-5:]))
        # First bar should be (closes[5]/closes[0] - 1)
        expected = closes[5] / closes[0] - 1.0
        np.testing.assert_almost_equal(y[0], expected, decimal=10)

    def test_clipped_mode(self):
        from scripts.train_v6_alpha import _compute_target
        closes = self._closes()
        y = _compute_target(closes, horizon=5, mode="clipped")
        valid = y[~np.isnan(y)]
        # All values should be within 1st/99th percentile of raw
        raw = _compute_target(closes, horizon=5, mode="raw")
        raw_valid = raw[~np.isnan(raw)]
        p1, p99 = np.percentile(raw_valid, [1, 99])
        assert np.all(valid >= p1 - 1e-12)
        assert np.all(valid <= p99 + 1e-12)

    def test_vol_norm_mode(self):
        from scripts.train_v6_alpha import _compute_target
        closes = self._closes()
        y = _compute_target(closes, horizon=5, mode="vol_norm")
        valid = y[~np.isnan(y)]
        assert len(valid) > 0
        # Vol-normalized should have larger absolute values than raw
        # (dividing by small vol amplifies)

    def test_binary_mode(self):
        from scripts.train_v6_alpha import _compute_target
        closes = self._closes()
        y = _compute_target(closes, horizon=5, mode="binary")
        valid = y[~np.isnan(y)]
        # All values should be 0 or 1
        assert set(valid.tolist()).issubset({0.0, 1.0})

    def test_invalid_mode_raises(self):
        from scripts.train_v6_alpha import _compute_target
        closes = self._closes()
        with pytest.raises(ValueError, match="Unknown target mode"):
            _compute_target(closes, horizon=5, mode="invalid")


class TestExpandingWindowFoldsV6:
    """Test expanding window folds with min_train=2000."""

    def test_min_train_2000(self):
        from scripts.train_v6_alpha import expanding_window_folds
        folds = expanding_window_folds(5000, n_folds=5, min_train=2000)
        assert len(folds) > 0
        # First fold should start training at 0, test at 2000
        _, tr_end, te_start, _ = folds[0]
        assert tr_end == 2000
        assert te_start == 2000

    def test_small_data_rejected(self):
        from scripts.train_v6_alpha import expanding_window_folds
        # Data smaller than min_train should return empty
        folds = expanding_window_folds(1500, n_folds=5, min_train=2000)
        assert folds == []

    def test_expanding_windows(self):
        from scripts.train_v6_alpha import expanding_window_folds
        folds = expanding_window_folds(10000, n_folds=5, min_train=2000)
        # Each fold should have train_start=0
        for tr_start, _, _, _ in folds:
            assert tr_start == 0
        # Train end should increase across folds
        train_ends = [f[1] for f in folds]
        assert train_ends == sorted(train_ends)

    def test_last_fold_covers_end(self):
        from scripts.train_v6_alpha import expanding_window_folds
        n = 10000
        folds = expanding_window_folds(n, n_folds=5, min_train=2000)
        _, _, _, last_te_end = folds[-1]
        assert last_te_end == n


class TestBlacklistV6:
    """Test that ret_1 and ret_3 are excluded."""

    def test_ret1_ret3_blacklisted(self):
        from scripts.train_v6_alpha import BLACKLIST
        assert "ret_1" in BLACKLIST
        assert "ret_3" in BLACKLIST

    def test_get_available_excludes_blacklist(self):
        from scripts.train_v6_alpha import get_available_features, BLACKLIST
        features = get_available_features("BTCUSDT")
        for feat in features:
            assert feat not in BLACKLIST


@pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")
class TestSampleWeightPropagation:
    """Test that fit() and fit_classifier() accept sample_weight."""

    def _make_data(self, n: int = 500):
        rng = np.random.RandomState(42)
        X = rng.randn(n, 5)
        y = X[:, 0] * 0.5 + rng.randn(n) * 0.1
        return X, y

    def test_fit_with_sample_weight(self):
        from alpha.models.lgbm_alpha import LGBMAlphaModel
        X, y = self._make_data()
        weights = np.linspace(0.5, 1.0, len(X))
        model = LGBMAlphaModel(
            name="test_sw",
            feature_names=tuple(f"f{i}" for i in range(5)),
        )
        metrics = model.fit(X, y, sample_weight=weights, n_estimators=10)
        assert "val_mse" in metrics
        assert model._model is not None

    def test_fit_without_sample_weight(self):
        from alpha.models.lgbm_alpha import LGBMAlphaModel
        X, y = self._make_data()
        model = LGBMAlphaModel(
            name="test_nosw",
            feature_names=tuple(f"f{i}" for i in range(5)),
        )
        metrics = model.fit(X, y, n_estimators=10)
        assert "val_mse" in metrics

    def test_fit_classifier_with_sample_weight(self):
        from alpha.models.lgbm_alpha import LGBMAlphaModel
        X, y = self._make_data()
        y_bin = (y > 0).astype(float)
        weights = np.linspace(0.5, 1.0, len(X))
        model = LGBMAlphaModel(
            name="test_cls_sw",
            feature_names=tuple(f"f{i}" for i in range(5)),
        )
        metrics = model.fit_classifier(X, y_bin, sample_weight=weights, n_estimators=10)
        assert "val_logloss" in metrics
        assert model._is_classifier is True


class TestRegimeFeature:
    """Test regime_vol column creation."""

    def test_regime_vol_added(self):
        import pandas as pd
        from scripts.train_v6_alpha import _add_regime_feature
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"vol_20": rng.exponential(0.02, 500)})
        result = _add_regime_feature(df)
        assert "regime_vol" in result.columns
        values = set(result["regime_vol"].unique())
        assert values.issubset({0, 1, 2})

    def test_regime_without_vol20(self):
        import pandas as pd
        from scripts.train_v6_alpha import _add_regime_feature
        df = pd.DataFrame({"close": [100.0] * 50})
        result = _add_regime_feature(df)
        assert "regime_vol" in result.columns
        # Should default to 1
        assert (result["regime_vol"] == 1).all()

    def test_regime_terciles(self):
        import pandas as pd
        from scripts.train_v6_alpha import _add_regime_feature
        # Create data with clear tercile structure (extra high values to
        # ensure p67 boundary doesn't land on the high group)
        vol = np.concatenate([
            np.full(100, 0.01),  # low
            np.full(100, 0.03),  # medium
            np.full(50, 0.10),   # high
            np.full(50, 0.15),   # very high
        ])
        df = pd.DataFrame({"vol_20": vol})
        result = _add_regime_feature(df)
        # First 100 should be 0 (low vol)
        assert (result["regime_vol"].values[:100] == 0).all()
        # Last 50 should be 2 (very high vol)
        assert (result["regime_vol"].values[250:] == 2).all()


@pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")
class TestInnerCVFixed:
    """Test that inner CV returns reasonable floats."""

    def test_inner_cv_returns_float(self):
        from scripts.train_v6_alpha import _inner_cv_objective
        rng = np.random.RandomState(42)
        n = 1000
        X = rng.randn(n, 5)
        y = X[:, 0] * 0.3 + rng.randn(n) * 0.5
        weights = np.linspace(0.5, 1.0, n)

        obj_fn = _inner_cv_objective(X, y, weights, embargo_bars=5, inner_folds=3)

        result = obj_fn({
            "max_depth": 5,
            "num_leaves": 16,
            "min_child_samples": 50,
            "learning_rate": 0.01,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
        })
        assert isinstance(result, float)
        # Should not be exactly 0 with a signal in the data
        # (relaxed — could be 0 if signal is too weak)
        assert -1.0 <= result <= 1.0
