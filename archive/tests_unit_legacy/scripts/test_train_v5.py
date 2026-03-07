"""Tests for V5 training pipeline."""
from __future__ import annotations

import numpy as np
import pytest


class TestSampleWeights:

    def test_monotonically_increasing(self):
        from scripts.train_v5_signal import _compute_sample_weights
        w = _compute_sample_weights(100, decay=0.5)
        assert len(w) == 100
        assert all(w[i] <= w[i + 1] for i in range(len(w) - 1))

    def test_bounds(self):
        from scripts.train_v5_signal import _compute_sample_weights
        w = _compute_sample_weights(100, decay=0.5)
        assert w[0] == pytest.approx(0.5)
        assert w[-1] == pytest.approx(1.0)

    def test_decay_parameter(self):
        from scripts.train_v5_signal import _compute_sample_weights
        w1 = _compute_sample_weights(100, decay=0.1)
        w2 = _compute_sample_weights(100, decay=0.9)
        assert w1[0] < w2[0]
        assert w1[-1] == w2[-1]  # both end at 1.0

    def test_single_sample(self):
        from scripts.train_v5_signal import _compute_sample_weights
        w = _compute_sample_weights(1, decay=0.5)
        assert len(w) == 1


class TestExpandingWindowFolds:

    def test_basic_folds(self):
        from scripts.train_v5_signal import expanding_window_folds
        folds = expanding_window_folds(2000, n_folds=5, min_train=500)
        assert len(folds) == 5
        for tr_start, tr_end, te_start, te_end in folds:
            assert tr_start == 0
            assert te_start == tr_end
            assert te_end > te_start

    def test_non_overlapping_test(self):
        from scripts.train_v5_signal import expanding_window_folds
        folds = expanding_window_folds(2000, n_folds=5, min_train=500)
        for i in range(len(folds) - 1):
            assert folds[i][3] <= folds[i + 1][2]

    def test_insufficient_data(self):
        from scripts.train_v5_signal import expanding_window_folds
        folds = expanding_window_folds(100, n_folds=5, min_train=500)
        assert folds == []

    def test_expanding_train(self):
        from scripts.train_v5_signal import expanding_window_folds
        folds = expanding_window_folds(2000, n_folds=5, min_train=500)
        train_sizes = [tr_end - tr_start for tr_start, tr_end, _, _ in folds]
        assert all(train_sizes[i] <= train_sizes[i + 1] for i in range(len(train_sizes) - 1))


class TestV5Constants:

    def test_embargo_larger_than_v3(self):
        from scripts.train_v5_signal import EMBARGO_BARS
        assert EMBARGO_BARS == 7

    def test_interaction_features_include_v5(self):
        from scripts.train_v5_signal import INTERACTION_FEATURES
        names = [n for n, _, _ in INTERACTION_FEATURES]
        assert "cvd_x_oi_chg" in names
        assert "vol_of_vol_x_range" in names

    def test_search_space_dimensions(self):
        from scripts.train_v5_signal import V5_SEARCH_SPACE
        assert V5_SEARCH_SPACE.dimension_count == 8  # 3 int + 5 float

    def test_blacklist_preserved(self):
        from scripts.train_v5_signal import BLACKLIST
        assert "avg_trade_size" in BLACKLIST


class TestICIRIntegration:

    def test_icir_select_called_with_correct_interface(self):
        """Verify icir_select works with the expected V5 interface."""
        from features.dynamic_selector import icir_select
        rng = np.random.RandomState(42)
        n, p = 1200, 10
        X = rng.randn(n, p)
        y = 0.3 * X[:, 0] + 0.05 * rng.randn(n)
        names = [f"feat_{i}" for i in range(p)]
        selected = icir_select(X, y, names, top_k=5, ic_window=200, n_windows=5)
        assert isinstance(selected, list)
        assert len(selected) <= 5
        assert "feat_0" in selected  # strong signal feature


class TestVolNormalizedTarget:

    def test_basic_target(self):
        from scripts.train_v5_signal import vol_normalized_target
        closes = np.linspace(100, 110, 200)
        target = vol_normalized_target(closes, horizon=5, vol_window=20)
        assert len(target) == 200
        # Last 5 should be NaN (no forward return)
        assert np.isnan(target[-1])
        # Some valid values after warmup
        valid = ~np.isnan(target)
        assert valid.sum() > 100
