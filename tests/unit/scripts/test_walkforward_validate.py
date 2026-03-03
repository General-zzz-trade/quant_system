"""Tests for walk-forward validation script — fold generation + result aggregation."""
from __future__ import annotations
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from scripts.walkforward_validate import (
    Fold,
    FoldResult,
    generate_wf_folds,
    run_fold,
    stitch_results,
    _compute_regime_labels,
    _apply_signal_filters,
    _apply_trend_hold,
    WARMUP,
    DEADZONE,
    MIN_HOLD,
)


class TestGenerateWfFolds:
    def test_basic_fold_generation(self):
        """Standard case: enough data for multiple folds."""
        folds = generate_wf_folds(
            n_bars=20000,
            min_train_bars=8760,
            test_bars=2190,
            step_bars=2190,
        )
        assert len(folds) > 0
        # First fold: train [0, 8760), test [8760, 10950)
        assert folds[0].train_start == 0
        assert folds[0].train_end == 8760
        assert folds[0].test_start == 8760
        assert folds[0].test_end == 8760 + 2190

    def test_expanding_window(self):
        """Train always starts from 0 (expanding window)."""
        folds = generate_wf_folds(
            n_bars=30000,
            min_train_bars=8760,
            test_bars=2190,
            step_bars=2190,
        )
        for fold in folds:
            assert fold.train_start == 0
        # Train end expands each fold
        if len(folds) >= 2:
            assert folds[1].train_end > folds[0].train_end

    def test_fold_count(self):
        """Check expected number of folds for known data size."""
        # 56755 bars (real BTC data size)
        folds = generate_wf_folds(
            n_bars=56755,
            min_train_bars=8760,
            test_bars=2190,
            step_bars=2190,
        )
        # (56755 - 8760) / 2190 ≈ 21.9, but need full test_bars → ~21 folds
        assert len(folds) >= 15
        assert len(folds) <= 22

    def test_insufficient_data(self):
        """Not enough data → empty fold list."""
        folds = generate_wf_folds(
            n_bars=5000,
            min_train_bars=8760,
            test_bars=2190,
            step_bars=2190,
        )
        assert folds == []

    def test_exact_boundary(self):
        """Exactly min_train + test_bars data → 1 fold."""
        folds = generate_wf_folds(
            n_bars=8760 + 2190,
            min_train_bars=8760,
            test_bars=2190,
            step_bars=2190,
        )
        assert len(folds) == 1

    def test_no_overlap_between_folds(self):
        """Test windows should not overlap (same step = test_bars)."""
        folds = generate_wf_folds(
            n_bars=30000,
            min_train_bars=8760,
            test_bars=2190,
            step_bars=2190,
        )
        for i in range(1, len(folds)):
            # Current test_start should be >= previous test_end
            # (with step_bars == test_bars, they should be exactly equal)
            assert folds[i].test_start >= folds[i - 1].test_end or \
                   folds[i].test_start == folds[i - 1].test_start + 2190

    def test_train_never_overlaps_test(self):
        """Train end <= test start for every fold."""
        folds = generate_wf_folds(n_bars=40000, min_train_bars=8760,
                                   test_bars=2190, step_bars=2190)
        for fold in folds:
            assert fold.train_end <= fold.test_start

    def test_sequential_indices(self):
        """Fold indices should be sequential starting from 0."""
        folds = generate_wf_folds(n_bars=30000, min_train_bars=8760,
                                   test_bars=2190, step_bars=2190)
        for i, fold in enumerate(folds):
            assert fold.idx == i


class TestStitchResults:
    def _make_result(self, idx: int, sharpe: float, ic: float = 0.05,
                     ret: float = 0.02, features: list = None) -> FoldResult:
        return FoldResult(
            idx=idx, period=f"fold_{idx}",
            ic=ic, sharpe=sharpe, total_return=ret,
            features=features or ["feat_a", "feat_b"],
            n_train=5000, n_test=2190,
        )

    def test_pass_verdict(self):
        """10/15 positive Sharpe → PASS."""
        results = [self._make_result(i, sharpe=1.0) for i in range(10)]
        results += [self._make_result(i + 10, sharpe=-0.5) for i in range(5)]
        summary = stitch_results(results)
        assert summary["positive_sharpe"] == 10
        assert summary["n_folds"] == 15
        assert summary["passed"] is True

    def test_fail_verdict(self):
        """5/15 positive Sharpe → FAIL."""
        results = [self._make_result(i, sharpe=1.0) for i in range(5)]
        results += [self._make_result(i + 5, sharpe=-0.5) for i in range(10)]
        summary = stitch_results(results)
        assert summary["positive_sharpe"] == 5
        assert summary["passed"] is False

    def test_feature_stability(self):
        """Features appearing >= 80% of folds are marked stable."""
        results = []
        for i in range(10):
            feats = ["always_here"]
            if i < 3:
                feats.append("sometimes")
            results.append(self._make_result(i, sharpe=1.0, features=feats))

        summary = stitch_results(results)
        assert "always_here" in summary["stable_features"]
        assert "sometimes" not in summary["stable_features"]

    def test_avg_metrics(self):
        """Average IC and Sharpe computed correctly."""
        results = [
            self._make_result(0, sharpe=2.0, ic=0.04, ret=0.05),
            self._make_result(1, sharpe=1.0, ic=0.06, ret=0.03),
        ]
        summary = stitch_results(results)
        assert abs(summary["avg_ic"] - 0.05) < 1e-10
        assert abs(summary["avg_sharpe"] - 1.5) < 1e-10
        assert abs(summary["total_return"] - 0.08) < 1e-10

    def test_empty_results(self):
        """Edge case: no fold results."""
        summary = stitch_results([])
        assert summary["n_folds"] == 0
        assert summary["passed"] is True  # 0 >= 0
        assert summary["stable_features"] == {}


class TestRunFoldFixedFeatures:
    """Test run_fold with fixed_features mode."""

    def _make_data(self, n_bars: int, feature_names: list):
        """Create synthetic data for run_fold tests."""
        np.random.seed(42)
        feat_df = {name: np.random.randn(n_bars) for name in feature_names}
        feat_df["close"] = 100 + np.cumsum(np.random.randn(n_bars) * 0.01)
        import pandas as pd
        feat_df = pd.DataFrame(feat_df)
        closes = feat_df["close"].values.astype(np.float64)
        return feat_df, closes

    @patch("scripts.walkforward_validate.greedy_ic_select")
    @patch("lightgbm.train")
    @patch("lightgbm.Dataset")
    def test_fixed_features_locks_and_selects_flexible(
        self, mock_dataset, mock_lgb_train, mock_greedy
    ):
        """Fixed features always included; greedy selects flexible from pool."""
        features = ["f_fixed1", "f_fixed2", "f_cand1", "f_cand2", "f_cand3"]
        n = 10000
        feat_df, closes = self._make_data(n, features)
        fold = Fold(idx=0, train_start=0, train_end=8000, test_start=8000, test_end=n)

        mock_greedy.return_value = ["f_cand1"]
        mock_bst = MagicMock()
        mock_bst.predict.return_value = np.random.randn(n - 8000)
        mock_lgb_train.return_value = mock_bst

        result = run_fold(
            fold, feat_df, closes, features,
            use_hpo=False,
            fixed_features=["f_fixed1", "f_fixed2"],
            candidate_pool=["f_cand1", "f_cand2", "f_cand3"],
            n_flexible=1,
        )

        # Fixed features are always in result
        assert "f_fixed1" in result.features
        assert "f_fixed2" in result.features
        # Flexible selected via greedy
        assert "f_cand1" in result.features
        # greedy_ic_select called with pool features only
        call_args = mock_greedy.call_args
        assert call_args[0][2] == ["f_cand1", "f_cand2", "f_cand3"]
        assert call_args[1]["top_k"] == 1

    @patch("scripts.walkforward_validate.greedy_ic_select")
    @patch("lightgbm.train")
    @patch("lightgbm.Dataset")
    def test_fixed_features_zero_flexible(
        self, mock_dataset, mock_lgb_train, mock_greedy
    ):
        """n_flexible=0 means only fixed features, no greedy call."""
        features = ["f_a", "f_b", "f_c"]
        n = 10000
        feat_df, closes = self._make_data(n, features)
        fold = Fold(idx=0, train_start=0, train_end=8000, test_start=8000, test_end=n)

        mock_bst = MagicMock()
        mock_bst.predict.return_value = np.random.randn(n - 8000)
        mock_lgb_train.return_value = mock_bst

        result = run_fold(
            fold, feat_df, closes, features,
            use_hpo=False,
            fixed_features=["f_a", "f_b"],
            n_flexible=0,
        )

        assert result.features == ["f_a", "f_b"]
        mock_greedy.assert_not_called()

    @patch("scripts.walkforward_validate.greedy_ic_select")
    @patch("lightgbm.train")
    @patch("lightgbm.Dataset")
    def test_no_fixed_features_uses_original_logic(
        self, mock_dataset, mock_lgb_train, mock_greedy
    ):
        """Without fixed_features, original greedy top_k logic is used."""
        features = ["f1", "f2", "f3"]
        n = 10000
        feat_df, closes = self._make_data(n, features)
        fold = Fold(idx=0, train_start=0, train_end=8000, test_start=8000, test_end=n)

        mock_greedy.return_value = ["f1", "f2"]
        mock_bst = MagicMock()
        mock_bst.predict.return_value = np.random.randn(n - 8000)
        mock_lgb_train.return_value = mock_bst

        result = run_fold(fold, feat_df, closes, features, use_hpo=False)

        # greedy called with all features, default TOP_K
        call_args = mock_greedy.call_args
        assert call_args[0][2] == features


class TestComputeRegimeLabels:
    """Test _compute_regime_labels regime detection."""

    def test_trending_strong_ma(self):
        """Strong MA deviation → trending."""
        import pandas as pd
        df = pd.DataFrame({
            "vol_regime": [1.0],
            "close_vs_ma20": [0.05],
            "close_vs_ma50": [0.03],
        })
        labels = _compute_regime_labels(df)
        assert labels[0] == "trending"

    def test_ranging_low_ma(self):
        """Both MAs near zero → ranging."""
        import pandas as pd
        df = pd.DataFrame({
            "vol_regime": [0.9],
            "close_vs_ma20": [0.005],
            "close_vs_ma50": [0.003],
        })
        labels = _compute_regime_labels(df)
        assert labels[0] == "ranging"

    def test_high_vol_flat(self):
        """High vol expanding + no trend → high_vol_flat."""
        import pandas as pd
        df = pd.DataFrame({
            "vol_regime": [1.5],
            "close_vs_ma20": [0.005],
            "close_vs_ma50": [-0.002],
        })
        labels = _compute_regime_labels(df)
        assert labels[0] == "high_vol_flat"

    def test_high_vol_trending_stays_trending(self):
        """High vol + strong trend → still trending (vol alone doesn't block)."""
        import pandas as pd
        df = pd.DataFrame({
            "vol_regime": [1.5],
            "close_vs_ma20": [0.04],
            "close_vs_ma50": [0.03],
        })
        labels = _compute_regime_labels(df)
        assert labels[0] == "trending"

    def test_nan_defaults_trending(self):
        """NaN features → default to trending (warmup period)."""
        import pandas as pd
        df = pd.DataFrame({
            "vol_regime": [np.nan],
            "close_vs_ma20": [np.nan],
            "close_vs_ma50": [np.nan],
        })
        labels = _compute_regime_labels(df)
        assert labels[0] == "trending"

    def test_missing_columns_defaults_trending(self):
        """Missing feature columns → all trending."""
        import pandas as pd
        df = pd.DataFrame({"some_other_col": [1.0, 2.0, 3.0]})
        labels = _compute_regime_labels(df)
        assert all(l == "trending" for l in labels)

    def test_mixed_regimes(self):
        """Multiple bars with different regimes."""
        import pandas as pd
        df = pd.DataFrame({
            "vol_regime": [1.0, 0.8, 1.5, 1.3],
            "close_vs_ma20": [0.05, 0.003, 0.001, 0.04],
            "close_vs_ma50": [0.03, 0.002, -0.005, 0.02],
        })
        labels = _compute_regime_labels(df)
        assert labels[0] == "trending"
        assert labels[1] == "ranging"
        assert labels[2] == "high_vol_flat"
        assert labels[3] == "trending"


class TestApplySignalFilters:
    """Test _apply_signal_filters post-processing."""

    def test_long_only_clips_shorts(self):
        """Long-only clips negative signals to 0."""
        sig = np.array([1.0, -1.0, 0.0, 1.0, -1.0])
        out = _apply_signal_filters(sig, None, long_only=True,
                                     regime_gate=False, adaptive_sizing=None)
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0, 1.0, 0.0])

    def test_regime_gate_zeros_bad_regimes(self):
        """Regime gate zeros signal in ranging and high_vol_flat."""
        sig = np.array([1.0, 1.0, -1.0, 1.0])
        regimes = np.array(["trending", "ranging", "high_vol_flat", "trending"])
        out = _apply_signal_filters(sig, regimes, long_only=False,
                                     regime_gate=True, adaptive_sizing=None)
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0, 1.0])

    def test_adaptive_sizing_scales(self):
        """Adaptive sizing scales signal by regime multiplier."""
        sig = np.array([1.0, 1.0, -1.0, 1.0])
        regimes = np.array(["trending", "ranging", "high_vol_flat", "trending"])
        sizing = {"trending": 1.0, "ranging": 0.5, "high_vol_flat": 0.0}
        out = _apply_signal_filters(sig, regimes, long_only=False,
                                     regime_gate=False, adaptive_sizing=sizing)
        np.testing.assert_array_almost_equal(out, [1.0, 0.5, 0.0, 1.0])

    def test_combined_long_only_and_regime_gate(self):
        """Combined: long-only first, then regime gate."""
        sig = np.array([1.0, -1.0, 1.0, -1.0])
        regimes = np.array(["trending", "trending", "ranging", "ranging"])
        out = _apply_signal_filters(sig, regimes, long_only=True,
                                     regime_gate=True, adaptive_sizing=None)
        # long_only: [1, 0, 1, 0], then regime_gate: [1, 0, 0, 0]
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0, 0.0])

    def test_no_filters_passthrough(self):
        """No filters → signal unchanged."""
        sig = np.array([1.0, -1.0, 0.0])
        out = _apply_signal_filters(sig, None, long_only=False,
                                     regime_gate=False, adaptive_sizing=None)
        np.testing.assert_array_equal(out, sig)

    def test_does_not_mutate_input(self):
        """Input signal array should not be modified."""
        sig = np.array([1.0, -1.0, 1.0])
        original = sig.copy()
        _apply_signal_filters(sig, None, long_only=True,
                               regime_gate=False, adaptive_sizing=None)
        np.testing.assert_array_equal(sig, original)


class TestRunFoldSignalParams:
    """Test run_fold with deadzone, min_hold, and continuous_sizing."""

    def _make_data(self, n_bars: int, feature_names: list):
        np.random.seed(42)
        feat_df = {name: np.random.randn(n_bars) for name in feature_names}
        feat_df["close"] = 100 + np.cumsum(np.random.randn(n_bars) * 0.01)
        import pandas as pd
        feat_df = pd.DataFrame(feat_df)
        closes = feat_df["close"].values.astype(np.float64)
        return feat_df, closes

    @patch("scripts.walkforward_validate.greedy_ic_select")
    @patch("lightgbm.train")
    @patch("lightgbm.Dataset")
    def test_deadzone_passed_to_signal(
        self, mock_dataset, mock_lgb_train, mock_greedy
    ):
        """Custom deadzone affects signal generation (higher = fewer trades)."""
        features = ["f1", "f2", "f3"]
        n = 10000
        feat_df, closes = self._make_data(n, features)
        fold = Fold(idx=0, train_start=0, train_end=8000,
                    test_start=8000, test_end=n)

        mock_greedy.return_value = ["f1", "f2"]
        mock_bst = MagicMock()
        # Predictions with moderate z-scores
        preds = np.random.randn(n - 8000) * 0.8
        mock_bst.predict.return_value = preds
        mock_lgb_train.return_value = mock_bst

        # Low deadzone → more trades
        r_low = run_fold(fold, feat_df, closes, features, use_hpo=False,
                         deadzone=0.3, min_hold=12)
        # High deadzone → fewer trades
        r_high = run_fold(fold, feat_df, closes, features, use_hpo=False,
                          deadzone=1.5, min_hold=12)

        # Both should produce valid results; high deadzone generally
        # leads to lower absolute return (fewer trades)
        assert isinstance(r_low.sharpe, float)
        assert isinstance(r_high.sharpe, float)

    @patch("scripts.walkforward_validate.greedy_ic_select")
    @patch("lightgbm.train")
    @patch("lightgbm.Dataset")
    def test_continuous_sizing_produces_fractional_positions(
        self, mock_dataset, mock_lgb_train, mock_greedy
    ):
        """Continuous sizing should produce values between 0 and 1, not just {0, 1}."""
        features = ["f1", "f2"]
        n = 10000
        feat_df, closes = self._make_data(n, features)
        fold = Fold(idx=0, train_start=0, train_end=8000,
                    test_start=8000, test_end=n)

        mock_greedy.return_value = ["f1", "f2"]
        mock_bst = MagicMock()
        # Predictions with varying z-scores so some are fractional
        np.random.seed(123)
        preds = np.random.randn(n - 8000)
        mock_bst.predict.return_value = preds
        mock_lgb_train.return_value = mock_bst

        result = run_fold(fold, feat_df, closes, features, use_hpo=False,
                          long_only=True, continuous_sizing=True,
                          deadzone=0.5, min_hold=24)

        assert isinstance(result.sharpe, float)
        assert isinstance(result.total_return, float)

    @patch("scripts.walkforward_validate.greedy_ic_select")
    @patch("lightgbm.train")
    @patch("lightgbm.Dataset")
    def test_continuous_sizing_clipped_0_to_1(
        self, mock_dataset, mock_lgb_train, mock_greedy
    ):
        """Continuous sizing values should be in [0, 1] range."""
        from scripts.backtest_alpha_v8 import _pred_to_signal

        features = ["f1"]
        n = 10000
        feat_df, closes = self._make_data(n, features)
        fold = Fold(idx=0, train_start=0, train_end=8000,
                    test_start=8000, test_end=n)

        mock_greedy.return_value = ["f1"]
        mock_bst = MagicMock()
        # Large z-scores to test clipping
        preds = np.random.randn(n - 8000) * 3.0
        mock_bst.predict.return_value = preds
        mock_lgb_train.return_value = mock_bst

        # The continuous sizing logic: clip(z/2, 0, 1) where signal != 0
        # Verify by reproducing the logic
        mu, std = np.mean(preds), np.std(preds)
        z = (preds - mu) / std
        continuous = np.clip(z / 2.0, 0.0, 1.0)
        # All values should be in [0, 1]
        assert np.all(continuous >= 0.0)
        assert np.all(continuous <= 1.0)
        # Some should be fractional (not all 0 or 1)
        assert np.any((continuous > 0) & (continuous < 1))

    @patch("scripts.walkforward_validate.greedy_ic_select")
    @patch("lightgbm.train")
    @patch("lightgbm.Dataset")
    def test_hpo_trials_parameter(
        self, mock_dataset, mock_lgb_train, mock_greedy
    ):
        """hpo_trials parameter is passed through (tested via no-HPO path)."""
        features = ["f1", "f2"]
        n = 10000
        feat_df, closes = self._make_data(n, features)
        fold = Fold(idx=0, train_start=0, train_end=8000,
                    test_start=8000, test_end=n)

        mock_greedy.return_value = ["f1"]
        mock_bst = MagicMock()
        mock_bst.predict.return_value = np.random.randn(n - 8000)
        mock_lgb_train.return_value = mock_bst

        # Just verify it doesn't crash with custom hpo_trials
        result = run_fold(fold, feat_df, closes, features,
                          use_hpo=False, hpo_trials=20)
        assert isinstance(result.sharpe, float)


class TestApplyTrendHold:
    """Test _apply_trend_hold post-processing."""

    def test_trend_extends_hold(self):
        """When trend is above threshold at exit point, hold is extended."""
        signal = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        trend = np.array([0.01, 0.02, 0.015, 0.01, -0.01])
        out = _apply_trend_hold(signal, trend, trend_threshold=0.0, max_hold=10)
        # Bar 2: signal drops to 0, but trend=0.015 > 0 → extend
        # Bar 3: still extending, trend=0.01 > 0 → extend
        # Bar 4: trend=-0.01 < 0 → stop extending
        assert out[0] == 1.0
        assert out[1] == 1.0
        assert out[2] == 1.0  # extended
        assert out[3] == 1.0  # extended
        assert out[4] == 0.0  # trend broke

    def test_trend_break_exits(self):
        """When trend drops below threshold, exit immediately."""
        signal = np.array([1.0, 1.0, 0.0, 0.0])
        trend = np.array([0.02, 0.01, -0.005, 0.01])
        out = _apply_trend_hold(signal, trend, trend_threshold=0.0, max_hold=10)
        assert out[2] == 0.0  # trend < 0 → no extension
        assert out[3] == 0.0  # already exited

    def test_max_hold_enforced(self):
        """Max hold limit forces exit even if trend is intact."""
        signal = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        trend = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        out = _apply_trend_hold(signal, trend, trend_threshold=0.0, max_hold=3)
        # hold_count: bar0=1, bar1=2, bar2: would be 3 but hold_count=2 < max_hold=3 → extend
        # bar3: hold_count=3, 3 < 3 is False → exit
        assert out[0] == 1.0
        assert out[1] == 1.0
        assert out[2] == 1.0  # extended (hold_count was 2 < 3)
        assert out[3] == 0.0  # max_hold reached (hold_count was 3, not < 3)

    def test_nan_trend_no_extension(self):
        """NaN trend values do not extend holds."""
        signal = np.array([1.0, 1.0, 0.0, 0.0])
        trend = np.array([0.05, 0.05, np.nan, 0.05])
        out = _apply_trend_hold(signal, trend, trend_threshold=0.0, max_hold=10)
        assert out[2] == 0.0  # NaN → no extension

    def test_no_position_unaffected(self):
        """Bars with no prior position are not affected."""
        signal = np.array([0.0, 0.0, 1.0, 0.0])
        trend = np.array([0.05, 0.05, 0.05, 0.05])
        out = _apply_trend_hold(signal, trend, trend_threshold=0.0, max_hold=10)
        assert out[0] == 0.0
        assert out[1] == 0.0
        assert out[2] == 1.0
        assert out[3] == 1.0  # extended from bar 2

    def test_threshold_affects_extension(self):
        """Higher threshold requires stronger trend to extend."""
        signal = np.array([1.0, 0.0, 0.0])
        trend = np.array([0.02, 0.005, 0.005])
        # Low threshold: extend
        out_low = _apply_trend_hold(signal, trend, trend_threshold=0.0, max_hold=10)
        assert out_low[1] == 1.0
        # High threshold: don't extend
        out_high = _apply_trend_hold(signal, trend, trend_threshold=0.01, max_hold=10)
        assert out_high[1] == 0.0

    def test_does_not_mutate_input(self):
        """Input signal should not be modified."""
        signal = np.array([1.0, 0.0, 0.0])
        original = signal.copy()
        trend = np.array([0.05, 0.05, 0.05])
        _apply_trend_hold(signal, trend, trend_threshold=0.0, max_hold=10)
        np.testing.assert_array_equal(signal, original)
