"""Tests for 15m walk-forward validation script."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestImports:
    """Verify the 15m WF module can be imported."""

    def test_import_module(self):
        from alpha.walkforward import walkforward_validate_15m
        assert hasattr(walkforward_validate_15m, "main")
        assert hasattr(walkforward_validate_15m, "run_fold")
        assert hasattr(walkforward_validate_15m, "generate_wf_folds")
        assert hasattr(walkforward_validate_15m, "stitch_results")

    def test_import_constants(self):
        from alpha.walkforward.walkforward_validate_15m import (
            BARS_PER_DAY, BARS_PER_MONTH, MIN_TRAIN_BARS,
            TEST_BARS, STEP_BARS, BLACKLIST_15M,
        )
        assert BARS_PER_DAY == 96
        assert BARS_PER_MONTH == 2880
        assert MIN_TRAIN_BARS == 34560  # 12 months
        assert TEST_BARS == 8640        # 3 months
        assert STEP_BARS == 8640
        assert "fgi_normalized" in BLACKLIST_15M

    def test_import_dataclasses(self):
        from alpha.walkforward.walkforward_validate_15m import Fold, FoldResult
        f = Fold(idx=0, train_start=0, train_end=100, test_start=100, test_end=200)
        assert f.idx == 0
        r = FoldResult(idx=0, period="test", ic=0.1, sharpe=1.5,
                       total_return=0.05, features=["a"], n_train=100, n_test=50)
        assert r.sharpe == 1.5


class TestFoldGeneration:
    """Test expanding-window fold splitting for 15m data."""

    def test_basic_fold_generation(self):
        from alpha.walkforward.walkforward_validate_15m import generate_wf_folds
        # 18 months of data = 18 * 2880 = 51840 bars
        # min_train = 34560 (12 months), test = 8640 (3 months), step = 8640
        # Fold 0: train [0:34560], test [34560:43200]
        # Fold 1: train [0:43200], test [43200:51840]
        folds = generate_wf_folds(51840)
        assert len(folds) == 2
        assert folds[0].train_start == 0
        assert folds[0].train_end == 34560
        assert folds[0].test_start == 34560
        assert folds[0].test_end == 43200
        assert folds[1].train_end == 43200
        assert folds[1].test_end == 51840

    def test_insufficient_data(self):
        from alpha.walkforward.walkforward_validate_15m import generate_wf_folds
        # Less than min_train + test bars = 34560 + 8640 = 43200
        folds = generate_wf_folds(40000)
        assert len(folds) == 0

    def test_expanding_window(self):
        from alpha.walkforward.walkforward_validate_15m import generate_wf_folds
        # 2.5 years = 30 months = 86400 bars
        folds = generate_wf_folds(86400)
        # All folds start from bar 0 (expanding window)
        for f in folds:
            assert f.train_start == 0
        # Train end grows by STEP_BARS each fold
        for i in range(1, len(folds)):
            assert folds[i].train_end == folds[i-1].train_end + 8640

    def test_fold_indices_sequential(self):
        from alpha.walkforward.walkforward_validate_15m import generate_wf_folds
        folds = generate_wf_folds(86400)
        for i, f in enumerate(folds):
            assert f.idx == i

    def test_custom_parameters(self):
        from alpha.walkforward.walkforward_validate_15m import generate_wf_folds
        folds = generate_wf_folds(
            20000,
            min_train_bars=10000,
            test_bars=4000,
            step_bars=4000,
        )
        assert len(folds) == 2
        assert folds[0].train_end == 10000
        assert folds[0].test_end == 14000


class TestHelpers:
    """Test helper functions."""

    def test_compute_target(self):
        from alpha.walkforward.walkforward_validate_15m import compute_target
        closes = np.array([100.0, 102.0, 104.0, 106.0, 108.0])
        y = compute_target(closes, horizon=2)
        assert len(y) == 5
        # y[0] = 104/100 - 1 = 0.04
        assert abs(y[0] - 0.04) < 1e-6
        # Last 2 should be NaN
        assert np.isnan(y[-1])
        assert np.isnan(y[-2])

    def test_compute_target_clipping(self):
        from alpha.walkforward.walkforward_validate_15m import compute_target
        closes = np.concatenate([
            np.ones(100) * 100,
            [200.0],  # extreme spike
            np.ones(100) * 100,
        ])
        y = compute_target(closes, horizon=1)
        # The extreme return should be clipped to 99th percentile
        valid = y[~np.isnan(y)]
        assert valid.max() < 1.0  # Should not be 100% return

    def test_fast_ic(self):
        from alpha.walkforward.walkforward_validate_15m import fast_ic
        x = np.arange(100, dtype=float)
        y = np.arange(100, dtype=float) + np.random.randn(100) * 0.1
        ic = fast_ic(x, y)
        assert ic > 0.9  # Nearly perfect correlation

    def test_fast_ic_insufficient_data(self):
        from alpha.walkforward.walkforward_validate_15m import fast_ic
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        ic = fast_ic(x, y)
        assert ic == 0.0  # Less than 50 samples

    def test_fast_ic_with_nans(self):
        from alpha.walkforward.walkforward_validate_15m import fast_ic
        x = np.arange(100, dtype=float)
        y = np.arange(100, dtype=float)
        x[50] = np.nan
        y[60] = np.nan
        ic = fast_ic(x, y)
        assert ic > 0.9  # Should skip NaN pairs


class TestStitchResults:
    """Test result aggregation."""

    def test_basic_stitching(self):
        from alpha.walkforward.walkforward_validate_15m import FoldResult, stitch_results
        results = [
            FoldResult(idx=0, period="2024-01→2024-03", ic=0.05, sharpe=2.0,
                       total_return=0.1, features=["a", "b"], n_train=1000, n_test=500, n_trades=20),
            FoldResult(idx=1, period="2024-04→2024-06", ic=0.03, sharpe=-1.0,
                       total_return=-0.05, features=["a", "c"], n_train=2000, n_test=500, n_trades=15),
            FoldResult(idx=2, period="2024-07→2024-09", ic=0.04, sharpe=1.5,
                       total_return=0.08, features=["a", "b"], n_train=3000, n_test=500, n_trades=18),
        ]
        summary = stitch_results(results)
        assert summary["n_folds"] == 3
        assert summary["positive_sharpe"] == 2
        assert summary["total_trades"] == 53
        assert abs(summary["avg_ic"] - 0.04) < 1e-6

    def test_pass_threshold(self):
        from alpha.walkforward.walkforward_validate_15m import FoldResult, stitch_results
        # 3/5 = 60% exactly, threshold is 60%
        results = [
            FoldResult(idx=i, period="", ic=0.0, sharpe=(1.0 if i < 3 else -1.0),
                       total_return=0.0, features=[], n_train=0, n_test=0, n_trades=0)
            for i in range(5)
        ]
        summary = stitch_results(results)
        assert summary["positive_sharpe"] == 3
        assert summary["passed"] is True  # 3 >= 3 (60% of 5)

    def test_fail_threshold(self):
        from alpha.walkforward.walkforward_validate_15m import FoldResult, stitch_results
        # 2/5 = 40% < 60%
        results = [
            FoldResult(idx=i, period="", ic=0.0, sharpe=(1.0 if i < 2 else -1.0),
                       total_return=0.0, features=[], n_train=0, n_test=0, n_trades=0)
            for i in range(5)
        ]
        summary = stitch_results(results)
        assert summary["passed"] is False

    def test_feature_stability(self):
        from alpha.walkforward.walkforward_validate_15m import FoldResult, stitch_results
        # Feature "a" appears in all 5 folds (100%), "b" in 4/5 (80%), "c" in 2/5 (40%)
        results = [
            FoldResult(idx=0, period="", ic=0.0, sharpe=1.0, total_return=0.0,
                       features=["a", "b", "c"], n_train=0, n_test=0, n_trades=0),
            FoldResult(idx=1, period="", ic=0.0, sharpe=1.0, total_return=0.0,
                       features=["a", "b"], n_train=0, n_test=0, n_trades=0),
            FoldResult(idx=2, period="", ic=0.0, sharpe=1.0, total_return=0.0,
                       features=["a", "b"], n_train=0, n_test=0, n_trades=0),
            FoldResult(idx=3, period="", ic=0.0, sharpe=1.0, total_return=0.0,
                       features=["a", "b", "c"], n_train=0, n_test=0, n_trades=0),
            FoldResult(idx=4, period="", ic=0.0, sharpe=1.0, total_return=0.0,
                       features=["a"], n_train=0, n_test=0, n_trades=0),
        ]
        summary = stitch_results(results)
        assert "a" in summary["stable_features"]  # 5/5 >= 80%
        assert "b" in summary["stable_features"]  # 4/5 = 80% >= 80%
        assert "c" not in summary["stable_features"]  # 2/5 = 40% < 80%


class TestT1Check:
    """Test T-1 cross-market feature checking logic."""

    def test_cross_market_features_defined(self):
        from alpha.walkforward.walkforward_validate_15m import CROSS_MARKET_FEATURES
        assert "spy_ret_1d" in CROSS_MARKET_FEATURES
        assert "dvol_z" in CROSS_MARKET_FEATURES
        assert "etf_vol_zscore_14" in CROSS_MARKET_FEATURES
        # Non-cross-market features should NOT be in the set
        assert "rsi_14" not in CROSS_MARKET_FEATURES
        assert "atr_norm_14" not in CROSS_MARKET_FEATURES
