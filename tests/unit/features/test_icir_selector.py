"""Tests for ICIR feature selection in dynamic_selector.py."""
from __future__ import annotations

import numpy as np

from features.dynamic_selector import icir_select, compute_feature_icir_report


def _make_data(n_samples: int, n_features: int, seed: int = 42):
    """Create synthetic feature matrix and target."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    return X, y


class TestIcirSelect:

    def test_basic_selection(self):
        """Stable features should be selected over noise."""
        rng = np.random.RandomState(42)
        n = 1200
        X = rng.randn(n, 5)
        # Feature 0: stable positive IC across all windows
        y = 0.3 * X[:, 0] + 0.05 * rng.randn(n)
        names = [f"f{i}" for i in range(5)]
        selected = icir_select(X, y, names, top_k=3, ic_window=200, n_windows=5)
        assert "f0" in selected

    def test_returns_list(self):
        X, y = _make_data(1200, 5)
        names = [f"f{i}" for i in range(5)]
        result = icir_select(X, y, names, top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_top_k_respected(self):
        rng = np.random.RandomState(42)
        n = 1200
        X = rng.randn(n, 10)
        y = 0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.05 * rng.randn(n)
        names = [f"f{i}" for i in range(10)]
        selected = icir_select(X, y, names, top_k=5)
        assert len(selected) <= 5

    def test_unstable_feature_filtered(self):
        """Feature with alternating IC signs should be filtered."""
        rng = np.random.RandomState(42)
        n = 1200
        X = rng.randn(n, 3)
        # f0: stable signal
        y = 0.3 * X[:, 0] + 0.05 * rng.randn(n)

        # f1: flip sign every window (200 bars)
        for w in range(6):
            ws = w * 200
            we = min(ws + 200, n)
            sign = 1.0 if w % 2 == 0 else -1.0
            X[ws:we, 1] = sign * np.abs(X[ws:we, 1])

        names = ["stable", "unstable", "noise"]
        selected = icir_select(X, y, names, top_k=2, ic_window=200, n_windows=5,
                               min_icir=0.5)
        assert "stable" in selected

    def test_too_few_samples_fallback(self):
        """With insufficient data, return first top_k features."""
        X, y = _make_data(30, 5)
        names = [f"f{i}" for i in range(5)]
        selected = icir_select(X, y, names, top_k=3, ic_window=200, n_windows=5)
        assert selected == ["f0", "f1", "f2"]

    def test_min_icir_filter(self):
        """All-noise features below min_icir should be filtered out (but fallback fills)."""
        X, y = _make_data(1200, 5)
        names = [f"f{i}" for i in range(5)]
        # With pure noise, ICIR should be low — but fallback fills to top_k
        selected = icir_select(X, y, names, top_k=3, min_icir=0.3)
        assert len(selected) == 3  # fallback fills

    def test_consecutive_negative_filter(self):
        """Feature with 3+ consecutive negative IC windows should be discarded."""
        rng = np.random.RandomState(42)
        n = 1200
        X = rng.randn(n, 3)
        y = 0.3 * X[:, 0] + 0.05 * rng.randn(n)
        # f2: negatively correlated in last 3 windows
        for w in range(2, 5):
            ws = (n - 1000) + w * 200
            we = min(ws + 200, n)
            X[ws:we, 2] = -np.abs(X[ws:we, 2]) * y[ws:we]

        names = ["stable", "noise", "neg_streak"]
        selected = icir_select(X, y, names, top_k=2, ic_window=200, n_windows=5,
                               max_consecutive_negative=3)
        assert "stable" in selected

    def test_no_duplicates(self):
        rng = np.random.RandomState(42)
        n = 1200
        X = rng.randn(n, 10)
        y = 0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.05 * rng.randn(n)
        names = [f"f{i}" for i in range(10)]
        selected = icir_select(X, y, names, top_k=5)
        assert len(selected) == len(set(selected))


class TestComputeFeatureIcirReport:

    def test_report_keys(self):
        rng = np.random.RandomState(42)
        n = 1200
        X = rng.randn(n, 3)
        y = 0.3 * X[:, 0] + 0.05 * rng.randn(n)
        names = ["good", "noise1", "noise2"]
        report = compute_feature_icir_report(X, y, names, ic_window=200, n_windows=5)
        assert set(report.keys()) == set(names)
        for name, entry in report.items():
            assert "mean_ic" in entry
            assert "std_ic" in entry
            assert "icir" in entry
            assert "max_consec_neg" in entry
            assert "pct_positive" in entry

    def test_good_feature_high_icir(self):
        rng = np.random.RandomState(42)
        n = 1200
        X = rng.randn(n, 2)
        y = 0.5 * X[:, 0] + 0.05 * rng.randn(n)
        names = ["signal", "noise"]
        report = compute_feature_icir_report(X, y, names, ic_window=200, n_windows=5)
        assert report["signal"]["icir"] > report["noise"]["icir"]
        assert report["signal"]["pct_positive"] > 0.5

    def test_empty_on_insufficient_data(self):
        X, y = _make_data(100, 3)
        names = ["a", "b", "c"]
        report = compute_feature_icir_report(X, y, names, ic_window=200, n_windows=5)
        assert report == {}
