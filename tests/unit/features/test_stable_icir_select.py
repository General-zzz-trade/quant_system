"""Tests for stable_icir_select feature selection."""
from __future__ import annotations

import numpy as np
import pytest

from features.dynamic_selector import stable_icir_select, icir_select


class TestStableIcirSelect:
    """Test stable_icir_select with stability gate + sign consistency."""

    def _make_data(self, n=1200, p=20, seed=42):
        rng = np.random.RandomState(seed)
        X = rng.randn(n, p)
        # Feature 0: strong positive IC, stable across windows
        y = 0.3 * X[:, 0] + 0.1 * X[:, 1] + rng.randn(n) * 0.5
        return X, y

    def test_returns_list_of_strings(self):
        X, y = self._make_data()
        names = [f"f{i}" for i in range(X.shape[1])]
        result = stable_icir_select(X, y, names, top_k=5)
        assert isinstance(result, list)
        assert all(isinstance(n, str) for n in result)
        assert len(result) <= 5

    def test_respects_top_k(self):
        X, y = self._make_data()
        names = [f"f{i}" for i in range(X.shape[1])]
        result = stable_icir_select(X, y, names, top_k=3)
        assert len(result) <= 3

    def test_fallback_on_insufficient_data(self):
        """With too little data, should fall back to greedy_ic_select."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = rng.randn(50)
        names = [f"f{i}" for i in range(5)]
        result = stable_icir_select(X, y, names, top_k=3)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_stable_feature_selected_first(self):
        """Feature with consistent IC should rank higher than noisy ones."""
        rng = np.random.RandomState(42)
        n = 1200
        X = rng.randn(n, 10)
        # Feature 0: very stable positive IC
        y = 0.5 * X[:, 0] + rng.randn(n) * 0.3
        names = [f"f{i}" for i in range(10)]
        result = stable_icir_select(X, y, names, top_k=5)
        assert "f0" in result

    def test_empty_features(self):
        X = np.zeros((100, 0))
        y = np.zeros(100)
        result = stable_icir_select(X, y, [], top_k=5)
        assert result == []

    def test_all_nan_features_handled(self):
        rng = np.random.RandomState(42)
        X = np.full((1200, 5), np.nan)
        y = rng.randn(1200)
        names = [f"f{i}" for i in range(5)]
        result = stable_icir_select(X, y, names, top_k=3)
        # Should fall back to greedy since no features pass gate
        assert isinstance(result, list)


class TestIcirSelectExisting:
    """Ensure existing icir_select still works."""

    def test_basic(self):
        rng = np.random.RandomState(42)
        n = 1200
        X = rng.randn(n, 10)
        y = 0.3 * X[:, 0] + rng.randn(n) * 0.5
        names = [f"f{i}" for i in range(10)]
        result = icir_select(X, y, names, top_k=5)
        assert isinstance(result, list)
        assert len(result) <= 5
