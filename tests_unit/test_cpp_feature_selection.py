"""Tests for C++ feature selection: correlation and mutual information."""
from __future__ import annotations

import math
import random

import pytest

try:
    from features._quant_rolling import (
        cpp_correlation_select,
        cpp_mutual_info_select,
    )
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ extension not built")


# ---------------------------------------------------------------------------
# Correlation Select
# ---------------------------------------------------------------------------
class TestCppCorrelationSelect:
    def test_perfect_correlation(self):
        """Feature = 2*target → |cor| = 1.0."""
        target = [random.gauss(0, 1) for _ in range(500)]
        feat = [2 * t for t in target]
        result = cpp_correlation_select([feat], target)
        assert result[0] == pytest.approx(1.0, abs=1e-10)

    def test_negative_correlation(self):
        """Feature = -target → |cor| = 1.0."""
        target = [random.gauss(0, 1) for _ in range(500)]
        feat = [-t for t in target]
        result = cpp_correlation_select([feat], target)
        assert result[0] == pytest.approx(1.0, abs=1e-10)

    def test_uncorrelated(self):
        """Independent features → low correlation."""
        random.seed(42)
        target = [random.gauss(0, 1) for _ in range(5000)]
        feat = [random.gauss(0, 1) for _ in range(5000)]
        result = cpp_correlation_select([feat], target)
        assert result[0] < 0.05

    def test_multiple_features(self):
        random.seed(1)
        T = 1000
        target = [random.gauss(0, 1) for _ in range(T)]
        f_perfect = [t + random.gauss(0, 0.01) for t in target]
        f_noise = [random.gauss(0, 1) for _ in range(T)]
        f_half = [0.5 * t + random.gauss(0, 0.5) for t in target]
        result = cpp_correlation_select([f_perfect, f_noise, f_half], target)
        assert len(result) == 3
        assert result[0] > result[2] > result[1]

    def test_constant_feature(self):
        """Constant feature → correlation = 0."""
        target = [1.0, 2.0, 3.0, 4.0, 5.0]
        feat = [5.0] * 5
        result = cpp_correlation_select([feat], target)
        assert result[0] == 0.0

    def test_empty(self):
        assert cpp_correlation_select([], [1.0, 2.0]) == []

    def test_short_series(self):
        result = cpp_correlation_select([[1.0]], [2.0])
        assert result[0] == 0.0

    def test_matches_python(self):
        """Verify C++ matches Python FeatureSelector correlation."""
        import features.auto.selector as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        T = 2000
        F = 30
        target = [random.gauss(0, 1) for _ in range(T)]
        features = {f"feat_{i}": [random.gauss(0, 1) for _ in range(T)] for i in range(F)}

        sel = mod.FeatureSelector(method="correlation", top_k=F)
        py_scores = sel.select(features, target)
        mod._USING_CPP = saved

        py_map = {s.name: s.score for s in py_scores}

        names = list(features.keys())
        feat_mat = [list(features[n]) for n in names]
        cpp_vec = cpp_correlation_select(feat_mat, target)

        for i, name in enumerate(names):
            assert cpp_vec[i] == pytest.approx(py_map[name], rel=1e-9), \
                f"mismatch at {name}"

    def test_dispatch_integration(self):
        """FeatureSelector uses C++ when available."""
        from features.auto.selector import FeatureSelector
        random.seed(99)
        T = 100
        target = [random.gauss(0, 1) for _ in range(T)]
        features = {f"f{i}": [random.gauss(0, 1) for _ in range(T)] for i in range(5)}
        sel = FeatureSelector(method="correlation", top_k=3)
        scores = sel.select(features, target)
        assert len(scores) == 3
        assert all(0 <= s.score <= 1 for s in scores)


# ---------------------------------------------------------------------------
# Mutual Info Select
# ---------------------------------------------------------------------------
class TestCppMutualInfoSelect:
    def test_identical(self):
        """MI(X, X) > 0."""
        random.seed(42)
        target = [random.gauss(0, 1) for _ in range(1000)]
        result = cpp_mutual_info_select([target], target, 30)
        assert result[0] > 0.5

    def test_independent(self):
        """Independent → MI ~ 0."""
        random.seed(42)
        target = [random.gauss(0, 1) for _ in range(5000)]
        feat = [random.gauss(0, 1) for _ in range(5000)]
        result = cpp_mutual_info_select([feat], target, 30)
        assert result[0] < 0.1

    def test_nonlinear_dependency(self):
        """X^2 has MI with X (nonlinear)."""
        random.seed(42)
        target = [random.gauss(0, 1) for _ in range(2000)]
        feat_sq = [t ** 2 for t in target]
        feat_ind = [random.gauss(0, 1) for _ in range(2000)]
        result = cpp_mutual_info_select([feat_sq, feat_ind], target, 30)
        assert result[0] > result[1]

    def test_multiple_features_ranking(self):
        random.seed(1)
        T = 2000
        target = [random.gauss(0, 1) for _ in range(T)]
        f_strong = [t + random.gauss(0, 0.1) for t in target]
        f_weak = [t + random.gauss(0, 2.0) for t in target]
        f_noise = [random.gauss(0, 1) for _ in range(T)]
        result = cpp_mutual_info_select([f_strong, f_weak, f_noise], target, 30)
        assert result[0] > result[1] > result[2]

    def test_constant_values(self):
        """Constant feature → MI = 0."""
        target = list(range(100))
        feat = [5.0] * 100
        result = cpp_mutual_info_select([feat], [float(t) for t in target], 10)
        assert result[0] == 0.0

    def test_empty(self):
        assert cpp_mutual_info_select([], [1.0, 2.0], 5) == []

    def test_non_negative(self):
        """MI should always be >= 0."""
        random.seed(42)
        T = 500
        target = [random.gauss(0, 1) for _ in range(T)]
        feats = [[random.gauss(0, 1) for _ in range(T)] for _ in range(20)]
        result = cpp_mutual_info_select(feats, target, 20)
        assert all(v >= 0 for v in result)

    def test_matches_python(self):
        """Verify C++ matches Python FeatureSelector mutual_info."""
        import features.auto.selector as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        T = 2000
        F = 30
        target = [random.gauss(0, 1) for _ in range(T)]
        features = {f"feat_{i}": [random.gauss(0, 1) for _ in range(T)] for i in range(F)}

        sel = mod.FeatureSelector(method="mutual_info", top_k=F)
        py_scores = sel.select(features, target)
        mod._USING_CPP = saved

        py_map = {s.name: s.score for s in py_scores}

        names = list(features.keys())
        feat_mat = [list(features[n]) for n in names]
        n_bins = max(5, int(math.sqrt(T)))
        cpp_vec = cpp_mutual_info_select(feat_mat, [float(t) for t in target], n_bins)

        for i, name in enumerate(names):
            assert cpp_vec[i] == pytest.approx(py_map[name], rel=1e-9), \
                f"mismatch at {name}"

    def test_dispatch_integration(self):
        """FeatureSelector uses C++ when available."""
        from features.auto.selector import FeatureSelector
        random.seed(99)
        T = 200
        target = [random.gauss(0, 1) for _ in range(T)]
        features = {f"f{i}": [random.gauss(0, 1) for _ in range(T)] for i in range(5)}
        sel = FeatureSelector(method="mutual_info", top_k=3)
        scores = sel.select(features, target)
        assert len(scores) == 3
        assert all(s.score >= 0 for s in scores)
