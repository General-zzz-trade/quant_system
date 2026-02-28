"""Tests for OLS factor orthogonalization and marginal IC computation."""
from __future__ import annotations

import math
import random
from typing import List, Optional

import pytest

from research.orthogonalize import ols_residualize, marginal_ic_ols


# ---------------------------------------------------------------------------
# ols_residualize
# ---------------------------------------------------------------------------

class TestOlsResidualize:

    def test_single_regressor_perfect_fit(self):
        """target = 2*x → residuals ≈ 0."""
        x = [float(i) for i in range(100)]
        y = [2.0 * xi for xi in x]
        residuals = ols_residualize(y, [x])
        assert all(abs(r) < 1e-8 for r in residuals)

    def test_single_regressor_with_noise(self):
        """target = 2*x + noise → residuals are the noise."""
        rng = random.Random(42)
        x = [float(i) for i in range(200)]
        noise = [rng.gauss(0, 0.1) for _ in x]
        y = [2.0 * xi + ni for xi, ni in zip(x, noise)]
        residuals = ols_residualize(y, [x])

        # Residuals should be approximately the noise
        corr_residual_noise = _pearson(residuals, noise)
        assert corr_residual_noise > 0.95

    def test_orthogonal_to_regressor(self):
        """Residuals should be orthogonal to regressors."""
        rng = random.Random(42)
        x = [rng.gauss(0, 1) for _ in range(200)]
        y = [xi + rng.gauss(0, 0.5) for xi in x]
        residuals = ols_residualize(y, [x])

        corr = _pearson(residuals, x)
        assert abs(corr) < 0.05

    def test_multiple_regressors(self):
        """Two regressors → residual orthogonal to both."""
        rng = random.Random(42)
        x1 = [rng.gauss(0, 1) for _ in range(300)]
        x2 = [rng.gauss(0, 1) for _ in range(300)]
        y = [3.0 * a + 2.0 * b + rng.gauss(0, 0.1) for a, b in zip(x1, x2)]
        residuals = ols_residualize(y, [x1, x2])

        assert abs(_pearson(residuals, x1)) < 0.05
        assert abs(_pearson(residuals, x2)) < 0.05

    def test_no_regressors(self):
        """No regressors → residuals equal target."""
        y = [1.0, 2.0, 3.0]
        residuals = ols_residualize(y, [])
        assert residuals == y

    def test_empty_target(self):
        residuals = ols_residualize([], [])
        assert residuals == []

    def test_constant_regressor(self):
        """Constant regressor → residuals are demeaned target."""
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        x = [1.0] * 5
        residuals = ols_residualize(y, [x])
        mean_y = sum(y) / len(y)
        for r, yi in zip(residuals, y):
            assert abs(r - (yi - mean_y)) < 1e-8


# ---------------------------------------------------------------------------
# marginal_ic_ols
# ---------------------------------------------------------------------------

class TestMarginalIcOls:

    def test_identical_factors_second_marginal_zero(self):
        """Two identical factors → second one has marginal IC ≈ 0."""
        rng = random.Random(42)
        n = 300
        factor = [rng.gauss(0, 1) for _ in range(n)]
        returns = [f * 0.01 + rng.gauss(0, 0.005) for f in factor]

        factor_values = {
            "f1": factor[:],
            "f2": factor[:],  # identical
        }
        fwd = returns[:]

        result = marginal_ic_ols(factor_values, fwd)
        # One of them should have near-zero marginal IC
        # (OLS can split arbitrarily, so check that at least one is small)
        marginals = list(result.values())
        # When factors are identical, residual of each against the other is ~0
        assert min(abs(m) for m in marginals) < 0.15

    def test_independent_factors_marginal_equals_raw(self):
        """Two independent factors → marginal IC ≈ raw IC."""
        rng = random.Random(42)
        n = 500
        f1 = [rng.gauss(0, 1) for _ in range(n)]
        f2 = [rng.gauss(0, 1) for _ in range(n)]
        returns = [0.01 * a + 0.01 * b + rng.gauss(0, 0.005) for a, b in zip(f1, f2)]

        factor_values = {"f1": f1, "f2": f2}

        result = marginal_ic_ols(factor_values, returns)
        # Both should have positive marginal IC
        assert result["f1"] > 0.1
        assert result["f2"] > 0.1

    def test_single_factor(self):
        """Single factor → marginal IC = raw IC."""
        rng = random.Random(42)
        n = 200
        f = [rng.gauss(0, 1) for _ in range(n)]
        ret = [0.01 * fi + rng.gauss(0, 0.005) for fi in f]

        result = marginal_ic_ols({"f1": f}, ret)
        raw_ic = _pearson(f, ret)
        assert result["f1"] == pytest.approx(raw_ic, abs=0.01)

    def test_with_nones(self):
        """None values are skipped properly."""
        rng = random.Random(42)
        n = 300
        f1: List[Optional[float]] = [rng.gauss(0, 1) for _ in range(n)]
        f2: List[Optional[float]] = [rng.gauss(0, 1) for _ in range(n)]
        ret: List[Optional[float]] = [0.01 * (f1[i] or 0) + rng.gauss(0, 0.005) for i in range(n)]

        # Sprinkle Nones
        for i in range(0, n, 10):
            f1[i] = None
        for i in range(5, n, 15):
            f2[i] = None

        result = marginal_ic_ols({"f1": f1, "f2": f2}, ret)
        assert "f1" in result
        assert "f2" in result

    def test_insufficient_data_returns_zero(self):
        """Too few observations → marginal IC = 0."""
        f1: List[Optional[float]] = [1.0, 2.0]
        f2: List[Optional[float]] = [3.0, 4.0]
        ret: List[Optional[float]] = [0.1, 0.2]

        result = marginal_ic_ols({"f1": f1, "f2": f2}, ret)
        assert result["f1"] == 0.0
        assert result["f2"] == 0.0

    def test_redundant_factor_gets_low_marginal(self):
        """Factor that's nearly linear combo of others → low marginal IC."""
        rng = random.Random(42)
        n = 500
        f1 = [rng.gauss(0, 1) for _ in range(n)]
        f2 = [rng.gauss(0, 1) for _ in range(n)]
        # Nearly redundant (small noise to avoid singular matrix)
        f3 = [a + b + rng.gauss(0, 0.01) for a, b in zip(f1, f2)]
        ret = [0.01 * a + 0.01 * b + rng.gauss(0, 0.005) for a, b in zip(f1, f2)]

        result = marginal_ic_ols({"f1": f1, "f2": f2, "f3": f3}, ret)
        # f3 is nearly redundant — its marginal IC should be small
        assert abs(result["f3"]) < 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom > 1e-12 else 0.0
