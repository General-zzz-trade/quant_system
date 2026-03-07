"""Tests for C++ Black-Litterman posterior computation."""
from __future__ import annotations

import random

import pytest

try:
    from features._quant_rolling import cpp_black_litterman_posterior
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ extension not built")


def _make_symmetric_cov(n, seed=42):
    """Generate a symmetric positive-definite covariance matrix."""
    random.seed(seed)
    # Random lower triangular, then L @ L'
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            L[i][j] = random.gauss(0, 0.01)
        L[i][i] = abs(L[i][i]) + 0.001  # ensure positive diagonal
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(min(i, j) + 1):
                s += L[i][k] * L[j][k]
            cov[i][j] = s
    return cov


class TestCppBlackLittermanPosterior:
    def test_no_views(self):
        """No views → posterior = equilibrium."""
        N = 5
        cov = _make_symmetric_cov(N)
        w = [1.0 / N] * N
        mu_post, post_cov, eq_pi = cpp_black_litterman_posterior(
            cov, w, [], [], [], 0.05, 2.5)
        # With no views, posterior = equilibrium
        assert len(mu_post) == N
        assert len(eq_pi) == N
        for i in range(N):
            assert mu_post[i] == pytest.approx(eq_pi[i])

    def test_single_absolute_view(self):
        """Single absolute view shifts posterior toward view."""
        N = 3
        cov = _make_symmetric_cov(N, seed=1)
        w = [1.0 / N] * N
        # View: asset 0 expected return = 0.10 with high confidence
        P = [[1.0, 0.0, 0.0]]
        Q = [0.10]
        conf = [10.0]
        mu_post, _, eq_pi = cpp_black_litterman_posterior(
            cov, w, P, Q, conf, 0.05, 2.5)
        # Posterior for asset 0 should be closer to 0.10 than equilibrium
        assert abs(mu_post[0] - 0.10) < abs(eq_pi[0] - 0.10)

    def test_relative_view(self):
        """Relative view: asset 0 outperforms asset 1."""
        N = 4
        cov = _make_symmetric_cov(N, seed=2)
        w = [0.25] * N
        P = [[1.0, -1.0, 0.0, 0.0]]
        Q = [0.05]  # asset 0 outperforms asset 1 by 5%
        conf = [5.0]
        mu_post, _, _ = cpp_black_litterman_posterior(
            cov, w, P, Q, conf, 0.05, 2.5)
        # Posterior spread between asset 0 and 1 should be positive
        assert mu_post[0] - mu_post[1] > 0

    def test_multiple_views(self):
        """Multiple views all applied."""
        N = 5
        cov = _make_symmetric_cov(N, seed=3)
        w = [0.2] * N
        P = [
            [1.0, 0.0, 0.0, 0.0, 0.0],  # abs view on asset 0
            [0.0, 1.0, -1.0, 0.0, 0.0],  # relative: asset 1 vs 2
        ]
        Q = [0.08, 0.03]
        conf = [5.0, 3.0]
        mu_post, post_cov, eq_pi = cpp_black_litterman_posterior(
            cov, w, P, Q, conf, 0.05, 2.5)
        assert len(mu_post) == N
        assert len(post_cov) == N
        assert len(post_cov[0]) == N

    def test_posterior_covariance_symmetric(self):
        """Posterior covariance should be symmetric."""
        N = 10
        cov = _make_symmetric_cov(N, seed=4)
        w = [1.0 / N] * N
        P = [[0.0] * N for _ in range(3)]
        P[0][0] = 1.0
        P[1][2] = 1.0
        P[2][4] = 1.0; P[2][5] = -1.0
        Q = [0.05, 0.03, 0.02]
        conf = [2.0, 3.0, 1.0]
        _, post_cov, _ = cpp_black_litterman_posterior(
            cov, w, P, Q, conf, 0.05, 2.5)
        for i in range(N):
            for j in range(N):
                assert post_cov[i][j] == pytest.approx(post_cov[j][i], rel=1e-10)

    def test_high_confidence_dominates(self):
        """Very high confidence → posterior ≈ view."""
        N = 3
        cov = _make_symmetric_cov(N, seed=5)
        w = [1.0 / N] * N
        P = [[1.0, 0.0, 0.0]]
        Q = [0.20]
        conf = [1000.0]  # extremely confident
        mu_post, _, _ = cpp_black_litterman_posterior(
            cov, w, P, Q, conf, 0.05, 2.5)
        assert mu_post[0] == pytest.approx(0.20, abs=0.02)

    def test_matches_python(self):
        """Verify C++ matches Python BlackLittermanModel.posterior."""
        import portfolio.optimizer.black_litterman as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        N = 20
        symbols = [f"S{i}" for i in range(N)]
        cov = {}
        for s1 in symbols:
            row = {}
            for s2 in symbols:
                row[s2] = random.gauss(0, 0.001)
            row[s1] = random.uniform(0.001, 0.01)
            cov[s1] = row
        # Make symmetric
        for s1 in symbols:
            for s2 in symbols:
                avg = (cov[s1][s2] + cov[s2][s1]) / 2
                cov[s1][s2] = avg
                cov[s2][s1] = avg

        mkt_w = {s: 1.0 / N for s in symbols}
        views = [
            mod.ViewSpec(assets=(symbols[0],), weights=(1.0,),
                        expected_return=0.05, confidence=2.0),
            mod.ViewSpec(assets=(symbols[1], symbols[2]),
                        weights=(1.0, -1.0), expected_return=0.02, confidence=1.5),
            mod.ViewSpec(assets=(symbols[3],), weights=(1.0,),
                        expected_return=-0.01, confidence=1.0),
        ]

        config = mod.BlackLittermanConfig(tau=0.05, risk_aversion=2.5)
        model = mod.BlackLittermanModel(config)
        py_result = model.posterior(mkt_w, cov, views)
        mod._USING_CPP = saved

        # C++ path
        mod._USING_CPP = True
        cpp_result = model.posterior(mkt_w, cov, views)
        mod._USING_CPP = saved

        for s in symbols:
            assert cpp_result.posterior_returns[s] == pytest.approx(
                py_result.posterior_returns[s], rel=1e-9), f"return mismatch at {s}"
            assert cpp_result.equilibrium_returns[s] == pytest.approx(
                py_result.equilibrium_returns[s], rel=1e-9), f"eq mismatch at {s}"
            for s2 in symbols:
                assert cpp_result.posterior_covariance[s][s2] == pytest.approx(
                    py_result.posterior_covariance[s][s2], rel=1e-9), \
                    f"cov mismatch at ({s},{s2})"

    def test_dispatch_integration(self):
        """BlackLittermanModel uses C++ dispatch."""
        from portfolio.optimizer.black_litterman import (
            BlackLittermanModel, BlackLittermanConfig, ViewSpec,
        )
        random.seed(99)
        N = 5
        symbols = [f"S{i}" for i in range(N)]
        cov = {}
        for s1 in symbols:
            row = {}
            for s2 in symbols:
                row[s2] = random.gauss(0, 0.001)
            row[s1] = random.uniform(0.001, 0.01)
            cov[s1] = row
        for s1 in symbols:
            for s2 in symbols:
                avg = (cov[s1][s2] + cov[s2][s1]) / 2
                cov[s1][s2] = avg
                cov[s2][s1] = avg

        mkt_w = {s: 1.0 / N for s in symbols}
        views = [ViewSpec(assets=(symbols[0],), weights=(1.0,),
                          expected_return=0.05, confidence=2.0)]
        model = BlackLittermanModel(BlackLittermanConfig())
        result = model.posterior(mkt_w, cov, views)
        assert symbols[0] in result.posterior_returns
        assert len(result.posterior_covariance) == N
