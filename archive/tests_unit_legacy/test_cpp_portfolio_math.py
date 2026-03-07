"""Tests for C++ portfolio math: covariance, correlation, portfolio variance."""
from __future__ import annotations

import math
import random

import pytest

try:
    from features._quant_rolling import (
        cpp_sample_covariance,
        cpp_ewma_covariance,
        cpp_rolling_correlation,
        cpp_portfolio_variance,
    )
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ extension not built")


# ---------------------------------------------------------------------------
# Sample Covariance
# ---------------------------------------------------------------------------
class TestCppSampleCovariance:
    def test_identity(self):
        """Uncorrelated unit variance → diagonal ~1."""
        random.seed(42)
        n = 10000
        matrix = [
            [random.gauss(0, 1) for _ in range(n)],
            [random.gauss(0, 1) for _ in range(n)],
        ]
        cov = cpp_sample_covariance(matrix)
        assert cov[0][0] == pytest.approx(1.0, abs=0.05)
        assert cov[1][1] == pytest.approx(1.0, abs=0.05)
        assert abs(cov[0][1]) < 0.05
        assert cov[0][1] == pytest.approx(cov[1][0])

    def test_perfect_correlation(self):
        """x and 2x → cov(x,2x) = 2*var(x)."""
        x = [float(i) for i in range(100)]
        matrix = [x, [2 * v for v in x]]
        cov = cpp_sample_covariance(matrix)
        assert cov[0][1] == pytest.approx(2 * cov[0][0], rel=1e-10)

    def test_symmetry(self):
        random.seed(1)
        M = 5
        matrix = [[random.gauss(0, 1) for _ in range(50)] for _ in range(M)]
        cov = cpp_sample_covariance(matrix)
        for i in range(M):
            for j in range(M):
                assert cov[i][j] == pytest.approx(cov[j][i], rel=1e-12)

    def test_empty(self):
        assert cpp_sample_covariance([]) == []

    def test_single_obs(self):
        cov = cpp_sample_covariance([[1.0], [2.0]])
        assert cov[0][0] == 0.0
        assert cov[0][1] == 0.0

    def test_matches_python(self):
        """Verify C++ matches Python SampleCovariance."""
        from portfolio.risk_model.covariance.sample import SampleCovariance, _USING_CPP
        import portfolio.risk_model.covariance.sample as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        N, M = 20, 200
        symbols = [f"S{i}" for i in range(N)]
        returns = {s: [random.gauss(0.001, 0.02) for _ in range(M)] for s in symbols}

        sc = SampleCovariance()
        py_result = sc.estimate(symbols, returns)
        mod._USING_CPP = saved

        matrix = [list(returns[s][:M]) for s in symbols]
        cpp_cov = cpp_sample_covariance(matrix)

        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                assert cpp_cov[i][j] == pytest.approx(py_result[s1][s2], rel=1e-10), \
                    f"mismatch at ({s1},{s2})"

    def test_dispatch_integration(self):
        from portfolio.risk_model.covariance.sample import SampleCovariance
        random.seed(99)
        symbols = ["A", "B", "C"]
        returns = {s: [random.gauss(0, 0.01) for _ in range(50)] for s in symbols}
        result = SampleCovariance().estimate(symbols, returns)
        assert "A" in result and "B" in result["A"]
        assert result["A"]["A"] > 0


# ---------------------------------------------------------------------------
# EWMA Covariance
# ---------------------------------------------------------------------------
class TestCppEwmaCovariance:
    def test_basic(self):
        random.seed(42)
        matrix = [[random.gauss(0, 1) for _ in range(100)] for _ in range(3)]
        cov = cpp_ewma_covariance(matrix, 0.06)
        assert len(cov) == 3
        for i in range(3):
            assert cov[i][i] > 0  # variance positive

    def test_symmetry(self):
        random.seed(2)
        matrix = [[random.gauss(0, 1) for _ in range(50)] for _ in range(5)]
        cov = cpp_ewma_covariance(matrix, 0.1)
        for i in range(5):
            for j in range(5):
                assert cov[i][j] == pytest.approx(cov[j][i], rel=1e-12)

    def test_empty(self):
        assert cpp_ewma_covariance([], 0.1) == []

    def test_matches_python(self):
        """Verify C++ matches Python EWMACovariance."""
        from portfolio.risk_model.covariance.ewma import EWMACovariance
        import portfolio.risk_model.covariance.ewma as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        N, M = 15, 200
        symbols = [f"S{i}" for i in range(N)]
        returns = {s: [random.gauss(0.001, 0.02) for _ in range(M)] for s in symbols}

        ewma = EWMACovariance(span=30)
        py_result = ewma.estimate(symbols, returns)
        mod._USING_CPP = saved

        matrix = [list(returns[s][:M]) for s in symbols]
        cpp_cov = cpp_ewma_covariance(matrix, ewma.alpha)

        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                assert cpp_cov[i][j] == pytest.approx(py_result[s1][s2], rel=1e-9), \
                    f"mismatch at ({s1},{s2})"

    def test_dispatch_integration(self):
        from portfolio.risk_model.covariance.ewma import EWMACovariance
        random.seed(99)
        symbols = ["A", "B"]
        returns = {s: [random.gauss(0, 0.01) for _ in range(50)] for s in symbols}
        result = EWMACovariance(span=20).estimate(symbols, returns)
        assert result["A"]["A"] > 0


# ---------------------------------------------------------------------------
# Rolling Correlation
# ---------------------------------------------------------------------------
class TestCppRollingCorrelation:
    def test_self_correlation(self):
        """Diagonal = 1.0."""
        matrix = [[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]]
        corr = cpp_rolling_correlation(matrix, 5)
        assert corr[0][0] == pytest.approx(1.0)
        assert corr[1][1] == pytest.approx(1.0)

    def test_perfect_negative(self):
        """Perfectly inversely correlated → -1."""
        x = [float(i) for i in range(20)]
        matrix = [x, [-v for v in x]]
        corr = cpp_rolling_correlation(matrix, 20)
        assert corr[0][1] == pytest.approx(-1.0, abs=1e-10)

    def test_perfect_positive(self):
        """Perfectly correlated → 1."""
        x = [float(i) for i in range(20)]
        matrix = [x, [2 * v + 3 for v in x]]
        corr = cpp_rolling_correlation(matrix, 20)
        assert corr[0][1] == pytest.approx(1.0, abs=1e-10)

    def test_symmetry(self):
        random.seed(3)
        matrix = [[random.gauss(0, 1) for _ in range(60)] for _ in range(5)]
        corr = cpp_rolling_correlation(matrix, 60)
        for i in range(5):
            for j in range(5):
                assert corr[i][j] == pytest.approx(corr[j][i], rel=1e-12)

    def test_window_slicing(self):
        """Uses only last `window` observations."""
        # First 50 obs: correlated. Last 10: uncorrelated.
        random.seed(10)
        x = [float(i) for i in range(50)] + [random.gauss(0, 1) for _ in range(10)]
        y = [float(i) for i in range(50)] + [random.gauss(0, 1) for _ in range(10)]
        # Window=10 should see uncorrelated data
        corr = cpp_rolling_correlation([x, y], 10)
        assert abs(corr[0][1]) < 0.9  # not perfect correlation

    def test_matches_python(self):
        """Verify C++ matches Python RollingCorrelation."""
        from portfolio.risk_model.correlation.rolling import RollingCorrelation
        import portfolio.risk_model.correlation.rolling as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        N = 10
        symbols = [f"S{i}" for i in range(N)]
        returns = {s: [random.gauss(0, 0.02) for _ in range(200)] for s in symbols}

        rc = RollingCorrelation(window=60)
        py_result = rc.estimate(symbols, returns)
        mod._USING_CPP = saved

        matrix = [list(returns[s]) for s in symbols]
        cpp_corr = cpp_rolling_correlation(matrix, 60)

        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                assert cpp_corr[i][j] == pytest.approx(py_result[s1][s2], rel=1e-9), \
                    f"mismatch at ({s1},{s2})"

    def test_invalid_window(self):
        with pytest.raises(Exception):
            cpp_rolling_correlation([[1.0]], 0)

    def test_dispatch_integration(self):
        from portfolio.risk_model.correlation.rolling import RollingCorrelation
        random.seed(99)
        symbols = ["A", "B", "C"]
        returns = {s: [random.gauss(0, 0.01) for _ in range(100)] for s in symbols}
        result = RollingCorrelation(window=30).estimate(symbols, returns)
        assert result["A"]["A"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Portfolio Variance
# ---------------------------------------------------------------------------
class TestCppPortfolioVariance:
    def test_equal_weight(self):
        """Equal weight on 2 uncorrelated assets."""
        cov = [[0.04, 0.0], [0.0, 0.04]]
        weights = [0.5, 0.5]
        var = cpp_portfolio_variance(weights, cov)
        # 0.5^2 * 0.04 + 0.5^2 * 0.04 = 0.02
        assert var == pytest.approx(0.02)

    def test_single_asset(self):
        cov = [[0.09]]
        var = cpp_portfolio_variance([1.0], cov)
        assert var == pytest.approx(0.09)

    def test_correlated(self):
        """Two perfectly correlated assets with equal weight."""
        cov = [[0.04, 0.04], [0.04, 0.04]]
        var = cpp_portfolio_variance([0.5, 0.5], cov)
        # 0.5*0.5*0.04 * 4 = 0.04
        assert var == pytest.approx(0.04)

    def test_empty(self):
        assert cpp_portfolio_variance([], []) == 0.0

    def test_matches_python(self):
        """Verify C++ matches Python _portfolio_variance."""
        from portfolio.optimizer.objectives import _portfolio_variance

        random.seed(42)
        N = 20
        symbols = [f"S{i}" for i in range(N)]
        weights_dict = {s: random.uniform(0, 0.1) for s in symbols}
        total = sum(weights_dict.values())
        weights_dict = {s: w / total for s, w in weights_dict.items()}

        cov_dict = {}
        for s1 in symbols:
            row = {}
            for s2 in symbols:
                row[s2] = random.gauss(0, 0.001)
            row[s1] = random.uniform(0.0001, 0.001)
            cov_dict[s1] = row
        # Make symmetric
        for s1 in symbols:
            for s2 in symbols:
                cov_dict[s1][s2] = cov_dict[s2][s1] = (cov_dict[s1].get(s2, 0) + cov_dict[s2].get(s1, 0)) / 2

        py_var = _portfolio_variance(weights_dict, cov_dict)

        w_vec = [weights_dict[s] for s in symbols]
        cov_mat = [[cov_dict[s1][s2] for s2 in symbols] for s1 in symbols]
        cpp_var = cpp_portfolio_variance(w_vec, cov_mat)

        assert cpp_var == pytest.approx(py_var, rel=1e-10)

    def test_dispatch_integration(self):
        """Verify compute_portfolio_risk dispatches to C++."""
        from portfolio.risk_model.aggregation.portfolio_risk import compute_portfolio_risk
        cov = {"A": {"A": 0.04, "B": 0.01}, "B": {"A": 0.01, "B": 0.09}}
        weights = {"A": 0.6, "B": 0.4}
        result = compute_portfolio_risk(weights, cov)
        assert result.variance > 0
        assert result.volatility > 0
