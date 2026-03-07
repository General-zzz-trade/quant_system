"""Tests for C++ factor model math: exposures, factor model covariance, specific risk."""
from __future__ import annotations

import random

import pytest

try:
    from features._quant_rolling import (
        cpp_compute_exposures,
        cpp_factor_model_covariance,
        cpp_estimate_specific_risk,
    )
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ extension not built")


# ---------------------------------------------------------------------------
# Compute Exposures (batch beta)
# ---------------------------------------------------------------------------
class TestCppComputeExposures:
    def test_perfect_beta(self):
        """asset = 2 * factor → beta = 2.0."""
        T = 200
        random.seed(42)
        factor = [random.gauss(0, 0.01) for _ in range(T)]
        asset = [2 * f for f in factor]
        result = cpp_compute_exposures([asset], [factor])
        assert result[0][0] == pytest.approx(2.0, abs=0.01)

    def test_zero_variance_factor(self):
        """Constant factor → beta = 0."""
        asset = [0.01] * 50
        factor = [0.0] * 50
        result = cpp_compute_exposures([asset], [factor])
        assert result[0][0] == 0.0

    def test_multiple_assets_factors(self):
        """Univariate betas with independent factors."""
        random.seed(1)
        T = 1000
        # Use truly independent factors for clean univariate beta recovery
        f0 = [random.gauss(0, 0.01) for _ in range(T)]
        f1 = [random.gauss(0, 0.01) for _ in range(T)]
        f2 = [random.gauss(0, 0.01) for _ in range(T)]
        factors = [f0, f1, f2]
        assets = [
            [1.5 * f0[t] + random.gauss(0, 0.001) for t in range(T)],
            [2.0 * f2[t] + random.gauss(0, 0.001) for t in range(T)],
        ]
        result = cpp_compute_exposures(assets, factors)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert result[0][0] == pytest.approx(1.5, abs=0.15)
        assert abs(result[0][1]) < 0.15  # no loading on f1
        assert result[1][2] == pytest.approx(2.0, abs=0.15)

    def test_empty(self):
        assert cpp_compute_exposures([], []) == []
        result = cpp_compute_exposures([[1.0, 2.0]], [])
        assert result == [[]]

    def test_matches_python(self):
        """Verify C++ matches Python compute_exposures."""
        import portfolio.risk_model.factor.exposure as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        N, F, T = 20, 5, 200
        symbols = [f"S{i}" for i in range(N)]
        factor_names = [f"F{i}" for i in range(F)]
        returns = {s: [random.gauss(0.001, 0.02) for _ in range(T)] for s in symbols}
        factor_returns = {f: [random.gauss(0, 0.01) for _ in range(T)] for f in factor_names}

        py_result = mod.compute_exposures(symbols, returns, factor_returns)
        mod._USING_CPP = saved

        a_mat = [list(returns[s]) for s in symbols]
        f_mat = [list(factor_returns[f]) for f in factor_names]
        cpp_mat = cpp_compute_exposures(a_mat, f_mat)

        for i, s in enumerate(symbols):
            for j, f in enumerate(factor_names):
                assert cpp_mat[i][j] == pytest.approx(py_result[s][f], rel=1e-9), \
                    f"mismatch at ({s},{f})"

    def test_dispatch_integration(self):
        from portfolio.risk_model.factor.exposure import compute_exposures
        random.seed(99)
        symbols = ["A", "B"]
        returns = {s: [random.gauss(0, 0.01) for _ in range(50)] for s in symbols}
        factors = {"MKT": [random.gauss(0, 0.01) for _ in range(50)]}
        result = compute_exposures(symbols, returns, factors)
        assert "A" in result and "MKT" in result["A"]


# ---------------------------------------------------------------------------
# Factor Model Covariance
# ---------------------------------------------------------------------------
class TestCppFactorModelCovariance:
    def test_single_factor(self):
        """One factor, two assets with different betas."""
        exposures = [[1.0], [2.0]]  # asset 0: beta=1, asset 1: beta=2
        factor_cov = [[0.01]]  # factor variance = 0.01
        specific = [0.001, 0.002]
        result = cpp_factor_model_covariance(exposures, factor_cov, specific)
        # cov(0,0) = 1*0.01*1 + 0.001 = 0.011
        assert result[0][0] == pytest.approx(0.011)
        # cov(0,1) = 1*0.01*2 = 0.02
        assert result[0][1] == pytest.approx(0.02)
        # cov(1,1) = 2*0.01*2 + 0.002 = 0.042
        assert result[1][1] == pytest.approx(0.042)

    def test_symmetry(self):
        random.seed(3)
        N, F = 10, 3
        exposures = [[random.gauss(0, 1) for _ in range(F)] for _ in range(N)]
        factor_cov = [[random.gauss(0, 0.01) for _ in range(F)] for _ in range(F)]
        # Make factor_cov symmetric
        for i in range(F):
            for j in range(i + 1, F):
                factor_cov[j][i] = factor_cov[i][j]
        specific = [random.uniform(0, 0.01) for _ in range(N)]
        result = cpp_factor_model_covariance(exposures, factor_cov, specific)
        for i in range(N):
            for j in range(N):
                assert result[i][j] == pytest.approx(result[j][i], rel=1e-12)

    def test_no_factors(self):
        """No factors → only diagonal specific risk."""
        result = cpp_factor_model_covariance([[]], [], [0.05, 0.03])
        # With no factors, result should be zero (exposures are empty)
        # Actually, exposures is [[]], factor_cov is [], so F=0
        assert result[0][0] == pytest.approx(0.05)

    def test_matches_python(self):
        """Verify C++ matches Python factor_model_covariance."""
        import portfolio.risk_model.factor.covariance as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        N, F = 20, 5
        symbols = [f"S{i}" for i in range(N)]
        factor_names = [f"F{i}" for i in range(F)]

        exposures = {s: {f: random.gauss(0, 1) for f in factor_names} for s in symbols}
        factor_cov = {}
        for f1 in factor_names:
            row = {}
            for f2 in factor_names:
                row[f2] = random.gauss(0, 0.001)
            row[f1] = random.uniform(0.0001, 0.01)
            factor_cov[f1] = row
        specific_risk = {s: random.uniform(0.0001, 0.001) for s in symbols}

        py_result = mod.factor_model_covariance(symbols, exposures, factor_cov, specific_risk)
        mod._USING_CPP = saved

        exp_mat = [[exposures[s][f] for f in factor_names] for s in symbols]
        fcov_mat = [[factor_cov[f1][f2] for f2 in factor_names] for f1 in factor_names]
        sr_vec = [specific_risk[s] for s in symbols]
        cpp_mat = cpp_factor_model_covariance(exp_mat, fcov_mat, sr_vec)

        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                assert cpp_mat[i][j] == pytest.approx(py_result[s1][s2], rel=1e-9), \
                    f"mismatch at ({s1},{s2})"

    def test_dispatch_integration(self):
        from portfolio.risk_model.factor.covariance import factor_model_covariance
        symbols = ["A", "B"]
        exposures = {"A": {"MKT": 1.0}, "B": {"MKT": 0.5}}
        factor_cov = {"MKT": {"MKT": 0.01}}
        specific = {"A": 0.001, "B": 0.002}
        result = factor_model_covariance(symbols, exposures, factor_cov, specific)
        assert "A" in result and "B" in result["A"]
        assert result["A"]["A"] > 0


# ---------------------------------------------------------------------------
# Estimate Specific Risk
# ---------------------------------------------------------------------------
class TestCppEstimateSpecificRisk:
    def test_perfect_fit(self):
        """asset = beta * factor → residual ~0, specific risk ~0."""
        T = 200
        random.seed(42)
        factor = [random.gauss(0, 0.01) for _ in range(T)]
        asset = [1.5 * f for f in factor]
        result = cpp_estimate_specific_risk([asset], [factor], [[1.5]])
        assert result[0] == pytest.approx(0.0, abs=1e-20)

    def test_pure_noise(self):
        """asset = noise, no factor loading → specific risk = var(asset)."""
        T = 5000
        random.seed(42)
        asset = [random.gauss(0, 0.05) for _ in range(T)]
        factor = [random.gauss(0, 0.01) for _ in range(T)]
        # beta = 0 → residual = asset itself
        result = cpp_estimate_specific_risk([asset], [factor], [[0.0]])
        expected_var = sum((a - sum(asset) / T) ** 2 for a in asset) / (T - 1)
        assert result[0] == pytest.approx(expected_var, rel=1e-9)

    def test_multiple_assets(self):
        T = 200
        random.seed(1)
        factor = [random.gauss(0, 0.01) for _ in range(T)]
        assets = [
            [1.0 * factor[t] + random.gauss(0, 0.005) for t in range(T)],
            [2.0 * factor[t] + random.gauss(0, 0.010) for t in range(T)],
        ]
        result = cpp_estimate_specific_risk(assets, [factor], [[1.0], [2.0]])
        assert len(result) == 2
        # Specific risk for asset 1 should be higher (more noise)
        assert result[1] > result[0]

    def test_empty(self):
        assert cpp_estimate_specific_risk([], [], []) == []

    def test_matches_python(self):
        """Verify C++ matches Python estimate_specific_risk."""
        import portfolio.risk_model.factor.specific_risk as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        N, F, T = 20, 5, 200
        symbols = [f"S{i}" for i in range(N)]
        factor_names = [f"F{i}" for i in range(F)]
        returns = {s: [random.gauss(0.001, 0.02) for _ in range(T)] for s in symbols}
        factor_returns = {f: [random.gauss(0, 0.01) for _ in range(T)] for f in factor_names}
        exposures = {s: {f: random.gauss(0, 1) for f in factor_names} for s in symbols}

        py_result = mod.estimate_specific_risk(symbols, returns, factor_returns, exposures)
        mod._USING_CPP = saved

        a_mat = [list(returns[s][:T]) for s in symbols]
        f_mat = [list(factor_returns[f][:T]) for f in factor_names]
        exp_mat = [[exposures[s][f] for f in factor_names] for s in symbols]
        cpp_vec = cpp_estimate_specific_risk(a_mat, f_mat, exp_mat)

        for i, s in enumerate(symbols):
            assert cpp_vec[i] == pytest.approx(py_result[s], rel=1e-9), \
                f"mismatch at {s}"

    def test_dispatch_integration(self):
        from portfolio.risk_model.factor.specific_risk import estimate_specific_risk
        random.seed(99)
        symbols = ["A", "B"]
        returns = {s: [random.gauss(0, 0.01) for _ in range(50)] for s in symbols}
        factors = {"MKT": [random.gauss(0, 0.01) for _ in range(50)]}
        exposures = {"A": {"MKT": 1.0}, "B": {"MKT": 0.5}}
        result = estimate_specific_risk(symbols, returns, factors, exposures)
        assert "A" in result and result["A"] >= 0


# ---------------------------------------------------------------------------
# Estimate Factor Covariance (reuses cpp_sample_covariance)
# ---------------------------------------------------------------------------
class TestEstimateFactorCovarianceDispatch:
    def test_dispatch(self):
        from portfolio.risk_model.factor.covariance import estimate_factor_covariance
        random.seed(42)
        factors = {"MKT": [random.gauss(0, 0.01) for _ in range(100)],
                   "SMB": [random.gauss(0, 0.005) for _ in range(100)]}
        result = estimate_factor_covariance(factors)
        assert "MKT" in result and "SMB" in result["MKT"]
        assert result["MKT"]["MKT"] > 0

    def test_matches_python(self):
        import portfolio.risk_model.factor.covariance as mod
        saved = mod._USING_CPP
        mod._USING_CPP = False

        random.seed(42)
        factors = {f"F{i}": [random.gauss(0, 0.01) for _ in range(200)] for i in range(5)}
        py_result = mod.estimate_factor_covariance(factors)
        mod._USING_CPP = saved

        cpp_result = mod.estimate_factor_covariance(factors)
        for f1 in factors:
            for f2 in factors:
                assert cpp_result[f1][f2] == pytest.approx(py_result[f1][f2], rel=1e-9)
