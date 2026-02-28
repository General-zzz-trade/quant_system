"""Tests for portfolio/risk_model/covariance estimators."""
from __future__ import annotations

import math

from portfolio.risk_model.covariance.sample import SampleCovariance
from portfolio.risk_model.covariance.ewma import EWMACovariance
from portfolio.risk_model.covariance.shrinkage import ShrinkageCovariance
from portfolio.risk_model.covariance.cleaning import CovarianceCleaning


# ── SampleCovariance ───────────────────────────────────────────────


class TestSampleCovariance:
    def test_known_2_asset(self):
        """Hand-computed sample covariance for 2 assets, 3 observations."""
        sc = SampleCovariance()
        returns = {
            "A": [0.01, -0.02, 0.03],
            "B": [0.02, 0.01, -0.01],
        }
        result = sc.estimate(["A", "B"], returns)
        # mean_A = (0.01 - 0.02 + 0.03)/3 = 0.02/3
        # mean_B = (0.02 + 0.01 - 0.01)/3 = 0.02/3
        ma = sum(returns["A"]) / 3
        mb = sum(returns["B"]) / 3
        var_a = sum((r - ma) ** 2 for r in returns["A"]) / 2
        var_b = sum((r - mb) ** 2 for r in returns["B"]) / 2
        cov_ab = sum(
            (returns["A"][i] - ma) * (returns["B"][i] - mb) for i in range(3)
        ) / 2
        assert abs(result["A"]["A"] - var_a) < 1e-12
        assert abs(result["B"]["B"] - var_b) < 1e-12
        assert abs(result["A"]["B"] - cov_ab) < 1e-12

    def test_n_less_than_2(self):
        sc = SampleCovariance()
        result = sc.estimate(["A", "B"], {"A": [0.01], "B": [0.02]})
        assert result["A"]["A"] == 0.0
        assert result["A"]["B"] == 0.0

    def test_symmetry(self):
        sc = SampleCovariance()
        returns = {
            "X": [0.01, -0.02, 0.03, 0.005, -0.01],
            "Y": [0.02, 0.01, -0.01, 0.03, -0.02],
        }
        result = sc.estimate(["X", "Y"], returns)
        assert abs(result["X"]["Y"] - result["Y"]["X"]) < 1e-12

    def test_diagonal_positive(self):
        sc = SampleCovariance()
        returns = {
            "A": [0.01, -0.02, 0.03, -0.01],
            "B": [0.02, 0.01, -0.01, 0.015],
        }
        result = sc.estimate(["A", "B"], returns)
        assert result["A"]["A"] > 0
        assert result["B"]["B"] > 0

    def test_empty_symbols(self):
        sc = SampleCovariance()
        result = sc.estimate([], {})
        assert result == {}


# ── EWMACovariance ─────────────────────────────────────────────────


class TestEWMACovariance:
    def test_alpha_computation(self):
        ec = EWMACovariance(span=9)
        assert abs(ec.alpha - 0.2) < 1e-12

    def test_ewma_recursion(self):
        """Verify EWMA recursion: cov_t = alpha*r1_t*r2_t + (1-alpha)*cov_{t-1}."""
        ec = EWMACovariance(span=9)
        alpha = 0.2
        returns = {
            "A": [0.01, -0.02, 0.03],
            "B": [0.02, 0.01, -0.01],
        }
        result = ec.estimate(["A", "B"], returns)

        # Manual recursion for cov(A, B)
        cov = returns["A"][0] * returns["B"][0]
        cov = alpha * returns["A"][1] * returns["B"][1] + (1 - alpha) * cov
        cov = alpha * returns["A"][2] * returns["B"][2] + (1 - alpha) * cov
        assert abs(result["A"]["B"] - cov) < 1e-12

    def test_n_less_than_2(self):
        ec = EWMACovariance()
        result = ec.estimate(["A", "B"], {"A": [0.01], "B": [0.02]})
        assert result["A"]["A"] == 0.0

    def test_self_covariance_positive(self):
        ec = EWMACovariance(span=5)
        returns = {
            "A": [0.01, -0.02, 0.03, -0.01, 0.005],
            "B": [0.02, 0.01, -0.01, 0.015, -0.005],
        }
        result = ec.estimate(["A", "B"], returns)
        assert result["A"]["A"] > 0
        assert result["B"]["B"] > 0


# ── ShrinkageCovariance ───────────────────────────────────────────


class TestShrinkageCovariance:
    def test_target_construction(self):
        """Shrinkage target is scaled identity (avg variance on diagonal)."""
        shc = ShrinkageCovariance()
        returns = {
            "A": [0.01, -0.02, 0.03, -0.01, 0.02],
            "B": [0.02, 0.01, -0.01, 0.015, -0.005],
        }
        result = shc.estimate(["A", "B"], returns)
        # Result should be a blend of sample and identity target
        assert result["A"]["A"] > 0
        assert result["B"]["B"] > 0

    def test_blending(self):
        """Result = alpha*target + (1-alpha)*sample."""
        shc = ShrinkageCovariance()
        sc = SampleCovariance()
        returns = {
            "A": [0.01, -0.02, 0.03, -0.01, 0.02,
                  0.005, -0.015, 0.025, -0.008, 0.012],
            "B": [0.02, 0.01, -0.01, 0.015, -0.005,
                  0.008, -0.012, 0.018, -0.003, 0.007],
        }
        sample = sc.estimate(["A", "B"], returns)
        result = shc.estimate(["A", "B"], returns)

        n_obs = 10
        alpha = max(0.0, min(1.0, 1.0 / math.sqrt(n_obs)))
        avg_var = (sample["A"]["A"] + sample["B"]["B"]) / 2

        # Check off-diagonal is blended correctly
        expected_ab = alpha * 0.0 + (1 - alpha) * sample["A"]["B"]
        assert abs(result["A"]["B"] - expected_ab) < 1e-12

        # Check diagonal is blended
        expected_aa = alpha * avg_var + (1 - alpha) * sample["A"]["A"]
        assert abs(result["A"]["A"] - expected_aa) < 1e-12

    def test_shrinkage_coefficient_few_obs(self):
        """With < 4 observations, alpha defaults to 0.5."""
        shc = ShrinkageCovariance()
        returns = {"A": [0.01, -0.02, 0.03], "B": [0.02, 0.01, -0.01]}
        sc = SampleCovariance()
        sample = sc.estimate(["A", "B"], returns)
        result = shc.estimate(["A", "B"], returns)
        avg_var = (sample["A"]["A"] + sample["B"]["B"]) / 2
        expected_ab = 0.5 * 0.0 + 0.5 * sample["A"]["B"]
        assert abs(result["A"]["B"] - expected_ab) < 1e-12

    def test_single_asset(self):
        shc = ShrinkageCovariance()
        returns = {"A": [0.01, -0.02, 0.03]}
        result = shc.estimate(["A"], returns)
        # n < 2 symbols => returns sample directly
        assert result["A"]["A"] > 0


# ── CovarianceCleaning ─────────────────────────────────────────────


class TestCovarianceCleaning:
    def test_min_diagonal(self):
        """Diagonal must be at least min_eigenvalue."""
        cc = CovarianceCleaning(min_eigenvalue=0.001)
        # Returns with near-zero variance for one asset
        returns = {
            "A": [0.0001, 0.0001, 0.0001, 0.0001],
            "B": [0.01, -0.02, 0.03, -0.01],
        }
        result = cc.estimate(["A", "B"], returns)
        assert result["A"]["A"] >= 0.001

    def test_symmetry_restoration(self):
        cc = CovarianceCleaning(min_eigenvalue=1e-10)
        returns = {
            "X": [0.01, -0.02, 0.03, -0.01, 0.02],
            "Y": [0.02, 0.01, -0.01, 0.015, -0.005],
        }
        result = cc.estimate(["X", "Y"], returns)
        assert abs(result["X"]["Y"] - result["Y"]["X"]) < 1e-15

    def test_single_asset(self):
        cc = CovarianceCleaning(min_eigenvalue=1e-8)
        returns = {"A": [0.01, -0.02, 0.03]}
        result = cc.estimate(["A"], returns)
        assert result["A"]["A"] > 0

    def test_preserves_valid_diagonal(self):
        """If diagonal is already above min_eigenvalue, leave it alone."""
        cc = CovarianceCleaning(min_eigenvalue=1e-10)
        returns = {
            "A": [0.01, -0.02, 0.03, -0.01],
            "B": [0.02, 0.01, -0.01, 0.015],
        }
        sc = SampleCovariance()
        sample = sc.estimate(["A", "B"], returns)
        result = cc.estimate(["A", "B"], returns)
        # Large enough variance => diagonal should match sample (both > 1e-10)
        assert abs(result["A"]["A"] - sample["A"]["A"]) < 1e-12
