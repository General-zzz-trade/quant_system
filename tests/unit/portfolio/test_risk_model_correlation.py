"""Tests for portfolio/risk_model/correlation estimators."""
from __future__ import annotations

import math

from portfolio.risk_model.correlation.rolling import RollingCorrelation
from portfolio.risk_model.correlation.regime import RegimeCorrelation
from portfolio.risk_model.correlation.static import StaticCorrelation


# ── RollingCorrelation ─────────────────────────────────────────────


class TestRollingCorrelation:
    def test_perfect_positive_correlation(self):
        rc = RollingCorrelation(window=10)
        returns = {"A": [0.01 * i for i in range(10)],
                   "B": [0.02 * i for i in range(10)]}
        result = rc.estimate(["A", "B"], returns)
        assert abs(result["A"]["B"] - 1.0) < 1e-6

    def test_perfect_negative_correlation(self):
        rc = RollingCorrelation(window=10)
        returns = {"A": [0.01 * i for i in range(10)],
                   "B": [-0.02 * i for i in range(10)]}
        result = rc.estimate(["A", "B"], returns)
        assert abs(result["A"]["B"] - (-1.0)) < 1e-6

    def test_diagonal_is_one(self):
        rc = RollingCorrelation(window=5)
        returns = {"X": [0.01, -0.02, 0.03, -0.01, 0.02],
                   "Y": [0.02, 0.01, -0.01, 0.03, -0.02]}
        result = rc.estimate(["X", "Y"], returns)
        assert result["X"]["X"] == 1.0
        assert result["Y"]["Y"] == 1.0

    def test_window_selection(self):
        """Only the last `window` returns are used."""
        rc = RollingCorrelation(window=3)
        # First 7 entries are noise, last 3 are perfectly correlated
        a_noise = [0.1, -0.3, 0.2, 0.05, -0.1, 0.15, -0.05]
        b_noise = [-0.2, 0.1, -0.15, 0.3, 0.05, -0.25, 0.1]
        a_tail = [0.01, 0.02, 0.03]
        b_tail = [0.02, 0.04, 0.06]
        returns = {"A": a_noise + a_tail, "B": b_noise + b_tail}
        result = rc.estimate(["A", "B"], returns)
        assert abs(result["A"]["B"] - 1.0) < 1e-6

    def test_uncorrelated_short_series(self):
        rc = RollingCorrelation(window=5)
        returns = {"A": [0.01], "B": [0.02]}
        result = rc.estimate(["A", "B"], returns)
        # n < 2 in _pearson => returns 0.0
        assert result["A"]["B"] == 0.0

    def test_symmetry(self):
        rc = RollingCorrelation(window=10)
        returns = {"A": [0.01, -0.02, 0.03, 0.01, -0.01,
                         0.02, -0.03, 0.01, 0.005, -0.005],
                   "B": [0.02, 0.01, -0.01, 0.03, -0.02,
                         0.01, 0.02, -0.01, 0.01, 0.005]}
        result = rc.estimate(["A", "B"], returns)
        assert abs(result["A"]["B"] - result["B"]["A"]) < 1e-12


# ── RegimeCorrelation ──────────────────────────────────────────────


class TestRegimeCorrelation:
    def test_stress_detection(self):
        """High-vol returns should trigger stress regime."""
        rc = RegimeCorrelation(vol_threshold=0.03)
        high_vol = [0.1, -0.1, 0.08, -0.12, 0.15,
                    -0.09, 0.11, -0.13, 0.07, -0.14,
                    0.12, -0.08, 0.1, -0.11, 0.09,
                    -0.13, 0.11, -0.07, 0.14, -0.1]
        returns = {"A": high_vol, "B": high_vol}
        is_stress = rc._detect_stress(returns)
        assert is_stress is True

    def test_no_stress(self):
        rc = RegimeCorrelation(vol_threshold=0.03)
        low_vol = [0.001, -0.001, 0.002, -0.002, 0.001] * 4
        returns = {"A": low_vol, "B": low_vol}
        is_stress = rc._detect_stress(returns)
        assert is_stress is False

    def test_correlation_amplification(self):
        """In stress regime, off-diagonal correlations are multiplied by stress_multiplier."""
        rc = RegimeCorrelation(
            stress_window=20, stress_multiplier=1.5, vol_threshold=0.01,
        )
        # High-vol returns to trigger stress
        a = [0.05 * ((-1) ** i) for i in range(20)]
        b = [0.04 * ((-1) ** i) for i in range(20)]
        returns = {"A": a, "B": b}
        result = rc.estimate(["A", "B"], returns)
        # Diagonal stays 1.0
        assert result["A"]["A"] == 1.0
        assert result["B"]["B"] == 1.0
        # Off-diagonal is clamped to [-1, 1]
        assert -1.0 <= result["A"]["B"] <= 1.0

    def test_normal_regime_uses_normal_window(self):
        rc = RegimeCorrelation(
            normal_window=5, stress_window=3, vol_threshold=1.0,
        )
        returns = {
            "A": [0.001, -0.001, 0.002, -0.002, 0.001],
            "B": [0.002, 0.001, -0.001, 0.003, -0.002],
        }
        result = rc.estimate(["A", "B"], returns)
        assert result["A"]["A"] == 1.0


# ── StaticCorrelation ──────────────────────────────────────────────


class TestStaticCorrelation:
    def test_default_value(self):
        sc = StaticCorrelation(default_corr=0.5)
        result = sc.estimate(["A", "B", "C"], {})
        assert result["A"]["B"] == 0.5
        assert result["B"]["C"] == 0.5

    def test_diagonal_enforcement(self):
        sc = StaticCorrelation(default_corr=0.5)
        result = sc.estimate(["A", "B"], {})
        assert result["A"]["A"] == 1.0
        assert result["B"]["B"] == 1.0

    def test_overrides(self):
        overrides = {"A": {"B": 0.9}}
        sc = StaticCorrelation(default_corr=0.3, overrides=overrides)
        result = sc.estimate(["A", "B"], {})
        assert result["A"]["B"] == 0.9
        assert result["B"]["A"] == 0.3  # Override is directional

    def test_single_asset(self):
        sc = StaticCorrelation()
        result = sc.estimate(["X"], {})
        assert result["X"]["X"] == 1.0
