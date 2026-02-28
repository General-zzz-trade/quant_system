"""Tests for Kelly Criterion allocator."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from portfolio.allocator import PortfolioConstraints
from portfolio.allocator_kelly import KellyAllocator, _invert_matrix


# ── Test helpers ──────────────────────────────────────────────

class _MockAccount:
    equity = Decimal("100000")
    positions_qty = {}


class _MockPrices:
    _px = {"BTC": Decimal("40000"), "ETH": Decimal("2500"), "SOL": Decimal("100")}

    def price(self, symbol: str) -> Decimal:
        return self._px.get(symbol, Decimal("100"))


_CONSTRAINTS = PortfolioConstraints()
_ACCOUNT = _MockAccount()
_PRICES = _MockPrices()


# Simple 2-asset scenario
_COV_2 = {
    "BTC": {"BTC": 0.04, "ETH": 0.01},
    "ETH": {"BTC": 0.01, "ETH": 0.09},
}
_MU_2 = {"BTC": 0.05, "ETH": 0.08}


# 3-asset scenario (diagonal for simplicity)
_COV_3 = {
    "BTC": {"BTC": 0.04, "ETH": 0.0, "SOL": 0.0},
    "ETH": {"BTC": 0.0, "ETH": 0.09, "SOL": 0.0},
    "SOL": {"BTC": 0.0, "ETH": 0.0, "SOL": 0.16},
}
_MU_3 = {"BTC": 0.05, "ETH": 0.08, "SOL": 0.12}


# ── Matrix inversion tests ───────────────────────────────────

class TestMatrixInversion:
    def test_identity_inverse(self):
        cov = {"A": {"A": 1.0, "B": 0.0}, "B": {"A": 0.0, "B": 1.0}}
        inv = _invert_matrix(cov, ["A", "B"])
        assert abs(inv["A"]["A"] - 1.0) < 1e-10
        assert abs(inv["A"]["B"]) < 1e-10
        assert abs(inv["B"]["B"] - 1.0) < 1e-10

    def test_diagonal_inverse(self):
        cov = {"A": {"A": 4.0, "B": 0.0}, "B": {"A": 0.0, "B": 9.0}}
        inv = _invert_matrix(cov, ["A", "B"])
        assert abs(inv["A"]["A"] - 0.25) < 1e-10
        assert abs(inv["B"]["B"] - 1.0 / 9.0) < 1e-10

    def test_2x2_with_correlation(self):
        inv = _invert_matrix(_COV_2, ["BTC", "ETH"])
        # Verify: cov @ inv ≈ I
        for si in ["BTC", "ETH"]:
            for sj in ["BTC", "ETH"]:
                val = sum(_COV_2[si][sk] * inv[sk][sj] for sk in ["BTC", "ETH"])
                expected = 1.0 if si == sj else 0.0
                assert abs(val - expected) < 1e-10

    def test_3x3_inversion(self):
        inv = _invert_matrix(_COV_3, ["BTC", "ETH", "SOL"])
        # Diagonal: inv should be 1/cov[i][i]
        assert abs(inv["BTC"]["BTC"] - 25.0) < 1e-10  # 1/0.04
        assert abs(inv["ETH"]["ETH"] - 1.0 / 0.09) < 1e-10
        assert abs(inv["SOL"]["SOL"] - 1.0 / 0.16) < 1e-10

    def test_singular_matrix_raises(self):
        cov = {"A": {"A": 1.0, "B": 1.0}, "B": {"A": 1.0, "B": 1.0}}
        with pytest.raises(Exception):
            _invert_matrix(cov, ["A", "B"])


# ── KellyAllocator core tests ────────────────────────────────

class TestKellyAllocator:
    def test_full_kelly_diagonal(self):
        """With diagonal covariance, full Kelly weight = mu_i / var_i."""
        alloc = KellyAllocator(default_kelly_fraction=1.0)
        weights = alloc.compute_kelly_weights(
            ["BTC", "ETH", "SOL"], _MU_3, _COV_3, kelly_fraction=1.0,
        )
        # BTC: 0.05 / 0.04 = 1.25
        assert abs(weights["BTC"] - 1.25) < 1e-8
        # ETH: 0.08 / 0.09 ≈ 0.8889
        assert abs(weights["ETH"] - 0.08 / 0.09) < 1e-8
        # SOL: 0.12 / 0.16 = 0.75
        assert abs(weights["SOL"] - 0.75) < 1e-8

    def test_half_kelly_halves_weights(self):
        full = KellyAllocator().compute_kelly_weights(
            ["BTC", "ETH"], _MU_2, _COV_2, kelly_fraction=1.0,
        )
        half = KellyAllocator().compute_kelly_weights(
            ["BTC", "ETH"], _MU_2, _COV_2, kelly_fraction=0.5,
        )
        for s in ["BTC", "ETH"]:
            assert abs(half[s] - full[s] * 0.5) < 1e-10

    def test_quarter_kelly(self):
        full = KellyAllocator().compute_kelly_weights(
            ["BTC", "ETH"], _MU_2, _COV_2, kelly_fraction=1.0,
        )
        quarter = KellyAllocator().compute_kelly_weights(
            ["BTC", "ETH"], _MU_2, _COV_2, kelly_fraction=0.25,
        )
        for s in ["BTC", "ETH"]:
            assert abs(quarter[s] - full[s] * 0.25) < 1e-10

    def test_allocate_produces_plan(self):
        alloc = KellyAllocator()
        plan = alloc.allocate(
            ts="2024-01-01",
            symbols=["BTC", "ETH"],
            account=_ACCOUNT,
            prices=_PRICES,
            constraints=_CONSTRAINTS,
            inputs={
                "expected_returns": _MU_2,
                "covariance": _COV_2,
                "kelly_fraction": 0.5,
            },
        )
        assert len(plan.targets) == 2
        assert plan.diagnostics.equity == Decimal("100000")
        # Kelly tag should be present
        assert "kelly_allocator" in plan.tags

    def test_allocate_respects_concentration(self):
        alloc = KellyAllocator()
        plan = alloc.allocate(
            ts="2024-01-01",
            symbols=["BTC", "ETH", "SOL"],
            account=_ACCOUNT,
            prices=_PRICES,
            constraints=_CONSTRAINTS,
            inputs={
                "expected_returns": _MU_3,
                "covariance": _COV_3,
                "kelly_fraction": 1.0,
                "max_concentration": 0.3,
            },
        )
        for t in plan.targets:
            assert abs(t.target_weight) <= Decimal("0.31")  # tolerance for rounding

    def test_allocate_with_gross_leverage_cap(self):
        alloc = KellyAllocator()
        capped = PortfolioConstraints(max_gross_leverage=Decimal("1.0"))
        plan = alloc.allocate(
            ts="2024-01-01",
            symbols=["BTC", "ETH", "SOL"],
            account=_ACCOUNT,
            prices=_PRICES,
            constraints=capped,
            inputs={
                "expected_returns": _MU_3,
                "covariance": _COV_3,
                "kelly_fraction": 1.0,
            },
        )
        assert plan.diagnostics.leverage_after <= Decimal("1.01")

    def test_negative_expected_return(self):
        """Negative expected return should produce negative (short) weight."""
        alloc = KellyAllocator()
        weights = alloc.compute_kelly_weights(
            ["BTC"], {"BTC": -0.05}, {"BTC": {"BTC": 0.04}}, kelly_fraction=1.0,
        )
        assert weights["BTC"] < 0

    def test_zero_expected_return(self):
        """Zero expected return should produce zero weight."""
        weights = KellyAllocator().compute_kelly_weights(
            ["BTC"], {"BTC": 0.0}, {"BTC": {"BTC": 0.04}}, kelly_fraction=1.0,
        )
        assert abs(weights["BTC"]) < 1e-10

    def test_correlated_assets_adjust_weights(self):
        """With positive correlation, weights should be different from diagonal case."""
        diag_w = KellyAllocator().compute_kelly_weights(
            ["BTC", "ETH"], _MU_2,
            {"BTC": {"BTC": 0.04, "ETH": 0.0}, "ETH": {"BTC": 0.0, "ETH": 0.09}},
            kelly_fraction=1.0,
        )
        corr_w = KellyAllocator().compute_kelly_weights(
            ["BTC", "ETH"], _MU_2, _COV_2, kelly_fraction=1.0,
        )
        # With positive correlation, weights shift
        assert diag_w["BTC"] != pytest.approx(corr_w["BTC"], abs=1e-6)

    def test_missing_inputs_raises(self):
        alloc = KellyAllocator()
        with pytest.raises(Exception):
            alloc.allocate(
                ts="2024-01-01", symbols=["BTC"], account=_ACCOUNT,
                prices=_PRICES, constraints=_CONSTRAINTS, inputs=None,
            )

    def test_invalid_kelly_fraction_raises(self):
        alloc = KellyAllocator()
        with pytest.raises(Exception):
            alloc.allocate(
                ts="2024-01-01", symbols=["BTC"], account=_ACCOUNT,
                prices=_PRICES, constraints=_CONSTRAINTS,
                inputs={
                    "expected_returns": {"BTC": 0.05},
                    "covariance": {"BTC": {"BTC": 0.04}},
                    "kelly_fraction": 0.0,
                },
            )

    def test_empty_symbols_raises(self):
        alloc = KellyAllocator()
        with pytest.raises(Exception):
            alloc.allocate(
                ts="2024-01-01", symbols=[], account=_ACCOUNT,
                prices=_PRICES, constraints=_CONSTRAINTS,
                inputs={
                    "expected_returns": {},
                    "covariance": {},
                },
            )
