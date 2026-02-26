# tests/unit/portfolio/test_factor_constraints.py
"""Tests for factor exposure constraints."""
from __future__ import annotations

import pytest

from portfolio.optimizer.factor_constraints import (
    FactorExposureConstraint,
    SectorExposureConstraint,
)


# ── FactorExposureConstraint ─────────────────────────────────────────

class TestFactorExposureConstraint:
    def test_feasible_within_bounds(self):
        """Weights that produce exposure within bounds are feasible."""
        c = FactorExposureConstraint(
            factor_loadings={"A": 1.2, "B": 0.8, "C": -0.5},
            max_exposure=0.5,
            min_exposure=-0.5,
        )
        # exposure = 0.3*1.2 + 0.3*0.8 + 0.4*(-0.5) = 0.36 + 0.24 - 0.20 = 0.40
        weights = {"A": 0.3, "B": 0.3, "C": 0.4}
        assert c.is_feasible(weights) is True

    def test_infeasible_above_max(self):
        """Weights that exceed max exposure are infeasible."""
        c = FactorExposureConstraint(
            factor_loadings={"A": 1.5, "B": 1.0},
            max_exposure=0.5,
            min_exposure=-0.5,
        )
        # exposure = 0.5*1.5 + 0.5*1.0 = 1.25 > 0.5
        weights = {"A": 0.5, "B": 0.5}
        assert c.is_feasible(weights) is False

    def test_infeasible_below_min(self):
        """Weights that go below min exposure are infeasible."""
        c = FactorExposureConstraint(
            factor_loadings={"A": -2.0, "B": -1.0},
            max_exposure=0.1,
            min_exposure=-0.5,
        )
        # exposure = 0.5*(-2.0) + 0.5*(-1.0) = -1.5 < -0.5
        weights = {"A": 0.5, "B": 0.5}
        assert c.is_feasible(weights) is False

    def test_exposure_computation(self):
        """Verify exposure calculation."""
        c = FactorExposureConstraint(
            factor_loadings={"A": 1.0, "B": -1.0},
        )
        assert c.exposure({"A": 0.6, "B": 0.4}) == pytest.approx(0.2)

    def test_project_brings_within_bounds(self):
        """Projection should bring exposure within [min, max]."""
        c = FactorExposureConstraint(
            factor_loadings={"A": 1.5, "B": 0.5},
            max_exposure=0.5,
            min_exposure=-0.5,
        )
        weights = {"A": 0.6, "B": 0.4}
        # exposure = 0.6*1.5 + 0.4*0.5 = 1.1 > 0.5

        projected = c.project(weights)
        assert c.is_feasible(projected)

    def test_project_no_change_when_feasible(self):
        """Projection of feasible weights returns same weights."""
        c = FactorExposureConstraint(
            factor_loadings={"A": 0.5, "B": -0.5},
            max_exposure=1.0,
            min_exposure=-1.0,
        )
        weights = {"A": 0.5, "B": 0.5}
        projected = c.project(weights)
        for s in weights:
            assert projected[s] == pytest.approx(weights[s])

    def test_beta_neutral_constraint(self):
        """Market-neutral: beta exposure close to zero."""
        c = FactorExposureConstraint(
            factor_loadings={"A": 1.2, "B": 0.8, "C": 0.5},
            max_exposure=0.05,
            min_exposure=-0.05,
        )
        # High beta exposure
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}
        assert c.is_feasible(weights) is False

        projected = c.project(weights)
        assert c.is_feasible(projected)


# ── SectorExposureConstraint ─────────────────────────────────────────

class TestSectorExposureConstraint:
    def test_feasible(self):
        c = SectorExposureConstraint(
            sector_map={"A": "tech", "B": "fin", "C": "tech"},
            max_sector_weight=0.6,
        )
        weights = {"A": 0.3, "B": 0.4, "C": 0.3}
        # tech = 0.6, fin = 0.4 -> tech at boundary
        assert c.is_feasible(weights) is True

    def test_infeasible(self):
        c = SectorExposureConstraint(
            sector_map={"A": "tech", "B": "fin", "C": "tech"},
            max_sector_weight=0.5,
        )
        weights = {"A": 0.4, "B": 0.2, "C": 0.4}
        # tech = 0.8 > 0.5
        assert c.is_feasible(weights) is False

    def test_project_scales_sector(self):
        c = SectorExposureConstraint(
            sector_map={"A": "tech", "B": "fin", "C": "tech"},
            max_sector_weight=0.5,
        )
        weights = {"A": 0.4, "B": 0.2, "C": 0.4}
        projected = c.project(weights)
        sector_w = c.sector_weights(projected)
        assert sector_w["tech"] <= 0.5 + 1e-10
