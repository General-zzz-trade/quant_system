"""Tests for position limits."""
from __future__ import annotations

from portfolio.optimizer.constraints import MaxWeightConstraint


def test_max_weight_feasible():
    c = MaxWeightConstraint(max_weight=0.3)
    weights = {"A": 0.2, "B": 0.3, "C": 0.1}
    assert c.is_feasible(weights)


def test_max_weight_infeasible():
    c = MaxWeightConstraint(max_weight=0.3)
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert not c.is_feasible(weights)


def test_max_weight_project():
    c = MaxWeightConstraint(max_weight=0.3)
    weights = {"A": 0.5, "B": -0.4}
    projected = c.project(weights)
    assert projected["A"] == 0.3
    assert projected["B"] == -0.3
