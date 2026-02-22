# portfolio/optimizer/diagnostics.py
"""Optimization diagnostics — post-optimization quality checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from portfolio.optimizer.base import OptimizationResult
from portfolio.optimizer.constraints import OptConstraint


@dataclass(frozen=True, slots=True)
class DiagnosticItem:
    """单项诊断结果。"""
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """完整诊断报告。"""
    items: tuple[DiagnosticItem, ...]
    all_passed: bool

    @staticmethod
    def build(items: Sequence[DiagnosticItem]) -> DiagnosticReport:
        return DiagnosticReport(
            items=tuple(items),
            all_passed=all(i.passed for i in items),
        )


def check_convergence(result: OptimizationResult) -> DiagnosticItem:
    """检查优化是否收敛。"""
    return DiagnosticItem(
        name="convergence",
        passed=result.converged,
        detail=result.message,
    )


def check_feasibility(
    result: OptimizationResult,
    constraints: Sequence[OptConstraint],
) -> DiagnosticItem:
    """检查结果是否满足所有约束。"""
    violations = []
    for c in constraints:
        if not c.is_feasible(result.weights):
            violations.append(c.name)
    passed = len(violations) == 0
    detail = "" if passed else f"violated: {', '.join(violations)}"
    return DiagnosticItem(name="feasibility", passed=passed, detail=detail)


def check_weight_sum(
    result: OptimizationResult, tolerance: float = 0.01
) -> DiagnosticItem:
    """检查权重之和是否接近 1。"""
    total = sum(result.weights.values())
    passed = abs(total - 1.0) <= tolerance
    return DiagnosticItem(
        name="weight_sum",
        passed=passed,
        detail=f"sum={total:.6f}",
    )


def check_concentration(
    result: OptimizationResult, max_weight: float = 0.5
) -> DiagnosticItem:
    """检查是否过度集中。"""
    if not result.weights:
        return DiagnosticItem(name="concentration", passed=True)
    max_w = max(abs(v) for v in result.weights.values())
    passed = max_w <= max_weight
    return DiagnosticItem(
        name="concentration",
        passed=passed,
        detail=f"max_abs_weight={max_w:.6f}",
    )


def run_diagnostics(
    result: OptimizationResult,
    constraints: Sequence[OptConstraint] = (),
    max_weight: float = 0.5,
    weight_sum_tolerance: float = 0.01,
) -> DiagnosticReport:
    """运行全部诊断。"""
    items = [
        check_convergence(result),
        check_feasibility(result, constraints),
        check_weight_sum(result, weight_sum_tolerance),
        check_concentration(result, max_weight),
    ]
    return DiagnosticReport.build(items)
