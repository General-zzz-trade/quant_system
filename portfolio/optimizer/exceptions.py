# portfolio/optimizer/exceptions.py
"""Optimization exceptions."""
from __future__ import annotations


class OptimizationError(RuntimeError):
    """优化过程通用错误。"""


class InfeasibleError(OptimizationError):
    """约束不可行 — 无法找到满足所有约束的解。"""


class NumericalError(OptimizationError):
    """数值计算错误。"""


class ConvergenceError(OptimizationError):
    """迭代未收敛。"""
