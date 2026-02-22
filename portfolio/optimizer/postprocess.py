# portfolio/optimizer/postprocess.py
"""Post-optimization weight processing."""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from portfolio.optimizer.base import OptimizationResult


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """归一化权重使其和为 1。"""
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def clip_small_weights(
    weights: dict[str, float], threshold: float = 1e-4
) -> dict[str, float]:
    """将过小权重置零。"""
    return {k: (v if abs(v) >= threshold else 0.0) for k, v in weights.items()}


def round_weights(
    weights: dict[str, float], precision: int = 6
) -> dict[str, float]:
    """四舍五入权重到指定精度。"""
    return {k: round(v, precision) for k, v in weights.items()}


def remove_zero_weights(weights: dict[str, float]) -> dict[str, float]:
    """移除零权重品种。"""
    return {k: v for k, v in weights.items() if v != 0.0}


def postprocess(
    result: OptimizationResult,
    *,
    clip_threshold: float = 1e-4,
    precision: int = 6,
    do_normalize: bool = True,
    do_remove_zeros: bool = True,
) -> OptimizationResult:
    """后处理优化结果: clip → normalize → round → remove zeros。"""
    w = dict(result.weights)
    w = clip_small_weights(w, clip_threshold)
    if do_normalize:
        w = normalize_weights(w)
    w = round_weights(w, precision)
    if do_remove_zeros:
        w = remove_zero_weights(w)
    return OptimizationResult(
        weights=w,
        objective_value=result.objective_value,
        converged=result.converged,
        iterations=result.iterations,
        message=result.message,
        diagnostics=result.diagnostics,
    )
