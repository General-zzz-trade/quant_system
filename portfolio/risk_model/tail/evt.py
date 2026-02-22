# portfolio/risk_model/tail/evt.py
"""Extreme Value Theory (simplified GPD tail estimation)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class EVTEstimate:
    """EVT 尾部估计结果。"""
    threshold: float
    shape: float        # ξ (shape parameter)
    scale: float        # β (scale parameter)
    n_exceedances: int
    tail_var_99: float


def fit_gpd(
    returns: Sequence[float],
    threshold_pct: float = 10.0,
) -> EVTEstimate:
    """拟合广义帕累托分布到尾部损失。

    使用简化的矩估计法。
    """
    losses = [-r for r in returns if r < 0]
    if len(losses) < 10:
        return EVTEstimate(0.0, 0.0, 0.0, 0, 0.0)

    sorted_losses = sorted(losses)
    threshold_idx = int(len(sorted_losses) * (1 - threshold_pct / 100))
    threshold_idx = max(0, min(threshold_idx, len(sorted_losses) - 2))
    threshold = sorted_losses[threshold_idx]

    exceedances = [l - threshold for l in sorted_losses if l > threshold]
    n_exc = len(exceedances)
    if n_exc < 3:
        return EVTEstimate(threshold, 0.0, 0.0, n_exc, 0.0)

    # 矩估计
    mean_exc = sum(exceedances) / n_exc
    var_exc = sum((e - mean_exc) ** 2 for e in exceedances) / (n_exc - 1)

    if mean_exc <= 0:
        return EVTEstimate(threshold, 0.0, 0.0, n_exc, 0.0)

    # ξ = 0.5 * (mean²/var - 1),  β = mean * (1 + ξ) / 2
    ratio = mean_exc ** 2 / max(var_exc, 1e-12)
    xi = 0.5 * (ratio - 1)
    beta = mean_exc * (1 + xi) / 2

    # GPD 99% VaR
    n = len(returns)
    p = 0.01
    if n_exc > 0 and xi != 0 and beta > 0:
        prob_exceed = n_exc / n
        tail_var = threshold + beta / xi * ((p / prob_exceed) ** (-xi) - 1)
    else:
        tail_var = threshold + mean_exc * 2.326

    return EVTEstimate(
        threshold=threshold,
        shape=xi,
        scale=beta,
        n_exceedances=n_exc,
        tail_var_99=-tail_var,  # 转回收益率符号
    )
