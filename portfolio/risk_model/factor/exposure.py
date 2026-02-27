# portfolio/risk_model/factor/exposure.py
"""Factor exposure (beta) computation."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

try:
    from features._quant_rolling import cpp_compute_exposures as _cpp_compute_exposures
    _USING_CPP = True
except ImportError:
    _USING_CPP = False


def compute_beta(
    asset_returns: Sequence[float],
    factor_returns: Sequence[float],
) -> float:
    """计算资产对因子的 beta 暴露。"""
    n = min(len(asset_returns), len(factor_returns))
    if n < 2:
        return 0.0
    mx = sum(factor_returns[:n]) / n
    my = sum(asset_returns[:n]) / n
    cov = sum(
        (factor_returns[i] - mx) * (asset_returns[i] - my) for i in range(n)
    ) / (n - 1)
    var_x = sum((factor_returns[i] - mx) ** 2 for i in range(n)) / (n - 1)
    if var_x < 1e-12:
        return 0.0
    return cov / var_x


def compute_exposures(
    symbols: Sequence[str],
    returns: Mapping[str, Sequence[float]],
    factor_returns: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, float]]:
    """批量计算所有资产对所有因子的暴露。"""
    if _USING_CPP and symbols and factor_returns:
        factor_names = list(factor_returns.keys())
        a_mat = [list(returns.get(s, ())) for s in symbols]
        f_mat = [list(factor_returns[f]) for f in factor_names]
        exp_mat = _cpp_compute_exposures(a_mat, f_mat)
        return {
            s: {f: exp_mat[i][j] for j, f in enumerate(factor_names)}
            for i, s in enumerate(symbols)
        }

    result: dict[str, dict[str, float]] = {}
    for s in symbols:
        asset_ret = returns.get(s, ())
        row: dict[str, float] = {}
        for f_name, f_ret in factor_returns.items():
            row[f_name] = compute_beta(asset_ret, f_ret)
        result[s] = row
    return result
