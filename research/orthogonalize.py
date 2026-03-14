"""OLS-based factor orthogonalization for marginal IC computation.

Provides proper OLS residualization (Gram-Schmidt-like) to compute
the true marginal contribution of each factor beyond what others explain.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from research.alpha_factor import _pearson_corr


def ols_residualize(
    target: List[float],
    regressors: List[List[float]],
) -> List[float]:
    """Compute OLS residuals: target - X @ (X'X)^-1 X'target.

    Uses Gaussian elimination (no external dependencies).

    Parameters
    ----------
    target : 1D array of floats
    regressors : list of 1D arrays (each a regressor column)

    Returns
    -------
    Residual vector (target minus fitted values).
    """
    n = len(target)
    k = len(regressors)

    if k == 0 or n == 0:
        return list(target)

    # Verify dimensions
    for r in regressors:
        if len(r) != n:
            raise ValueError("All regressors must have same length as target")

    # Build X'X and X'y (demeaned for numerical stability)
    x_means = [sum(regressors[j][i] for i in range(n)) / n for j in range(k)]
    y_mean = sum(target) / n

    XtX = [[0.0] * k for _ in range(k)]
    Xty = [0.0] * k

    for i in range(n):
        for j in range(k):
            xj = regressors[j][i] - x_means[j]
            Xty[j] += xj * (target[i] - y_mean)
            for m in range(j, k):
                val = xj * (regressors[m][i] - x_means[m])
                XtX[j][m] += val
                if m != j:
                    XtX[m][j] += val

    # Gaussian elimination with partial pivoting
    beta = _solve_linear(XtX, Xty, k)

    # Compute residuals: y - (y_mean + sum(beta_j * (x_j - x_mean_j)))
    bias = y_mean - sum(beta[j] * x_means[j] for j in range(k))
    residuals = [0.0] * n
    for i in range(n):
        fitted = bias + sum(beta[j] * regressors[j][i] for j in range(k))
        residuals[i] = target[i] - fitted

    return residuals


def marginal_ic_ols(
    factor_values: Mapping[str, List[Optional[float]]],
    fwd_returns: List[Optional[float]],
) -> Dict[str, float]:
    """Compute marginal IC for each factor via OLS residualization.

    For each factor, residualize it against all other factors,
    then compute Pearson IC of the residual against forward returns.

    Parameters
    ----------
    factor_values : dict mapping factor name → list of optional float values
    fwd_returns : forward return series (same length)

    Returns
    -------
    Dict mapping factor name → marginal IC.
    """
    names = list(factor_values.keys())
    n = len(fwd_returns)
    marginal: Dict[str, float] = {}

    for i, name in enumerate(names):
        other_names = [n for j, n in enumerate(names) if j != i]

        if not other_names:
            # Single factor: marginal IC = raw IC
            xs, ys = _paired_clean(factor_values[name], fwd_returns)
            marginal[name] = _pearson_corr(xs, ys) if len(xs) >= 20 else 0.0
            continue

        # Build paired-clean arrays: own, others, returns
        own_vals = factor_values[name]
        other_vals_list = [factor_values[on] for on in other_names]

        valid_own: List[float] = []
        valid_rets: List[float] = []
        valid_others: List[List[float]] = [[] for _ in other_names]

        for idx in range(min(n, len(own_vals))):
            ov = own_vals[idx]
            rv = fwd_returns[idx]
            if ov is None or rv is None:
                continue
            all_ok = True
            for ol in other_vals_list:
                if idx >= len(ol) or ol[idx] is None:
                    all_ok = False
                    break
            if not all_ok:
                continue

            valid_own.append(ov)
            valid_rets.append(rv)
            for j, ol in enumerate(other_vals_list):
                valid_others[j].append(ol[idx])

        if len(valid_own) < 20:
            marginal[name] = 0.0
            continue

        # Residualize factor against others
        residuals = ols_residualize(valid_own, valid_others)
        marginal[name] = _pearson_corr(residuals, valid_rets)

    return marginal


def _paired_clean(
    a: List[Optional[float]], b: List[Optional[float]],
) -> tuple[List[float], List[float]]:
    """Extract paired non-None values."""
    xs: List[float] = []
    ys: List[float] = []
    for va, vb in zip(a, b):
        if va is not None and vb is not None:
            xs.append(va)
            ys.append(vb)
    return xs, ys


def _solve_linear(A: list[list[float]], b: list[float], k: int) -> list[float]:
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    aug = [A[i][:] + [b[i]] for i in range(k)]

    for col in range(k):
        max_row = col
        for row in range(col + 1, k):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            continue

        for row in range(col + 1, k):
            factor = aug[row][col] / pivot
            for j in range(col, k + 1):
                aug[row][j] -= factor * aug[col][j]

    x = [0.0] * k
    for i in range(k - 1, -1, -1):
        if abs(aug[i][i]) < 1e-12:
            continue
        x[i] = aug[i][k]
        for j in range(i + 1, k):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]

    return x
