# portfolio/optimizer/black_litterman.py
"""Black-Litterman model: market equilibrium + investor views -> posterior."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

try:
    from features._quant_rolling import cpp_black_litterman_posterior as _cpp_bl_posterior
    _USING_CPP = True
except ImportError:
    _USING_CPP = False

logger = logging.getLogger(__name__)


# ── Pure-Python matrix helpers (for small N, typically <20 assets) ───


def _mat_zeros(n: int, m: int) -> list[list[float]]:
    return [[0.0] * m for _ in range(n)]


def _mat_identity(n: int) -> list[list[float]]:
    mat = _mat_zeros(n, n)
    for i in range(n):
        mat[i][i] = 1.0
    return mat


def _mat_transpose(a: list[list[float]]) -> list[list[float]]:
    if not a:
        return []
    rows, cols = len(a), len(a[0])
    result = _mat_zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            result[j][i] = a[i][j]
    return result


def _mat_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = _mat_zeros(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def _mat_scale(a: list[list[float]], scalar: float) -> list[list[float]]:
    return [[v * scalar for v in row] for row in a]


def _mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows, cols = len(a), len(a[0])
    return [[a[i][j] + b[i][j] for j in range(cols)] for i in range(rows)]


def _mat_inverse(mat: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan elimination for matrix inverse. Works for small N."""
    n = len(mat)
    # Augmented matrix [A | I]
    aug = _mat_zeros(n, 2 * n)
    for i in range(n):
        for j in range(n):
            aug[i][j] = mat[i][j]
        aug[i][n + i] = 1.0

    for col in range(n):
        # Partial pivoting
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-15:
            raise ValueError("Singular matrix, cannot invert")
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    # Extract inverse from right half
    inv = _mat_zeros(n, n)
    for i in range(n):
        for j in range(n):
            inv[i][j] = aug[i][n + j]
    return inv


def _mat_vec_multiply(mat: list[list[float]], vec: list[float]) -> list[float]:
    """Matrix-vector multiply: mat @ vec."""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def _vec_to_col(vec: list[float]) -> list[list[float]]:
    """Convert vector to column matrix."""
    return [[v] for v in vec]


def _col_to_vec(col: list[list[float]]) -> list[float]:
    """Convert column matrix to vector."""
    return [row[0] for row in col]


# ── Data classes ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BlackLittermanConfig:
    """Configuration for Black-Litterman model."""
    tau: float = 0.05       # Uncertainty scaling for prior covariance
    risk_aversion: float = 2.5  # Market risk aversion parameter (delta)


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """A single investor view: P @ mu = Q with uncertainty omega.

    For an absolute view on asset X: assets=("X",), weights=(1.0,), expected_return=0.05
    For a relative view (X outperforms Y by 2%):
        assets=("X","Y"), weights=(1.0,-1.0), expected_return=0.02
    """
    assets: tuple[str, ...]
    weights: tuple[float, ...]
    expected_return: float
    confidence: float = 1.0  # Higher = more confident (used to compute omega)


@dataclass(frozen=True, slots=True)
class BlackLittermanResult:
    """Result of Black-Litterman posterior computation."""
    posterior_returns: dict[str, float]
    posterior_covariance: dict[str, dict[str, float]]
    equilibrium_returns: dict[str, float]


# ── Model ────────────────────────────────────────────────────────────


class BlackLittermanModel:
    """Black-Litterman model: market equilibrium + investor views -> posterior.

    Computes posterior expected returns by combining market equilibrium
    (implied from market cap weights) with investor views.
    """

    def __init__(self, config: BlackLittermanConfig | None = None) -> None:
        self._config = config or BlackLittermanConfig()

    @property
    def config(self) -> BlackLittermanConfig:
        return self._config

    def equilibrium_returns(
        self,
        market_weights: Mapping[str, float],
        cov: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        """Implied equilibrium returns: pi = delta * Sigma * w_mkt."""
        delta = self._config.risk_aversion
        symbols = list(market_weights.keys())
        n = len(symbols)

        # Build Sigma matrix and w vector
        sigma = _mat_zeros(n, n)
        w = [0.0] * n
        for i, si in enumerate(symbols):
            w[i] = market_weights[si]
            row = cov.get(si, {})
            for j, sj in enumerate(symbols):
                sigma[i][j] = row.get(sj, 0.0)

        # pi = delta * Sigma @ w
        sigma_w = _mat_vec_multiply(sigma, w)
        pi = {symbols[i]: delta * sigma_w[i] for i in range(n)}
        return pi

    def posterior(
        self,
        market_weights: Mapping[str, float],
        cov: Mapping[str, Mapping[str, float]],
        views: Sequence[ViewSpec],
    ) -> BlackLittermanResult:
        """Compute posterior returns incorporating investor views.

        Formula:
            mu_post = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
                    * [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]
        """
        symbols = list(market_weights.keys())
        n = len(symbols)
        sym_idx = {s: i for i, s in enumerate(symbols)}
        tau = self._config.tau
        delta = self._config.risk_aversion

        if _USING_CPP and views:
            # Build matrices for C++ dispatch
            sigma_mat = [[cov.get(si, {}).get(sj, 0.0) for sj in symbols] for si in symbols]
            w_vec = [market_weights[s] for s in symbols]
            k = len(views)
            P_mat = [[0.0] * n for _ in range(k)]
            Q_vec = [0.0] * k
            conf_vec = [0.0] * k
            for v_idx, view in enumerate(views):
                Q_vec[v_idx] = view.expected_return
                conf_vec[v_idx] = view.confidence
                for asset, weight in zip(view.assets, view.weights):
                    if asset in sym_idx:
                        P_mat[v_idx][sym_idx[asset]] = weight

            mu_post, post_cov, eq_pi = _cpp_bl_posterior(
                sigma_mat, w_vec, P_mat, Q_vec, conf_vec, tau, delta)

            return BlackLittermanResult(
                posterior_returns={symbols[i]: mu_post[i] for i in range(n)},
                posterior_covariance={
                    symbols[i]: {symbols[j]: post_cov[i][j] for j in range(n)}
                    for i in range(n)
                },
                equilibrium_returns={symbols[i]: eq_pi[i] for i in range(n)},
            )

        # Equilibrium returns
        eq_ret = self.equilibrium_returns(market_weights, cov)
        pi = [eq_ret[s] for s in symbols]

        # Build Sigma matrix
        sigma = _mat_zeros(n, n)
        for i, si in enumerate(symbols):
            row = cov.get(si, {})
            for j, sj in enumerate(symbols):
                sigma[i][j] = row.get(sj, 0.0)

        # tau * Sigma
        tau_sigma = _mat_scale(sigma, tau)

        if not views:
            # No views: posterior = equilibrium
            return BlackLittermanResult(
                posterior_returns=dict(eq_ret),
                posterior_covariance=_matrix_to_dict(sigma, symbols),
                equilibrium_returns=dict(eq_ret),
            )

        k = len(views)  # number of views

        # Build P matrix (k x n): each row is a view's asset weights
        P = _mat_zeros(k, n)
        Q = [0.0] * k
        for v_idx, view in enumerate(views):
            Q[v_idx] = view.expected_return
            for asset, weight in zip(view.assets, view.weights):
                if asset in sym_idx:
                    P[v_idx][sym_idx[asset]] = weight

        # Omega (k x k): diagonal uncertainty matrix
        # omega_ii = (1 / confidence_i) * P_i' * tau * Sigma * P_i
        omega = _mat_zeros(k, k)
        for v_idx in range(k):
            p_row = P[v_idx]
            # p' * tau_sigma * p (scalar)
            tau_sigma_p = _mat_vec_multiply(tau_sigma, p_row)
            view_var = sum(p_row[j] * tau_sigma_p[j] for j in range(n))
            confidence = views[v_idx].confidence
            if confidence <= 0:
                confidence = 1e-6
            omega[v_idx][v_idx] = view_var / confidence

        # Compute posterior:
        # tau_sigma_inv = (tau * Sigma)^-1
        tau_sigma_inv = _mat_inverse(tau_sigma)

        # omega_inv = Omega^-1
        omega_inv = _mat_inverse(omega)

        # P' * Omega^-1 * P
        Pt = _mat_transpose(P)
        Pt_omega_inv = _mat_multiply(Pt, omega_inv)
        Pt_omega_inv_P = _mat_multiply(Pt_omega_inv, P)

        # M = (tau_sigma_inv + P'*Omega^-1*P)^-1
        M_inv = _mat_add(tau_sigma_inv, Pt_omega_inv_P)
        M = _mat_inverse(M_inv)

        # tau_sigma_inv * pi
        pi_col = _vec_to_col(pi)
        tau_sigma_inv_pi = _col_to_vec(_mat_multiply(tau_sigma_inv, pi_col))

        # P' * Omega^-1 * Q
        Q_col = _vec_to_col(Q)
        Pt_omega_inv_Q = _col_to_vec(_mat_multiply(Pt_omega_inv, Q_col))

        # Combined: tau_sigma_inv*pi + P'*Omega^-1*Q
        combined = [tau_sigma_inv_pi[i] + Pt_omega_inv_Q[i] for i in range(n)]

        # Posterior mean: M @ combined
        mu_post = _mat_vec_multiply(M, combined)

        # Posterior covariance: M (the inverse of the precision matrix)
        # Full posterior cov = Sigma + M, but M alone is the posterior uncertainty
        # For weight optimization, use M + Sigma as the posterior covariance
        post_cov = _mat_add(sigma, M)

        posterior_returns = {symbols[i]: mu_post[i] for i in range(n)}

        return BlackLittermanResult(
            posterior_returns=posterior_returns,
            posterior_covariance=_matrix_to_dict(post_cov, symbols),
            equilibrium_returns=dict(eq_ret),
        )


def _matrix_to_dict(
    mat: list[list[float]], symbols: list[str],
) -> dict[str, dict[str, float]]:
    """Convert NxN matrix to nested dict keyed by symbol names."""
    n = len(symbols)
    return {
        symbols[i]: {symbols[j]: mat[i][j] for j in range(n)}
        for i in range(n)
    }
