# portfolio/optimizer/black_litterman.py
"""Black-Litterman model: market equilibrium + investor views -> posterior."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

from _quant_hotpath import cpp_black_litterman_posterior as _cpp_bl_posterior

logger = logging.getLogger(__name__)


# ── Minimal matrix helpers (used by equilibrium_returns for no-views case) ──


def _mat_zeros(n: int, m: int) -> list[list[float]]:
    return [[0.0] * m for _ in range(n)]


def _mat_vec_multiply(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


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

        if views:
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

        # No views: return equilibrium
        eq_ret = self.equilibrium_returns(market_weights, cov)
        sigma = _mat_zeros(n, n)
        for i, si in enumerate(symbols):
            row = cov.get(si, {})
            for j, sj in enumerate(symbols):
                sigma[i][j] = row.get(sj, 0.0)
        return BlackLittermanResult(
            posterior_returns=dict(eq_ret),
            posterior_covariance={
                symbols[i]: {symbols[j]: sigma[i][j] for j in range(n)}
                for i in range(n)
            },
            equilibrium_returns=dict(eq_ret),
        )
