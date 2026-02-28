"""Kelly Criterion allocator — optimal position sizing based on expected returns and risk."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from portfolio.allocator import (
    AccountSnapshot,
    AllocationPlan,
    AllocatorError,
    PortfolioConstraints,
    PriceProvider,
    TargetWeightAllocator,
    _abs,
    _d,
)


def _invert_2x2(a: float, b: float, c: float, d: float) -> Tuple[float, float, float, float]:
    """Invert a 2x2 matrix [[a,b],[c,d]]."""
    det = a * d - b * c
    if abs(det) < 1e-15:
        raise AllocatorError("Covariance matrix is singular")
    return d / det, -b / det, -c / det, a / det


def _invert_matrix(cov: Dict[str, Dict[str, float]], symbols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Invert a symmetric positive-definite covariance matrix via Gauss-Jordan elimination."""
    n = len(symbols)
    if n == 0:
        return {}

    # Build augmented matrix [cov | I]
    aug = [[0.0] * (2 * n) for _ in range(n)]
    for i, si in enumerate(symbols):
        for j, sj in enumerate(symbols):
            aug[i][j] = cov.get(si, {}).get(sj, 0.0)
        aug[i][n + i] = 1.0

    # Gauss-Jordan with partial pivoting
    for col in range(n):
        # Find pivot
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-15:
            raise AllocatorError("Covariance matrix is singular or near-singular")
        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        # Scale pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate column
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]

    # Extract inverse
    inv: Dict[str, Dict[str, float]] = {}
    for i, si in enumerate(symbols):
        inv[si] = {}
        for j, sj in enumerate(symbols):
            inv[si][sj] = aug[i][n + j]
    return inv


@dataclass(frozen=True, slots=True)
class KellyAllocator:
    """
    Kelly Criterion position sizing allocator.

    Multi-asset Kelly formula: w* = Σ⁻¹ · μ
    where Σ is the covariance matrix and μ is expected excess returns.

    In practice, uses fractional Kelly (default half-Kelly) to reduce
    parameter estimation risk and drawdown volatility.

    inputs:
      - expected_returns: Mapping[str, float] — expected excess returns per symbol
      - covariance: Mapping[str, Mapping[str, float]] — covariance matrix
      - kelly_fraction: float (default 0.5) — fraction of full Kelly (0.5 = half Kelly)
      - max_concentration: float (optional) — max weight for any single asset

    The final weights are passed through TargetWeightAllocator for constraint enforcement.
    """
    name: str = "kelly_allocator"
    default_kelly_fraction: float = 0.5
    default_max_concentration: Optional[float] = None

    def allocate(
        self,
        *,
        ts: Any,
        symbols: Sequence[str],
        account: AccountSnapshot,
        prices: PriceProvider,
        constraints: PortfolioConstraints,
        inputs: Mapping[str, Any] | None = None,
        tags: Tuple[str, ...] = (),
    ) -> AllocationPlan:
        if inputs is None:
            raise AllocatorError("KellyAllocator needs inputs")
        if "expected_returns" not in inputs or "covariance" not in inputs:
            raise AllocatorError("KellyAllocator needs inputs['expected_returns'] and inputs['covariance']")

        mu: Mapping[str, Any] = inputs["expected_returns"]
        cov: Mapping[str, Mapping[str, Any]] = inputs["covariance"]
        kelly_frac = float(inputs.get("kelly_fraction", self.default_kelly_fraction))
        max_conc = inputs.get("max_concentration", self.default_max_concentration)

        if not (0 < kelly_frac <= 1.0):
            raise AllocatorError(f"kelly_fraction must be in (0, 1], got {kelly_frac}")

        n = len(symbols)
        if n == 0:
            raise AllocatorError("symbols must not be empty")

        # Convert to float for matrix math
        mu_f = {s: float(mu.get(s, 0.0)) for s in symbols}
        cov_f: Dict[str, Dict[str, float]] = {}
        for si in symbols:
            cov_f[si] = {}
            for sj in symbols:
                cov_f[si][sj] = float(cov.get(si, {}).get(sj, 0.0))

        # w* = Σ⁻¹ · μ  (full Kelly)
        cov_inv = _invert_matrix(cov_f, symbols)

        raw_weights: Dict[str, float] = {}
        for si in symbols:
            w = 0.0
            for sj in symbols:
                w += cov_inv[si][sj] * mu_f[sj]
            raw_weights[si] = w

        # Apply fractional Kelly
        scaled_weights = {s: raw_weights[s] * kelly_frac for s in symbols}

        # Apply concentration constraint
        if max_conc is not None:
            max_c = float(max_conc)
            for s in symbols:
                if abs(scaled_weights[s]) > max_c:
                    scaled_weights[s] = max_c if scaled_weights[s] > 0 else -max_c

        # Convert to Decimal for TargetWeightAllocator
        target_weights = {s: _d(scaled_weights[s]) for s in symbols}

        return TargetWeightAllocator().allocate(
            ts=ts,
            symbols=symbols,
            account=account,
            prices=prices,
            constraints=constraints,
            inputs={"target_weights": target_weights, "weight_residual_to_cash": True},
            tags=tags + (self.name,),
        )

    def compute_kelly_weights(
        self,
        symbols: Sequence[str],
        expected_returns: Mapping[str, float],
        covariance: Mapping[str, Mapping[str, float]],
        kelly_fraction: float = 0.5,
    ) -> Dict[str, float]:
        """Compute raw Kelly weights without going through the full allocator pipeline.

        Useful for analysis and backtesting.
        """
        cov_inv = _invert_matrix(
            {si: {sj: float(covariance.get(si, {}).get(sj, 0.0)) for sj in symbols} for si in symbols},
            symbols,
        )
        weights: Dict[str, float] = {}
        for si in symbols:
            w = 0.0
            for sj in symbols:
                w += cov_inv[si][sj] * float(expected_returns.get(sj, 0.0))
            weights[si] = w * kelly_fraction
        return weights
