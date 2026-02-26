# portfolio/optimizer/factor_constraints.py
"""Factor exposure constraints for portfolio optimization."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FactorExposureConstraint:
    """Constrain portfolio factor exposure (e.g., beta neutrality).

    Enforces: min_exposure <= sum(w_i * loading_i) <= max_exposure.
    """

    name: str = "factor_exposure"
    factor_loadings: Mapping[str, float] = field(default_factory=dict)
    max_exposure: float = 0.1
    min_exposure: float = -0.1

    def exposure(self, weights: Mapping[str, float]) -> float:
        """Compute current portfolio factor exposure."""
        return sum(
            weights.get(s, 0.0) * self.factor_loadings.get(s, 0.0)
            for s in set(weights) | set(self.factor_loadings)
        )

    def is_feasible(self, weights: Mapping[str, float]) -> bool:
        """Check if portfolio factor exposure is within bounds."""
        exp = self.exposure(weights)
        tol = 1e-10
        return (self.min_exposure - tol) <= exp <= (self.max_exposure + tol)

    def project(self, weights: dict[str, float]) -> dict[str, float]:
        """Project weights to satisfy factor exposure bounds.

        If exposure is out of bounds, adjust weights along the direction
        of factor loadings to bring exposure within bounds.
        """
        exp = self.exposure(weights)

        if self.min_exposure <= exp <= self.max_exposure:
            return weights

        # Determine target exposure (nearest bound)
        if exp > self.max_exposure:
            target = self.max_exposure
        else:
            target = self.min_exposure

        excess = exp - target

        # Compute sum of squared loadings for assets in the portfolio
        loading_sq_sum = 0.0
        for s in weights:
            loading = self.factor_loadings.get(s, 0.0)
            loading_sq_sum += loading * loading

        if loading_sq_sum < 1e-15:
            # No loadings overlap with portfolio; cannot project
            return dict(weights)

        # Adjust each weight proportionally to its loading
        # w_new_i = w_i - excess * loading_i / sum(loading_j^2)
        result = {}
        for s, w in weights.items():
            loading = self.factor_loadings.get(s, 0.0)
            result[s] = w - excess * loading / loading_sq_sum

        return result


@dataclass(frozen=True, slots=True)
class SectorExposureConstraint:
    """Constrain portfolio sector exposure.

    Groups assets by sector and enforces weight limits per sector.
    """

    name: str = "sector_exposure"
    sector_map: Mapping[str, str] = field(default_factory=dict)  # symbol -> sector
    max_sector_weight: float = 0.4
    min_sector_weight: float = 0.0

    def sector_weights(self, weights: Mapping[str, float]) -> dict[str, float]:
        """Compute weight per sector."""
        sector_w: dict[str, float] = {}
        for sym, w in weights.items():
            sector = self.sector_map.get(sym, "unknown")
            sector_w[sector] = sector_w.get(sector, 0.0) + w
        return sector_w

    def is_feasible(self, weights: Mapping[str, float]) -> bool:
        """Check if all sector weights are within bounds."""
        for sector, sw in self.sector_weights(weights).items():
            if sw > self.max_sector_weight or sw < self.min_sector_weight:
                return False
        return True

    def project(self, weights: dict[str, float]) -> dict[str, float]:
        """Scale down weights in overweight sectors."""
        result = dict(weights)
        sector_w = self.sector_weights(result)

        for sector, sw in sector_w.items():
            if sw <= self.max_sector_weight:
                continue
            if abs(sw) < 1e-15:
                continue

            scale = self.max_sector_weight / sw
            for sym in result:
                if self.sector_map.get(sym, "unknown") == sector:
                    result[sym] *= scale

        return result
