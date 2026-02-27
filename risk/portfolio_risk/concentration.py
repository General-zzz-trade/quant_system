"""Concentration monitoring — tracks portfolio concentration risk.

Monitors position sizes, sector exposure, and HHI concentration index.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConcentrationMetrics:
    """Concentration risk metrics for a portfolio."""
    hhi: float  # Herfindahl-Hirschman Index (0 to 1)
    max_weight: float  # largest single position weight
    max_weight_symbol: str
    top3_weight: float  # sum of top 3 position weights
    effective_positions: float  # 1/HHI — effective number of positions
    sector_hhi: float  # HHI across sectors (if sector data available)


@dataclass(frozen=True, slots=True)
class ConcentrationAlert:
    """Alert when concentration exceeds threshold."""
    metric: str
    value: float
    threshold: float
    message: str


class ConcentrationMonitor:
    """Monitors portfolio concentration and generates alerts.

    Tracks position-level and sector-level concentration using
    Herfindahl-Hirschman Index (HHI) and weight-based thresholds.
    """

    def __init__(
        self,
        *,
        max_single_weight: float = 0.25,
        max_top3_weight: float = 0.60,
        max_hhi: float = 0.30,
        max_sector_weight: float = 0.40,
    ) -> None:
        self._max_single_weight = max_single_weight
        self._max_top3_weight = max_top3_weight
        self._max_hhi = max_hhi
        self._max_sector_weight = max_sector_weight
        self._sector_map: Dict[str, str] = {}  # symbol → sector

    def set_sector_map(self, mapping: Dict[str, str]) -> None:
        """Set symbol-to-sector mapping."""
        self._sector_map = dict(mapping)

    def compute_metrics(
        self,
        positions: Dict[str, Decimal],
        prices: Dict[str, Decimal],
    ) -> ConcentrationMetrics:
        """Compute concentration metrics for current portfolio."""
        # Calculate position values
        values: Dict[str, float] = {}
        for symbol, qty in positions.items():
            price = prices.get(symbol, Decimal("0"))
            values[symbol] = float(abs(qty) * price)

        total_value = sum(values.values())
        if total_value <= 0:
            return ConcentrationMetrics(
                hhi=0.0,
                max_weight=0.0,
                max_weight_symbol="",
                top3_weight=0.0,
                effective_positions=0.0,
                sector_hhi=0.0,
            )

        # Position weights
        weights = {s: v / total_value for s, v in values.items()}

        # HHI = sum of squared weights
        hhi = sum(w ** 2 for w in weights.values())

        # Top positions
        sorted_weights = sorted(weights.values(), reverse=True)
        max_weight = sorted_weights[0] if sorted_weights else 0.0
        max_symbol = max(weights, key=weights.get) if weights else ""  # type: ignore[arg-type]
        top3_weight = sum(sorted_weights[:3])

        # Effective positions
        effective = 1.0 / hhi if hhi > 0 else 0.0

        # Sector HHI
        sector_hhi = self._compute_sector_hhi(values, total_value)

        return ConcentrationMetrics(
            hhi=hhi,
            max_weight=max_weight,
            max_weight_symbol=max_symbol,
            top3_weight=top3_weight,
            effective_positions=effective,
            sector_hhi=sector_hhi,
        )

    def check_alerts(
        self, metrics: ConcentrationMetrics,
    ) -> List[ConcentrationAlert]:
        """Check concentration metrics against thresholds."""
        alerts: List[ConcentrationAlert] = []

        if metrics.max_weight > self._max_single_weight:
            alerts.append(ConcentrationAlert(
                metric="max_single_weight",
                value=metrics.max_weight,
                threshold=self._max_single_weight,
                message=f"{metrics.max_weight_symbol} weight {metrics.max_weight:.1%} exceeds {self._max_single_weight:.1%}",
            ))

        if metrics.top3_weight > self._max_top3_weight:
            alerts.append(ConcentrationAlert(
                metric="top3_weight",
                value=metrics.top3_weight,
                threshold=self._max_top3_weight,
                message=f"Top 3 concentration {metrics.top3_weight:.1%} exceeds {self._max_top3_weight:.1%}",
            ))

        if metrics.hhi > self._max_hhi:
            alerts.append(ConcentrationAlert(
                metric="hhi",
                value=metrics.hhi,
                threshold=self._max_hhi,
                message=f"HHI {metrics.hhi:.3f} exceeds {self._max_hhi:.3f}",
            ))

        if metrics.sector_hhi > self._max_sector_weight:
            alerts.append(ConcentrationAlert(
                metric="sector_hhi",
                value=metrics.sector_hhi,
                threshold=self._max_sector_weight,
                message=f"Sector HHI {metrics.sector_hhi:.3f} exceeds {self._max_sector_weight:.3f}",
            ))

        return alerts

    def _compute_sector_hhi(
        self, values: Dict[str, float], total: float,
    ) -> float:
        """Compute HHI across sectors."""
        if not self._sector_map or total <= 0:
            return 0.0

        sector_values: Dict[str, float] = {}
        for symbol, val in values.items():
            sector = self._sector_map.get(symbol, "unknown")
            sector_values[sector] = sector_values.get(sector, 0.0) + val

        sector_weights = [v / total for v in sector_values.values()]
        return sum(w ** 2 for w in sector_weights)
