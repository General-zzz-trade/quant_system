# runner/gates/carry_cost_gate.py
"""Carry cost gate — adjust position scale based on funding + basis carry.

When holding a position, funding and basis create a carry cost/benefit:
  - Long + negative funding → RECEIVE carry → boost
  - Long + positive funding → PAY carry → reduce
  - High basis → contango carry cost for longs
  - Low basis → backwardation carry benefit for longs

Unlike FundingAlphaGate (which handles 100x extreme cases), this gate
provides continuous carry-adjusted sizing for standard leverage (1-20x).

Carry impact = (annualized_funding + basis_carry) × leverage × hold_time_fraction
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from runner.gate_chain import GateResult

_log = logging.getLogger(__name__)


@dataclass
class CarryCostConfig:
    """Configuration for carry cost gate."""
    enabled: bool = True

    # Cost thresholds (annualized %)
    favorable_carry_pct: float = 5.0    # > 5% annualized carry in our favor → boost
    costly_carry_pct: float = 10.0      # > 10% annualized carry against us → reduce
    extreme_carry_pct: float = 30.0     # > 30% annualized → significant reduction

    # Scaling factors
    favorable_scale: float = 1.15       # carry helps → modest boost
    costly_scale: float = 0.7           # carry hurts → reduce
    extreme_scale: float = 0.4          # extreme carry against → aggressive reduce

    # Leverage for carry impact calculation
    leverage: float = 10.0


class CarryCostGate:
    """Gate: adjust position size by carry cost/benefit.

    Reads from context:
      - funding_rate: current 8h funding rate
      - funding_annualized: annualized funding (rate × 3 × 365)
      - basis: futures-spot spread (%)
      - basis_carry_adj: basis + annualized funding
      - signal: trade direction (+1/-1)

    Computes net carry direction relative to signal and scales accordingly.
    """

    name = "CarryCost"

    def __init__(self, cfg: CarryCostConfig | None = None) -> None:
        self._cfg = cfg or CarryCostConfig()
        self._total_checks = 0
        self._boosted_count = 0
        self._reduced_count = 0

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        if not self._cfg.enabled:
            return GateResult(allowed=True, scale=1.0)

        self._total_checks += 1
        cfg = self._cfg

        # Get signal direction
        signal = 0
        meta = getattr(ev, "metadata", None) or {}
        if isinstance(meta, dict):
            signal = int(meta.get("signal", 0))
        if signal == 0:
            signal = int(context.get("signal", 0))

        if signal == 0:
            return GateResult(allowed=True, scale=1.0)

        # Get carry components
        funding_rate = _safe_float(context.get("funding_rate"), 0.0)
        basis = _safe_float(context.get("basis"), 0.0)

        # Compute annualized carry impact
        # funding_rate is per 8h, annualize: rate × 3 × 365
        funding_annual = funding_rate * 3 * 365 * 100  # as percentage

        # basis is already in % terms (futures premium)
        # For a long position: negative funding = receive, positive basis = pay contango
        # For a short position: positive funding = receive, negative basis = receive backwardation

        if signal > 0:  # Long
            # Longs pay positive funding, pay contango (positive basis)
            net_carry_cost = funding_annual + basis * 365  # annualized
        else:  # Short
            # Shorts receive positive funding, receive contango
            net_carry_cost = -(funding_annual + basis * 365)

        # net_carry_cost > 0 means we PAY carry
        # net_carry_cost < 0 means we RECEIVE carry

        abs_carry = abs(net_carry_cost)

        if net_carry_cost < 0:
            # We receive carry — favorable
            if abs_carry >= cfg.favorable_carry_pct:
                self._boosted_count += 1
                scale = cfg.favorable_scale
                reason = f"carry_favorable net={net_carry_cost:.1f}%/yr"
            else:
                scale = 1.0
                reason = ""
        else:
            # We pay carry — costly
            if abs_carry >= cfg.extreme_carry_pct:
                self._reduced_count += 1
                scale = cfg.extreme_scale
                reason = f"carry_extreme net={net_carry_cost:.1f}%/yr"
            elif abs_carry >= cfg.costly_carry_pct:
                self._reduced_count += 1
                scale = cfg.costly_scale
                reason = f"carry_costly net={net_carry_cost:.1f}%/yr"
            else:
                scale = 1.0
                reason = ""

        if reason:
            _log.debug(
                "CarryCost: signal=%d funding_ann=%.1f%% basis=%.4f net=%.1f%%/yr → scale=%.2f",
                signal, funding_annual, basis, net_carry_cost, scale,
            )

        return GateResult(allowed=True, scale=scale, reason=reason)

    @property
    def stats(self) -> dict:
        return {
            "total_checks": self._total_checks,
            "boosted": self._boosted_count,
            "reduced": self._reduced_count,
        }


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to float, returning default for NaN/None/missing."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if f != f else f
    except (ValueError, TypeError):
        return default
