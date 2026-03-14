"""Drift-threshold rebalancing — trigger when weights deviate from target."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from state.snapshot import StateSnapshot


@dataclass(frozen=True, slots=True)
class ThresholdRebalance:
    """Rebalance when any position drifts beyond ``drift_pct`` from target.

    Parameters
    ----------
    drift_pct : Decimal
        Maximum allowed drift as a fraction (e.g., 0.05 = 5%).
    min_delta_qty : Decimal
        Minimum absolute position change to trigger rebalance.
    """
    drift_pct: Decimal = Decimal("0.05")
    min_delta_qty: Decimal = Decimal("0")

    def should_rebalance(self, snapshot: StateSnapshot) -> bool:
        positions = getattr(snapshot, "positions", {})
        if not positions:
            return True

        # Check if any position has non-trivial size (basic heuristic)
        total_abs = sum(abs(p.qty) for p in positions.values())
        if total_abs == 0:
            return True

        # Check each position for drift
        for sym, pos in positions.items():
            if abs(pos.qty) >= self.min_delta_qty:
                weight = abs(pos.qty) / total_abs if total_abs > 0 else Decimal("0")
                target_weight = Decimal("1") / Decimal(str(max(1, len(positions))))
                drift = abs(weight - target_weight)
                if drift > self.drift_pct:
                    return True

        return False
