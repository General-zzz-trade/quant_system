"""Isolated-margin liquidation distance and the rule that a protective stop must sit well inside it."""
from __future__ import annotations


def liquidation_distance(isolated_leverage: float, maintenance_margin_rate: float) -> float:
    """Adverse price move (fraction) that liquidates an isolated position."""
    return 1.0 / isolated_leverage - maintenance_margin_rate


def check_stop_inside_liquidation(stop_pct: float, leverage: int, margin: str | None, *,
                                  maintenance_margin_rate: float = 0.005, fraction: float = 0.5) -> None:
    """With isolated margin, refuse a protective stop farther than ``fraction`` of the liquidation distance."""
    if margin != "ISOLATED":
        return
    limit = fraction * liquidation_distance(leverage, maintenance_margin_rate)
    if float(stop_pct) > limit:
        raise RuntimeError(f"protective stop {float(stop_pct):.2%} is not inside {fraction:.0%} of the "
                           f"{leverage}x isolated liquidation distance; use --stop-pct <= {limit:.4f}")


__all__ = ["liquidation_distance", "check_stop_inside_liquidation"]
