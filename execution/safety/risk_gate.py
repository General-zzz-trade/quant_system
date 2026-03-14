# execution/safety/risk_gate.py
"""RiskGate — pre-execution risk checks before orders reach the venue.

Validates:
- Position limits (per-symbol and total notional)
- Order size sanity (no fat-finger)
- Kill switch not triggered
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RiskGateConfig:
    max_position_notional: float = 100_000.0
    max_order_notional: float = 50_000.0
    max_open_orders: int = 20
    max_portfolio_notional: float = 500_000.0


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    allowed: bool
    reason: str = ""


@dataclass
class RiskGate:
    """Pre-execution risk gate. Must pass before orders reach the venue."""

    config: RiskGateConfig = field(default_factory=RiskGateConfig)
    get_positions: Optional[Callable[[], Mapping[str, Any]]] = None
    get_open_order_count: Optional[Callable[[], int]] = None
    is_killed: Optional[Callable[[], bool]] = None

    def check(self, cmd: Any) -> RiskCheckResult:
        """Check if an order command passes risk validation."""
        # Kill switch check
        if self.is_killed is not None and self.is_killed():
            return RiskCheckResult(allowed=False, reason="kill_switch_active")

        # Open order limit
        if self.get_open_order_count is not None:
            count = self.get_open_order_count()
            if count >= self.config.max_open_orders:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"max_open_orders:{count}>={self.config.max_open_orders}",
                )

        # Order notional check — fail-closed: reject if qty or price missing
        qty = _get_qty(cmd)
        price = _get_price(cmd)
        if qty is None or price is None:
            logger.warning("RiskGate: cannot extract qty=%s price=%s from order, rejecting", qty, price)
            return RiskCheckResult(
                allowed=False,
                reason=f"missing_qty_or_price:qty={qty},price={price}",
            )

        notional = abs(qty * price)
        if notional > self.config.max_order_notional:
            return RiskCheckResult(
                allowed=False,
                reason=f"order_notional:{notional:.2f}>{self.config.max_order_notional:.2f}",
            )

        # Position limit check
        if self.get_positions is not None:
            symbol = str(getattr(cmd, "symbol", ""))
            positions = self.get_positions()
            current_notional = _position_notional(positions, symbol, price)
            order_notional = abs(qty * price)
            new_notional = current_notional + order_notional

            if new_notional > self.config.max_position_notional:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"position_notional:{new_notional:.2f}>{self.config.max_position_notional:.2f}",
                )

            # Portfolio-wide check — use order price only for the order's symbol
            total = sum(
                _position_notional(positions, s, price if s == symbol else 0.0)
                for s in positions
            ) + order_notional
            if total > self.config.max_portfolio_notional:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"portfolio_notional:{total:.2f}>{self.config.max_portfolio_notional:.2f}",
                )

        return RiskCheckResult(allowed=True)


def _get_qty(cmd: Any) -> Optional[float]:
    for attr in ("qty", "quantity", "size"):
        v = getattr(cmd, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _get_price(cmd: Any) -> Optional[float]:
    for attr in ("price", "limit_price"):
        v = getattr(cmd, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _position_notional(
    positions: Mapping[str, Any], symbol: str, fallback_price: float,
) -> float:
    pos = positions.get(symbol)
    if pos is None:
        return 0.0
    qty = getattr(pos, "qty", None) or getattr(pos, "quantity", None) or 0
    # Use the position's own price (mark or entry) rather than the order's price
    pos_price = getattr(pos, "mark_price", None) or getattr(pos, "entry_price", None)
    if pos_price is None:
        pos_price = fallback_price
    try:
        return abs(float(qty)) * float(pos_price)
    except (TypeError, ValueError):
        return 0.0
