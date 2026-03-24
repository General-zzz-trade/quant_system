# execution/safety/risk_gate.py
"""RiskGate — pre-execution risk checks before orders reach the venue.

Delegates numeric validation (notional, position, portfolio limits) to
RustRiskGate via a single FFI call. Python layer handles dynamic callbacks
(get_positions, get_open_order_count, is_killed) and attribute extraction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from _quant_hotpath import (  # type: ignore[import-untyped]
    RustRiskGate as _RustRiskGate,
    RustRiskResult as _RustRiskResult,
)

logger = logging.getLogger(__name__)

# Rust-accelerated risk gate and result types — available for
# single-FFI-call risk evaluation in the binary hot path.
RustRiskGateType = _RustRiskGate
RustRiskResultType = _RustRiskResult


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
    """Pre-execution risk gate. Must pass before orders reach the venue.

    Numeric checks (notional limits, open order count) are delegated to
    RustRiskGate.  Python handles callback resolution and attribute extraction.
    """

    config: RiskGateConfig = field(default_factory=RiskGateConfig)
    get_positions: Optional[Callable[[], Mapping[str, Any]]] = None
    get_open_order_count: Optional[Callable[[], int]] = None
    is_killed: Optional[Callable[[], bool]] = None

    def __post_init__(self) -> None:
        self._rust = _RustRiskGate(
            max_open_orders=self.config.max_open_orders,
            max_order_notional=self.config.max_order_notional,
            max_position_notional=self.config.max_position_notional,
            max_portfolio_notional=self.config.max_portfolio_notional,
        )

    def check(self, cmd: Any) -> RiskCheckResult:
        """Check if an order command passes risk validation."""
        # Resolve dynamic state from callbacks
        kill_switch_armed = self.is_killed is not None and self.is_killed()
        open_order_count = self.get_open_order_count() if self.get_open_order_count is not None else 0

        # Extract qty, price, symbol from command object
        qty = _get_qty(cmd)
        price = _get_price(cmd)
        symbol = str(getattr(cmd, "symbol", ""))

        if qty is None or price is None:
            logger.warning(
                "RiskGate: cannot extract qty=%s price=%s (checked mark_price/price/limit_price) from order, rejecting",
                qty, price,
            )
            return RiskCheckResult(
                allowed=False,
                reason=f"missing_qty_or_price:qty={qty},price={price}",
            )

        # Compute position/portfolio notionals from callbacks
        position_notional = 0.0
        portfolio_notional = 0.0
        if self.get_positions is not None:
            positions = self.get_positions()
            position_notional = _position_notional(positions, symbol, price)
            # Portfolio-wide: sum all symbols (use order price only for order's symbol)
            portfolio_notional = sum(
                _position_notional(positions, s, price if s == symbol else 0.0)
                for s in positions
            )

        # Single FFI call to Rust for all numeric checks
        allowed, reason = self._rust.check(
            symbol=symbol,
            side=str(getattr(cmd, "side", "BUY")),
            qty=abs(qty),
            price=price,
            open_order_count=open_order_count,
            position_notional=position_notional,
            portfolio_notional=portfolio_notional,
            kill_switch_armed=kill_switch_armed,
        )

        if allowed:
            return RiskCheckResult(allowed=True)
        return RiskCheckResult(allowed=False, reason=reason or "")


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
    """Extract price for notional check. Prefer mark_price over order price."""
    for attr in ("mark_price", "price", "limit_price"):
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
