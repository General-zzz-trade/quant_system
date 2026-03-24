# execution/safety/limits.py
"""Pre-flight order limits — qty, notional, rate, and position size checks.

Delegates to RustOrderLimiter for lock-free, high-performance checking.
Python OrderLimiter is a thin wrapper that preserves the existing API
(Decimal inputs, LimitCheckResult output).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from _quant_hotpath import RustOrderLimiter  # type: ignore[import-untyped]

# Re-export for callers that reference the Rust type directly.
RustOrderLimiterType = RustOrderLimiter


@dataclass(frozen=True, slots=True)
class OrderLimitsConfig:
    max_order_qty: Optional[Decimal] = None
    max_order_notional: Optional[Decimal] = None
    max_position_notional: Optional[Decimal] = None
    max_orders_per_second: Optional[float] = None
    max_daily_orders: Optional[int] = None
    max_daily_notional: Optional[Decimal] = None


@dataclass(frozen=True, slots=True)
class LimitCheckResult:
    allowed: bool
    violated_rule: Optional[str] = None
    detail: Optional[str] = None


class OrderLimiter:
    """Rust-backed order limiter with same API as original Python version.

    All numeric checks and rate/daily tracking are delegated to RustOrderLimiter.
    """

    def __init__(self, cfg: Optional[OrderLimitsConfig] = None) -> None:
        self._cfg = cfg or OrderLimitsConfig()
        c = self._cfg
        self._rust = RustOrderLimiter(
            max_order_qty=float(c.max_order_qty) if c.max_order_qty is not None else None,
            max_order_notional=float(c.max_order_notional) if c.max_order_notional is not None else None,
            max_position_notional=(
                float(c.max_position_notional) if c.max_position_notional is not None else None
            ),
            max_orders_per_sec=c.max_orders_per_second,
            max_daily_orders=c.max_daily_orders,
            max_daily_notional=float(c.max_daily_notional) if c.max_daily_notional is not None else None,
        )

    def check_order(
        self,
        *,
        qty: Decimal,
        price: Optional[Decimal] = None,
        current_position_notional: Decimal = Decimal("0"),
    ) -> LimitCheckResult:
        effective_price = float(price) if price is not None else 0.0
        allowed, reason = self._rust.check(
            qty=float(qty),
            price=effective_price,
            current_position_notional=float(current_position_notional),
        )
        if allowed:
            # Record the order so rate/daily counters are updated
            notional = float(qty) * effective_price
            self._rust.record_order(notional)
            return LimitCheckResult(allowed=True)

        # Map Rust reason strings to violated_rule names.
        rule = _parse_rule(reason)
        return LimitCheckResult(allowed=False, violated_rule=rule, detail=reason)

    def reset_daily(self) -> None:
        self._rust.reset_daily()


def _parse_rule(reason: str | None) -> str | None:
    """Extract the rule name from Rust reason string (e.g. 'max_order_qty: ...' -> 'max_order_qty')."""
    if not reason:
        return None
    return reason.split(":")[0].strip()
