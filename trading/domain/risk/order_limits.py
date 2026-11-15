"""Order limits: the rules of the Rust limiter this replaces, as a pure function (migration step 13b).

Checked in this order, the first that fails decides: order quantity, order notional, position notional
after the order, orders in the last second, orders today, notional today. The reason text keeps the Rust
limiter's wording, and ``violated_rule`` is its first word. Counting is the caller's: only an allowed order
counts, and "today" is whatever day the caller passes counts for.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

RATE_RULE = "max_orders_per_second"


@dataclass(frozen=True, slots=True)
class OrderLimitsConfig:
    max_order_qty: Decimal | None = None
    max_order_notional: Decimal | None = None
    max_position_notional: Decimal | None = None
    max_orders_per_second: float | None = None
    max_daily_orders: int | None = None
    max_daily_notional: Decimal | None = None


@dataclass(frozen=True, slots=True)
class LimitCheckResult:
    allowed: bool
    violated_rule: str | None = None
    detail: str | None = None


def _refuse(detail: str) -> LimitCheckResult:
    return LimitCheckResult(False, detail.split(":")[0].strip(), detail)


def check_limits(config: OrderLimitsConfig, *, qty: Decimal, price: Decimal, position_notional: Decimal,
                 orders_last_second: int, orders_today: int, notional_today: Decimal) -> LimitCheckResult:
    notional = qty * price
    c = config
    if c.max_order_qty is not None and qty > c.max_order_qty:
        return _refuse(f"max_order_qty: qty={qty} > max={c.max_order_qty}")
    if c.max_order_notional is not None and notional > c.max_order_notional:
        return _refuse(f"max_order_notional: notional={notional:.2f} > max={c.max_order_notional:.2f}")
    if c.max_position_notional is not None and position_notional + notional > c.max_position_notional:
        return _refuse(f"max_position_notional: projected={position_notional + notional:.2f} "
                       f"> max={c.max_position_notional:.2f}")
    if c.max_orders_per_second is not None and orders_last_second >= c.max_orders_per_second:
        return _refuse(f"{RATE_RULE}: rate={orders_last_second}/s")
    if c.max_daily_orders is not None and orders_today >= c.max_daily_orders:
        return _refuse(f"max_daily_orders: count={orders_today}")
    if c.max_daily_notional is not None and notional_today + notional > c.max_daily_notional:
        return _refuse(f"max_daily_notional: total={notional_today + notional:.2f}")
    return LimitCheckResult(True)


__all__ = ["LimitCheckResult", "OrderLimitsConfig", "RATE_RULE", "check_limits"]
