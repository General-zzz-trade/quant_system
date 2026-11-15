"""The order limiter: today's counts kept in a store, per UTC day (migration step 13b; ADR-0007 2b).

The rules are ``trading.domain.risk.order_limits``. The orders-per-second window lives in memory on the
monotonic clock, as before; a restart forgets at most one second of it. Today's order count and notional are
read from and added to the store, so a restart keeps them. A count that cannot be read, or an allowed order
whose count cannot be saved, refuses the order: the day's allowance is treated as used up.
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal

from trading.application.ports import Clock, LimitStore
from trading.domain.risk.drawdown import periods
from trading.domain.risk.order_limits import LimitCheckResult, OrderLimitsConfig, check_limits

UNREADABLE = "max_daily_orders"


class OrderLimiter:
    def __init__(self, config: OrderLimitsConfig, *, store: LimitStore, clock: Clock, monotonic: Clock) -> None:
        self._cfg, self.store, self.clock, self._monotonic = config, store, clock, monotonic
        self._recent: deque[float] = deque()

    def check_order(self, *, qty: Decimal, price: Decimal | None = None,
                    current_position_notional: Decimal = Decimal(0)) -> LimitCheckResult:
        price = price if price is not None else Decimal(0)
        day = periods(self.clock())["day"]
        try:
            orders, notional = self.store.load(day)
        except Exception as exc:
            return LimitCheckResult(False, UNREADABLE, f"{UNREADABLE}: today's count unreadable "
                                                       f"({type(exc).__name__}); treated as used up")
        now = self._monotonic()
        while self._recent and self._recent[0] <= now - 1.0:
            self._recent.popleft()
        result = check_limits(self._cfg, qty=qty, price=price, position_notional=current_position_notional,
                              orders_last_second=len(self._recent), orders_today=orders, notional_today=notional)
        if not result.allowed:
            return result
        try:
            self.store.add(day, qty * price)
        except Exception as exc:
            return LimitCheckResult(False, UNREADABLE, f"{UNREADABLE}: today's count not saved "
                                                       f"({type(exc).__name__}); order refused")
        self._recent.append(now)
        return result


__all__ = ["OrderLimiter"]
