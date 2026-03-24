# execution/safety/limits.py
"""Pre-flight order limits — qty, notional, rate, and position size checks."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from threading import RLock
from time import monotonic
from typing import Optional

from _quant_hotpath import RustOrderLimiter  # type: ignore[import-untyped]

# Rust-backed order limiter — available for latency-sensitive paths where
# Python lock contention is a concern.  Same semantics as OrderLimiter.
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
    """订单预检限制器 — 防止代码 bug 导致异常订单。"""

    def __init__(self, cfg: Optional[OrderLimitsConfig] = None) -> None:
        self._cfg = cfg or OrderLimitsConfig()
        self._lock = RLock()
        self._order_timestamps: list[float] = []
        self._daily_order_count: int = 0
        self._daily_notional: Decimal = Decimal("0")
        self._day_start: float = monotonic()

    def check_order(
        self,
        *,
        qty: Decimal,
        price: Optional[Decimal] = None,
        current_position_notional: Decimal = Decimal("0"),
    ) -> LimitCheckResult:
        now = monotonic()
        notional = qty * price if price is not None else Decimal("0")

        with self._lock:
            self._maybe_reset_daily(now)

            if self._cfg.max_order_qty is not None and qty > self._cfg.max_order_qty:
                return LimitCheckResult(False, "max_order_qty",
                    f"qty={qty} > max={self._cfg.max_order_qty}")

            if self._cfg.max_order_notional is not None and notional > self._cfg.max_order_notional:
                return LimitCheckResult(False, "max_order_notional",
                    f"notional={notional} > max={self._cfg.max_order_notional}")

            if self._cfg.max_position_notional is not None:
                projected = current_position_notional + notional
                if projected > self._cfg.max_position_notional:
                    return LimitCheckResult(False, "max_position_notional",
                        f"projected={projected} > max={self._cfg.max_position_notional}")

            if self._cfg.max_orders_per_second is not None:
                cutoff = now - 1.0
                self._order_timestamps = [t for t in self._order_timestamps if t > cutoff]
                if len(self._order_timestamps) >= self._cfg.max_orders_per_second:
                    return LimitCheckResult(False, "max_orders_per_second",
                        f"rate={len(self._order_timestamps)}/s")

            if self._cfg.max_daily_orders is not None and self._daily_order_count >= self._cfg.max_daily_orders:
                return LimitCheckResult(False, "max_daily_orders",
                    f"count={self._daily_order_count}")

            if self._cfg.max_daily_notional is not None:
                if self._daily_notional + notional > self._cfg.max_daily_notional:
                    return LimitCheckResult(False, "max_daily_notional",
                        f"total={self._daily_notional + notional}")

            self._order_timestamps.append(now)
            self._daily_order_count += 1
            self._daily_notional += notional

        return LimitCheckResult(allowed=True)

    def reset_daily(self) -> None:
        with self._lock:
            self._daily_order_count = 0
            self._daily_notional = Decimal("0")
            self._day_start = monotonic()

    def _maybe_reset_daily(self, now: float) -> None:
        if now - self._day_start > 86400.0:
            self._daily_order_count = 0
            self._daily_notional = Decimal("0")
            self._day_start = now
