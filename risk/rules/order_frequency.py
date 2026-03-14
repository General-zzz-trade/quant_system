# quant_system/risk/rules/order_frequency.py
"""Order frequency limit rule — prevents runaway order loops.

NOTE: This rule is currently INACTIVE in production. The required meta fields
(order_rate_per_min, recent_order_count) are not populated by LiveMetaBuilder.
To activate, wire an order rate tracker into meta_builder_live.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from event.types import IntentEvent, OrderEvent
from risk.decisions import (
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
)


class OrderFrequencyRuleError(RuntimeError):
    pass


def _get_from_meta(meta: Mapping[str, Any], *keys: str, default=None):
    for k in keys:
        if k in meta:
            return meta[k]
    return default


@dataclass(frozen=True, slots=True)
class OrderFrequencyRule:
    """
    订单频率限制规则（防止策略 bug 导致高频下单循环）

    从 meta 读取：
      - "order_rate_per_min"  : float — 近期每分钟订单数
      - "recent_order_count"  : int   — 窗口内总订单数

    由 RiskEvalMetaBuilder 负责注入上述字段。
    """

    name: str = "order_frequency"

    max_orders_per_minute: int = 30
    max_orders_per_window: int = 100
    window_seconds: int = 300

    def evaluate_intent(self, intent: IntentEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        rate = _get_from_meta(meta, "order_rate_per_min", default=None)
        if rate is not None and float(rate) > self.max_orders_per_minute:
            v = RiskViolation(
                code=RiskCode.OMS_DEGRADED,
                message="Intent pre-check: order rate exceeds per-minute limit",
                scope=RiskScope.GLOBAL,
                severity="warn",
                details={
                    "rate_per_min": str(rate),
                    "limit_per_min": str(self.max_orders_per_minute),
                },
            )
            return RiskDecision.reject((v,), scope=RiskScope.GLOBAL, tags=(self.name,))
        return RiskDecision.allow(tags=(self.name,))

    def evaluate_order(self, order: OrderEvent, *, meta: Mapping[str, Any]) -> RiskDecision:
        violations: list[RiskViolation] = []

        rate = _get_from_meta(meta, "order_rate_per_min", default=None)
        if rate is not None and float(rate) > self.max_orders_per_minute:
            violations.append(RiskViolation(
                code=RiskCode.OMS_DEGRADED,
                message="Order rate exceeds per-minute limit",
                scope=RiskScope.GLOBAL,
                severity="error",
                details={
                    "rate_per_min": str(rate),
                    "limit_per_min": str(self.max_orders_per_minute),
                },
            ))

        count = _get_from_meta(meta, "recent_order_count", "order_count_in_window", default=None)
        if count is not None and int(count) >= self.max_orders_per_window:
            violations.append(RiskViolation(
                code=RiskCode.OMS_DEGRADED,
                message="Order count exceeds window limit",
                scope=RiskScope.GLOBAL,
                severity="error",
                details={
                    "count": str(count),
                    "window_seconds": str(self.window_seconds),
                    "limit": str(self.max_orders_per_window),
                },
            ))

        if not violations:
            return RiskDecision.allow(tags=(self.name,))

        return RiskDecision.reject(tuple(violations), scope=RiskScope.GLOBAL, tags=(self.name,))
