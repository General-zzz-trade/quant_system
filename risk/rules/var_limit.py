"""VaR limit rule — rejects orders that would push portfolio VaR beyond limit.

Reads pre-computed VaR from risk meta dict.

NOTE: This rule is currently INACTIVE in production. The required meta fields
(portfolio_var_95, portfolio_var_99, post_trade_var_95) are not populated by
LiveMetaBuilder. To activate, wire a VaR calculator into meta_builder_live.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from event.types import IntentEvent, OrderEvent
from risk.decisions import RiskAction, RiskCode, RiskDecision, RiskScope, RiskViolation


@dataclass(frozen=True, slots=True)
class VaRLimitRule:
    """Limits portfolio Value-at-Risk.

    NOTE: This rule requires portfolio VaR values (portfolio_var_95, portfolio_var_99)
    to be computed and injected into the risk meta dict. Currently no upstream
    component computes these values, so this rule is effectively inactive.
    Enable by wiring a VaR calculator into RiskAggregator meta builder.

    Reads from meta:
        - portfolio_var_95: float — current 95% VaR as % of equity
        - portfolio_var_99: float — current 99% VaR as % of equity
        - post_trade_var_95: float — projected VaR after trade
    """

    name: str = "var_limit"
    max_var_95_pct: float = 5.0    # Max 5% daily VaR at 95%
    max_var_99_pct: float = 10.0   # Max 10% daily VaR at 99%

    def evaluate_intent(
        self, intent: IntentEvent, *, meta: Mapping[str, Any] = {},
    ) -> RiskDecision:
        # Check current VaR
        var_95 = meta.get("portfolio_var_95")
        if var_95 is not None and var_95 > self.max_var_95_pct:
            return RiskDecision(
                action=RiskAction.REJECT,
                violations=(RiskViolation(
                    code=RiskCode.MAX_GROSS,
                    scope=RiskScope.PORTFOLIO,
                    message=(
                        f"Portfolio VaR(95%) {var_95:.2f}% "
                        f"exceeds limit {self.max_var_95_pct:.2f}%"
                    ),
                ),),
            )

        var_99 = meta.get("portfolio_var_99")
        if var_99 is not None and var_99 > self.max_var_99_pct:
            return RiskDecision(
                action=RiskAction.REJECT,
                violations=(RiskViolation(
                    code=RiskCode.MAX_GROSS,
                    scope=RiskScope.PORTFOLIO,
                    message=(
                        f"Portfolio VaR(99%) {var_99:.2f}% "
                        f"exceeds limit {self.max_var_99_pct:.2f}%"
                    ),
                ),),
            )

        # Check post-trade VaR
        post_var = meta.get("post_trade_var_95")
        if post_var is not None and post_var > self.max_var_95_pct:
            return RiskDecision(
                action=RiskAction.REJECT,
                violations=(RiskViolation(
                    code=RiskCode.MAX_GROSS,
                    scope=RiskScope.PORTFOLIO,
                    message=(
                        f"Post-trade VaR(95%) {post_var:.2f}% "
                        f"would exceed limit {self.max_var_95_pct:.2f}%"
                    ),
                ),),
            )

        return RiskDecision(action=RiskAction.ALLOW)

    def evaluate_order(
        self, order: OrderEvent, *, meta: Mapping[str, Any] = {},
    ) -> RiskDecision:
        return self.evaluate_intent(order, meta=meta)
