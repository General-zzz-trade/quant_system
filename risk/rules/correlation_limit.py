"""Correlation concentration limit — prevents excessive correlated exposure.

Rejects new positions when portfolio correlation concentration exceeds threshold.

NOTE: Pre-trade correlation check also exists in risk/correlation_gate.py
(used in the order gate chain). This rule provides deeper portfolio-level
evaluation via RiskAggregator, while CorrelationGate is a fast per-order filter.

NOTE: This rule is currently INACTIVE in production. The required meta fields
(portfolio_avg_correlation, position_correlation_to_portfolio) are not populated
by LiveMetaBuilder. To activate, wire the corresponding data source into
meta_builder_live.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from event.types import IntentEvent, OrderEvent
from risk.decisions import RiskAction, RiskCode, RiskDecision, RiskScope, RiskViolation


@dataclass(frozen=True, slots=True)
class CorrelationLimitRule:
    """Limits portfolio concentration in highly-correlated assets.

    Reads from meta:
        - portfolio_avg_correlation: float — average pairwise correlation
        - position_correlation_to_portfolio: float — new position's correlation to existing
    """

    name: str = "correlation_limit"
    max_avg_correlation: float = 0.7
    max_position_correlation: float = 0.85

    def evaluate_intent(
        self, intent: IntentEvent, *, meta: Mapping[str, Any] = {},
    ) -> RiskDecision:
        avg_corr = meta.get("portfolio_avg_correlation")
        if avg_corr is not None and avg_corr > self.max_avg_correlation:
            return RiskDecision(
                action=RiskAction.REJECT,
                violations=(RiskViolation(
                    code=RiskCode.MAX_GROSS,
                    scope=RiskScope.PORTFOLIO,
                    message=(
                        f"Portfolio avg correlation {avg_corr:.2f} "
                        f"exceeds limit {self.max_avg_correlation:.2f}"
                    ),
                ),),
            )

        pos_corr = meta.get("position_correlation_to_portfolio")
        if pos_corr is not None and pos_corr > self.max_position_correlation:
            return RiskDecision(
                action=RiskAction.REJECT,
                violations=(RiskViolation(
                    code=RiskCode.MAX_GROSS,
                    scope=RiskScope.SYMBOL,
                    message=(
                        f"Position correlation to portfolio {pos_corr:.2f} "
                        f"exceeds limit {self.max_position_correlation:.2f}"
                    ),
                ),),
            )

        return RiskDecision(action=RiskAction.ALLOW)

    def evaluate_order(
        self, order: OrderEvent, *, meta: Mapping[str, Any] = {},
    ) -> RiskDecision:
        return self.evaluate_intent(order, meta=meta)
