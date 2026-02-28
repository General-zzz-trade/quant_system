"""Correlation gate — pre-trade check using CorrelationComputer.

Blocks new positions when portfolio correlation concentration is too high.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from risk.correlation_computer import CorrelationComputer
from risk.decisions import (
    RiskAction,
    RiskCode,
    RiskDecision,
    RiskScope,
    RiskViolation,
)


@dataclass(frozen=True, slots=True)
class CorrelationGateConfig:
    """Thresholds for the correlation gate."""

    max_avg_correlation: float = 0.7
    max_position_correlation: float = 0.85
    min_data_points: int = 20


class CorrelationGate:
    """Pre-trade correlation check using live CorrelationComputer data.

    Integrates with the existing CorrelationComputer to check whether
    adding a new symbol would create excessive correlation concentration.
    """

    def __init__(
        self,
        computer: CorrelationComputer,
        config: CorrelationGateConfig = CorrelationGateConfig(),
    ) -> None:
        self._computer = computer
        self._config = config

    def should_allow(
        self,
        symbol: str,
        existing_symbols: Sequence[str],
    ) -> RiskDecision:
        """Check if adding symbol to portfolio is acceptable.

        Parameters
        ----------
        symbol : symbol to add
        existing_symbols : currently held symbols

        Returns
        -------
        RiskDecision — ALLOW or REJECT with violation details.
        """
        if not existing_symbols:
            return RiskDecision(action=RiskAction.ALLOW)

        # Check data sufficiency
        sym_returns = self._computer._returns.get(symbol, [])
        if len(sym_returns) < self._config.min_data_points:
            # Insufficient data → allow (don't block on missing data)
            return RiskDecision(action=RiskAction.ALLOW)

        # Check position correlation (new symbol vs existing)
        pos_corr = self._computer.position_correlation(symbol, existing_symbols)
        if pos_corr is not None and pos_corr > self._config.max_position_correlation:
            return RiskDecision(
                action=RiskAction.REJECT,
                violations=(RiskViolation(
                    code=RiskCode.MAX_GROSS,
                    scope=RiskScope.SYMBOL,
                    message=(
                        f"Position correlation {pos_corr:.2f} "
                        f"exceeds limit {self._config.max_position_correlation:.2f}"
                    ),
                    symbol=symbol,
                ),),
            )

        # Check portfolio average correlation (if we add this symbol)
        all_symbols = list(existing_symbols) + [symbol]
        avg_corr = self._computer.portfolio_avg_correlation(all_symbols)
        if avg_corr is not None and avg_corr > self._config.max_avg_correlation:
            return RiskDecision(
                action=RiskAction.REJECT,
                violations=(RiskViolation(
                    code=RiskCode.MAX_GROSS,
                    scope=RiskScope.PORTFOLIO,
                    message=(
                        f"Portfolio avg correlation {avg_corr:.2f} "
                        f"exceeds limit {self._config.max_avg_correlation:.2f}"
                    ),
                ),),
            )

        return RiskDecision(action=RiskAction.ALLOW)
