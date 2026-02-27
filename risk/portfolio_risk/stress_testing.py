"""Portfolio-level stress testing — wraps core StressEngine with portfolio context.

Provides scenario-based stress tests for portfolio risk management.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

from risk.stress import (
    AccountExposure,
    PositionExposure,
    PriceShock,
    StressEngine,
    StressScenario,
    StressThresholds,
    build_default_stress_scenarios,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Simplified portfolio snapshot for stress testing."""
    equity: Decimal
    cash: Decimal
    positions: Dict[str, Decimal]  # symbol → qty
    prices: Dict[str, Decimal]  # symbol → current price


@dataclass(frozen=True, slots=True)
class StressTestResult:
    """Result of a portfolio stress test."""
    scenario_name: str
    pre_equity: Decimal
    post_equity: Decimal
    pnl: Decimal
    drawdown_pct: float
    breaches: List[str]  # threshold violations


def _build_exposure(snapshot: PortfolioSnapshot) -> AccountExposure:
    """Convert PortfolioSnapshot to AccountExposure for StressEngine."""
    positions: Dict[str, PositionExposure] = {}
    for symbol, qty in snapshot.positions.items():
        price = snapshot.prices.get(symbol, Decimal("0"))
        positions[symbol] = PositionExposure(
            symbol=symbol,
            qty=qty,
            mark_price=price,
        )
    return AccountExposure(
        equity=snapshot.equity,
        balance=snapshot.cash,
        positions=positions,
    )


class PortfolioStressTester:
    """Runs stress tests on portfolio snapshots.

    Wraps the core StressEngine with portfolio-aware scenario generation
    and threshold checking.
    """

    def __init__(
        self,
        *,
        max_drawdown_pct: float = 0.20,
        min_equity_ratio: float = 0.50,
    ) -> None:
        self._max_drawdown_pct = Decimal(str(max_drawdown_pct))
        self._min_equity_ratio = min_equity_ratio
        self._custom_scenarios: List[StressScenario] = []

    def add_scenario(self, scenario: StressScenario) -> None:
        self._custom_scenarios.append(scenario)

    def run(
        self,
        snapshot: PortfolioSnapshot,
        *,
        include_defaults: bool = True,
    ) -> List[StressTestResult]:
        """Run stress tests on portfolio snapshot."""
        account = _build_exposure(snapshot)

        # Set minimum equity threshold based on current equity
        min_equity = snapshot.equity * Decimal(str(self._min_equity_ratio))
        thresholds = StressThresholds(
            max_drawdown_pct=self._max_drawdown_pct,
            min_equity=min_equity,
        )

        scenarios: List[StressScenario] = list(self._custom_scenarios)
        if include_defaults:
            symbols = list(snapshot.positions.keys())
            defaults = build_default_stress_scenarios(symbols=symbols)
            scenarios.extend(defaults)

        if not scenarios:
            return []

        engine = StressEngine(thresholds=thresholds)
        report = engine.run(
            account=account,
            scenarios=scenarios,
        )

        results: List[StressTestResult] = []
        for sr in report.results:
            breaches: List[str] = []
            dd = float(sr.drawdown_pct)
            if sr.drawdown_pct > self._max_drawdown_pct:
                breaches.append(
                    f"drawdown {dd:.1%} > {float(self._max_drawdown_pct):.1%}"
                )
            if sr.equity_after < min_equity:
                breaches.append(
                    f"equity {sr.equity_after} < min {min_equity}"
                )

            results.append(StressTestResult(
                scenario_name=sr.scenario,
                pre_equity=sr.equity_before,
                post_equity=sr.equity_after,
                pnl=sr.pnl,
                drawdown_pct=dd,
                breaches=breaches,
            ))

        if any(r.breaches for r in results):
            logger.warning(
                "Stress test breaches detected in %d scenarios",
                sum(1 for r in results if r.breaches),
            )

        return results

    @staticmethod
    def worst_case(
        results: List[StressTestResult],
    ) -> Optional[StressTestResult]:
        """Return the worst-case scenario by drawdown."""
        if not results:
            return None
        return max(results, key=lambda r: r.drawdown_pct)
