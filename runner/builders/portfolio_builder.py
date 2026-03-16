# runner/builders/portfolio_builder.py
"""Phase 3: portfolio and correlation — allocator, burn-in gate, correlation, attribution.

Extracted from LiveRunner._build_portfolio_and_correlation().
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_portfolio_and_correlation(config: Any) -> tuple:
    """Phase 3: portfolio allocator, burn-in gate, correlation, attribution.

    Returns (portfolio_allocator, correlation_computer, _update_correlation,
             attribution_tracker, correlation_gate).
    """
    # ── Portfolio Allocator (Direction 19) ──
    portfolio_allocator = None
    if config.enable_portfolio_risk:
        from portfolio.live_allocator import LivePortfolioAllocator, LiveAllocatorConfig
        portfolio_allocator = LivePortfolioAllocator(
            config=LiveAllocatorConfig(
                max_gross_leverage=config.max_gross_leverage,
                max_net_leverage=config.max_net_leverage,
                max_concentration=config.max_concentration,
            ),
        )
        logger.info(
            "Portfolio allocator enabled: gross=%.1f net=%.1f concentration=%.1f",
            config.max_gross_leverage, config.max_net_leverage,
            config.max_concentration,
        )

    # ── Burn-in Gate (Direction 14) ──
    if config.enable_burnin_gate and not config.testnet:
        from runner.preflight import BurninGate
        burnin_gate = BurninGate(report_path=config.burnin_report_path)
        burnin_check = burnin_gate.check(testnet=config.testnet)
        if not burnin_check.passed:
            raise RuntimeError(
                f"Burn-in gate FAILED: {burnin_check.message}\n"
                "Complete paper→shadow→testnet phases before production."
            )
        logger.info("Burn-in gate passed: %s", burnin_check.message)

    # ── CorrelationComputer (created early for on_snapshot) ──
    from risk.correlation_computer import CorrelationComputer
    correlation_computer = CorrelationComputer(window=60)

    def _update_correlation(snapshot: Any) -> None:
        markets = getattr(snapshot, "markets", {})
        for sym, mkt in markets.items():
            close = getattr(mkt, "close", None)
            if close is not None:
                correlation_computer.update(sym, float(close))

    # ── AttributionTracker ───────────────────────────────
    from attribution.tracker import AttributionTracker
    attribution_tracker = AttributionTracker()

    # ── CorrelationGate ──────────────────────────────────
    from risk.correlation_gate import CorrelationGate, CorrelationGateConfig
    correlation_gate = CorrelationGate(
        computer=correlation_computer,
        config=CorrelationGateConfig(max_avg_correlation=config.max_avg_correlation),
    )

    return (
        portfolio_allocator, correlation_computer, _update_correlation,
        attribution_tracker, correlation_gate,
    )
