# portfolio/risk_model
"""Portfolio risk models — covariance estimation, correlation, factor models.

NOTE: This module is EXPERIMENTAL and not wired into the production risk layer.
Production risk management uses:
- RiskAggregator with 6 active rules (risk/aggregator.py)
- DrawdownCircuitBreaker (risk/drawdown_breaker.py)
- CorrelationGate (risk/correlation_gate.py)

These risk models are available for research via scripts/backtest_portfolio.py.
"""
from portfolio.risk_model.base import RiskEstimate, RiskModel
from portfolio.risk_model.input import RiskModelInput
from portfolio.risk_model.registry import RiskModelRegistry

__all__ = [
    "RiskEstimate",
    "RiskModel",
    "RiskModelInput",
    "RiskModelRegistry",
]
