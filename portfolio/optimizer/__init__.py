# portfolio/optimizer
"""Portfolio optimization — QP/SOCP solvers, Black-Litterman, constraints.

NOTE: This module is EXPERIMENTAL and not wired into the production gate chain.
Production position sizing uses:
- LivePortfolioAllocator (portfolio/live_allocator.py) via PortfolioAllocatorGate
- RegimePositionSizer (portfolio/regime_sizer.py) via RegimeSizerGate

These optimizer modules are available for research and backtesting via
scripts/backtest_portfolio.py and research/walk_forward_optimizer.py.
"""
from portfolio.optimizer.base import Optimizer, OptimizationResult
from portfolio.optimizer.constraints import (
    FullInvestmentConstraint,
    LongOnlyConstraint,
    MaxWeightConstraint,
    OptConstraint,
)
from portfolio.optimizer.exceptions import (
    ConvergenceError,
    InfeasibleError,
    NumericalError,
    OptimizationError,
)
from portfolio.optimizer.input import OptimizationInput
from portfolio.optimizer.objectives import MaxSharpe, MinVariance, Objective, RiskParity

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "FullInvestmentConstraint",
    "LongOnlyConstraint",
    "MaxWeightConstraint",
    "OptConstraint",
    "ConvergenceError",
    "InfeasibleError",
    "NumericalError",
    "OptimizationError",
    "OptimizationInput",
    "MaxSharpe",
    "MinVariance",
    "Objective",
    "RiskParity",
]
