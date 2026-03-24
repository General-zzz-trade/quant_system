"""State management — Rust-native types + snapshots.

All state types are Rust PyO3 classes from _quant_hotpath.
Backward-compatible aliases (MarketState, PositionState, etc.) point to Rust types.
"""
from _quant_hotpath import (  # type: ignore[import-untyped]
    RustMarketState as MarketState,
    RustPositionState as PositionState,
    RustAccountState as AccountState,
    RustPortfolioState as PortfolioState,
    RustRiskState as RiskState,
    RustRiskLimits as RiskLimits,
)

from state.snapshot import StateSnapshot

__all__ = [
    "MarketState",
    "PositionState",
    "AccountState",
    "PortfolioState",
    "RiskState",
    "RiskLimits",
    "StateSnapshot",
]
