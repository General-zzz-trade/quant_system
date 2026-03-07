# DEPRECATED: Python reducers are superseded by Rust fast path (RustStateStore / rust_pipeline_apply).
# Retained for: slow-path fallback, parity tests, and Reducer Protocol interface.
from state.reducers.base import ReducerResult, apply_one
from state.reducers.market import MarketReducer
from state.reducers.position import PositionReducer
from state.reducers.account import AccountReducer
from state.reducers.portfolio import PortfolioReducer
from state.reducers.risk import RiskReducer

__all__ = [
    "ReducerResult",
    "apply_one",
    "MarketReducer",
    "PositionReducer",
    "AccountReducer",
    "PortfolioReducer",
    "RiskReducer",
]
