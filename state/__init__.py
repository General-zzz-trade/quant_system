"""State management — snapshots, projections, diffing, and versioning."""
from state.registry import StateProjector
from state.snapshot import StateSnapshot
from state.portfolio import PortfolioState
from state.risk import RiskState, RiskLimits
from state.diff import SnapshotDiff, compute_diff
from state.versioning import SchemaVersion, check_compatibility

__all__ = [
    "StateProjector",
    "StateSnapshot",
    "PortfolioState",
    "RiskState",
    "RiskLimits",
    "SnapshotDiff",
    "compute_diff",
    "SchemaVersion",
    "check_compatibility",
]
