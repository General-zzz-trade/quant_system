"""State layer exceptions — aligned with core/errors.py hierarchy.

All state-layer exceptions inherit from ``core.errors.StateError`` so
callers can catch at the granularity they want.
"""
from __future__ import annotations

from infra.errors import StateError


class ReducerError(StateError):
    """Raised when a reducer cannot apply an event due to contract violation."""


class ProjectionError(StateError):
    """Raised when projector cannot produce a consistent snapshot."""


class SnapshotConsistencyError(StateError):
    """Raised when a snapshot fails internal consistency checks.

    Examples: negative balance, qty/price mismatch, missing required fields.
    """


class SchemaVersionError(StateError):
    """Raised when stored state has an incompatible schema version."""

    def __init__(self, stored: int, current: int) -> None:
        self.stored = stored
        self.current = current
        super().__init__(
            f"Schema version mismatch: stored={stored}, current={current}"
        )
