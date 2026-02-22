from __future__ import annotations

class StateError(Exception):
    """Base exception for state layer."""

class ReducerError(StateError):
    """Raised when a reducer cannot apply an event due to contract violation."""

class ProjectionError(StateError):
    """Raised when projector cannot produce a consistent snapshot."""
