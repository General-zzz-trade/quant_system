from __future__ import annotations

class DecisionError(RuntimeError):
    """Base error for decision layer."""


class PolicyViolation(DecisionError):
    """Raised when decision violates a configured policy/constraint."""
