"""Operator kill switch state."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KillState:
    """``active`` blocks new exposure; ``flatten`` also asks the runtime to close positions."""

    active: bool
    flatten: bool
    reason: str
