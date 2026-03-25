# strategy/gates/types.py
"""Shared gate types — GateResult lives here to break runner ↔ strategy cycle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class GateResult:
    """Result of a single gate check."""
    allowed: bool
    scale: float = 1.0   # qty multiplier (1.0 = no change)
    reason: str = ""


class Gate(Protocol):
    """Protocol for order gates."""
    name: str

    def check(self, ev: Any, context: Dict[str, Any]) -> GateResult:
        ...
