"""Alpha model protocol + Signal data class.

Signal is now canonical in strategy/signals/base.py.
"""
from __future__ import annotations

from strategy.signals.base import Signal  # noqa: F401
from typing import Optional, Protocol
from datetime import datetime


class AlphaModel(Protocol):
    """Protocol for alpha models that produce signals."""
    name: str

    def predict(self, *, symbol: str, ts: datetime,
                features: dict[str, float]) -> Optional[Signal]: ...
