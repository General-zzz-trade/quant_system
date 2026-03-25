"""Base signal types — no dependencies on decision/ to avoid circular imports."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SignalResult:
    """Lightweight signal result (strategy-local, avoids decision.types circular)."""
    symbol: str
    side: str
    score: Decimal
    confidence: Decimal = Decimal("0")


class SignalModel(Protocol):
    """Protocol for all signal models."""
    name: str

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        ...


@dataclass(frozen=True, slots=True)
class NullSignal:
    name: str = "null"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))


@dataclass(frozen=True)
class Signal:
    """A minimal trading signal.

    side: "long" | "short" | "flat"
    strength: 0..1 continuous
    """
    symbol: str
    ts: Any  # datetime
    side: str
    strength: float = 1.0
    meta: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.meta is None:
            object.__setattr__(self, "meta", {})
