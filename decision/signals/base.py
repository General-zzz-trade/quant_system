from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from decision.types import SignalResult


class SignalModel(Protocol):
    """Protocol for all signal models.

    Set ``experimental = True`` on research-only signals to mark them
    as non-production.  The attribute is optional — production signals
    need not declare it.
    """
    name: str

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        ...


@dataclass(frozen=True, slots=True)
class NullSignal:
    name: str = "null"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
