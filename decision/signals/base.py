from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional, Protocol

from decision.types import SignalResult


class SignalModel(Protocol):
    name: str

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        ...


@dataclass(frozen=True, slots=True)
class NullSignal:
    name: str = "null"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
