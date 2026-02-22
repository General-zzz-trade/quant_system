from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from decision.types import SignalResult


@dataclass(frozen=True, slots=True)
class CointegrationSignal:
    """Placeholder: pair trading needs multi-symbol state; neutral by default."""
    name: str = "cointegration"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        return SignalResult(symbol=symbol, side="flat", score=Decimal("0"), confidence=Decimal("0"))
