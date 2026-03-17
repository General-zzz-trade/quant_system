from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from state._util import ensure_utc


def _check_finite(value: Optional[Decimal], name: str) -> None:
    """Raise ValueError if value is a Decimal NaN or Inf."""
    if value is not None and (value != value or abs(value) == Decimal("Inf")):
        raise ValueError(f"{name} must be finite, got {value}")


@dataclass(frozen=True, slots=True)
class PositionState:
    """Single-symbol position facts (SSOT). Updated only by fills."""

    symbol: str
    qty: Decimal = Decimal("0")
    avg_price: Optional[Decimal] = None
    last_price: Optional[Decimal] = None
    last_ts: Optional[datetime] = None

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @classmethod
    def empty(cls, symbol: str) -> "PositionState":
        return cls(symbol=symbol)

    def with_update(
        self,
        *,
        qty: Decimal,
        avg_price: Optional[Decimal],
        last_price: Optional[Decimal],
        ts: Optional[datetime],
    ) -> "PositionState":
        _check_finite(qty, "qty")
        _check_finite(avg_price, "avg_price")
        _check_finite(last_price, "last_price")
        return PositionState(
            symbol=self.symbol,
            qty=qty,
            avg_price=avg_price,
            last_price=last_price,
            last_ts=ensure_utc(ts) if ts is not None else self.last_ts,
        )
