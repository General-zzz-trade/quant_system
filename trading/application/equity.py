"""Account equity used for sizing: a value older than two minutes, or not positive, sizes to zero."""
from __future__ import annotations

from decimal import Decimal

from trading.application.ports import Clock

EQUITY_MAX_AGE = 120.0


class EquitySource:
    def __init__(self, clock: Clock) -> None:
        self.value, self.at, self.clock = Decimal(0), 0.0, clock

    def update(self, equity: Decimal) -> None:
        self.value, self.at = Decimal(str(equity)), self.clock()

    def current(self) -> Decimal:
        if self.value <= 0 or self.clock() - self.at > EQUITY_MAX_AGE:
            return Decimal(0)  # stale equity sizes to zero
        return self.value


__all__ = ["EQUITY_MAX_AGE", "EquitySource"]
