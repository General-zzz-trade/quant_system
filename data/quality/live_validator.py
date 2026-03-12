"""Live market event validator used by market-data runtime."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True, slots=True)
class LiveBarValidator:
    """Lightweight validator for live market events.

    The market-data runtime only needs a fast yes/no gate before dispatching
    events into the live engine. This validator intentionally stays small:
    it rejects obviously invalid market payloads while allowing partial event
    shapes that appear across adapters and replay sources.
    """

    require_positive_close: bool = True

    def validate(self, event: Any) -> bool:
        market = getattr(event, "market", None)
        if market is None:
            return True

        close = self._as_float(getattr(market, "close", None))
        high = self._as_float(getattr(market, "high", None))
        low = self._as_float(getattr(market, "low", None))
        open_ = self._as_float(getattr(market, "open", None))

        if self.require_positive_close and close is not None and close <= 0:
            return False
        if high is not None and low is not None and high < low:
            return False
        if high is not None and open_ is not None and high < open_:
            return False
        if high is not None and close is not None and high < close:
            return False
        if low is not None and open_ is not None and low > open_:
            return False
        if low is not None and close is not None and low > close:
            return False
        return True

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None
