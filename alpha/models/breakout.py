from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from ..base import Signal


@dataclass
class BreakoutAlpha:
    """Simple breakout signal.

    Expects features to contain "high" and "low" arrays.
    A breakout is detected when close exceeds recent high (long) or recent low (short).
    """

    lookback: int = 50
    name: str = "breakout"

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[Signal]:
        high = features.get("high")
        low = features.get("low")
        close = features.get("close")
        if not high or not low or not close:
            return None
        if len(close) < self.lookback + 1:
            return None

        window_high = max(float(x) for x in high[-self.lookback - 1 : -1])
        window_low = min(float(x) for x in low[-self.lookback - 1 : -1])
        c = float(close[-1])

        if c > window_high:
            return Signal(symbol=symbol, ts=ts, side="long", strength=1.0, meta={"lookback": self.lookback})
        if c < window_low:
            return Signal(symbol=symbol, ts=ts, side="short", strength=1.0, meta={"lookback": self.lookback})
        return Signal(symbol=symbol, ts=ts, side="flat", strength=0.0, meta={"lookback": self.lookback})
