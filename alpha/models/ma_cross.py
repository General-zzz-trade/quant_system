from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from features.technical import sma
from alpha.base import Signal


@dataclass
class MACrossAlpha:
    """Moving average cross signal.

    Expects features to contain a list/series of closes under key "close".
    """

    fast: int = 20
    slow: int = 50
    name: str = "ma_cross"

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[Signal]:
        closes = features.get("close")
        if not closes or len(closes) < max(self.fast, self.slow) + 2:
            return None

        fast_ma = sma(closes, self.fast)
        slow_ma = sma(closes, self.slow)

        i = len(closes) - 1
        fast_prev, slow_prev = fast_ma[i - 1], slow_ma[i - 1]
        fast_curr, slow_curr = fast_ma[i], slow_ma[i]
        if fast_prev is None or slow_prev is None or fast_curr is None or slow_curr is None:
            return None

        prev_diff = fast_prev - slow_prev
        curr_diff = fast_curr - slow_curr

        if prev_diff <= 0.0 and curr_diff > 0.0:
            return Signal(symbol=symbol, ts=ts, side="long", strength=1.0, meta={"fast": self.fast, "slow": self.slow})
        if prev_diff >= 0.0 and curr_diff < 0.0:
            return Signal(symbol=symbol, ts=ts, side="short", strength=1.0, meta={"fast": self.fast, "slow": self.slow})
        return Signal(symbol=symbol, ts=ts, side="flat", strength=0.0, meta={"fast": self.fast, "slow": self.slow})
