from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from .base import RegimeLabel


class TrendRegimeDetector:
    """Simple trend regime based on fast/slow moving averages.

    Expects features:
      - ma_fast
      - ma_slow
      - atr (optional)

    Returns values:
      - "up"
      - "down"
      - "flat"
    """

    name = "trend"

    def __init__(self, *, threshold: float = 0.001) -> None:
        self.threshold = float(threshold)

    def detect(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[RegimeLabel]:
        fast = features.get("ma_fast")
        slow = features.get("ma_slow")
        if fast is None or slow is None:
            return None
        f = float(fast)
        s = float(slow)
        if s == 0.0:
            return None

        diff = (f - s) / s
        if diff > self.threshold:
            val = "up"
        elif diff < -self.threshold:
            val = "down"
        else:
            val = "flat"

        score = min(abs(diff) / max(self.threshold, 1e-12), 5.0) / 5.0
        return RegimeLabel(name=self.name, ts=ts, value=val, score=score, meta={"diff": diff})
