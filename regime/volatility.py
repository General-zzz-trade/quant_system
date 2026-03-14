from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from .base import RegimeLabel


class VolatilityRegimeDetector:
    """Volatility regime based on rolling volatility.

    Expects features:
      - vol

    Returns values:
      - "low"
      - "mid"
      - "high"
    """

    name = "volatility"

    def __init__(self, *, low: float = 0.002, high: float = 0.01) -> None:
        self.low = float(low)
        self.high = float(high)

    def detect(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[RegimeLabel]:
        v = features.get("vol")
        if v is None:
            return None
        vol = abs(float(v))
        if vol < self.low:
            val = "low"
        elif vol > self.high:
            val = "high"
        else:
            val = "mid"

        # normalize into 0..1
        score = 0.0
        if vol <= self.low:
            score = 0.2
        elif vol >= self.high:
            score = 1.0
        else:
            score = 0.2 + 0.8 * ((vol - self.low) / max(self.high - self.low, 1e-12))

        return RegimeLabel(name=self.name, ts=ts, value=val, score=score, meta={"vol": vol})
