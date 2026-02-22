from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from .base import RegimeDetector, RegimeLabel


class CorrelationRegimeDetector:
    """Correlation regime based on a rolling correlation coefficient.

    Expects features:
      - corr (correlation)

    Returns values:
      - "neg"
      - "neutral"
      - "pos"
    """

    name = "correlation"

    def __init__(self, *, neg: float = -0.3, pos: float = 0.3) -> None:
        self.neg = float(neg)
        self.pos = float(pos)

    def detect(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[RegimeLabel]:
        c = features.get("corr")
        if c is None:
            return None
        c0 = float(c)
        if c0 < self.neg:
            val = "neg"
        elif c0 > self.pos:
            val = "pos"
        else:
            val = "neutral"
        score = min(abs(c0), 1.0)
        return RegimeLabel(name=self.name, ts=ts, value=val, score=score, meta={"corr": c0})
