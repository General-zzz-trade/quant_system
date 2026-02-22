from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from .base import RegimeDetector, RegimeLabel


class LiquidityRegimeDetector:
    """Liquidity regime based on volume z-score.

    Expects features:
      - volume_z (z-score of volume)

    Returns values:
      - "thin"
      - "normal"
      - "thick"
    """

    name = "liquidity"

    def __init__(self, *, thin: float = -0.5, thick: float = 0.5) -> None:
        self.thin = float(thin)
        self.thick = float(thick)

    def detect(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[RegimeLabel]:
        z = features.get("volume_z")
        if z is None:
            return None
        z0 = float(z)
        if z0 < self.thin:
            val = "thin"
        elif z0 > self.thick:
            val = "thick"
        else:
            val = "normal"
        score = min(abs(z0) / 3.0, 1.0)
        return RegimeLabel(name=self.name, ts=ts, value=val, score=score, meta={"z": z0})
