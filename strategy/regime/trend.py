from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Optional

from .base import RegimeLabel


class TrendRegimeDetector:
    """ADX-based trend regime detection.

    Uses close_vs_ma20, close_vs_ma50 for direction and adx_14 for strength.

    Returns values:
      - "strong_up":   MA20 > 0 and MA50 > 0 and ADX >= 25
      - "weak_up":     MA20 > 0 and MA50 > 0 and ADX < 25
      - "strong_down": MA20 < 0 and MA50 < 0 and ADX >= 25
      - "weak_down":   MA20 < 0 and MA50 < 0 and ADX < 25
      - "ranging":     MAs disagree on direction, or ADX < 15
    """

    name = "trend"

    def __init__(
        self,
        *,
        adx_strong: float = 25.0,
        adx_ranging: float = 15.0,
    ) -> None:
        self.adx_strong = float(adx_strong)
        self.adx_ranging = float(adx_ranging)

    def detect(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[RegimeLabel]:
        ma20_raw = features.get("close_vs_ma20")
        ma50_raw = features.get("close_vs_ma50")
        adx_raw = features.get("adx_14")

        if ma20_raw is None or ma50_raw is None:
            return None

        ma20 = float(ma20_raw)
        ma50 = float(ma50_raw)

        if not (math.isfinite(ma20) and math.isfinite(ma50)):
            return None

        adx = float(adx_raw) if adx_raw is not None and math.isfinite(adx_raw) else 0.0

        # Direction from MA signals
        up20 = ma20 > 0
        up50 = ma50 > 0
        agree = up20 == up50

        # Ranging: MAs disagree or ADX very low
        if not agree or adx < self.adx_ranging:
            val = "ranging"
            score = 0.2
        elif up20 and up50:
            if adx >= self.adx_strong:
                val = "strong_up"
                score = min(adx / 50.0, 1.0)
            else:
                val = "weak_up"
                score = adx / 50.0
        else:
            if adx >= self.adx_strong:
                val = "strong_down"
                score = min(adx / 50.0, 1.0)
            else:
                val = "weak_down"
                score = adx / 50.0

        meta: Dict[str, Any] = {
            "close_vs_ma20": ma20,
            "close_vs_ma50": ma50,
            "adx_14": adx,
        }

        return RegimeLabel(name=self.name, ts=ts, value=val, score=score, meta=meta)
