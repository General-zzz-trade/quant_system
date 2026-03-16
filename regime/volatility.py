from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

from .base import RegimeLabel


class VolatilityRegimeDetector:
    """Percentile-based dynamic volatility regime classification.

    Uses parkinson_vol, vol_of_vol, and bb_width_20 from RustFeatureEngine
    to classify the current volatility regime into one of four buckets:
      - "low_vol":    parkinson_vol below 25th percentile
      - "normal_vol": 25th-75th percentile
      - "high_vol":   75th-95th percentile
      - "crisis":     vol_of_vol above 95th percentile

    Maintains a rolling 720-bar history for percentile calculation.
    Falls back to static thresholds if fewer than 30 bars of history.
    """

    name = "volatility"

    def __init__(self, *, window: int = 720, min_bars: int = 30) -> None:
        self.window = int(window)
        self.min_bars = int(min_bars)
        self._vol_history: deque[float] = deque(maxlen=self.window)
        self._vov_history: deque[float] = deque(maxlen=self.window)

    def detect(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[RegimeLabel]:
        parkinson = features.get("parkinson_vol")
        vol_of_vol = features.get("vol_of_vol")
        bb_width = features.get("bb_width_20")

        # Need at least parkinson_vol to classify
        if parkinson is None or not math.isfinite(parkinson):
            return None

        vol = abs(float(parkinson))
        vov = abs(float(vol_of_vol)) if vol_of_vol is not None and math.isfinite(vol_of_vol) else None

        self._vol_history.append(vol)
        if vov is not None:
            self._vov_history.append(vov)

        if len(self._vol_history) < self.min_bars:
            return None

        # Compute percentiles on sorted history
        vol_sorted = sorted(self._vol_history)
        n = len(vol_sorted)
        p25 = vol_sorted[int(n * 0.25)]
        p75 = vol_sorted[int(n * 0.75)]
        p95 = vol_sorted[int(n * 0.95)]

        # Crisis detection: vol_of_vol above 95th percentile
        is_crisis = False
        if vov is not None and len(self._vov_history) >= self.min_bars:
            vov_sorted = sorted(self._vov_history)
            vov_p95 = vov_sorted[int(len(vov_sorted) * 0.95)]
            is_crisis = vov > vov_p95

        if is_crisis:
            val = "crisis"
            score = 1.0
        elif vol > p95:
            val = "high_vol"
            score = 0.9
        elif vol > p75:
            val = "high_vol"
            score = 0.7
        elif vol < p25:
            val = "low_vol"
            score = 0.2
        else:
            val = "normal_vol"
            # Normalize within the normal band
            band = max(p75 - p25, 1e-12)
            score = 0.3 + 0.4 * ((vol - p25) / band)

        meta: Dict[str, Any] = {
            "parkinson_vol": vol,
            "p25": p25,
            "p75": p75,
            "p95": p95,
            "bars": n,
        }
        if vov is not None:
            meta["vol_of_vol"] = vov
        if bb_width is not None and math.isfinite(bb_width):
            meta["bb_width_20"] = float(bb_width)

        return RegimeLabel(name=self.name, ts=ts, value=val, score=score, meta=meta)
