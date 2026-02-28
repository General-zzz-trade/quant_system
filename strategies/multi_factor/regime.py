from __future__ import annotations

from enum import Enum
from typing import Optional

from strategies.multi_factor.feature_computer import MultiFactorFeatures


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOL = "high_vol"


def classify_regime(
    features: MultiFactorFeatures,
    atr_extreme_pct: float = 85.0,
    slope_threshold: float = 0.001,
) -> Optional[Regime]:
    if features.atr_percentile is None or features.ma_slope is None:
        return None
    if features.sma_fast is None or features.sma_slow is None:
        return None

    if features.atr_percentile > atr_extreme_pct:
        return Regime.HIGH_VOL

    if features.ma_slope > slope_threshold and features.sma_fast > features.sma_slow:
        return Regime.TRENDING_UP

    if features.ma_slope < -slope_threshold and features.sma_fast < features.sma_slow:
        return Regime.TRENDING_DOWN

    return Regime.RANGING
