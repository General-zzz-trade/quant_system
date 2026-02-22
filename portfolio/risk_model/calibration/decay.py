# portfolio/risk_model/calibration/decay.py
"""Decay schemes for weighting historical observations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class DecayType(Enum):
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    STEP = "step"


@dataclass(frozen=True, slots=True)
class DecayScheme:
    """衰减方案。"""
    decay_type: DecayType = DecayType.EXPONENTIAL
    halflife: int = 30

    def weights(self, n: int) -> list[float]:
        """为 n 个观测值生成权重（最近的权重最大）。"""
        if n <= 0:
            return []
        if self.decay_type == DecayType.UNIFORM:
            return [1.0 / n] * n
        elif self.decay_type == DecayType.EXPONENTIAL:
            lam = math.log(2) / max(self.halflife, 1)
            raw = [math.exp(-lam * (n - 1 - i)) for i in range(n)]
        elif self.decay_type == DecayType.LINEAR:
            raw = [float(i + 1) for i in range(n)]
        elif self.decay_type == DecayType.STEP:
            raw = [1.0 if i >= n - self.halflife else 0.5 for i in range(n)]
        else:
            raw = [1.0] * n
        total = sum(raw)
        return [w / total for w in raw] if total > 0 else [1.0 / n] * n
