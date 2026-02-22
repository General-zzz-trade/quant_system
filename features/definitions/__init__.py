# features/definitions
"""Feature definitions — declarative feature specifications."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class FeatureDefinition:
    """特征定义。"""
    name: str
    category: str = "technical"       # technical / fundamental / ml
    lookback: int = 20
    dependencies: tuple[str, ...] = ()
    description: str = ""


# 标准技术特征
STANDARD_FEATURES: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("sma_20", "technical", 20, description="20期简单移动平均"),
    FeatureDefinition("ema_12", "technical", 12, description="12期指数移动平均"),
    FeatureDefinition("rsi_14", "technical", 14, description="14期RSI"),
    FeatureDefinition("atr_14", "technical", 14, description="14期ATR"),
    FeatureDefinition("bbands_20", "technical", 20, description="20期布林带"),
    FeatureDefinition("volume_sma_20", "technical", 20, description="20期成交量均线"),
)
