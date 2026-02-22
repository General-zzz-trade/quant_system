# portfolio/risk_model/factor/definitions.py
"""Standard factor definitions for crypto markets."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class FactorType(Enum):
    """因子类型。"""
    MARKET = "market"
    SIZE = "size"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    MEAN_REVERSION = "mean_reversion"


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    """因子定义。"""
    name: str
    factor_type: FactorType
    description: str = ""
    lookback: int = 30


# 标准加密因子
CRYPTO_FACTORS: tuple[FactorDefinition, ...] = (
    FactorDefinition("market", FactorType.MARKET, "整体市场因子", 1),
    FactorDefinition("size", FactorType.SIZE, "市值因子", 30),
    FactorDefinition("momentum", FactorType.MOMENTUM, "动量因子", 20),
    FactorDefinition("volatility", FactorType.VOLATILITY, "波动率因子", 30),
    FactorDefinition("liquidity", FactorType.LIQUIDITY, "流动性因子", 30),
)
