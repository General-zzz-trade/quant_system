# portfolio/risk_model/stress/scenarios.py
"""Stress scenario definitions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ScenarioType(Enum):
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    FACTOR_SHOCK = "factor_shock"


@dataclass(frozen=True, slots=True)
class StressScenario:
    """压力测试场景。"""
    name: str
    scenario_type: ScenarioType
    shocks: Mapping[str, float]  # symbol/factor → shock magnitude
    description: str = ""


# 预定义加密市场压力场景
CRYPTO_CRASH_2022 = StressScenario(
    name="crypto_crash_2022",
    scenario_type=ScenarioType.HISTORICAL,
    shocks={"BTC": -0.50, "ETH": -0.60, "SOL": -0.80},
    description="2022年加密市场崩盘",
)

FLASH_CRASH = StressScenario(
    name="flash_crash",
    scenario_type=ScenarioType.HYPOTHETICAL,
    shocks={"BTC": -0.20, "ETH": -0.25, "SOL": -0.35},
    description="闪崩情景 — 突然下跌20-35%",
)

CORRELATION_SPIKE = StressScenario(
    name="correlation_spike",
    scenario_type=ScenarioType.HYPOTHETICAL,
    shocks={"BTC": -0.15, "ETH": -0.15, "SOL": -0.15},
    description="相关性飙升 — 所有资产同步下跌",
)
