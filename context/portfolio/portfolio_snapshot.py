# context/portfolio/portfolio_snapshot.py
"""Portfolio snapshot — aggregated portfolio view."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ContextPortfolioSnapshot:
    """组合快照 — 汇总持仓、暴露、风险指标。"""
    total_equity: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    leverage: Decimal
    position_count: int
    long_exposure: Decimal = Decimal("0")
    short_exposure: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
