# monitoring/dashboards
"""Monitoring dashboards — metrics visualization config."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class PanelConfig:
    """仪表盘面板配置。"""
    title: str
    metric_names: tuple[str, ...]
    panel_type: str = "line"  # line / gauge / table / heatmap


@dataclass(frozen=True)
class DashboardConfig:
    """仪表盘配置。"""
    name: str
    panels: tuple[PanelConfig, ...]
    refresh_seconds: int = 5


# 预定义仪表盘
TRADING_DASHBOARD = DashboardConfig(
    name="trading",
    panels=(
        PanelConfig("PnL", ("realized_pnl", "unrealized_pnl", "total_pnl")),
        PanelConfig("Positions", ("position_count", "total_exposure"), "gauge"),
        PanelConfig("Orders", ("orders_submitted", "orders_filled", "orders_rejected")),
        PanelConfig("Risk", ("portfolio_var", "max_drawdown", "sharpe_ratio"), "gauge"),
    ),
)
