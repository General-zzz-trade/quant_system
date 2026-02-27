"""Grafana dashboard panel definitions.

Each factory function returns a ``GrafanaPanel`` representing a single
Grafana panel backed by Prometheus queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class GrafanaPanel:
    """Single Grafana dashboard panel definition."""

    title: str
    panel_type: str  # "graph" | "gauge" | "stat" | "heatmap" | "table"
    targets: tuple[dict[str, Any], ...]  # Prometheus queries
    grid_pos: dict[str, int]  # x, y, w, h
    options: dict[str, Any] = field(default_factory=dict)


# ── Panel factories ──────────────────────────────────────────


def pnl_panel(*, row: int = 0) -> GrafanaPanel:
    """PnL curve panel."""
    return GrafanaPanel(
        title="PnL Curve",
        panel_type="graph",
        targets=(
            {"expr": "portfolio_equity", "legendFormat": "equity"},
            {"expr": "portfolio_realized_pnl", "legendFormat": "realized_pnl"},
        ),
        grid_pos={"x": 0, "y": row, "w": 12, "h": 8},
        options={"tooltip": {"mode": "multi"}, "legend": {"displayMode": "table"}},
    )


def position_heatmap(*, row: int = 8) -> GrafanaPanel:
    """Position size heatmap across symbols."""
    return GrafanaPanel(
        title="Position Heatmap",
        panel_type="heatmap",
        targets=(
            {"expr": "quant_position_qty", "legendFormat": "{{symbol}}"},
        ),
        grid_pos={"x": 0, "y": row, "w": 12, "h": 8},
        options={"color": {"scheme": "RdYlGn"}, "yAxis": {"unit": "short"}},
    )


def execution_latency_panel(*, row: int = 16) -> GrafanaPanel:
    """Execution latency percentiles."""
    return GrafanaPanel(
        title="Execution Latency",
        panel_type="graph",
        targets=(
            {
                "expr": 'histogram_quantile(0.50, rate(quant_execution_latency_bucket[5m]))',
                "legendFormat": "p50",
            },
            {
                "expr": 'histogram_quantile(0.95, rate(quant_execution_latency_bucket[5m]))',
                "legendFormat": "p95",
            },
            {
                "expr": 'histogram_quantile(0.99, rate(quant_execution_latency_bucket[5m]))',
                "legendFormat": "p99",
            },
        ),
        grid_pos={"x": 0, "y": row, "w": 12, "h": 8},
        options={"tooltip": {"mode": "multi"}, "fieldConfig": {"defaults": {"unit": "ms"}}},
    )


def system_health_panel(*, row: int = 24) -> GrafanaPanel:
    """System health gauges (CPU, memory, connections)."""
    return GrafanaPanel(
        title="System Health",
        panel_type="gauge",
        targets=(
            {"expr": "process_cpu_seconds_total", "legendFormat": "cpu"},
            {"expr": "process_resident_memory_bytes", "legendFormat": "memory"},
            {"expr": "quant_ws_connected", "legendFormat": "ws_connected"},
        ),
        grid_pos={"x": 0, "y": row, "w": 12, "h": 8},
        options={
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "orientation": "horizontal",
        },
    )


def strategy_row(strategy_name: str, *, row: int) -> list[GrafanaPanel]:
    """Generate panels for a single strategy."""
    return [
        GrafanaPanel(
            title=f"{strategy_name} - PnL",
            panel_type="graph",
            targets=(
                {
                    "expr": f'quant_strategy_pnl{{strategy="{strategy_name}"}}',
                    "legendFormat": "pnl",
                },
            ),
            grid_pos={"x": 0, "y": row, "w": 6, "h": 8},
        ),
        GrafanaPanel(
            title=f"{strategy_name} - Signals",
            panel_type="graph",
            targets=(
                {
                    "expr": f'quant_strategy_signal{{strategy="{strategy_name}"}}',
                    "legendFormat": "{{symbol}}",
                },
            ),
            grid_pos={"x": 6, "y": row, "w": 6, "h": 8},
        ),
    ]
