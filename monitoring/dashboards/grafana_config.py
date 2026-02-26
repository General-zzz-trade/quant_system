"""Grafana dashboard JSON generator.

Produces complete Grafana dashboard JSON configs that can be imported via
the Grafana HTTP API or provisioned via JSON files.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monitoring.dashboards.panels import (
    GrafanaPanel,
    execution_latency_panel,
    pnl_panel,
    position_heatmap,
    strategy_row,
    system_health_panel,
)

logger = logging.getLogger(__name__)

_PANEL_TYPE_MAP = {
    "graph": "timeseries",
    "gauge": "gauge",
    "stat": "stat",
    "heatmap": "heatmap",
    "table": "table",
}


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Configuration for Grafana dashboard generation."""

    title: str = "Quant System Dashboard"
    refresh_interval: str = "10s"
    time_range: str = "24h"
    strategies: tuple[str, ...] = ()
    datasource: str = "Prometheus"
    uid: str = "quant-system-main"


class GrafanaDashboardGenerator:
    """Generate Grafana JSON dashboard configuration."""

    def __init__(self, config: DashboardConfig | None = None) -> None:
        self._config = config or DashboardConfig()
        self._panel_id = 0

    def generate(self) -> dict[str, Any]:
        """Generate complete Grafana dashboard JSON."""
        self._panel_id = 0
        panels: list[dict[str, Any]] = []

        panels.append(self._panel_to_dict(pnl_panel()))
        panels.append(self._panel_to_dict(position_heatmap()))
        panels.append(self._panel_to_dict(execution_latency_panel()))
        panels.append(self._panel_to_dict(system_health_panel()))

        row = 32
        for strategy in self._config.strategies:
            for p in strategy_row(strategy, row=row):
                panels.append(self._panel_to_dict(p))
            row += 8

        return {
            "dashboard": {
                "uid": self._config.uid,
                "title": self._config.title,
                "panels": panels,
                "refresh": self._config.refresh_interval,
                "time": {
                    "from": f"now-{self._config.time_range}",
                    "to": "now",
                },
                "schemaVersion": 39,
                "editable": True,
                "tags": ["quant", "trading"],
            },
        }

    def _panel_to_dict(self, panel: GrafanaPanel) -> dict[str, Any]:
        """Convert GrafanaPanel to Grafana JSON format."""
        self._panel_id += 1
        targets = []
        for idx, t in enumerate(panel.targets):
            target = {
                "refId": chr(65 + idx),
                "datasource": {"type": "prometheus", "uid": self._config.datasource},
                **t,
            }
            targets.append(target)

        result: dict[str, Any] = {
            "id": self._panel_id,
            "title": panel.title,
            "type": _PANEL_TYPE_MAP.get(panel.panel_type, panel.panel_type),
            "gridPos": panel.grid_pos,
            "targets": targets,
            "datasource": {
                "type": "prometheus",
                "uid": self._config.datasource,
            },
        }
        if panel.options:
            result["options"] = panel.options
        return result

    def save(self, path: str | Path) -> None:
        """Save dashboard JSON to file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.generate()
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Dashboard saved to %s", p)
