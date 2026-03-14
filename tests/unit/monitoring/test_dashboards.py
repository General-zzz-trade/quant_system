"""Tests for Grafana dashboard generation."""
from __future__ import annotations

import json


from monitoring.dashboards.panels import (
    execution_latency_panel,
    pnl_panel,
    position_heatmap,
    strategy_row,
    system_health_panel,
)
from monitoring.dashboards.grafana_config import (
    DashboardConfig,
    GrafanaDashboardGenerator,
)


# ── Panel tests ──────────────────────────────────────────────


class TestPanelFactories:
    def test_pnl_panel_type(self):
        panel = pnl_panel()
        assert panel.panel_type == "graph"
        assert panel.title == "PnL Curve"
        assert len(panel.targets) >= 1

    def test_position_heatmap_type(self):
        panel = position_heatmap()
        assert panel.panel_type == "heatmap"

    def test_execution_latency_type(self):
        panel = execution_latency_panel()
        assert panel.panel_type == "graph"
        assert len(panel.targets) == 3  # p50, p95, p99

    def test_system_health_type(self):
        panel = system_health_panel()
        assert panel.panel_type == "gauge"

    def test_custom_row_offset(self):
        panel = pnl_panel(row=42)
        assert panel.grid_pos["y"] == 42

    def test_strategy_row_returns_two_panels(self):
        panels = strategy_row("momentum", row=0)
        assert len(panels) == 2
        assert "momentum" in panels[0].title
        assert "momentum" in panels[1].title

    def test_strategy_row_has_correct_grid_positions(self):
        panels = strategy_row("arb", row=16)
        assert panels[0].grid_pos["y"] == 16
        assert panels[1].grid_pos["y"] == 16
        assert panels[0].grid_pos["x"] == 0
        assert panels[1].grid_pos["x"] == 6


# ── Generator tests ──────────────────────────────────────────


class TestGrafanaDashboardGenerator:
    def test_generate_produces_valid_structure(self):
        gen = GrafanaDashboardGenerator()
        result = gen.generate()
        assert "dashboard" in result
        dash = result["dashboard"]
        assert "title" in dash
        assert "panels" in dash
        assert "refresh" in dash
        assert "time" in dash
        assert dash["schemaVersion"] == 39

    def test_default_has_four_panels(self):
        gen = GrafanaDashboardGenerator()
        result = gen.generate()
        assert len(result["dashboard"]["panels"]) == 4

    def test_with_strategies_adds_panels(self):
        cfg = DashboardConfig(strategies=("momentum", "mean_revert"))
        gen = GrafanaDashboardGenerator(cfg)
        result = gen.generate()
        # 4 base + 2 per strategy * 2 strategies = 8
        assert len(result["dashboard"]["panels"]) == 8

    def test_panels_have_unique_ids(self):
        cfg = DashboardConfig(strategies=("alpha",))
        gen = GrafanaDashboardGenerator(cfg)
        result = gen.generate()
        ids = [p["id"] for p in result["dashboard"]["panels"]]
        assert len(ids) == len(set(ids))

    def test_panels_have_datasource(self):
        gen = GrafanaDashboardGenerator()
        result = gen.generate()
        for panel in result["dashboard"]["panels"]:
            assert "datasource" in panel

    def test_custom_config(self):
        cfg = DashboardConfig(
            title="Custom Dashboard",
            refresh_interval="30s",
            time_range="7d",
        )
        gen = GrafanaDashboardGenerator(cfg)
        result = gen.generate()
        assert result["dashboard"]["title"] == "Custom Dashboard"
        assert result["dashboard"]["refresh"] == "30s"
        assert result["dashboard"]["time"]["from"] == "now-7d"

    def test_save_writes_json_file(self, tmp_path):
        gen = GrafanaDashboardGenerator()
        out = tmp_path / "dashboard.json"
        gen.save(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert "dashboard" in data

    def test_generate_is_json_serializable(self):
        cfg = DashboardConfig(strategies=("a", "b"))
        gen = GrafanaDashboardGenerator(cfg)
        result = gen.generate()
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
