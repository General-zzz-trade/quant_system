"""Tests for live validation dashboard and slippage analyzer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestDashboardImport:

    def test_import(self):
        from monitoring.dashboard import build_dashboard, print_terminal, print_markdown
        assert callable(build_dashboard)

    def test_build_dashboard_no_data(self, monkeypatch):
        """Dashboard should work with missing data files."""
        from monitoring import dashboard as mod
        monkeypatch.setattr(mod, "HEALTH_STATUS", Path("/nonexistent/health.json"))
        monkeypatch.setattr(mod, "IC_HEALTH", Path("/nonexistent/ic.json"))
        monkeypatch.setattr(mod, "TRACK_RECORD", Path("/nonexistent/track.json"))
        monkeypatch.setattr(mod, "SIGNAL_RECONCILE", Path("/nonexistent/reconcile.json"))

        dashboard = mod.build_dashboard()
        assert dashboard["days_running"] == 0
        assert dashboard["total_trades"] == 0
        assert not dashboard["all_pass"]


class TestParseTrackRecord:

    def test_empty_track(self):
        from monitoring.dashboard import _parse_track_record
        result = _parse_track_record({"daily": {}})
        assert result["days"] == 0
        assert result["total_pnl"] == 0

    def test_basic_metrics(self):
        from monitoring.dashboard import _parse_track_record
        data = {
            "daily": {
                "2026-03-01": {
                    "total_pnl_usd": 100.0,
                    "total_trades": 5,
                    "max_drawdown": 2.0,
                    "symbols": {
                        "BTCUSDT": {"bars": 24, "signals": {"long": 10, "short": 5, "flat": 9},
                                    "trades": 3, "pnl_usd": 80.0},
                    },
                },
                "2026-03-02": {
                    "total_pnl_usd": -50.0,
                    "total_trades": 3,
                    "max_drawdown": 5.0,
                    "symbols": {
                        "BTCUSDT": {"bars": 24, "signals": {"long": 5, "short": 10, "flat": 9},
                                    "trades": 2, "pnl_usd": -30.0},
                    },
                },
            }
        }
        result = _parse_track_record(data)
        assert result["days"] == 2
        assert result["total_pnl"] == 50.0
        assert result["total_trades"] == 8
        assert result["max_dd"] == 5.0
        assert "BTCUSDT" in result["per_runner"]
        assert result["per_runner"]["BTCUSDT"]["trades"] == 5

    def test_sharpe_computation(self):
        from monitoring.dashboard import _parse_track_record
        # 30 days with varying positive returns
        daily = {}
        for i in range(30):
            daily[f"2026-03-{i+1:02d}"] = {
                "total_pnl_usd": 10.0 + i * 0.5,  # varying to get nonzero std
                "total_trades": 2,
                "max_drawdown": 0.5,
                "symbols": {},
            }
        result = _parse_track_record({"daily": daily})
        assert result["sharpe_30d"] > 0


class TestChecklist:

    def test_all_pass(self):
        from monitoring.dashboard import build_dashboard
        # Can't easily mock all paths; just verify structure
        dashboard = build_dashboard()
        assert "checklist" in dashboard
        assert "all_pass" in dashboard
        assert isinstance(dashboard["checklist"], dict)

    def test_checklist_criteria(self):
        """Verify checklist has the expected items."""
        from monitoring.dashboard import build_dashboard
        dashboard = build_dashboard()
        expected_keys = [
            "Sharpe > 1.0 (30d rolling)",
            "Signal match rate > 85%",
            "MaxDD < 20%",
            "At least 50 trades",
        ]
        for key in expected_keys:
            assert key in dashboard["checklist"], f"Missing checklist item: {key}"


class TestICHealth:

    def test_empty(self):
        from monitoring.dashboard import _check_ic_health
        assert _check_ic_health(None) == {}

    def test_parse_list(self):
        from monitoring.dashboard import _check_ic_health
        data = {"models": [
            {"model": "BTC_4h", "overall_status": "GREEN"},
            {"model": "ETH_1h", "overall_status": "YELLOW"},
        ]}
        result = _check_ic_health(data)
        assert result["BTC_4h"] == "GREEN"
        assert result["ETH_1h"] == "YELLOW"

    def test_parse_dict(self):
        from monitoring.dashboard import _check_ic_health
        data = {"models": {"BTC_4h": {"status": "GREEN"}}}
        result = _check_ic_health(data)
        assert result["BTC_4h"] == "GREEN"


class TestHealthStatus:

    def test_healthy(self):
        from monitoring.dashboard import _check_health_status
        data = {"checks": {"svc1": {"problems": []}, "svc2": {"problems": []}}}
        assert _check_health_status(data) == "HEALTHY"

    def test_issues(self):
        from monitoring.dashboard import _check_health_status
        data = {"checks": {"svc1": {"problems": ["stale data"]}}}
        assert "ISSUES" in _check_health_status(data)


class TestSignalReconcile:

    def test_missing(self):
        from monitoring.dashboard import _check_signal_reconcile
        assert _check_signal_reconcile(None) == -1.0

    def test_rate(self):
        from monitoring.dashboard import _check_signal_reconcile
        assert _check_signal_reconcile({"match_rate": 92.5}) == 92.5


class TestSlippageAnalyzer:

    def test_import(self):
        from monitoring.slippage import parse_log, compute_stats

    def test_missing_log(self, tmp_path):
        from monitoring.slippage import parse_log
        result = parse_log(tmp_path / "nonexistent.log", 24)
        assert "error" in result

    def test_compute_stats_empty(self):
        from monitoring.slippage import compute_stats
        result = compute_stats({"fills": [], "maker_fills": 0, "market_fallbacks": 0})
        assert result["n_fills"] == 0
        assert result["maker_fill_rate"] == 0

    def test_compute_stats_basic(self):
        from monitoring.slippage import compute_stats
        fills = [
            {"symbol": "BTCUSDT", "slippage_bps": 1.5},
            {"symbol": "BTCUSDT", "slippage_bps": -0.5},
            {"symbol": "ETHUSDT", "slippage_bps": 2.0},
        ]
        result = compute_stats({"fills": fills, "maker_fills": 8, "market_fallbacks": 2})
        assert result["n_fills"] == 3
        assert abs(result["avg_slippage_bps"] - 1.0) < 1e-6
        assert result["maker_fill_rate"] == 80.0
        assert "BTCUSDT" in result["per_symbol"]
        assert "ETHUSDT" in result["per_symbol"]

    def test_cost_comparison(self):
        from monitoring.slippage import compute_stats
        # Very low slippage → BETTER than backtest
        fills = [{"symbol": "BTCUSDT", "slippage_bps": 0.5} for _ in range(10)]
        result = compute_stats({"fills": fills, "maker_fills": 10, "market_fallbacks": 0})
        assert result["cost_vs_assumption"] == "BETTER"

        # High slippage → WORSE
        fills2 = [{"symbol": "BTCUSDT", "slippage_bps": 10.0} for _ in range(10)]
        result2 = compute_stats({"fills": fills2, "maker_fills": 0, "market_fallbacks": 10})
        assert result2["cost_vs_assumption"] == "WORSE"
