# tests/unit/scripts/test_ops_tools.py
"""Tests for ops tools: shadow_mode_check, ops_dashboard, pre_live_checklist."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch



# ── shadow_mode_check ──────────────────────────────────────

class TestShadowModeCheck:
    def test_shadow_report_dataclass(self):
        from scripts.ops.shadow_mode_check import ShadowReport
        r = ShadowReport(window_hours=24)
        assert r.window_hours == 24
        assert r.signals_long == 0
        assert r.trade_attempts == 0

    def test_parse_log_entries_empty(self, tmp_path):
        from scripts.ops.shadow_mode_check import parse_log_entries
        log = tmp_path / "test.log"
        log.write_text("")
        report = parse_log_entries(str(log), hours=24)
        assert report.log_lines_parsed == 0

    def test_parse_log_entries_missing_file(self):
        from scripts.ops.shadow_mode_check import parse_log_entries
        report = parse_log_entries("/nonexistent/file.log", hours=24)
        assert report.log_lines_parsed == 0

    def test_parse_bar_lines(self, tmp_path):
        from scripts.ops.shadow_mode_check import parse_log_entries
        log = tmp_path / "test.log"
        log.write_text(textwrap.dedent("""\
            2099-01-01 10:00:00,000 INFO __main__: WS ETHUSDT bar 201: $2100.0 z=-0.223 sig=0 hold=1 regime=active
            2099-01-01 11:00:00,000 INFO __main__: WS ETHUSDT bar 202: $2110.0 z=+0.500 sig=1 hold=1 regime=active
            2099-01-01 12:00:00,000 INFO __main__: WS ETHUSDT bar 203: $2090.0 z=-0.800 sig=-1 hold=1 regime=quiet
            2099-01-01 12:30:00,000 INFO __main__: HEARTBEAT cycle=6 signals={'ETHUSDT': -1}
        """))
        report = parse_log_entries(str(log), hours=999999)
        assert report.total_bars == 3
        assert report.signals_long == 1
        assert report.signals_short == 1
        assert report.signals_flat == 1
        assert report.heartbeats == 1
        assert report.regimes == {"active": 2, "quiet": 1}
        assert report.symbol_bars == {"ETHUSDT": 3}

    def test_parse_trade_lines(self, tmp_path):
        from scripts.ops.shadow_mode_check import parse_log_entries
        log = tmp_path / "trades.log"
        t1 = ("2099-01-01 10:00:00,000 INFO __main__: WS ETHUSDT bar 201:"
              " $2100.0 z=-0.5 sig=-1 hold=1"
              " TRADE={'side': 'sell', 'qty': 0.48,"
              " 'result': {'status': 'ok'}}")
        t2 = ("2099-01-01 11:00:00,000 INFO __main__: WS ETHUSDT bar 202:"
              " $2100.0 z=-0.5 sig=-1 hold=2"
              " TRADE={'side': 'sell', 'qty': 0.48,"
              " 'result': {'status': 'error', 'retCode': 110007}}")
        log.write_text(t1 + "\n" + t2 + "\n")
        report = parse_log_entries(str(log), hours=999999)
        assert report.trade_attempts == 2
        assert report.trade_errors == 1
        assert report.trade_success_rate == 0.5

    def test_parse_combo_signals(self, tmp_path):
        from scripts.ops.shadow_mode_check import parse_log_entries
        log = tmp_path / "combo.log"
        line = ("2099-01-01 10:00:00,000 INFO __main__: WS ETHUSDT bar 201:"
                " $2100.0 z=-1.0 sig=-1 hold=1"
                " COMBO={'from': 0, 'to': -1,"
                " 'result': {'status': 'ok'}}")
        log.write_text(line + "\n")
        report = parse_log_entries(str(log), hours=999999)
        assert report.combo_signals == 1

    def test_window_filtering(self, tmp_path):
        """Lines outside the time window should be excluded."""
        from scripts.ops.shadow_mode_check import parse_log_entries
        log = tmp_path / "old.log"
        log.write_text(textwrap.dedent("""\
            2020-01-01 10:00:00,000 INFO __main__: WS ETHUSDT bar 201: $2100.0 z=0.0 sig=0 hold=1
            2099-01-01 10:00:00,000 INFO __main__: WS ETHUSDT bar 202: $2100.0 z=0.0 sig=0 hold=1
        """))
        report = parse_log_entries(str(log), hours=24)
        assert report.total_bars == 1

    def test_signal_rate_property(self):
        from scripts.ops.shadow_mode_check import ShadowReport
        r = ShadowReport(
            total_bars=10, signals_long=3, signals_short=2, signals_flat=5
        )
        assert r.signal_rate == 0.5

    def test_signal_rate_zero_bars(self):
        from scripts.ops.shadow_mode_check import ShadowReport
        r = ShadowReport(total_bars=0)
        assert r.signal_rate == 0.0

    def test_trade_success_rate_no_trades(self):
        from scripts.ops.shadow_mode_check import ShadowReport
        r = ShadowReport()
        assert r.trade_success_rate == 0.0

    def test_format_report_issues(self):
        """Report with no bars should show issues."""
        from scripts.ops.shadow_mode_check import ShadowReport, format_report
        report = ShadowReport(window_hours=24)
        text = format_report(report)
        assert "No bars processed" in text

    def test_format_report_healthy(self):
        from scripts.ops.shadow_mode_check import ShadowReport, format_report
        report = ShadowReport(
            window_hours=24,
            total_bars=100,
            signals_long=20,
            signals_short=15,
            signals_flat=65,
            heartbeats=24,
            regimes={"active": 80, "quiet": 20},
            symbol_bars={"ETHUSDT": 50, "BTCUSDT": 50},
        )
        text = format_report(report)
        assert "STATUS: OK" in text
        assert "ETHUSDT" in text

    def test_multi_symbol_bar_parsing(self, tmp_path):
        from scripts.ops.shadow_mode_check import parse_log_entries
        log = tmp_path / "multi.log"
        log.write_text(textwrap.dedent("""\
            2099-01-01 10:00:00,000 INFO __main__: WS ETHUSDT bar 201: $2100.0 z=-0.2 sig=0 hold=1
            2099-01-01 10:00:00,000 INFO __main__: WS BTCUSDT bar 201: $84000.0 z=+0.1 sig=1 hold=1
            2099-01-01 10:15:00,000 INFO __main__: WS ETHUSDT_15m bar 801: $2101.0 z=-0.5 sig=-1 hold=1
        """))
        report = parse_log_entries(str(log), hours=999999)
        assert report.total_bars == 3
        assert report.symbol_bars["ETHUSDT"] == 1
        assert report.symbol_bars["BTCUSDT"] == 1
        assert report.symbol_bars["ETHUSDT_15m"] == 1


# ── ops_dashboard ──────────────────────────────────────────

class TestOpsDashboard:
    def test_check_active_host_services(self, monkeypatch):
        from scripts.ops.ops_dashboard import check_active_host_services

        states = {
            "bybit-alpha.service": "active",
            "bybit-mm.service": "inactive",
        }
        monkeypatch.setattr(
            "scripts.ops.ops_dashboard.check_service_health",
            lambda service_name: states[service_name],
        )

        result = check_active_host_services()

        assert result == states

    def test_check_burnin_status_no_file(self, monkeypatch):
        from scripts.ops.ops_dashboard import check_burnin_status
        monkeypatch.setattr(Path, "exists", lambda self: False)
        result = check_burnin_status()
        assert isinstance(result, dict)
        assert result.get("status") == "not_started"

    def test_check_model_versions_with_data(self, tmp_path):
        from scripts.ops.ops_dashboard import check_model_versions
        m1 = tmp_path / "ETHUSDT_gate_v2"
        m1.mkdir()
        (m1 / "config.json").write_text(json.dumps({
            "horizon": 24, "trained_at": "2026-03-01", "features": ["a", "b"],
        }))
        m2 = tmp_path / "BTCUSDT_gate_v2"
        m2.mkdir()
        (m2 / "config.json").write_text(json.dumps({"horizon": 96}))
        result = check_model_versions(str(tmp_path))
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert "ETHUSDT_gate_v2" in names

    def test_check_model_versions_default(self):
        from scripts.ops.ops_dashboard import check_model_versions
        result = check_model_versions()
        assert isinstance(result, list)

    def test_check_model_versions_empty_dir(self, tmp_path):
        from scripts.ops.ops_dashboard import check_model_versions
        d = tmp_path / "empty"
        d.mkdir()
        result = check_model_versions(str(d))
        assert result == []

    def test_check_model_versions_nonexistent(self, tmp_path):
        from scripts.ops.ops_dashboard import check_model_versions
        result = check_model_versions(str(tmp_path / "nope"))
        assert result == []

    def test_check_model_versions_bad_json(self, tmp_path):
        from scripts.ops.ops_dashboard import check_model_versions
        m = tmp_path / "BAD"
        m.mkdir()
        (m / "config.json").write_text("not json")
        result = check_model_versions(str(tmp_path))
        assert len(result) == 1
        assert result[0]["trained"] == "error"

    def test_check_model_versions_no_config(self, tmp_path):
        from scripts.ops.ops_dashboard import check_model_versions
        m = tmp_path / "NO_CFG"
        m.mkdir()
        result = check_model_versions(str(tmp_path))
        assert result[0]["trained"] == "no config"

    def test_check_system_health_no_systemctl(self):
        from scripts.ops.ops_dashboard import check_system_health
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = check_system_health()
        assert result == "systemctl_not_found"

    def test_check_system_health_exception(self):
        from scripts.ops.ops_dashboard import check_system_health
        with patch("subprocess.run", side_effect=OSError("test")):
            result = check_system_health()
        assert result == "unknown"

    def test_format_dashboard_smoke(self):
        from scripts.ops.ops_dashboard import format_dashboard
        data = {
            "service_status": "active",
            "service_statuses": {
                "bybit-alpha.service": "active",
                "bybit-mm.service": "inactive",
            },
            "burnin": {"status": "not_started", "phases": {}},
            "models": [],
            "signals": {
                "hours": 24, "total_bars": 0,
                "nonzero_signals": 0, "signal_rate": 0,
            },
            "data_freshness": [],
            "disk_usage": {},
        }
        text = format_dashboard(data)
        assert "OPS DASHBOARD" in text
        assert "active" in text
        assert "bybit-mm.service" in text

    def test_format_dashboard_with_data(self):
        from scripts.ops.ops_dashboard import format_dashboard
        data = {
            "service_status": "active",
            "service_statuses": {
                "bybit-alpha.service": "active",
                "bybit-mm.service": "active",
            },
            "burnin": {
                "status": "passed",
                "phases": {
                    "A": {"passed": True, "duration_hours": 170},
                    "B": {"passed": True, "duration_hours": 172},
                    "C": {"passed": True, "duration_hours": 75},
                },
            },
            "models": [
                {
                    "name": "ETHUSDT_gate_v2", "horizon": 24,
                    "n_features": 14, "trained": "2026-03-01",
                },
            ],
            "signals": {
                "hours": 24, "total_bars": 100,
                "nonzero_signals": 30, "signal_rate": 0.3,
            },
            "data_freshness": [
                {
                    "file": "ETHUSDT_1h.csv",
                    "modified": "2026-03-17 10:00",
                    "age_hours": 2.0,
                },
            ],
            "disk_usage": {"logs": "5.0 MB"},
        }
        text = format_dashboard(data)
        assert "ETHUSDT_gate_v2" in text
        assert "PASS" in text

    def test_check_recent_signals_missing_log(self, tmp_path):
        from scripts.ops.ops_dashboard import check_recent_signals
        result = check_recent_signals(str(tmp_path / "missing.log"))
        assert result["total_bars"] == 0
        assert "error" in result

    def test_check_recent_signals_with_data(self, tmp_path):
        from scripts.ops.ops_dashboard import check_recent_signals
        log = tmp_path / "test.log"
        log.write_text(textwrap.dedent("""\
            2099-01-01 10:00:00,000 INFO __main__: WS ETHUSDT bar 201: $2100.0 z=-0.2 sig=0 hold=1
            2099-01-01 11:00:00,000 INFO __main__: WS ETHUSDT bar 202: $2110.0 z=+0.5 sig=1 hold=1
        """))
        result = check_recent_signals(str(log), hours=999999)
        assert result["total_bars"] == 2
        assert result["nonzero_signals"] == 1


# ── pre_live_checklist ─────────────────────────────────────

class TestPreLiveChecklist:
    def test_check_api_keys_present(self, monkeypatch):
        from scripts.ops.pre_live_checklist import check_api_keys
        monkeypatch.setenv("BYBIT_API_KEY", "test_key")
        monkeypatch.setenv("BYBIT_API_SECRET", "test_secret")
        ok, missing = check_api_keys()
        assert ok is True
        assert missing == []

    def test_check_api_keys_missing(self, monkeypatch):
        from scripts.ops.pre_live_checklist import check_api_keys
        monkeypatch.delenv("BYBIT_API_KEY", raising=False)
        monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
        ok, missing = check_api_keys()
        assert ok is False
        assert "BYBIT_API_KEY" in missing

    def test_check_api_keys_partial(self, monkeypatch):
        from scripts.ops.pre_live_checklist import check_api_keys
        monkeypatch.setenv("BYBIT_API_KEY", "key")
        monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
        ok, missing = check_api_keys()
        assert ok is False
        assert "BYBIT_API_SECRET" in missing
        assert "BYBIT_API_KEY" not in missing

    def test_check_max_order_notional(self):
        from scripts.ops.pre_live_checklist import check_max_order_notional
        ok, val = check_max_order_notional()
        assert ok is True
        assert val == 5000

    def test_check_burnin_passed_no_file(self):
        from scripts.ops.pre_live_checklist import check_burnin_passed
        ok, detail = check_burnin_passed(report_path="/nonexistent.json")
        assert ok is False

    def test_check_burnin_passed_all_passed_list(self, tmp_path):
        from scripts.ops.pre_live_checklist import check_burnin_passed
        report = tmp_path / "burnin.json"
        report.write_text(json.dumps([
            {"phase": "A", "passed": True},
            {"phase": "B", "passed": True},
            {"phase": "C", "passed": True},
        ]))
        ok, detail = check_burnin_passed(report_path=str(report))
        assert ok is True

    def test_check_burnin_passed_incomplete(self, tmp_path):
        from scripts.ops.pre_live_checklist import check_burnin_passed
        report = tmp_path / "burnin.json"
        report.write_text(json.dumps([
            {"phase": "A", "passed": True},
            {"phase": "B", "passed": False},
        ]))
        ok, detail = check_burnin_passed(report_path=str(report))
        assert ok is False

    def test_check_burnin_passed_dict_format(self, tmp_path):
        """Dict format with phases.X.status keys."""
        from scripts.ops.pre_live_checklist import check_burnin_passed
        report = tmp_path / "burnin.json"
        report.write_text(json.dumps({
            "phases": {
                "A": {"status": "passed"},
                "B": {"status": "passed"},
                "C": {"status": "passed"},
            }
        }))
        ok, detail = check_burnin_passed(report_path=str(report))
        assert ok is True

    def test_check_burnin_passed_dict_incomplete(self, tmp_path):
        from scripts.ops.pre_live_checklist import check_burnin_passed
        report = tmp_path / "burnin.json"
        report.write_text(json.dumps({
            "phases": {
                "A": {"status": "passed"},
                "B": {"status": "in_progress"},
            }
        }))
        ok, detail = check_burnin_passed(report_path=str(report))
        assert ok is False

    def test_check_burnin_passed_bad_json(self, tmp_path):
        from scripts.ops.pre_live_checklist import check_burnin_passed
        report = tmp_path / "bad.json"
        report.write_text("not json")
        ok, detail = check_burnin_passed(report_path=str(report))
        assert ok is False
        assert "read error" in str(detail)

    def test_check_models_exist(self, tmp_path):
        from scripts.ops.pre_live_checklist import check_models_exist
        (tmp_path / "ETHUSDT_gate_v2").mkdir()
        (tmp_path / "BTCUSDT_gate_v2").mkdir()
        ok, missing = check_models_exist(str(tmp_path))
        assert ok is True
        assert missing == []

    def test_check_models_missing(self, tmp_path):
        from scripts.ops.pre_live_checklist import check_models_exist
        ok, missing = check_models_exist(str(tmp_path))
        assert ok is False
        assert len(missing) > 0

    def test_check_models_partial(self, tmp_path):
        from scripts.ops.pre_live_checklist import check_models_exist
        (tmp_path / "ETHUSDT_gate_v2").mkdir()
        ok, missing = check_models_exist(str(tmp_path))
        assert ok is False
        assert "BTCUSDT_gate_v2" in missing
        assert "ETHUSDT_gate_v2" not in missing

    def test_run_checks_returns_list(self):
        from scripts.ops.pre_live_checklist import run_checks
        results = run_checks([
            ("always pass", lambda: (True, "ok")),
            ("always fail", lambda: (False, "nope")),
        ])
        assert len(results) == 2
        assert results[0]["status"] == "PASS"
        assert results[1]["status"] == "FAIL"

    def test_run_checks_handles_exception(self):
        from scripts.ops.pre_live_checklist import run_checks

        def boom():
            raise RuntimeError("kaboom")

        results = run_checks([("boom", boom)])
        assert results[0]["status"] == "SKIP"
        assert "kaboom" in results[0]["detail"]

    def test_format_checklist_smoke(self):
        from scripts.ops.pre_live_checklist import format_checklist
        results = [
            {"name": "test", "status": "PASS", "detail": "ok"},
            {"name": "test2", "status": "FAIL", "detail": "bad"},
        ]
        text = format_checklist(results)
        assert "PRE-LIVE CHECKLIST" in text
        assert "AUTOMATED: FAIL" in text

    def test_format_checklist_all_pass(self):
        from scripts.ops.pre_live_checklist import format_checklist
        results = [{"name": "test", "status": "PASS", "detail": "ok"}]
        text = format_checklist(results)
        assert "AUTOMATED: PASS" in text
