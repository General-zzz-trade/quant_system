"""Tests for Phase 3 portfolio risk and risk reporting modules."""
from __future__ import annotations

import json
import tempfile
import pytest
from decimal import Decimal
from pathlib import Path

from risk.portfolio_risk.stress_testing import (
    PortfolioStressTester, PortfolioSnapshot, StressTestResult,
)
from risk.portfolio_risk.concentration import (
    ConcentrationMonitor, ConcentrationMetrics, ConcentrationAlert,
)
from risk.reporting.risk_report import (
    RiskReportGenerator, RiskSummary, PositionRisk,
)


# ── F2: Portfolio stress testing ──

class TestPortfolioStressTester:
    def _make_snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            equity=Decimal("100000"),
            cash=Decimal("50000"),
            positions={
                "BTCUSDT": Decimal("1"),
                "ETHUSDT": Decimal("10"),
            },
            prices={
                "BTCUSDT": Decimal("40000"),
                "ETHUSDT": Decimal("3000"),
            },
        )

    def test_run_with_defaults(self):
        tester = PortfolioStressTester()
        snapshot = self._make_snapshot()
        results = tester.run(snapshot)
        assert len(results) > 0
        # Should have global + per-symbol scenarios
        names = [r.scenario_name for r in results]
        assert any("global" in n for n in names)
        assert any("BTCUSDT" in n for n in names)

    def test_stress_pnl_negative_on_crash(self):
        tester = PortfolioStressTester()
        snapshot = self._make_snapshot()
        results = tester.run(snapshot)
        # Find a crash scenario
        crashes = [r for r in results if "crash" in r.scenario_name or "down" in r.scenario_name]
        assert len(crashes) > 0
        # At least one should have negative PnL
        assert any(r.pnl < 0 for r in crashes)

    def test_breach_detection(self):
        tester = PortfolioStressTester(max_drawdown_pct=0.05)  # Very tight
        snapshot = self._make_snapshot()
        results = tester.run(snapshot)
        # With 5% max DD, some scenarios should breach
        breached = [r for r in results if r.breaches]
        assert len(breached) > 0

    def test_worst_case(self):
        tester = PortfolioStressTester()
        snapshot = self._make_snapshot()
        results = tester.run(snapshot)
        worst = PortfolioStressTester.worst_case(results)
        assert worst is not None
        assert worst.drawdown_pct >= 0

    def test_custom_scenario(self):
        from risk.stress import StressScenario, PriceShock

        tester = PortfolioStressTester()
        tester.add_scenario(StressScenario(
            name="custom_btc_50pct_crash",
            shocks={"BTCUSDT": PriceShock(pct=Decimal("-0.50"))},
        ))
        snapshot = self._make_snapshot()
        results = tester.run(snapshot, include_defaults=False)
        assert len(results) == 1
        assert results[0].scenario_name == "custom_btc_50pct_crash"
        assert results[0].pnl < 0

    def test_empty_portfolio(self):
        tester = PortfolioStressTester()
        snapshot = PortfolioSnapshot(
            equity=Decimal("10000"),
            cash=Decimal("10000"),
            positions={},
            prices={},
        )
        results = tester.run(snapshot)
        # Global scenarios still run but with 0 PnL
        for r in results:
            assert r.pnl == 0


# ── F2: Concentration monitoring ──

class TestConcentrationMonitor:
    def test_compute_metrics_basic(self):
        monitor = ConcentrationMonitor()
        positions = {
            "BTCUSDT": Decimal("1"),
            "ETHUSDT": Decimal("10"),
            "SOLUSDT": Decimal("100"),
        }
        prices = {
            "BTCUSDT": Decimal("40000"),
            "ETHUSDT": Decimal("3000"),
            "SOLUSDT": Decimal("100"),
        }
        metrics = monitor.compute_metrics(positions, prices)
        assert metrics.hhi > 0
        assert metrics.max_weight > 0
        assert metrics.max_weight_symbol == "BTCUSDT"  # Largest position value
        assert metrics.effective_positions > 0
        assert abs(metrics.top3_weight - 1.0) < 0.01  # Only 3 positions

    def test_single_position_hhi_1(self):
        monitor = ConcentrationMonitor()
        positions = {"BTCUSDT": Decimal("1")}
        prices = {"BTCUSDT": Decimal("50000")}
        metrics = monitor.compute_metrics(positions, prices)
        assert metrics.hhi == pytest.approx(1.0)
        assert metrics.max_weight == pytest.approx(1.0)
        assert metrics.effective_positions == pytest.approx(1.0)

    def test_equal_weights_low_hhi(self):
        monitor = ConcentrationMonitor()
        # 4 equal positions
        positions = {f"SYM{i}": Decimal("100") for i in range(4)}
        prices = {f"SYM{i}": Decimal("100") for i in range(4)}
        metrics = monitor.compute_metrics(positions, prices)
        assert metrics.hhi == pytest.approx(0.25)  # 4 * (0.25)^2
        assert metrics.effective_positions == pytest.approx(4.0)

    def test_empty_portfolio(self):
        monitor = ConcentrationMonitor()
        metrics = monitor.compute_metrics({}, {})
        assert metrics.hhi == 0.0
        assert metrics.max_weight == 0.0

    def test_alerts_triggered(self):
        monitor = ConcentrationMonitor(
            max_single_weight=0.30,
            max_top3_weight=0.80,
            max_hhi=0.20,
        )
        # Concentrated portfolio
        positions = {
            "BTCUSDT": Decimal("1"),
            "ETHUSDT": Decimal("1"),
        }
        prices = {
            "BTCUSDT": Decimal("90000"),  # 90% weight
            "ETHUSDT": Decimal("10000"),  # 10% weight
        }
        metrics = monitor.compute_metrics(positions, prices)
        alerts = monitor.check_alerts(metrics)
        assert len(alerts) >= 2  # max_single_weight and hhi

    def test_no_alerts_diversified(self):
        monitor = ConcentrationMonitor()
        positions = {f"SYM{i}": Decimal("100") for i in range(10)}
        prices = {f"SYM{i}": Decimal("100") for i in range(10)}
        metrics = monitor.compute_metrics(positions, prices)
        alerts = monitor.check_alerts(metrics)
        assert len(alerts) == 0

    def test_sector_hhi(self):
        monitor = ConcentrationMonitor()
        monitor.set_sector_map({
            "BTCUSDT": "crypto",
            "ETHUSDT": "crypto",
            "AAPL": "tech",
        })
        positions = {
            "BTCUSDT": Decimal("1"),
            "ETHUSDT": Decimal("10"),
            "AAPL": Decimal("50"),
        }
        prices = {
            "BTCUSDT": Decimal("40000"),
            "ETHUSDT": Decimal("3000"),
            "AAPL": Decimal("200"),
        }
        metrics = monitor.compute_metrics(positions, prices)
        assert metrics.sector_hhi > 0


# ── F3: Risk reporting ──

class TestRiskReportGenerator:
    def test_generate_basic(self):
        gen = RiskReportGenerator()
        summary = gen.generate(
            equity=100000.0,
            positions={
                "BTCUSDT": {"qty": 1.0, "price": 50000.0, "pnl": 5000.0},
                "ETHUSDT": {"qty": 10.0, "price": 3000.0, "pnl": -200.0},
            },
        )
        assert summary.total_equity == 100000.0
        assert summary.position_count == 2
        assert summary.leverage > 0
        assert len(summary.positions) == 2

    def test_position_weights_sum_to_one(self):
        gen = RiskReportGenerator()
        summary = gen.generate(
            equity=100000.0,
            positions={
                "A": {"qty": 10.0, "price": 100.0, "pnl": 0.0},
                "B": {"qty": 20.0, "price": 200.0, "pnl": 0.0},
            },
        )
        total_weight = sum(p.weight for p in summary.positions)
        assert total_weight == pytest.approx(1.0)

    def test_with_risk_metrics(self):
        gen = RiskReportGenerator()
        summary = gen.generate(
            equity=100000.0,
            positions={"A": {"qty": 10.0, "price": 100.0, "pnl": 0.0}},
            var_95=5000.0,
            var_99=8000.0,
            max_drawdown=0.05,
            hhi=0.25,
            alerts=["High concentration in BTCUSDT"],
        )
        assert summary.var_95 == 5000.0
        assert summary.var_99 == 8000.0
        assert summary.max_drawdown == 0.05
        assert len(summary.alerts) == 1

    def test_format_text(self):
        gen = RiskReportGenerator()
        summary = gen.generate(
            equity=100000.0,
            positions={"BTCUSDT": {"qty": 1.0, "price": 50000.0, "pnl": 0.0}},
            var_95=5000.0,
        )
        text = gen.format_text(summary)
        assert "RISK REPORT" in text
        assert "BTCUSDT" in text
        assert "VaR" in text

    def test_save_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = RiskReportGenerator(output_dir=tmpdir)
            summary = gen.generate(
                equity=100000.0,
                positions={"A": {"qty": 1.0, "price": 1000.0, "pnl": 0.0}},
            )
            files = list(Path(tmpdir).glob("risk_report_*.json"))
            assert len(files) == 1
            data = json.loads(files[0].read_text())
            assert data["total_equity"] == 100000.0

    def test_empty_portfolio(self):
        gen = RiskReportGenerator()
        summary = gen.generate(equity=100000.0, positions={})
        assert summary.position_count == 0
        assert summary.leverage == 0.0
