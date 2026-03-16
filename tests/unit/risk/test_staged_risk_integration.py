# tests/unit/risk/test_staged_risk_integration.py
"""Tests for StagedRiskManager integration with gate chain."""
from __future__ import annotations

import pytest

from risk.staged_risk import StagedRiskManager


class TestStagedRiskStages:
    def test_initial_stage_survival(self):
        mgr = StagedRiskManager(initial_equity=200.0)
        assert mgr.stage.label == "survival"
        assert mgr.risk_fraction == 0.50
        assert mgr.leverage == 3.0

    def test_initial_stage_growth(self):
        mgr = StagedRiskManager(initial_equity=500.0)
        assert mgr.stage.label == "growth"

    def test_initial_stage_stable(self):
        mgr = StagedRiskManager(initial_equity=1000.0)
        assert mgr.stage.label == "stable"

    def test_initial_stage_safe(self):
        mgr = StagedRiskManager(initial_equity=3000.0)
        assert mgr.stage.label == "safe"

    def test_initial_stage_institutional(self):
        mgr = StagedRiskManager(initial_equity=10000.0)
        assert mgr.stage.label == "institutional"

    def test_upgrade_immediate(self):
        mgr = StagedRiskManager(initial_equity=200.0)
        assert mgr.stage.label == "survival"
        mgr.update_equity(500.0)
        assert mgr.stage.label == "growth"

    def test_downgrade_with_hysteresis(self):
        """Downgrade requires equity to drop 10% below boundary."""
        mgr = StagedRiskManager(initial_equity=500.0)
        assert mgr.stage.label == "growth"
        # Drop to just below boundary (300) — not enough for hysteresis
        mgr.update_equity(295.0)
        assert mgr.stage.label == "growth"  # Still growth (hysteresis)
        # Drop below 300 * 0.9 = 270
        mgr.update_equity(260.0)
        assert mgr.stage.label == "survival"


class TestStagedRiskDrawdown:
    def test_no_drawdown_full_scale(self):
        mgr = StagedRiskManager(initial_equity=500.0)
        assert mgr.position_scale() == 1.0

    def test_drawdown_reduces_scale(self):
        mgr = StagedRiskManager(initial_equity=500.0)
        # Growth stage max_drawdown = 20%
        # 30% of 20% = 6% drawdown → transition from 1.0 to 0.7
        mgr.update_equity(470.0)  # 6% DD
        assert mgr.position_scale() < 1.0

    def test_deep_drawdown_halts(self):
        mgr = StagedRiskManager(initial_equity=500.0)
        # Growth stage max_drawdown = 20%
        mgr.update_equity(380.0)  # 24% DD
        assert mgr.can_trade is False
        assert mgr.position_scale() == 0.0

    def test_halt_recovery(self):
        mgr = StagedRiskManager(initial_equity=500.0)
        # Halt
        mgr.update_equity(380.0)
        assert mgr.can_trade is False
        # Recover to 380 * 1.05 = 399
        mgr.update_equity(400.0)
        assert mgr.can_trade is True


class TestStagedRiskNotional:
    def test_compute_notional_basic(self):
        mgr = StagedRiskManager(initial_equity=500.0)
        notional = mgr.compute_notional(price=2000.0)
        # growth: 500 * 0.25 * 3.0 * 1.0 = 375
        assert notional == pytest.approx(375.0)

    def test_min_notional_enforcement(self):
        """If computed notional < min, scale up to min if safe."""
        mgr = StagedRiskManager(initial_equity=500.0, min_notional=400.0)
        notional = mgr.compute_notional(price=2000.0)
        # Computed = 375 < 400, max safe = 500*3.0*0.8=1200, so scale up
        assert notional == 400.0

    def test_halted_returns_zero(self):
        mgr = StagedRiskManager(initial_equity=500.0)
        mgr.update_equity(380.0)  # Halt
        assert mgr.compute_notional(price=2000.0) == 0.0


class TestStagedRiskGateIntegration:
    def test_gate_allows_when_can_trade(self):
        from runner.gate_chain import StagedRiskGate
        mgr = StagedRiskManager(initial_equity=500.0)
        gate = StagedRiskGate(mgr)
        result = gate.check(None, {})
        assert result.allowed is True
        assert result.scale == 1.0

    def test_gate_blocks_when_halted(self):
        from runner.gate_chain import StagedRiskGate
        mgr = StagedRiskManager(initial_equity=500.0)
        mgr.update_equity(380.0)  # Halt
        gate = StagedRiskGate(mgr)
        result = gate.check(None, {})
        assert result.allowed is False
        assert "halted" in result.reason

    def test_gate_scales_on_drawdown(self):
        from runner.gate_chain import StagedRiskGate
        mgr = StagedRiskManager(initial_equity=500.0)
        mgr.update_equity(450.0)  # 10% DD → some scaling
        gate = StagedRiskGate(mgr)
        result = gate.check(None, {})
        assert result.allowed is True
        assert result.scale <= 1.0
