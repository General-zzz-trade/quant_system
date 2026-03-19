"""Tests for RiskMonitor."""

import time
import pytest
from execution.market_maker.config import MarketMakerConfig
from execution.market_maker.risk_monitor import RiskMonitor


@pytest.fixture
def risk():
    cfg = MarketMakerConfig(
        daily_loss_limit=10.0,
        circuit_breaker_losses=3,
        circuit_breaker_pause_s=1.0,
    )
    return RiskMonitor(cfg)


class TestRiskMonitor:
    def test_initial_state_running(self, risk):
        assert risk.state == "running"
        assert risk.can_quote

    def test_daily_loss_kills(self, risk):
        state = risk.check(daily_pnl=-11.0, consecutive_losses=0)
        assert state == "killed"
        assert risk.is_killed
        assert not risk.can_quote

    def test_circuit_breaker_pauses(self, risk):
        state = risk.check(daily_pnl=-5.0, consecutive_losses=3)
        assert state == "paused"
        assert not risk.can_quote

    def test_circuit_breaker_expires(self, risk):
        risk.check(daily_pnl=-5.0, consecutive_losses=3)
        assert risk.state == "paused"
        # Wait for cooldown (1 second in test config)
        time.sleep(1.1)
        assert risk.state == "running"

    def test_consecutive_losses_below_threshold(self, risk):
        state = risk.check(daily_pnl=-5.0, consecutive_losses=2)
        assert state == "running"

    def test_kill_is_permanent(self, risk):
        risk.check(daily_pnl=-15.0, consecutive_losses=0)
        assert risk.is_killed
        # Even with good PnL, still killed
        risk.check(daily_pnl=5.0, consecutive_losses=0)
        # Once killed by daily loss, stays killed (PnL can't recover past limit)
        # Actually the check uses <= -limit, so positive PnL won't trigger kill
        # But the state is already killed and we don't auto-recover
        assert risk.is_killed

    def test_force_kill(self, risk):
        risk.force_kill("test_reason")
        assert risk.is_killed
        assert not risk.can_quote

    def test_reset(self, risk):
        risk.force_kill("test")
        assert risk.is_killed
        risk.reset()
        assert risk.state == "running"
        assert risk.can_quote

    def test_pause_recovers_when_losses_clear(self, risk):
        risk.check(daily_pnl=-5.0, consecutive_losses=3)
        assert risk.state == "paused"
        # Losses cleared (e.g., a win happened)
        risk.check(daily_pnl=-5.0, consecutive_losses=1)
        assert risk.state == "running"
