"""Tests for DrawdownCircuitBreaker."""
import logging
import time

from risk.kill_switch import KillSwitch, KillMode
from risk.drawdown_breaker import DrawdownCircuitBreaker, DrawdownBreakerConfig


class TestDrawdownCircuitBreaker:
    def _make(self, **kwargs):
        ks = KillSwitch()
        cfg = DrawdownBreakerConfig(**kwargs)
        return DrawdownCircuitBreaker(kill_switch=ks, config=cfg), ks

    def _now(self):
        return time.time()

    def test_normal_state(self):
        breaker, ks = self._make()
        t = self._now()
        state = breaker.on_equity_update(10000.0, now_ts=t)
        assert state == "normal"
        assert ks.is_killed() is None

    def test_no_trigger_below_warning(self):
        """DD < warning threshold -> no kill switch trigger."""
        breaker, ks = self._make(warning_pct=10.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        # 5% drawdown, below 10% warning
        state = breaker.on_equity_update(9500.0, now_ts=t + 1)
        assert state == "normal"
        assert ks.is_killed() is None
        assert breaker.current_drawdown_pct == 5.0

    def test_warning_threshold_logs(self, caplog):
        """DD crosses warning threshold -> warning logged, no kill switch."""
        breaker, ks = self._make(warning_pct=10.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        with caplog.at_level(logging.WARNING, logger="risk.drawdown_breaker"):
            state = breaker.on_equity_update(8900.0, now_ts=t + 1)
        assert state == "warning"
        assert ks.is_killed() is None  # warning does NOT trigger kill switch
        assert any("WARNING" in rec.message or "DrawdownBreaker" in rec.message for rec in caplog.records)

    def test_reduce_threshold_triggers_reduce_only(self):
        """DD crosses reduce threshold -> kill_switch.trigger(REDUCE_ONLY)."""
        breaker, ks = self._make(reduce_pct=15.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        state = breaker.on_equity_update(8400.0, now_ts=t + 1)
        assert state == "reduce_only"
        rec = ks.is_killed()
        assert rec is not None
        assert rec.mode == KillMode.REDUCE_ONLY

    def test_kill_threshold_triggers_hard_kill(self):
        """DD crosses kill threshold -> kill_switch.trigger(HARD_KILL)."""
        breaker, ks = self._make(kill_pct=20.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        state = breaker.on_equity_update(7900.0, now_ts=t + 1)
        assert state == "killed"
        rec = ks.is_killed()
        assert rec is not None
        assert rec.mode == KillMode.HARD_KILL

    def test_velocity_detection_triggers_hard_kill(self):
        """Equity drops >5% in 15min -> immediate hard kill."""
        breaker, ks = self._make(velocity_pct=5.0, velocity_window_sec=900)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        # 6% drop in 60 seconds (well within 900s window)
        state = breaker.on_equity_update(9400.0, now_ts=t + 60)
        assert state == "killed"
        rec = ks.is_killed()
        assert rec is not None
        assert rec.mode == KillMode.HARD_KILL

    def test_hwm_checkpoint_restore(self):
        """checkpoint() saves HWM, restore_checkpoint() restores it."""
        breaker, ks = self._make(velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(12345.0, now_ts=t)
        assert breaker.equity_hwm == 12345.0

        cp = breaker.checkpoint()
        assert cp["equity_hwm"] == 12345.0

        # Create fresh breaker and restore
        breaker2, _ = self._make()
        assert breaker2.equity_hwm == 0.0
        breaker2.restore_checkpoint(cp)
        assert breaker2.equity_hwm == 12345.0

    def test_hwm_updates_on_new_high(self):
        """HWM tracks highest equity seen."""
        breaker, _ = self._make(velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        assert breaker.equity_hwm == 10000.0

        breaker.on_equity_update(10500.0, now_ts=t + 1)
        assert breaker.equity_hwm == 10500.0

        # Equity drops -- HWM should NOT decrease
        breaker.on_equity_update(10200.0, now_ts=t + 2)
        assert breaker.equity_hwm == 10500.0

        # New high
        breaker.on_equity_update(11000.0, now_ts=t + 3)
        assert breaker.equity_hwm == 11000.0

    def test_warning_threshold(self):
        breaker, ks = self._make(warning_pct=10.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        state = breaker.on_equity_update(8900.0, now_ts=t + 1)
        assert state == "warning"
        assert ks.is_killed() is None  # warning doesn't trigger kill switch

    def test_no_velocity_breach_outside_window(self):
        breaker, ks = self._make(velocity_pct=5.0, velocity_window_sec=900, kill_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        # 6% drop but after window expired
        state = breaker.on_equity_update(9400.0, now_ts=t + 1000)
        # Should be warning (6% DD from HWM) but NOT velocity kill
        assert state != "killed"

    def test_recovery_from_warning(self):
        breaker, _ = self._make(warning_pct=10.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        breaker.on_equity_update(8900.0, now_ts=t + 1)
        assert breaker.state == "warning"
        # Equity recovers above warning
        breaker.on_equity_update(9200.0, now_ts=t + 2)
        assert breaker.state == "normal"

    def test_reset(self):
        breaker, ks = self._make(kill_pct=20.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        breaker.on_equity_update(7900.0, now_ts=t + 1)
        assert breaker.state == "killed"
        breaker.reset(new_hwm=7900.0)
        assert breaker.state == "normal"
        assert ks.is_killed() is None

    def test_get_status(self):
        breaker, _ = self._make(velocity_pct=99.0, warning_pct=5.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        breaker.on_equity_update(9500.0, now_ts=t + 1)
        status = breaker.get_status()
        assert status["state"] == "warning"
        assert status["drawdown_pct"] == 5.0
        assert status["equity_hwm"] == 10000.0

    def test_escalation_warning_to_reduce_to_kill(self):
        breaker, ks = self._make(warning_pct=5.0, reduce_pct=10.0, kill_pct=15.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)

        breaker.on_equity_update(9400.0, now_ts=t + 1)
        assert breaker.state == "warning"

        breaker.on_equity_update(8900.0, now_ts=t + 2)
        assert breaker.state == "reduce_only"

        breaker.on_equity_update(8400.0, now_ts=t + 3)
        assert breaker.state == "killed"
