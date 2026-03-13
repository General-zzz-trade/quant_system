"""Tests for DrawdownCircuitBreaker."""
import time

from risk.kill_switch import KillSwitch, KillMode, KillScope
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

    def test_warning_threshold(self):
        breaker, ks = self._make(warning_pct=10.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        state = breaker.on_equity_update(8900.0, now_ts=t + 1)
        assert state == "warning"
        assert ks.is_killed() is None  # warning doesn't trigger kill switch

    def test_reduce_only_threshold(self):
        breaker, ks = self._make(reduce_pct=15.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        state = breaker.on_equity_update(8400.0, now_ts=t + 1)
        assert state == "reduce_only"
        rec = ks.is_killed()
        assert rec is not None
        assert rec.mode == KillMode.REDUCE_ONLY

    def test_hard_kill_threshold(self):
        breaker, ks = self._make(kill_pct=20.0, velocity_pct=99.0)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        state = breaker.on_equity_update(7900.0, now_ts=t + 1)
        assert state == "killed"
        rec = ks.is_killed()
        assert rec is not None
        assert rec.mode == KillMode.HARD_KILL

    def test_velocity_breach(self):
        breaker, ks = self._make(velocity_pct=5.0, velocity_window_sec=900)
        t = self._now()
        breaker.on_equity_update(10000.0, now_ts=t)
        # 6% drop in 60 seconds (< 900s window)
        state = breaker.on_equity_update(9400.0, now_ts=t + 60)
        assert state == "killed"
        rec = ks.is_killed()
        assert rec is not None
        assert rec.mode == KillMode.HARD_KILL

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
