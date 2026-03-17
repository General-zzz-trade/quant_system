"""Parity test: RustDrawdownBreaker vs Python DrawdownCircuitBreaker."""
import pytest

try:
    from _quant_hotpath import RustDrawdownBreaker
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not built")


def make_py_breaker(config=None):
    """Force Python path (no Rust)."""
    from collections import deque
    from unittest.mock import MagicMock
    from risk.kill_switch import KillSwitch
    from risk.drawdown_breaker import DrawdownBreakerConfig, DrawdownCircuitBreaker
    ks = MagicMock(spec=KillSwitch)
    b = DrawdownCircuitBreaker.__new__(DrawdownCircuitBreaker)
    b._kill_switch = ks
    b._config = config or DrawdownBreakerConfig()
    b._alert_manager = None
    b._use_rust = False
    b._equity_hwm = 0.0
    b._current_dd_pct = 0.0
    b._state = "normal"
    b._equity_history = deque(maxlen=1000)
    b._last_warning_ts = 0.0
    return b, ks


class TestRustDrawdownBreakerDirect:
    """Tests directly against the Rust class."""

    def test_escalation_chain(self):
        """normal → warning → reduce_only → killed."""
        rust = RustDrawdownBreaker(warning_pct=10.0, reduce_pct=15.0, kill_pct=20.0, velocity_pct=99.0)

        # Normal
        state, action = rust.on_equity_update(1000.0, 0.0)
        assert state == "normal"
        assert action is None

        # Warning at 11% dd (hwm=1000, current=890)
        state, action = rust.on_equity_update(890.0, 1.0)
        assert state == "warning"
        assert action is None

        # Reduce at 16% dd
        state, action = rust.on_equity_update(840.0, 2.0)
        assert state == "reduce_only"
        assert action is not None
        assert action[0] == "reduce_only"

        # Kill at 21% dd
        state, action = rust.on_equity_update(790.0, 3.0)
        assert state == "killed"
        assert action is not None
        assert action[0] == "hard_kill"

    def test_velocity_detection(self):
        """5%+ drop within 900s triggers hard kill."""
        rust = RustDrawdownBreaker(velocity_pct=5.0, velocity_window_sec=900.0)
        rust.on_equity_update(1000.0, 0.0)
        # Drop 6% within window (60s < 900s) → velocity breach
        state, action = rust.on_equity_update(940.0, 60.0)
        assert action is not None
        assert action[0] == "hard_kill"
        assert "velocity_breach" in action[1]

    def test_velocity_no_breach_outside_window(self):
        """Drop outside window does not trigger velocity kill."""
        rust = RustDrawdownBreaker(velocity_pct=5.0, velocity_window_sec=900.0, kill_pct=99.0)
        rust.on_equity_update(1000.0, 0.0)
        # Drop 6% but after window expired (1100s > 900s window)
        state, action = rust.on_equity_update(940.0, 1100.0)
        assert state != "killed", f"Should not kill outside window, got: {state}"

    def test_recovery_warning_to_normal(self):
        rust = RustDrawdownBreaker(velocity_pct=99.0)
        rust.on_equity_update(1000.0, 0.0)
        rust.on_equity_update(890.0, 1.0)  # → warning (11% dd)
        assert rust.state == "warning"
        rust.on_equity_update(960.0, 2.0)  # 4% dd < 10% → normal
        assert rust.state == "normal"

    def test_killed_is_terminal(self):
        rust = RustDrawdownBreaker(velocity_pct=99.0)
        rust.on_equity_update(1000.0, 0.0)
        rust.on_equity_update(790.0, 1.0)  # → killed (21% dd)
        assert rust.state == "killed"
        # No action on further updates (already killed)
        state, action = rust.on_equity_update(1000.0, 2.0)
        assert state == "killed"
        assert action is None

    def test_checkpoint_restore(self):
        rust = RustDrawdownBreaker(velocity_pct=99.0)
        rust.on_equity_update(1000.0, 0.0)
        ckpt = rust.checkpoint()
        assert "equity_hwm" in ckpt
        assert ckpt["equity_hwm"] == pytest.approx(1000.0)
        assert "state" in ckpt

        rust2 = RustDrawdownBreaker()
        rust2.restore_checkpoint(ckpt)
        assert rust2.equity_hwm == pytest.approx(1000.0)
        # restore_checkpoint only restores HWM (not state — matches Python)
        assert rust2.state == "normal"

    def test_reset_clears(self):
        rust = RustDrawdownBreaker(velocity_pct=99.0)
        rust.on_equity_update(1000.0, 0.0)
        rust.on_equity_update(790.0, 1.0)  # killed
        state, action = rust.reset()
        assert state == "normal"
        assert action is not None
        assert action[0] == "clear"

    def test_reset_with_new_hwm(self):
        rust = RustDrawdownBreaker(velocity_pct=99.0)
        rust.on_equity_update(1000.0, 0.0)
        rust.reset(new_hwm=800.0)
        assert rust.equity_hwm == pytest.approx(800.0)
        assert rust.state == "normal"

    def test_equity_zero_ignored(self):
        rust = RustDrawdownBreaker()
        state, action = rust.on_equity_update(0.0, 0.0)
        assert state == "normal"
        assert action is None

    def test_getters(self):
        rust = RustDrawdownBreaker(velocity_pct=99.0)
        rust.on_equity_update(1000.0, 0.0)
        rust.on_equity_update(890.0, 1.0)  # 11% dd → warning
        assert rust.state == "warning"
        assert rust.current_drawdown_pct == pytest.approx(11.0)
        assert rust.equity_hwm == pytest.approx(1000.0)

    def test_get_status(self):
        rust = RustDrawdownBreaker(velocity_pct=99.0, warning_pct=5.0)
        rust.on_equity_update(10000.0, 0.0)
        rust.on_equity_update(9500.0, 1.0)  # 5% dd → warning
        status = rust.get_status()
        assert status["state"] == "warning"
        assert status["drawdown_pct"] == pytest.approx(5.0, abs=0.01)
        assert status["equity_hwm"] == pytest.approx(10000.0)
        assert "thresholds" in status
        assert "warning_pct" in status["thresholds"]

    def test_reduce_not_repeated(self):
        rust = RustDrawdownBreaker(velocity_pct=99.0)
        rust.on_equity_update(1000.0, 0.0)
        _, action1 = rust.on_equity_update(840.0, 1.0)  # → reduce_only
        assert action1 is not None
        assert action1[0] == "reduce_only"
        _, action2 = rust.on_equity_update(838.0, 2.0)  # still in reduce range
        assert action2 is None

    def test_hwm_updates_on_new_high(self):
        rust = RustDrawdownBreaker(velocity_pct=99.0)
        rust.on_equity_update(10000.0, 0.0)
        assert rust.equity_hwm == pytest.approx(10000.0)
        rust.on_equity_update(10500.0, 1.0)
        assert rust.equity_hwm == pytest.approx(10500.0)
        rust.on_equity_update(10200.0, 2.0)
        assert rust.equity_hwm == pytest.approx(10500.0)  # HWM should not decrease


class TestDrawdownBreakerParity:
    """Parity tests: Rust vs Python paths produce the same outcomes."""

    def test_python_wrapper_uses_rust(self):
        """DrawdownCircuitBreaker uses Rust when available."""
        from unittest.mock import MagicMock
        from risk.kill_switch import KillSwitch
        from risk.drawdown_breaker import DrawdownBreakerConfig, DrawdownCircuitBreaker
        ks = MagicMock(spec=KillSwitch)
        b = DrawdownCircuitBreaker(ks, DrawdownBreakerConfig())
        assert b._use_rust is True

    def test_python_wrapper_escalation_calls_kill_switch(self):
        """Verify DrawdownCircuitBreaker wrapper calls KillSwitch correctly."""
        from unittest.mock import MagicMock
        from risk.kill_switch import KillSwitch
        from risk.drawdown_breaker import DrawdownBreakerConfig, DrawdownCircuitBreaker
        ks = MagicMock(spec=KillSwitch)
        cfg = DrawdownBreakerConfig(velocity_pct=99.0)
        b = DrawdownCircuitBreaker(ks, cfg)

        b.on_equity_update(1000.0, 0.0)
        b.on_equity_update(840.0, 1.0)  # → reduce_only (16% dd)
        ks.trigger.assert_called()

    def test_python_wrapper_hard_kill(self):
        """KillSwitch triggered with HARD_KILL on kill threshold breach."""
        from unittest.mock import MagicMock
        from risk.kill_switch import KillSwitch, KillMode
        from risk.drawdown_breaker import DrawdownBreakerConfig, DrawdownCircuitBreaker
        ks = MagicMock(spec=KillSwitch)
        cfg = DrawdownBreakerConfig(velocity_pct=99.0)
        b = DrawdownCircuitBreaker(ks, cfg)

        b.on_equity_update(1000.0, 0.0)
        b.on_equity_update(790.0, 1.0)  # → killed (21% dd)
        # Verify trigger was called with HARD_KILL mode
        calls = ks.trigger.call_args_list
        assert len(calls) >= 1
        last_call = calls[-1]
        kwargs = last_call[1] if last_call[1] else {}
        args = last_call[0] if last_call[0] else ()
        # mode can be positional or keyword
        mode_val = kwargs.get("mode") or (args[2] if len(args) > 2 else None)
        assert mode_val == KillMode.HARD_KILL

    def test_python_wrapper_reduce_only(self):
        """KillSwitch triggered with REDUCE_ONLY on reduce threshold breach."""
        from unittest.mock import MagicMock
        from risk.kill_switch import KillSwitch, KillMode
        from risk.drawdown_breaker import DrawdownBreakerConfig, DrawdownCircuitBreaker
        ks = MagicMock(spec=KillSwitch)
        cfg = DrawdownBreakerConfig(velocity_pct=99.0)
        b = DrawdownCircuitBreaker(ks, cfg)

        b.on_equity_update(1000.0, 0.0)
        b.on_equity_update(840.0, 1.0)  # → reduce_only (16% dd)
        calls = ks.trigger.call_args_list
        assert len(calls) >= 1
        last_call = calls[-1]
        kwargs = last_call[1] if last_call[1] else {}
        args = last_call[0] if last_call[0] else ()
        mode_val = kwargs.get("mode") or (args[2] if len(args) > 2 else None)
        assert mode_val == KillMode.REDUCE_ONLY

    def test_python_wrapper_reset_clears_kill_switch(self):
        """reset() calls kill_switch.clear()."""
        from unittest.mock import MagicMock
        from risk.kill_switch import KillSwitch
        from risk.drawdown_breaker import DrawdownBreakerConfig, DrawdownCircuitBreaker
        ks = MagicMock(spec=KillSwitch)
        cfg = DrawdownBreakerConfig(velocity_pct=99.0)
        b = DrawdownCircuitBreaker(ks, cfg)

        b.on_equity_update(1000.0, 0.0)
        b.on_equity_update(790.0, 1.0)  # killed
        b.reset()
        ks.clear.assert_called()

    def test_python_wrapper_checkpoint_restore(self):
        """checkpoint() and restore_checkpoint() work via Rust path."""
        from unittest.mock import MagicMock
        from risk.kill_switch import KillSwitch
        from risk.drawdown_breaker import DrawdownBreakerConfig, DrawdownCircuitBreaker
        ks = MagicMock(spec=KillSwitch)
        cfg = DrawdownBreakerConfig(velocity_pct=99.0)
        b = DrawdownCircuitBreaker(ks, cfg)

        b.on_equity_update(1000.0, 0.0)
        ckpt = b.checkpoint()
        assert ckpt["equity_hwm"] == pytest.approx(1000.0)

        b2 = DrawdownCircuitBreaker(ks, cfg)
        b2.restore_checkpoint(ckpt)
        assert b2.equity_hwm == pytest.approx(1000.0)

    def test_python_wrapper_get_status(self):
        """get_status() returns correct structure."""
        from unittest.mock import MagicMock
        from risk.kill_switch import KillSwitch
        from risk.drawdown_breaker import DrawdownBreakerConfig, DrawdownCircuitBreaker
        ks = MagicMock(spec=KillSwitch)
        cfg = DrawdownBreakerConfig(velocity_pct=99.0, warning_pct=5.0)
        b = DrawdownCircuitBreaker(ks, cfg)

        b.on_equity_update(10000.0, 0.0)
        b.on_equity_update(9500.0, 1.0)  # 5% dd → warning
        status = b.get_status()
        assert status["state"] == "warning"
        assert "drawdown_pct" in status
        assert "equity_hwm" in status
        assert "thresholds" in status

    def test_python_wrapper_properties(self):
        """state, current_drawdown_pct, equity_hwm properties work."""
        from unittest.mock import MagicMock
        from risk.kill_switch import KillSwitch
        from risk.drawdown_breaker import DrawdownBreakerConfig, DrawdownCircuitBreaker
        ks = MagicMock(spec=KillSwitch)
        cfg = DrawdownBreakerConfig(velocity_pct=99.0)
        b = DrawdownCircuitBreaker(ks, cfg)

        b.on_equity_update(1000.0, 0.0)
        assert b.equity_hwm == pytest.approx(1000.0)
        assert b.state == "normal"
        assert b.current_drawdown_pct == pytest.approx(0.0)

    def test_rust_vs_python_threshold_parity(self):
        """Rust and Python paths produce the same state transitions."""
        from risk.drawdown_breaker import DrawdownBreakerConfig
        py_b, _ = make_py_breaker(DrawdownBreakerConfig(velocity_pct=99.0))
        rust = RustDrawdownBreaker(warning_pct=10.0, reduce_pct=15.0, kill_pct=20.0, velocity_pct=99.0)

        equity_sequence = [
            (1000.0, 0.0),   # normal
            (950.0, 1.0),    # normal (5% dd)
            (890.0, 2.0),    # warning (11% dd)
            (920.0, 3.0),    # normal (recovery < 10%)
            (840.0, 4.0),    # reduce_only (16% dd)
        ]

        for eq, ts in equity_sequence:
            py_state = py_b.on_equity_update(eq, ts)
            rust_state, _ = rust.on_equity_update(eq, ts)
            assert py_state == rust_state, (
                f"Parity mismatch at equity={eq}, ts={ts}: "
                f"Python={py_state}, Rust={rust_state}"
            )

    def test_rust_vs_python_velocity_parity(self):
        """Velocity detection matches between Python and Rust."""
        from risk.drawdown_breaker import DrawdownBreakerConfig
        py_b, _ = make_py_breaker(DrawdownBreakerConfig(
            velocity_pct=5.0, velocity_window_sec=900.0,
            warning_pct=99.0, reduce_pct=99.0, kill_pct=99.0,
        ))
        rust = RustDrawdownBreaker(
            velocity_pct=5.0, velocity_window_sec=900.0,
            warning_pct=99.0, reduce_pct=99.0, kill_pct=99.0,
        )

        py_b.on_equity_update(1000.0, 0.0)
        rust.on_equity_update(1000.0, 0.0)

        # 6% drop within window → both should kill
        py_state = py_b.on_equity_update(940.0, 60.0)
        rust_state, action = rust.on_equity_update(940.0, 60.0)

        assert py_state == rust_state == "killed"
        assert action is not None
        assert action[0] == "hard_kill"
