# tests/unit/runner/test_live_runner.py
"""Tests for LiveRunner — full production live trading stack."""
from __future__ import annotations

import time
import threading
from typing import Any, List, Optional
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from runner.live_runner import LiveRunner, LiveRunnerConfig, _reconcile_startup
from risk.kill_switch import KillMode, KillScope, KillSwitch
from risk.margin_monitor import MarginConfig, MarginMonitor


# ── Fake transport ────────────────────────────────────────────

class _FakeTransport:
    """Minimal WsTransport stub for testing."""

    def __init__(self, messages: list[str] | None = None):
        self._messages = list(messages or [])
        self._idx = 0

    def connect(self, url: str) -> None:
        pass

    def recv(self, timeout_s: float = 5.0) -> Optional[str]:
        if self._idx >= len(self._messages):
            time.sleep(0.01)
            return None
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    def close(self) -> None:
        pass


class _FakeVenueClient:
    """Minimal venue client that records orders and returns empty results."""

    def __init__(self) -> None:
        self.orders: List[Any] = []

    def send_order(self, order_event: Any) -> list:
        self.orders.append(order_event)
        return []


# ── Build tests ────────────────────────────────────────────────

class TestBuild:
    def test_build_creates_all_components(self):
        config = LiveRunnerConfig(symbols=("BTCUSDT",))
        venue_client = _FakeVenueClient()

        runner = LiveRunner.build(
            config,
            venue_clients={"binance": venue_client},
            transport=_FakeTransport(),
        )

        assert runner.coordinator is not None
        assert runner.loop is not None
        assert runner.runtime is not None
        assert runner.kill_switch is not None
        assert runner.shutdown_handler is not None

    def test_build_with_monitoring(self):
        config = LiveRunnerConfig(enable_monitoring=True)
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.health is not None

    def test_build_without_monitoring(self):
        config = LiveRunnerConfig(enable_monitoring=False)
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.health is None

    def test_build_with_reconcile(self):
        config = LiveRunnerConfig(enable_reconcile=True)
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
            fetch_venue_state=lambda: {"positions": {}, "balances": {}},
        )
        assert runner.reconcile_scheduler is not None

    def test_build_without_reconcile_no_fetcher(self):
        config = LiveRunnerConfig(enable_reconcile=True)
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.reconcile_scheduler is None

    def test_build_with_margin_monitor(self):
        config = LiveRunnerConfig()
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
            fetch_margin=lambda: {"margin_ratio": 0.5},
        )
        assert runner.margin_monitor is not None

    def test_build_without_margin_monitor(self):
        config = LiveRunnerConfig()
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.margin_monitor is None

    def test_build_missing_venue_client_raises(self):
        config = LiveRunnerConfig(venue="binance")
        with pytest.raises(ValueError, match="No venue client"):
            LiveRunner.build(
                config,
                venue_clients={"other": _FakeVenueClient()},
                transport=_FakeTransport(),
            )

    def test_build_multi_symbol(self):
        config = LiveRunnerConfig(symbols=("BTCUSDT", "ETHUSDT"))
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        view = runner.coordinator.get_state_view()
        assert "BTCUSDT" in view["markets"]
        assert "ETHUSDT" in view["markets"]


# ── Lifecycle tests ────────────────────────────────────────────

class TestLifecycle:
    def test_stop_idempotent(self):
        config = LiveRunnerConfig()
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        runner.stop()
        runner.stop()

    def test_start_stop_in_background(self):
        config = LiveRunnerConfig()
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )

        t = threading.Thread(target=runner.start, daemon=True)
        t.start()

        deadline = time.monotonic() + 3.0
        while not runner._running and time.monotonic() < deadline:
            time.sleep(0.05)

        assert runner._running is True
        runner.stop()
        t.join(timeout=3.0)
        assert runner._running is False

    def test_start_reconnects_user_stream_after_step_error(self, monkeypatch):
        monkeypatch.setattr(LiveRunner, "_apply_perf_tuning", staticmethod(lambda: None))

        class _FakeCoordinator:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        class _FakeLoop:
            def start_background(self) -> None:
                pass

            def stop_background(self) -> None:
                pass

        class _FakeRuntime:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        class _FakeUserStream:
            def __init__(self, owner: LiveRunner) -> None:
                self.owner = owner
                self.connect_calls = 0
                self.step_calls = 0
                self.close_calls = 0

            def connect(self) -> None:
                self.connect_calls += 1

            def step(self) -> None:
                self.step_calls += 1
                if self.step_calls == 1:
                    raise RuntimeError("user stream dropped")
                self.owner._running = False

            def close(self) -> None:
                self.close_calls += 1

        runner = LiveRunner(
            loop=_FakeLoop(),
            coordinator=_FakeCoordinator(),
            runtime=_FakeRuntime(),
            kill_switch=KillSwitch(),
        )
        runner.user_stream = _FakeUserStream(runner)

        real_sleep = time.sleep

        def _fast_sleep(seconds: float) -> None:
            real_sleep(0.01)

        monkeypatch.setattr("runner.live_runner.time.sleep", _fast_sleep)

        runner.start()

        assert runner.user_stream.connect_calls >= 2
        assert runner.user_stream.step_calls >= 2
        assert runner.user_stream.close_calls == 1

    def test_start_checks_timeout_tracker_each_loop(self, monkeypatch):
        monkeypatch.setattr(LiveRunner, "_apply_perf_tuning", staticmethod(lambda: None))

        class _FakeCoordinator:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        class _FakeLoop:
            def start_background(self) -> None:
                pass

            def stop_background(self) -> None:
                pass

        class _FakeRuntime:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        class _FakeTimeoutTracker:
            def __init__(self, owner: LiveRunner) -> None:
                self.owner = owner
                self.calls = 0

            def check_timeouts(self) -> list[str]:
                self.calls += 1
                self.owner._running = False
                return ["order-1"]

        runner = LiveRunner(
            loop=_FakeLoop(),
            coordinator=_FakeCoordinator(),
            runtime=_FakeRuntime(),
            kill_switch=KillSwitch(),
        )
        runner.timeout_tracker = _FakeTimeoutTracker(runner)

        real_sleep = time.sleep

        def _fast_sleep(seconds: float) -> None:
            real_sleep(0.01)

        monkeypatch.setattr("runner.live_runner.time.sleep", _fast_sleep)

        runner.start()

        assert runner.timeout_tracker.calls >= 1


class TestPerfTuning:
    def test_apply_perf_tuning_ignores_null_nohz_cpu_list(self, monkeypatch):
        from io import StringIO

        monkeypatch.setattr("builtins.open", lambda *args, **kwargs: StringIO("(null)\n"))
        sched = MagicMock()
        nice = MagicMock()
        monkeypatch.setattr("runner.live_runner.os.sched_setaffinity", sched, raising=False)
        monkeypatch.setattr("runner.live_runner.os.nice", nice, raising=False)

        LiveRunner._apply_perf_tuning()

        sched.assert_not_called()
        nice.assert_called_once_with(-10)


class TestStartupReconcile:
    def test_detects_position_mismatch(self):
        mismatches = _reconcile_startup(
            local_view={
                "positions": {"BTCUSDT": SimpleNamespace(qty=1.0)},
                "account": SimpleNamespace(balance=1000.0),
            },
            venue_state={
                "positions": {"BTCUSDT": {"qty": 0.5}},
                "balance": 1000.0,
            },
            symbols=("BTCUSDT",),
        )

        assert mismatches == ["BTCUSDT position: local=1.0, venue=0.5"]

    def test_detects_balance_mismatch_from_account_view(self):
        mismatches = _reconcile_startup(
            local_view={
                "positions": {},
                "account": SimpleNamespace(balance=1000.0),
            },
            venue_state={
                "positions": {},
                "balance": 995.0,
            },
            symbols=("BTCUSDT",),
        )

        assert mismatches == ["Balance: local=1000.00, venue=995.00"]


# ── Fills tracking ─────────────────────────────────────────────

class TestFills:
    def test_fills_initially_empty(self):
        config = LiveRunnerConfig()
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        assert runner.fills == []

    def test_fills_returns_copy(self):
        config = LiveRunnerConfig()
        runner = LiveRunner.build(
            config,
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        f1 = runner.fills
        f2 = runner.fills
        assert f1 is not f2


# ── MarginMonitor (production) unit tests ─────────────────────

class TestMarginMonitor:
    def test_critical_triggers_kill_switch(self):
        ks = KillSwitch()

        monitor = MarginMonitor(
            config=MarginConfig(critical_margin_ratio=0.08, warning_margin_ratio=0.15),
            fetch_margin=lambda: {"margin_ratio": 0.05},
            kill_switch=ks,
        )

        status = monitor.check_once()
        assert status["margin_ok"] is False
        assert ks.is_killed() is not None

    def test_warning_does_not_trigger_kill_switch(self):
        ks = KillSwitch()

        monitor = MarginMonitor(
            config=MarginConfig(critical_margin_ratio=0.08, warning_margin_ratio=0.15),
            fetch_margin=lambda: {"margin_ratio": 0.12},
            kill_switch=ks,
        )

        status = monitor.check_once()
        assert ks.is_killed() is None
        assert len(status["alerts"]) == 1

    def test_healthy_margin_no_alert(self):
        ks = KillSwitch()

        monitor = MarginMonitor(
            config=MarginConfig(critical_margin_ratio=0.08, warning_margin_ratio=0.15),
            fetch_margin=lambda: {"margin_ratio": 0.50},
            kill_switch=ks,
        )

        status = monitor.check_once()
        assert ks.is_killed() is None
        assert len(status["alerts"]) == 0
