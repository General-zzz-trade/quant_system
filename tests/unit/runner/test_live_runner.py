# tests/unit/runner/test_live_runner.py
"""Tests for LiveRunner — full production live trading stack."""
from __future__ import annotations

import time
import threading
from typing import Any, List, Optional
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest


from research.model_registry.registry import ModelRegistry
from monitoring.alerts.base import Alert, Severity
from monitoring.alerts.manager import AlertManager
from runner.live_runner import LiveRunner, LiveRunnerConfig, _reconcile_startup
from risk.kill_switch import KillMode, KillSwitch
from risk.margin_monitor import MarginConfig, MarginMonitor
from execution.store.event_log import InMemoryEventLog


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


class _RecordingAlertSink:
    def __init__(self) -> None:
        self.alerts: List[Any] = []

    def emit(self, alert: Any) -> None:
        self.alerts.append(alert)


class _TimeoutTracker:
    def __init__(self, timed_out: list[str], timeout_sec: float = 30.0) -> None:
        self._timed_out = list(timed_out)
        self.timeout_sec = timeout_sec

    def check_timeouts(self) -> list[str]:
        current = list(self._timed_out)
        self._timed_out.clear()
        return current


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
        baseline_threads = {
            t.ident for t in threading.enumerate() if t.ident is not None
        }

        t = threading.Thread(target=runner.start, daemon=True)
        t.start()

        deadline = time.monotonic() + 3.0
        while not runner._running and time.monotonic() < deadline:
            time.sleep(0.05)

        assert runner._running is True
        runner.stop()
        t.join(timeout=3.0)
        assert runner._running is False
        assert not t.is_alive()
        leaked = [
            th.name
            for th in threading.enumerate()
            if th.ident is not None
            and th.ident not in baseline_threads
            and th is not threading.main_thread()
        ]
        assert leaked == []

    def test_start_startup_failure_stops_already_started_subsystems(self, monkeypatch):
        monkeypatch.setattr(LiveRunner, "_apply_perf_tuning", staticmethod(lambda: None))

        class _FakeCoordinator:
            def __init__(self) -> None:
                self.started = 0
                self.stopped = 0

            def start(self) -> None:
                self.started += 1

            def stop(self) -> None:
                self.stopped += 1

        class _FakeLoop:
            def __init__(self) -> None:
                self.started = 0
                self.stopped = 0

            def start_background(self) -> None:
                self.started += 1

            def stop_background(self) -> None:
                self.stopped += 1

        class _FakeRuntime:
            def __init__(self) -> None:
                self.stopped = 0

            def start(self) -> None:
                raise RuntimeError("runtime boom")

            def stop(self) -> None:
                self.stopped += 1

        class _FakeMonitor:
            def __init__(self) -> None:
                self.started = 0
                self.stopped = 0

            def start(self) -> None:
                self.started += 1

            def stop(self) -> None:
                self.stopped += 1

        class _FakeAlertManager:
            def __init__(self) -> None:
                self.started = 0
                self.stopped = 0

            def start_periodic(self) -> None:
                self.started += 1

            def stop(self) -> None:
                self.stopped += 1

        class _FakeCheckpointer:
            def __init__(self) -> None:
                self.started = 0
                self.stopped = 0

            def start(self) -> None:
                self.started += 1

            def stop(self) -> None:
                self.stopped += 1

        coordinator = _FakeCoordinator()
        loop = _FakeLoop()
        runtime = _FakeRuntime()
        health = _FakeMonitor()
        alert_manager = _FakeAlertManager()
        checkpointer = _FakeCheckpointer()
        runner = LiveRunner(
            loop=loop,
            coordinator=coordinator,
            runtime=runtime,
            kill_switch=KillSwitch(),
            health=health,
            alert_manager=alert_manager,
            periodic_checkpointer=checkpointer,
        )

        with pytest.raises(RuntimeError, match="runtime boom"):
            runner.start()

        assert coordinator.started == 1
        assert coordinator.stopped == 1
        assert health.started == 1
        assert health.stopped == 1
        assert alert_manager.started == 1
        assert alert_manager.stopped == 1
        assert checkpointer.started == 1
        assert checkpointer.stopped == 1
        assert runtime.stopped == 1
        assert loop.started == 0
        assert loop.stopped == 1

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


@pytest.mark.forked
class TestOperatorControl:
    def test_halt_sets_global_hard_kill(self):
        runner = LiveRunner.build(
            LiveRunnerConfig(),
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )

        rec = runner.halt(reason="manual_halt")

        assert rec.mode == KillMode.HARD_KILL
        assert rec.reason == "manual_halt"
        assert runner.kill_switch.is_killed() is not None

    def test_reduce_only_sets_global_reduce_only(self):
        runner = LiveRunner.build(
            LiveRunnerConfig(),
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )

        rec = runner.reduce_only(reason="manual_reduce_only")

        assert rec.mode == KillMode.REDUCE_ONLY
        assert rec.reason == "manual_reduce_only"
        assert runner.kill_switch.is_killed() is not None

    def test_resume_clears_global_kill_switch(self):
        runner = LiveRunner.build(
            LiveRunnerConfig(),
            venue_clients={"binance": _FakeVenueClient()},
            transport=_FakeTransport(),
        )
        runner.halt(reason="manual_halt")

        assert runner.resume(reason="manual_resume") is True
        assert runner.kill_switch.is_killed() is None

    def test_flush_runs_reconcile_once_when_available(self):
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            reconcile_scheduler=SimpleNamespace(run_once=lambda: {"ok": True}),
        )

        assert runner.flush(reason="manual_flush") == {"ok": True}

    def test_apply_control_dispatches_commands(self):
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            reconcile_scheduler=SimpleNamespace(run_once=lambda: {"ok": True}),
        )

        halt_result = runner.apply_control(SimpleNamespace(command="halt", reason="manual stop"))
        assert halt_result.mode == KillMode.HARD_KILL
        assert runner.kill_switch.is_killed() is not None

        ro_result = runner.apply_control(SimpleNamespace(command="reduce_only", reason="manual ro"))
        assert ro_result.mode == KillMode.REDUCE_ONLY
        assert runner.kill_switch.is_killed() is not None

        flush_result = runner.apply_control(SimpleNamespace(command="flush", reason="manual flush"))
        assert flush_result == {"ok": True}

        assert runner.apply_control(SimpleNamespace(command="resume", reason="manual resume")) is True
        assert runner.kill_switch.is_killed() is None

    def test_apply_control_shutdown_stops_runner(self):
        runtime = MagicMock()
        loop = MagicMock()
        coordinator = MagicMock()
        runner = LiveRunner(
            loop=loop,
            coordinator=coordinator,
            runtime=runtime,
            kill_switch=KillSwitch(),
        )

        runner.apply_control(SimpleNamespace(command="shutdown", reason="manual shutdown"))

        runtime.stop.assert_called_once()
        loop.stop_background.assert_called_once()
        coordinator.stop.assert_called_once()

    def test_apply_control_rejects_unknown_command(self):
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
        )

        with pytest.raises(ValueError, match="Unsupported control command"):
            runner.apply_control(SimpleNamespace(command="bogus", reason="bad"))

    def test_operator_status_reflects_kill_switch_reconcile_and_last_control(self):
        report = SimpleNamespace(ok=False, should_halt=True, all_drifts=("d1", "d2"))
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            reconcile_scheduler=SimpleNamespace(last_report=report, run_once=lambda: report),
        )

        runner.reduce_only(reason="manual_reduce_only")
        status = runner.operator_status()

        assert status["kill_switch"] is not None
        assert status["kill_switch"]["mode"] == "reduce_only"
        assert status["stream_status"] == "ok"
        assert status["incident_state"] == "critical"
        assert status["recommended_action"] == "halt"
        assert status["last_reconcile"] == {"ok": False, "should_halt": True, "drift_count": 2}
        assert status["last_control"]["command"] == "reduce_only"
        assert status["last_control"]["result"] == "reduce_only"
        assert status["last_incident_category"] == "operator_control"
        assert status["last_incident_ts"] is not None

    def test_operator_status_snapshot_exposes_structured_schema(self):
        report = SimpleNamespace(ok=True, should_halt=False, all_drifts=())
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            reconcile_scheduler=SimpleNamespace(last_report=report, run_once=lambda: report),
        )

        runner.halt(reason="manual_halt")
        snap = runner.operator_status_snapshot()

        assert snap.running is False
        assert snap.stopped is False
        assert snap.stream_status == "ok"
        assert snap.incident_state == "critical"
        assert snap.recommended_action == "halt"
        assert snap.kill_switch is not None
        assert snap.kill_switch.mode == "hard_kill"
        assert snap.last_reconcile is not None
        assert snap.last_reconcile.ok is True
        assert snap.last_control is not None
        assert snap.last_control.command == "halt"
        assert snap.last_incident_category == "operator_control"
        assert snap.last_incident_ts is not None

    def test_operator_status_marks_reduce_only_as_degraded(self):
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
        )

        runner.reduce_only(reason="manual_reduce_only")
        status = runner.operator_status()

        assert status["stream_status"] == "ok"
        assert status["incident_state"] == "degraded"
        assert status["recommended_action"] == "reduce_only"

    def test_control_history_records_flush_outcome(self):
        report = SimpleNamespace(ok=False, should_halt=False, all_drifts=("d1",))
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            reconcile_scheduler=SimpleNamespace(last_report=report, run_once=lambda: report),
        )

        runner.flush(reason="manual_flush")

        assert runner.control_history[-1].command == "flush"
        assert runner.control_history[-1].result == "drift"

    def test_control_actions_emit_alerts(self):
        sink = _RecordingAlertSink()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=sink),
            reconcile_scheduler=SimpleNamespace(last_report=None, run_once=lambda: SimpleNamespace(ok=False,
                should_halt=False, all_drifts=("d1",))),
        )

        runner.halt(reason="manual_halt")
        runner.flush(reason="manual_flush")
        runner.resume(reason="manual_resume")

        assert [a.title for a in sink.alerts] == [
            "operator_halt",
            "operator_flush",
            "execution-reconcile-drift",
            "operator_resume",
        ]
        assert sink.alerts[0].severity == Severity.CRITICAL
        assert sink.alerts[1].severity == Severity.WARNING
        assert sink.alerts[2].severity == Severity.WARNING
        assert sink.alerts[3].severity == Severity.INFO

    def test_control_actions_append_to_event_log_when_available(self):
        rows: list[dict[str, object]] = []

        class _EventLog:
            def append(self, *, event_type, payload, correlation_id=None):
                rows.append(
                    {
                        "event_type": event_type,
                        "payload": dict(payload),
                        "correlation_id": correlation_id,
                    }
                )
                return len(rows)

        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            event_log=_EventLog(),
        )

        runner.halt(reason="manual_halt", source="ops")

        assert rows and rows[-1]["event_type"] == "operator_control"
        assert rows[-1]["correlation_id"] == "halt"
        assert rows[-1]["payload"]["result"] == "hard_kill"

    def test_flush_with_drift_emits_reconcile_alert(self):
        sink = _RecordingAlertSink()
        report = SimpleNamespace(ok=False, should_halt=False, all_drifts=(SimpleNamespace(symbol="BTCUSDT"),),
            venue="binance")
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=sink),
            reconcile_scheduler=SimpleNamespace(last_report=report, run_once=lambda: report),
        )

        runner.flush(reason="manual_flush")

        titles = [a.title for a in sink.alerts]
        assert "operator_flush" in titles
        assert "execution-reconcile-drift" in titles

    def test_flush_with_drift_persists_execution_incident_to_event_log(self):
        rows: list[dict[str, object]] = []

        class _EventLog:
            def append(self, *, event_type, payload, correlation_id=None):
                rows.append(
                    {
                        "event_type": event_type,
                        "payload": dict(payload),
                        "correlation_id": correlation_id,
                    }
                )
                return len(rows)

        report = SimpleNamespace(ok=False, should_halt=False, all_drifts=(SimpleNamespace(symbol="BTCUSDT"),),
            venue="binance")
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            event_log=_EventLog(),
            reconcile_scheduler=SimpleNamespace(last_report=report, run_once=lambda: report),
            alert_manager=AlertManager(sink=_RecordingAlertSink()),
        )

        runner.flush(reason="manual_flush")

        assert any(row["event_type"] == "execution_incident" for row in rows)
        incident = next(row for row in rows if row["event_type"] == "execution_incident")
        assert incident["payload"]["title"] == "execution-reconcile-drift"
        assert incident["payload"]["category"] == "execution_reconcile"

    def test_start_emits_timeout_alerts_for_timed_out_orders(self, monkeypatch):
        monkeypatch.setattr(LiveRunner, "_apply_perf_tuning", staticmethod(lambda: None))

        class _FakeCoordinator:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        class _FakeLoop:
            def __init__(self) -> None:
                self.owner = None

            def start_background(self) -> None:
                pass

            def stop_background(self) -> None:
                pass

        class _FakeRuntime:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        tracker = _TimeoutTracker(["ord-timeout-1"], timeout_sec=15.0)
        original_check = tracker.check_timeouts

        def _check_timeouts_once() -> list[str]:
            result = original_check()
            if loop.owner is not None:
                loop.owner._running = False
            return result

        sink = _RecordingAlertSink()
        loop = _FakeLoop()
        tracker.check_timeouts = _check_timeouts_once  # type: ignore[method-assign]
        runner = LiveRunner(
            loop=loop,
            coordinator=_FakeCoordinator(),
            runtime=_FakeRuntime(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=sink),
            timeout_tracker=tracker,
        )
        loop.owner = runner
        runner._config = LiveRunnerConfig(venue="binance")

        runner.start()

        assert any(a.title == "execution-timeout" for a in sink.alerts)

    def test_start_persists_timeout_incident_to_event_log(self, monkeypatch):
        monkeypatch.setattr(LiveRunner, "_apply_perf_tuning", staticmethod(lambda: None))

        class _FakeCoordinator:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        class _FakeLoop:
            def __init__(self) -> None:
                self.owner = None

            def start_background(self) -> None:
                pass

            def stop_background(self) -> None:
                pass

        class _FakeRuntime:
            def start(self) -> None:
                pass

            def stop(self) -> None:
                pass

        class _EventLog:
            def __init__(self) -> None:
                self.rows: list[dict[str, object]] = []

            def append(self, *, event_type, payload, correlation_id=None):
                self.rows.append(
                    {
                        "event_type": event_type,
                        "payload": dict(payload),
                        "correlation_id": correlation_id,
                    }
                )
                return len(self.rows)

            def list_recent(self, *, event_type: str | None = None, limit: int = 20):
                rows = list(self.rows)
                if event_type is not None:
                    rows = [r for r in rows if r["event_type"] == event_type]
                return rows[-limit:][::-1]

        tracker = _TimeoutTracker(["ord-timeout-1"], timeout_sec=15.0)
        original_check = tracker.check_timeouts

        def _check_timeouts_once() -> list[str]:
            result = original_check()
            if loop.owner is not None:
                loop.owner._running = False
            return result

        sink = _RecordingAlertSink()
        loop = _FakeLoop()
        event_log = _EventLog()
        tracker.check_timeouts = _check_timeouts_once  # type: ignore[method-assign]
        runner = LiveRunner(
            loop=loop,
            coordinator=_FakeCoordinator(),
            runtime=_FakeRuntime(),
            kill_switch=KillSwitch(),
            event_log=event_log,
            alert_manager=AlertManager(sink=sink),
            timeout_tracker=tracker,
        )
        loop.owner = runner
        runner._config = LiveRunnerConfig(venue="binance")

        runner.start()

        assert any(a.title == "execution-timeout" for a in sink.alerts)
        assert any(row["event_type"] == "execution_incident" for row in event_log.rows)
        incident = next(row for row in event_log.rows if row["event_type"] == "execution_incident")
        assert incident["payload"]["title"] == "execution-timeout"
        assert incident["payload"]["category"] == "execution_timeout"

    def test_execution_alert_history_filters_execution_categories(self):
        sink = _RecordingAlertSink()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=sink),
        )

        runner.alert_manager.emit_direct(Alert(
            title="execution-timeout",
            message="timeout",
            severity=Severity.WARNING,
            source="execution:test",
            meta={"category": "execution_timeout"},
        ))
        runner.alert_manager.emit_direct(Alert(
            title="operator_halt",
            message="manual halt",
            severity=Severity.CRITICAL,
            source="runner:control",
            meta={"category": "operator_control"},
        ))

        rows = runner.execution_alert_history()

        assert len(rows) == 1
        assert rows[0]["title"] == "execution-timeout"

    def test_model_alert_history_filters_model_categories(self):
        sink = _RecordingAlertSink()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=sink),
        )

        runner.alert_manager.emit_direct(Alert(
            title="model_reload_failed",
            message="reload failed",
            severity=Severity.ERROR,
            source="model:test",
            meta={"category": "model_reload", "outcome": "failed"},
        ))
        runner.alert_manager.emit_direct(Alert(
            title="execution-timeout",
            message="timeout",
            severity=Severity.WARNING,
            source="execution:test",
            meta={"category": "execution_timeout"},
        ))

        rows = runner.model_alert_history()

        assert len(rows) == 1
        assert rows[0]["title"] == "model_reload_failed"

    def test_operator_status_reports_degraded_stream_after_user_stream_failure(self):
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=_RecordingAlertSink()),
        )
        runner._config = LiveRunnerConfig(venue="binance")
        runner.user_stream = object()

        runner._record_user_stream_failure(kind="step")

        status = runner.operator_status()

        assert status["stream_status"] == "degraded"
        assert status["incident_state"] == "degraded"
        assert status["recommended_action"] == "review"
        assert status["last_incident_category"] == "execution_stream"
        assert status["last_incident_ts"] is not None

    def test_ops_audit_snapshot_exposes_incident_aggregate(self):
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=_RecordingAlertSink()),
        )
        runner._config = LiveRunnerConfig(venue="binance")
        runner.user_stream = object()

        runner._record_user_stream_failure(kind="connect")
        snapshot = runner.ops_audit_snapshot()

        assert snapshot["stream_status"] == "degraded"
        assert snapshot["incident_state"] == "degraded"
        assert snapshot["recommended_action"] == "review"
        assert snapshot["last_incident_category"] == "execution_stream"
        assert snapshot["last_incident_ts"] is not None
        assert snapshot["operator"]["stream_status"] == "degraded"

    def test_ops_audit_snapshot_collects_all_configured_model_actions(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry.db")
        mv_btc = registry.register(name="alpha_btc", params={}, features=[], metrics={"sharpe": 1.0})
        mv_eth = registry.register(name="alpha_eth", params={}, features=[], metrics={"sharpe": 1.1})
        registry.promote(mv_btc.model_id, reason="btc_go", actor="ops")
        registry.promote(mv_eth.model_id, reason="eth_go", actor="ops")

        sink = _RecordingAlertSink()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=sink),
        )
        runner._config = LiveRunnerConfig(
            model_registry_db=str(tmp_path / "registry.db"),
            model_names=("alpha_btc", "alpha_eth"),
        )

        snapshot = runner.ops_audit_snapshot()

        names = {row["model"] for row in snapshot["model_actions"]}
        assert names == {"alpha_btc", "alpha_eth"}

    def test_ops_audit_snapshot_includes_model_status_from_loader(self):
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
        )
        runner._config = LiveRunnerConfig(model_registry_db="registry.db", model_names=("alpha_btc",))
        runner.model_loader = SimpleNamespace(
            inspect_production_models=lambda names: [
                {
                    "name": "alpha_btc",
                    "available": True,
                    "model_id": "m-2",
                    "version": 2,
                    "loaded_model_id": "m-1",
                    "autoload_pending": True,
                }
            ]
        )

        snapshot = runner.ops_audit_snapshot()

        assert snapshot["model_status"][0]["name"] == "alpha_btc"
        assert snapshot["model_status"][0]["autoload_pending"] is True
        assert snapshot["model_reload"] is None

    def test_ops_audit_snapshot_includes_last_model_reload_status(self):
        sink = _RecordingAlertSink()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=sink),
        )
        runner._record_model_reload(
            outcome="reloaded",
            model_names=("alpha_btc",),
            detail={"reloaded_count": 1},
        )

        snapshot = runner.ops_audit_snapshot()

        assert snapshot["model_reload"] is not None
        assert snapshot["model_reload"]["outcome"] == "reloaded"
        assert snapshot["model_reload"]["model_names"] == ["alpha_btc"]
        assert snapshot["model_reload"]["detail"]["reloaded_count"] == 1
        assert snapshot["model_alerts"][0]["title"] == "model_reload_reloaded"

    def test_ops_audit_snapshot_includes_failed_model_reload_status(self):
        sink = _RecordingAlertSink()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=sink),
        )
        runner._record_model_reload(
            outcome="failed",
            model_names=("alpha_btc",),
            detail=None,
            error="model_hot_reload_failed",
        )

        snapshot = runner.ops_audit_snapshot()

        assert snapshot["model_reload"] is not None
        assert snapshot["model_reload"]["outcome"] == "failed"
        assert snapshot["model_reload"]["model_names"] == ["alpha_btc"]
        assert snapshot["model_reload"]["detail"] is None
        assert snapshot["model_reload"]["error"] == "model_hot_reload_failed"
        assert snapshot["model_alerts"][0]["title"] == "model_reload_failed"

    def test_ops_timeline_merges_control_execution_and_model_rows(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry.db")
        mv = registry.register(name="alpha_btc", params={}, features=[], metrics={"sharpe": 1.0})
        registry.promote(mv.model_id, reason="shadow_win", actor="ops")

        sink = _RecordingAlertSink()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            alert_manager=AlertManager(sink=sink),
        )
        runner._config = LiveRunnerConfig(
            model_registry_db=str(tmp_path / "registry.db"),
            model_names=("alpha_btc",),
        )

        runner.halt(reason="manual_halt", source="ops")
        runner.alert_manager.emit_direct(Alert(
            title="execution-timeout",
            message="timeout",
            severity=Severity.WARNING,
            source="execution:test",
            meta={"category": "execution_timeout"},
        ))
        runner._record_model_reload(
            outcome="failed",
            model_names=("alpha_btc",),
            detail=None,
            error="model_hot_reload_failed",
        )

        timeline = runner.ops_timeline()

        kinds = {row["kind"] for row in timeline}
        assert "control" in kinds
        assert "execution_alert" in kinds
        assert "model_alert" in kinds
        assert "model_action" in kinds
        assert timeline == sorted(timeline, key=lambda row: row["ts"], reverse=True)

    def test_ops_timeline_uses_event_log_for_persistent_control_and_model_reload(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry.db")
        mv = registry.register(name="alpha_btc", params={}, features=[], metrics={"sharpe": 1.0})
        registry.promote(mv.model_id, reason="shadow_win", actor="ops")

        event_log = InMemoryEventLog()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            event_log=event_log,
        )
        runner._config = LiveRunnerConfig(
            model_registry_db=str(tmp_path / "registry.db"),
            model_names=("alpha_btc",),
        )

        runner.halt(reason="manual_halt", source="ops")
        runner._emit_execution_incident(
            Alert(
                title="execution-timeout",
                message="timeout",
                severity=Severity.WARNING,
                source="execution:test",
                meta={"category": "execution_timeout", "routing_key": "binance:BTCUSDT:timeout"},
            )
        )
        runner._record_model_reload(
            outcome="failed",
            model_names=("alpha_btc",),
            detail=None,
            error="model_hot_reload_failed",
        )

        timeline = runner.ops_timeline()

        kinds = {row["kind"] for row in timeline}
        assert "control" in kinds
        assert "execution_incident" in kinds
        assert "model_reload" in kinds
        assert "model_action" in kinds
        assert any(row["title"] == "operator_halt" for row in timeline)
        assert any(row["title"] == "model_reload_failed" for row in timeline)

    def test_emit_execution_incident_persists_fill_and_rejection_categories(self):
        event_log = InMemoryEventLog()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            event_log=event_log,
        )

        runner._emit_execution_incident(
            Alert(
                title="execution-synthetic-fill",
                message="synthetic fill",
                severity=Severity.INFO,
                source="execution:bridge",
                meta={"category": "execution_fill", "routing_key": "binance:BTCUSDT:fill"},
            )
        )
        runner._emit_execution_incident(
            Alert(
                title="execution-rejected",
                message="rejected",
                severity=Severity.ERROR,
                source="execution:bridge",
                meta={"category": "execution_rejection", "routing_key": "binance:BTCUSDT:reject"},
            )
        )

        rows = event_log.list_recent(event_type="execution_incident", limit=10)

        categories = {row["payload"]["category"] for row in rows}
        assert "execution_fill" in categories
        assert "execution_rejection" in categories
        assert any(row["payload"]["title"] == "execution-synthetic-fill" for row in rows)
        assert any(row["payload"]["title"] == "execution-rejected" for row in rows)

    def test_ops_timeline_limit_is_sorted_and_trimmed(self, tmp_path):
        registry = ModelRegistry(tmp_path / "registry.db")
        mv = registry.register(name="alpha_btc", params={}, features=[], metrics={"sharpe": 1.0})
        registry.promote(mv.model_id, reason="shadow_win", actor="ops")

        event_log = InMemoryEventLog()
        runner = LiveRunner(
            loop=MagicMock(),
            coordinator=MagicMock(),
            runtime=MagicMock(),
            kill_switch=KillSwitch(),
            event_log=event_log,
        )
        runner._config = LiveRunnerConfig(
            model_registry_db=str(tmp_path / "registry.db"),
            model_names=("alpha_btc",),
        )

        event_log.append(
            event_type="operator_control",
            correlation_id="halt",
            payload={
                "command": "halt",
                "reason": "manual_halt",
                "source": "ops",
                "result": "hard_kill",
                "ts": "2099-03-13T00:00:01+00:00",
            },
        )
        event_log.append(
            event_type="execution_incident",
            correlation_id="binance:BTCUSDT:timeout",
            payload={
                "title": "execution-timeout",
                "category": "execution_timeout",
                "source": "execution:test",
                "ts": "2099-03-13T00:00:02+00:00",
            },
        )
        event_log.append(
            event_type="model_reload",
            correlation_id="failed",
            payload={
                "outcome": "failed",
                "model_names": ["alpha_btc"],
                "error": "model_hot_reload_failed",
                "ts": "2099-03-13T00:00:03+00:00",
            },
        )

        timeline = runner.ops_timeline(limit=2)

        assert len(timeline) == 2
        assert timeline == sorted(timeline, key=lambda row: row["ts"], reverse=True)
        assert timeline[0]["kind"] == "model_reload"
        assert timeline[1]["kind"] == "execution_incident"


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
