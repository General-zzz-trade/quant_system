from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from monitoring.alerts.manager import AlertManager
from risk.kill_switch import KillSwitch
from runner.control_plane import OperatorControlPlane, OperatorControlRequest
from runner.live_runner import LiveRunner


class _RecordingSink:
    def __init__(self) -> None:
        self.alerts: list[object] = []

    def emit(self, alert: object) -> None:
        self.alerts.append(alert)


def test_control_plane_accepts_mapping_request_and_returns_stable_result() -> None:
    runner = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
    )
    plane = OperatorControlPlane(runner)

    result = plane.execute({"command": "halt", "reason": "manual_halt", "source": "api"})

    assert result.accepted is True
    assert result.command == "halt"
    assert result.outcome == "hard_kill"
    assert result.source == "api"
    assert result.status is not None
    assert result.status["kill_switch"]["mode"] == "hard_kill"


def test_control_plane_returns_rejected_result_for_unknown_command() -> None:
    runner = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
    )
    plane = OperatorControlPlane(runner)

    result = plane.execute(OperatorControlRequest(command="bogus", reason="bad"))

    assert result.accepted is False
    assert result.command == "bogus"
    assert result.outcome == "rejected"
    assert result.error_code == "invalid_command"
    assert "unsupported control command" in (result.error or "")


def test_control_plane_rejects_missing_command() -> None:
    runner = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
    )
    plane = OperatorControlPlane(runner)

    result = plane.execute({"reason": "bad"})

    assert result.accepted is False
    assert result.error_code == "missing_command"


def test_control_plane_lists_available_commands() -> None:
    assert OperatorControlPlane.available_commands() == (
        "halt",
        "reduce_only",
        "resume",
        "flush",
        "shutdown",
    )


def test_control_plane_flush_returns_drift_outcome_and_status() -> None:
    report = SimpleNamespace(ok=False, should_halt=True, all_drifts=("d1", "d2"))
    sink = _RecordingSink()
    runner = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
        reconcile_scheduler=SimpleNamespace(last_report=report, run_once=lambda: report),
        alert_manager=AlertManager(sink=sink),
    )
    plane = OperatorControlPlane(runner)

    result = plane.execute({"command": "flush", "reason": "manual_flush", "source": "api"})

    assert result.accepted is True
    assert result.outcome == "drift"
    assert result.status is not None
    assert result.status["last_reconcile"] == {"ok": False, "should_halt": True, "drift_count": 2}
    # Flush runs reconcile, which produces a reconcile-drift alert (not operator_flush)
    assert sink.alerts[-1].title == "execution-reconcile-drift"
