# runner/operator_control.py
"""Operator control mixin — halt, resume, flush, shutdown, apply_control."""
from __future__ import annotations

import logging
from typing import Any, Optional

from execution.observability.incidents import reconcile_report_to_alert
from monitoring.alerts.base import Severity
from risk.kill_switch import KillMode, KillScope

logger = logging.getLogger(__name__)


class OperatorControlMixin:
    """Mixin providing operator control methods for LiveRunner.

    Expects the following attributes on ``self``:
        kill_switch, reconcile_scheduler, alert_manager,
        _record_control(), _emit_control_alert(), _emit_execution_incident(),
        stop()
    """

    def halt(self, *, reason: str = "operator_halt", source: str = "operator") -> Any:
        """Trigger a global HARD_KILL without stopping the process."""
        rec = self.kill_switch.trigger(
            scope=KillScope.GLOBAL,
            key="*",
            mode=KillMode.HARD_KILL,
            reason=reason,
            source=source,
        )
        self._record_control(
            command="halt", reason=reason, source=source, result=rec.mode.value)
        self._emit_control_alert(
            command="halt", reason=reason, source=source,
            result=rec.mode.value, severity=Severity.CRITICAL)
        logger.warning("LiveRunner halted: reason=%s source=%s", reason, source)
        return rec

    def reduce_only(self, *, reason: str = "operator_reduce_only", source: str = "operator") -> Any:
        """Trigger a global REDUCE_ONLY gate without stopping the process."""
        rec = self.kill_switch.trigger(
            scope=KillScope.GLOBAL,
            key="*",
            mode=KillMode.REDUCE_ONLY,
            reason=reason,
            source=source,
        )
        self._record_control(
            command="reduce_only", reason=reason, source=source, result=rec.mode.value)
        self._emit_control_alert(
            command="reduce_only", reason=reason, source=source,
            result=rec.mode.value, severity=Severity.WARNING)
        logger.warning("LiveRunner reduce-only enabled: reason=%s source=%s", reason, source)
        return rec

    def resume(self, *, reason: str = "operator_resume") -> bool:
        """Clear the global kill-switch scope and resume order flow."""
        cleared = self.kill_switch.clear(scope=KillScope.GLOBAL, key="*")
        self._record_control(
            command="resume", reason=reason, source="operator",
            result="cleared" if cleared else "noop")
        self._emit_control_alert(
            command="resume",
            reason=reason,
            source="operator",
            result="cleared" if cleared else "noop",
            severity=Severity.INFO if cleared else Severity.WARNING,
        )
        logger.info("LiveRunner resume requested: cleared=%s reason=%s", cleared, reason)
        return cleared

    def flush(self, *, reason: str = "operator_flush") -> Optional[Any]:
        """Run one immediate reconcile pass when available."""
        logger.info("LiveRunner flush requested: reason=%s", reason)
        if self.reconcile_scheduler is None:
            self._record_control(
                command="flush", reason=reason, source="operator", result="unavailable")
            self._emit_control_alert(
                command="flush", reason=reason, source="operator",
                result="unavailable", severity=Severity.WARNING)
            return None
        report = self.reconcile_scheduler.run_once()
        result = "ok"
        severity = Severity.INFO
        if report is None:
            result = "failed"
            severity = Severity.ERROR
        elif hasattr(report, "ok") and not bool(getattr(report, "ok")):
            result = "drift"
            severity = Severity.WARNING
        self._record_control(
            command="flush", reason=reason, source="operator", result=result)
        self._emit_control_alert(
            command="flush", reason=reason, source="operator",
            result=result, severity=severity)
        if (report is not None and self.alert_manager is not None
                and hasattr(report, "ok") and not bool(getattr(report, "ok"))):
            try:
                self._emit_execution_incident(reconcile_report_to_alert(report))
            except Exception:
                logger.exception("reconcile alert emit failed during flush")
        return report

    def shutdown(self, *, reason: str = "operator_shutdown", source: str = "operator") -> None:
        """Halt order flow and stop the runner."""
        self.halt(reason=reason, source=source)
        self._record_control(
            command="shutdown", reason=reason, source=source, result="stopping")
        self._emit_control_alert(
            command="shutdown", reason=reason, source=source,
            result="stopping", severity=Severity.CRITICAL)
        self.stop()

    def apply_control(self, control: Any) -> Any:
        """Apply a ControlEvent-like command to the live runner."""
        if isinstance(control, dict):
            command = str(control.get("command", "")).lower().strip()
            reason = str(control.get("reason", "")).strip() or "control_event"
            source = str(control.get("source", "")).strip() or "operator"
        else:
            command = str(getattr(control, "command", "")).lower().strip()
            reason = str(getattr(control, "reason", "")).strip() or "control_event"
            source = str(getattr(control, "source", "")).strip() or "operator"

        if command == "halt":
            return self.halt(reason=reason, source=source)
        if command == "reduce_only":
            return self.reduce_only(reason=reason, source=source)
        if command == "resume":
            return self.resume(reason=reason)
        if command == "flush":
            return self.flush(reason=reason)
        if command == "shutdown":
            self.shutdown(reason=reason, source=source)
            return None

        raise ValueError(f"Unsupported control command: {command}")
