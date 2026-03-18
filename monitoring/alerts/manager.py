"""Rule-based alert engine with cooldown and routing.

Evaluates a set of ``AlertRule`` conditions on a periodic basis and fires
alerts through the configured ``AlertSink`` pipeline.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from monitoring.alerts.base import Alert, AlertSink, Severity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlertRule:
    """A single alert rule.

    Parameters
    ----------
    name : str
        Unique rule identifier.
    condition : Callable[[], bool]
        Returns True when the alert should fire.
    severity : Severity
        Alert severity level.
    message_template : str
        Alert message body (may contain descriptive text).
    cooldown_sec : float
        Minimum time between consecutive firings of this rule.
    """

    name: str
    condition: Callable[[], bool]
    severity: Severity
    message_template: str
    cooldown_sec: float = 300.0


class AlertManager:
    """Rule-based alert engine with cooldown and routing."""

    def __init__(
        self,
        *,
        sink: Optional[AlertSink] = None,
        rules: Sequence[AlertRule] = (),
        max_history: int = 256,
    ) -> None:
        self._sink = sink
        self._rules: list[AlertRule] = list(rules)
        self._last_fired: dict[str, float] = {}
        self._history: list[Alert] = []
        self._max_history = max(1, int(max_history))
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def add_rule(self, rule: AlertRule) -> None:
        with self._lock:
            self._rules.append(rule)

    def remove_rule(self, name: str) -> None:
        with self._lock:
            self._rules = [r for r in self._rules if r.name != name]

    @property
    def rules(self) -> list[AlertRule]:
        with self._lock:
            return list(self._rules)

    def check_all(self) -> list[Alert]:
        """Evaluate all rules and fire alerts for matching ones."""
        fired: list[Alert] = []
        now = time.monotonic()

        with self._lock:
            rules_snapshot = list(self._rules)

        for rule in rules_snapshot:
            last = self._last_fired.get(rule.name, float("-inf"))
            if now - last < rule.cooldown_sec:
                continue

            try:
                if rule.condition():
                    alert = Alert(
                        title=rule.name,
                        message=rule.message_template,
                        severity=rule.severity,
                        source="alert_manager",
                        ts=datetime.now(timezone.utc),
                    )
                    if self._sink is not None:
                        try:
                            self._sink.emit(alert)
                        except Exception:
                            logger.exception("Sink emit failed for rule %s", rule.name)
                    self._record(alert)
                    fired.append(alert)
                    self._last_fired[rule.name] = now
            except Exception:
                logger.exception("Rule condition check failed: %s", rule.name)

        return fired

    def emit_direct(self, alert: Alert) -> Alert:
        """Emit a structured alert immediately, bypassing rule evaluation."""
        if self._sink is not None:
            try:
                self._sink.emit(alert)
            except Exception:
                logger.exception("Direct sink emit failed for alert %s", alert.title)
        self._record(alert)
        return alert

    def history(
        self,
        *,
        limit: int = 50,
        category: str | None = None,
        source_prefix: str | None = None,
    ) -> list[Alert]:
        with self._lock:
            rows = list(self._history)
        if category is not None:
            rows = [a for a in rows if str((a.meta or {}).get("category", "")) == category]
        if source_prefix is not None:
            rows = [a for a in rows if str(a.source).startswith(source_prefix)]
        return rows[-int(limit):][::-1]

    def start_periodic(self, interval_sec: float = 10.0) -> None:
        """Start periodic rule checking in background thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(interval_sec,),
            name="alert-manager",
            daemon=True,
        )
        self._thread.start()
        logger.info("AlertManager periodic check started (interval=%.1fs)", interval_sec)

    def stop(self) -> None:
        """Stop the periodic check thread."""
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None
        logger.info("AlertManager stopped")

    def _run_loop(self, interval_sec: float) -> None:
        while not self._stop_event.wait(interval_sec):
            try:
                self.check_all()
            except Exception:
                logger.exception("Periodic check_all failed")

    def _record(self, alert: Alert) -> None:
        with self._lock:
            self._history.append(alert)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
