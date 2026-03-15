# runner/observability.py
"""Operator observability mixin — status, timeline, audit, alerting."""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from execution.observability.incidents import (
    IncidentCategory,
    IncidentState,
    RecommendedAction,
)
from monitoring.alerts.base import Alert, Severity
from risk.kill_switch import KillMode

from runner.config import (
    OperatorControlRecord,
    OperatorKillSwitchStatus,
    OperatorReconcileStatus,
    OperatorStatusSnapshot,
)

logger = logging.getLogger(__name__)


class OperatorObservabilityMixin:
    """Mixin providing operator status, timeline, audit, and alerting methods.

    Expects the following attributes on ``self``:
        _running, _stopped, kill_switch, reconcile_scheduler, _control_history,
        _user_stream_failure_count, _last_user_stream_failure_at,
        _last_user_stream_failure_kind, user_stream, alert_manager,
        event_log, model_loader, coordinator, _fills,
        _last_model_reload_status
    """

    # ── Snapshot / status ──────────────────────────────────────

    def operator_status_snapshot(self) -> OperatorStatusSnapshot:
        """Return the current operator-facing control state snapshot."""
        kill = self.kill_switch.is_killed()
        last_reconcile = getattr(self.reconcile_scheduler, "last_report", None)
        last_control = self._control_history[-1] if self._control_history else None
        stream_status = self._stream_status()
        incident_state = self._incident_state(kill=kill, last_reconcile=last_reconcile, stream_status=stream_status)
        last_incident_category, last_incident_ts = self._last_incident()
        return OperatorStatusSnapshot(
            running=self._running,
            stopped=self._stopped,
            stream_status=stream_status,
            incident_state=incident_state,
            last_incident_category=last_incident_category,
            last_incident_ts=last_incident_ts,
            recommended_action=self._recommended_action(
                kill=kill,
                last_reconcile=last_reconcile,
                stream_status=stream_status,
            ),
            kill_switch=None if kill is None else OperatorKillSwitchStatus(
                scope=kill.scope.value,
                key=kill.key,
                mode=kill.mode.value,
                reason=kill.reason,
                source=kill.source,
            ),
            last_reconcile=None if last_reconcile is None else OperatorReconcileStatus(
                ok=bool(last_reconcile.ok),
                should_halt=bool(last_reconcile.should_halt),
                drift_count=len(last_reconcile.all_drifts),
            ),
            last_control=last_control,
        )

    def operator_status(self) -> Dict[str, Any]:
        """Return a dict-compatible operator status view for APIs and tooling."""
        snap = self.operator_status_snapshot()
        return {
            "running": snap.running,
            "stopped": snap.stopped,
            "stream_status": snap.stream_status,
            "incident_state": snap.incident_state,
            "last_incident_category": snap.last_incident_category,
            "last_incident_ts": None if snap.last_incident_ts is None else snap.last_incident_ts.isoformat(),
            "recommended_action": snap.recommended_action,
            "kill_switch": None if snap.kill_switch is None else asdict(snap.kill_switch),
            "last_reconcile": None if snap.last_reconcile is None else asdict(snap.last_reconcile),
            "last_control": None if snap.last_control is None else {
                "command": snap.last_control.command,
                "reason": snap.last_control.reason,
                "source": snap.last_control.source,
                "result": snap.last_control.result,
                "ts": snap.last_control.ts.isoformat(),
            },
        }

    # ── Alert histories ────────────────────────────────────────

    def execution_alert_history(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent execution-scoped alerts in API-friendly dict form."""
        if self.alert_manager is None:
            return []
        return [
            alert.to_dict()
            for alert in self.alert_manager.history(limit=limit)
            if str((alert.meta or {}).get("category", "")).startswith("execution_")
        ]

    def model_alert_history(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent model-ops alerts in API-friendly dict form."""
        if self.alert_manager is None:
            return []
        return [
            alert.to_dict()
            for alert in self.alert_manager.history(limit=limit)
            if str((alert.meta or {}).get("category", "")).startswith("model_")
        ]

    # ── Timeline / audit ───────────────────────────────────────

    def ops_timeline(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return a recent merged operator / execution / model timeline."""
        rows: List[Dict[str, Any]] = []

        if self.event_log is not None and hasattr(self.event_log, "list_recent"):
            for row in self.event_log.list_recent(event_type="operator_control", limit=limit):
                payload = dict(row.get("payload") or {})
                rows.append(
                    {
                        "kind": "control",
                        "ts": payload.get("ts") or datetime.fromtimestamp(float(row["ts"]), timezone.utc).isoformat(),
                        "title": f"operator_{payload.get('command', row.get('correlation_id', 'control'))}",
                        "category": "operator_control",
                        "source": payload.get("source", ""),
                        "detail": payload,
                    }
                )
            for row in self.event_log.list_recent(event_type="model_reload", limit=limit):
                payload = dict(row.get("payload") or {})
                rows.append(
                    {
                        "kind": "model_reload",
                        "ts": payload.get("ts") or datetime.fromtimestamp(float(row["ts"]), timezone.utc).isoformat(),
                        "title": f"model_reload_{payload.get('outcome', 'unknown')}",
                        "category": "model_reload",
                        "source": "model:reload",
                        "detail": payload,
                    }
                )
            for row in self.event_log.list_recent(event_type="execution_incident", limit=limit):
                payload = dict(row.get("payload") or {})
                rows.append(
                    {
                        "kind": "execution_incident",
                        "ts": payload.get("ts") or datetime.fromtimestamp(float(row["ts"]), timezone.utc).isoformat(),
                        "title": str(payload.get("title", "execution_incident")),
                        "category": str(payload.get("category", "execution_incident")),
                        "source": str(payload.get("source", "")),
                        "detail": payload,
                    }
                )
        else:
            for rec in self.control_history[-limit:]:
                rows.append(
                    {
                        "kind": "control",
                        "ts": rec.ts.isoformat(),
                        "title": f"operator_{rec.command}",
                        "category": "operator_control",
                        "source": rec.source,
                        "detail": {
                            "command": rec.command,
                            "reason": rec.reason,
                            "result": rec.result,
                        },
                    }
                )

        for row in self.execution_alert_history(limit=limit):
            rows.append(
                {
                    "kind": "execution_alert",
                    "ts": row["ts"],
                    "title": row["title"],
                    "category": str((row.get("meta") or {}).get("category", "")),
                    "source": row.get("source", ""),
                    "detail": dict(row.get("meta") or {}),
                }
            )

        for row in self.model_alert_history(limit=limit):
            rows.append(
                {
                    "kind": "model_alert",
                    "ts": row["ts"],
                    "title": row["title"],
                    "category": str((row.get("meta") or {}).get("category", "")),
                    "source": row.get("source", ""),
                    "detail": dict(row.get("meta") or {}),
                }
            )

        cfg = getattr(self, "_config", None)
        if cfg is not None and getattr(cfg, "model_registry_db", None) and getattr(cfg, "model_names", None):
            from research.model_registry.registry import ModelRegistry

            registry = ModelRegistry(cfg.model_registry_db)
            seen_action_ids: set[int] = set()
            for model_name in tuple(cfg.model_names):
                for action in registry.list_actions(model_name, limit=limit):
                    if action.action_id in seen_action_ids:
                        continue
                    seen_action_ids.add(action.action_id)
                    rows.append(
                        {
                            "kind": "model_action",
                            "ts": action.created_at.isoformat(),
                            "title": f"model_{action.action}",
                            "category": "model_action",
                            "source": action.actor or "model_registry",
                            "detail": {
                                "model": action.name,
                                "action_id": action.action_id,
                                "action": action.action,
                                "from_model_id": action.from_model_id,
                                "to_model_id": action.to_model_id,
                                "reason": action.reason,
                                "metadata": action.metadata,
                            },
                        }
                    )

        rows.sort(key=lambda row: row["ts"], reverse=True)
        return rows[:limit]

    def ops_audit_snapshot(self, *, limit: int = 50) -> Dict[str, Any]:
        """Return a unified operator / execution / model-ops audit snapshot."""
        operator = self.operator_status()
        model_actions: List[Dict[str, Any]] = []
        model_status: List[Dict[str, Any]] = []
        cfg = getattr(self, "_config", None)
        if cfg is not None and getattr(cfg, "model_registry_db", None) and getattr(cfg, "model_names", None):
            from research.model_registry.registry import ModelRegistry

            registry = ModelRegistry(cfg.model_registry_db)
            seen_action_ids: set[int] = set()
            for model_name in tuple(cfg.model_names):
                for row in registry.list_actions(model_name, limit=limit):
                    if row.action_id in seen_action_ids:
                        continue
                    seen_action_ids.add(row.action_id)
                    model_actions.append(
                        {
                            "action_id": row.action_id,
                            "model": row.name,
                            "action": row.action,
                            "from_model_id": row.from_model_id,
                            "to_model_id": row.to_model_id,
                            "reason": row.reason,
                            "actor": row.actor,
                            "created_at": row.created_at.isoformat(),
                            "metadata": row.metadata,
                        }
                    )
            model_actions.sort(key=lambda row: row["action_id"], reverse=True)
            model_actions = model_actions[:limit]
            if self.model_loader is not None and hasattr(self.model_loader, "inspect_production_models"):
                try:
                    model_status = list(self.model_loader.inspect_production_models(tuple(cfg.model_names)))
                except Exception:
                    logger.exception("model status inspection failed during ops audit snapshot")

        return {
            "stream_status": operator["stream_status"],
            "incident_state": operator["incident_state"],
            "last_incident_category": operator["last_incident_category"],
            "last_incident_ts": operator["last_incident_ts"],
            "recommended_action": operator["recommended_action"],
            "operator": self.operator_status(),
            "control_history": [
                {
                    "command": rec.command,
                    "reason": rec.reason,
                    "source": rec.source,
                    "result": rec.result,
                    "ts": rec.ts.isoformat(),
                }
                for rec in self.control_history[-limit:][::-1]
            ],
            "execution_alerts": self.execution_alert_history(limit=limit),
            "model_alerts": self.model_alert_history(limit=limit),
            "model_actions": model_actions,
            "model_status": model_status,
            "model_reload": None if self._last_model_reload_status is None else dict(self._last_model_reload_status),
            "timeline": self.ops_timeline(limit=limit),
        }

    # ── Properties ─────────────────────────────────────────────

    @property
    def control_history(self) -> List[OperatorControlRecord]:
        return list(self._control_history)

    @property
    def fills(self) -> List[Dict[str, Any]]:
        return list(self._fills)

    @property
    def event_index(self) -> int:
        return self.coordinator.get_state_view().get("event_index", 0)

    # ── Internal recording helpers ─────────────────────────────

    def _record_control(self, *, command: str, reason: str, source: str, result: str) -> None:
        record = OperatorControlRecord(
            command=command,
            reason=reason,
            source=source,
            ts=datetime.now(timezone.utc),
            result=result,
        )
        self._control_history.append(record)
        if self.event_log is not None:
            self.event_log.append(
                event_type="operator_control",
                correlation_id=command,
                payload={
                    "command": record.command,
                    "reason": record.reason,
                    "source": record.source,
                    "result": record.result,
                    "ts": record.ts.isoformat(),
                },
            )

    def _record_user_stream_connect(self) -> None:
        return None

    def _record_user_stream_failure(self, *, kind: str) -> None:
        self._user_stream_failure_count += 1
        self._last_user_stream_failure_at = datetime.now(timezone.utc)
        self._last_user_stream_failure_kind = kind
        if self.alert_manager is not None:
            try:
                from execution.observability.alerts import build_execution_alert

                venue = str(getattr(getattr(self, "_config", None), "venue", ""))
                severity = Severity.ERROR if self._user_stream_failure_count >= 2 else Severity.WARNING
                self.alert_manager.emit_direct(
                    build_execution_alert(
                        title="execution-user-stream",
                        message=f"user stream {kind} failure count={self._user_stream_failure_count}",
                        severity=severity,
                        category="execution_stream",
                        routing_key=f"{venue}:*:user_stream",
                        source="execution:user_stream",
                        meta={
                            "venue": venue,
                            "failure_kind": kind,
                            "failure_count": self._user_stream_failure_count,
                            "stream_status": self._stream_status(),
                        },
                    )
                )
            except Exception:
                logger.exception("user stream alert emit failed")

    def _stream_status(self) -> str:
        if self.user_stream is None:
            return "ok"
        if self._user_stream_failure_count <= 0:
            return "ok"
        if self._user_stream_failure_count == 1:
            return "degraded"
        return "down"

    def _incident_state(self, *, kill: Any, last_reconcile: Any, stream_status: str) -> str:
        if kill is not None and getattr(kill, "mode", None) == KillMode.HARD_KILL:
            return IncidentState.CRITICAL
        if last_reconcile is not None and bool(getattr(last_reconcile, "should_halt", False)):
            return IncidentState.CRITICAL
        if kill is not None and getattr(kill, "mode", None) == KillMode.REDUCE_ONLY:
            return IncidentState.DEGRADED
        if stream_status in {"degraded", "down"}:
            return IncidentState.DEGRADED
        if last_reconcile is not None and not bool(getattr(last_reconcile, "ok", True)):
            return IncidentState.DEGRADED
        return IncidentState.NORMAL

    def _recommended_action(self, *, kill: Any, last_reconcile: Any, stream_status: str) -> str:
        if kill is not None and getattr(kill, "mode", None) == KillMode.HARD_KILL:
            return RecommendedAction.HALT
        if last_reconcile is not None and bool(getattr(last_reconcile, "should_halt", False)):
            return RecommendedAction.HALT
        if kill is not None and getattr(kill, "mode", None) == KillMode.REDUCE_ONLY:
            return RecommendedAction.REDUCE_ONLY
        if stream_status == "down":
            return RecommendedAction.REDUCE_ONLY
        if stream_status == "degraded":
            return RecommendedAction.REVIEW
        if last_reconcile is not None and not bool(getattr(last_reconcile, "ok", True)):
            return RecommendedAction.REVIEW
        return RecommendedAction.NONE

    def _last_incident(self) -> tuple[Optional[str], Optional[datetime]]:
        latest_category: Optional[str] = None
        latest_ts: Optional[datetime] = None

        if self._last_user_stream_failure_at is not None:
            latest_category = IncidentCategory.EXECUTION_STREAM
            latest_ts = self._last_user_stream_failure_at

        if self._control_history:
            control = self._control_history[-1]
            if latest_ts is None or control.ts >= latest_ts:
                latest_category = IncidentCategory.OPERATOR_CONTROL
                latest_ts = control.ts

        if self.alert_manager is not None:
            alerts = self.alert_manager.history(limit=1)
            if alerts:
                alert = alerts[0]
                alert_ts = alert.ts or datetime.now(timezone.utc)
                category = str((alert.meta or {}).get("category", "")) or alert.title
                if latest_ts is None or alert_ts >= latest_ts:
                    latest_category = category
                    latest_ts = alert_ts

        return latest_category, latest_ts

    def _record_model_reload(
        self,
        *,
        outcome: str,
        model_names: Sequence[str],
        detail: Optional[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> None:
        self._last_model_reload_status = {
            "outcome": outcome,
            "model_names": list(model_names),
            "detail": None if detail is None else dict(detail),
            "error": error,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if self.event_log is not None:
            self.event_log.append(
                event_type="model_reload",
                correlation_id=outcome,
                payload=dict(self._last_model_reload_status),
            )
        if self.alert_manager is not None:
            severity = Severity.INFO
            if outcome == "failed":
                severity = Severity.ERROR
            elif outcome == "noop":
                severity = Severity.WARNING
            self.alert_manager.emit_direct(
                Alert(
                    title=f"model_reload_{outcome}",
                    message=f"model reload outcome={outcome} models={','.join(model_names)}",
                    severity=severity,
                    source="model:reload",
                    ts=datetime.now(timezone.utc),
                    meta={
                        "category": "model_reload",
                        "routing_key": f"model:{outcome}",
                        "outcome": outcome,
                        "model_names": list(model_names),
                        "detail": None if detail is None else dict(detail),
                        "error": error,
                    },
                )
            )

    def _emit_execution_incident(self, alert: Alert) -> None:
        if self.event_log is not None:
            payload = {
                "title": alert.title,
                "category": str((alert.meta or {}).get("category", "")),
                "source": alert.source,
                "ts": (alert.ts or datetime.now(timezone.utc)).isoformat(),
                "meta": dict(alert.meta or {}),
            }
            self.event_log.append(
                event_type="execution_incident",
                correlation_id=str((alert.meta or {}).get("routing_key", alert.title)),
                payload=payload,
            )
        if self.alert_manager is not None:
            self.alert_manager.emit_direct(alert)

    def _emit_control_alert(
        self,
        *,
        command: str,
        reason: str,
        source: str,
        result: str,
        severity: Severity,
    ) -> None:
        if self.alert_manager is None:
            return
        self.alert_manager.emit_direct(
            Alert(
                title=f"operator_{command}",
                message=f"operator control command={command} result={result} reason={reason}",
                severity=severity,
                source=source,
                ts=datetime.now(timezone.utc),
                meta={
                    "category": "operator_control",
                    "command": command,
                    "reason": reason,
                    "result": result,
                },
            )
        )
