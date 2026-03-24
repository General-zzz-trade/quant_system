"""Audit snapshot helper for OperatorObservabilityMixin.

Extracted from observability.py to keep it under 500 lines.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def ops_audit_snapshot(mixin: Any, *, limit: int = 50) -> Dict[str, Any]:
    """Return a unified operator / execution / model-ops audit snapshot."""
    operator = mixin.operator_status()
    model_actions: List[Dict[str, Any]] = []
    model_status: List[Dict[str, Any]] = []
    cfg = getattr(mixin, "_config", None)
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
        if mixin.model_loader is not None and hasattr(mixin.model_loader, "inspect_production_models"):
            try:
                model_status = list(mixin.model_loader.inspect_production_models(tuple(cfg.model_names)))
            except Exception:
                logger.exception("model status inspection failed during ops audit snapshot")

    return {
        "stream_status": operator["stream_status"],
        "incident_state": operator["incident_state"],
        "last_incident_category": operator["last_incident_category"],
        "last_incident_ts": operator["last_incident_ts"],
        "recommended_action": operator["recommended_action"],
        "operator": mixin.operator_status(),
        "control_history": [
            {
                "command": rec.command,
                "reason": rec.reason,
                "source": rec.source,
                "result": rec.result,
                "ts": rec.ts.isoformat(),
            }
            for rec in mixin.control_history[-limit:][::-1]
        ],
        "execution_alerts": mixin.execution_alert_history(limit=limit),
        "model_alerts": mixin.model_alert_history(limit=limit),
        "model_actions": model_actions,
        "model_status": model_status,
        "model_reload": None if mixin._last_model_reload_status is None else dict(mixin._last_model_reload_status),
        "timeline": mixin.ops_timeline(limit=limit),
    }
