"""SQLite-backed model metadata registry."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Immutable snapshot of a registered model version."""
    model_id: str
    name: str
    version: int
    params: dict[str, Any]
    features: tuple[str, ...]
    metrics: dict[str, float]
    created_at: datetime
    is_production: bool = False
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelAction:
    """Immutable audit record for production model operations."""
    action_id: int
    name: str
    action: str
    from_model_id: str | None
    to_model_id: str
    reason: str | None
    actor: str | None
    created_at: datetime
    metadata: dict[str, Any]


class ModelRegistry:
    """SQLite-backed model metadata registry.

    Tracks model versions, parameters, features, and metrics. Supports
    promotion of a model to production status and side-by-side comparison.

    Usage:
        registry = ModelRegistry("models.db")
        mv = registry.register(name="alpha_v1", params={...}, features=["sma_20"], metrics={"sharpe": 1.5})
        registry.promote(mv.model_id)
        prod = registry.get_production("alpha_v1")
    """

    def __init__(self, db_path: str | Path = "model_registry.db") -> None:
        self._db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    params TEXT NOT NULL,
                    features TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_production INTEGER NOT NULL DEFAULT 0,
                    tags TEXT NOT NULL DEFAULT '[]'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_name ON models(name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_production ON models(name, is_production)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_actions (
                    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    from_model_id TEXT,
                    to_model_id TEXT NOT NULL,
                    reason TEXT,
                    actor TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_actions_name_created
                ON model_actions(name, created_at DESC, action_id DESC)
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def register(
        self,
        *,
        name: str,
        params: dict[str, Any],
        features: Sequence[str],
        metrics: dict[str, float],
        tags: Sequence[str] = (),
    ) -> ModelVersion:
        """Register a new model version. Auto-increments version number."""
        model_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM models WHERE name = ?",
                (name,),
            ).fetchone()
            next_version = row[0] + 1

            conn.execute(
                """INSERT INTO models
                   (model_id, name, version, params, features, metrics, created_at, is_production, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (
                    model_id,
                    name,
                    next_version,
                    json.dumps(params),
                    json.dumps(list(features)),
                    json.dumps(metrics),
                    now.isoformat(),
                    json.dumps(list(tags)),
                ),
            )

        mv = ModelVersion(
            model_id=model_id,
            name=name,
            version=next_version,
            params=params,
            features=tuple(features),
            metrics=metrics,
            created_at=now,
            is_production=False,
            tags=tuple(tags),
        )
        logger.info("Registered model %s v%d (id=%s)", name, next_version, model_id)
        return mv

    def get(self, model_id: str) -> Optional[ModelVersion]:
        """Retrieve a model version by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_model(row)

    def list_versions(self, name: str) -> list[ModelVersion]:
        """List all versions of a model, ordered by version number."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM models WHERE name = ? ORDER BY version",
                (name,),
            ).fetchall()
        return [self._row_to_model(r) for r in rows]

    def _check_promotion_preconditions(self, model_id: str) -> list[str]:
        """Check preconditions before promotion. Returns list of warnings."""
        warnings: list[str] = []
        with self._connect() as conn:
            # 1. Verify the model artifact exists in the registry
            row = conn.execute(
                "SELECT model_id, name FROM models WHERE model_id = ?", (model_id,),
            ).fetchone()
            if row is None:
                warnings.append(f"Model {model_id} not found in registry")
                return warnings  # No point checking further

            name = row[1]

            # 2. Check if a shadow_compare action exists for this model
            shadow_row = conn.execute(
                "SELECT COUNT(*) FROM model_actions WHERE to_model_id = ? AND action = 'shadow_compare'",
                (model_id,),
            ).fetchone()
            if shadow_row[0] == 0:
                warnings.append(
                    f"No shadow_compare action found for model {model_id} (name={name}) — "
                    "promoting without shadow comparison"
                )

        return warnings

    def promote(
        self,
        model_id: str,
        *,
        reason: str | None = None,
        actor: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark model as production. Demotes current production model for that name."""
        # Check preconditions and log warnings
        precondition_warnings = self._check_promotion_preconditions(model_id)
        for warning in precondition_warnings:
            logger.warning("Promotion precondition: %s", warning)

        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM models WHERE model_id = ?", (model_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Model {model_id} not found")
            name = row[0]
            current_row = conn.execute(
                "SELECT model_id FROM models WHERE name = ? AND is_production = 1",
                (name,),
            ).fetchone()
            previous_model_id = None if current_row is None else current_row[0]

            conn.execute(
                "UPDATE models SET is_production = 0 WHERE name = ? AND is_production = 1",
                (name,),
            )
            conn.execute(
                "UPDATE models SET is_production = 1 WHERE model_id = ?",
                (model_id,),
            )
            conn.execute(
                """INSERT INTO model_actions
                   (name, action, from_model_id, to_model_id, reason, actor, created_at, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    "promote",
                    previous_model_id,
                    model_id,
                    reason,
                    actor,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(metadata or {}),
                ),
            )
        logger.info("Promoted model %s to production (name=%s)", model_id, name)

    def get_production(self, name: str) -> Optional[ModelVersion]:
        """Get the current production model for a given name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE name = ? AND is_production = 1",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_model(row)

    def rollback_to_previous(
        self,
        name: str,
        *,
        to_model_id: str | None = None,
        to_version: int | None = None,
        reason: str | None = None,
        actor: str | None = None,
    ) -> ModelVersion:
        """Rollback production to a previous or explicitly selected version."""
        current = self.get_production(name)
        if current is None:
            raise ValueError(f"No production model for {name}")

        if to_model_id is not None and to_version is not None:
            raise ValueError("Specify at most one rollback target: to_model_id or to_version")

        versions = self.list_versions(name)
        if to_model_id is not None:
            target = next((v for v in versions if v.model_id == to_model_id), None)
            if target is None:
                raise ValueError(f"Rollback target model_id not found for {name}: {to_model_id}")
        elif to_version is not None:
            target = next((v for v in versions if v.version == to_version), None)
            if target is None:
                raise ValueError(f"Rollback target version not found for {name}: v{to_version}")
        else:
            candidates = [v for v in versions if v.version < current.version]
            if not candidates:
                raise ValueError(f"No previous version available for {name}")
            target = candidates[-1]

        if target.model_id == current.model_id:
            raise ValueError(f"Rollback target is already production for {name}")

        rollback_reason = reason or f"rollback:{current.model_id}->{target.model_id}"
        self.promote(
            target.model_id,
            reason=rollback_reason,
            actor=actor,
            metadata={
                "rollback_from_model_id": current.model_id,
                "rollback_from_version": current.version,
                "rollback_to_version": target.version,
            },
        )
        return self.get(target.model_id)  # type: ignore[return-value]

    def list_actions(self, name: str, *, limit: int = 20) -> list[ModelAction]:
        """List recent promotion / rollback audit records for a model name."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT action_id, name, action, from_model_id, to_model_id, reason, actor, created_at, metadata
                   FROM model_actions WHERE name = ?
                   ORDER BY created_at DESC, action_id DESC
                   LIMIT ?""",
                (name, limit),
            ).fetchall()
        return [self._row_to_action(r) for r in rows]

    def compare(self, id_a: str, id_b: str) -> dict[str, Any]:
        """Compare two model versions side by side."""
        a = self.get(id_a)
        b = self.get(id_b)
        if a is None or b is None:
            raise ValueError(f"One or both models not found: {id_a}, {id_b}")

        all_metric_keys = sorted(set(a.metrics) | set(b.metrics))
        metric_comparison = {}
        for key in all_metric_keys:
            val_a = a.metrics.get(key)
            val_b = b.metrics.get(key)
            diff = None
            if val_a is not None and val_b is not None:
                diff = val_b - val_a
            metric_comparison[key] = {"a": val_a, "b": val_b, "diff": diff}

        param_diff = {}
        all_param_keys = sorted(set(a.params) | set(b.params))
        for key in all_param_keys:
            val_a = a.params.get(key)
            val_b = b.params.get(key)
            if val_a != val_b:
                param_diff[key] = {"a": val_a, "b": val_b}

        return {
            "model_a": {"id": id_a, "name": a.name, "version": a.version},
            "model_b": {"id": id_b, "name": b.name, "version": b.version},
            "metrics": metric_comparison,
            "param_diff": param_diff,
            "features_a_only": sorted(set(a.features) - set(b.features)),
            "features_b_only": sorted(set(b.features) - set(a.features)),
            "features_shared": sorted(set(a.features) & set(b.features)),
        }

    @staticmethod
    def _row_to_model(row: tuple) -> ModelVersion:
        return ModelVersion(
            model_id=row[0],
            name=row[1],
            version=row[2],
            params=json.loads(row[3]),
            features=tuple(json.loads(row[4])),
            metrics=json.loads(row[5]),
            created_at=datetime.fromisoformat(row[6]),
            is_production=bool(row[7]),
            tags=tuple(json.loads(row[8])),
        )

    @staticmethod
    def _row_to_action(row: tuple) -> ModelAction:
        return ModelAction(
            action_id=row[0],
            name=row[1],
            action=row[2],
            from_model_id=row[3],
            to_model_id=row[4],
            reason=row[5],
            actor=row[6],
            created_at=datetime.fromisoformat(row[7]),
            metadata=json.loads(row[8]),
        )
