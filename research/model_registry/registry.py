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

    def promote(self, model_id: str) -> None:
        """Mark model as production. Demotes current production model for that name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM models WHERE model_id = ?", (model_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Model {model_id} not found")
            name = row[0]

            conn.execute(
                "UPDATE models SET is_production = 0 WHERE name = ? AND is_production = 1",
                (name,),
            )
            conn.execute(
                "UPDATE models SET is_production = 1 WHERE model_id = ?",
                (model_id,),
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
