"""Data lineage tracking — records source, version, and processing metadata."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LineageRecord:
    """Provenance record for a data artifact."""
    artifact_id: str  # e.g., "BTCUSDT_1h"
    source: str  # e.g., "binance_api", "csv_import"
    version: str  # data version / hash
    created_at: str  # ISO timestamp
    row_count: int
    time_range_start: str
    time_range_end: str
    schema_hash: str  # hash of column names
    processing_steps: tuple[str, ...] = ()
    parent_artifacts: tuple[str, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)


class LineageTracker:
    """Tracks data lineage with JSONL-based persistent storage.

    Each data operation (download, transform, merge) creates a LineageRecord
    that can be queried for auditing and debugging.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, LineageRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            for line in self._path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                data = json.loads(line)
                rec = LineageRecord(
                    artifact_id=data["artifact_id"],
                    source=data["source"],
                    version=data["version"],
                    created_at=data["created_at"],
                    row_count=data["row_count"],
                    time_range_start=data["time_range_start"],
                    time_range_end=data["time_range_end"],
                    schema_hash=data["schema_hash"],
                    processing_steps=tuple(data.get("processing_steps", ())),
                    parent_artifacts=tuple(data.get("parent_artifacts", ())),
                    meta=data.get("meta", {}),
                )
                self._records[rec.artifact_id] = rec
        except Exception:
            logger.warning("Failed to load lineage from %s", self._path)

    def record(
        self,
        artifact_id: str,
        source: str,
        row_count: int,
        time_range_start: str,
        time_range_end: str,
        columns: Sequence[str],
        processing_steps: Sequence[str] = (),
        parent_artifacts: Sequence[str] = (),
        meta: Optional[Dict[str, Any]] = None,
    ) -> LineageRecord:
        """Record a data lineage entry."""
        schema_hash = hashlib.sha256(",".join(columns).encode()).hexdigest()[:12]
        content_hash = hashlib.sha256(
            f"{artifact_id}:{row_count}:{time_range_start}:{time_range_end}".encode()
        ).hexdigest()[:12]

        rec = LineageRecord(
            artifact_id=artifact_id,
            source=source,
            version=content_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            row_count=row_count,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            schema_hash=schema_hash,
            processing_steps=tuple(processing_steps),
            parent_artifacts=tuple(parent_artifacts),
            meta=meta or {},
        )

        self._records[artifact_id] = rec
        self._append(rec)
        return rec

    def get(self, artifact_id: str) -> Optional[LineageRecord]:
        return self._records.get(artifact_id)

    def list_all(self) -> List[LineageRecord]:
        return list(self._records.values())

    def trace(self, artifact_id: str) -> List[LineageRecord]:
        """Trace the full lineage chain for an artifact."""
        result: list[LineageRecord] = []
        visited: set[str] = set()
        stack = [artifact_id]
        while stack:
            aid = stack.pop()
            if aid in visited:
                continue
            visited.add(aid)
            rec = self._records.get(aid)
            if rec is not None:
                result.append(rec)
                for parent in rec.parent_artifacts:
                    stack.append(parent)
        return result

    def _append(self, rec: LineageRecord) -> None:
        data = {
            "artifact_id": rec.artifact_id,
            "source": rec.source,
            "version": rec.version,
            "created_at": rec.created_at,
            "row_count": rec.row_count,
            "time_range_start": rec.time_range_start,
            "time_range_end": rec.time_range_end,
            "schema_hash": rec.schema_hash,
            "processing_steps": list(rec.processing_steps),
            "parent_artifacts": list(rec.parent_artifacts),
            "meta": rec.meta,
        }
        with self._path.open("a") as f:
            f.write(json.dumps(data) + "\n")
