"""Artifact storage for model weights, configs, and reports."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ArtifactMeta:
    """Metadata for a stored artifact."""
    model_id: str
    artifact_type: str
    path: str
    size_bytes: int
    created_at: datetime


class ArtifactStore:
    """Store model artifacts (weights, configs, reports) on disk.

    Artifacts are organized as ``<root>/<model_id>/<artifact_type>``.

    Usage:
        store = ArtifactStore("artifacts")
        meta = store.save("model-123", "weights", model_bytes)
        data = store.load("model-123", "weights")
    """

    def __init__(self, root: str | Path = "artifacts") -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, model_id: str, artifact_type: str, data: bytes) -> ArtifactMeta:
        """Save artifact bytes to disk."""
        artifact_dir = self._root / model_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / artifact_type
        artifact_path.write_bytes(data)

        meta = ArtifactMeta(
            model_id=model_id,
            artifact_type=artifact_type,
            path=str(artifact_path),
            size_bytes=len(data),
            created_at=datetime.now(timezone.utc),
        )
        logger.info(
            "Saved artifact %s/%s (%d bytes)",
            model_id, artifact_type, len(data),
        )
        return meta

    def load(self, model_id: str, artifact_type: str) -> Optional[bytes]:
        """Load artifact bytes from disk. Returns None if not found."""
        artifact_path = self._root / model_id / artifact_type
        if not artifact_path.exists():
            return None
        return artifact_path.read_bytes()

    def list_artifacts(self, model_id: str) -> list[ArtifactMeta]:
        """List all artifacts for a model."""
        artifact_dir = self._root / model_id
        if not artifact_dir.exists():
            return []

        results: list[ArtifactMeta] = []
        for p in sorted(artifact_dir.iterdir()):
            if p.is_file():
                stat = p.stat()
                results.append(ArtifactMeta(
                    model_id=model_id,
                    artifact_type=p.name,
                    path=str(p),
                    size_bytes=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                ))
        return results

    def delete(self, model_id: str, artifact_type: str) -> bool:
        """Delete an artifact. Returns True if deleted, False if not found."""
        artifact_path = self._root / model_id / artifact_type
        if artifact_path.exists():
            artifact_path.unlink()
            logger.info("Deleted artifact %s/%s", model_id, artifact_type)
            return True
        return False
