from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    """Minimal placeholder. Route B will implement structured diffs."""
    changed: bool
    details: Mapping[str, Any]
