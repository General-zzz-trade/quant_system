from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True, slots=True)
class RegimeLabel:
    """A discrete regime label."""

    name: str
    ts: datetime
    value: str
    score: float = 1.0
    meta: Dict[str, Any] | None = None


class RegimeDetector(Protocol):
    name: str

    def detect(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[RegimeLabel]:
        ...
