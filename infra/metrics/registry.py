from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Metrics:
    """Simple metrics container for runners."""

    values: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def inc(self, key: str, n: int = 1) -> None:
        self.values[key] = int(self.values.get(key, 0)) + int(n)
