from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import RegimeLabel


@dataclass
class RegimeState:
    """Stores the latest regime labels for a symbol."""

    symbol: str
    labels: Dict[str, RegimeLabel] = field(default_factory=dict)

    def update(self, label: RegimeLabel) -> None:
        self.labels[label.name] = label

    def get(self, name: str) -> Optional[RegimeLabel]:
        return self.labels.get(name)
