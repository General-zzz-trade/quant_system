from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class Signal:
    """A minimal trading signal.

    side: "long" | "short" | "flat"
    strength: 0..1 continuous
    """

    symbol: str
    ts: datetime
    side: str
    strength: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


class AlphaModel(Protocol):
    """Alpha model interface.

    Implementations should be deterministic for a given input.
    """

    name: str

    def predict(self, *, symbol: str, ts: datetime, features: Dict[str, Any]) -> Optional[Signal]:
        ...
