"""Strategy protocol -- interface all strategies implement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class Signal:
    """Universal signal emitted by any strategy.

    Attributes
    ----------
    direction : int
        +1 long, -1 short, 0 flat.
    confidence : float
        Strength in [0.0, 1.0].
    meta : dict
        Arbitrary strategy-specific metadata.
    """

    direction: int  # +1 long, -1 short, 0 flat
    confidence: float  # 0.0 to 1.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0, or +1, got {self.direction}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


class StrategyProtocol(Protocol):
    """Interface that all strategies must implement."""

    name: str
    version: str
    venue: str
    timeframe: str

    def generate_signal(self, features: Dict[str, Any]) -> Signal:
        """Produce a trading signal from the given feature dict."""
        ...

    def validate_config(self) -> bool:
        """Return True if the strategy is correctly configured."""
        ...

    def describe(self) -> str:
        """Human-readable summary of the strategy and its parameters."""
        ...
