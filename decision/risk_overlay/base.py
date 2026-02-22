from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from state.snapshot import StateSnapshot


class RiskOverlay(Protocol):
    def allow(self, snapshot: StateSnapshot) -> tuple[bool, Sequence[str]]:
        ...
