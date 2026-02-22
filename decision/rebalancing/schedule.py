from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from state.snapshot import StateSnapshot


@dataclass(frozen=True, slots=True)
class AlwaysRebalance:
    def should_rebalance(self, snapshot: StateSnapshot) -> bool:
        return True
