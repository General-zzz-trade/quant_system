from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Protocol

from state.snapshot import StateSnapshot


class PositionSizer(Protocol):
    def target_qty(self, snapshot: StateSnapshot, symbol: str, weight: Decimal) -> Decimal:
        ...
