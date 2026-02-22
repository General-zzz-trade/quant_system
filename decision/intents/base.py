from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from state.snapshot import StateSnapshot
from decision.types import TargetPosition, OrderSpec


class IntentBuilder(Protocol):
    def build(self, snapshot: StateSnapshot, target: TargetPosition) -> OrderSpec | None:
        ...
