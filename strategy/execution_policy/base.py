from __future__ import annotations

from typing import Protocol

from state.snapshot import StateSnapshot
from decision.types import OrderSpec


class ExecutionPolicy(Protocol):
    name: str

    def apply(self, snapshot: StateSnapshot, order: OrderSpec) -> OrderSpec:
        ...
