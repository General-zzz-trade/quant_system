"""Intent building — converting target positions into executable order specs.

An IntentBuilder bridges the gap between abstract portfolio targets
(what position we *want*) and concrete order specifications (what
order to place).  The builder inspects the current snapshot to
determine position deltas and produces the appropriate OrderSpec,
or ``None`` if no action is needed.

Implementations
---------------
- ``TargetPositionIntentBuilder`` — delta-based, emits a single order
  to move from current qty to target qty.
- ``IntentValidator`` — validates orders against notional / qty minimums.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from state.snapshot import StateSnapshot
from decision.types import TargetPosition, OrderSpec


class IntentBuilder(Protocol):
    """Protocol for converting a target position into an order specification.

    Parameters
    ----------
    snapshot : StateSnapshot
        Current system state (positions, market data, account).
    target : TargetPosition
        The desired position for a given symbol.

    Returns
    -------
    OrderSpec | None
        An order to place, or None if the position is already at target.
    """
    def build(self, snapshot: StateSnapshot, target: TargetPosition) -> OrderSpec | None:
        ...
