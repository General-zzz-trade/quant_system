from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from state.snapshot import StateSnapshot


@dataclass(frozen=True, slots=True)
class ThresholdRebalance:
    min_delta_qty: Decimal = Decimal("0")

    def should_rebalance(self, snapshot: StateSnapshot) -> bool:
        return True
