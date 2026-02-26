"""Protocol definitions for bar and tick storage backends."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Protocol, Sequence, Tuple

from data.store import Bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Tick:
    """Single trade tick."""

    ts: datetime
    symbol: str
    price: Decimal
    qty: Decimal
    side: str  # "buy" | "sell"
    trade_id: str = ""


class BarStore(Protocol):
    """Protocol for bar storage backends."""

    def write_bars(self, symbol: str, bars: Sequence[Bar]) -> None: ...

    def read_bars(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Bar]: ...

    def symbols(self) -> List[str]: ...

    def date_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]: ...


class TickStore(Protocol):
    """Protocol for tick storage backends."""

    def write_ticks(self, symbol: str, ticks: Sequence[Tick]) -> None: ...

    def read_ticks(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Tick]: ...

    def count(self, symbol: str) -> int: ...
