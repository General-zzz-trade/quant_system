"""Parquet-backed BarStore wrapping the existing TimeSeriesStore."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from data.backends.base import Bar
from data.store import TimeSeriesStore

logger = logging.getLogger(__name__)


class ParquetBarStore:
    """BarStore implementation backed by Parquet files via TimeSeriesStore."""

    def __init__(self, root: str | Path) -> None:
        self._store = TimeSeriesStore(root)

    def write_bars(self, symbol: str, bars: Sequence[Bar]) -> None:
        if not bars:
            return
        self._store.write_bars(symbol, bars)

    def read_bars(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Bar]:
        return self._store.read_bars(symbol, start=start, end=end)

    def symbols(self) -> List[str]:
        return self._store.list_symbols()

    def date_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        bars = self._store.read_bars(symbol)
        if not bars:
            return None
        return (bars[0].ts, bars[-1].ts)
