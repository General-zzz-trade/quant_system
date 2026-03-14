"""Data store types -- minimal stubs for the data layer.

The data layer (data/) is NOT part of the default production runtime.
These types are used by data collectors, backfill scripts, and storage backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True, slots=True)
class Bar:
    """OHLCV bar data point."""

    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    symbol: str = ""


class TimeSeriesStore:
    """Parquet-based time series storage for bar data.

    Stores bars as parquet files partitioned by symbol under a root directory.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _symbol_path(self, symbol: str) -> Path:
        return self._root / f"{symbol}.parquet"

    def write_bars(self, symbol: str, bars: Sequence[Bar]) -> None:
        """Write bars to parquet file for the given symbol."""
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as e:
            raise RuntimeError("TimeSeriesStore requires pyarrow") from e

        records = [
            {
                "ts": b.ts,
                "open": float(b.open),
                "high": float(b.high),
                "low": float(b.low),
                "close": float(b.close),
                "volume": float(b.volume),
                "symbol": b.symbol or symbol,
            }
            for b in bars
        ]
        table = pa.Table.from_pylist(records)
        path = self._symbol_path(symbol)

        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path)

    def read_bars(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Bar]:
        """Read bars for a symbol, optionally filtered by time range."""
        path = self._symbol_path(symbol)
        if not path.exists():
            return []

        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as e:
            raise RuntimeError("TimeSeriesStore requires pyarrow") from e

        table = pq.read_table(path)
        rows = table.to_pylist()

        bars = []
        for r in rows:
            ts = r["ts"]
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            bars.append(
                Bar(
                    ts=ts,
                    open=Decimal(str(r["open"])),
                    high=Decimal(str(r["high"])),
                    low=Decimal(str(r["low"])),
                    close=Decimal(str(r["close"])),
                    volume=Decimal(str(r["volume"])),
                    symbol=r.get("symbol", symbol),
                )
            )
        return bars

    def list_symbols(self) -> List[str]:
        """List all symbols with stored data."""
        return sorted(
            p.stem for p in self._root.glob("*.parquet") if p.is_file()
        )
